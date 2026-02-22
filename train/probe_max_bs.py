import argparse
import gc
import math
import os

import torch
from open_clip import get_input_dtype, ClipLoss
from torch import optim

from clip_train.precision import get_autocast
from clip_train.scheduler import cosine_lr
from clip_train.train import backward, unwrap_model
from utils import create_default_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument(
        "--precision",
        choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "pure_bf16", "pure_fp16", "fp32"],
        default="amp",
        required=False,
    )
    return p.parse_args()


def dummy_batch(bs: int, device: str, image_size: int = 224, text_len: int = 77, vocab_size: int = 49408):
    images = torch.randn(bs, 3, image_size, image_size, device=device)
    texts = torch.randint(0, vocab_size, (bs, text_len), device=device, dtype=torch.long)
    return images, texts


def train_one_step(
        model,
        loss_fn,
        optimizer,
        scheduler,
        device,
        images,
        texts,
        precision,
        start_step: int = 0,
):
    device = torch.device(device)
    autocast = get_autocast(precision, device_type=device.type)
    input_dtype = get_input_dtype(precision)

    model.train()

    total_loss = None

    for step in range(start_step, start_step + 1):
        scheduler(step)

        images_step = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts_step = texts.to(device=device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            model_out = model(images_step, texts_step)
            losses = loss_fn(**model_out, output_dict=True)
            total_loss = sum(losses.values())

        backward(total_loss, None)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()

        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

    return total_loss.detach()


def largest_power_of_two_leq(x: int) -> int:
    return 1 << (x.bit_length() - 1)


def find_max_local_bs(
        model, loss_fn, optimizer, scheduler, device, precision,
        image_size: int = 224, text_len: int = 77,
        max_bs: int = 4096,
):
    assert torch.cuda.is_available()

    probe_batches = [16, 32]
    peaks = {}
    images = texts = None
    for bs in probe_batches:
        try:
            print(f"probing batch size: {bs}")

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            images, texts = dummy_batch(
                bs, device=device, image_size=image_size, text_len=text_len
            )

            _ = train_one_step(
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                images=images,
                texts=texts,
                precision=precision
            )

            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            peaks[bs] = peak

            print(f"batch {bs} | peak VRAM: {peak:.2f} GB")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise RuntimeError(f"OOM even at batch size {bs}, cannot estimate safely")
            raise
        finally:
            _cuda_cleanup(images, texts)

    bs1, bs2 = probe_batches
    m1, m2 = peaks[bs1], peaks[bs2]

    per_sample_mem = (m2 - m1) / (bs2 - bs1)

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    total_vram = props.total_memory / 1024 ** 3
    usable_vram = total_vram * 0.95  # a buffer of 5 percent, assuming there is no max. utilization

    extra_samples = int((usable_vram - m1) / per_sample_mem)
    extra_samples = max(0, extra_samples)

    predicted_total = bs1 + extra_samples
    predicted_total = max(1, min(predicted_total, max_bs))

    best = largest_power_of_two_leq(predicted_total)

    print(
        f"Predicted max batch ≈ {predicted_total}, "
        f"choosing power-of-2 batch size: {best}"
    )

    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        images, texts = dummy_batch(
            best, device=device, image_size=image_size, text_len=text_len
        )

        _ = train_one_step(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            images=images,
            texts=texts,
            precision=precision
        )

        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"verified batch {best} | peak VRAM: {peak:.2f} GB")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            best //= 2
            print(f"verification failed, falling back to {best}")
        else:
            raise
    finally:
        _cuda_cleanup(images, texts)

    return best


def _cuda_cleanup(*tensors):
    for t in tensors:
        try:
            del t
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    probe_max_bs(args.model, args.precision)


def probe_max_bs(model_str, precision):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print_vram("Before model load")

    model = create_default_model(model_str)
    model = model.to(device=device)

    print_vram("After model load")

    exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n, p: not exclude(n, p)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

    optimizer = optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": 0.2},
        ],
        lr=0.001,
        betas=(0.98, 0.9),
        eps=1e-6,
    )

    scheduler = cosine_lr(optimizer, 0.001, 0, 10)

    loss = ClipLoss(
        local_loss=True,
        gather_with_grad=True,
        cache_labels=True,
    )

    return find_max_local_bs(
        model=model,
        loss_fn=loss,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        precision=precision,
    )


def print_vram(prefix=""):
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)

    total = props.total_memory / 1024 ** 3
    allocated = torch.cuda.memory_allocated(device) / 1024 ** 3
    reserved = torch.cuda.memory_reserved(device) / 1024 ** 3
    free = total - reserved

    print(
        f"PID={os.getpid()} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"PCI_BUS_ID={props.pci_bus_id} "
        f"GPU={props.name} "
        f"{prefix} VRAM | "
        f"total: {total:.2f} GB | "
        f"allocated: {allocated:.2f} GB | "
        f"reserved: {reserved:.2f} GB | "
        f"free (est.): {free:.2f} GB"
    )



if __name__ == "__main__":
    main()
