import argparse
import math
from pathlib import Path

import torch
from open_clip import get_input_dtype, ClipLoss
from torch import optim

from open_clip_train.precision import get_autocast
from open_clip_train.scheduler import cosine_lr
from open_clip_train.train import backward, unwrap_model
from utils import create_default_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--partition", required=True)
    p.add_argument("--model", required=True)
    p.add_argument(
        "--precision",
        choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "pure_bf16", "pure_fp16", "fp32"],
        default="bf16",
        required=False,
    )
    p.add_argument(
        "--save_to",
        default="bs_probe_cache",
        required=True,
    )
    p.add_argument(
        "--trial_id",
        type=int,
        required=True,
    )
    return p.parse_args()


def dummy_batch(bs: int, device: str, image_size: int = 224, text_len: int = 77, vocab_size: int = 49408):
    images = torch.randn(bs, 3, image_size, image_size, device=device)
    texts = torch.randint(0, vocab_size, (bs, text_len), device=device, dtype=torch.long)
    return images, texts


def train_four_steps(
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

    for step in range(start_step, start_step + 4):
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


def find_max_local_bs(
        model, loss_fn, optimizer, scheduler, device, precision, save_to, trial_id,
        image_size: int = 224, text_len: int = 77
):
    candidates = [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
    best = 0

    for bs in candidates:
        try:
            print(f"probing batch size:{bs}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            images, texts = dummy_batch(bs, device=device, image_size=image_size, text_len=text_len)

            _ = train_four_steps(
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                images=images,
                texts=texts,
                precision=precision
            )

            best = bs
            if "train/" in save_to:
                save_to = save_to.replace("train/", "")

            save_dir = Path(save_to)
            save_dir.mkdir(parents=True, exist_ok=True)

            out_path = save_dir / f"config_{trial_id}.txt"
            out_path.write_text(str(best) + "\n")
            break

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"batch size {bs} causes out of memory error: {e}")
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise
    return best


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = create_default_model(args.model)
    model = model.to(device=device)

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

    find_max_local_bs(
        model=model,
        loss_fn=loss,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        precision=args.precision,
        save_to=args.save_to,
        trial_id=args.trial_id
    )


if __name__ == "__main__":
    main()
