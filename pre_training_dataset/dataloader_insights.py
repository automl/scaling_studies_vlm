import sys

import torch

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from open_clip import create_model_and_transforms, get_tokenizer

from train.open_clip_train.distributed import init_distributed_device
from train.open_clip_train.params import parse_args
from train.open_clip_train.data import expand_urls, get_data
from train.open_clip_train.data import tarfile_to_samples_nothrow
from train.open_clip_train.data import filter_no_caption_or_no_image

def main(args):
    args = parse_args(args)

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    device = init_distributed_device(args)
    model_kwargs = {}
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        force_context_length=args.force_context_length,
        image_mean=args.image_mean,
        image_std=args.image_std,
        image_interpolation=args.image_interpolation,
        image_resize_mode=args.image_resize_mode,  # only effective for inference
        aug_cfg=args.aug_cfg,
        pretrained_image=args.pretrained_image,
        output_dict=True,
        cache_dir=args.cache_dir,
        **model_kwargs,
    )

    tokenizer = get_tokenizer(args.model, cache_dir=args.cache_dir, context_length=args.force_context_length)

    data = get_data(
        args,
        (preprocess_train, preprocess_val),
        epoch=0,
        tokenizer=tokenizer,
    )

    seen = set()
    num_batches = 0
    num_samples = 0
    dataloader = data["train"].dataloader

    for bidx, batch in enumerate(dataloader):
        num_batches += 1
        images, texts, keys = batch
        num_samples += len(keys)
        for k in keys:
            if isinstance(k, bytes):
                k = k.decode("utf-8", errors="ignore")
            seen.add(k)

    print(f"[LOADER] batches={num_batches}, samples(total)={num_samples}, unique_seen={len(seen)}")

    def collect_keys_from_tars(input_shards):
        urls, _ = expand_urls(input_shards)

        keys_all = set()
        keys_valid = set()
        filtered_out = set()

        for url in urls:
            src = [{"url": url}]
            try:
                for sample in tarfile_to_samples_nothrow(src):
                    k = sample.get("__key__")
                    if isinstance(k, bytes):
                        k = k.decode("utf-8", errors="ignore")
                    if not k:
                        continue

                    keys_all.add(k)
                    if filter_no_caption_or_no_image(sample):
                        keys_valid.add(k)
                    else:
                        filtered_out.add(k)
            except Exception as e:
                print(f"Failed reading {url}: {e}")

        return keys_all, keys_valid, filtered_out

    input_shards = args.train_data
    keys_all, keys_valid, filtered_out = collect_keys_from_tars(input_shards)

    print(
        f"[TARS] total_in_shards={len(keys_all)}, valid_per_filter={len(keys_valid)}, filtered_out={len(filtered_out)}")

    missing_in_loader = keys_valid - seen
    unexpected = seen - keys_valid

    print("\n=== Comparison ===")
    print(f"Valid but missing in loader: {len(missing_in_loader)}")
    print(f"Filtered out by pipeline  : {len(filtered_out)}")
    print(f"Unexpected seen (not valid): {len(unexpected)}")

    def peek(name, s, n=10):
        if not s:
            print(f"{name}: []")
        else:
            print(f"{name} (showing up to {n}): {sorted(list(s))[:n]}")

    peek("Missing in loader", missing_in_loader)
    peek("Filtered out", filtered_out)
    peek("Unexpected seen", unexpected)


if __name__ == "__main__":
    main(sys.argv[1:])
