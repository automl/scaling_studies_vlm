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

    def collect_keys_from_tars(input_shards):
        urls, _ = expand_urls(input_shards)

        keys_all = set()
        keys_valid = set()
        filtered_out = set()

        corrupted_tars = set()
        empty_tars = set()

        per_tar_stats = {}

        for url in urls:
            src = [{"url": url}]
            total = 0
            valid = 0
            filtered = 0

            try:
                for sample in tarfile_to_samples_nothrow(src):
                    k = sample.get("__key__")
                    if isinstance(k, bytes):
                        k = k.decode("utf-8", errors="ignore")
                    if not k:
                        continue

                    total += 1
                    keys_all.add(k)

                    if filter_no_caption_or_no_image(sample):
                        valid += 1
                        keys_valid.add(k)
                    else:
                        filtered += 1
                        filtered_out.add(k)

            except Exception as e:
                print(f"Failed reading {url}: {e}")
                corrupted_tars.add(url)
                continue

            if total == 0:
                empty_tars.add(url)

            per_tar_stats[url] = {
                "total": total,
                "valid": valid,
                "filtered": filtered,
            }

        return keys_all, keys_valid, filtered_out, corrupted_tars, empty_tars, per_tar_stats

    input_shards = args.train_data
    keys_all, keys_valid, filtered_out, corrupted_tars, empty_tars, per_tar_stats = collect_keys_from_tars(input_shards)

    missing_in_loader = keys_valid - seen
    unexpected = seen - keys_valid

    write_stats_to_file(
        "dataset_stats.txt",
        corrupted_tars,
        empty_tars,
        per_tar_stats,
        missing_in_loader,
        filtered_out,
        unexpected,
        keys_all,
        keys_valid,num_batches, num_samples, len(seen))
    print("Saved dataset statistics to dataset_stats.txt")


def write_stats_to_file(filename,
                        corrupted_tars,
                        empty_tars,
                        per_tar_stats,
                        missing_in_loader,
                        filtered_out,
                        unexpected,
                        keys_all,
                        keys_valid,
                        num_batches,
                        num_samples,
                        seen):
    with open(filename, "w") as f:
        f.write("============= TAR SUMMARY =============\n\n")

        f.write("[1] CORRUPTED SHARDS\n")
        if corrupted_tars:
            for url in sorted(corrupted_tars):
                f.write(f"  - {url}\n")
        else:
            f.write("  None\n")
        f.write("\n")

        f.write("[2] EMPTY SHARDS\n")
        if empty_tars:
            for url in sorted(empty_tars):
                f.write(f"  - {url}\n")
        else:
            f.write("  None\n")
        f.write("\n")

        f.write("[3] PER-SHARD STATISTICS\n")
        if per_tar_stats:
            for url, stats in sorted(per_tar_stats.items()):
                f.write(
                    f"  - {url}: total={stats['total']}, "
                    f"valid={stats['valid']}, filtered={stats['filtered']}\n"
                )
        else:
            f.write("  No stats available\n")

        f.write("\n============= SUMMARY =============\n")
        f.write(f"Total shards analyzed: {len(per_tar_stats) + len(corrupted_tars)}\n")
        f.write(f"Corrupted shards:      {len(corrupted_tars)}\n")
        f.write(f"Empty shards:          {len(empty_tars)}\n")
        f.write("====================================\n\n")

        f.write("[4] COMPARISON AGAINST LOADER\n")
        f.write(f"Valid but missing in loader: {len(missing_in_loader)}\n")
        f.write(f"Filtered out by pipeline:    {len(filtered_out)}\n")
        f.write(f"Unexpected seen samples:     {len(unexpected)}\n")

        f.write("====================================\n\n")

        f.write(
            f"[TARS] total_in_shards={len(keys_all)}, "
            f"valid_per_filter={len(keys_valid)}, "
            f"filtered_out={len(filtered_out)}\n"
        )

        f.write(
            f"[LOADER] batches={num_batches}, "
            f"samples(total)={num_samples}, "
            f"unique_seen={seen}"
        )


        def _write(name, s):
            f.write(f"\n{name}:\n")
            if not s:
                f.write("  []\n")
            else:
                for item in list(sorted(s))[:20]:
                    f.write(f"  - {item}\n")

        _write("Missing in loader(20 samples)", missing_in_loader)
        _write("Filtered out(20 samples)", filtered_out)
        _write("Unexpected", unexpected)


if __name__ == "__main__":
    main(sys.argv[1:])