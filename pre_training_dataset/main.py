import argparse
import json
import os
import shutil
import time
from pathlib import Path

from huggingface_hub import snapshot_download
from img2dataset import download

from pre_training_dataset.resizing import resize


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=Path,
        required=False,
        default=Path(os.path.dirname(__file__)) / "laion400m",
        help="Path to directory where the parquet files are found/will be downloaded.",
    )

    parser.add_argument(
        "--shard_data_dir",
        type=str,
        required=False,
        default="train_shards",
        help="Path to directory where the data (webdataset shards) will be downloaded.",
    )

    parser.add_argument(
        "--processes_count",
        type=int,
        required=False,
        default=16,
        help="Number of processes for download for img2dataset.",
    )

    parser.add_argument(
        "--thread_count",
        type=int,
        required=False,
        default=128,
        help="Number of threads for download for img2dataset.",
    )

    parser.add_argument(
        "--image_size",
        type=int,
        required=False,
        default=256,
        help="Size images need to be downloaded to.",
    )

    parser.add_argument(
        "--encode_format",
        type=str,
        required=False,
        choices=["png", "jpg", "webp"],
        default="jpg",
        help="Images encoding format.",
    )

    parser.add_argument(
        "--output_format",
        type=str,
        required=False,
        choices=["webdataset", "tfrecord", "parquet", "files"],
        default="webdataset",
        help="Output format used by img2dataset when downloading images.",
    )

    parser.add_argument(
        "--retries",
        type=int,
        required=False,
        default=2,
        help="Number of time a download should be retried (default 2) for img2dataset.",
    )
    parser.add_argument(
        "--enable_wandb",
        action="store_true",
        default=False,
        help="Whether to enable wandb logging (default False)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        required=False,
        default="laion400m",
        help="Name of W&B project used (default laion400m)",
    )

    parser.add_argument(
        "--download_parquet_files",
        action="store_true",
        default=False,
        help="Whether to consider downloading parquet files from HF.",
    )

    parser.add_argument(
        "--convert_parquet_to_tar",
        action="store_true",
        default=False,
        help="Whether to convert downloaded parquet files to tar files.",
    )

    parser.add_argument(
        "--disable_all_reencoding",
        action="store_true",
        default=False,
        help="Whether to download raw images without any resizing.",
    )

    parser.add_argument(
        "--parquet_files",
        type=str,
        nargs="*",
        default=None,
        help="Specific parquet files to convert to tar files. If not provided, all parquet files in the folder will be used.",
    )

    parser.add_argument(
        "--separate_resizing",
        action="store_true",
        default=False,
        help="Whether to resize images after the raw download.",
    )

    return parser


def download_parquet_files(args):
    metadata_dir = args.data_dir / "parquet_files"

    if not args.download_parquet_files:
        return

    if metadata_dir.exists():
        print(f"Parquet directory already exists at {metadata_dir}, skipping.")
        return

    metadata_dir.mkdir(parents=True)
    print(f"Downloading parquet files to {metadata_dir}...")

    cache_dir = metadata_dir.parent / "hf"

    hf_snapshot_args = dict(
        repo_id="laion/relaion400m",
        revision="refs/convert/parquet",
        allow_patterns=[
            "default/train/0*.parquet",
        ],
        local_dir=metadata_dir,
        cache_dir=cache_dir,
        repo_type="dataset",
        max_workers=args.processes_count,
        token="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",
    )

    snapshot_download(**hf_snapshot_args)

    shutil.rmtree(cache_dir, ignore_errors=True)

    print("Downloaded parquet files.")


def download_tar_shards(args):
    if not args.convert_parquet_to_tar:
        return

    shard_dir = args.data_dir / args.shard_data_dir

    shard_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading images to {shard_dir}...")

    parquet_dir = args.data_dir / "parquet_files/default/train"
    if args.parquet_files:
        parquet_files = [parquet_dir / fname for fname in args.parquet_files]
    else:
        parquet_files = sorted(parquet_dir.glob("*.parquet"))

    for parquet_path in parquet_files:
        print(f"Processing parquet: {parquet_path}")
        download(
            url_list=parquet_path,
            image_size=args.image_size,
            output_folder=str(shard_dir),
            processes_count=args.processes_count,
            thread_count=args.thread_count,
            encode_format=args.encode_format,
            output_format=args.output_format,
            input_format="parquet",
            url_col="url",
            caption_col="caption",
            number_sample_per_shard=10000,
            oom_shard_count=8,
            retries=args.retries,
            timeout=2,
            incremental_mode="extend",
            extract_exif=False,
            compute_hash=None,
            disable_all_reencoding=args.disable_all_reencoding,
            enable_wandb=args.enable_wandb,
            max_shard_retry=0,
            wandb_project=args.wandb_project,
        )

    print("Downloaded shards.")


def infer_resize_start(shard_dir):
    shard_ids = []

    for p in shard_dir.glob("*.tar"):
        stem = p.stem
        if stem.isdigit():
            shard_ids.append(int(stem))

    if not shard_ids:
        return 0

    return max(shard_ids) + 1


if __name__ == "__main__":
    args = get_parser().parse_args()

    shard_dir = args.data_dir / args.shard_data_dir
    resize_start_at = infer_resize_start(shard_dir)

    time_dir = Path("time_tracking")
    time_dir.mkdir(exist_ok=True, parents=True)
    time_file = time_dir / "time_log.jsonl"

    run_log = {}

    download_start = time.time()

    download_parquet_files(args)
    download_tar_shards(args)

    download_end = time.time()

    run_log["download"] = {
        "start": download_start,
        "end": download_end,
        "duration_sec": round(download_end - download_start, 2)
    }

    if args.separate_resizing:
        resizing_proc = args.processes_count // 4
        resize_start = time.time()
        resize(root_dir=str(shard_dir),
               start_at=resize_start_at,
               processes=resizing_proc)
        resize_end = time.time()

        run_log["resize"] = {
            "start": resize_start,
            "end": resize_end,
            "duration_sec": round(resize_end - resize_start, 2)
        }
    with open(time_file, "a") as f:
        f.write(json.dumps(run_log) + "\n")
