import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from img2dataset import download


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=Path,
        required=False,
        default=Path(os.path.dirname(__file__)) / "laion400m",
        help="Path to directory where the data (webdataset shards) will be stored.",
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

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()

    metadata_dir = args.data_dir / "parquet_files"

    if args.download_parquet_files:
        if not metadata_dir.exists():
            metadata_dir.mkdir(parents=True)

            print(f"Downloading parquet files to {metadata_dir}...")

            cache_dir = metadata_dir.parent / f"hf"
            hf_snapshot_args = dict(
                repo_id="laion/relaion400m",
                revision="refs/convert/parquet",
                allow_patterns=[
                    "default/train/0*.parquet",
                ],
                local_dir=metadata_dir,
                cache_dir=cache_dir,
                repo_type="dataset",
                max_workers=16,
                token="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",
            )

            snapshot_download(**hf_snapshot_args)
            shutil.rmtree(cache_dir, ignore_errors=True)

            print("Downloaded parquet files")

    if args.convert_parquet_to_tar:
        shard_dir = args.data_dir / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading images to {shard_dir}")

        download(
            url_list=str(metadata_dir / "default/train/0000.parquet"),
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
            timeout=1,
            extract_exif=False,
            compute_hash=None,
            disable_all_reencoding = True,
            enable_wandb=args.enable_wandb,
            max_shard_retry=0,
            wandb_project=args.wandb_project
        )

        print("Downloaded shards")
