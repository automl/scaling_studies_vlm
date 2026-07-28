import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root_dir",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--default_gpu",
        type=str,
        default="NVIDIA A100-SXM4-40GB",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root_dir: Path = args.root_dir
    default_gpu: str = args.default_gpu

    total_eval_time_per_gpu = {}
    DEFAULT_GPU = "NVIDIA A100-SXM4-40GB"

    for config_dir in root_dir.iterdir():

        results_jsonl = config_dir / "checkpoints" / "results.jsonl"

        if not results_jsonl.exists():
            continue

        with open(results_jsonl, "r") as f:
            for line in f:
                if not line.strip():
                    continue

                row = json.loads(line)

                gpu = row.get("gpu_name") or default_gpu

                total_time = (
                        row.get("val_time_sec", 0.0)
                        + row.get("test_time_sec", 0.0)
                )

                total_eval_time_per_gpu[gpu] = (
                        total_eval_time_per_gpu.get(gpu, 0.0) + total_time
                )

    print("Total eval time per GPU (seconds):")
    for gpu, t in total_eval_time_per_gpu.items():
        print(f"{gpu}: {t}")

    print("\nTotal eval time per GPU (hours):")
    for gpu, t in total_eval_time_per_gpu.items():
        print(f"{gpu}: {t / 3600}")
