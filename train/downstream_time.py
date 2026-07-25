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
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root_dir: Path = args.root_dir

    total_eval_time_per_gpu = {}
    DEFAULT_GPU = "NVIDIA A100-SXM4-40GB"

    for config_dir in root_dir.iterdir():
        if not config_dir.is_dir() or not config_dir.name.startswith("train_config"):
            continue

        downstream_results = config_dir / "downstream"

        if not downstream_results.exists():
            continue

        for per_epoch_result in downstream_results.iterdir():
            if (
                not per_epoch_result.is_dir()
                or not per_epoch_result.name.startswith("epoch_")
            ):
                continue

            eval_results_path = per_epoch_result / "eval_time.json"

            if not eval_results_path.exists():
                continue

            with eval_results_path.open("r", encoding="utf-8") as file:
                result = json.load(file)

            eval_time = float(result["eval_time_seconds"])

            gpu = result.get("gpu_name") or DEFAULT_GPU

            total_eval_time_per_gpu[gpu] = (
                    total_eval_time_per_gpu.get(gpu, 0.0) + eval_time
            )

    print("Total eval time per GPU (seconds):")
    for gpu, t in total_eval_time_per_gpu.items():
        print(f"{gpu}: {t}")

    print("\nTotal eval time per GPU (hours):")
    for gpu, t in total_eval_time_per_gpu.items():
        print(f"{gpu}: {t / 3600}")