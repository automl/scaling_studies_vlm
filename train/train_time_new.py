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

    NUM_GPUS = 16

    total_training_seconds = 0.0

    total_eval_time_per_gpu = {}
    DEFAULT_GPU = "H100"

    for config_dir in root_dir.iterdir():

        results_jsonl = config_dir / "checkpoints" / "train_time.jsonl"
        if not results_jsonl.exists():
            continue

        with results_jsonl.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                result = json.loads(line)
                total_training_seconds += result.get(
                    "epoch_training_duration_seconds",
                    0.0,
                )

    total_gpu_seconds = total_training_seconds * NUM_GPUS
    total_gpu_hours = total_gpu_seconds / 3600

    print(f"Training time: {total_training_seconds:.2f} seconds")
    print(f"GPU training time: {total_gpu_seconds:.2f} GPU-seconds")
    print(f"GPU training time: {total_gpu_hours:.2f} GPU-hours")
