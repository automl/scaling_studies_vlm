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
    root_dir: Path = args.root_dir / "individual_training_logs"

    total_eval_time_seconds = 0.0
    total_eval_time_per_config = {}
    processed_files = 0
    processed_rows = 0

    total_eval_time_per_gpu = {}
    DEFAULT_GPU = "H100"

    for config_dir in root_dir.iterdir():
        if not config_dir.is_dir() or not config_dir.name.startswith("train_config"):
            continue

        downstream_results = config_dir / "downstream"
        config_eval_time = 0.0

        if not downstream_results.exists():
            continue

        for per_epoch_result in downstream_results.iterdir():
            if (
                    not per_epoch_result.is_dir()
                    or not per_epoch_result.name.startswith("epoch_")
            ):
                continue

            eval_results_path = per_epoch_result / "eval_results.jsonl"

            if not eval_results_path.exists():
                continue

            processed_files += 1
            epoch_eval_time = 0.0

            with eval_results_path.open("r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    line = line.strip()

                    if not line:
                        continue

                    try:
                        result = json.loads(line)
                    except json.JSONDecodeError as error:
                        print(
                            f"Skipping invalid JSON in "
                            f"{eval_results_path}:{line_number}: {error}"
                        )
                        continue

                    eval_time = result.get("eval_time_seconds")

                    if eval_time is None:
                        print(
                            f"Missing eval_time_seconds in "
                            f"{eval_results_path}:{line_number}"
                        )
                        continue

                    try:
                        eval_time = float(eval_time)
                    except (TypeError, ValueError):
                        print(
                            f"Invalid eval_time_seconds in "
                            f"{eval_results_path}:{line_number}: {eval_time}"
                        )
                        continue

                    epoch_eval_time += eval_time
                    config_eval_time += eval_time
                    total_eval_time_seconds += eval_time
                    processed_rows += 1

            print(
                f"{config_dir.name}/{per_epoch_result.name}: "
                f"{epoch_eval_time:.2f} seconds "
                f"({epoch_eval_time / 60:.2f} minutes)"
            )

        total_eval_time_per_config[config_dir.name] = config_eval_time

        print("\nEvaluation time per configuration:")
        for config_name, eval_time in sorted(total_eval_time_per_config.items()):
            print(
                f"{config_name}: "
                f"{eval_time:.2f} seconds "
                f"({eval_time / 3600:.2f} hours)"
            )

        print("\nOverall evaluation time:")
        print(f"Processed files: {processed_files}")
        print(f"Processed rows: {processed_rows}")
        print(f"Seconds: {total_eval_time_seconds:.2f}")
        print(f"Hours: {total_eval_time_seconds / 3600:.2f}")
        print(f"GPU days: {total_eval_time_seconds / 86_400:.2f}")
