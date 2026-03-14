import json
import re
import subprocess
import time
from pathlib import Path


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--config-id", type=int, required=True)
    parser.add_argument("--root-dir", type=str, required=True)

    return parser.parse_args()


def _get_epoch_from_checkpoint(ckpt_path):
    name = ckpt_path.name
    match = re.search(r"epoch[_\-]?(\d+)", name)
    if match:
        return int(match.group(1))
    return None


def main():
    args = parse_args()

    root_dir = Path(args.root_dir).resolve()
    config_id = args.config_id

    model_cfg_dir = root_dir / "model_cfg" / f"config_{config_id}"
    train_config_dir = root_dir / "individual_training_logs" / f"train_config{config_id}"
    ckpt_dir = train_config_dir / "checkpoints"
    downstream_dir = train_config_dir / "downstream"

    data_dir = Path("/work/dlclarge1/sinanid-VLM-scaling-law/datacomp/eval_data")
    evaluate_script = Path("evaluate.py").resolve()

    if not model_cfg_dir.exists():
        raise FileNotFoundError(f"Model config directory not found: {model_cfg_dir}")

    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    if not evaluate_script.exists():
        raise FileNotFoundError(f"evaluate.py not found: {evaluate_script}")

    checkpoints = list(ckpt_dir.glob("*.pt"))

    if not checkpoints:
        raise RuntimeError(f"No checkpoints found in {ckpt_dir}")

    print(f"Config {config_id}: found {len(checkpoints)} checkpoints")

    for ckpt in checkpoints:
        epoch = _get_epoch_from_checkpoint(ckpt)

        if epoch is None:
            epoch = "diverged"
        train_output_dir = downstream_dir / f"epoch_{epoch}"
        train_output_dir.mkdir(parents=True, exist_ok=True)

        use_model = f"local-dir:{model_cfg_dir} {ckpt}"

        cmd = [
            "python", "evaluate.py",
            "--train_output_dir", str(train_output_dir),
            "--data_dir", str(data_dir),
            "--use_model", use_model,
        ]

        print(f"\n[Config {config_id}] Evaluating checkpoint: {ckpt.name}")
        print(" ".join(cmd))

        start_time = time.time()

        subprocess.run(cmd, check=True)

        end_time = time.time()
        elapsed = end_time - start_time

        time_info = {
            "config_id": config_id,
            "checkpoint": ckpt.name,
            "epoch": epoch,
            "eval_time_seconds": elapsed,
        }

        with open(train_output_dir / "eval_time.json", "w") as f:
            json.dump(time_info, f, indent=2)

        print(f"Evaluation time: {elapsed:.2f} seconds")

    print(f"\nEvaluated checkpoints for config {config_id}")


if __name__ == "__main__":
    main()
