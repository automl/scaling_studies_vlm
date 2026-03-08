import json
import os
import re
import subprocess
from pathlib import Path


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--config-id", type=int, required=True)
    parser.add_argument("--models-cfg-dir", type=str, required=True)
    parser.add_argument("--logs-dir", type=str, required=True)

    parser.add_argument("--val-data", type=str, required=True)
    parser.add_argument("--test-data", type=str, required=True)

    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-dir", type=str, required=True)
    parser.add_argument("--command-dir", type=str, required=True)

    return parser.parse_args()


def _prepare_runtime(train_dir: str, command_dir: str) -> dict:
    train_dir_p = Path(train_dir).resolve()
    command_dir_p = Path(command_dir).resolve()

    os.chdir(train_dir_p)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{command_dir_p}:{env.get('PYTHONPATH', '')}"
    return env


def _should_skip_eval_on_epoch(epoch, checkpoints_dir):
    results_path = Path(checkpoints_dir) / "results.jsonl"

    if not results_path.exists():
        return False

    with results_path.open("r") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            if rec.get("epoch") == epoch:
                has_val = any(k in rec for k in ["val_loss", "clip_val_loss"])
                has_test = any(k in rec for k in ["test_loss", "clip_test_loss"])

                if has_val and has_test:
                    return True

    return False


def _get_epoch_from_checkpoint(ckpt_path):
    name = ckpt_path.name
    match = re.search(r"epoch[_\-]?(\d+)", name)
    if match:
        return int(match.group(1))
    return None


def main():
    args = parse_args()

    config_id = args.config_id

    model_cfg_base_path = Path(args.models_cfg_dir)

    if model_cfg_base_path.name == f"config_{config_id}":
        model_cfg_dir = model_cfg_base_path
    else:
        model_cfg_dir = model_cfg_base_path / f"config_{config_id}"

    if not str(model_cfg_dir).startswith("local-dir:"):
        model_cfg_dir = f"local-dir:{model_cfg_dir}"

    ckpt_dir = Path(args.logs_dir) / f"train_config{config_id}" / "checkpoints"
    eval_logs_dir = Path(args.logs_dir)

    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    eval_logs_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = list(ckpt_dir.glob("*.pt"))

    if not checkpoints:
        raise RuntimeError(f"No checkpoints found in {ckpt_dir}")

    print(f"Config {config_id}: found {len(checkpoints)} checkpoints")

    run_name = f"train_config{config_id}"

    for ckpt in checkpoints:
        epoch = _get_epoch_from_checkpoint(ckpt)

        if epoch is not None and _should_skip_eval_on_epoch(epoch, ckpt_dir):
            print(f"Skipping checkpoint {ckpt.name}, epoch {epoch} already evaluated")
            continue

        cmd = [
            "python", "-u", "-m", "clip_train.main",
            "--val-data", args.val_data,
            "--test-data", args.test_data,
            "--dataset-type", "webdataset",
            "--precision", "fp32",
            "--model", str(model_cfg_dir),
            "--seed", str(args.seed),
            "--name", run_name,
            "--logs", str(args.logs_dir),
            "--pipeline-id", str(args.config_id),
            "--resume", str(ckpt),
        ]

        print(f"\n[Config {config_id}] Evaluating checkpoint: {ckpt.name}")
        print(" ".join(cmd))
        env = _prepare_runtime(args.train_dir, args.command_dir)
        subprocess.run(cmd, check=True, env=env)

    print(f"\nEvaluated checkpoints for config {config_id}")


if __name__ == "__main__":
    main()
