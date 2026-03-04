import argparse
import json
import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from utils import read_last_json_line


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root_dir",
        type=Path,
        required=True,
        help='Example: "results/model=ViT-S-32/optimizer=experimental/seed=10"',
    )

    p.add_argument(
        "--start_from",
        type=int,
        required=False,
        help='First config id from which the status is generated',
        default=1,
    )

    p.add_argument(
        "--end_at",
        type=int,
        required=False,
        help='Last config id from which the status is generated',
        default=100,
    )
    return p.parse_args()


def read_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def extract_zeroshot_acc1(root_dir: Path, latest_epoch: int) -> dict:
    zero_shot_dir = root_dir / "zero_shot"

    out = {
        "cifar10_acc1": None,
        "imagenet1k_acc1": None,
    }

    cifar_path = zero_shot_dir / f"cifar10_epoch_{latest_epoch}.json"
    if cifar_path.is_file():
        try:
            out["cifar10_acc1"] = read_json(cifar_path).get("metrics", {}).get("acc1")
        except Exception:
            pass

    imagenet_path = zero_shot_dir / f"imagenet1k_epoch_{latest_epoch}.json"
    if imagenet_path.is_file():
        try:
            out["imagenet1k_acc1"] = read_json(imagenet_path).get("metrics", {}).get("acc1")
        except Exception:
            pass

    return out


def extract_mscoco_retrieval(root_dir: Path, latest_epoch: int) -> dict:
    retrieval_dir = root_dir / "retrieval"

    out = {
        "mscoco_image_recall@5": None,
        "mscoco_text_recall@5": None,
    }

    mscoco_path = retrieval_dir / f"mscoco_captions_epoch_{latest_epoch}.json"
    if mscoco_path.is_file():
        try:
            metrics = read_json(mscoco_path).get("metrics", {})
            out["mscoco_image_recall@5"] = metrics.get("image_retrieval_recall@5")
            out["mscoco_text_recall@5"] = metrics.get("text_retrieval_recall@5")
        except Exception:
            pass

    return out


def fetch_downstream_eval_results(path, latest_epoch):
    zero_shot_results = extract_zeroshot_acc1(path, latest_epoch)
    retrieval_results = extract_mscoco_retrieval(path, latest_epoch)
    return {**zero_shot_results, **retrieval_results}


def fetch_hyperparameters(config_dir: Path) -> dict:
    config_path = config_dir / "config.yaml"
    if not config_path.is_file():
        return {}

    with config_path.open("r") as f:
        params = yaml.safe_load(f) or {}

    params.pop("train_num_samples", None)

    return params


def fetch_text_vision_width(model_cfg):
    with open(model_cfg, "r") as f:
        cfg = json.load(f)

    vision_width = cfg["model_cfg"]["vision_cfg"]["width"]
    text_width = cfg["model_cfg"]["text_cfg"]["width"]
    return {"text_width": text_width, "vision_width": vision_width}


def fetch_train_num_samples_per_epoch(
        train_scripts_dir: Path,
        config_id: int,
        epoch: int
):
    script_path = train_scripts_dir / f"train_config{config_id}.sh"

    if not script_path.exists():
        return {}

    text = script_path.read_text()

    samples_match = re.search(r'--train-num-samples\s+(\S+)', text)
    epochs_match = re.search(r'--epochs\s+(\S+)', text)

    if not samples_match or not epochs_match:
        return {}

    train_num_samples = int(float(samples_match.group(1)))
    total_epochs = int(float(epochs_match.group(1)))

    total_samples_planned = train_num_samples * total_epochs
    samples_seen = epoch * train_num_samples
    samples_seen_frac = samples_seen / total_samples_planned

    return {
        "train_num_samples_per_epoch": train_num_samples,
        "total_epochs": total_epochs,
        "total_samples_planned": total_samples_planned,
        "samples_seen": samples_seen,
        "samples_seen_frac": samples_seen_frac,
    }


def fetch_global_bs(train_scripts_dir: Path, config_id: int):
    script_path = train_scripts_dir / f"train_config{config_id}.sh"

    if not script_path.exists():
        return {}

    text = script_path.read_text()

    nodes_match = re.search(r"#SBATCH\s+--nodes=(\d+)", text)
    nodes = int(nodes_match.group(1)) if nodes_match else 1

    gpu_match = re.search(r"#SBATCH\s+--gres=gpu:(\d+)", text)
    gpus_per_node = int(gpu_match.group(1)) if gpu_match else 1

    bs_match = re.search(r"--batch-size\s+(\d+)", text)
    batch_size = int(bs_match.group(1)) if bs_match else None

    if batch_size is None:
        return {}

    global_batch_size = nodes * gpus_per_node * batch_size
    return {
        "global_batch_size": global_batch_size,
    }


def fetch_nparams(checkpoints_dir: Path) -> dict:
    result = {}

    params_path = checkpoints_dir / "model_params.txt"
    if params_path.is_file():
        with params_path.open("r") as f:
            for line in f:
                if ":" not in line:
                    continue
                key, value = line.strip().split(":", 1)
                result[key.strip()] = float(value.strip())

    return {
        "mparams": result.get("total_mparams"),
        "vision_mparams": result.get("image_mparams"),
        "text_mparams": result.get("text_mparams"),
        "text_mparams_non_emb": result.get("text_mparams_non_emb"),
    }


def fetch_training_gpu_used_gpu_training_time(checkpoints_dir: Path, epoch) -> dict:
    out = {"train_gpu_name": None, "world_size": None, "train_duration(s)": None}
    train_results_path = checkpoints_dir / "train_loss.jsonl"
    train_time_path = checkpoints_dir / "train_time.jsonl"

    try:
        out["train_gpu_name"] = read_last_json_line(train_results_path).get("GPU_trained_on")
        out["world_size"] = read_last_json_line(train_results_path).get("world_size")
        out["train_duration(s)"] = read_last_json_line(train_time_path).get("epoch_training_duration_seconds")
    except Exception:
        pass

    return out


def fetch_losses(checkpoints_dir: Path) -> dict:
    out = {"train_loss": None, "val_loss": None, "test_loss": None}

    epoch1_path = checkpoints_dir / "epoch_1.pt"
    results_path = checkpoints_dir / "results.jsonl"
    train_results_path = checkpoints_dir / "train_loss.jsonl"

    if not (epoch1_path.is_file() and results_path.is_file() and train_results_path.is_file()):
        return out

    try:
        train_results = read_last_json_line(train_results_path) or {}
        val_test_results = read_last_json_line(results_path) or {}

        out["train_loss"] = train_results.get("train_loss")
        out["val_loss"] = val_test_results.get("clip_val_loss")
        out["test_loss"] = val_test_results.get("clip_test_loss")

    except Exception:
        pass

    return out


def fetch_gpu_names_and_world_size(checkpoints_dir: Path) -> dict:
    out = {"train_gpu": None, "val_test_gpu": None, "downstream_gpu": None}
    results_path = checkpoints_dir / "results.jsonl"
    train_results_path = checkpoints_dir / "train_loss.jsonl"
    downstream_eval_results_path = checkpoints_dir / "eval_gpu_info.jsonl"

    try:
        train_results = read_last_json_line(train_results_path) or {}
        val_test_results = read_last_json_line(results_path) or {}
        downstream_eval_results = read_last_json_line(downstream_eval_results_path) or {}
        out["train_gpu"] = train_results.get("GPU_trained_on")
        out["val_test_gpu"] = val_test_results.get("gpu_name")
        out["downstream_gpu"] = downstream_eval_results.get("gpu_name")
        out["train_world_size"] = train_results.get("world_size")

    except Exception:
        pass

    return out


def find_latest_epoch(checkpoints_dir: Path) -> int | None:
    _epoch_re = re.compile(r"^epoch_(\d+)\.pt$")

    epochs = []
    for p in checkpoints_dir.glob("epoch_*.pt"):
        m = _epoch_re.match(p.name)
        if m:
            epochs.append(int(m.group(1)))
    return max(epochs) if epochs else None


def has_diverged(checkpoints_dir: Path) -> bool:
    if not checkpoints_dir.is_dir():
        return False
    return any(checkpoints_dir.glob("nan_*.pt"))


def is_completed_trial(checkpoints_dir: Path) -> bool:
    required_files = [
        "compute_info.json",
        "model_params.txt",
        "train_loss.jsonl",
        "train_loss_over_data_seen.jsonl",
    ]
    if not checkpoints_dir.is_dir():
        return False

    if any(not (checkpoints_dir / fname).is_file() for fname in required_files):
        return False

    return True


def fetch_progress_per_epoch(checkpoints_dir: Path, epoch) -> dict:
    out = {
        "global_steps_completed": None,
        "total_global_steps": None,
        "in_warmup": None,
        "warmup_steps": None,
        "warmup_progress": None,
        "training_progress": None,
    }

    results_path = checkpoints_dir / "epoch_steps_info.jsonl"

    if not results_path.exists():
        return out

    with open(results_path, "r") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)

            if int(data.get("epoch")) == int(epoch):
                for key in out.keys():
                    out[key] = data.get(key)

                if out["in_warmup"] is not None:
                    out["in_warmup"] = int(out["in_warmup"])

                return out

    return out


def fetch_losses_for_epoch(checkpoints_dir: Path, target_epoch: int) -> dict:
    jsonl_path = checkpoints_dir / "results.jsonl"

    if not jsonl_path.exists():
        return {}

    with open(jsonl_path, "r") as f:
        for line in f:
            record = json.loads(line)

            if record.get("epoch") == target_epoch:
                return {
                    "clip_val_loss": record.get("clip_val_loss"),
                    "clip_test_loss": record.get("clip_test_loss"),
                }

    return {}


def compute_lr_value_for_epoch(progress: dict, base_row: dict) -> float:
    base_lr = base_row["lr"]
    step = progress["global_steps_completed"]
    total_steps = progress["total_global_steps"]
    warmup_steps = progress["warmup_steps"]

    e = step - warmup_steps
    es = total_steps - warmup_steps

    return 0.5 * (1.0 + math.cos(math.pi * e / es)) * base_lr


def list_available_epochs(checkpoints_dir: Path):
    epochs = []

    for file in checkpoints_dir.glob("epoch_*.pt"):
        match = re.match(r"epoch_(\d+)\.pt", file.name)
        if match:
            epochs.append(int(match.group(1)))

    return sorted(epochs)


def build_rows_per_trial(trial_id: str, root_dir: Path):
    train_logs_root = root_dir / "individual_training_logs"
    trial_dir = train_logs_root / f"train_config{trial_id}"

    train_scripts_dir = root_dir / "train_scripts"
    checkpoints_dir = trial_dir / "checkpoints"
    configs_dir = root_dir / "configs"
    model_cfgs_dir = root_dir / "model_cfg"
    config_dir = configs_dir / f"config_{trial_id}"
    model_cfg = model_cfgs_dir / f"config_{trial_id}/open_clip_config.json"

    diverged = has_diverged(checkpoints_dir)

    if diverged: # ignore for now
        return []

    # Fields agnostic to epoch
    base_row: dict[str, Any] = {
        "config_id": trial_id,
        "diverged": diverged,
        **fetch_global_bs(train_scripts_dir, config_id=int(trial_id)),
        **fetch_hyperparameters(config_dir),
        **fetch_text_vision_width(model_cfg),
        **fetch_nparams(checkpoints_dir),
    }
    epochs = list_available_epochs(checkpoints_dir)
    rows = []
    for epoch in epochs:
        progress = fetch_progress_per_epoch(checkpoints_dir, epoch)
        samples = fetch_train_num_samples_per_epoch(train_scripts_dir=train_scripts_dir, config_id=int(trial_id),
                                                    epoch=epoch)
        losses = fetch_losses_for_epoch(checkpoints_dir, epoch)

        lr_value = compute_lr_value_for_epoch(progress, base_row)

        row = deepcopy(base_row)
        row.update({
            "epoch": epoch,
            "current_lr_value": lr_value,
            "lr_ratio": lr_value / base_row["lr"],
            **progress,
            **losses,
            **samples
        })

        rows.append(row)

    return rows


if __name__ == "__main__":
    args = parse_args()

    root_dir: Path = args.root_dir
    out_csv = root_dir / "configs_summary.csv"

    trial_ids = list(range(args.start_from, args.end_at))

    rows: list[dict[str, Any]] = []
    completed_trial_ids: list[int] = []
    diverged_trial_ids: list[int] = []
    failed_trial_ids: list[int] = []

    for trial_id in trial_ids:
        try:
            trial_rows = build_rows_per_trial(str(trial_id), root_dir)

            if not trial_rows:
                failed_trial_ids.append(trial_id)
                continue

            rows.extend(trial_rows)

            if any(r.get("diverged", False) for r in trial_rows):
                diverged_trial_ids.append(trial_id)

            checkpoints_dir = root_dir / "individual_training_logs" / f"train_config{trial_id}" / "checkpoints"
            if is_completed_trial(checkpoints_dir):
                completed_trial_ids.append(trial_id)

        except Exception as e:
            failed_trial_ids.append(trial_id)
            print(f"Trial {trial_id} failed with error: {e}")

    failed_trial_ids = sorted(set(failed_trial_ids))
    print("Failed trials:", failed_trial_ids)

    diverged_trial_ids = sorted(set(diverged_trial_ids))
    print("Diverged trials:", diverged_trial_ids)

    completed_trial_ids = sorted(set(completed_trial_ids))
    print("Completed trials:", completed_trial_ids)

    df = pd.DataFrame(rows)

    if "epoch" in df.columns:
        df = df.set_index(["config_id", "epoch"]).sort_index()
    else:
        df = df.set_index("config_id").sort_index()

    df.to_csv(out_csv)
    print("Summary csv file has been written to:", out_csv)
