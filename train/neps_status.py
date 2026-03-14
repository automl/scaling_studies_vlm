import argparse
import json
import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


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
        checkpoints_dir: Path,
        train_scripts_dir: Path,
        config_id: int,
        epoch: int,
        is_diverged_epoch: bool,
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

    if is_diverged_epoch:
        early_stop_path = checkpoints_dir / "early_stop_info.json"

        samples_processed_current_epoch = None

        if early_stop_path.exists():
            with open(early_stop_path, "r") as f:
                data = json.load(f)
                samples_processed_current_epoch = data.get("samples_processed_current_epoch")
        if samples_processed_current_epoch is None:
            return {}

        samples_seen = ((epoch - 1) * train_num_samples) + samples_processed_current_epoch
    else:
        samples_seen = epoch * train_num_samples

    samples_seen_frac = samples_seen / total_samples_planned if total_samples_planned > 0 else 0.0

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


def fetch_epoch_training_time(checkpoints_dir: Path, epoch: int) -> dict:
    train_time_path = checkpoints_dir / "train_time.jsonl"

    if not train_time_path.exists():
        return {}

    with open(train_time_path, "r") as f:
        for line in f:
            record = json.loads(line)

            if record.get("epoch") == epoch:
                return {
                    "train_duration(s)": record.get("epoch_training_duration_seconds")
                }

    return {}


def fetch_compute_per_epoch(checkpoints_dir: Path, epoch: int) -> dict:
    compute_path = checkpoints_dir / "compute_info.json"

    if not compute_path.exists():
        return {}

    with open(compute_path, "r") as f:
        record = json.load(f)

    compute_per_epoch = record.get("compute_per_epoch")
    total_compute = record.get("total_compute")
    total_epochs = record.get("epochs")

    if compute_per_epoch is None:
        return {}

    return {
        "compute_seen": compute_per_epoch * epoch if epoch != total_epochs else total_compute,
        "Total compute(GLOPs)": total_compute,
        "total_compute_planned": total_epochs * compute_per_epoch
    }


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


def fetch_workers(params_dir: Path):
    params_path = params_dir / "params.txt"

    if not params_path.exists():
        return None

    text = params_path.read_text()

    match = re.search(r"workers:\s*(\d+)", text)
    return int(match.group(1)) if match else None


def fetch_completed_steps_from_nan_checkpoint(checkpoints_dir: Path):
    for file in checkpoints_dir.glob("*.pt"):
        if "nan" in file.name.lower():
            match = re.search(r"step(\d+)", file.name)
            if match:
                return int(match.group(1))
    return None


def fetch_progress_for_diverged_epoch(train_scripts_dir: Path, config_id: int, checkpoints_dir: Path,
                                      params_dir) -> dict:
    out = {
        "global_steps_completed": None,
        "total_global_steps": None,
        "in_warmup": None,
        "warmup_steps": None,
        "warmup_progress": None,
        "training_progress": None,
    }

    script_path = train_scripts_dir / f"train_config{config_id}.sh"

    if not script_path.exists():
        return out

    text = script_path.read_text()

    nodes_match = re.search(r"#SBATCH\s+--nodes=(\d+)", text)
    nodes = int(nodes_match.group(1)) if nodes_match else 1

    gpu_match = re.search(r"#SBATCH\s+--gres=gpu:(\d+)", text)
    gpus_per_node = int(gpu_match.group(1)) if gpu_match else 1

    bs_match = re.search(r"--batch-size\s+(\d+)", text)
    batch_size = int(bs_match.group(1)) if bs_match else None

    epochs_match = re.search(r"--epochs\s+(\d+)", text)
    epochs = int(epochs_match.group(1)) if epochs_match else None

    samples_match = re.search(r"--train-num-samples\s+(\d+)", text)
    train_num_samples = int(samples_match.group(1)) if samples_match else None

    warmup_match = re.search(r"--warmup-fraction\s+([\d.]+)", text)
    warmup_fraction = float(warmup_match.group(1)) if warmup_match else 0.0

    if None in (batch_size, epochs, train_num_samples):
        return out

    workers = fetch_workers(params_dir)
    completed_steps = fetch_completed_steps_from_nan_checkpoint(checkpoints_dir)

    global_batch_size = nodes * gpus_per_node * batch_size

    num_batches = math.ceil(train_num_samples / global_batch_size)
    num_worker_batches = math.ceil(num_batches / workers)
    num_batches = num_worker_batches * workers

    steps_per_epoch = num_batches
    total_steps = steps_per_epoch * epochs

    warmup_steps = int(total_steps * warmup_fraction)

    in_warmup = int(warmup_steps > 0 and completed_steps <= warmup_steps)
    warmup_progress = (
        min(completed_steps / warmup_steps, 1.0) if warmup_steps > 0 else 0.0
    )

    training_progress = completed_steps / total_steps if total_steps > 0 else 0.0

    out["global_steps_completed"] = completed_steps
    out["total_global_steps"] = total_steps
    out["in_warmup"] = in_warmup
    out["warmup_steps"] = warmup_steps
    out["warmup_progress"] = warmup_progress
    out["training_progress"] = training_progress

    return out


def fetch_progress_per_epoch(checkpoints_dir: Path, epoch, has_diverged, train_scripts_dir, trial_dir,
                             trial_id) -> dict:
    if has_diverged:
        return fetch_progress_for_diverged_epoch(train_scripts_dir=train_scripts_dir, config_id=trial_id,
                                                 checkpoints_dir=checkpoints_dir, params_dir=trial_dir)
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
    losses = {}

    jsonl_path = checkpoints_dir / "results.jsonl"
    train_loss_path = checkpoints_dir / "train_loss.jsonl"

    if not jsonl_path.exists() or not train_loss_path.exists():
        return {}

    with open(jsonl_path, "r") as f:
        for line in f:
            record = json.loads(line)

            if record.get("epoch") == target_epoch:
                val = record.get("clip_val_loss")
                test = record.get("clip_test_loss")

                if val is None or math.isnan(val) or math.isinf(val):
                    val = 5.0
                if test is None or math.isnan(test) or math.isinf(test):
                    test = 5.0

                losses["val_loss"] = val
                losses["test_loss"] = test
                break

    with open(train_loss_path, "r") as f:
        for line in f:
            record = json.loads(line)

            if record.get("epoch") == target_epoch - 1:
                losses["train_loss"] = record.get("train_loss")
                losses["world_size"] = record.get("world_size")
                losses["train_gpu_name"] = record.get("GPU_trained_on")
                break

    return losses


def compute_lr_value_for_step(progress: dict, base_row: dict) -> float:
    base_lr = float(base_row["lr"])
    step = int(progress["global_steps_completed"])
    warmup_steps = int(progress["warmup_steps"])
    total_steps = int(progress["total_global_steps"])

    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps

    e = step - warmup_steps
    es = total_steps - warmup_steps
    return 0.5 * (1 + math.cos(math.pi * e / es)) * base_lr


def list_available_epochs(checkpoints_dir: Path):
    epochs = []
    has_nan_checkpoint = False

    for file in checkpoints_dir.glob("*.pt"):
        name = file.name

        match = re.match(r"epoch_(\d+)\.pt$", name)
        if match:
            epochs.append(int(match.group(1)))
        elif "nan" in file.stem.lower():
            has_nan_checkpoint = True

    epochs = sorted(set(epochs))

    if has_nan_checkpoint:
        next_epoch = (max(epochs) + 1) if epochs else 1
        epochs.append(next_epoch)

    return epochs, has_nan_checkpoint


def build_rows_per_trial(trial_id: str, root_dir: Path):
    train_logs_root = root_dir / "individual_training_logs"
    trial_dir = train_logs_root / f"train_config{trial_id}"

    train_scripts_dir = root_dir / "train_scripts"
    checkpoints_dir = trial_dir / "checkpoints"
    configs_dir = root_dir / "configs"
    model_cfgs_dir = root_dir / "model_cfg"
    config_dir = configs_dir / f"config_{trial_id}"
    model_cfg = model_cfgs_dir / f"config_{trial_id}/open_clip_config.json"

    # Fields agnostic to epoch
    base_row: dict[str, Any] = {
        "config_id": trial_id,
        **fetch_global_bs(train_scripts_dir, config_id=int(trial_id)),
        **fetch_hyperparameters(config_dir),
        **fetch_text_vision_width(model_cfg),
        **fetch_nparams(checkpoints_dir),
    }
    epochs, has_nan_checkpoint = list_available_epochs(checkpoints_dir)

    rows = []
    for epoch in epochs:
        is_diverged_epoch = has_nan_checkpoint and epoch == epochs[-1]

        progress = fetch_progress_per_epoch(checkpoints_dir, epoch, is_diverged_epoch, train_scripts_dir, trial_dir,
                                            trial_id)
        samples = fetch_train_num_samples_per_epoch(checkpoints_dir=checkpoints_dir,
                                                    train_scripts_dir=train_scripts_dir, config_id=int(trial_id),
                                                    epoch=epoch, is_diverged_epoch=is_diverged_epoch)
        losses = fetch_losses_for_epoch(checkpoints_dir, epoch)

        lr_value = compute_lr_value_for_step(progress, base_row)

        row = deepcopy(base_row)
        row.update({
            "epoch": epoch,
            "epoch_diverged": is_diverged_epoch,
            "current_lr_value": lr_value,
            "lr_ratio": lr_value / base_row["lr"],
            **progress,
            **losses,
            **samples,
            **fetch_compute_per_epoch(checkpoints_dir, epoch),
            **fetch_epoch_training_time(checkpoints_dir, epoch),
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

            if any(r.get("epoch_diverged", False) for r in trial_rows):
                diverged_trial_ids.append(trial_id)

            checkpoints_dir = root_dir / "individual_training_logs" / f"train_config{trial_id}" / "checkpoints"
            if is_completed_trial(checkpoints_dir):
                completed_trial_ids.append(trial_id)

        except Exception as e:
            failed_trial_ids.append(trial_id)
            print(f"Trial {trial_id} failed with error: {e}")

    df = pd.DataFrame(rows)

    if "epoch" in df.columns:
        df = df.set_index(["config_id", "epoch"]).sort_index()
    else:
        df = df.set_index("config_id").sort_index()

    print("Diverged trial ids:", diverged_trial_ids)
    print("Trial ids not seen:", failed_trial_ids)
    df.to_csv(out_csv)
    print("Summary csv file has been written to:", out_csv)
