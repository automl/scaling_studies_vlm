import argparse
import json
from pathlib import Path

import neps
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
    return p.parse_args()


def read_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def extract_zeroshot_acc1(root_dir: Path) -> dict:
    zero_shot_dir = root_dir / "zero_shot"

    out = {
        "cifar10_acc1": None,
        "imagenet1k_acc1": None,
    }

    cifar_path = zero_shot_dir / f"cifar10_epoch_1.json"
    if cifar_path.is_file():
        try:
            out["cifar10_acc1"] = read_json(cifar_path).get("metrics", {}).get("acc1")
        except Exception:
            pass

    imagenet_path = zero_shot_dir / f"imagenet1k_epoch_1.json"
    if imagenet_path.is_file():
        try:
            out["imagenet1k_acc1"] = read_json(imagenet_path).get("metrics", {}).get("acc1")
        except Exception:
            pass

    return out


def extract_mscoco_retrieval(root_dir: Path) -> dict:
    retrieval_dir = root_dir / "retrieval"

    out = {
        "mscoco_image_recall@5": None,
        "mscoco_text_recall@5": None,
    }

    mscoco_path = retrieval_dir / f"mscoco_captions_epoch_1.json"
    if mscoco_path.is_file():
        try:
            metrics = read_json(mscoco_path).get("metrics", {})
            out["mscoco_image_recall@5"] = metrics.get("image_retrieval_recall@5")
            out["mscoco_text_recall@5"] = metrics.get("text_retrieval_recall@5")
        except Exception:
            pass

    return out


def fetch_downstream_eval_results(path):
    zero_shot_results = extract_zeroshot_acc1(path)
    retrieval_results = extract_mscoco_retrieval(path)
    return {**zero_shot_results, **retrieval_results}


def fetch_scale_parameters(config_dir: Path) -> dict:
    config_path = config_dir / "config.yaml"
    if not config_path.is_file():
        return {}
    with config_path.open("r") as f:
        return yaml.safe_load(f) or {}


def fetch_total_compute_and_nparams(config_dir: Path) -> dict:
    config_path = config_dir / "report.yaml"
    if not config_path.is_file():
        return {}
    with config_path.open("r") as f:
        data = yaml.safe_load(f) or {}

    return {
        "Total compute(GLOPs)": data.get("cost"),
        "mparams": data.get("extra", {}).get("total_mparams"),
        "vision_mparams": data.get("extra", {}).get("image_mparams"),
        "text_mparams": data.get("extra", {}).get("text_mparams")
    }


def fetch_training_gpu_used_gpu_training_time(checkpoints_dir: Path) -> dict:
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


if __name__ == "__main__":
    args = parse_args()

    root_dir: Path = args.root_dir
    train_logs_root = root_dir / "individual_training_logs"
    configs_dir = root_dir / "configs"
    out_csv = root_dir / "configs_summary.csv"

    full_df, _ = neps.status(str(root_dir))
    trial_ids = [int(i) for i in full_df.index.tolist()]
    completed_trial_ids = []
    rows = []

    for trial_id in trial_ids:
        trial_dir = train_logs_root / f"train_config{trial_id}"
        checkpoints_dir = trial_dir / "checkpoints"
        downstream_eval_path = trial_dir / "eval"
        config_dir = configs_dir / f"config_{trial_id}"

        if (checkpoints_dir / "epoch_1.pt").is_file():
            completed_trial_ids.append(trial_id)

        row = {
            "config_id": trial_id,
            **fetch_losses(checkpoints_dir),
            **fetch_downstream_eval_results(downstream_eval_path),
            **fetch_scale_parameters(config_dir),
            **fetch_total_compute_and_nparams(config_dir),
            **fetch_training_gpu_used_gpu_training_time(checkpoints_dir),
            # fetch gpu used during evaluations, time it took
        }
        rows.append(row)

    failed_trial_ids = sorted(set(trial_ids) - set(completed_trial_ids))
    print("Failed trials:", failed_trial_ids)

    df = pd.DataFrame(rows).set_index("config_id")
    df.to_csv(out_csv)
    print("Summary csv file has been written to:", out_csv)
