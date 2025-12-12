import json
from glob import glob
from pathlib import Path


def successful_downloaded_pairs_in_shards():
    stats_files = glob("../pre_training_dataset/laion400m/shards/*.json")

    total_successes = 0

    for path in stats_files:
        with open(path, "r") as f:
            data = json.load(f)
            total_successes += data.get("successes", 0)

    print("Total successes:", total_successes)


def unwrap_model(model):
    """Unwrapping for DDP"""
    if hasattr(model, "module"):
        return unwrap_model(model.module)

    return model


def find_latest_checkpoint(checkpoints_dir: Path) -> str:
    """
    Finds the newest checkpoint (epoch_X.pt) inside the directory.
    """
    checkpoints = sorted(
        checkpoints_dir.glob("epoch_*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if not checkpoints:
        raise FileNotFoundError(f"No epoch_*.pt files found in {checkpoints_dir}")
    latest = checkpoints[0]
    return latest.name


def _build_output_name(dataset: str, epoch: int) -> str:
    return f"{dataset}_epoch_{epoch}.json"
