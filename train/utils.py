import json
from glob import glob


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
