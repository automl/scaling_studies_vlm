import json
from glob import glob

if __name__ == "__main__":
    # first find the number of total pairs in all shards
    stats_files = glob("../pre_training_dataset/laion400m/shards/*.json")

    total_successes = 0

    for path in stats_files:
        with open(path, "r") as f:
            data = json.load(f)
            total_successes += data.get("successes", 0)

    print("Total successes:", total_successes)
