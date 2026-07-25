import argparse
from pathlib import Path

import pandas as pd

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

    df = pd.read_csv("surrogate_data.csv")

    df = df[df["config_id"] < 1001] # do not take into account the new runs

    df["gpu_hours"] = df["train_duration(s)"] * df["world_size"] / 3600
    result = df.groupby("train_gpu_name")["gpu_hours"].sum()
    print(result)
