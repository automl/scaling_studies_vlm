import os
import re
import shlex
import subprocess
from pathlib import Path
from time import time
from typing import Dict, Optional, Union

import neps
from neps.optimizers import GridSearch
from neps.search_spaces.search_space import pipeline_space_from_yaml, SearchSpace

from params import get_parser

SHARDS = "../pre_training_dataset/laion400m/shards/{00000000..00000282}.tar"

def build_openclip_command(
    learning_rate: float,
    logs_dir: str,
    args,
) -> str:
    """
    Construct the OpenCLIP training command as a single string.
    """
    cmd = (
        f"srun --cpu_bind=v --accel-bind=gn python "
        f"-m open_clip_train.main "  
        f'--train-data="{SHARDS}" '
        f"--batch-size={args.batch_size} "
        f"--epochs={args.epochs} "
        f"--workers={args.num_workers} "
        f"--model {args.model} "
        f"--seed {args.seed} "
        f"--train-num-samples {args.train_num_samples} "
        f"--lr {learning_rate} "
        f"--local-loss "
        f"--gather-with-grad "
        f"--dataset-type=webdataset "
        f'--logs="{logs_dir}" '
    )
    return cmd


def get_latest_contrastive_loss(log_path: str):
    """
    Reads a log file and returns the most recent 'Contrastive_loss' value.
    """
    pattern = re.compile(r"Contrastive_loss[:=]\s*([0-9]*\.?[0-9]+)")
    last_value = None

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                last_value = float(match.group(1))

    return last_value


def get_latest_folder_name(parent_dir: str) -> str | None:
    """
    Returns the name of the most recently modified folder
    inside the given parent directory.
    """
    folders = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]
    if not folders:
        return None
    latest_folder = max(folders, key=lambda f: os.path.getmtime(os.path.join(parent_dir, f)))
    return latest_folder


def train_and_evaluate(learning_rate: float, root_dir: str, args) -> Dict[str, Optional[float]]:
    """
    Run one HPO trial.
    """
    logs_dir = os.path.join(root_dir, "individual_training_logs")
    os.makedirs(logs_dir, exist_ok=True)

    cmd = build_openclip_command(learning_rate=learning_rate, logs_dir=logs_dir, args=args)
    print("Running command:\n", cmd)

    proc = subprocess.Popen(
        shlex.split(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")

    proc.wait()

    if proc.returncode != 0:
        print(f"Training process exited with code {proc.returncode}.")

    latest_folder = get_latest_folder_name(logs_dir)
    if latest_folder is None:
        print("No training log folder found.")
        return {"loss": float("nan")}

    out_log = os.path.join(logs_dir, latest_folder, "out.log")
    loss = get_latest_contrastive_loss(out_log)
    if loss is None:
        print(f"Could not parse Contrastive_loss from {out_log}")
        return {"loss": float("nan")}

    return {"loss": loss}


def resolve_search_space(path: Union[str, Path]) -> SearchSpace:
    """
    Load a NEPS search space from YAML. Accepts either the raw dict format
    or the dict with a 'pipeline_space' key, and returns a SearchSpace object.
    """
    space = pipeline_space_from_yaml(path)
    if isinstance(space, SearchSpace):
        return space
    if isinstance(space, dict):
        if "pipeline_space" in space:
            return SearchSpace(**space["pipeline_space"])
        return SearchSpace(**space)
    raise TypeError(f"Unsupported search space type: {type(space)}")


if __name__ == "__main__":
    args = get_parser().parse_args()

    root_dir = f"./results/model={args.model}/seed={args.seed}"
    os.makedirs(root_dir, exist_ok=True)

    space = resolve_search_space("search_space.yaml")
    searcher = GridSearch(space)

    print("Starting HPO...")
    start = time()

    neps.run(
        run_pipeline=lambda learning_rate: train_and_evaluate(learning_rate, root_dir, args),
        pipeline_space="./search_space.yaml",
        root_directory=root_dir,
        max_evaluations_total=10,
        post_run_summary=True,
        overwrite_working_directory=True,
        searcher=searcher,
    )

    end = time()
    elapsed = end - start

    with open(os.path.join(root_dir, "time.txt"), "w", encoding="utf-8") as f:
        f.write(f"{elapsed}")
    print(f"Finished HPO in {elapsed:.2f} seconds")

    best_config = neps.get_summary_dict(root_dir)["best_config"]
    print("-" * 80)
    print(f"Best config: {best_config}")
    print("-" * 80)
