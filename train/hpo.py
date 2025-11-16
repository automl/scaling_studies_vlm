import os
import sys
from pathlib import Path
from time import time
from typing import Dict, Optional, Union

import neps
from neps.optimizers import GridSearch
from neps.search_spaces.search_space import pipeline_space_from_yaml, SearchSpace

from train.open_clip_train import params
from train.slurm import SlurmManager, build_cmd


def train_and_evaluate(learning_rate: float, root_dir: str, args) -> Dict[str, Optional[float]]:
    """
    Run one HPO trial.
    """
    logs_dir = os.path.join(root_dir, "individual_training_logs")
    os.makedirs(logs_dir, exist_ok=True)

    args.lr = learning_rate
    args.logs = logs_dir

    cmd = build_cmd(args, learning_rate, logs_dir)

    slurm = SlurmManager(root_dir=root_dir)
    slurm.submit_job(cmd)

    return {"loss": 1.0}


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
    args = params.parse_args(sys.argv[1:])

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
        overwrite_working_directory=False,
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
