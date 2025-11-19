import os
import sys
from time import time
from typing import Dict, Optional, Any

import neps

from train import params
from train.config import GRID_SPACE, TrainConfig
from train.slurm import SlurmManager, build_cmd


def train_and_evaluate(hps: Dict[str, Any], root_dir: str, args) -> Dict[str, Optional[float]]:
    """
    Run one HPO trial.
    """
    logs_dir = os.path.join(root_dir, "individual_training_logs")
    os.makedirs(logs_dir, exist_ok=True)

    config = TrainConfig(
        model=args.model,
        train_num_samples=args.train_num_samples,
        epochs=args.epochs,
        seed=args.seed,
        logs=logs_dir,
        **hps,
    )

    cmd = build_cmd(config)

    job_name = f"openclip_train:{config.model}_ns{config.train_num_samples}_e{config.epochs}_s{config.seed}"


    slurm = SlurmManager(job_name=job_name, partition=args.partition, nodes=args.nodes, gpus_per_node=args.gpus_per_node,
                         root_dir=root_dir)
    slurm.submit_job(cmd)

    return {"loss": 1.0}


def main(args):
    root_dir = f"./train/results/model={args.model}/seed={args.seed}"

    print("Starting HPO...")
    start = time()

    def pipeline(**config):
        return train_and_evaluate(hps=config, root_dir=root_dir, args=args)

    neps.run(
        run_pipeline=pipeline,
        root_directory=root_dir,
        post_run_summary=True,
        overwrite_working_directory=False,
        **GRID_SPACE.run_args,
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


if __name__ == "__main__":
    args = params.parse_args(sys.argv[1:])
    main(args)
