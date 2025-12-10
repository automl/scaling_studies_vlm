import json
import os
import sys
from pathlib import Path
from time import time
from typing import Dict, Any

import neps

import multiprocessing as mp
from train import params
from train.config import GRID_SPACE, TrainConfig
from train.slurm import SlurmManager, build_cmd


def fetch_result(root_dir, pipeline_id):
    sync_dir = Path(root_dir) / "sync_results"
    result_file = sync_dir / f"{pipeline_id}.json"

    with open(result_file, "r") as f:
        return json.load(f)


def train_and_evaluate(hps: Dict[str, Any], root_dir: str, pipeline_id, args):
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
        pipeline_id=pipeline_id,
        sync_neps=args.sync_neps,
        logs=logs_dir,
        **hps,
    )

    cmd = build_cmd(config)

    job_name = f"openclip_train:{config.model}_ns{config.train_num_samples}_e{config.epochs}_s{config.seed}"

    slurm = SlurmManager(job_name=job_name, partition=args.partition, nodes=args.nodes,
                         gpus_per_node=args.gpus_per_node,
                         root_dir=root_dir)
    slurm.submit_job(cmd)

    if args.sync_neps:
        result = fetch_result(root_dir, pipeline_id)
        return result
    return None


def main(args, root_dir):
    def pipeline(pipeline_id, **config):
        return train_and_evaluate(hps=config, root_dir=root_dir, pipeline_id=pipeline_id, args=args)

    neps.run(
        evaluate_pipeline=pipeline,
        root_directory=root_dir,
        **GRID_SPACE.run_args,
    )


def worker_proc(rank, args, root_dir):
    print(f"Starting NePS worker {rank}", flush=True)
    main(args, root_dir)


if __name__ == "__main__":
    args = params.parse_args(sys.argv[1:])

    root_dir = f"./train/results/model={args.model}/seed={args.seed}"

    print("Starting HPO...")
    start = time()

    processes = []
    for rank in range(args.num_neps_workers):
        p = mp.Process(target=worker_proc, args=(rank, args, root_dir))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    end = time()
    elapsed = end - start

    with open(os.path.join(root_dir, "time.txt"), "w", encoding="utf-8") as f:
        f.write(f"{elapsed}")
    print(f"Finished HPO in {elapsed:.2f} seconds")
