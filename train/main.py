import json
import os
import subprocess
import sys
from pathlib import Path
from time import time
from typing import Dict, Any

import neps

import multiprocessing as mp

from neps import plot

from train import params
from train.config import GRID_SPACE, TrainConfig
from train.slurm import SlurmManager, build_srun_cmd, build_torchrun_cmd


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

    if args.sync_neps:
        # multi-gpu case is ignored!
        results = sync_submitter(config, args.gpus_per_node)
        return results

    async_submitter(root_dir, config, args)


def async_submitter(root_dir, train_config, args):
    cmd = build_srun_cmd(train_config)
    job_name = f"openclip_train:{train_config.model}_ns{train_config.train_num_samples}_e{train_config.epochs}_s{train_config.seed}"

    slurm = SlurmManager(job_name=job_name, partition=args.partition, nodes=args.nodes,
                         gpus_per_node=args.gpus_per_node,
                         root_dir=root_dir)
    slurm.submit_job(cmd)


def sync_submitter(train_config, nproc_per_node, root_dir):
    cmd = build_torchrun_cmd(train_config, nproc_per_node)
    os.chdir("train")
    process = subprocess.Popen(cmd)
    process.wait()
    os.chdir("..")

    return fetch_result(root_dir, pipeline_id=train_config.pipeline_id)


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
    plot(root_directory=root_dir)
