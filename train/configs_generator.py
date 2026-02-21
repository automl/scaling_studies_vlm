import logging
import sys
from time import time

import neps

import params
from clip_train.main import random_seed
from config import HPO_CONFIGS
from optimizers import set_workspace_dir


def main(args, root_dir):
    # Set workspace directory for disk-based FLOP cache
    set_workspace_dir(args.workspace_dir)
    
    neps.run(
        evaluate_pipeline=lambda **kwargs: None,
        root_directory=root_dir,
        **HPO_CONFIGS.get(args.hpo_optimizer).run_args,
    )


def worker_proc(rank, args, root_dir):
    print(f"Starting NePS worker {rank}", flush=True)
    main(args, root_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    args = params.parse_args(sys.argv[1:])
    random_seed(args.seed, 0)

    root_dir = f"{args.workspace_dir}/train/results/optimizer={args.hpo_optimizer}/seed={args.seed}"

    print("Starting HPO...")
    start = time()

    main(args, root_dir)

    end = time()
    elapsed = end - start

    print(f"Finished HPO in {elapsed:.2f} seconds")
