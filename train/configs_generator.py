import multiprocessing as mp
import sys
from time import time

import neps
import neps.optimizers
import neps.optimizers.chinchilla_guided_scaling

import params
from config import HPO_CONFIGS


def main(args, root_dir):
    neps.run(
        evaluate_pipeline=lambda **kwargs: None,
        root_directory=root_dir,
        **HPO_CONFIGS.get(args.hpo_optimizer).run_args,
    )


def worker_proc(rank, args, root_dir):
    print(f"Starting NePS worker {rank}", flush=True)
    main(args, root_dir)


if __name__ == "__main__":
    args = params.parse_args(sys.argv[1:])

    root_dir = f"{args.workspace_dir}/train/results/optimizer={args.hpo_optimizer}/seed={args.seed}"

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

    print(f"Finished HPO in {elapsed:.2f} seconds")
