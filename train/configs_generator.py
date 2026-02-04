import multiprocessing as mp
import os
import sys
from time import time

import neps

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
    from chinchilla import chinchilla_optimizer_fine_grained_dp
    args = params.parse_args(sys.argv[1:])

    root_dir = f"/work/dlclarge1/sinanid-VLM-scaling-law/scaling_studies_vlm/train/results/optimizer={args.hpo_optimizer}/seed={args.seed}"

    neps.extrapolate(root_directory=root_dir, max_target_flops=10_000_000, optimizer=chinchilla_optimizer_fine_grained_dp)
    # print("Starting HPO...")
    # start = time()

    # processes = []
    # for rank in range(args.num_neps_workers):
    #     p = mp.Process(target=worker_proc, args=(rank, args, root_dir))
    #     p.start()
    #     processes.append(p)

    # for p in processes:
    #     p.join()

    # end = time()
    # elapsed = end - start

    # # with open(os.path.join(root_dir, "time.txt"), "w+", encoding="utf-8") as f:
    # #     f.write(f"{elapsed}")
    # print(f"Finished HPO in {elapsed:.2f} seconds")
