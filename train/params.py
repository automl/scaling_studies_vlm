import argparse


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hpo-optimizer",
        type=str,
        default="experimental",
        help="HPO optimizer type."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for reproducibility.",
    )

    # args for slurm
    parser.add_argument(
        "--partition",
        type=str,
        default="testdlc_gpu-rtx2080",
        help="which partition to use for training",
    )

    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Number of nodes for training.",
    )

    parser.add_argument(
        "--gpus-per-node",
        type=int,
        default=1,
        help="Number of gpus per node for training.",
    )

    # args for HPO
    parser.add_argument(
        "--num-neps-workers",
        type=int,
        default=1,
        help="Number of parallel workers.",
    )

    parser.add_argument(
        "--sync-neps",
        default=False,
        action="store_true",
        help="Whether to use sync neps.")

    parser.add_argument(
        "--sampling-only",
        default=False,
        action="store_true",
        help="Use NePs to only sample configurations, but do not train models.")

    args = parser.parse_args(args)
    return args
