import argparse


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root_dir",
        type=str,
        default="",
        help="NePs root directory to iterate over.",
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

    args = parser.parse_args(args)
    return args
