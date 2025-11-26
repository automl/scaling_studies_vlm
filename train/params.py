import argparse


def parse_args(args):
    parser = argparse.ArgumentParser()

    # args that override openclip properties
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-S-32",
        help="Name of the vision backbone to use.",
    )

    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=30,
        help="Number of samples in dataset.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs to train model for.",
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

    args = parser.parse_args(args)
    return args
