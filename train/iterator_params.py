import argparse


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root_dir",
        type=str,
        help="NePs root directory to iterate over.",
        required=True,
    )

    parser.add_argument(
        "--workspace_dir",
        type=str,
        help="Directory to the workspace.",
        default="/work/dlclarge1/sinanid-VLM-scaling-law/scaling_studies_vlm",
    )

    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="scan-space",
        help="Wandb project name.",
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
