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
        "--workspace_dir",
        type=str,
        help="Directory to the workspace.",
        default="/work/dlclarge1/sinanid-VLM-scaling-law/scaling_studies_vlm",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for reproducibility.",
    )

    args = parser.parse_args(args)
    return args
