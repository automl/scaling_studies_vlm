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
        required=True,
    )

    parser.add_argument(
        "--command_dir",
        type=str,
        help="Directory where the train code is located (for PYTHONPATH).",
        required=True,
    )

    parser.add_argument(
        "--cluster",
        type=str,
        choices=["meta", "juwels", "horeka"],
        required=True,
        help="Cluster the jobs will be running on. Necessary for data paths.",
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
        required=True,
        help="which partition to use for training",
    )

    parser.add_argument(
        "--worker_id",
        type=int,
        default=0,
        help="worker id.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,  # match juwels
        help="Number of GPUs that will be used to generate scripts.",
    )

    parser.add_argument(
        "--bias_scale_sampling",
        default=False,
        action="store_true",
        help="Whether to bias sampling for scale space, towards smaller spaces."
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="parameter for biased scale sampling, check utils.py sample_warped_log_uniform() method for a brief description",
    )

    parser.add_argument(
        "--group_by_train_num_samples",
        default=False,
        action="store_true",
        help="Whether to group the configurations by their train_num_samples, and set train bigger train_num_samples first and set checkpoint_path to the checkpoint of the previous smaller train_num_samples configuration." \
        "Useful incase you have set warmup_fraction to 0, then no need to train the same model multiple times",
    )

    parser.add_argument(
        "--venv_path",
        type=str,
        default=None,
        help="Path to the python virtual environment to use.",
    )

    args = parser.parse_args(args)
    return args
