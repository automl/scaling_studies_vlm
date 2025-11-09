import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="Arguments for OpenClip model training")

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Local batch size per GPU during training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs to train.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=6,
        help="Number of data loader workers per GPU.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="ViT-S-16",
        choices=["ViT-S-16", "ViT-B-16"],
        help="Vision tower to use for training.",
    )

    parser.add_argument(
        "--train_num_samples",
        type=int,
        default=1754784,
        help="Number of training samples. Required for WebDataset if not available in metadata.",
    )

    return parser
