import argparse
from pathlib import Path

import torch

from open_clip_train.profiler import get_model_flops
from utils import precision_to_dtype, create_default_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train_num_samples", type=int, required=True)
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--precision", default="bf16", type=str, required=False)
    parser.add_argument("--trial_id", type=int, required=True)
    args = parser.parse_args()

    if "train/" in args.root_dir:
        args.root_dir = args.root_dir.replace("train/", "")

    model = create_default_model(args.model)
    if torch.cuda.is_available():
        model = model.cuda()

    total_compute = get_model_flops(model, args.train_num_samples, dtype=precision_to_dtype(args.precision))

    root = Path(args.root_dir)
    output = root / f"trial_{args.trial_id}.txt"
    output.write_text(f"{total_compute}\n")


if __name__ == "__main__":
    main()
