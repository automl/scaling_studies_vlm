import bisect
import json
import re
from enum import Enum
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt


def successful_downloaded_pairs_in_shards():
    stats_files = glob("../pre_training_dataset/laion400m/shards/*.json")

    total_successes = 0

    for path in stats_files:
        with open(path, "r") as f:
            data = json.load(f)
            total_successes += data.get("successes", 0)

    print("Total successes:", total_successes)


def unwrap_model(model):
    """Unwrapping for DDP"""
    if hasattr(model, "module"):
        return unwrap_model(model.module)

    return model


def find_latest_checkpoint(checkpoints_dir: Path) -> str:
    """
    Finds the newest checkpoint (epoch_X.pt) inside the directory.
    """
    checkpoints = sorted(
        checkpoints_dir.glob("epoch_*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if not checkpoints:
        raise FileNotFoundError(f"No epoch_*.pt files found in {checkpoints_dir}")
    latest = checkpoints[0]
    return latest.name


def _build_output_name(dataset: str, epoch: int) -> str:
    return f"{dataset}_epoch_{epoch}.json"


####################################################################
# methods for batch size heuristics
####################################################################

def monotonic_bs(total_compute, batch_size, buffer_gflops=5.0):
    order = np.argsort(total_compute)
    c_sorted = np.array(total_compute, dtype=float)[order]
    bs_sorted = np.array(batch_size, dtype=int)[order]

    bs_mono = np.maximum.accumulate(bs_sorted)

    ranges = []
    start = 0.0
    current_bs = int(bs_mono[0])

    for k in range(1, len(c_sorted)):
        if int(bs_mono[k]) != current_bs:
            boundary = float(c_sorted[k]) + buffer_gflops
            ranges.append({"min": start, "max": boundary, "bs": current_bs})
            start = boundary
            current_bs = int(bs_mono[k])

    ranges.append({"min": start, "max": None, "bs": current_bs})

    with open("bs_ranges.json", "w", encoding="utf-8") as f:
        json.dump(ranges, f, indent=2)

    return ranges


def load_ranges(path="bs_ranges.json"):
    with open(path, "r", encoding="utf-8") as f:
        ranges = json.load(f)

    ranges.sort(key=lambda r: float(r["min"]))
    return ranges


def query_bs(compute_value):
    ranges = load_ranges()
    ends = [r["max"] for r in ranges if r["max"] is not None]
    idx = bisect.bisect_right(ends, float(compute_value))
    return int(ranges[idx]["bs"])


####################################################################
# methods for insights
####################################################################
class Plotting(Enum):
    VALIDATION_ERROR = "validation_error"
    TEST_ERROR = "test_error"
    IMAGENET = "imagenet1k"
    MSCOCO_IMG = "mscoco_img"


def parse_samples(x):
    x = str(x).lower().strip()
    if x.endswith("m"):
        return float(x[:-1]) * 1e6
    return float(x)


def extract_imagenet_acc(x):
    match = re.search(r"IN1k acc1=([0-9.]+)", x)
    return float(match.group(1)) if match else None


def extract_mscoco_img(x):
    match = re.search(r"img=([0-9.]+)", x)
    return float(match.group(1)) if match else None


def resolve_metric_value(row, plotting: Plotting):
    if plotting == Plotting.VALIDATION_ERROR:
        return row["Validation error"]
    if plotting == Plotting.TEST_ERROR:
        return row["Test error"]
    if plotting == Plotting.IMAGENET:
        return extract_imagenet_acc(row["Downstream performance"])
    if plotting == Plotting.MSCOCO_IMG:
        return extract_mscoco_img(row["Downstream performance"])

    raise ValueError(f"Unknown plotting: {plotting}")


def plot_compute_vs_metric(runs_config_path, plotting: Plotting):
    df = pd.read_csv(runs_config_path)

    df["samples"] = df["Nr. Samples trained on"].apply(parse_samples)
    df["compute"] = df["samples"] * df["GFLOPs per sample"]

    df["metric_value"] = df.apply(lambda row: resolve_metric_value(row, plotting), axis=1)

    if plotting in {Plotting.IMAGENET, Plotting.MSCOCO_IMG}:
        df["metric_value"] = 1 - df["metric_value"]

    df["metric_value"] = df["metric_value"].clip(lower=1e-12)

    plt.figure(figsize=(7, 5))

    plt.loglog(df["compute"], df["metric_value"], "o-", markersize=8)

    for _, row in df.iterrows():
        label = f"{row['Model']} ({row['compute']:.2e})"
        plt.annotate(label,
                     (row["compute"], row["metric_value"]),
                     xytext=(5, 5),
                     textcoords="offset points")

    label_map = {
        Plotting.VALIDATION_ERROR: "Validation Error",
        Plotting.TEST_ERROR: "Test Error",
        Plotting.IMAGENET: "ImageNet Loss [Error rate]",
        Plotting.MSCOCO_IMG: "MSCOCO Image Retrieval R@5 [Error rate]"
    }

    plt.xlabel("Compute (samples × GFLOPs)")
    plt.ylabel(label_map[plotting])
    plt.title(f"Compute vs {label_map[plotting]}")
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


def precision_to_dtype(precision: str) -> torch.dtype | None:
    p = precision.lower()

    if p == "fp32":
        return torch.float32

    if p in {"fp16", "pure_fp16"}:
        return torch.float16

    if p in {"bf16", "pure_bf16", "amp_bf16", "amp_bfloat16", "amp_bfloat16"}:
        return torch.bfloat16

    if p == "amp":
        return None

    raise ValueError(f"Unknown precision: {precision}")
