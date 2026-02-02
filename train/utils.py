import bisect
import json
import re
import tempfile
from enum import Enum
from glob import glob
from pathlib import Path
from typing import Union, Dict, Any

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from open_clip import create_model


def successful_downloaded_pairs_in_shards():
    stats_files = glob("../pre_training_dataset/laion400m/shards/*.json")

    total_successes = 0

    for path in stats_files:
        with open(path, "r") as f:
            data = json.load(f)
            total_successes += data.get("successes", 0)

    print("Total successes:", total_successes)


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


def create_default_model(model_str) -> torch.nn.Module:
    return create_model(
        model_str,
        precision="amp",
        output_dict=True
    )


def get_local_model_str(model_cfg_dir: Union[str, Path]):
    model_cfg_dir = Path(model_cfg_dir)
    p = model_cfg_dir.as_posix().replace("train/", "")
    return f"local-dir:{p}"


def build_model_cfg(
        *,
        text_width: int,
        vision_width: int,
        out_path: Union[str, Path],
        embed_dim: int = 384,
        image_size: int = 224,
        vision_layers: int = 12,
        patch_size: int = 32,
        context_length: int = 77,
        vocab_size: int = 49408,
        text_heads: int = 8,
        text_layers: int = 12,
        **kwargs
):
    filename = "open_clip_config.json"
    if isinstance(out_path, str):
        out_path = Path(out_path)
    out_file = out_path / filename

    if out_file.exists():
        return None

    cfg: Dict[str, Any] = {
        "model_cfg": {
            "embed_dim": embed_dim,
            "vision_cfg": {
                "image_size": image_size,
                "layers": vision_layers,
                "width": vision_width,
                "patch_size": patch_size,
            },
            "text_cfg": {
                "context_length": context_length,
                "vocab_size": vocab_size,
                "width": text_width,
                "heads": text_heads,
                "layers": text_layers,
            },
        }
    }

    out_file.parent.mkdir(parents=True, exist_ok=True)

    with out_file.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")


def build_model_cfg_temp(
        *,
        text_width: int,
        vision_width: int,
        embed_dim: int = 384,
        image_size: int = 224,
        vision_layers: int = 12,
        patch_size: int = 32,
        context_length: int = 77,
        vocab_size: int = 49408,
        text_heads: int = 8,
        text_layers: int = 12,
        **kwargs
) -> str:
    """
    Creates a model config and writes it to a temporary directory.
    Returns the path to the temporary directory (for use with get_local_model_str).
    """
    cfg: Dict[str, Any] = {
        "model_cfg": {
            "embed_dim": embed_dim,
            "vision_cfg": {
                "image_size": image_size,
                "layers": vision_layers,
                "width": vision_width,
                "patch_size": patch_size,
            },
            "text_cfg": {
                "context_length": context_length,
                "vocab_size": vocab_size,
                "width": text_width,
                "heads": text_heads,
                "layers": text_layers,
            },
        }
    }

    temp_dir = tempfile.mkdtemp()
    config_file = Path(temp_dir) / "open_clip_config.json"
    
    with config_file.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    
    return temp_dir



def read_last_json_line(path):
    last = None
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                last = line
    return json.loads(last)

def sample_warped_log_uniform(
        x,
        alpha: float = 1.0,
):
    """
    alpha = 1.0  -> standard log-uniform
    alpha > 1.0  -> bias toward smaller values
    alpha < 1.0  -> bias toward larger values
    """

    return 0.6 * (60 / 0.6) ** (x ** alpha)


def map_uniform_to_range(u, lower=0.6, upper=60.0):
    return lower + u * (upper - lower)


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


####################################################################
# methods for batch size cache
####################################################################

def _cache_key(partition: str, model_name: str) -> str:
    return f"{partition}_{model_name}"


def _read_cached_bs(cache_dir: Path, key: str) -> int | None:
    p = cache_dir / f"{key}.json"
    if not p.exists():
        return None
    return int(json.loads(p.read_text())["max_bs"])


def _write_cached_bs(cache_dir: Path, key: str, max_bs: int) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {"max_bs": int(max_bs)}
    (cache_dir / f"{key}.json").write_text(json.dumps(payload, indent=2))
