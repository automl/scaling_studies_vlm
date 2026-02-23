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
from matplotlib.ticker import LogLocator, LogFormatterSciNotation
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
        vision_head_width: int = 64,
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
                "head_width": vision_head_width,
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
        vision_head_width: int = 64,
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
                "head_width": vision_head_width,
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
        x, low: float = 0.6, high: float = 60, alpha: float = 1.0,
):
    """
    alpha = 1.0  -> standard log-uniform
    alpha > 1.0  -> bias toward smaller values
    alpha < 1.0  -> bias toward larger values
    """
    x = (x - low) / (high - low)  # map to [0,1]
    return low * (high / low) ** (x ** alpha)  # apply inverse warped log uniform transform


def sample_small_favoring_index(n: int = 9, alpha: float = 1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    k = np.arange(n)
    weights = 1.0 / (k + 1) ** alpha
    probs = weights / weights.sum()

    return int(rng.choice(k, p=probs))


####################################################################
# methods for batch size heuristics
####################################################################

def monotonic_bs(samples_seen, batch_size):
    order = np.argsort(samples_seen)
    s_sorted = np.array(samples_seen, dtype=float)[order]
    bs_sorted = np.array(batch_size, dtype=int)[order]

    bs_mono = np.maximum.accumulate(bs_sorted)

    ranges = []
    start = 0.0
    current_bs = int(bs_mono[0])

    for k in range(1, len(s_sorted)):
        if int(bs_mono[k]) != current_bs:
            prev_s = float(s_sorted[k - 1])
            next_s = float(s_sorted[k])

            alpha = 0.1
            boundary = prev_s + alpha * (next_s - prev_s)

            ranges.append({
                "min": start,
                "max": boundary,
                "bs": current_bs
            })

            start = boundary
            current_bs = int(bs_mono[k])

    ranges.append({"min": start, "max": None, "bs": current_bs})
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


####################################################################
# methods for GFLOPs to GPU hours mapping
####################################################################

def add_gpu_hours_column(
        df: pd.DataFrame,
        *,
        world_size_col: str = "world_size",
        duration_s_col: str = "train_duration(s)",
        out_col: str = "gpu_hours",
) -> pd.DataFrame:
    df = df.copy()
    df[out_col] = df[world_size_col] * (df[duration_s_col] / 3600.0)
    return df


def plot_measured_vs_predicted_gpu_hours(
        df: pd.DataFrame,
        *,
        gpu_name: str,
        model,
        compute_col: str = "Total compute(GLOPs)",
        measured_gpu_hours_col: str = "gpu_hours",
        x_query=None,
        xlim=(1e7, 1e9),
        title=None,
        n_fit_points: int = 200,
):
    df_gpu = df[df["train_gpu_name"] == gpu_name]
    if df_gpu.empty:
        raise ValueError(f"No rows for gpu_name {gpu_name!r}")

    if x_query is None:
        x_query = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 6)

    y_pred = model.gpu_hours(x_query, k=3, mode="global_linear")

    x_fit = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), n_fit_points)
    y_fit = model.gpu_hours(x_fit, k=3, mode="global_linear")

    plt.figure(figsize=(8, 5))
    plt.scatter(
        df_gpu[compute_col],
        df_gpu[measured_gpu_hours_col],
        alpha=0.8,
        label="Measured",
    )
    plt.scatter(
        x_query,
        y_pred,
        marker="x",
        s=80,
        linewidths=1,
        label="Predicted points",
    )
    plt.plot(
        x_fit,
        y_fit,
        linewidth=0.5,
        label="Fitted curve",
    )

    plt.xscale("log")
    plt.xlim(*xlim)

    ax = plt.gca()
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_major_formatter(LogFormatterSciNotation())

    plt.xlabel("Total compute (GFLOPs)")
    plt.ylabel(f"GPU hours ({gpu_name})")
    plt.title(title or f"Total compute vs GPU hours for {gpu_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    return (x_query, y_pred), (x_fit, y_fit)
