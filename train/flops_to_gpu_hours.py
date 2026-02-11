from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Union, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

Number = Union[int, float, np.number]


@dataclass(frozen=True)
class GpuSpec:
    name: str
    peak_gflops: float


@dataclass(frozen=True)
class BsRule:
    min_compute: float
    max_compute: float
    bs: int


class GpuHoursModel:
    """
        Batch size aware GPU-hours model:
        - Fit one MFU per batch-size bucket (defined by bs_rules over compute ranges)
        - For a query compute, select the right MFU bucket
    """

    def __init__(self, spec: GpuSpec, bs_rules: List[BsRule], mfu_by_bs: Dict[int, float]):
        self.spec = spec
        self.bs_rules = bs_rules
        self.mfu_by_bs = mfu_by_bs  # For a GPU, how efficient was I at each batch size range

    @staticmethod
    def _weighted_mfu(df: pd.DataFrame, *, compute_col: str, duration_s_col: str, world_size_col: str,
                      peak_gflops: float) -> float:
        total_compute = df[compute_col].sum()
        total_peak = peak_gflops * (df[duration_s_col] * df[world_size_col]).sum()
        return float(total_compute / total_peak)

    @classmethod
    def fit_from_df(
            cls,
            df: pd.DataFrame,
            gpu_name: str,
            *,
            peak_gflops: float,
            bs_rules: List[dict],
            gpu_name_col: str = "train_gpu_name",
            compute_col: str = "Total compute(GLOPs)",
            duration_s_col: str = "train_duration(s)",
            world_size_col: str = "world_size",
    ) -> "GpuHoursModel":
        df_gpu = df[df[gpu_name_col] == gpu_name].copy()
        if df_gpu.empty:
            raise ValueError(f"No rows found for gpu_name {gpu_name!r}")

        rules = [BsRule(r["min"], r["max"], r["bs"]) for r in bs_rules]

        mfu_by_bs: Dict[int, float] = {}
        for r in rules:
            bucket = df_gpu[(df_gpu[compute_col] >= r.min_compute) & (df_gpu[compute_col] < r.max_compute)]
            if bucket.empty:
                continue
            mfu_by_bs[r.bs] = cls._weighted_mfu(
                bucket,
                compute_col=compute_col,
                duration_s_col=duration_s_col,
                world_size_col=world_size_col,
                peak_gflops=peak_gflops,
            )

        if not mfu_by_bs:
            raise ValueError(
                f"No MFU buckets could be fit for gpu_name {gpu_name!r}. "
            )

        return cls(GpuSpec(gpu_name, float(peak_gflops)), rules, mfu_by_bs)

    def _select_bs(self, total_gflops: float) -> int:
        for r in self.bs_rules:
            if r.min_compute <= total_gflops < r.max_compute and r.bs in self.mfu_by_bs:
                return r.bs
        available = [r for r in self.bs_rules if r.bs in self.mfu_by_bs]
        if not available:
            raise ValueError("No available MFU buckets.")
        mid = np.array([(r.min_compute + r.max_compute) / 2 for r in available], dtype=float)
        idx = int(np.argmin(np.abs(mid - float(total_gflops))))
        return available[idx].bs

    def mfu_for(self, total_gflops: Number) -> float:
        bs = self._select_bs(float(total_gflops))
        if bs in self.mfu_by_bs:
            return self.mfu_by_bs[bs]
        # if bucket is missing then fallback to mean mfu of existing buckets
        return float(np.mean(list(self.mfu_by_bs.values())))

    def gpu_hours(self, total_gflops: Union[Number, np.ndarray]) -> Union[float, np.ndarray]:
        x = np.asarray(total_gflops, dtype=float)
        mfus = np.array([self.mfu_for(v) for v in x.ravel()], dtype=float).reshape(x.shape)
        hours = (x / (self.spec.peak_gflops * mfus)) / 3600.0
        return float(hours) if np.isscalar(total_gflops) else hours

    def mapping_gflops_to_gpu_hours(
            self,
            gflops_values: Iterable[Number],
            *,
            as_dict: bool = True,
    ):
        xs = np.array(list(gflops_values), dtype=float)
        ys = self.gpu_hours(xs)
        if as_dict:
            return {float(x): float(y) for x, y in zip(xs, ys)}
        return pd.Series(ys, index=xs, name="gpu_hours")

    def debug_bucket_table(self) -> pd.DataFrame:
        rows = []
        for r in self.bs_rules:
            if r.bs in self.mfu_by_bs:
                rows.append(
                    {"bs": r.bs, "min_compute": r.min_compute, "max_compute": r.max_compute,
                     "mfu": self.mfu_by_bs[r.bs]}
                )
        return pd.DataFrame(rows).sort_values("bs")


WIDTH_LEVELS = list(range(64, 513, 64))

TRAIN_SAMPLES_MIN = 0.6e6
TRAIN_SAMPLES_MAX = 60e6

COMPUTE_MIN = 1e6
COMPUTE_MAX = 1e9

DEFAULT_COMPUTE_TICKS = (1e6, 1e7, 1e8, 1e9)
DEFAULT_TRAIN_TICKS = (0.6e6, 1e6, 2e6, 5e6, 1e7, 2e7, 6e7)


def _log_edges(xmin: float, xmax: float, bins: int) -> np.ndarray:
    xmin = max(float(xmin), 1e-12)
    xmax = max(float(xmax), xmin * 1.000001)
    return np.logspace(np.log10(xmin), np.log10(xmax), bins + 1)


def _linear_edges(xmin: float, xmax: float, bins: int) -> np.ndarray:
    xmin = float(xmin)
    xmax = float(xmax)
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        return np.linspace(0.0, 1.0, bins + 1)
    return np.linspace(xmin, xmax, bins + 1)


def _fmt_sci(x: float, _pos: int) -> str:
    return f"{x:.0e}"


def _fmt_millions(x: float, _pos: int) -> str:
    return f"{x / 1e6:g}M"


@dataclass(frozen=True)
class FeatureSpec:
    col: str
    label: Optional[str] = None
    kind: str = "auto"
    bins: int = 25
    log_range: Optional[Tuple[float, float]] = None
    fixed_ticks: Optional[Sequence[float]] = None
    ordered_categories: Optional[Sequence[int]] = None


class SamplingStrategyPlotter:

    def __init__(self, dfs: Dict[str, pd.DataFrame]):
        self.dfs = dfs

    def plot_features_grid(
            self,
            specs: Sequence[FeatureSpec],
            density: bool = False,
            title: str = "Distributions per sampling strategy",
            figsize_per_cell: Tuple[float, float] = (4.4, 2.9),
    ) -> None:
        strategy_items = list(self.dfs.items())
        n_rows = len(strategy_items)
        n_cols = len(specs)

        fig_w = figsize_per_cell[0] * n_cols
        fig_h = figsize_per_cell[1] * n_rows
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

        for r, (strategy_name, df) in enumerate(strategy_items):
            for c, spec in enumerate(specs):
                ax = axes[r][c]
                if spec.col not in df.columns:
                    ax.set_axis_off()
                    continue

                kind = spec.kind if spec.kind != "auto" else self._infer_kind(spec)
                label = spec.label or spec.col

                if kind == "discrete_bar":
                    self._plot_discrete_bar(ax, df, spec, density=density)
                elif kind == "log_hist":
                    self._plot_log_hist(ax, df, spec, density=density)
                elif kind == "hist":
                    self._plot_hist(ax, df, spec, density=density)
                else:
                    raise ValueError(f"Unknown kind {kind} for col {spec.col}")

                if r == 0:
                    ax.set_title(label)

                if c == 0:
                    ax.annotate(
                        strategy_name,
                        xy=(0, 0.5),
                        xycoords="axes fraction",
                        xytext=(-ax.yaxis.labelpad - 28, 0),
                        textcoords="offset points",
                        ha="right",
                        va="center",
                        rotation=90,
                    )

                if r == n_rows - 1:
                    ax.set_xlabel(label)

        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

    def plot_default_suite(
            self,
            compute_col: str = "total_compute",
            vision_col: str = "vision_width",
            text_col: str = "text_width",
            train_samples_col: str = "train_num_samples",
            bins_compute: int = 25,
            bins_train: int = 25,
            title: str = "Sampling strategy distributions",
            density: bool = False,
    ) -> None:
        specs = [
            FeatureSpec(
                col=compute_col,
                label="Total compute",
                kind="log_hist",
                bins=bins_compute,
                log_range=(COMPUTE_MIN, COMPUTE_MAX),
                fixed_ticks=DEFAULT_COMPUTE_TICKS,
            ),
            FeatureSpec(
                col=train_samples_col,
                label="Train samples",
                kind="log_hist",
                bins=bins_train,
                log_range=(TRAIN_SAMPLES_MIN, TRAIN_SAMPLES_MAX),
                fixed_ticks=DEFAULT_TRAIN_TICKS,
            ),
            FeatureSpec(
                col=vision_col,
                label="Vision width",
                kind="discrete_bar",
                ordered_categories=WIDTH_LEVELS,
            ),
            FeatureSpec(
                col=text_col,
                label="Text width",
                kind="discrete_bar",
                ordered_categories=WIDTH_LEVELS,
            ),
        ]
        self.plot_features_grid(specs=specs, density=density, title=title)

    def _global_positive_range(self, col: str) -> Tuple[float, float]:
        vals = []
        for df in self.dfs.values():
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce").to_numpy()
                s = s[np.isfinite(s)]
                s = s[s > 0]
                if s.size:
                    vals.append(s)
        if not vals:
            return 1.0, 10.0
        allv = np.concatenate(vals)
        return float(allv.min()), float(allv.max())

    def _infer_kind(self, spec: FeatureSpec) -> str:
        if spec.ordered_categories is not None:
            return "discrete_bar"
        if spec.log_range is not None:
            return "log_hist"
        lo, hi = self._global_positive_range(spec.col)
        if hi / max(lo, 1e-12) >= 1e3:
            return "log_hist"
        return "hist"

    def _plot_discrete_bar(self, ax, df: pd.DataFrame, spec: FeatureSpec, density: bool) -> None:
        cats = list(spec.ordered_categories or [])
        if not cats:
            raise ValueError(f"discrete_bar requires ordered_categories for col={spec.col}")

        s = pd.to_numeric(df[spec.col], errors="coerce").dropna().astype(int)
        counts = s.value_counts().reindex(cats, fill_value=0)

        if density:
            total = counts.sum()
            y = (counts / total).to_numpy() if total > 0 else counts.to_numpy(dtype=float)
            ax.set_ylabel("Density")
        else:
            y = counts.to_numpy()
            ax.set_ylabel("Count")

        ax.bar([str(c) for c in cats], y, alpha=0.85)
        ax.grid(True, axis="y", alpha=0.2)

    def _plot_log_hist(self, ax, df: pd.DataFrame, spec: FeatureSpec, density: bool) -> None:
        s = pd.to_numeric(df[spec.col], errors="coerce").dropna().to_numpy()
        s = s[np.isfinite(s)]
        s = s[s > 0]

        if spec.log_range is None:
            xmin, xmax = self._global_positive_range(spec.col)
        else:
            xmin, xmax = spec.log_range

        edges = _log_edges(xmin, xmax, spec.bins)
        ax.hist(s, bins=edges, density=density, alpha=0.85)
        ax.set_xscale("log")
        ax.set_ylabel("Density" if density else "Count")
        ax.grid(True, axis="x", alpha=0.25)

        if spec.fixed_ticks is not None:
            ax.xaxis.set_major_locator(FixedLocator(list(spec.fixed_ticks)))
            if spec.log_range == (TRAIN_SAMPLES_MIN, TRAIN_SAMPLES_MAX):
                ax.xaxis.set_major_formatter(FuncFormatter(_fmt_millions))
            else:
                ax.xaxis.set_major_formatter(FuncFormatter(_fmt_sci))

    def _plot_hist(self, ax, df: pd.DataFrame, spec: FeatureSpec, density: bool) -> None:
        s = pd.to_numeric(df[spec.col], errors="coerce").dropna().to_numpy()
        s = s[np.isfinite(s)]
        if s.size == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_axis_off()
            return

        xmin, xmax = (float(s.min()), float(s.max()))
        edges = _linear_edges(xmin, xmax, spec.bins)
        ax.hist(s, bins=edges, density=density, alpha=0.85)
        ax.set_ylabel("Density" if density else "Count")
        ax.grid(True, axis="x", alpha=0.25)
