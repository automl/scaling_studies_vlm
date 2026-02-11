from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Union, List

import numpy as np
import pandas as pd

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
