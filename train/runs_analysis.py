from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Sequence, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HP_COLS = ["lr", "wd", "warmup_fraction", "beta2", "eps"]


def load_runs_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    numeric_cols = [
        "trial_id",
        "lr", "wd", "warmup_fraction", "beta2", "eps",
        "val_loss", "test_loss", "imagenet1k_acc1",
        "mscoco_image_recall@5",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(subset=[c for c in numeric_cols if c in df.columns])


def add_log_hp(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "lr" in out.columns:
        out["log10_lr"] = np.log10(out["lr"].clip(lower=np.finfo(float).tiny))

    if "wd" in out.columns:
        out["log10_wd"] = np.log10(out["wd"].clip(lower=np.finfo(float).tiny))

    if "eps" in out.columns:
        out["log10_eps"] = np.log10(out["eps"].clip(lower=np.finfo(float).tiny))

    return out


class HyperparamEffectAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def available_hps(self) -> list[str]:
        hps: list[str] = []
        for hp in ["lr", "wd", "warmup_fraction", "grad_clip_norm", "beta2", "eps"]:
            if hp in self.df.columns:
                hps.append(hp)
        return hps

    def _resolve_x_col(self, hp: str) -> Tuple[str, str]:
        if hp == "lr":
            if "log10_lr" not in self.df.columns:
                raise ValueError("Missing column 'log10_lr'. Call add_log_hp(df) before plotting lr.")
            return "log10_lr", "log10(lr)"

        if hp == "wd":
            if "log10_wd" not in self.df.columns:
                raise ValueError("Missing column 'log10_wd'. Call add_log_hp(df) before plotting wd.")
            return "log10_wd", "log10(wd)"

        if hp == "warmup_fraction":
            return "warmup_fraction", "warmup_fraction"

        if hp == "grad_clip_norm":
            return "grad_clip_norm", "grad_clip_norm"

        if hp == "beta2":
            return "beta2", "beta2"

        if hp == "eps":
            if "log10_eps" in self.df.columns:
                return "log10_eps", "log10(eps)"
            return "eps", "eps"

        raise ValueError(f"Unknown hp: {hp}. Expected one of {self.available_hps()}.")

    def plot_effect_2x2(
            self,
            hp: str,
            out_path: Optional[Path] = None,
            title_prefix: str = "",
    ) -> None:
        x_col, x_label = self._resolve_x_col(hp)

        required = [x_col, "val_loss", "test_loss", "imagenet1k_acc1", "mscoco_image_recall@5"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(
                f"Missing columns for plotting: {missing}. "
                f"Available columns: {list(self.df.columns)}"
            )

        data = self.df.dropna(subset=required).copy()

        fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

        plots = [
            ("val_loss", "Validation loss"),
            ("test_loss", "Test loss"),
            ("imagenet1k_acc1", "ImageNet1k acc@1"),
            ("mscoco_image_recall@5", "MSCOCO image recall@5"),
        ]

        for ax, (y_col, y_label) in zip(axes.ravel(), plots):
            ax.scatter(data[x_col], data[y_col], s=30)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(f"{y_label} vs {x_label}")

        if title_prefix:
            fig.suptitle(f"{title_prefix} {hp}", fontsize=14)

        if out_path is None:
            plt.show()
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=160, bbox_inches="tight")
            plt.close(fig)


@dataclass
class MetricCfg:
    label: str
    maximize: bool


class IncumbentAnalyzer:
    def __init__(self, df: pd.DataFrame, trial_col: str = "trial_id"):
        self.df = df.copy()
        self.trial_col = trial_col

        self.df[self.trial_col] = pd.to_numeric(self.df[self.trial_col], errors="coerce")
        self.df = self.df.dropna(subset=[self.trial_col])
        self.df[self.trial_col] = self.df[self.trial_col].astype(int)
        self.df = self.df.sort_values(self.trial_col)

    def incumbent_series(self, metric: str, maximize: bool) -> pd.DataFrame:
        data = self.df[[self.trial_col, metric]].copy()
        data[metric] = pd.to_numeric(data[metric], errors="coerce")
        data = data.dropna(subset=[metric])

        if maximize:
            data["incumbent"] = data[metric].cummax()
        else:
            data["incumbent"] = data[metric].cummin()

        return data

    def plot_incumbent_trajectory_2x2(
            self,
            out_path: Optional[Path] = None,
            title: str = "Incumbent trajectory vs trial id",
    ) -> None:
        metrics: Dict[str, MetricCfg] = {
            "val_loss": MetricCfg("Validation loss (best so far)", maximize=False),
            "test_loss": MetricCfg("Test loss (best so far)", maximize=False),
            "imagenet1k_acc1": MetricCfg("ImageNet1k acc@1 (best so far)", maximize=True),
            "mscoco_image_recall@5": MetricCfg("MSCOCO image recall@5 (best so far)", maximize=True),
        }

        missing = [m for m in metrics.keys() if m not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

        for ax, (metric, cfg) in zip(axes.ravel(), metrics.items()):
            traj = self.incumbent_series(metric=metric, maximize=cfg.maximize)

            ax.scatter(traj[self.trial_col], traj[metric], s=20)
            ax.plot(traj[self.trial_col], traj["incumbent"])

            ax.set_xlabel("trial_id")
            ax.set_ylabel(cfg.label)
            ax.set_title(metric)

        fig.suptitle(title, fontsize=14)

        if out_path is None:
            plt.show()
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=160, bbox_inches="tight")
            plt.close(fig)


@dataclass(frozen=True)
class ScatterMetricCfg:
    y_col: str
    y_label: str


class LossDownstreamAnalyzer:

    def __init__(self, df: pd.DataFrame, x_col: str = "val_loss"):
        self.df = df.copy()
        self.x_col = x_col

        self.df[self.x_col] = pd.to_numeric(self.df[self.x_col], errors="coerce")

    def _ensure_numeric(self, col: str) -> None:
        self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

    def plot_2x2(
            self,
            out_path: Optional[Path] = None,
            title: str = "Validation loss vs downstream performance",
    ) -> None:
        metrics: Dict[str, ScatterMetricCfg] = {
            "imagenet": ScatterMetricCfg("imagenet1k_acc1", "ImageNet1k acc@1"),
            "cifar10": ScatterMetricCfg("cifar10_acc1", "CIFAR 10 acc@1"),
            "mscoco_img": ScatterMetricCfg("mscoco_image_recall@5", "MSCOCO image recall@5"),
            "mscoco_txt": ScatterMetricCfg("mscoco_text_recall@5", "MSCOCO text recall@5"),
        }

        missing = [cfg.y_col for cfg in metrics.values() if cfg.y_col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns in df: {missing}")

        for cfg in metrics.values():
            self._ensure_numeric(cfg.y_col)

        required = [self.x_col] + [cfg.y_col for cfg in metrics.values()]
        data = self.df.dropna(subset=required).copy()

        fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

        for ax, cfg in zip(axes.ravel(), metrics.values()):
            ax.scatter(data[self.x_col], data[cfg.y_col], s=30)

            ax.set_xlabel(self.x_col)
            ax.set_ylabel(cfg.y_label)
            ax.set_title(f"{cfg.y_label} vs {self.x_col}")

        fig.suptitle(title, fontsize=14)

        if out_path is None:
            plt.show()
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=160, bbox_inches="tight")
            plt.close(fig)


@dataclass(frozen=True)
class TopKBoxplotSpec:
    metric: str
    maximize: bool
    k: int = 10
    title: str = ""


class TopKRangeBoxplotter:
    def __init__(
            self,
            df: pd.DataFrame,
            hp_candidates: Sequence[str] = ("lr", "wd", "warmup_fraction", "grad_clip_norm"),
    ):
        self.df = df.copy()
        self.hp_candidates = list(hp_candidates)

    def available_hps(self) -> List[str]:
        return [hp for hp in self.hp_candidates if hp in self.df.columns]

    def _topk_subset(self, metric: str, k: int, maximize: bool, hp_list: Sequence[str]) -> pd.DataFrame:
        if metric not in self.df.columns:
            raise ValueError(f"Metric '{metric}' not found. Available: {list(self.df.columns)}")

        cols = [metric, *hp_list]
        data = self.df[cols].copy()

        data[metric] = pd.to_numeric(data[metric], errors="coerce")
        for hp in hp_list:
            data[hp] = pd.to_numeric(data[hp], errors="coerce")

        data = data.dropna(subset=cols)
        data = data.sort_values(metric, ascending=not maximize).head(k)
        return data

    @staticmethod
    def _log_transform_if_needed(values: pd.Series, hp: str) -> tuple[np.ndarray, str]:
        if hp in ("lr", "wd"):
            v = values.to_numpy(dtype=float)
            v = np.log10(np.clip(v, a_min=np.finfo(float).tiny, a_max=None))
            return v, f"log10({hp})"
        return values.to_numpy(dtype=float), hp

    def plot_boxplot_topk_per_hp(
            self,
            metric: str,
            maximize: bool,
            k: int = 10,
            out_dir: Optional[Path] = None,
            title_prefix: str = "",
            hp_list: Optional[Sequence[str]] = None,
            log_scale_lr_wd: bool = True,
            whis: tuple[int, int] = (0, 100),
    ) -> None:
        hps = list(hp_list) if hp_list is not None else self.available_hps()
        if not hps:
            raise ValueError("No hyperparameters available for boxplots.")

        topk = self._topk_subset(metric=metric, k=k, maximize=maximize, hp_list=hps)

        direction = "maximize" if maximize else "minimize"

        for hp in hps:
            series = topk[hp].dropna()
            if series.empty:
                continue

            if log_scale_lr_wd and hp in ("lr", "wd"):
                vals, x_label = self._log_transform_if_needed(series, hp)
            else:
                vals = series.to_numpy(dtype=float)
                x_label = hp

            fig = plt.figure(figsize=(5.5, 4.5), constrained_layout=True)
            ax = fig.add_subplot(111)

            ax.boxplot([vals], labels=[x_label], whis=whis, showfliers=True)

            ax.set_ylabel("Value (top-k subset)")
            if title_prefix:
                ax.set_title(f"{title_prefix} {hp}\nTop-{k} by {metric} ({direction})")
            else:
                ax.set_title(f"{hp}: Top-{k} by {metric} ({direction})")

            if out_dir is None:
                plt.show()
            else:
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{hp}_top{k}_by_{metric}.png"
                fig.savefig(out_path, dpi=160, bbox_inches="tight")
                plt.close(fig)


@dataclass(frozen=True)
class HeatmapSpec:
    x_hp: str = "warmup_fraction"
    y_hp: str = "wd"
    metric: str = "val_loss"
    agg: str = "mean"
    log_y: bool = True


class HyperparamHeatmapAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def plot(
            self,
            spec: HeatmapSpec,
            out_path: Optional[Path] = None,
            title: str = "",
    ) -> None:
        data = self.df[[spec.x_hp, spec.y_hp, spec.metric]].copy()

        data[spec.x_hp] = pd.to_numeric(data[spec.x_hp], errors="coerce")
        data[spec.y_hp] = pd.to_numeric(data[spec.y_hp], errors="coerce")
        data[spec.metric] = pd.to_numeric(data[spec.metric], errors="coerce")
        data = data.dropna()

        y_col = spec.y_hp
        y_label = spec.y_hp

        if spec.log_y and spec.y_hp in ("wd", "lr"):
            data[f"log10_{spec.y_hp}"] = np.log10(
                np.clip(data[spec.y_hp].to_numpy(float), np.finfo(float).tiny, None)
            )
            y_col = f"log10_{spec.y_hp}"
            y_label = f"log10({spec.y_hp})"

        grid = (
            data.pivot_table(
                index=y_col,
                columns=spec.x_hp,
                values=spec.metric,
                aggfunc=spec.agg,
            )
            .sort_index()
            .sort_index(axis=1)
        )

        Z = grid.to_numpy()

        fig = plt.figure(figsize=(7.5, 5.5), constrained_layout=True)
        ax = fig.add_subplot(111)

        im = ax.imshow(Z, aspect="auto", origin="lower")
        fig.colorbar(im, ax=ax, label=spec.metric)

        ax.set_xlabel(spec.x_hp)
        ax.set_ylabel(y_label)

        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"{spec.metric} heatmap: {spec.y_hp} vs {spec.x_hp}")

        ax.set_xticks(np.arange(grid.shape[1]))
        ax.set_yticks(np.arange(grid.shape[0]))
        ax.set_xticklabels([f"{v:.3g}" for v in grid.columns.to_numpy()], rotation=45, ha="right")
        ax.set_yticklabels([f"{v:.3g}" for v in grid.index.to_numpy()])

        if out_path is None:
            plt.show()
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=160, bbox_inches="tight")
            plt.close(fig)


if __name__ == "__main__":
    df = load_runs_csv(Path("results/model=ViT-S-32/optimizer=experimental/seed=1/exp.csv"))
    df = add_log_hp(df)

    hp_effect = HyperparamEffectAnalyzer(df)

    for hp in hp_effect.available_hps():
        hp_effect.plot_effect_2x2(
            hp=hp,
            out_path=Path("hp_analysis_out/hp_analyzer") / f"{hp}_vs_metrics.png",
            title_prefix="ViT-S-32 random search:",
        )

    incumbentAnalyzer = IncumbentAnalyzer(df)
    incumbentAnalyzer.plot_incumbent_trajectory_2x2(
        out_path=Path("hp_analysis_out/incumbent_analyzer") / "incumbent_trajectory.png"
    )

    valLossAnalyzer = LossDownstreamAnalyzer(df, x_col="val_loss")

    valLossAnalyzer.plot_2x2(
        out_path=Path("hp_analysis_out/loss_downstream_analyzer") / "val_loss_vs_downstreams.png"
    )

    testLossAnalyzer = LossDownstreamAnalyzer(df, x_col="test_loss")

    testLossAnalyzer.plot_2x2(
        out_path=Path("hp_analysis_out/loss_downstream_analyzer") / "test_loss_vs_downstreams.png"
    )

    boxplotter = TopKRangeBoxplotter(df)

    k = 10
    boxplotter.plot_boxplot_topk_per_hp(
        metric="imagenet1k_acc1",
        maximize=True,
        k=k,
        out_dir=Path("hp_analysis_out/ranges_topk_per_hp") / "top10_by_imagenet1k_acc1",
        title_prefix="ViT-S-32 random search",
    )

    boxplotter.plot_boxplot_topk_per_hp(
        metric="val_loss",
        maximize=False,
        k=k,
        out_dir=Path("hp_analysis_out/ranges_topk_per_hp") / "top10_by_val_loss",
        title_prefix="ViT-S-32 random search",
    )

    boxplotter.plot_boxplot_topk_per_hp(
        metric="mscoco_image_recall@5",
        maximize=True,
        k=k,
        out_dir=Path("hp_analysis_out/ranges_topk_per_hp") / "top10_by_mscoco_image_recall@5",
        title_prefix="ViT-S-32 random search",
    )

    heat = HyperparamHeatmapAnalyzer(df)

    heat.plot(
        HeatmapSpec(x_hp="warmup_fraction", y_hp="wd", metric="val_loss", log_y=True),
        out_path=Path("hp_analysis_out/heatmaps") / "wd_vs_warmup_val_loss.png",
    )
    heat.plot(
        HeatmapSpec(x_hp="warmup_fraction", y_hp="wd", metric="test_loss", log_y=True),
        out_path=Path("hp_analysis_out/heatmaps") / "wd_vs_warmup_test_loss.png",
    )
    heat.plot(
        HeatmapSpec(x_hp="warmup_fraction", y_hp="wd", metric="imagenet1k_acc1", log_y=True),
        out_path=Path("hp_analysis_out/heatmaps") / "wd_vs_warmup_imagenet1k_acc1.png",
    )
    heat.plot(
        HeatmapSpec(x_hp="warmup_fraction", y_hp="wd", metric="mscoco_image_recall@5", log_y=True),
        out_path=Path("hp_analysis_out/heatmaps") / "wd_vs_warmup_mscoco_image_recall@5.png",
    )
    heat.plot(
        HeatmapSpec(x_hp="warmup_fraction", y_hp="wd", metric="mscoco_text_recall@5", log_y=True),
        out_path=Path("hp_analysis_out/heatmaps") / "wd_vs_warmup_mscoco_text_recall@5.png",
    )

    heat.plot(
        HeatmapSpec(x_hp="beta2", y_hp="eps", metric="val_loss", log_y=True),
        out_path=Path("hp_analysis_out/heatmaps") / "beta2_vs_eps_val_loss.png",
    )
    heat.plot(
        HeatmapSpec(x_hp="beta2", y_hp="eps", metric="test_loss", log_y=True),
        out_path=Path("hp_analysis_out/heatmaps") / "beta2_vs_eps_test_loss.png",
    )
    heat.plot(
        HeatmapSpec(x_hp="beta2", y_hp="eps", metric="imagenet1k_acc1", log_y=True),
        out_path=Path("hp_analysis_out/heatmaps") / "beta2_vs_eps_imagenet1k_acc1.png",
    )
    heat.plot(
        HeatmapSpec(x_hp="beta2", y_hp="eps", metric="mscoco_image_recall@5", log_y=True),
        out_path=Path("hp_analysis_out/heatmaps") / "beta2_vs_eps_mscoco_image_recall@5.png",
    )
    heat.plot(
        HeatmapSpec(x_hp="beta2", y_hp="eps", metric="mscoco_text_recall@5", log_y=True),
        out_path=Path("hp_analysis_out/heatmaps") / "beta2_vs_eps_mscoco_text_recall@5.png",
    )
