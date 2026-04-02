import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

PREDICTOR_COLORS = {
    "tabpfn": "#4C72B0",
    "xgb": "#DD8452",
    "lightgbm": "#55A868",
    "mix": "#C44E52",
    "autogluon": "#8172B3",
}

MODES = ["with_intermediate", "no_intermediate"]
METRICS = ["rmse", "mae", "mdae", "marpd", "r2", "r", "spearman"]
TARGETS = ["val_loss"]


def get_per_mode_plot_path(base_dir, target, metric, mode, fit_mode, eval_mode):
    return Path(base_dir) / "per_mode" / target / metric / fit_mode / eval_mode / f"{mode}.png"


def get_with_vs_without_plot_path(base_dir, target, metric, fit_mode, eval_mode):
    return Path(base_dir) / "with_vs_without" / target / fit_mode / eval_mode / f"{metric}.png"


def load_results(root_dir):
    root = Path(root_dir)
    data = {}

    for predictor_dir in root.iterdir():
        if not predictor_dir.is_dir():
            continue

        predictor = predictor_dir.name
        data[predictor] = {}

        for seed_dir in predictor_dir.iterdir():
            if not seed_dir.is_dir():
                continue

            seed = int(seed_dir.name.split("_")[-1])

            for fit_mode_dir in seed_dir.iterdir():
                if not fit_mode_dir.is_dir():
                    continue

                for eval_mode_dir in fit_mode_dir.iterdir():
                    results_path = eval_mode_dir / "results.json"
                    if not eval_mode_dir.is_dir() or not results_path.exists():
                        continue

                    with open(results_path) as f:
                        results_json = json.load(f)

                    eval_mode = eval_mode_dir.name
                    data[predictor].setdefault(eval_mode, {}).setdefault(fit_mode_dir.name, {})[seed] = results_json

    return data


def extract_curve(results_json, target, metric):
    pairs = [
        (int(n_cfg), values[target][metric])
        for n_cfg, values in results_json["results"].items()
        if target in values and metric in values[target]
    ]

    if not pairs:
        return [], []

    pairs.sort()
    xs, ys = zip(*pairs)
    return list(xs), list(ys)


def aggregate_curves_over_seeds(seed_results, target, metric):
    values_per_config = {}

    for results_json in seed_results.values():
        xs, ys = extract_curve(results_json, target, metric)
        for x, y in zip(xs, ys):
            values_per_config.setdefault(x, []).append(y)

    if not values_per_config:
        return np.array([]), np.array([]), np.array([])

    xs = np.array(sorted(values_per_config), dtype=float)
    mean = np.array([np.mean(values_per_config[x]) for x in xs], dtype=float)
    std = np.array([np.std(values_per_config[x]) for x in xs], dtype=float)

    mask = xs >= 50
    return xs[mask], mean[mask], std[mask]


def setup_ax(ax, title, ylabel, fit_mode, eval_mode):
    ax.set_xscale("log")
    ax.set_xticks([50, 100, 200, 500, 700])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel("Number of configs", fontsize=12)
    ax.set_ylabel(ylabel.upper(), fontsize=12)
    ax.set_title(f"{title}\nFit Mode: {fit_mode}, Eval Mode: {eval_mode}", fontsize=14, pad=12)
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_and_show(fig, save_path=None):
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.tight_layout()
    plt.show()


def plot_metric(all_results, mode, target, metric, save_path=None, fit_mode=None, eval_mode=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    for predictor, predictor_data in sorted(all_results.items()):
        if mode not in predictor_data:
            continue

        xs, mean, std = aggregate_curves_over_seeds(predictor_data[mode], target, metric)
        if len(xs) == 0:
            continue

        color = PREDICTOR_COLORS.get(predictor, "#333333")
        ax.plot(xs, mean, label=predictor, color=color, linewidth=2.5, alpha=0.95)
        ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.15)

    setup_ax(
        ax,
        title=f"{target.replace('_', ' ')} {metric.upper()} across predictors\n{mode.replace('_', ' ')}",
        ylabel=metric,
        fit_mode=fit_mode,
        eval_mode=eval_mode
    )

    ax.legend(title="Predictor", loc="best", frameon=True, fontsize=10, title_fontsize=10)
    save_and_show(fig, save_path)


def plot_metric_with_vs_without(all_results, target, metric, save_path=None, fit_mode=None, eval_mode=None):
    fig, ax = plt.subplots(figsize=(11, 6))

    linestyles = {
        "with_intermediate": "-",
        "no_intermediate": "--",
    }
    band_alpha = {
        "with_intermediate": 0.12,
        "no_intermediate": 0.04,
    }

    predictor = "tabpfn"
    predictor_data = all_results.get(predictor, {})
    color = PREDICTOR_COLORS.get(predictor, "#333333")

    for mode in MODES:
        if mode not in predictor_data:
            continue

        xs, mean, std = aggregate_curves_over_seeds(predictor_data[mode], target, metric)
        if len(xs) == 0:
            continue

        ax.plot(
            xs,
            mean,
            color=color,
            linestyle=linestyles[mode],
            linewidth=2.3,
            alpha=0.95,
        )
        ax.fill_between(xs, mean - std, mean + std, color=color, alpha=band_alpha[mode])

    setup_ax(
        ax,
        title=f"{target.replace('_', ' ')} {metric.upper()}: with vs without intermediate points",
        ylabel=metric,
        fit_mode=fit_mode,
        eval_mode=eval_mode
    )

    style_handles = [
        Line2D([0], [0], color="black", lw=2.3, linestyle="-", label="with intermediate"),
        Line2D([0], [0], color="black", lw=2.3, linestyle="--", label="without intermediate"),
    ]
    ax.legend(handles=style_handles, title="Mode", loc="best", frameon=True, fontsize=9, title_fontsize=10)
    save_and_show(fig, save_path)


def main():
    results = load_results("results/predictors")
    plot_root = Path("plots")

    for target in TARGETS:
        for mode in MODES:
            for metric in METRICS:
                for predictor, predictor_data in results.items():
                    for eval_mode, eval_data in predictor_data.items():
                        for fit_mode, fit_data in eval_data.items():
                            plot_metric(
                                fit_data,
                                mode=mode,
                                target=target,
                                metric=metric,
                                save_path=get_per_mode_plot_path(plot_root, target, metric, mode, fit_mode, eval_mode),
                                fit_mode=fit_mode,
                                eval_mode=eval_mode
                            )

        for metric in METRICS:
            for predictor, predictor_data in results.items():
                for eval_mode, eval_data in predictor_data.items():
                    for fit_mode, fit_data in eval_data.items():
                        plot_metric_with_vs_without(
                            fit_data,
                            target=target,
                            metric=metric,
                            save_path=get_with_vs_without_plot_path(plot_root, target, metric, fit_mode, eval_mode),
                            fit_mode=fit_mode,
                            eval_mode=eval_mode
                        )


if __name__ == "__main__":
    main()
