import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch


def convert_width_to_one_hot(width):
    width_choices = [32, 64, 128, 192, 256, 320, 384, 448, 512]

    one_hot = torch.zeros(len(width_choices))
    idx = width_choices.index(width)
    one_hot[idx] = 1

    return one_hot


def plot_correlation_matrix(corr_mat, keys, filename):
    sns.clustermap(
        corr_mat,
        xticklabels=keys,
        yticklabels=keys,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 6},
    )

    plt.savefig(filename)
    plt.close()


def load_results(root_dir: str):
    root = Path(root_dir)
    data = {}

    for predictor_dir in root.iterdir():
        if not predictor_dir.is_dir():
            continue

        predictor_name = predictor_dir.name
        data[predictor_name] = {}

        for mode_dir in predictor_dir.iterdir():
            if not mode_dir.is_dir():
                continue

            results_path = mode_dir / "results.json"
            if not results_path.exists():
                continue

            with open(results_path) as f:
                data[predictor_name][mode_dir.name] = json.load(f)

    return data


def extract_curve(results_json, target, metric):
    results = results_json["results"]

    xs = []
    ys = []

    for n_cfg, values in results.items():
        n_cfg = int(n_cfg)

        metric_value = values[target][metric]

        xs.append(n_cfg)
        ys.append(metric_value)

    pairs = sorted(zip(xs, ys))

    xs, ys = zip(*pairs)

    return xs, ys


def plot_metric(all_results, mode, target, metric):
    plt.figure(figsize=(8, 5))

    for predictor, predictor_data in all_results.items():

        if mode not in predictor_data:
            continue

        xs, ys = extract_curve(
            predictor_data[mode],
            target=target,
            metric=metric
        )

        plt.plot(xs, ys, marker="o", label=predictor)

    plt.xlabel("Number of configs")
    plt.ylabel(metric)
    plt.title(f"{target} {metric} ({mode})")

    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    root_dir = "results/predictors"

    results = load_results(root_dir)

    modes = [
        "with_intermediate_points",
        "no_intermediate_points"
    ]

    for mode in modes:
        plot_metric(
            results,
            mode=mode,
            target="val_loss",
            metric="rmse"
        )

        plot_metric(
            results,
            mode=mode,
            target="val_loss",
            metric="spearman"
        )


if __name__ == "__main__":
    main()
