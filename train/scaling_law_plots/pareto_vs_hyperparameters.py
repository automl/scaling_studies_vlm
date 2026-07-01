import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize


def load_final_non_diverged(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    mask = (
            (df["epoch"] == df["total_epochs"])
            & (df["epoch_diverged"] == False)
    )

    return df[mask].copy()


def get_pareto_mask(
        flops: np.ndarray,
        objective: np.ndarray,
) -> np.ndarray:
    pareto_mask = np.ones(len(flops), dtype=bool)

    for i in range(len(flops)):
        dominated = (
                (flops <= flops[i])
                & (objective <= objective[i])
                & (
                        (flops < flops[i])
                        | (objective < objective[i])
                )
        )

        dominated[i] = False

        if dominated.any():
            pareto_mask[i] = False

    return pareto_mask


def plot_pareto_vs_hyperparameter(
        csv_path: str,
        hyperparam_col: str,
        output_path: str,
        objective_col: str = "val_loss",
        flops_col: str = "compute_seen",
        log_x: bool = True,
        log_y: bool = True,
):
    df = load_final_non_diverged(csv_path)

    required_columns = {
        flops_col,
        hyperparam_col,
        objective_col,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    flops = pd.to_numeric(
        df[flops_col],
        errors="coerce",
    ).to_numpy(dtype=float)

    hyperparameter = pd.to_numeric(
        df[hyperparam_col],
        errors="coerce",
    ).to_numpy(dtype=float)

    objective = pd.to_numeric(
        df[objective_col],
        errors="coerce",
    ).to_numpy(dtype=float)

    valid_mask = (
            np.isfinite(flops)
            & np.isfinite(hyperparameter)
            & np.isfinite(objective)
            & (flops > 0)
    )

    if log_y:
        valid_mask &= hyperparameter > 0

    flops = flops[valid_mask]
    hyperparameter = hyperparameter[valid_mask]
    objective = objective[valid_mask]

    if flops.size == 0:
        print(f"No valid points found for {hyperparam_col}.")
        return

    pareto_mask = get_pareto_mask(
        flops=flops,
        objective=objective,
    )

    pareto_flops = flops[pareto_mask]
    pareto_hyperparameter = hyperparameter[pareto_mask]

    fig, ax = plt.subplots(
        figsize=(10, 4),
        constrained_layout=True,
    )

    norm = Normalize(
        vmin=float(objective.min()),
        vmax=float(objective.max()),
    )

    scatter = ax.scatter(
        flops,
        hyperparameter,
        c=objective,
        cmap="viridis",
        marker="o",
        alpha=0.7,
        norm=norm,
        label="All points",
    )

    ax.scatter(
        pareto_flops,
        pareto_hyperparameter,
        facecolors="none",
        edgecolors="black",
        linewidths=2.0,
        s=95,
        label="Pareto optimal",
        zorder=3,
    )

    if log_x:
        ax.set_xscale("log")

    if log_y:
        ax.set_yscale("log")

    ax.set_xlabel("GFLOPs")
    ax.set_ylabel(hyperparam_col)
    ax.set_title(f"GFLOPs vs {hyperparam_col}")

    ax.grid(
        True,
        linestyle="--",
        alpha=0.4,
        which="major",
    )

    ax.legend(loc="upper left")

    colorbar = fig.colorbar(
        scatter,
        ax=ax,
    )
    colorbar.set_label("Loss")

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    hyperparameters = {
        "lr": True,
        "wd": True,
        "warmup_fraction": False,
        "vision_width": False,
        "text_width": False,
        "beta1": False,
        "beta2": False,
        "eps": True,
        "samples_seen": True,
    }

    for hyperparam, log_y in hyperparameters.items():
        output_path = (
            f"{hyperparam}_vs_flops.png"
        )

        plot_pareto_vs_hyperparameter(
            csv_path="../surrogate_data.csv",
            hyperparam_col=hyperparam,
            output_path=str(output_path),
            log_y=log_y,
        )

    plot_vision_text_ratio(
        csv_path="../surrogate_data.csv",
        output_path=str("pareto_vs_vision_text_ratio.png"
                        ),
        objective_col="val_loss",
    )


def plot_vision_text_ratio(
        csv_path: str,
        output_path: str,
        objective_col: str = "val_loss",
        flops_col: str = "compute_seen",
        vision_width_col: str = "vision_width",
        text_width_col: str = "text_width",
):
    df = load_final_non_diverged(csv_path)

    required_columns = {
        flops_col,
        objective_col,
        vision_width_col,
        text_width_col,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    flops = pd.to_numeric(
        df[flops_col],
        errors="coerce",
    ).to_numpy(dtype=float)

    objective = pd.to_numeric(
        df[objective_col],
        errors="coerce",
    ).to_numpy(dtype=float)

    vision_width = pd.to_numeric(
        df[vision_width_col],
        errors="coerce",
    ).to_numpy(dtype=float)

    text_width = pd.to_numeric(
        df[text_width_col],
        errors="coerce",
    ).to_numpy(dtype=float)

    valid_mask = (
            np.isfinite(flops)
            & np.isfinite(objective)
            & np.isfinite(vision_width)
            & np.isfinite(text_width)
            & (flops > 0)
            & (objective > 0)
            & (vision_width > 0)
            & (text_width > 0)
    )

    flops = flops[valid_mask]
    objective = objective[valid_mask]
    vision_width = vision_width[valid_mask]
    text_width = text_width[valid_mask]

    if flops.size == 0:
        print("No valid rows found.")
        return

    vision_text_ratio = vision_width / text_width

    pareto_mask = get_pareto_mask(
        flops=flops,
        objective=objective,
    )

    pareto_flops = flops[pareto_mask]
    pareto_ratio = vision_text_ratio[pareto_mask]

    fig, ax = plt.subplots(
        figsize=(10, 4),
        constrained_layout=True,
    )

    norm = Normalize(
        vmin=float(objective.min()),
        vmax=float(objective.max()),
    )

    scatter = ax.scatter(
        flops,
        vision_text_ratio,
        c=objective,
        cmap="viridis",
        marker="o",
        alpha=0.7,
        norm=norm,
        label="All points",
    )

    ax.scatter(
        pareto_flops,
        pareto_ratio,
        facecolors="none",
        edgecolors="black",
        linewidths=2.0,
        s=95,
        label="Pareto optimal",
        zorder=3,
    )

    fit_mask = (
            np.isfinite(pareto_flops)
            & np.isfinite(pareto_ratio)
            & (pareto_flops > 0)
            & (pareto_ratio > 0)
    )

    fit_flops = pareto_flops[fit_mask]
    fit_ratio = pareto_ratio[fit_mask]

    if len(fit_flops) >= 2:
        log_flops = np.log(fit_flops)
        log_ratio = np.log(fit_ratio)

        slope, intercept = np.polyfit(
            log_flops,
            log_ratio,
            1,
        )

        line_flops = np.logspace(
            np.log10(fit_flops.min()),
            np.log10(fit_flops.max()),
            200,
        )

        line_ratio = np.exp(intercept) * line_flops ** slope

        ax.plot(
            line_flops,
            line_ratio,
            "r--",
            alpha=0.7,
            linewidth=2.5,
            label=f"Fit on Pareto (slope={slope:.4f})",
        )

        scaler = np.exp(intercept)

        line_eq = (
            f"log(y) = {slope:.4f}·log(x) + {intercept:.4f}   →   "
            f"y = {scaler:.4e}·x^{slope:.4f}"
        )

        fig.text(
            0.5,
            0.02,
            line_eq,
            ha="center",
            fontsize=9,
            bbox=dict(
                boxstyle="round",
                facecolor="wheat",
                alpha=0.5,
            ),
        )

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("GFLOPs")
    ax.set_ylabel("vision-text-ratio")
    ax.set_title("GFLOPs vs vision-text-ratio")

    ax.grid(
        True,
        linestyle="--",
        alpha=0.4,
        which="major",
    )

    ax.legend(loc="upper left")

    colorbar = fig.colorbar(
        scatter,
        ax=ax,
    )
    colorbar.set_label("Loss")

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


if __name__ == "__main__":
    main()
