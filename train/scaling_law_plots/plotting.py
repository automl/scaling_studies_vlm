import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize


def plot_flops_vs_params_from_csv(
        csv_path: str,
        *,
        flops_col: str = "Total compute(GLOPs)",
        params_col: str = "mparams",
        params_are_millions: bool = True,
) -> plt.Figure | None:
    """
    Read a CSV and plot FLOPs vs Params (log-log).
    Expects columns like:
      - Total compute(GLOPs)
      - mparams (millions of parameters)
    """

    df = pd.read_csv(csv_path)

    if flops_col not in df.columns or params_col not in df.columns:
        print(f"Missing required columns. Need: {flops_col!r} and {params_col!r}")
        return None

    x = pd.to_numeric(df[flops_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[params_col], errors="coerce").to_numpy(dtype=float)

    if params_are_millions:
        y = y * 1_000_000

    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]

    if x.size == 0:
        print("No valid FLOPs/Params rows to plot.")
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.scatter(x, y, marker="o", alpha=0.9)
    ax.set_xlabel("GFLOPs")
    ax.set_ylabel("Params")
    ax.set_title("GFLOPs vs Params")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    return fig


def plot_flops_per_objective_from_csv(
        csv_path: str,
        *,
        objective_col: str = "val_loss",
        flops_col: str = "Total compute(GLOPs)",
        mparams_col: str = "mparams",
        params_are_millions: bool = True,
) -> plt.Figure | None:
    """
    Plot Objective vs FLOPs (log-log), colored by parameter count, from a CSV.

    Expected columns (defaults):
      - Total compute(GLOPs)
      - mparams
      - objective_col (e.g. val_loss, train_loss, imagenet1k_acc1, ...)
    """

    df = pd.read_csv(csv_path)

    needed = {objective_col, flops_col, mparams_col}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"Missing required columns: {missing}")
        return None

    obj = pd.to_numeric(df[objective_col], errors="coerce").to_numpy(dtype=float)
    flops = pd.to_numeric(df[flops_col], errors="coerce").to_numpy(dtype=float)
    n_params = pd.to_numeric(df[mparams_col], errors="coerce").to_numpy(dtype=float)

    if params_are_millions:
        n_params = n_params * 1_000_000

    mask = np.isfinite(obj) & np.isfinite(flops) & np.isfinite(n_params)
    mask &= (flops > 0) & (obj > 0) & (n_params > 0)

    obj = obj[mask]
    flops = flops[mask]
    n_params = n_params[mask]

    if flops.size == 0:
        print("No valid rows to plot after filtering.")
        return None

    fig, ax = plt.subplots(figsize=(6, 4))

    norm = Normalize(vmin=float(n_params.min()), vmax=float(n_params.max()))
    sc = ax.scatter(flops, obj, c=n_params, cmap="inferno", marker="o", alpha=0.9, norm=norm)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("GFLOPs")
    ax.set_ylabel(objective_col)
    ax.set_title(f"{objective_col} vs GFLOPs (log-log scale)")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.colorbar(sc, ax=ax, label="Parameters")
    plt.tight_layout()

    return fig


def plot_params_vs_loss_from_csv(
        csv_path: str,
        *,
        objective_col: str = "val_loss",
        params_col: str = "mparams",
        time_col: str = "train_duration(s)",
        params_are_millions: bool = True,
) -> plt.Figure | None:
    """
    Parameters vs objective (colored by time) from CSV.

    Defaults assume your runs CSV has:
      - mparams
      - train_duration(s)
      - val_loss (or any objective_col you pass)
    """
    df = pd.read_csv(csv_path)

    needed = {objective_col, params_col, time_col}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"Missing required columns: {missing}")
        return None

    obj = pd.to_numeric(df[objective_col], errors="coerce").to_numpy(dtype=float)
    params = pd.to_numeric(df[params_col], errors="coerce").to_numpy(dtype=float)
    times = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)

    if params_are_millions:
        params = params * 1_000_000

    mask = np.isfinite(obj) & np.isfinite(params) & np.isfinite(times) & (params > 0)
    obj = obj[mask]
    params = params[mask]
    times = times[mask]

    if params.size == 0:
        print("No valid rows to plot.")
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(params, obj, c=times, cmap="coolwarm", marker="o", alpha=0.9)

    ax.set_xscale("log")
    ax.set_xlabel("Parameters")
    ax.set_ylabel(objective_col)
    ax.set_title("Objective vs Parameters (colored by Time)")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.colorbar(sc, ax=ax, label=time_col)

    return fig


def plot_data_vs_loss_from_csv(
        csv_path: str,
        *,
        objective_col: str = "val_loss",
        data_col: str = "train_num_samples",
        flops_col: str = "Total compute(GLOPs)",
) -> plt.Figure | None:
    """
    Data points vs objective (colored by FLOPs) from CSV.

    Defaults assume your runs CSV has:
      - train_num_samples
      - Total compute(GLOPs)
      - val_loss (or any objective_col you pass)
    """
    df = pd.read_csv(csv_path)

    needed = {objective_col, data_col, flops_col}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"Missing required columns: {missing}")
        return None

    obj = pd.to_numeric(df[objective_col], errors="coerce").to_numpy(dtype=float)
    n_data = pd.to_numeric(df[data_col], errors="coerce").to_numpy(dtype=float)
    flops = pd.to_numeric(df[flops_col], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(obj) & np.isfinite(n_data) & np.isfinite(flops)
    obj = obj[mask]
    n_data = n_data[mask]
    flops = flops[mask]

    if n_data.size == 0:
        print("No valid rows to plot.")
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    norm = Normalize(vmin=float(flops.min()), vmax=float(flops.max()))
    sc = ax.scatter(n_data, obj, c=flops, cmap="plasma", marker="o", alpha=0.9, norm=norm)

    ax.set_xlabel("Number of Data Points")
    ax.set_ylabel(objective_col)
    ax.set_title("Objective vs Data Points")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.colorbar(sc, ax=ax, label="GFLOPs")

    return fig


def plot_accumulated_flops_per_objective_from_csv(
        csv_path: str,
        *,
        objective_col: str = "val_loss",
        flops_col: str = "Total compute(GLOPs)",
        sort_by: str = "config_id",
) -> plt.Figure | None:
    """
    Accumulated FLOPs vs objective from a CSV.

    We sort rows to define the accumulation order, then compute:
      accumulated_flops[i] = sum_{j<=i} flops[j]
    and plot objective vs accumulated_flops.

    Note: if objective is a loss, lower is better; for accuracy, higher is better.
    """

    df = pd.read_csv(csv_path)

    needed = {objective_col, flops_col}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"Missing required columns: {missing}")
        return None

    if sort_by is not None and sort_by in df.columns:
        df = df.sort_values(sort_by, kind="mergesort")
    else:
        df = df.reset_index(drop=True)

    obj = pd.to_numeric(df[objective_col], errors="coerce").to_numpy(dtype=float)
    flops = pd.to_numeric(df[flops_col], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(obj) & np.isfinite(flops)
    obj = obj[mask]
    flops = flops[mask]

    if flops.size == 0:
        print("No valid rows to plot.")
        return None

    accumulated_flops = np.cumsum(flops)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(accumulated_flops, obj, marker="o", linestyle="-", alpha=0.9)
    ax.set_xlabel("Accumulated FLOPs (GLOPs)")
    ax.set_ylabel(objective_col)
    ax.set_title("Objective vs Accumulated GFLOPs")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    return fig


def _compute_pareto_front(trial_data: list[tuple]) -> list[tuple]:
    """Compute Pareto front from trial data (n_params, n_data, obj, flops)."""
    pareto = []
    for i, (n_i, d_i, l_i, f_i) in enumerate(trial_data):
        dominated = False
        for j, (_, _, l_j, f_j) in enumerate(trial_data):
            if i == j:
                continue
            if (f_j <= f_i and l_j <= l_i) and (f_j < f_i or l_j < l_i):
                dominated = True
                break
        if not dominated:
            pareto.append((n_i, d_i, l_i, f_i))
    return pareto


def plot_pareto_front_from_csv(
        csv_path: str,
        *,
        objective_col: str = "val_loss",
        flops_col: str = "Total compute(GLOPs)",
        params_col: str = "mparams",
        data_col: str = "train_num_samples",
        params_are_millions: bool = True,
        compare_csv_path: str | None = None,
) -> tuple[plt.Figure, pd.DataFrame]:
    """
    Pareto front plot from CSV (non-dominated points) in log-log scale.
    Points are Pareto-optimal w.r.t. minimizing (objective, flops).
    Colored by parameter count.
    """

    df = pd.read_csv(csv_path)

    needed = {objective_col, flops_col, params_col, data_col}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"Missing required columns: {missing}")
        return None

    obj = pd.to_numeric(df[objective_col], errors="coerce").to_numpy(dtype=float)
    flops = pd.to_numeric(df[flops_col], errors="coerce").to_numpy(dtype=float)
    n_params = pd.to_numeric(df[params_col], errors="coerce").to_numpy(dtype=float)
    n_data = pd.to_numeric(df[data_col], errors="coerce").to_numpy(dtype=float)

    if params_are_millions:
        n_params = n_params * 1_000_000

    mask = np.isfinite(obj) & np.isfinite(flops) & np.isfinite(n_params) & np.isfinite(n_data)
    mask &= (obj > 0) & (flops > 0) & (n_params > 0) & (n_data > 0)

    obj = obj[mask]
    flops = flops[mask]
    n_params = n_params[mask]
    n_data = n_data[mask]

    if flops.size == 0:
        return None

    filtered_df = df[mask].reset_index(drop=True)

    rows = list(zip(n_params.tolist(), n_data.tolist(), obj.tolist(), flops.tolist()))
    pareto_front = _compute_pareto_front(rows)

    if not pareto_front:
        return None

    indexed_rows = [
        (i, n_params[i], n_data[i], obj[i], flops[i])
        for i in range(len(obj))
    ]

    tuple_to_idx = {}
    for idx, npv, ndv, lv, fv in indexed_rows:
        key = (npv, ndv, lv, fv)
        if key not in tuple_to_idx:
            tuple_to_idx[key] = idx

    pareto_indices = [tuple_to_idx[(p[0], p[1], p[2], p[3])] for p in pareto_front]
    pareto_df = filtered_df.iloc[pareto_indices].copy()

    pareto_front.sort(key=lambda r: r[3])

    n_params_front = [r[0] for r in pareto_front]
    objs_front = [r[2] for r in pareto_front]
    flops_front = [r[3] for r in pareto_front]

    log_flops_front = np.log(np.array(flops_front))
    log_objs_front = np.log(np.array(objs_front))

    slope = intercept = None
    fitted_objs = None
    try:
        if np.all(np.isfinite(log_flops_front)) and np.all(np.isfinite(log_objs_front)) and len(log_flops_front) >= 2:
            coeffs = np.polyfit(log_flops_front, log_objs_front, 1)
            slope, intercept = float(coeffs[0]), float(coeffs[1])
            fitted_objs = np.exp(np.poly1d(coeffs)(log_flops_front))
    except Exception as e:
        print(f"Could not fit linear trend to Pareto front: {e}")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        flops, obj,
        c=n_params, cmap="gray",
        marker="o", alpha=0.2, s=30,
        label="All points",
    )

    norm_front = Normalize(vmin=min(n_params_front), vmax=max(n_params_front))
    sc = ax.scatter(
        flops_front, objs_front,
        c=n_params_front, cmap="inferno",
        marker="D", s=100, alpha=1.0,
        edgecolors="black", linewidth=1.5,
        norm=norm_front,
        label="Pareto front (actual)",
    )

    if compare_csv_path is not None and (
            objective_col == "imagenet1k_acc1_error_rate" or objective_col == "mscoco_image_recall@5_error_rate" or objective_col == "mscoco_text_recall@5_error_rate"):
        df2 = None
        if objective_col == "imagenet1k_acc1_error_rate":
            df2 = pd.read_csv("results/openclip_in1k_pareto.csv")

        if objective_col == "mscoco_image_recall@5_error_rate":
            df2 = pd.read_csv("results/openclip_msococo_img_ret_pareto.csv")

        if objective_col == "mscoco_text_recall@5_error_rate":
            df2 = pd.read_csv("results/openclip_msococo_img_ret_pareto.csv")
            df2["mscoco_text_recall@5_error_rate"] = 1 - df2["mscoco_text_recall@5_error_rate"]

        needed2 = {objective_col, flops_col, params_col}
        missing2 = [c for c in needed2 if c not in df2.columns]
        if missing2:
            print(f"[compare] Missing required columns: {missing2}")
        else:
            obj2 = pd.to_numeric(df2[objective_col], errors="coerce").to_numpy(float)
            flops2 = pd.to_numeric(df2[flops_col], errors="coerce").to_numpy(float)
            n_params2 = pd.to_numeric(df2[params_col], errors="coerce").to_numpy(float)

            if params_are_millions:
                n_params2 *= 1_000_000

            mask2 = np.isfinite(obj2) & np.isfinite(flops2) & np.isfinite(n_params2)
            mask2 &= (obj2 > 0) & (flops2 > 0) & (n_params2 > 0)

            obj2 = obj2[mask2]
            flops2 = flops2[mask2]
            n_params2 = n_params2[mask2]

            if len(flops2) > 0:
                norm_front = Normalize(
                    vmin=min(norm_front.vmin, float(np.min(n_params2))),
                    vmax=max(norm_front.vmax, float(np.max(n_params2))),
                )
                sc.set_norm(norm_front)

                ax.scatter(
                    flops2, obj2,
                    c=n_params2, cmap="inferno",
                    marker="^", s=120,
                    edgecolors="black", linewidth=1.5,
                    norm=norm_front,
                    label="External points",
                )

    ax.plot(flops_front, objs_front, "k-", alpha=0.3, linewidth=1)

    if fitted_objs is not None and slope is not None and intercept is not None:
        ax.plot(
            flops_front, fitted_objs,
            "r--", alpha=0.7, linewidth=2.5,
            label=f"Fit on Pareto (slope={slope:.4f})",
        )

        scaler = np.exp(intercept)
        line_eq = (
            f"log(y) = {slope:.4f}·log(x) + {intercept:.4f}   →   "
            f"y = {scaler:.4e}·x^{slope:.4f}"
        )
        fig.text(
            0.5, 0.02, line_eq,
            ha="center", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("FLOPs", fontsize=12)
    ax.set_ylabel(objective_col, fontsize=12)
    ax.set_title(f"Pareto Front (log-log) ({len(pareto_front)}/{len(rows)} points)", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.3, which="both")
    ax.legend(loc="best")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.colorbar(sc, ax=ax, label="Parameters")

    return fig, pareto_df


def plot_flop_vs_data_from_pareto_frontier(
        csv_path: str,
        *,
        flops_col: str = "Total compute(GLOPs)",
        data_col: str = "train_num_samples",
) -> plt.Figure | None:
    df = pd.read_csv(csv_path)

    if flops_col not in df.columns or data_col not in df.columns:
        print(f"Missing required columns. Need: {flops_col!r} and {data_col!r}")
        return None

    x = pd.to_numeric(df[flops_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[data_col], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[mask], y[mask]

    if x.size == 0:
        return None

    order = np.argsort(x)
    x, y = x[order], y[order]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.scatter(x, y, marker="o", alpha=0.9)
    ax.set_xlabel("GFLOPs")
    ax.set_ylabel("Number of datapoints")
    ax.set_title("Pareto: GFLOPs vs Dataset Size")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def plot_flops_vs_params_from_pareto(
        csv_path: str,
        *,
        flops_col: str = "Total compute(GLOPs)",
        params_col: str = "mparams",
        params_are_millions: bool = True,
) -> plt.Figure | None:
    df = pd.read_csv(csv_path)

    if flops_col not in df.columns or params_col not in df.columns:
        return None

    x = pd.to_numeric(df[flops_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[params_col], errors="coerce").to_numpy(dtype=float)

    if params_are_millions:
        y *= 1_000_000

    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[mask], y[mask]

    if x.size == 0:
        return None

    order = np.argsort(x)
    x, y = x[order], y[order]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.scatter(x, y, alpha=0.9)

    ax.set_xlabel("Compute (FLOPs)")
    ax.set_ylabel("Parameters")
    ax.set_title("Pareto: Compute vs Model Size")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    return fig


OBJECTIVE_DIRNAME = {
    "imagenet1k_acc1": "imagenet1k_acc1",
    "mscoco_image_recall@5": "mscoco_image_recall_at_5",
    "mscoco_text_recall@5": "mscoco_text_recall_at_5",
    "val_loss": "val_loss",
    "test_loss": "test_loss",
}

NON_OBJECTIVE_PLOTS = {
    "flops_vs_params",
    "accumulated_flops_objective",
}

OBJECTIVE_PLOTS = {
    "flops_per_objective",
    "params_vs_loss",
    "data_vs_loss",
    "pareto_front",
}

import pandas as pd
from pathlib import Path

ACCURACY_OBJECTIVES = {
    "imagenet1k_acc1",
    "mscoco_image_recall@5",
    "mscoco_text_recall@5",
}


def _prepare_objective_csv(csv_path: str, objective_col: str) -> tuple[str, str]:
    """
    If objective is accuracy -> create temp CSV with error column.
    Returns (csv_to_use, objective_column_to_use)
    """
    if objective_col not in ACCURACY_OBJECTIVES:
        return csv_path, objective_col

    df = pd.read_csv(csv_path)

    error_col = f"{objective_col}_error_rate"
    df[error_col] = 1.0 - pd.to_numeric(df[objective_col], errors="coerce")

    tmp_path = Path(csv_path).with_name(f"_tmp_{error_col}.csv")
    df.to_csv(tmp_path, index=False)

    return str(tmp_path), error_col


def plot_all_scaling_law_figures(csv_path: str, objective_col: str, dpi: int = 200):
    """
    Creates and saves plots for one objective.
    Accuracy metrics are automatically converted to error rate.
    """

    csv_used, obj_used = _prepare_objective_csv(csv_path, objective_col)

    figures = {
        "flops_per_objective": plot_flops_per_objective_from_csv(csv_used, objective_col=obj_used),
        "params_vs_loss": plot_params_vs_loss_from_csv(csv_used, objective_col=obj_used),
        "data_vs_loss": plot_data_vs_loss_from_csv(csv_used, objective_col=obj_used),
        "pareto_front": plot_pareto_front_from_csv(csv_used, objective_col=obj_used,
                                                   compare_csv_path="results/openclip_in1k_pareto.csv"),
        "accumulated_flops_objective": plot_accumulated_flops_per_objective_from_csv(csv_used, objective_col=obj_used),
    }

    csv_path = Path(csv_path)
    parent_dir = csv_path.parent

    obj_dirname = OBJECTIVE_DIRNAME.get(objective_col, objective_col)
    obj_dir = parent_dir / obj_dirname
    obj_dir.mkdir(parents=True, exist_ok=True)

    for name, item in figures.items():

        if item is None:
            continue

        if not isinstance(item, tuple):
            fig = item
            out_path = obj_dir / f"{name}.png"
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            continue

        fig, df = item

        if fig is not None:
            fig_path = obj_dir / f"{name}.png"
            fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")

        if df is not None:
            csv_out = obj_dir / f"{name}.csv"
            df.to_csv(csv_out, index=False)

            if name == "pareto_front":
                flops_vs_data_pareto = plot_flop_vs_data_from_pareto_frontier(str(csv_out))
                flops_vs_n_params_pareto = plot_flops_vs_params_from_pareto(str(csv_out))
                if flops_vs_data_pareto is not None:
                    flops_vs_data_pareto.savefig(obj_dir / "pareto_flops_vs_data.png", dpi=dpi, bbox_inches="tight")

                if flops_vs_n_params_pareto is not None:
                    flops_vs_n_params_pareto.savefig(obj_dir / "pareto_flops_vs_n_params.png", dpi=dpi,
                                                     bbox_inches="tight")

    return figures


def plot_all_objectives(csv_path: str, dpi: int = 200):
    """
    Iterates over the desired objectives and generates the objective-dependent plots
    into their own folders. Objective-independent plots are still saved once to parent.
    """
    objectives = [
        "imagenet1k_acc1",
        "mscoco_image_recall@5",
        "mscoco_text_recall@5",
        "val_loss",
        "test_loss",
    ]

    csv_path_p = Path(csv_path)
    parent_dir = csv_path_p.parent

    base_figs = {
        "flops_vs_params": plot_flops_vs_params_from_csv(csv_path),
    }
    for name, fig in base_figs.items():
        if fig is None:
            continue
        fig.savefig(parent_dir / f"{name}.png", dpi=dpi, bbox_inches="tight")

    all_figs = {"__base__": base_figs}
    for obj in objectives:
        all_figs[obj] = plot_all_scaling_law_figures(csv_path, objective_col=obj, dpi=dpi)

    return all_figs
