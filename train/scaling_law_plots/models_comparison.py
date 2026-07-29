import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_scaling_curve(df, benchmark_metric):
    selected_widths = [32, 64, 128, 256, 320, 448]

    models = df[['vision_width', 'text_width']].drop_duplicates()

    models = models[
        (models['vision_width'].isin(selected_widths)) &
        (models['text_width'].isin(selected_widths)) &
        (models['text_width'] == models['vision_width'])
        ]

    models = models.sort_values(
        by=['vision_width', 'text_width']
    ).reset_index(drop=True)

    plt.figure(figsize=(8, 5))

    # same coloring style as their implementation
    mode_colors = plt.cm.viridis_r(
        np.linspace(0.4, 0.9, len(models))
    )

    for i, (_, row) in enumerate(models.iterrows()):
        vw = row['vision_width']
        tw = row['text_width']

        config_df = df[
            (df['vision_width'] == vw) &
            (df['text_width'] == tw)
            ].copy()

        # sort by compute
        config_df = config_df.sort_values('Total compute(GLOPs)')

        # build pareto costs array
        costs = np.array([
            config_df['Total compute(GLOPs)'].values,
            config_df[benchmark_metric].values
        ]).T

        # keep only pareto optimal points
        pareto_mask = paretoset_naive(costs)

        pareto_df = config_df.iloc[pareto_mask]

        x = pareto_df['Total compute(GLOPs)']
        y = pareto_df[benchmark_metric]

        plt.plot(
            x,
            y,
            'o-',
            color=mode_colors[i],
            label=f'vw={vw}, tw={tw}',
            lw=0.8,
            alpha=0.6
        )

    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel('Compute [GFLOPs]')
    plt.ylabel(benchmark_metric)

    plt.grid(True, which='both', linestyle='--', alpha=0.5)

    plt.legend()

    plt.tight_layout()
    plt.savefig("models_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def build_failure_labels(df: pd.DataFrame) -> pd.DataFrame:
    fail_df = (
        df.groupby("config_id", as_index=False)["epoch_diverged"]
        .any()
        .rename(columns={"epoch_diverged": "failed"})
    )
    fail_df["failed"] = fail_df["failed"].astype(int)
    return fail_df


def keep_only_nonfailed_last_epoch(df: pd.DataFrame) -> pd.DataFrame:
    fail_df = build_failure_labels(df)

    df = df.merge(fail_df, on="config_id", how="left")
    df_nonfailed = df[df["failed"] == 0].copy()

    df_last = (
        df_nonfailed
        .sort_values(["config_id", "epoch"])
        .drop_duplicates(subset="config_id", keep="last")
        .reset_index(drop=True)
    )

    return df_last


def paretoset_naive(costs, distinct=True):
    n_costs, _ = costs.shape

    is_efficient = np.zeros(n_costs, dtype=np.bool_)

    for i in range(n_costs):
        this_cost = costs[i, :]

        at_least_as_good = np.logical_not(np.any(costs > this_cost, axis=1))
        any_better = np.any(costs <= this_cost, axis=1)

        dominated_by = np.logical_and(at_least_as_good, any_better)

        if distinct and np.any(is_efficient):
            if np.any(np.all(costs[is_efficient] == this_cost, axis=1)):
                continue

        if not (np.any(dominated_by[:i]) or np.any(dominated_by[i + 1:])):
            is_efficient[i] = True

    return is_efficient


if __name__ == "__main__":
    df = pd.read_csv("train/surrogate_data.csv")
    df = keep_only_nonfailed_last_epoch(df)
    plot_scaling_curve(df, "val_loss")
