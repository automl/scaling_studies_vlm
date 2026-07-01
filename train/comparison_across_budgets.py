import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import spearmanr

from train.scaling_law_plots.models_comparison import build_failure_labels

TASK_NAME_MAP = {
    "vtab_caltech101_mean_per_class_recall": "Caltech-101",
    "cifar10_mean_per_class_recall": "CIFAR-10",
    "vtab_cifar100_mean_per_class_recall": "CIFAR-100",
    "vtab_clevr_count_all_acc1": "CLEVR Counts",
    "vtab_clevr_closest_object_distance_acc1": "CLEVR Distance",
    "country211_acc1": "Country211",
    "vtab_dtd_acc1": "Describable Textures",
    "vtab_eurosat_acc1": "EuroSAT",
    "fgvc_aircraft_mean_per_class_recall": "FGVC Aircraft",
    "food101_acc1": "Food-101",
    "gtsrb_acc1": "GTSRB",
    "imagenet1k_acc1": "ImageNet 1k",
    "imagenet_sketch_acc1": "ImageNet Sketch",
    "imagenetv2_acc1": "ImageNet v2",
    "imagenet-a_acc1": "ImageNet-A",
    "imagenet-o_acc1": "ImageNet-O",
    "imagenet-r_acc1": "ImageNet-R",
    "vtab_kitti_closest_vehicle_distance_acc1": "KITTI Vehicle Distance",
    "mnist_acc1": "MNIST",
    "objectnet_acc1": "ObjectNet",
    "vtab_flowers_mean_per_class_recall": "Oxford Flowers-102",
    "vtab_pets_mean_per_class_recall": "Oxford-IIIT Pet",
    "voc2007_acc1": "Pascal VOC 2007",
    "vtab_pcam_acc1": "PatchCamelyon",
    "renderedsst2_acc1": "Rendered SST2",
    "vtab_resisc45_acc1": "RESISC45",
    "cars_acc1": "Stanford Cars",
    "stl10_acc1": "STL-10",
    "sun397_acc1": "SUN397",
    "vtab_svhn_acc1": "SVHN",
    "retrieval_flickr_1k_test_image_text_retrieval_mean_recall@1":
        "Flickr",
    "retrieval_mscoco_2014_5k_test_image_text_retrieval_mean_recall@1":
        "MSCOCO",
    "misc_winogavil_jaccard_score_10-12": "WinoGAViL",
    "wilds_iwildcam_F1-macro_all": "iWildCam",
    "wilds_camelyon17_acc1": "Camelyon17",
    "wilds_fmow_acc_worst_region": "FMoW",
    "fairness_dollar_street_acc_top5_wg": "Dollar Street",
    "fairness_geode_acc_wg": "GeoDE",
    "fairness_fairface_acc_race_avg": "FairFace",
    "fairness_utkface_acc_race_avg": "UTKFace",
    "val_loss": "Validation loss",
    "test_loss": "Test loss",
}


def clean_task_label(task: str) -> str:
    return TASK_NAME_MAP.get(task, task)


def keep_only_nonfailed(df: pd.DataFrame) -> pd.DataFrame:
    fail_df = build_failure_labels(df)

    df = df.merge(fail_df, on="config_id", how="left")

    df_nonfailed = (
        df[df["failed"] == 0]
        .copy()
        .reset_index(drop=True)
    )

    return df_nonfailed


def compute_fidelity_correlation(
        df: pd.DataFrame,
        fidelity_a: float,
        fidelity_b: float,
        metric_col: str,
):
    df_a = df[
        df["training_progress_rounded"] == fidelity_a
        ][["config_id", metric_col]]

    df_b = df[
        df["training_progress_rounded"] == fidelity_b
        ][["config_id", metric_col]]

    df_a = df_a.rename(
        columns={metric_col: f"{metric_col}_a"}
    )

    df_b = df_b.rename(
        columns={metric_col: f"{metric_col}_b"}
    )

    merged = pd.merge(
        df_a,
        df_b,
        on="config_id",
        how="inner"
    )

    corr, p_value = spearmanr(
        merged[f"{metric_col}_a"],
        merged[f"{metric_col}_b"]
    )

    return corr, p_value, len(merged)


def build_fidelity_matrices(
        df: pd.DataFrame,
        metric_col: str,
        min_shared_configs: int = 10,
):
    fidelities = sorted(df["training_progress_rounded"].unique())

    corr_matrix = pd.DataFrame(
        index=fidelities,
        columns=fidelities,
        dtype=float
    )

    count_matrix = pd.DataFrame(
        index=fidelities,
        columns=fidelities,
        dtype=float
    )

    for fidelity_a in fidelities:
        for fidelity_b in fidelities:
            corr, _, n_shared = compute_fidelity_correlation(
                df=df,
                fidelity_a=fidelity_a,
                fidelity_b=fidelity_b,
                metric_col=metric_col,
            )

            count_matrix.loc[fidelity_a, fidelity_b] = n_shared

            if n_shared >= min_shared_configs:
                corr_matrix.loc[fidelity_a, fidelity_b] = corr
            else:
                corr_matrix.loc[fidelity_a, fidelity_b] = float("nan")

    return corr_matrix, count_matrix


def plot_heatmap(
        matrix: pd.DataFrame,
        title: str,
        output_path: str,
):
    plot_matrix = matrix.copy()

    plot_matrix = plot_matrix.replace(0, float("nan"))

    plt.figure(figsize=(12, 10))

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="white")

    plt.imshow(
        plot_matrix.astype(float),
        aspect="auto",
        cmap=cmap
    )

    plt.colorbar()

    plt.xticks(
        ticks=range(len(plot_matrix.columns)),
        labels=plot_matrix.columns,
        rotation=90
    )

    plt.yticks(
        ticks=range(len(plot_matrix.index)),
        labels=plot_matrix.index
    )

    plt.title(title)
    plt.xlabel("Training progress")
    plt.ylabel("Training progress")

    plt.tight_layout()

    plt.savefig(output_path, dpi=300)

    plt.show()

    plt.close()


def compute_correlation_to_final_fidelity(
        df: pd.DataFrame,
        metric_col: str,
):
    fidelities = sorted(
        df["training_progress_rounded"].unique()
    )

    final_fidelity = max(fidelities)

    rows = []

    for fidelity in fidelities:
        corr, p_value, n_shared = (
            compute_fidelity_correlation(
                df=df,
                fidelity_a=fidelity,
                fidelity_b=final_fidelity,
                metric_col=metric_col,
            )
        )

        rows.append({
            "fidelity": fidelity,
            "final_fidelity": final_fidelity,
            "spearman_corr": corr,
            "p_value": p_value,
            "n_shared": n_shared,
        })

    result_df = pd.DataFrame(rows)

    return result_df


def plot_correlation_to_final_fidelity(
        result_df: pd.DataFrame,
        output_path: str,
):
    result_df = result_df[
        result_df["fidelity"] != 1.0
        ].copy()
    plt.figure(figsize=(10, 6))

    plt.plot(
        result_df["fidelity"],
        result_df["spearman_corr"],
        color="gray",
        linewidth=1.5,
        alpha=0.7,
        zorder=1,
    )

    scatter = plt.scatter(
        result_df["fidelity"],
        result_df["spearman_corr"],
        c=result_df["n_shared"],
        marker="^",
        s=90,
        edgecolors="black",
        linewidths=0.6,
        zorder=2,
    )

    cbar = plt.colorbar(scatter)
    cbar.set_label("Shared configs with final fidelity")

    plt.xlabel("Training progress fidelity")
    plt.ylabel("Spearman correlation to final fidelity")
    plt.title("Fidelity correlation to fully trained models")

    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()


def compute_downstream_fidelity_correlations(
        df: pd.DataFrame,
        downstream_cols: list[str],
        fidelities_to_compare: list[float],
        final_fidelity: float = 1.0,
):
    rows = []

    for task_col in downstream_cols:
        for fidelity in fidelities_to_compare:
            corr, p_value, n_shared = compute_fidelity_correlation(
                df=df,
                fidelity_a=fidelity,
                fidelity_b=final_fidelity,
                metric_col=task_col,
            )

            rows.append({
                "task": task_col,
                "fidelity_pair": f"({fidelity}, {final_fidelity})",
                "fidelity": fidelity,
                "final_fidelity": final_fidelity,
                "spearman_corr": corr,
                "p_value": p_value,
                "n_shared": n_shared,
            })

    return pd.DataFrame(rows)


def clean_task_label(task: str) -> str:
    exact_mapping = {
        "vtab_kitti_closest_vehicle_distance": "vtab_kitti_cvd",
        "vtab_clevr_closest_object_distance": "vtab_clevr_cod",
        "retrieval_flickr_1k_test_image_text_retrieval": "flickr",
        "retrieval_mscoco_2014_5k_test_image_text_retrieval": "mscoco"
    }

    suffixes_to_remove = [
        "_mean_per_class_recall",
        "_mean_recall@1",
        "_jaccard_score_10-12",
        "_F1-macro_all",
        "_acc_worst_region",
        "_acc_top5_wg",
        "_acc_race_avg",
        "_acc_wg",
        "_acc1",
    ]

    label = task

    for suffix in suffixes_to_remove:
        label = label.replace(suffix, "")

    if label in exact_mapping:
        label = exact_mapping[label]

    return label


def plot_downstream_fidelity_correlations(
        corr_df: pd.DataFrame,
        output_path: str,
):
    corr_df = corr_df.copy()

    # Remove accidental whitespace and map readable names
    corr_df["task"] = corr_df["task"].astype(str).str.strip()

    corr_df["task_label"] = (
        corr_df["task"]
        .map(TASK_NAME_MAP)
        .fillna(corr_df["task"])
    )

    task_order = (
        corr_df.groupby("task")["spearman_corr"]
        .mean()
        .sort_values()
        .index
        .tolist()
    )

    x_positions = {
        task: i
        for i, task in enumerate(task_order)
    }

    colors = {
        0.25: "#1f77b4",
        0.5: "#ff7f0e",
        0.75: "#2ca02c",
    }

    plt.figure(figsize=(14, 5))

    for fidelity in sorted(corr_df["fidelity"].unique()):
        sub = corr_df[corr_df["fidelity"] == fidelity]

        plt.scatter(
            [x_positions[task] for task in sub["task"]],
            sub["spearman_corr"],
            marker="v",
            color=colors.get(fidelity, "gray"),
            s=80,
            edgecolors="black",
            linewidths=0.5,
            label=f"({fidelity}, 1.0)",
            alpha=0.9,
        )

    task_labels = (
        corr_df[["task", "task_label"]]
        .drop_duplicates("task")
        .set_index("task")["task_label"]
    )

    plt.xticks(
        ticks=range(len(task_order)),
        labels=[task_labels.loc[task] for task in task_order],
        rotation=45,
        ha="right",
    )

    plt.ylim(0, 1.05)
    plt.ylabel("Spearman correlation")
    plt.xlabel("Task")
    plt.title("Spearman rank correlation to final fidelity per task")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Training progress", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


if __name__ == "__main__":
    df = pd.read_csv("surrogate_data.csv")

    df = keep_only_nonfailed(df)

    print(f"Nonfailed configs: {df['config_id'].nunique()}")
    print(f"Nonfailed rows: {len(df)}")

    df["training_progress_rounded"] = (
        df["training_progress"].round(3)
    )

    unique_fidelities = sorted(
        df["training_progress_rounded"].unique()
    )

    print("\nUnique fidelity levels:")
    print(unique_fidelities)

    fidelity_counts = (
        df["training_progress_rounded"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    fidelity_counts.columns = [
        "training_progress",
        "num_points"
    ]

    print("\nPoints per fidelity:")
    print(fidelity_counts.to_string(index=False))

    metric_col = "val_loss"

    corr_matrix, count_matrix = build_fidelity_matrices(
        df=df,
        metric_col=metric_col,
        min_shared_configs=10,
    )

    print("\nSpearman correlation matrix:")
    print(corr_matrix)

    print("\nShared config count matrix:")
    print(count_matrix)

    corr_matrix.to_csv("fidelity_spearman_correlation_matrix.csv")
    count_matrix.to_csv("fidelity_shared_config_count_matrix.csv")

    plot_heatmap(
        matrix=corr_matrix,
        title=f"Spearman fidelity correlation using {metric_col}",
        output_path="fidelity_spearman_correlation_heatmap.png",
    )

    plot_heatmap(
        matrix=count_matrix,
        title="Number of shared configs between fidelities",
        output_path="fidelity_shared_config_count_heatmap.png",
    )

    final_corr_df = compute_correlation_to_final_fidelity(
        df=df,
        metric_col=metric_col,
    )

    print("\nCorrelation to final fidelity:")
    print(final_corr_df)

    final_corr_df.to_csv(
        "correlation_to_final_fidelity.csv",
        index=False
    )

    plot_correlation_to_final_fidelity(
        result_df=final_corr_df,
        output_path="correlation_to_final_fidelity.png",
    )

    downstream_cols = [
        "vtab_caltech101_mean_per_class_recall",
        "cifar10_mean_per_class_recall",
        "vtab_cifar100_mean_per_class_recall",
        "vtab_clevr_count_all_acc1",
        "vtab_clevr_closest_object_distance_acc1",
        "country211_acc1",
        "vtab_dtd_acc1",
        "vtab_eurosat_acc1",
        "fgvc_aircraft_mean_per_class_recall",
        "food101_acc1",
        "gtsrb_acc1",
        "imagenet1k_acc1",
        "imagenet_sketch_acc1",
        # "imagenetv2_acc1",
        "imagenet-a_acc1",
        "imagenet-o_acc1",
        "imagenet-r_acc1",
        "vtab_kitti_closest_vehicle_distance_acc1",
        "mnist_acc1",
        "objectnet_acc1",
        "vtab_flowers_mean_per_class_recall",
        "vtab_pets_mean_per_class_recall",
        "voc2007_acc1",
        "vtab_pcam_acc1",
        "renderedsst2_acc1",
        "vtab_resisc45_acc1",
        "cars_acc1",
        "stl10_acc1",
        "sun397_acc1",
        "vtab_svhn_acc1",
        "retrieval_flickr_1k_test_image_text_retrieval_mean_recall@1",
        "retrieval_mscoco_2014_5k_test_image_text_retrieval_mean_recall@1",
        "misc_winogavil_jaccard_score_10-12",
        "wilds_iwildcam_F1-macro_all",
        "wilds_camelyon17_acc1",
        "wilds_fmow_acc_worst_region",
        "fairness_dollar_street_acc_top5_wg",
        "fairness_geode_acc_wg",
        "fairness_fairface_acc_race_avg",
        "fairness_utkface_acc_race_avg",
        "val_loss",
        "test_loss"
    ]

    corr_df = compute_downstream_fidelity_correlations(
        df=df,
        downstream_cols=downstream_cols,
        fidelities_to_compare=[0.25, 0.5, 0.75],
        final_fidelity=1.0,
    )

    print(corr_df)

    plot_downstream_fidelity_correlations(
        corr_df=corr_df,
        output_path="downstream_fidelity_correlations.png",
    )
