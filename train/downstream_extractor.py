import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root_dir",
        type=Path,
        required=True,
    )
    return p.parse_args()


def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def has_nan_checkpoint(checkpoints_dir: Path) -> bool:
    pt_files = list(checkpoints_dir.glob("*.pt"))

    nan_files = [f for f in pt_files if "nan" in f.name.lower()]

    return len(nan_files) == 1


def build_results_dict(root_dir, metric_map):
    root_dir = Path(root_dir) / "individual_training_logs"
    results = {}
    total_eval_time_spent = {}

    for config_dir in root_dir.iterdir():
        if not config_dir.is_dir() or not config_dir.name.startswith("train_config"):
            continue

        config_id = config_dir.name.replace("train_config", "")
        results[config_id] = {}

        checkpoints_dir = config_dir / "checkpoints"
        if has_nan_checkpoint(checkpoints_dir):
            continue

        downstream_dir = config_dir / "downstream"
        if not downstream_dir.exists():
            continue

        for epoch_dir in downstream_dir.iterdir():
            if not epoch_dir.is_dir() or not epoch_dir.name.startswith("epoch_"):
                continue

            epoch = epoch_dir.name.replace("epoch_", "")

            jsonl_files = list(epoch_dir.glob("*.jsonl"))
            if not jsonl_files:
                continue

            time_record_json = epoch_dir / "eval_time.json"
            with open(time_record_json, "r") as f:
                train_data = json.load(f)

            gpu = train_data.get("gpu_name") or "NVIDIA A100-SXM4-40GB"
            train_time = train_data["eval_time_seconds"] / 3600

            total_eval_time_spent[gpu] = total_eval_time_spent.get(gpu, 0) + train_time

            file_path = jsonl_files[0]

            data = load_jsonl(file_path)

            epoch_results = {
                f"{dataset}_{metric_map.get(dataset, 'unknown_metric')}":
                    entry["metrics"].get(metric_map.get(dataset), None)
                for entry in data
                for dataset in [entry["key"].replace("/", "_")]
            }

            results[config_id][epoch] = epoch_results
    print(total_eval_time_spent)
    return results


def results_dict_to_csv(results: dict, output_csv: Path):
    rows = []

    for config_id, epochs_dict in results.items():
        for epoch, downstream_metrics in epochs_dict.items():
            row = {
                "config_id": config_id,
                "epoch": epoch,
            }
            row.update(downstream_metrics)
            rows.append(row)

    df = pd.DataFrame(rows)

    df["config_id"] = pd.to_numeric(df["config_id"], errors="ignore")
    df["epoch"] = pd.to_numeric(df["epoch"], errors="ignore")
    df = df.sort_values(["config_id", "epoch"])

    df.to_csv(output_csv, index=False)
    return df


if __name__ == "__main__":
    args = parse_args()

    root_dir: Path = args.root_dir

    metric_map = {
        "vtab_caltech101": "mean_per_class_recall",
        "cifar10": "mean_per_class_recall",
        "vtab_cifar100": "mean_per_class_recall",
        "vtab_clevr_count_all": "acc1",
        "vtab_clevr_closest_object_distance": "acc1",
        "country211": "acc1",
        "vtab_dtd": "acc1",
        "vtab_eurosat": "acc1",
        "fgvc_aircraft": "mean_per_class_recall",
        "food101": "acc1",
        "gtsrb": "acc1",
        "imagenet1k": "acc1",
        "imagenet_sketch": "acc1",
        "imagenetv2": "acc1",
        "imagenet-a": "acc1",
        "imagenet-o": "acc1",
        "imagenet-r": "acc1",
        "vtab_kitti_closest_vehicle_distance": "acc1",
        "mnist": "acc1",
        "objectnet": "acc1",
        "vtab_flowers": "mean_per_class_recall",
        "vtab_pets": "mean_per_class_recall",
        "voc2007": "acc1",
        "vtab_pcam": "acc1",
        "renderedsst2": "acc1",
        "vtab_resisc45": "acc1",
        "cars": "acc1",
        "stl10": "acc1",
        "sun397": "acc1",
        "vtab_svhn": "acc1",
        "retrieval_flickr_1k_test_image_text_retrieval": "mean_recall@1",
        "retrieval_mscoco_2014_5k_test_image_text_retrieval": "mean_recall@1",
        "misc_winogavil": "jaccard_score_10-12",
        "wilds_iwildcam": "F1-macro_all",
        "wilds_camelyon17": "acc1",
        "wilds_fmow": "acc_worst_region",
        "fairness_dollar_street": "acc_top5_wg",
        "fairness_geode": "acc_wg",
        "fairness_fairface": "acc_race_avg",
        "fairness_utkface": "acc_race_avg",
    }
    downstream_results = build_results_dict(root_dir, metric_map)

    output_csv = root_dir / "downstream_results.csv"
    results_dict_to_csv(downstream_results, output_csv)
