import re
from pathlib import Path

import neps
import pandas as pd
import yaml


def safe_load_yaml(p: Path):
    try:
        return yaml.safe_load(p.read_text()) or {}
    except Exception:
        return None


def safe_get(dct, path, default=None):
    cur = dct
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def flatten_dict(d: dict, prefix: str = "hp", sep: str = ".") -> dict:
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, prefix=key, sep=sep))
        else:
            out[key] = v
    return out


def extract_imagenet1k_acc1(report: dict):
    zs = safe_get(report, ["extra", "eval_results", "zero_shot"], default=[])
    if not isinstance(zs, list):
        return None
    for item in zs:
        if isinstance(item, dict) and item.get("dataset") == "imagenet1k":
            return item.get("acc1")
    return None

def extract_cifar10_acc1(report: dict):
    zs = safe_get(report, ["extra", "eval_results", "zero_shot"], default=[])
    if not isinstance(zs, list):
        return None
    for item in zs:
        if isinstance(item, dict) and item.get("dataset") == "cifar10":
            return item.get("acc1")
    return None


def extract_mscoco_image_retrieval(report: dict):
    ret = safe_get(report, ["extra", "eval_results", "retrieval"], default=[])
    if not isinstance(ret, list):
        return None
    for item in ret:
        if isinstance(item, dict) and item.get("dataset") == "mscoco_captions":
            return item.get("image_retrieval")
    return None

def extract_mscoco_text_retrieval(report: dict):
    ret = safe_get(report, ["extra", "eval_results", "retrieval"], default=[])
    if not isinstance(ret, list):
        return None
    for item in ret:
        if isinstance(item, dict) and item.get("dataset") == "mscoco_captions":
            return item.get("text_retrieval")
    return None


def build_trial_dir_map(root: Path) -> dict[int, Path]:
    trial_dirs: dict[int, Path] = {}
    for rp in root.rglob("report.yaml"):
        trial_id = None
        for parent in rp.parents:
            m = re.fullmatch(r"config_(\d+)", parent.name)
            if m:
                trial_id = int(m.group(1))
                trial_dirs[trial_id] = parent
                break
    return trial_dirs


if __name__ == "__main__":
    root_dir = Path("results/model=ViT-S-32/seed=0")
    out_csv = root_dir / "trials_summary.csv"

    full_df, _ = neps.status(str(root_dir))
    trial_ids = [int(i) for i in full_df.index.tolist()]

    trial_dir_map = build_trial_dir_map(root_dir)

    rows = []
    for trial_id in trial_ids:
        trial_dir = trial_dir_map.get(trial_id)
        if trial_dir is None:
            continue

        report_path = trial_dir / "report.yaml"
        config_path = trial_dir / "config.yaml"

        if not report_path.exists() or not config_path.exists():
            continue

        report = safe_load_yaml(report_path)
        cfg = safe_load_yaml(config_path)
        if report is None or cfg is None:
            continue

        val_loss = safe_get(report, ["extra", "validation_loss"])
        test_loss = safe_get(report, ["extra", "test_loss"])
        train_loss = safe_get(report, ["extra", "train_loss"])

        if val_loss is None or test_loss is None or train_loss is None:
            continue

        row = {
            "trial_id": trial_id,
            "validation_loss": val_loss,
            "test_loss": test_loss,
            "train_loss": train_loss,
            "imagenet1k_acc1": extract_imagenet1k_acc1(report),
            "cifar10_acc1": extract_cifar10_acc1(report),
            "mscoco_image_retrieval": extract_mscoco_image_retrieval(report),
        }

        row.update(flatten_dict(cfg, prefix="hp"))

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("trial_id")
    df.to_csv(out_csv, index=False)

    print(f"Wrote: {out_csv}")
    print(f"Trials in neps.status: {len(trial_ids)}")
    print(f"Trials with trial_dir found: {len(trial_dir_map)}")
    print(f"Rows written: {len(df)}")
