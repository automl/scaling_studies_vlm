import argparse
from pathlib import Path

import neps
import yaml

from eval.downstream_eval import ClipBenchmarkEvaluator, ZeroShotDataset, RetrievalDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        required=True,
        help="Path to the run directory."
    )
    p.add_argument(
        "--model",
        required=True,
        type=str,
        help="Model to be evaluated."
    )
    p.add_argument(
        "--seed",
        required=True,
        type=int,
        help="Seed for reproducibility."
    )

    p.add_argument(
        "--pipeline_id",
        required=True,
        type=str,
        help="Id of the configuration being evaluated."
    )

    p.add_argument(
        "--cluster",
        default="meta",
        type=str,
        choices=["meta", "juwels"],
        help="Cluster used to run evaluations."
    )
    return p.parse_args()


def enrich_neps_state(neps_root, pipeline_id, eval_dict: dict):
    config_dir = neps_root / f"configs/config_{pipeline_id}"
    train_report_path = config_dir / "report_train.yaml"
    down_report_path = config_dir / "report_down.yaml"

    with open(train_report_path, "r") as f:
        state = yaml.safe_load(f) or {}

    info_dict = state.get("info_dict") or state.pop("extra", {}) or {}
    state["info_dict"] = info_dict

    info_dict["downstream_results"] = eval_dict

    with open(down_report_path, "w") as f:
        yaml.safe_dump(state, f, sort_keys=False)


def main(root: str, model: str, seed: int, pipeline_id: str, cluster: str = "meta"):
    """
    Run downstream evaluation and enrich neps state with results.
    
    Args:
        root: Path to the run directory
        model: Model to be evaluated
        seed: Seed for reproducibility
        pipeline_id: Id of the configuration being evaluated
        cluster: Cluster used to run evaluations (default: "meta")
    """
    neps_root = Path(root).resolve().parents[1]

    evaluator = ClipBenchmarkEvaluator(root=root, model=model, default_seed=seed, cluster=cluster)
    zero_shot_results = evaluator.zero_shot_eval([ZeroShotDataset.CIFAR10, ZeroShotDataset.IMAGENET])
    retrieval_results = evaluator.zero_shot_retrieval([RetrievalDataset.MS_COCO])
    full_results = {
        "zero_shot_results": zero_shot_results,
        "retrieval_results": retrieval_results
    }
    enrich_neps_state(neps_root=neps_root, pipeline_id=pipeline_id, eval_dict=full_results)


if __name__ == "__main__":
    args = parse_args()
    main(
        root=args.root,
        model=args.model,
        seed=args.seed,
        pipeline_id=args.pipeline_id,
        cluster=args.cluster,
    )
