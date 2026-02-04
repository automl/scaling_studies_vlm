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
        type=int,
        help="Id of the configuration being evaluated."
    )
    return p.parse_args()


def enrich_neps_state(neps_root, pipeline_id: int, eval_dict: dict):
    config_dir = neps_root / f"configs/config_{pipeline_id}"
    report_path = config_dir / "report.yaml"

    with open(report_path, "r") as f:
        state = yaml.safe_load(f) or {}

    info = state.setdefault("info_dict", {})
    info["downstream_results"] = eval_dict

    neps.save_pipeline_results(
        user_result=state,
        pipeline_id=str(pipeline_id),
        root_directory=neps_root,
    )


if __name__ == "__main__":
    args = parse_args()

    neps_root = Path(args.root).resolve().parents[1]

    evaluator = ClipBenchmarkEvaluator(root=args.root, model=args.model, default_seed=args.seed)
    zero_shot_results = evaluator.zero_shot_eval([ZeroShotDataset.CIFAR10, ZeroShotDataset.IMAGENET])
    retrieval_results = evaluator.zero_shot_retrieval([RetrievalDataset.MS_COCO])
    full_results = {
        "zero_shot_results": zero_shot_results,
        "retrieval_results": retrieval_results
    }
    enrich_neps_state(neps_root=neps_root, pipeline_id=args.pipeline_id, eval_dict=full_results)
