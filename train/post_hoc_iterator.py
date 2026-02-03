import argparse

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
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    evaluator = ClipBenchmarkEvaluator(root=args.root, model=args.model, default_seed=args.seed)
    zero_shot_results = evaluator.zero_shot_eval([ZeroShotDataset.CIFAR10])
    # retrieval_results = evaluator.zero_shot_retrieval([RetrievalDataset.MS_COCO])
