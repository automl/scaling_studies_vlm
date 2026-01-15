import argparse

from eval.downstream_eval import ClipBenchmarkEvaluator, ZeroShotDataset, RetrievalDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        required=True,
        help="Path to the run directory."
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    evaluator = ClipBenchmarkEvaluator(root=args.root)
    # zero_shot_results = evaluator.zero_shot_eval([ZeroShotDataset.CIFAR10, ZeroShotDataset.IMAGENET])
    # retrieval_results = evaluator.zero_shot_retrieval([RetrievalDataset.MS_COCO])
    zero_shot_results = evaluator.zero_shot_eval([ZeroShotDataset.CIFAR10])
