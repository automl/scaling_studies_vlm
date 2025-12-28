from config import ZeroShotDataset, RetrievalDataset
from eval.downstream_eval import ClipBenchmarkEvaluator

if __name__ == "__main__":
   evaluator = ClipBenchmarkEvaluator(root="logs/ViT-S-32_128e5_bs_512_4_L40s_seed_1_baseline")
   zero_shot_results = evaluator.zero_shot_eval([ZeroShotDataset.CIFAR10, ZeroShotDataset.IMAGENET])
   retrieval_results = evaluator.zero_shot_retrieval([RetrievalDataset.MS_COCO])
