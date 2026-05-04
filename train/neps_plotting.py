import neps
from neps.optimizers.algorithms import chinchilla_guided_scaling, random_search

from spaces import ScanPipeline

root_dir = f"/p/data1/datasets/scan/openclip/train/results/optimizer=bo_space_expansion_micro/seed=42/"


if __name__ == "__main__":
    neps.extrapolate(root_directory=root_dir, max_target_flops=10_000_000,
                     optimizer=lambda space: chinchilla_guided_scaling(space, base_optimizer=random_search(space, use_priors=True)))
