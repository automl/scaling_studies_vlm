import neps
from neps.optimizers.algorithms import chinchilla_guided_scaling, random_search

from spaces import ScanPipeline

root_dir = f"/work/dlclarge1/sinanid-VLM-scaling-law/scaling_studies_vlm/train/results/optimizer=scale_space_only_search_biased_scale_space/seed=42"

space = ScanPipeline.with_overrides(lr=2e-3, warmup_fraction=0.5, wd=1.0e-6)

if __name__ == "__main__":
    neps.extrapolate(root_directory=root_dir, max_target_flops=10_000_000,
                     optimizer=lambda space: chinchilla_guided_scaling(space, base_optimizer=random_search(space, use_priors=True)))
