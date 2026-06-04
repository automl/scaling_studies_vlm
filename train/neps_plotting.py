import neps
from neps.optimizers.algorithms import chinchilla_guided_scaling, random_search

from spaces import ScanPipeline

root_dir = f"/work/dlclarge1/sinanid-VLM-scaling-law/scaling_studies_vlm/train/results/optimizer=gs_micro/grid_search"
# root_dir3 = f"/work/dlclarge1/sinanid-VLM-scaling-law/scaling_studies_vlm/train/results/optimizer=rs_micro/seed=42"
root_dir2 = f"/work/dlclarge1/sinanid-VLM-scaling-law/scaling_studies_vlm/train/results/optimizer=bo_space_expansion_micro/bo_space_expansion"


if __name__ == "__main__":
    neps.extrapolate(root_directory=root_dir, max_target_flops=10_000_000,
                     optimizer=lambda space: chinchilla_guided_scaling(space, base_optimizer=random_search(space, use_priors=True)))
    
    neps.plot_study_artifacts(
        study_dirs=[root_dir, root_dir2],
        output_dir="/work/dlclarge1/sinanid-VLM-scaling-law/scaling_studies_vlm/train/results/combined_plots_gs_bo/",
        optimizer=lambda space: chinchilla_guided_scaling(space, base_optimizer=random_search(space, use_priors=True)),
    )
