import logging
import random
import sys
import time
from pathlib import Path
import time

import neps
import numpy as np
import yaml

import params
from config import HPO_CONFIGS
from optimizers import set_workspace_dir


def seed_generator_process(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def patch_neps_cuda_seed_restore() -> None:
    from neps.state.seed_snapshot import SeedSnapshot

    original_set_seed_state = SeedSnapshot.set_as_global_seed_state

    def set_seed_state_with_cuda_guard(self):
        cuda_rng = self.torch_cuda_rng
        if cuda_rng is None:
            return original_set_seed_state(self)

        try:
            import torch

            if not torch.cuda.is_available():
                self.torch_cuda_rng = None
                return original_set_seed_state(self)

            available_cuda_generators = len(torch.cuda.default_generators)
            if available_cuda_generators == 0:
                print(
                    "[WARN] NePS CUDA RNG snapshot restore skipped because "
                    "this process has no CUDA default generators."
                )
                self.torch_cuda_rng = None
                return original_set_seed_state(self)

            if len(cuda_rng) > available_cuda_generators:
                print(
                    "[WARN] NePS CUDA RNG snapshot has "
                    f"{len(cuda_rng)} states, but this process has "
                    f"{available_cuda_generators} CUDA generators. "
                    "Truncating CUDA RNG state restore."
                )
                self.torch_cuda_rng = cuda_rng[:available_cuda_generators]

            return original_set_seed_state(self)
        finally:
            self.torch_cuda_rng = cuda_rng

    SeedSnapshot.set_as_global_seed_state = set_seed_state_with_cuda_guard



def evaluate_pipeline(pipeline_directory, pipeline_id, **conf):

    print("in evaluate_pipeline with conf:", conf)
    import os
    
    # First, try to use the root_dir from environment (set in main())
    root_dir = os.environ.get('NEPS_ROOT_DIR')
    
    if root_dir:
        print(f"[DEBUG] Using root_dir from environment: {root_dir}")
    else:
        # Fallback: reconstruct from pipeline_directory  
        print(f"[DEBUG] NEPS_ROOT_DIR not in environment, reconstructing from pipeline_directory")
        pipeline_path = Path(pipeline_directory)
        
        # Convert to absolute if relative
        if not pipeline_path.is_absolute():
            pipeline_path = (Path.cwd() / pipeline_path).resolve()
        else:
            pipeline_path = pipeline_path.resolve()
        
        # Navigate up: configs/config_X -> configs -> root_dir
        root_dir = str(pipeline_path.parent.parent)
        
        print(f"[DEBUG] pipeline_directory input: {pipeline_directory}")
        print(f"[DEBUG] pipeline_path.is_absolute(): {pipeline_path.is_absolute()}")
        print(f"[DEBUG] pipeline_path resolved: {pipeline_path}")
        print(f"[DEBUG] reconstructed root_dir: {root_dir}")
    
    from configs_iterator import generate_scripts_per_config, normalize_scan_space
    seed = int(os.environ['NEPS_SEED'])
    rng = np.random.default_rng(seed)
    scan_space = normalize_scan_space(dict(conf), bias_scale_sampling=False, alpha=1.0, rng=rng)
    generate_scripts_per_config(
        config_id=pipeline_id,
        scan_space=scan_space,
        root_dir=root_dir,
        venv_path="/software/genoa/r24.10/Anaconda3/2024.06-1/bin/activate",
        wandb_project_name="",
        workspace_dir="/data/cat/ws/naal364i-primo/runs",
        cluster="tu_dresden",
        command_dir="/home/naal364i/scaling_studies_vlm/train",
        partition="capella",
        seed=seed,
    )
    report_path = Path(pipeline_directory).parent.parent / "configs" / f"config_{pipeline_id}" / "report_final.yaml"
    
    max_wait_time = 600  # 10 minutes timeout
    wait_interval = 2  # Check every 2 seconds
    elapsed = 0
    
    while not report_path.exists() and elapsed < max_wait_time:
        logging.info(f"Waiting for report.yaml at {report_path}... ({elapsed}s)")
        time.sleep(wait_interval)
        elapsed += wait_interval
    
    if not report_path.exists():
        logging.error(f"Timeout waiting for report.yaml at {report_path}")
        raise TimeoutError(f"report.yaml not generated within {max_wait_time} seconds")
    
    logging.info(f"Found report.yaml, reading results...")
    with open(report_path, "r") as f:
        report_data = yaml.safe_load(f) or {}
    
    # Extract data from report
    objective_to_minimize = report_data.get("objective_to_minimize")
    cost = report_data.get("cost")
    extra_info = (
        report_data.get("extra")
        or report_data.get("info_dict")
        or report_data.get("extra_info")
        or {}
    )
    print(f"Objective to minimize: {objective_to_minimize}, Cost: {cost}")
    return {
        "objective_to_minimize": objective_to_minimize,
        "cost": cost,
        "extra_info": extra_info,
    }

def main(args, root_dir):
    # Set workspace directory for disk-based FLOP cache
    set_workspace_dir(args.workspace_dir)
    patch_neps_cuda_seed_restore()
    
    # Store root_dir and seed in environment so evaluate_pipeline can access them reliably
    import os
    os.environ['NEPS_ROOT_DIR'] = str(Path(root_dir).resolve())
    os.environ['NEPS_SEED'] = str(args.seed)
    print(f"[DEBUG] Set NEPS_ROOT_DIR={os.environ['NEPS_ROOT_DIR']}")
    print(f"[DEBUG] Set NEPS_SEED={os.environ['NEPS_SEED']}")
    
    neps.run(
        evaluate_pipeline=evaluate_pipeline,
        root_directory=root_dir,
        # overwrite_root_directory=True,
        **HPO_CONFIGS.get(args.hpo_optimizer).run_args,
    )


def worker_proc(rank, args, root_dir):
    print(f"Starting NePS worker {rank}", flush=True)
    main(args, root_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    args = params.parse_args(sys.argv[1:])
    seed_generator_process(args.seed)

    root_dir = f"{args.workspace_dir}/train/results/optimizer={args.hpo_optimizer}/seed={args.seed}"

    print("Starting HPO...")
    start = time.time()

    main(args, root_dir)

    end = time.time()
    elapsed = end - start

    print(f"Finished HPO in {elapsed:.2f} seconds")
