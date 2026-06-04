import gc
import math
import os
import shlex
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import neps
import numpy as np
import pandas as pd
import torch

import iterator_params
from clip_train.profiler import get_model_flops
from config import EvalConfig
from probe_max_bs import probe_max_bs
from slurm import SlurmManager, build_srun_cmd
from utils import query_bs, build_model_cfg, get_local_model_str, precision_to_dtype, create_default_model, \
    sample_warped_log_uniform, sample_small_favoring_index


def group_configs_by_all_except_train_num_samples(df: pd.DataFrame, bias_scale_sampling: bool, alpha: float) -> Dict:
    """
    Convert all rows to scan_space dicts, then group by all keys except train_num_samples.
    Returns a Dict where each group key (frozenset) maps to a list of (config_id, scan_space) tuples.
    """
    groups = {}

    for config_id, row in df.iterrows():
        scan_space = build_scan_space(row, bias_scale_sampling, alpha)

        # Create a hashable key from all keys except train_num_samples
        group_key = frozenset((k, v) for k, v in scan_space.items() if k != 'train_num_samples')

        if group_key not in groups:
            groups[group_key] = []

        groups[group_key].append((config_id, scan_space))

    return groups


def infer_hardware_setup_and_local_bs(
        precision,
        model_name: str,
        global_batch_size: int,
        max_gpus_per_node: int = 4,
):
    local_bs = probe_max_bs(model_name, precision)

    if local_bs > global_batch_size:
        local_bs = global_batch_size

    if global_batch_size % local_bs != 0:
        raise ValueError(
            f"global_batch_size ({global_batch_size}) must be divisible by local_bs ({local_bs})"
        )

    gpus_needed = global_batch_size // local_bs
    gpus_per_node = min(gpus_needed, max_gpus_per_node)
    nodes_nr = math.ceil(gpus_needed / max_gpus_per_node)

    return gpus_per_node, nodes_nr, local_bs


def get_hp_space_train_num_samples_from_scan_space(scan_space):
    HP_KEYS = ["lr", "wd", "warmup_fraction", "beta1", "beta2", "eps"]
    SCALE_KEYS = ["train_num_samples", "vision_width", "text_width", "text_heads", "epochs"]

    hp_only = {k: v for k, v in scan_space.items() if k in HP_KEYS}
    scale_only = {k: v for k, v in scan_space.items() if k in SCALE_KEYS}

    if scale_only.get("train_num_samples") is not None:
        hp_only["train_num_samples"] = scale_only["train_num_samples"]
    
    if scale_only.get("epochs") is not None:
        hp_only["epochs"] = scale_only["epochs"]
    
    return hp_only


def build_scan_space(row, bias_scale_sampling, alpha, rng) -> Dict[str, Any]:
    scan_space = {
        col.replace("config.", ""): val
        for col, val in row.items()
        if col.startswith("config.")
    }
    return normalize_scan_space(scan_space, bias_scale_sampling, alpha, rng)


def _decode_text(idx: int) -> tuple[int, int]:
    # width, heads
    if idx == 0:
        return 32, 1
    return idx * 64, idx


def _decode_vision(idx: int) -> tuple[int, int]:
    # width, head_width
    if idx == 0:
        return 32, 32
    return idx * 64, 64


def normalize_scan_space(
        scan_space: Dict[str, Any],
        bias_scale_sampling: bool,
        alpha: float,
        rng,
) -> Dict[str, Any]:
    if scan_space.get("text_width") is not None:
        idx = sample_small_favoring_index(alpha=alpha, rng=rng) if bias_scale_sampling else int(
            scan_space["text_width"])
        width, heads = _decode_text(idx)
        scan_space["text_width"] = width
        scan_space["text_heads"] = heads

    if scan_space.get("vision_width") is not None:
        idx = sample_small_favoring_index(alpha=alpha, rng=rng) if bias_scale_sampling else int(
            scan_space["vision_width"])
        width, head_width = _decode_vision(idx)
        scan_space["vision_width"] = width
        scan_space["vision_head_width"] = head_width

    x = scan_space["train_num_samples"]
    if bias_scale_sampling:
        x = sample_warped_log_uniform(x=x, alpha=alpha)
    scan_space["train_num_samples"] = int(float(x) * 1_000_000)

    return scan_space


def create_directories(root_dir, config_id):
    logs_dir = os.path.join(root_dir, "individual_training_logs")
    scripts_dir = os.path.join(root_dir, "train_scripts")
    evaluation_scripts_path = os.path.join(root_dir, "evaluation_scripts")
    downstream_evaluation_scripts_path = os.path.join(root_dir, "downstream_evaluation_scripts")
    model_cfg_path = os.path.join(root_dir, "model_cfg", f"config_{config_id}")

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(scripts_dir, exist_ok=True)
    os.makedirs(evaluation_scripts_path, exist_ok=True)
    os.makedirs(downstream_evaluation_scripts_path, exist_ok=True)
    return logs_dir, scripts_dir, evaluation_scripts_path, downstream_evaluation_scripts_path, model_cfg_path


def compute_batch_and_resources(
        model_cfg: str,
        hp_scale_space: Dict[str, Any],
        precision: str,
        max_gpus_per_node: int,
) -> Tuple[int, int, int]:
    global_batch_size = query_bs(hp_scale_space.get("train_num_samples"))
    gpus_per_node, nodes_nr, local_bs = infer_hardware_setup_and_local_bs(
        model_name=model_cfg,
        precision=precision,
        global_batch_size=global_batch_size,
        max_gpus_per_node=max_gpus_per_node,
    )

    print(f"global_batch_size: {global_batch_size}")
    print(f"local_bs: {local_bs}")
    print(f"gpus_per_node: {gpus_per_node}")
    print(f"nodes_nr: {nodes_nr}")

    return gpus_per_node, nodes_nr, local_bs


def get_total_compute(model_str, train_num_samples, precision) -> int:
    model = create_default_model(model_str)
    if torch.cuda.is_available():
        model = model.cuda()
    try:
        return get_model_flops(
            model,
            train_num_samples,
            dtype=precision_to_dtype(precision),
        )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


def generate_scripts_per_config(config_id, scan_space, root_dir, venv_path, wandb_project_name, workspace_dir, cluster,
                                command_dir=None, config_id_to_train_num_samples=None, partition="", seed=1):
    precision = "amp"

    logs_dir, scripts_dir, evaluation_scripts_path, downstream_evaluation_scripts_path, model_cfg_path = create_directories(
        root_dir, config_id)
    
    # Convert to absolute path to ensure subprocess finds correct location
    logs_dir = str(Path(logs_dir).resolve())

    build_model_cfg(out_path=model_cfg_path, **scan_space)
    model_cfg = get_local_model_str(model_cfg_path)

    gpus_per_node, nodes_nr, local_bs = compute_batch_and_resources(
        model_cfg=model_cfg,
        hp_scale_space=scan_space,
        precision=precision,
        max_gpus_per_node=4,
    )

    hp_space_train_num_samples = get_hp_space_train_num_samples_from_scan_space(scan_space)
    job_name = f"train_config{config_id}"

    # If command_dir not specified, use workspace_dir (code in workspace)
    if not command_dir:
        command_dir = workspace_dir

    slurm = SlurmManager(job_name=job_name, partition=partition, nodes=nodes_nr,
                         gpus_per_node=gpus_per_node,
                         root_dir=root_dir, workspace_dir=workspace_dir, command_dir=command_dir, cluster=cluster,
                         venv_path=venv_path)

    if config_id_to_train_num_samples is None:
        config_id_to_train_num_samples = {}

    generate_train_script(
        config_id=config_id,
        model_cfg=model_cfg,
        local_bs=local_bs,
        hp_space_train_num_samples=hp_space_train_num_samples,
        seed=seed,
        slurmManager=slurm,
        logs_dir=logs_dir,
        train_scripts_dir=scripts_dir,
        wandb_project_name=wandb_project_name,
        cluster=cluster,
        config_id_to_train_num_samples=config_id_to_train_num_samples,
    )

    # generate_eval_script(
    #     config_id=config_id,
    #     model_cfg=model_cfg,
    #     seed=seed,
    #     slurmManager=slurm,
    #     logs_dir=logs_dir,
    #     eval_scripts_dir=evaluation_scripts_path,
    #     cluster=cluster
    # )

    # generate_downstream_eval_script(
    #     config_id=config_id,
    #     model_cfg=model_cfg,
    #     seed=seed,
    #     slurmManager=slurm,
    #     logs_dir=logs_dir,
    #     downstream_eval_scripts_dir=downstream_evaluation_scripts_path,
    #     cluster=cluster
    # )


def generate_train_script(model_cfg, local_bs, config_id, logs_dir, train_scripts_dir,
                          hp_space_train_num_samples, seed,
                          slurmManager, wandb_project_name, cluster, config_id_to_train_num_samples):
    from config import TrainConfig

    config = TrainConfig(
        model=model_cfg,
        seed=seed,
        # epochs=1,  # will be overwritten
        name=slurmManager.job_name,
        batch_size=local_bs,
        sync_neps=False,
        pipeline_id=config_id,
        logs=logs_dir,
        config_id_to_train_num_samples=config_id_to_train_num_samples,
        **hp_space_train_num_samples,
    )
    from slurm import build_cmd
    cmd_parts = build_cmd(config, wandb_project_name, cluster)

    # Build argument list for direct function call (no subprocess)
    args_list = [arg for part in cmd_parts for arg in shlex.split(part)]
    
    print(f"[Config {config_id}] Running training directly in same process")
    print(f"[Config {config_id}] Arguments: {args_list[:5]}... ({len(args_list)} total)")

    _prepare_env_for_in_process_training()

    # Import and call clip_train.main directly
    from clip_train.main import main as clip_train_main

    # Call main with the arguments - this will raise exceptions directly
    try:
        clip_train_main(args_list)
    except Exception as e:
        print(f"[Config {config_id}] Training failed with exception:")
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _prepare_env_for_in_process_training():
    # The generator itself runs inside a Slurm allocation. Direct in-process
    # training should behave like a single-process training invocation, not
    # inherit launcher rank metadata from the generator job.
    for key in (
            "LOCAL_RANK",
            "RANK",
            "WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "SLURM_PROCID",
            "SLURM_LOCALID",
            "SLURM_NTASKS",
            "PMI_RANK",
            "PMI_SIZE",
            "MPI_LOCALRANKID",
            "OMPI_COMM_WORLD_LOCAL_RANK",
            "OMPI_COMM_WORLD_RANK",
            "OMPI_COMM_WORLD_SIZE",
    ):
        os.environ.pop(key, None)

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices and "," in visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices.split(",", 1)[0]

    


def generate_eval_script(
        config_id,
        model_cfg,
        seed,
        slurmManager,
        logs_dir,
        eval_scripts_dir,
        cluster
):
    run_name = f"train_config{config_id}"

    eval_cfg = EvalConfig(
        model=model_cfg,
        seed=seed,
        logs=str(logs_dir),
        name=run_name,
        pipeline_id=config_id,
    )
    job_name = f"eval_config{config_id}"

    eval_script_path = Path(eval_scripts_dir + "/" + f"{job_name}.sh")
    slurmManager.build_evaluation_script(eval_cfg=eval_cfg,cluster=cluster, job_name=job_name, save_to=eval_script_path)
    return eval_script_path


def generate_downstream_eval_script(
        config_id,
        model_cfg,
        seed,
        slurmManager,
        logs_dir,
        downstream_eval_scripts_dir,
        cluster
):
    run_name = f"train_config{config_id}"
    downstream_eval_job_name = f"downstream_eval_config{config_id}"
    downstream_eval_script_path = Path(downstream_eval_scripts_dir + "/" + downstream_eval_job_name + ".sh")
    slurmManager.build_downstream_evaluation_script(job_name=downstream_eval_job_name,
                                                    root=logs_dir + "/" + run_name,
                                                    save_to=downstream_eval_script_path, model=model_cfg, seed=seed,
                                                    cluster=cluster,
                                                    config_id=config_id)

def run(root_dir, seed, bias_scale_sampling, alpha, venv_path, wandb_project_name, workspace_dir, cluster, command_dir=None, partition=""):
    full_df, result = neps.status(root_dir)
    rng = np.random.default_rng(seed)
    nr_configs = len(full_df)
    print(f"Total configs to process: {nr_configs}")
    for i, (config_id, row) in enumerate(full_df.iterrows()):
        scan_space = build_scan_space(row, bias_scale_sampling, alpha, rng)
        generate_scripts_per_config(config_id, scan_space, root_dir, venv_path, wandb_project_name,
                                    workspace_dir,
                                    cluster, command_dir=command_dir, partition=partition, seed=seed)

if __name__ == "__main__":
    args = iterator_params.parse_args(sys.argv[1:])
    run(
        root_dir=args.root_dir,
        seed=args.seed,
        bias_scale_sampling=args.bias_scale_sampling,
        alpha=args.alpha,
        venv_path=args.venv_path,
        wandb_project_name=args.wandb_project_name,
        workspace_dir=args.workspace_dir,
        cluster=args.cluster,
        command_dir=args.command_dir,
        partition=args.partition,
    )


    
