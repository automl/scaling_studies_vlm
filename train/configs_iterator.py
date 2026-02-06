import gc
import math
import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import neps
import torch

import iterator_params
from clip_train.profiler import get_model_flops
from probe_max_bs import probe_max_bs
from slurm import SlurmManager, build_srun_cmd, build_eval_cmd
from utils import query_bs, build_model_cfg, get_local_model_str, precision_to_dtype, create_default_model, \
    sample_warped_log_uniform


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
    SCALE_KEYS = ["train_num_samples", "vision_width", "text_width", "text_heads"]

    hp_only = {k: v for k, v in scan_space.items() if k in HP_KEYS}
    scale_only = {k: v for k, v in scan_space.items() if k in SCALE_KEYS}

    if scale_only.get("train_num_samples") is not None:
        hp_only["train_num_samples"] = scale_only["train_num_samples"]
    return hp_only


def build_scan_space(row, bias_scale_sampling, alpha) -> Dict[str, Any]:
    scan_space = {
        col.replace("config.", ""): val
        for col, val in row.items()
        if col.startswith("config.")
    }
    return normalize_scan_space(scan_space, bias_scale_sampling, alpha)


def normalize_scan_space(scan_space: Dict[str, Any], bias_scale_sampling: bool, alpha: float) -> Dict[str, Any]:
    """Normalize scan space values by applying scaling factors and transformations."""
    if scan_space.get("text_width") is not None:
        scan_space["text_heads"] = int(scan_space["text_width"])
        scan_space["text_width"] = int(scan_space["text_width"]) * 64


    if scan_space.get("vision_width") is not None:
        scan_space["vision_width"] = int(scan_space["vision_width"]) * 64

    if scan_space.get("train_num_samples") is not None:
        if bias_scale_sampling:
            scan_space["train_num_samples"] = sample_warped_log_uniform(scan_space["train_num_samples"], alpha)
        scan_space["train_num_samples"] = int(float(scan_space["train_num_samples"]) * 1_000_000)

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
    total_compute = get_total_compute(model_cfg, hp_scale_space.get("train_num_samples"), precision)
    global_batch_size = query_bs(total_compute)
    gpus_per_node, nodes_nr, local_bs = infer_hardware_setup_and_local_bs(
        model_name=model_cfg,
        precision=precision,
        global_batch_size=global_batch_size,
        max_gpus_per_node=max_gpus_per_node,
    )

    print(f"total_compute: {total_compute}")
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


def generate_scripts_per_config(config_id, row, root_dir, args, wandb_project_name, workspace_dir, bias_scale_sampling,
                                alpha, cluster):
    precision = "amp"
    scan_space = build_scan_space(row, bias_scale_sampling, alpha)

    logs_dir, scripts_dir, evaluation_scripts_path, downstream_evaluation_scripts_path, model_cfg_path = create_directories(
        root_dir, config_id)

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

    slurm = SlurmManager(job_name=job_name, partition=args.partition, nodes=nodes_nr,
                         gpus_per_node=gpus_per_node,
                         root_dir=root_dir, workspace_dir=workspace_dir)

    generate_train_script(
        config_id=config_id,
        model_cfg=model_cfg,
        local_bs=local_bs,
        hp_space_train_num_samples=hp_space_train_num_samples,
        seed=args.seed,
        slurmManager=slurm,
        logs_dir=logs_dir,
        train_scripts_dir=scripts_dir,
        wandb_project_name=wandb_project_name,
        cluster=cluster
    )

    generate_eval_script(
        config_id=config_id,
        model_cfg=model_cfg,
        seed=args.seed,
        slurmManager=slurm,
        logs_dir=logs_dir,
        eval_scripts_dir=evaluation_scripts_path,
        cluster=cluster
    )

    generate_downstream_eval_script(
        config_id=config_id,
        model_cfg=model_cfg,
        seed=args.seed,
        slurmManager=slurm,
        logs_dir=logs_dir,
        downstream_eval_scripts_dir=downstream_evaluation_scripts_path,
    )


def generate_train_script(model_cfg, local_bs, config_id, logs_dir, train_scripts_dir,
                          hp_space_train_num_samples, seed,
                          slurmManager, wandb_project_name, cluster):
    from config import TrainConfig

    config = TrainConfig(
        model=model_cfg,
        seed=seed,
        epochs=1,
        name=slurmManager.job_name,
        batch_size=local_bs,
        sync_neps=False,
        pipeline_id=config_id,
        logs=logs_dir,
        **hp_space_train_num_samples,
    )

    cmd = build_srun_cmd(config, wandb_project_name, cluster)

    script_name = slurmManager.job_name + ".sh"
    script_path = Path(train_scripts_dir + "/" + script_name)
    slurmManager.build_script(cmd, save_to=script_path)


def generate_eval_script(
        config_id,
        model_cfg,
        seed,
        slurmManager,
        logs_dir,
        eval_scripts_dir,
        cluster
):
    from config import EvalConfig

    run_name = f"train_config{config_id}"

    eval_cfg = EvalConfig(
        model=model_cfg,
        seed=seed,
        logs=str(logs_dir),
        name=run_name,
        pipeline_id=config_id,
    )
    eval_cmd = build_eval_cmd(eval_cfg, cluster)
    job_name = f"eval_config{config_id}"

    eval_script_path = Path(eval_scripts_dir + "/" + f"{job_name}.sh")
    slurmManager.build_evaluation_script(cmd=eval_cmd, job_name=job_name, save_to=eval_script_path)
    return eval_script_path


def generate_downstream_eval_script(
        config_id,
        model_cfg,
        seed,
        slurmManager,
        logs_dir,
        downstream_eval_scripts_dir,
):
    run_name = f"train_config{config_id}"
    downstream_eval_job_name = f"downstream_eval_config{config_id}"
    downstream_eval_script_path = Path(downstream_eval_scripts_dir + "/" + downstream_eval_job_name + ".sh")
    slurmManager.build_downstream_evaluation_script(job_name=downstream_eval_job_name,
                                                    root=logs_dir + "/" + run_name,
                                                    save_to=downstream_eval_script_path, model=model_cfg, seed=seed,
                                                    config_id=config_id)


if __name__ == "__main__":
    args = iterator_params.parse_args(sys.argv[1:])

    root_dir = args.root_dir
    full_df, result = neps.status(root_dir)
    nr_configs = len(full_df)
    for i, (config_id, row) in enumerate(full_df.iterrows()):
        if i % args.num_workers != args.worker_id:
            continue
        print(
            f"[worker {args.worker_id}/{args.num_workers}] "
            f"processing index {i} | config_id={config_id} | "
            f"total_configs={nr_configs}"
        )
        generate_scripts_per_config(config_id, row, root_dir, args, args.wandb_project_name, args.workspace_dir,
                                    args.bias_scale_sampling, args.alpha, args.cluster)
