import math
import os
import sys
from pathlib import Path

import neps
import torch

from train import iterator_params
from train.config import TrainConfig, EvalConfig
from train.open_clip_train.profiler import get_model_flops
from train.slurm import SlurmManager, build_srun_cmd, BatchSizeSlurmManager, build_eval_cmd
from train.utils import query_bs, precision_to_dtype, create_default_model, _cache_key, _read_cached_bs


def infer_local_bs(partition: str,
                   model_name: str,
                   ) -> int:
    cache_str = "bs_probe_cache"

    cache_dir = (Path.cwd().resolve() / "train" / cache_str)
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = _cache_key(partition, model_name)

    cached = _read_cached_bs(cache_dir, key)
    if cached is not None:
        print(f"Cache hit ({partition}, {model_name}) -> {cached}")
        return cached

    print(f"Cache miss ({partition}, {model_name}), submitting probe job")

    job_name = f"{partition}_{model_name}_max_local_bs_probe"

    scripts_dir = Path("bs_probing_scripts")
    scripts_dir.mkdir(parents=True, exist_ok=True)

    current_script_path = scripts_dir / f"{job_name}.sh"

    batch_size_slurm_manager = BatchSizeSlurmManager(partition=partition, job_name=job_name)

    cmd = (
        f'python probe_max_bs.py '
        f'--partition "{partition}" '
        f'--model "{model_name}" '
        f'--cache_dir "{str(cache_dir)}"'
    )

    batch_size_slurm_manager.build_script(cmd=cmd, save_to=current_script_path)
    batch_size_slurm_manager.submit_without_build(script_path=current_script_path)

    cached = _read_cached_bs(cache_dir, key)
    if cached is not None:
        return cached

    raise ValueError(
        f"Local batch size ({cached}) must not be None"
    )


def infer_hardware_setup_and_local_bs(model_name: str,
                                      partition: str,
                                      global_batch_size: int,
                                      max_gpus_per_node: int = 4):
    local_bs = infer_local_bs(partition=partition, model_name=model_name)

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


def submit_job_per_config(trial_id, row, root_dir, args):
    hps = {
        col.replace("config.", ""): val
        for col, val in row.items()
        if col.startswith("config.")
    }

    logs_dir = os.path.join(root_dir, "individual_training_logs")
    scripts_dir = os.path.join(root_dir, "slurm_scripts")
    evaluation_scripts_path = os.path.join(root_dir, "evaluation_scripts")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(scripts_dir, exist_ok=True)
    os.makedirs(evaluation_scripts_path, exist_ok=True)

    precision = "bf16"
    # TODO model to be created based on scale parameters
    model = create_default_model(args.model)
    if torch.cuda.is_available():
        model = model.cuda()
    total_compute = get_model_flops(model, args.train_num_samples, dtype=precision_to_dtype(precision))
    print(f"total_compute: {total_compute}")
    global_batch_size = query_bs(total_compute)
    print(f"global_batch_size: {global_batch_size}")
    gpus_per_node, nodes_nr, local_bs = infer_hardware_setup_and_local_bs(model_name=args.model,
                                                                          partition=args.partition,
                                                                          global_batch_size=global_batch_size)

    print(f"local_bs: {local_bs}")
    print(f"gpus_per_node: {gpus_per_node}")
    print(f"nodes_nr: {nodes_nr}")

    job_name = f"train_{args.model}_trial{trial_id}"

    config = TrainConfig(
        model=args.model,
        train_num_samples=args.train_num_samples,
        seed=args.seed,
        epochs=1,
        name=job_name,
        batch_size=local_bs,
        sync_neps=False,
        pipeline_id=trial_id,
        logs=logs_dir,
        **hps,
    )

    cmd = build_srun_cmd(config)

    slurm = SlurmManager(job_name=job_name, partition=args.partition, nodes=nodes_nr,
                         gpus_per_node=gpus_per_node,
                         root_dir=root_dir)
    script_name = job_name + ".sh"
    script_path = Path(scripts_dir + "/" + script_name)
    slurm.build_script(cmd, save_to=script_path)

    eval_config = EvalConfig(model=args.model, seed=args.seed, logs=str(logs_dir), name=job_name)
    eval_cmd = build_eval_cmd(eval_config)
    eval_job_name = f"eval_{args.model}_trial{trial_id}"

    script_name = eval_job_name + ".sh"
    eval_script_path = Path(evaluation_scripts_path + "/" + script_name)
    slurm.build_evaluation_script(cmd=eval_cmd, job_name=eval_job_name, save_to=eval_script_path)


if __name__ == "__main__":
    args = iterator_params.parse_args(sys.argv[1:])

    root_dir = args.root_dir

    full_df, result = neps.status(root_dir)
    for trial_id, row in full_df.iterrows():
        try:
            submit_job_per_config(trial_id, row, root_dir, args)
            # TODO post hoc evaluation

        except Exception as e:
            print(e)
