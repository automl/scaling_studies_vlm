import gc
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Tuple

import neps
import torch

from train import iterator_params
from train.clip_train.profiler import get_model_flops
from train.config import TrainConfig, EvalConfig
from train.probe_max_bs import probe_max_bs
from train.slurm import SlurmManager, build_srun_cmd, build_eval_cmd, eval_array_job_cmd
from train.spaces import __HP_SPACE, __SCALE_SPACE
from train.utils import query_bs, build_model_cfg, get_local_model_str, precision_to_dtype, create_default_model


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
    HP_KEYS = set(__HP_SPACE.keys())
    SCALE_KEYS = set(__SCALE_SPACE.keys())

    hp_only = {k: v for k, v in scan_space.items() if k in HP_KEYS}
    scale_only = {k: v for k, v in scan_space.items() if k in SCALE_KEYS}

    if scale_only.get("train_num_samples") is not None:
        hp_only["train_num_samples"] = scale_only["train_num_samples"]
    return hp_only


def build_scan_space(row) -> Dict[str, Any]:
    scan_space = {
        col.replace("config.", ""): val
        for col, val in row.items()
        if col.startswith("config.")
    }

    if scan_space.get("text_width") is not None:
        scan_space["text_width"] = int(scan_space["text_width"]) * 8

    if scan_space.get("vision_width") is not None:
        scan_space["vision_width"] = int(scan_space["vision_width"]) * 64

    if scan_space.get("train_num_samples") is not None:
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


def generate_scripts_per_config(config_id, row, root_dir, args, nr_configs):
    precision = "amp"
    scan_space = build_scan_space(row)

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
                         root_dir=root_dir)

    generate_train_script(
        config_id=config_id,
        model_cfg=model_cfg,
        local_bs=local_bs,
        hp_space_train_num_samples=hp_space_train_num_samples,
        seed=args.seed,
        slurmManager=slurm,
        logs_dir=logs_dir,
        train_scripts_dir=scripts_dir,
    )

    generate_eval_script(
        config_id=config_id,
        model_cfg=model_cfg,
        seed=args.seed,
        slurmManager=slurm,
        logs_dir=logs_dir,
        eval_scripts_dir=evaluation_scripts_path,
    )

    generate_downstream_eval_script(
        config_id=config_id,
        model_cfg=model_cfg,
        seed=args.seed,
        slurmManager=slurm,
        logs_dir=logs_dir,
        downstream_eval_scripts_dir=downstream_evaluation_scripts_path,
    )

    # TODO move array job creation to a separate part 
    array_job_eval_script_name = "array_job.sh"
    array_job_eval_script_path = Path(evaluation_scripts_path + "/" + array_job_eval_script_name)
    if not array_job_eval_script_path.exists():
        array_job_cmd = eval_array_job_cmd(model_cfg, args.seed, logs_dir)
        slurm.build_array_job_for_evaluation(cmd=array_job_cmd, nr_total_configs=nr_configs,
                                             save_to=array_job_eval_script_path)

    array_job_downstream_eval_script_path = Path(downstream_evaluation_scripts_path + "/" + array_job_eval_script_name)
    if not array_job_downstream_eval_script_path.exists():
        slurm.build_array_job_for_downstream_evaluation(root=logs_dir, nr_total_configs=nr_configs, seed=args.seed,
                                                        save_to=array_job_downstream_eval_script_path)
    if int(config_id) == nr_configs:
        bundle_train_scripts_in_array_jobs(root_dir=root_dir, logs_dir=logs_dir, train_scripts_dir=Path(scripts_dir),
                                           seed=args.seed)


def generate_train_script(model_cfg, local_bs, config_id, logs_dir, train_scripts_dir,
                          hp_space_train_num_samples, seed,
                          slurmManager):
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

    cmd = build_srun_cmd(config)

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
):
    eval_job_name = f"eval_config{config_id}"

    eval_cfg = EvalConfig(
        model=model_cfg,
        seed=seed,
        logs=str(logs_dir),
        name=eval_job_name,
    )
    eval_cmd = build_eval_cmd(eval_cfg)
    eval_script_path = Path(eval_scripts_dir + "/" + f"{eval_job_name}.sh")
    slurmManager.build_evaluation_script(cmd=eval_cmd, job_name=eval_job_name, save_to=eval_script_path)
    return eval_script_path


def generate_downstream_eval_script(
        config_id,
        model_cfg,
        seed,
        slurmManager,
        logs_dir,
        downstream_eval_scripts_dir,
):
    downstream_eval_job_name = f"downstream_eval_config{config_id}"
    downstream_eval_script_path = Path(downstream_eval_scripts_dir + "/" + downstream_eval_job_name + ".sh")
    slurmManager.build_downstream_evaluation_script(job_name=downstream_eval_job_name,
                                                    root=logs_dir + "/" + downstream_eval_job_name,
                                                    save_to=downstream_eval_script_path, model=model_cfg, seed=seed)


def bundle_train_scripts_in_array_jobs(root_dir, logs_dir, train_scripts_dir, seed: int):
    root_dir = str(root_dir)
    logs_dir = str(logs_dir)
    train_scripts_dir = Path(train_scripts_dir)

    if root_dir.startswith("train/"):
        root_dir = root_dir[len("train/"):]
    if logs_dir.startswith("train/"):
        logs_dir = logs_dir[len("train/"):]

    nodes_re = re.compile(r"^#SBATCH\s+--nodes=(\d+)")
    gpus_re = re.compile(r"^#SBATCH\s+--gres=gpu:(\d+)")
    part_re = re.compile(r"^#SBATCH\s+-(?:p)\s+(\S+)|^#SBATCH\s+--partition=(\S+)")

    samples_re = re.compile(r"--train-num-samples\s+([0-9]+(?:\.[0-9]+)?)")
    bs_re = re.compile(r"--batch-size\s+(\d+)")
    lr_re = re.compile(r"--lr\s+([0-9eE\.\+\-]+)")
    wd_re = re.compile(r"--wd\s+([0-9eE\.\+\-]+)")
    warmup_re = re.compile(r"--warmup-fraction\s+([0-9eE\.\+\-]+)")
    beta2_re = re.compile(r"--beta2\s+([0-9eE\.\+\-]+)")
    eps_re = re.compile(r"--eps\s+([0-9eE\.\+\-]+)")

    config_info = {}

    for sh in train_scripts_dir.glob("train_config*.sh"):
        m = re.search(r"train_config(\d+)\.sh$", sh.name)
        if not m:
            continue
        config_id = int(m.group(1))

        nodes = gpus = part = None
        samples = bs = None
        lr = wd = warmup = beta2 = eps = None

        for line in sh.read_text().splitlines():
            if nodes is None and (m2 := nodes_re.match(line)):
                nodes = int(m2.group(1))
            if gpus is None and (m2 := gpus_re.match(line)):
                gpus = int(m2.group(1))
            if part is None and (m2 := part_re.match(line)):
                part = m2.group(1) or m2.group(2)

            if samples is None and (m2 := samples_re.search(line)):
                samples = float(m2.group(1))
            if bs is None and (m2 := bs_re.search(line)):
                bs = int(m2.group(1))
            if lr is None and (m2 := lr_re.search(line)):
                lr = float(m2.group(1))
            if wd is None and (m2 := wd_re.search(line)):
                wd = float(m2.group(1))
            if warmup is None and (m2 := warmup_re.search(line)):
                warmup = float(m2.group(1))
            if beta2 is None and (m2 := beta2_re.search(line)):
                beta2 = float(m2.group(1))
            if eps is None and (m2 := eps_re.search(line)):
                eps = float(m2.group(1))

        config_info[config_id] = {
            "nodes": nodes,
            "gpus": gpus,
            "partition": part,
            "train_num_samples": samples,
            "batch_size": bs,
            "lr": lr,
            "wd": wd,
            "warmup_fraction": warmup,
            "beta2": beta2,
            "eps": eps,
        }
    groups = group_configs(config_info)
    group_to_txt = write_group_configs(config_info, groups, train_scripts_dir)
    for (part, bs, gpus, nodes), config_txt in group_to_txt.items():
        array_size = sum(1 for _ in config_txt.open())

        config_txt = str(config_txt).replace("\\", "/")

        if "train/" in config_txt:
            config_txt = config_txt.replace("train/", "")

        job_name = f"train_p{part}_bs{bs}_g{gpus}_n{nodes}"
        slurmManager = SlurmManager(partition=part, gpus_per_node=gpus, nodes=nodes, job_name=job_name)

        save_to = Path(train_scripts_dir) / f"{slurmManager.job_name}.sh"

        slurmManager.build_array_script(
            model_root=root_dir,
            logs_root=logs_dir,
            config_txt=config_txt,
            array_size=array_size,
            batch_size=bs,
            seed=seed,
            save_to=save_to,
        )


def group_configs(config_info: dict[int, dict]) -> dict[tuple, list[int]]:
    groups = defaultdict(list)
    for tid, info in config_info.items():
        key = (info["partition"], info["batch_size"], info["gpus"], info["nodes"])
        groups[key].append(tid)
    return {k: sorted(v) for k, v in groups.items()}


def write_group_configs(
        config_info: dict[int, dict],
        groups: dict[tuple, list[int]],
        out_dir: str | Path,
) -> dict[tuple, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_paths = {}
    for (part, bs, gpus, nodes), config_ids in groups.items():
        safe_part = "".join(c if c.isalnum() or c in "._" else "_" for c in str(part))
        p = out_dir / f"train_p{safe_part}_bs{bs}_g{gpus}_n{nodes}.txt"

        with p.open("w", encoding="utf-8") as f:
            for tid in config_ids:
                info = config_info[tid]
                f.write(
                    f"{tid} "
                    f"{info['train_num_samples']} "
                    f"{info['lr']} "
                    f"{info['wd']} "
                    f"{info['warmup_fraction']} "
                    f"{info['beta2']} "
                    f"{info['eps']}\n"
                )
        out_paths[(part, bs, gpus, nodes)] = p

    return out_paths


if __name__ == "__main__":
    args = iterator_params.parse_args(sys.argv[1:])

    root_dir = args.root_dir
    full_df, result = neps.status(root_dir)
    nr_configs = len(full_df)
    for config_id, row in full_df.iterrows():
        try:
            generate_scripts_per_config(config_id, row, root_dir, args, nr_configs)
        except Exception as e:
            print(e)
