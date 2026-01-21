import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any

import neps

from train import iterator_params
from train.config import TrainConfig, EvalConfig
from train.slurm import SlurmManager, build_srun_cmd, BatchSizeSlurmManager, build_eval_cmd, eval_array_job_cmd, \
    FLOPsSlurmManager
from train.spaces import __HP_SPACE, __SCALE_SPACE
from train.utils import query_bs, build_model_cfg, get_local_model_str


def infer_local_bs(partition: str,
                   model_name: str,
                   root_dir: str,
                   trial_id: int
                   ) -> int:
    job_name = f"trial{trial_id}_max_local_bs_probe"

    scripts_dir = Path(root_dir + "/bs_probing_scripts")
    scripts_dir.mkdir(parents=True, exist_ok=True)

    current_script_path = scripts_dir / f"trial{trial_id}.sh"

    batch_size_slurm_manager = BatchSizeSlurmManager(partition=partition, job_name=job_name)

    if "train/" in model_name:
        model_name = model_name.replace("train/", "")

    cmd = (
        f'python probe_max_bs.py '
        f'--partition "{partition}" '
        f'--model "{model_name}" '
        f'--save_to "{scripts_dir}" '
        f'--trial_id {trial_id}'
    )

    batch_size_slurm_manager.build_script(cmd=cmd, save_to=current_script_path)
    batch_size_slurm_manager.submit_without_build(script_path=current_script_path)

    bs_txt_file = scripts_dir / f"trial_{trial_id}.txt"
    if not bs_txt_file.exists():
        raise FileNotFoundError(
            f"Batch size output file was not found: {bs_txt_file}"
        )

    raw = bs_txt_file.read_text().strip()

    try:
        best_bs = int(raw)
    except ValueError:
        best_bs = int(float(raw))

    return best_bs


def infer_hardware_setup_and_local_bs(root_dir,
                                      trial_id,
                                      model_name: str,
                                      partition: str,
                                      global_batch_size: int,
                                      max_gpus_per_node: int = 4,
                                      ):
    local_bs = infer_local_bs(partition=partition, model_name=model_name, root_dir=root_dir, trial_id=trial_id)

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


def get_hp_space_train_num_samples_from_joint_space(hp_scale_space):
    HP_KEYS = set(__HP_SPACE.keys())
    SCALE_KEYS = set(__SCALE_SPACE.keys())

    hp_only = {k: v for k, v in hp_scale_space.items() if k in HP_KEYS}
    scale_only = {k: v for k, v in hp_scale_space.items() if k in SCALE_KEYS}

    if scale_only.get("train_num_samples") is not None:
        hp_only["train_num_samples"] = scale_only["train_num_samples"]
    return hp_only


def build_hp_scale_space(row) -> Dict[str, Any]:
    hp_scale_space = {
        col.replace("config.", ""): val
        for col, val in row.items()
        if col.startswith("config.")
    }

    if hp_scale_space.get("text_width") is not None:
        hp_scale_space["text_width"] = int(hp_scale_space["text_width"]) * 8

    if hp_scale_space.get("vision_width") is not None:
        hp_scale_space["vision_width"] = int(hp_scale_space["vision_width"]) * 64

    if hp_scale_space.get("train_num_samples") is not None:
        hp_scale_space["train_num_samples"] = int(float(hp_scale_space["train_num_samples"]) * 1_000_000)

    return hp_scale_space


def generate_scripts_per_config(trial_id, row, root_dir, args, nr_configs):
    hp_scale_space = build_hp_scale_space(row)

    logs_dir = os.path.join(root_dir, "individual_training_logs")
    scripts_dir = os.path.join(root_dir, "train_scripts")
    evaluation_scripts_path = os.path.join(root_dir, "evaluation_scripts")
    downstream_evaluation_scripts_path = os.path.join(root_dir, "downstream_evaluation_scripts")
    model_cfg_path = os.path.join(root_dir, "model_cfg", f"config_{trial_id}")

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(scripts_dir, exist_ok=True)
    os.makedirs(evaluation_scripts_path, exist_ok=True)
    os.makedirs(downstream_evaluation_scripts_path, exist_ok=True)

    precision = "amp"

    build_model_cfg(out_path=model_cfg_path, **hp_scale_space)
    model_cfg = get_local_model_str(model_cfg_path)

    total_compute = get_total_compute(model_cfg, hp_scale_space.get("train_num_samples"), precision, root_dir, trial_id)
    print(f"total_compute: {total_compute}")
    global_batch_size = query_bs(total_compute)
    print(f"global_batch_size: {global_batch_size}")
    gpus_per_node, nodes_nr, local_bs = infer_hardware_setup_and_local_bs(model_name=model_cfg,
                                                                          partition=args.partition,
                                                                          global_batch_size=global_batch_size,
                                                                          root_dir=root_dir, trial_id=trial_id)

    print(f"local_bs: {local_bs}")
    print(f"gpus_per_node: {gpus_per_node}")
    print(f"nodes_nr: {nodes_nr}")

    job_name = f"train_trial{trial_id}"

    hp_space = get_hp_space_train_num_samples_from_joint_space(hp_scale_space)

    config = TrainConfig(
        model=model_cfg,
        seed=args.seed,
        epochs=1,
        name=job_name,
        batch_size=local_bs,
        sync_neps=False,
        pipeline_id=trial_id,
        logs=logs_dir,
        **hp_space,
    )

    cmd = build_srun_cmd(config)

    slurm = SlurmManager(job_name=job_name, partition=args.partition, nodes=nodes_nr,
                         gpus_per_node=gpus_per_node,
                         root_dir=root_dir)
    script_name = job_name + ".sh"
    script_path = Path(scripts_dir + "/" + script_name)
    slurm.build_script(cmd, save_to=script_path)

    eval_config = EvalConfig(model=model_cfg, seed=args.seed, logs=str(logs_dir), name=job_name)
    eval_cmd = build_eval_cmd(eval_config)
    eval_job_name = f"eval_trial{trial_id}"

    eval_script_name = eval_job_name + ".sh"
    eval_script_path = Path(evaluation_scripts_path + "/" + eval_script_name)
    slurm.build_evaluation_script(cmd=eval_cmd, job_name=eval_job_name, save_to=eval_script_path)

    array_job_eval_script_name = "array_job.sh"
    array_job_eval_script_path = Path(evaluation_scripts_path + "/" + array_job_eval_script_name)
    if not array_job_eval_script_path.exists():
        array_job_cmd = eval_array_job_cmd(model_cfg, args.seed, logs_dir)
        slurm.build_array_job_for_evaluation(cmd=array_job_cmd, nr_total_configs=nr_configs,
                                             save_to=array_job_eval_script_path)

    downstream_eval_job_name = f"downstream_eval_trial{trial_id}"
    downstream_eval_script_path = Path(downstream_evaluation_scripts_path + "/" + downstream_eval_job_name + ".sh")
    slurm.build_downstream_evaluation_script(job_name=downstream_eval_job_name, root=logs_dir + "/" + job_name,
                                             save_to=downstream_eval_script_path, model=model_cfg, seed=args.seed)

    array_job_downstream_eval_script_path = Path(downstream_evaluation_scripts_path + "/" + array_job_eval_script_name)
    if not array_job_downstream_eval_script_path.exists():
        slurm.build_array_job_for_downstream_evaluation(root=logs_dir, nr_total_configs=nr_configs,
                                                        save_to=array_job_downstream_eval_script_path)
    if int(trial_id) == nr_configs:
        bundle_train_scripts_in_array_jobs(root_dir=root_dir, logs_dir=logs_dir, train_scripts_dir=Path(scripts_dir),
                                           seed=args.seed)


def get_total_compute(model_cfg, train_num_samples, precision, root_dir, trial_id) -> int:
    flops_scripts_path = Path(root_dir + "/" + "flops_scripts")
    os.makedirs(flops_scripts_path, exist_ok=True)
    current_script_path = flops_scripts_path / f"trial{trial_id}.sh"
    job_name = f"trial{trial_id}_total_compute"

    cmd_parts = [
        f'--model {model_cfg}',
        f'--train_num_samples {str(train_num_samples)}',
        f'--precision {precision}',
        f'--root_dir {str(flops_scripts_path)}',
        f'--trial_id {trial_id}',
    ]

    cmd = (
            "python flops_count.py \\\n    "
            + " \\\n    ".join(cmd_parts)
    )

    flops_slurm_manager = FLOPsSlurmManager(partition=args.partition, job_name=job_name)
    flops_slurm_manager.build_script(cmd, save_to=current_script_path)
    flops_slurm_manager.submit_without_build(script_path=current_script_path)

    flops_txt_path = flops_scripts_path / f"trial_{trial_id}.txt"
    if not flops_txt_path.exists():
        raise FileNotFoundError(
            f"FLOPs output file was not created: {flops_txt_path}\n"
            f"Slurm script: {current_script_path}"
        )

    raw = flops_txt_path.read_text().strip()

    try:
        total_compute = int(raw)
    except ValueError:
        total_compute = int(float(raw))

    return total_compute


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

    trial_info = {}

    for sh in train_scripts_dir.glob("train_trial*.sh"):
        m = re.search(r"train_trial(\d+)\.sh$", sh.name)
        if not m:
            continue
        trial_id = int(m.group(1))

        nodes = gpus = part = None
        samples = bs = None
        lr = wd = warmup = None

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

        trial_info[trial_id] = {
            "nodes": nodes,
            "gpus": gpus,
            "partition": part,
            "train_num_samples": samples,
            "batch_size": bs,
            "lr": lr,
            "wd": wd,
            "warmup_fraction": warmup,
        }
    groups = group_trials(trial_info)
    group_to_txt = write_group_configs(trial_info, groups, train_scripts_dir)
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


def group_trials(trial_info: dict[int, dict]) -> dict[tuple, list[int]]:
    groups = defaultdict(list)
    for tid, info in trial_info.items():
        key = (info["partition"], info["batch_size"], info["gpus"], info["nodes"])
        groups[key].append(tid)
    return {k: sorted(v) for k, v in groups.items()}


def write_group_configs(
        trial_info: dict[int, dict],
        groups: dict[tuple, list[int]],
        out_dir: str | Path,
) -> dict[tuple, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_paths = {}
    for (part, bs, gpus, nodes), trial_ids in groups.items():
        safe_part = "".join(c if c.isalnum() or c in "._" else "_" for c in str(part))
        p = out_dir / f"train_p{safe_part}_bs{bs}_g{gpus}_n{nodes}.txt"

        with p.open("w", encoding="utf-8") as f:
            for tid in trial_ids:
                info = trial_info[tid]
                f.write(
                    f"{tid} "
                    f"{info['train_num_samples']} "
                    f"{info['lr']} "
                    f"{info['wd']} "
                    f"{info['warmup_fraction']}\n"
                )
        out_paths[(part, bs, gpus, nodes)] = p

    return out_paths


if __name__ == "__main__":
    args = iterator_params.parse_args(sys.argv[1:])

    root_dir = args.root_dir
    full_df, result = neps.status(root_dir)
    nr_configs = len(full_df)
    for trial_id, row in full_df.iterrows():
        try:
            generate_scripts_per_config(trial_id, row, root_dir, args, nr_configs)
        except Exception as e:
            print(e)
