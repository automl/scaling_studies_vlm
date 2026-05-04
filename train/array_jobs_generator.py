"""
Group the scripts by their batch_size, partition, gpus and nodes and create array jobs for each group.
Usage:
python -m array_jobs_generator --root_dir results/optimizer=bo_space_expansion_micro/seed=42  --seed 42 --partition booster --workspace_dir /p/data1/datasets/scan/openclip --cluster juwels --command_dir /p/project1/projectnucleus/alipourgougeh1/scaling_studies_vlm/train  --wandb_project_name bo_space_expansion_micro --venv_path /p/project1/projectnucleus/alipourgougeh1/scaling_studies_vlm/venv/bin/activate
"""
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import iterator_params
from config import EvalConfig
from slurm import SlurmManager


def create_directories(neps_dir: Path):
    logs_dir = neps_dir / "individual_training_logs"
    evaluation_scripts_dir = neps_dir / "evaluation_scripts"
    downstream_evaluation_scripts_dir = neps_dir / "downstream_evaluation_scripts"
    models_cfg_dir = neps_dir / "model_cfg"
    train_scripts_dir = neps_dir / "train_scripts"

    evaluation_scripts_dir.mkdir(parents=True, exist_ok=True)
    downstream_evaluation_scripts_dir.mkdir(parents=True, exist_ok=True)
    models_cfg_dir.mkdir(parents=True, exist_ok=True)

    return logs_dir, evaluation_scripts_dir, downstream_evaluation_scripts_dir, models_cfg_dir, train_scripts_dir


def generate_eval_array_script(
        *,
        models_cfg_dir: Path,
        logs_dir: Path,
        evaluation_scripts_dir: Path,
        nr_configs: int,
        seed: int,
        slurm: SlurmManager,
        cluster: str
):
    out_path = evaluation_scripts_dir / "array_job.sh"
    if out_path.exists():
        return

    eval_cfg = EvalConfig(model=str(models_cfg_dir), seed=seed, name="", logs=str(logs_dir), pipeline_id=1)

    slurm.build_array_job_for_evaluation(eval_cfg=eval_cfg, cluster=cluster,
                                         nr_total_configs=nr_configs,
                                         save_to=out_path)


def generate_downstream_eval_array_script(
        *,
        logs_dir: Path,
        downstream_evaluation_scripts_dir: Path,
        models_cfg_dir: Path,
        nr_configs: int,
        seed: int,
        slurm: SlurmManager,
        cluster: str
):
    out_path = downstream_evaluation_scripts_dir / "array_job.sh"
    if out_path.exists():
        return

    slurm.build_array_job_for_downstream_evaluation(
        root=str(logs_dir),
        models_cfg_dir=models_cfg_dir,
        nr_total_configs=nr_configs,
        seed=seed,
        save_to=out_path,
        cluster=cluster
    )


def generate_array_scripts(neps_dir: Path, root_dir: str, seed: int, partition: str, nr_configs: int,
                           workspace_dir: str, cluster: str,
                           wandb_project_name: str, command_dir: str, venv_path: str | None = None):
    logs_dir, evaluation_scripts_dir, downstream_evaluation_scripts_dir, models_cfg_dir, train_scripts_dir = create_directories(
        neps_dir)

    slurm = SlurmManager(
        job_name="eval_arrays",
        partition=partition,
        nodes=1,
        gpus_per_node=1,
        root_dir=root_dir,
        workspace_dir=workspace_dir,
        cluster=cluster,
        command_dir=command_dir,
        venv_path=venv_path,
    )

    generate_eval_array_script(
        logs_dir=logs_dir,
        evaluation_scripts_dir=evaluation_scripts_dir,
        nr_configs=nr_configs,
        models_cfg_dir=models_cfg_dir,
        seed=seed,
        slurm=slurm,
        cluster=cluster,
    )

    generate_downstream_eval_array_script(
        logs_dir=logs_dir,
        downstream_evaluation_scripts_dir=downstream_evaluation_scripts_dir,
        models_cfg_dir=models_cfg_dir,
        nr_configs=nr_configs,
        seed=seed,
        slurm=slurm,
        cluster=cluster
    )

    bundle_train_scripts_in_array_jobs(root_dir=root_dir, logs_dir=logs_dir, train_scripts_dir=train_scripts_dir,
                                       seed=seed, workspace_dir=workspace_dir, cluster=cluster,
                                       wandb_project_name=wandb_project_name, command_dir=command_dir,
                                       venv_path=venv_path)


def bundle_train_scripts_in_array_jobs(root_dir, logs_dir, train_scripts_dir, workspace_dir, seed: int, cluster: str,
                                       wandb_project_name: str, command_dir: str, venv_path: str | None = None):
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
    beta1_re = re.compile(r"--beta1\s+([0-9eE\.\+\-]+)")
    beta2_re = re.compile(r"--beta2\s+([0-9eE\.\+\-]+)")
    eps_re = re.compile(r"--eps\s+([0-9eE\.\+\-]+)")
    epochs_re = re.compile(r"--epochs\s+(\d+)")

    config_info = {}

    for sh in train_scripts_dir.glob("train_config*.sh"):
        m = re.search(r"train_config(\d+)\.sh$", sh.name)
        if not m:
            continue
        config_id = int(m.group(1))

        if config_id > 100:
            continue

        nodes = gpus = part = None
        samples = bs = None
        lr = wd = warmup = beta2 = beta1 = eps = None
        epochs = None

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
            if beta1 is None and (m2 := beta1_re.search(line)):
                beta1 = float(m2.group(1))
            if beta2 is None and (m2 := beta2_re.search(line)):
                beta2 = float(m2.group(1))
            if eps is None and (m2 := eps_re.search(line)):
                eps = float(m2.group(1))
            if epochs is None and (m2 := epochs_re.search(line)):
                epochs = int(m2.group(1))

        config_info[config_id] = {
            "nodes": nodes,
            "gpus": gpus,
            "partition": part,
            "train_num_samples": samples,
            "batch_size": bs,
            "lr": lr,
            "epochs": epochs,
            "wd": wd,
            "warmup_fraction": warmup,
            "beta1": beta1,
            "beta2": beta2,
            "eps": eps,
        }
    groups = group_configs(config_info)
    group_to_txt = write_group_configs(config_info, groups, train_scripts_dir)
    for (part, bs, gpus, nodes), config_txt in group_to_txt.items():
        job_name = f"train_p{part}_bs{bs}_g{gpus}_n{nodes}"
        slurmManager = SlurmManager(partition=part, gpus_per_node=gpus, nodes=nodes, job_name=job_name,
                                    workspace_dir=workspace_dir, cluster=cluster, command_dir=command_dir,
                                    venv_path=venv_path)
        save_to = Path(train_scripts_dir) / f"{slurmManager.job_name}.sh"
        config_txt = str(config_txt).replace("\\", "/")

        if gpus == 1 and nodes == 1 and cluster == "juwels":
            num_lines = sum(1 for _ in Path(config_txt).open())
            pack = 4
            packed_array_size = math.ceil(num_lines / pack)

            slurmManager.build_array_script_for_single_gpu_runs(array_size=packed_array_size, seed=seed, batch_size=bs,
                                                                config_txt=config_txt, logs_root=logs_dir,
                                                                cluster=cluster, wandb_project_name=wandb_project_name,
                                                                save_to=save_to, model_root=root_dir)

        elif gpus == 2 and nodes == 1 and cluster == "juwels":
            num_lines = sum(1 for _ in Path(config_txt).open())
            pack = 2
            packed_array_size = math.ceil(num_lines / pack)

            slurmManager.build_array_script_for_two_gpu_runs(
                array_size=packed_array_size, seed=seed, batch_size=bs,
                config_txt=config_txt, logs_root=logs_dir, cluster=cluster,
                wandb_project_name=wandb_project_name, save_to=save_to, model_root=root_dir
            )
        else:
            array_size = sum(1 for _ in Path(config_txt).open())

            slurmManager.build_array_script(
                model_root=root_dir,
                logs_root=logs_dir,
                config_txt=config_txt,
                array_size=array_size,
                batch_size=bs,
                seed=seed,
                save_to=save_to,
                cluster=cluster,
                wandb_project_name=wandb_project_name
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

        sorted_ids = sorted(
            config_ids,
            key=lambda tid: config_info[tid]["train_num_samples"] * config_info[tid]["epochs"]
        )

        with p.open("w", encoding="utf-8") as f:
            for tid in sorted_ids:
                info = config_info[tid]
                f.write(
                    f"{tid} "
                    f"{info['train_num_samples']} "
                    f"{info['lr']} "
                    f"{info['wd']} "
                    f"{info['warmup_fraction']} "
                    f"{info['beta1']} "
                    f"{info['beta2']} "
                    f"{info['eps']} "
                    f"{info['epochs']}\n"
                )
        out_paths[(part, bs, gpus, nodes)] = p

    return out_paths


if __name__ == "__main__":
    args = iterator_params.parse_args(sys.argv[1:])

    root_dir = args.root_dir
    neps_dir = Path(args.workspace_dir + "/train/" + args.root_dir).resolve()
    # full_df, _ = neps.status(neps_dir)
    # nr_configs = len(full_df)

    generate_array_scripts(
        neps_dir=neps_dir,
        root_dir=root_dir,
        seed=args.seed,
        partition=args.partition,
        nr_configs=1000,
        workspace_dir=args.workspace_dir,
        cluster=args.cluster,
        command_dir=args.command_dir,
        wandb_project_name=args.wandb_project_name,
        venv_path=args.venv_path,
    )
