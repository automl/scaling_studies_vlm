import re
import sys
from collections import defaultdict
from pathlib import Path

import neps

import iterator_params
from slurm import eval_array_job_cmd, SlurmManager


def create_directories(root_dir: str):
    root = Path(root_dir)
    logs_dir = root / "individual_training_logs"
    evaluation_scripts_dir = root / "evaluation_scripts"
    downstream_evaluation_scripts_dir = root / "downstream_evaluation_scripts"
    models_cfg_dir = root / "model_cfg"
    train_scripts_dir = root / "train_scripts"

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
):
    out_path = evaluation_scripts_dir / "array_job.sh"
    if out_path.exists():
        return

    array_cmd = eval_array_job_cmd(seed, str(logs_dir))
    slurm.build_array_job_for_evaluation(
        cmd=array_cmd,
        models_cfg_dir=models_cfg_dir,
        nr_total_configs=nr_configs,
        save_to=out_path,
    )


def generate_downstream_eval_array_script(
        *,
        logs_dir: Path,
        downstream_evaluation_scripts_dir: Path,
        models_cfg_dir: Path,
        nr_configs: int,
        seed: int,
        slurm: SlurmManager,
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
    )


def generate_array_scripts(root_dir: str, seed: int, partition: str, nr_configs: int, workspace_dir: str):
    logs_dir, evaluation_scripts_dir, downstream_evaluation_scripts_dir, models_cfg_dir, train_scripts_dir = create_directories(
        root_dir)

    slurm = SlurmManager(
        job_name="eval_arrays",
        partition=partition,
        nodes=1,
        gpus_per_node=1,
        root_dir=root_dir,
        workspace_dir=workspace_dir
    )

    generate_eval_array_script(
        logs_dir=logs_dir,
        evaluation_scripts_dir=evaluation_scripts_dir,
        nr_configs=nr_configs,
        models_cfg_dir=models_cfg_dir,
        seed=seed,
        slurm=slurm,
    )

    generate_downstream_eval_array_script(
        logs_dir=logs_dir,
        downstream_evaluation_scripts_dir=downstream_evaluation_scripts_dir,
        models_cfg_dir=models_cfg_dir,
        nr_configs=nr_configs,
        seed=seed,
        slurm=slurm,
    )

    bundle_train_scripts_in_array_jobs(root_dir=root_dir, logs_dir=logs_dir, train_scripts_dir=train_scripts_dir,
                                       seed=args.seed, workspace_dir=workspace_dir)


def bundle_train_scripts_in_array_jobs(root_dir, logs_dir, train_scripts_dir,workspace_dir, seed: int):
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
        slurmManager = SlurmManager(partition=part, gpus_per_node=gpus, nodes=nodes, job_name=job_name, workspace_dir=workspace_dir)

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
    full_df, _ = neps.status(root_dir)
    nr_configs = len(full_df)

    generate_array_scripts(
        root_dir=root_dir,
        seed=args.seed,
        partition=args.partition,
        nr_configs=nr_configs,
        workspace_dir=args.workspace_dir
    )
