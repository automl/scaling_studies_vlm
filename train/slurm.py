import subprocess
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import TrainConfig, EvalConfig


class DatasetShards(str, Enum):
    LAION400M_TRAIN_DATA = "../pre_training_dataset/laion400m/train_data/{00000000..00012136}.tar",
    LAION400M_VALIDATION_DATA = "../pre_training_dataset/laion400m/validation_data/{00000000..00000015}.tar"
    LAION400M_TEST_DATA = "../pre_training_dataset/laion400m/test_data/{00000000..00000015}.tar"


eval_cmd_parts = [
    f'--val-data "{DatasetShards.LAION400M_VALIDATION_DATA.value}"',
    f'--test-data "{DatasetShards.LAION400M_TEST_DATA.value}"',
    "--dataset-type webdataset",
    "--precision fp32",
    "--resume latest",
    "--epochs 1",
]


def build_cmd(config: "TrainConfig", wandb_project_name) -> str:
    args = asdict(config)

    cmd_parts = [
        f'--train-data "{DatasetShards.LAION400M_TRAIN_DATA.value}"',
        f'--val-data "{DatasetShards.LAION400M_VALIDATION_DATA.value}"',
        f'--test-data "{DatasetShards.LAION400M_TEST_DATA.value}"',
        "--local-loss",
        "--gather-with-grad",
        "--dataset-type webdataset",
        "--precision amp",
        "--report-to wandb",
        f'--wandb-project-name "{wandb_project_name}"',
    ]

    for key, value in args.items():
        cli_key = key.replace("_", "-")

        if isinstance(value, bool):
            if value:
                cmd_parts.append(f"--{cli_key}")
            continue

        cmd_parts.append(f"--{cli_key} {value}")

    return cmd_parts


def eval_cmd(config: "EvalConfig") -> str:
    args = asdict(config)

    cmd_parts = eval_cmd_parts.copy()

    for key, value in args.items():
        cli_key = key.replace("_", "-")

        if isinstance(value, bool):
            if value:
                cmd_parts.append(f"--{cli_key}")
            continue

        cmd_parts.append(f"--{cli_key} {value}")

    return cmd_parts


def eval_array_job_cmd(seed, logs_dir):
    cmd_parts = eval_cmd_parts.copy()
    if "train/" in logs_dir:
        logs_dir = logs_dir.replace("train/", "")

    cmd_parts.append(f"--seed {seed}")
    cmd_parts.append(f"--logs {logs_dir}")

    return (
            "srun --cpu-bind=v --accel-bind=gn "
            "python -u -m clip_train.main \\\n    "
            + " \\\n    ".join(cmd_parts)
    )


def build_eval_cmd(config: "EvalConfig") -> str:
    cmd_parts = eval_cmd(config)
    return (
            "srun --cpu-bind=v --accel-bind=gn "
            "python -u -m clip_train.main \\\n    "
            + " \\\n    ".join(cmd_parts)
    )


def build_srun_cmd(config: "TrainConfig", wandb_project_name) -> str:
    cmd_parts = build_cmd(config, wandb_project_name)

    return (
            "srun --cpu-bind=v --accel-bind=gn "
            "python -u -m clip_train.main \\\n    "
            + " \\\n    ".join(cmd_parts)
    )


def build_torchrun_cmd(config: "TrainConfig", nproc_per_node) -> str:
    cmd_parts = build_cmd(config)

    return (
            f"torchrun --nproc_per_node {nproc_per_node} "
            "-m clip_train.main -- "
            + " ".join(cmd_parts)
    )


class SlurmManager:
    def __init__(
            self,
            job_name: str = "OpenClip_train",
            partition: str = "gpu",
            nodes: int = 1,
            gpus_per_node: int = 4,
            cpus_per_task: int = 8,
            mem_gb: int = 32,
            root_dir: Path | str = ".",
            workspace_dir: str = ".",
    ) -> None:
        """
        Manager to create and submit Slurm batch jobs.

        Args:
            job_name: Name of the Slurm job.
            partition: Slurm partition to use.
            nodes: Number of nodes.
            gpus_per_node: Number of GPUs per node.
            cpus_per_task: CPU cores per task.
            mem_gb: Memory in gigabytes.
            root_dir: root dir.
        """
        self.job_id = None
        self.job_name = job_name
        self.partition = partition
        self.nodes = nodes
        self.gpus_per_node = gpus_per_node
        self.cpus_per_task = cpus_per_task
        self.mem_gb = mem_gb
        self.script_path = Path(root_dir) / "train.sh"
        self.workspace_dir = workspace_dir

    def build_script(
            self,
            cmd: str,
            save_to: Path = None
    ) -> None:
        """
        Build a Slurm script that runs the given command.

        Args:
            cmd: Command to run.

        Returns:
            Path to the created script.
        """
        cuda_visible = ",".join(str(i) for i in range(self.gpus_per_node))

        script = f"""#!/bin/bash -x
#SBATCH --nodes={self.nodes}
#SBATCH --gres=gpu:{self.gpus_per_node}
#SBATCH --ntasks-per-node={self.gpus_per_node}
#SBATCH --cpus-per-task={self.cpus_per_task}
#SBATCH --wait-all-nodes=1
#SBATCH --job-name={self.job_name}
#SBATCH -p {self.partition}

#SBATCH --output logs/%x-%A.out
#SBATCH --error logs/%x-%A.err

{self.conda_env}

echo "Started at $(date)";

start=`date +%s`

echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

export CUDA_VISIBLE_DEVICES={cuda_visible}
export MASTER_PORT=12802

# first node to be used as the master node in multi-node distributed setup
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

PROJECT_ROOT="{self.workspace_dir}"
TRAIN_DIR="${{PROJECT_ROOT}}/train"

cd "${{TRAIN_DIR}}"

export PYTHONPATH="${{TRAIN_DIR}}:${{PYTHONPATH}}"

{cmd}

end=`date +%s`
runtime=$((end-start))

echo "Ended at $(date)";
echo "Job execution complete."
"""
        if save_to:
            save_to.write_text(script)
        else:
            self.script_path.write_text(script)

    @property
    def conda_env(self) -> str:
        """
        Get the command to activate the conda environment.

        Returns:
            str: The command to activate the environment.
        """
        return "source ~/miniconda3/bin/activate && conda activate test_scaling_laws"

    def submit_job(self, cmd: str):
        """
        Submit the Slurm script using sbatch.
        If wait=True, block until the job finishes.
        """

        self.build_script(cmd=cmd)
        self.submit_without_build()

    def submit_without_build(self, script_path: Path = None, wait: bool = True):
        if not self.script_path.exists() and not script_path.exists():
            raise FileNotFoundError("Slurm script not found.")

        sbatch_cmd = ["sbatch"]
        if wait:
            sbatch_cmd.append("--wait")

        sbatch_cmd.append(str(self.script_path) if script_path is None else str(script_path))

        print("Submitting:", " ".join(sbatch_cmd))

        result = subprocess.run(
            sbatch_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout = result.stdout.strip()
        job_id = None

        if stdout.startswith("Submitted batch job"):
            job_id = stdout.split()[-1]
            job_id = int(job_id)

        self.job_id = job_id
        print("Job ID:", job_id)

    def build_evaluation_script(self, cmd: str, job_name,
                                save_to: Path = None):

        script = f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task={self.cpus_per_task}
#SBATCH --job-name={job_name}
#SBATCH -p {self.partition}

#SBATCH --output evaluator_logs/%x-%A.out
#SBATCH --error evaluator_logs/%x-%A.err

{self.conda_env}

echo "Started at $(date)";

PROJECT_ROOT="{self.workspace_dir}"
TRAIN_DIR="${{PROJECT_ROOT}}/train"

cd "${{TRAIN_DIR}}"

export PYTHONPATH="${{TRAIN_DIR}}:${{PYTHONPATH}}"

{cmd}

echo "Ended at $(date)";
echo "Job execution complete."
"""
        if save_to:
            save_to.write_text(script)
        else:
            self.script_path.write_text(script)

    def build_downstream_evaluation_script(self, job_name, root, model, seed, save_to: Path = None):
        if "train/" in root:
            root = root.replace("train/", "")
        script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task={self.cpus_per_task}
#SBATCH -p {self.partition}

#SBATCH --output=downstream_evaluator_logs/%x-%j.out
#SBATCH --error=downstream_evaluator_logs/%x-%j.err

{self.conda_env}

PROJECT_ROOT="{self.workspace_dir}"
TRAIN_DIR="${{PROJECT_ROOT}}/train"

cd "${{TRAIN_DIR}}"

export PYTHONPATH="${{TRAIN_DIR}}:${{PYTHONPATH}}"

python post_hoc_iterator.py --root {root} --model {model} --seed {seed}
"""
        if save_to:
            save_to.write_text(script)
        else:
            self.script_path.write_text(script)

    def build_array_job_for_evaluation(self, cmd: str, models_cfg_dir, nr_total_configs, save_to: Path = None):
        model_name = f'local-dir:{models_cfg_dir}/config_${{SLURM_ARRAY_TASK_ID}}'

        cmd = cmd.strip()
        cmd = f'{cmd} \\\n    --model "{model_name}" \\\n    --name "${{RUN_NAME}}"'

        script = f"""#!/bin/bash
#SBATCH --nodes={self.nodes}
#SBATCH --gres=gpu:{self.gpus_per_node}
#SBATCH --cpus-per-task={self.cpus_per_task}
#SBATCH -p {self.partition}

#SBATCH --array=1-{nr_total_configs}

#SBATCH --job-name=eval_config
#SBATCH --output=evaluator_logs/%x-%A_%a.out
#SBATCH --error=evaluator_logs/%x-%A_%a.err

{self.conda_env}

echo "Started at $(date)"
echo "Array task id: ${{SLURM_ARRAY_TASK_ID}}"

PROJECT_ROOT="{self.workspace_dir}"
TRAIN_DIR="${{PROJECT_ROOT}}/train"

cd "${{TRAIN_DIR}}"
export PYTHONPATH="${{TRAIN_DIR}}:${{PYTHONPATH}}"

CONFIG_ID="${{SLURM_ARRAY_TASK_ID}}"
RUN_NAME="train_config${{CONFIG_ID}}"

{cmd}

echo "Ended at $(date)"
echo "Job execution complete."
"""
        if save_to:
            save_to.write_text(script)
        else:
            self.script_path.write_text(script)

    def build_array_job_for_downstream_evaluation(self, root, models_cfg_dir, nr_total_configs, seed,
                                                  save_to: Path = None):
        if "train/" in root:
            root = root.replace("train/", "")
        model_name = f'"local-dir:{models_cfg_dir}/config_${{SLURM_ARRAY_TASK_ID}}"'
        script = f"""#!/bin/bash
#SBATCH --job-name=downstream_eval_config
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task={self.cpus_per_task}
#SBATCH -p {self.partition}

#SBATCH --array=1-{nr_total_configs}

#SBATCH --output=downstream_evaluator_logs/%x-%A_%a.out
#SBATCH --error=downstream_evaluator_logs/%x-%A_%a.err

{self.conda_env}

PROJECT_ROOT="{self.workspace_dir}"
TRAIN_DIR="${{PROJECT_ROOT}}/train"

cd "${{TRAIN_DIR}}"
export PYTHONPATH="${{TRAIN_DIR}}:${{PYTHONPATH}}"

config_ID="${{SLURM_ARRAY_TASK_ID}}"
RUN_NAME="train_config${{config_ID}}"

ROOT_DIR={root}"/train_config${{config_ID}}"

echo "Running downstream eval for config ${{config_ID}}"
echo "Root dir: ${{ROOT_DIR}}"

python post_hoc_iterator.py --root "${{ROOT_DIR}}" --seed {seed} --model {model_name}
"""

        if save_to:
            save_to.write_text(script)
        else:
            self.script_path.write_text(script)

    def build_array_script(self,
                           model_root: str,
                           logs_root: str,
                           config_txt: str,
                           array_size: int,
                           save_to: Path,
                           batch_size: int,
                           seed: int
                           ) -> None:

        cuda_visible = ",".join(str(i) for i in range(self.gpus_per_node))

        script = f"""#!/bin/bash
#SBATCH --nodes={self.nodes}
#SBATCH --gres=gpu:{self.gpus_per_node}
#SBATCH --ntasks-per-node={self.gpus_per_node}
#SBATCH --cpus-per-task={self.cpus_per_task}
#SBATCH --wait-all-nodes=1
#SBATCH --job-name={self.job_name}
#SBATCH -p {self.partition}
#SBATCH --array=1-{array_size}

#SBATCH --output array_job_logs/%x-%A_%a.out
#SBATCH --error array_job_logs/%x-%A_%a.err

{self.conda_env}

echo "Started at $(date)";
start=`date +%s`

export CUDA_VISIBLE_DEVICES={cuda_visible}
export MASTER_PORT=$((12000 + SLURM_ARRAY_TASK_ID % 20000))

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

PROJECT_ROOT="{self.workspace_dir}"
TRAIN_DIR="${{PROJECT_ROOT}}/train"
cd "${{TRAIN_DIR}}"
export PYTHONPATH="${{TRAIN_DIR}}:${{PYTHONPATH}}"

CONFIG_TXT="{config_txt}"

LINE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "${{CONFIG_TXT}}")
[ -z "$LINE" ] && echo "No line for task $SLURM_ARRAY_TASK_ID" && exit 1

read -r config_ID TRAIN_NUM_SAMPLES LR WD WARMUP BETA2 EPS <<< "$LINE"

TRAIN_NUM_SAMPLES=${{TRAIN_NUM_SAMPLES%.*}}

srun --cpu-bind=v --accel-bind=gn python -u -m clip_train.main \\
    --train-data "../pre_training_dataset/laion400m/train_data/{{00000000..00012136}}.tar" \\
    --val-data "../pre_training_dataset/laion400m/validation_data/{{00000000..00000015}}.tar" \\
    --test-data "../pre_training_dataset/laion400m/test_data/{{00000000..00000015}}.tar" \\
    --local-loss \\
    --gather-with-grad \\
    --dataset-type webdataset \\
    --precision amp \\
    --report-to wandb \\
    --wandb-project-name open-clip-scan-space \\
    --model "local-dir:{model_root}/model_cfg/config_${{config_ID}}" \\
    --train-num-samples "${{TRAIN_NUM_SAMPLES}}" \\
    --epochs 1 \\
    --name "train_config${{config_ID}}" \\
    --logs "{logs_root}" \\
    --seed {seed} \\
    --batch-size {batch_size} \\
    --pipeline-id "${{config_ID}}" \\
    --lr "${{LR}}" \\
    --wd "${{WD}}" \\
    --warmup-fraction "${{WARMUP}}" \\
    --beta2 "${{BETA2}}" \\
    --eps "${{EPS}}"

end=`date +%s`
echo "Ended at $(date)";
echo "Job execution complete."
"""
        if save_to:
            save_to.write_text(script)
        else:
            self.script_path.write_text(script)
