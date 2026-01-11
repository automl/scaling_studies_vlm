import subprocess
from dataclasses import asdict
from enum import Enum
from pathlib import Path

from train.config import TrainConfig


class DatasetShards(str, Enum):
    LAION400M_TRAIN_DATA = "../pre_training_dataset/laion400m/shards/{00000000..00006193}.tar",
    LAION400M_VALIDATION_DATA = "../pre_training_dataset/laion400m/validation_data/{00000000..00000015}.tar"
    LAION400M_TEST_DATA = "../pre_training_dataset/laion400m/test_data/{00000000..00000015}.tar"


def build_cmd(config: TrainConfig) -> str:
    args = asdict(config)

    cmd_parts = [
        f'--train-data "{DatasetShards.LAION400M_TRAIN_DATA.value}"',
        f'--val-data "{DatasetShards.LAION400M_VALIDATION_DATA.value}"',
        f'--test-data "{DatasetShards.LAION400M_TEST_DATA.value}"',
        "--local-loss",
        "--gather-with-grad",
        "--dataset-type webdataset",
        "--precision bf16"
    ]

    for key, value in args.items():
        cli_key = key.replace("_", "-")

        if isinstance(value, bool):
            if value:
                cmd_parts.append(f"--{cli_key}")
            continue

        cmd_parts.append(f"--{cli_key} {value}")

    return cmd_parts


def build_srun_cmd(config: TrainConfig) -> str:
    cmd_parts = build_cmd(config)

    return (
            "srun --cpu-bind=v --accel-bind=gn "
            "python -u -m open_clip_train.main \\\n    "
            + " \\\n    ".join(cmd_parts)
    )


def build_torchrun_cmd(config: TrainConfig, nproc_per_node) -> str:
    cmd_parts = build_cmd(config)

    return (
            f"torchrun --nproc_per_node {nproc_per_node} "
            "-m open_clip_train.main -- "
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

PROJECT_ROOT="/work/dlclarge1/sinanid-VLM-scaling-law/scaling_studies_vlm"
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


class BatchSizeSlurmManager(SlurmManager):
    """
    Submits a short 1 GPU job that finds the max local batch size that fits on that GPU.
    """

    def __init__(
            self,
            partition: str,
            job_name: str = "OpenClip_bs_probe",
            **kwargs,
    ) -> None:
        super().__init__(
            job_name=job_name,
            partition=partition,
            nodes=1,
            gpus_per_node=1,
            **kwargs,
        )

    def build_script(self, cmd: str = "", save_to: Path = None) -> None:
        script = f"""#!/bin/bash
        #SBATCH --nodes={self.nodes}
        #SBATCH --gres=gpu:{self.gpus_per_node}
        #SBATCH --ntasks-per-node={self.gpus_per_node}
        #SBATCH --cpus-per-task={self.cpus_per_task}
        #SBATCH --job-name={self.job_name}
        #SBATCH -p {self.partition}

        #SBATCH --output bs_probing_logs/%x-%A.out
        #SBATCH --error bs_probing_logs/%x-%A.err

        {self.conda_env}

        echo "Started at $(date)";

        python probe_max_bs.py

        echo "Ended at $(date)";
        echo "Job execution complete."
        """
        if save_to:
            save_to.write_text(script)
        else:
            self.script_path.write_text(script)
