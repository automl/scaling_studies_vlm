import subprocess
from enum import Enum
from pathlib import Path


class DatasetShards(str, Enum):
    LAION400M = "../pre_training_dataset/laion400m/shards/{00000000..00000009}.tar"


def build_cmd(args, learning_rate: float, logs_dir: str) -> str:
    cmd_parts = [
        f'--train-data="{DatasetShards.LAION400M}"',
        f"--batch-size={args.batch_size}",
        f"--epochs={args.epochs}",
        f"--workers={args.workers}",
        f"--model {args.model}",
        f"--seed {args.seed}",
        f"--train-num-samples {args.train_num_samples}",
        f"--lr {learning_rate}",
        "--local-loss",
        "--gather-with-grad",
        "--dataset-type=webdataset",
        f'--logs="{logs_dir}"',
    ]
    return "srun --cpu-bind --accel-bind=gn python -u -m open_clip_train.main \\\n    " + " \\\n    ".join(cmd_parts)


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

{cmd}

end=`date +%s`
runtime=$((end-start))

echo "Ended at $(date)";
echo "Job execution complete."
"""

        self.script_path.write_text(script)

    @property
    def conda_env(self) -> str:
        """
        Get the command to activate the conda environment.

        Returns:
            str: The command to activate the environment.
        """
        return "source ~/miniconda3/bin/activate && conda activate vlm_scaling_law"

    def submit_job(self, cmd: str, wait: bool = False):
        """
        Submit the Slurm script using sbatch.
        If wait=True, block until the job finishes.
        """

        self.build_script(cmd=cmd)

        if not self.script_path.exists():
            raise FileNotFoundError("Slurm script not found.")

        sbatch_cmd = ["sbatch"]
        if wait:
            sbatch_cmd.append("--wait")

        sbatch_cmd.append(str(self.script_path))

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
