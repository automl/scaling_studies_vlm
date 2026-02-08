#!/bin/bash
# Usage: sbatch configs_iterator.sh [WORKSPACE_DIR] [CODE_DIR] [ROOT_DIR]
# Example: sbatch configs_iterator.sh /work/dlclarge1/sinanid-VLM-scaling-law/scaling_studies_vlm /home/alipourn/scaling_studies_vlm/train results/optimizer=scale_space_only_minimal_capacity_probe/seed=42

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --partition develbooster
#SBATCH --account projectnucleus
#SBATCH --job-name=individual_scripts_builder

#SBATCH --output configs_iterator_logs/%x-%A.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively. short: -o logs/%x-%A-job_name.out
#SBATCH --error configs_iterator_logs/%x-%A.err    # STDERR  short: -e logs/%x-%A-job_name.out

# Set defaults or use provided arguments
WORKSPACE_DIR="${1:-/p/data1/datasets/scan/openclip}"
COMMAND_DIR="${2:-/p/home/jusers/sinani1/juwels/sinani1/scaling_studies_vlm/train}"
ROOT_DIR="${3:-results/optimizer=experimental/seed=0}"
VENV_DIR="${4:-/p/project1/projectnucleus/sinani1/envs/openclip/bin/activate}"

echo "WORKSPACE_DIR: $WORKSPACE_DIR"
echo "COMMAND_DIR: $COMMAND_DIR"
echo "ROOT_DIR: $ROOT_DIR"

echo "Started at $(date)";

# Running the job
start=`date +%s`
cd "${WORKSPACE_DIR}/train"

export PYTHONPATH="${COMMAND_DIR}:${PYTHONPATH}"

srun -n $SLURM_NTASKS --gpus-per-task=1 \
     --gpu-bind=single:1 bash -lc '
python -u '"${COMMAND_DIR}"'/configs_iterator.py \
  --root_dir '"${ROOT_DIR}"' \
  --seed 0 \
  --worker_id $SLURM_PROCID \
  --num_workers $SLURM_NTASKS \
  --workspace_dir '"${WORKSPACE_DIR}"' \
  --cluster juwels \
  --partition develbooster \
  --wandb_project_name experimental \
  --command_dir '"${COMMAND_DIR}"' \
  --venv_path '"${VENV_DIR}"'
'

end=`date +%s`
runtime=$((end-start))

echo "Ended at $(date)";
echo Job execution complete.
