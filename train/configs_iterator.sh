#!/bin/bash
# Usage: sbatch configs_iterator.sh [WORKSPACE_DIR] [CODE_DIR] [ROOT_DIR]
# Example: sbatch configs_iterator.sh /work/dlclarge1/sinanid-VLM-scaling-law/scaling_studies_vlm /home/alipourn/scaling_studies_vlm/train results/optimizer=scale_space_only_minimal_capacity_probe/seed=42

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --partition testdlc2_gpu-l40s
#SBATCH --job-name=individual_scripts_builder
#SBATCH --time=00:30:00

#SBATCH --output configs_iterator_logs/%x-%A.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively. short: -o logs/%x-%A-job_name.out
#SBATCH --error configs_iterator_logs/%x-%A.err    # STDERR  short: -e logs/%x-%A-job_name.out
source ~/miniconda3/bin/activate && conda activate test_scaling_laws
# Set defaults or use provided arguments

WORKSPACE_DIR="${1:-/work/dlclarge1/sinanid-VLM-scaling-law/scaling_studies_vlm}"
COMMAND_DIR="${2:-/home/alipourn/scaling_studies_vlm/train}"
ROOT_DIR="${3:-results/optimizer=random_fixed_max_dp/seed=1}"
VENV_DIR="${4:-/home/alipourn/miniconda3/bin/activate}"

echo "WORKSPACE_DIR: $WORKSPACE_DIR"
echo "COMMAND_DIR: $COMMAND_DIR"
echo "ROOT_DIR: $ROOT_DIR"

echo "Started at $(date)";

start=`date +%s`
export PYTHONPATH="${COMMAND_DIR}:${PYTHONPATH}"

echo "which python: $(which python)"
srun -n $SLURM_NTASKS --gpus-per-task=1 \
     --gpu-bind=single:1 bash -lc '
cd '"${WORKSPACE_DIR}/train"'
source '"${VENV_DIR}"' && conda activate test_scaling_laws
python -u '"${COMMAND_DIR}"'/configs_iterator.py \
  --root_dir '"${ROOT_DIR}"' \
  --seed 42 \
  --worker_id $SLURM_PROCID \
  --num_workers $SLURM_NTASKS \
  --workspace_dir '"${WORKSPACE_DIR}"' \
  --cluster meta \
  --partition mldlc2_gpu-h200 \
  --wandb_project_name random_fixed_max_dp \
  --command_dir '"${COMMAND_DIR}"' \
  --venv_path '"${VENV_DIR}"' \
'

end=`date +%s`
runtime=$((end-start))

echo "Ended at $(date)";
echo Job execution complete.
