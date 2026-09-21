#!/bin/bash
# Usage: sbatch configs_generator.sh [WORKSPACE_DIR] [CODE_DIR] [HPO_OPTIMIZER] [SEED]
# Example: sbatch configs_generator.sh /work/dlc2workfs1/alipourn-primo/naal364i-primo /home/alipourn/scaling_studies_vlm/train bo_rw

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --partition capella
#SBATCH --job-name=configs_generator

#SBATCH --output configs_iterator_logs/%x-%A.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively.
#SBATCH --error configs_iterator_logs/%x-%A.err    # STDERR

# Set defaults or use provided arguments

source /home/alipourn/miniconda3/bin/activate && conda activate test_scaling_laws

WORKSPACE_DIR="${1:-/work/dlc2workfs1/alipourn-primo/naal364i-primo}"
CODE_DIR="${2:-/home/alipourn/scaling_studies_vlm/train}"
HPO_OPTIMIZER="${3:-primo}"
SEED="${4:-1}"
VENV_DIR="${5:-/home/alipourn/miniconda3/bin/activate}"

echo "WORKSPACE_DIR: $WORKSPACE_DIR"
echo "CODE_DIR: $CODE_DIR"
echo "HPO_OPTIMIZER: $HPO_OPTIMIZER"
echo "SEED: $SEED"

echo "Started at $(date)";

# Running the job
start=`date +%s`


cd "${WORKSPACE_DIR}/train"

export PYTHONPATH="${CODE_DIR}:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
export MASTER_PORT=$((12000 + SLURM_ARRAY_TASK_ID % 20000))
export NEPS_MAX_RETRIES_GET_NEXT_TRIAL=1 
export WORKSPACE_DIR CODE_DIR HPO_OPTIMIZER SEED

echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES before srun: ${CUDA_VISIBLE_DEVICES:-unset}"

srun --ntasks=1 --gpus-per-task=1 --gpu-bind=single:1 \
  bash -lc '
    source '"$VENV_DIR"' && conda activate test_scaling_laws
    echo "CUDA_VISIBLE_DEVICES inside srun: ${CUDA_VISIBLE_DEVICES:-unset}"
    python -u /home/alipourn/scaling_studies_vlm/train/configs_generator.py \
  --hpo-optimizer "$HPO_OPTIMIZER" \
  --seed "$SEED" \
  --workspace_dir "$WORKSPACE_DIR"
  '


end=`date +%s`
runtime=$((end-start))

echo "Ended at $(date)";
echo "Job execution complete. Runtime: ${runtime}s"
