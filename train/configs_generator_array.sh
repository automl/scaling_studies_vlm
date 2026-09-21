#!/bin/bash
# Usage: sbatch configs_generator_array.sh [WORKSPACE_DIR] [CODE_DIR]
# SLURM_ARRAY_TASK_ID maps: seed = 1 + TASK_ID

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --ntasks-per-node=1
#SBATCH --partition mldlc2_gpu-h200
#SBATCH --job-name=primo
#SBATCH --array=10,9,8,3
#SBATCH --time=24:00:00

#SBATCH --output configs_iterator_logs/%x-%A_%a.out
#SBATCH --error configs_iterator_logs/%x-%A_%a.err

source /home/alipourn/miniconda3/bin/activate && conda activate test_scaling_laws

WORKSPACE_DIR="${1:-/work/dlc2workfs1/alipourn-primo/naal364i-primo}"
CODE_DIR="${2:-/home/alipourn/scaling_studies_vlm/train}"
VENV_DIR="${3:-/home/alipourn/miniconda3/bin/activate}"

HPO_OPTIMIZER=primo
SEED=$SLURM_ARRAY_TASK_ID

echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "WORKSPACE_DIR: $WORKSPACE_DIR"
echo "CODE_DIR: $CODE_DIR"
echo "HPO_OPTIMIZER: $HPO_OPTIMIZER"
echo "SEED: $SEED"

echo "Started at $(date)";
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
