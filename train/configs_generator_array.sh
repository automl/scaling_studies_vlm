#!/bin/bash
# Usage: sbatch configs_generator_array.sh [WORKSPACE_DIR] [CODE_DIR]
# SLURM_ARRAY_TASK_ID maps: optimizer = TASK_ID % 3, seed = 7 + (TASK_ID / 3)

#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=200G
#SBATCH --ntasks-per-node=1
#SBATCH --partition capella
#SBATCH --time=UNLIMITED
#SBATCH --job-name=array_config_generator
#SBATCH --array=0-2
#SBATCH --time=24:00:00

#SBATCH --output configs_iterator_logs/%x-%A_%a.out
#SBATCH --error configs_iterator_logs/%x-%A_%a.err

source /software/genoa/r24.10/Anaconda3/2024.06-1/bin/activate && conda activate /home/naal364i/miniconda3/envs/test_scaling_laws

WORKSPACE_DIR="${1:-/data/cat/ws/naal364i-primo/runs}"
CODE_DIR="${2:-/home/naal364i/scaling_studies_vlm/train}"
VENV_DIR="${3:-/software/genoa/r24.10/Anaconda3/2024.06-1/bin/activate}"

HPO_OPTIMIZERS=(bo_rw_80M random_80M primo_80M)
OPTIMIZER_IDX=$((SLURM_ARRAY_TASK_ID % 3))
SEED=1

HPO_OPTIMIZER="${HPO_OPTIMIZERS[$OPTIMIZER_IDX]}"

echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "WORKSPACE_DIR: $WORKSPACE_DIR"
echo "CODE_DIR: $CODE_DIR"
echo "HPO_OPTIMIZER: $HPO_OPTIMIZER"
echo "SEED: $SEED"

echo "Started at $(date)";
start=`date +%s`

cd "${WORKSPACE_DIR}/train"

export PYTHONPATH="${CODE_DIR}:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=$((12000 + SLURM_ARRAY_TASK_ID % 20000))
export NEPS_MAX_RETRIES_GET_NEXT_TRIAL=1
export WORKSPACE_DIR CODE_DIR HPO_OPTIMIZER SEED

echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES before srun: ${CUDA_VISIBLE_DEVICES:-unset}"

srun --ntasks=1 --gpus-per-task=4 --gpu-bind=single:4 \
  bash -lc '
    source '"$VENV_DIR"' && conda activate /home/naal364i/miniconda3/envs/test_scaling_laws
    echo "CUDA_VISIBLE_DEVICES inside srun: ${CUDA_VISIBLE_DEVICES:-unset}"
    python -u /home/naal364i/scaling_studies_vlm/train/configs_generator.py \
  --hpo-optimizer "$HPO_OPTIMIZER" \
  --seed "$SEED" \
  --workspace_dir "$WORKSPACE_DIR"
  '

end=`date +%s`
runtime=$((end-start))

echo "Ended at $(date)";
echo "Job execution complete. Runtime: ${runtime}s"
