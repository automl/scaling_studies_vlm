#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -p capella-interactive
#SBATCH --time=00:30:00

#SBATCH --array=405,431,434

#SBATCH --job-name=eval_config_partial
#SBATCH --output=evaluator_logs/%x-%A_%a.out
#SBATCH --error=evaluator_logs/%x-%A_%a.err

module load Anaconda3
source $EBROOTANACONDA3/etc/profile.d/conda.sh
conda activate /data/cat/ws/dosi100i-scan_bench_suite/vlm_env

echo "Started at $(date)"
echo "Array task id: ${SLURM_ARRAY_TASK_ID}"

COMMAND_DIR="/data/cat/ws/dosi100i-scan_bench_suite/vlm_repo/scaling_studies_vlm/train"
TRAIN_DIR="/data/cat/ws/dosi100i-scan_bench_suite/runs/train"

export PYTHONPATH="${COMMAND_DIR}:${PYTHONPATH}"
cd "${TRAIN_DIR}"

LOG_DIR="/data/cat/ws/dosi100i-scan_bench_suite/runs/train/results/optimizer=scan_space_large_scale/seed=42/evaluation_scripts"
mkdir -p "${LOG_DIR}"

CONFIG_ID="${SLURM_ARRAY_TASK_ID}"

echo "Running evaluation for config ID: ${CONFIG_ID}"

python -u -m eval.upstream_eval \
  --models-cfg-dir /data/cat/ws/dosi100i-scan_bench_suite/runs/train/results/optimizer=scan_space_large_scale/seed=42/model_cfg \
  --logs-dir /data/cat/ws/dosi100i-scan_bench_suite/runs/train/results/optimizer=scan_space_large_scale/seed=42/individual_training_logs \
  --val-data "/data/cat/ws/dosi100i-scan_bench_suite/vlm_data/val_data/{00000000..00000015}.tar" \
  --test-data "/data/cat/ws/dosi100i-scan_bench_suite/vlm_data/test_data/{00000000..00000015}.tar" \
  --seed 42 \
  --train-dir /data/cat/ws/dosi100i-scan_bench_suite/runs/train \
  --command-dir /data/cat/ws/dosi100i-scan_bench_suite/vlm_repo/scaling_studies_vlm/train \
  --config-id "${CONFIG_ID}"


echo "Ended at $(date)"
echo "Job execution complete."
