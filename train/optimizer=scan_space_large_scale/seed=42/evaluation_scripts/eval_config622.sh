#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=eval_config622
#SBATCH -p capella
#SBATCH --time=02:00:00

#SBATCH --output evaluator_logs/%x-%A.out
#SBATCH --error evaluator_logs/%x-%A.err

module load Anaconda3
source $EBROOTANACONDA3/etc/profile.d/conda.sh
conda activate /data/cat/ws/dosi100i-scan_bench_suite/vlm_env

echo "Started at $(date)";


COMMAND_DIR="/data/cat/ws/dosi100i-scan_bench_suite/vlm_repo/scaling_studies_vlm/train"
TRAIN_DIR="/data/cat/ws/dosi100i-scan_bench_suite/runs/train"

export PYTHONPATH="${COMMAND_DIR}:${PYTHONPATH}"
cd "${TRAIN_DIR}"


python -u -m eval.upstream_eval \
          --config-id 622 \
          --models-cfg-dir /data/cat/ws/dosi100i-scan_bench_suite/runs/train/results/optimizer=scan_space_large_scale/seed=42/model_cfg/config_622 \
          --logs-dir /data/cat/ws/dosi100i-scan_bench_suite/runs/train/results/optimizer=scan_space_large_scale/seed=42/individual_training_logs \
          --val-data "/data/cat/ws/dosi100i-scan_bench_suite/vlm_data/validation_data/{00000000..00000015}.tar" \
          --test-data "/data/cat/ws/dosi100i-scan_bench_suite/vlm_data/test_data/{00000000..00000015}.tar" \
          --seed 42 \
          --train-dir /data/cat/ws/dosi100i-scan_bench_suite/runs/train \
          --command-dir /data/cat/ws/dosi100i-scan_bench_suite/vlm_repo/scaling_studies_vlm/train

echo "Ended at $(date)";
echo "Job execution complete."
