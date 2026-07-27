#!/bin/bash
#SBATCH --job-name=downstream_eval_config530
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -p capella

#SBATCH --output=downstream_evaluator_logs/%x-%j.out
#SBATCH --error=downstream_evaluator_logs/%x-%j.err

module load Anaconda3
source $EBROOTANACONDA3/etc/profile.d/conda.sh
conda activate /data/cat/ws/dosi100i-scan_bench_suite/vlm_env


COMMAND_DIR="/data/cat/ws/dosi100i-scan_bench_suite/vlm_repo/scaling_studies_vlm/train"
TRAIN_DIR="/data/cat/ws/dosi100i-scan_bench_suite/runs/train"

export PYTHONPATH="${COMMAND_DIR}:${PYTHONPATH}"
cd "${TRAIN_DIR}"


python post_hoc_iterator.py --root results/optimizer=scan_space_large_scale/seed=42/individual_training_logs/train_config530 --model local-dir:results/optimizer=scan_space_large_scale/seed=42/model_cfg/config_530 --seed 42 --pipeline_id 530 --cluster tu_dresden
