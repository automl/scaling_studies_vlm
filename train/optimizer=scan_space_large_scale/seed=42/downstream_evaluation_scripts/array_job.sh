#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH -p capella
#SBATCH --time=24:00:00

#SBATCH --array=1-250

#SBATCH --job-name=downstream_eval
#SBATCH --output=downstream_evaluator_logs/%x-%A_%a.out
#SBATCH --error=downstream_evaluator_logs/%x-%A_%a.err

module load Anaconda3
source $EBROOTANACONDA3/etc/profile.d/conda.sh
conda activate /data/cat/ws/dosi100i-scan_bench_suite/vlm_env

echo "Started at $(date)"
echo "Array task id: ${SLURM_ARRAY_TASK_ID}"


COMMAND_DIR="/data/cat/ws/dosi100i-scan_bench_suite/vlm_repo/scaling_studies_vlm/train"
TRAIN_DIR="/data/cat/ws/dosi100i-scan_bench_suite/runs/train"

export PYTHONPATH="${COMMAND_DIR}:${PYTHONPATH}"
cd "${TRAIN_DIR}"


LOG_DIR="/data/cat/ws/dosi100i-scan_bench_suite/runs/train/results/optimizer=scan_space_large_scale/seed=42/downstream_evaluation_scripts"
mkdir -p "${LOG_DIR}"
mkdir -p "${LOG_DIR}/downstream_array_eval_logs"

MODELS_CFG_DIR="/data/cat/ws/dosi100i-scan_bench_suite/runs/train/results/optimizer=scan_space_large_scale/seed=42/model_cfg"
ROOT_DIR="/data/cat/ws/dosi100i-scan_bench_suite/runs/train/results/optimizer=scan_space_large_scale/seed=42/individual_training_logs"
CLUSTER="tu_dresden"
SEED="42"

BASE_ID=$(( (SLURM_ARRAY_TASK_ID - 1) * 4 + 1 ))

echo "Array task ${SLURM_ARRAY_TASK_ID} uses config ids: ${BASE_ID}..$((BASE_ID+4-1))"

run_one () {
    local step_idx="$1"
    local config_ID="$2"

    if [ "${config_ID}" -gt "1000" ]; then
      echo "Skipping step ${step_idx} because config_ID=${config_ID} exceeds nr_total_configs=1000"
      return 0
    fi

    MODEL_NAME="local-dir:${MODELS_CFG_DIR}/config_${config_ID}"

    echo "Starting downstream eval step ${step_idx} (config_ID=${config_ID}) at $(date)"

    srun --exclusive --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
      --output "${LOG_DIR}/downstream_array_eval_logs/step_${step_idx}_%x-%A_%a.out" \
      --error  "${LOG_DIR}/downstream_array_eval_logs/step_${step_idx}_%x-%A_%a.err" \
      python -m post_hoc_iterator \
        --root "${ROOT_DIR}/train_config${config_ID}" \
        --model "${MODEL_NAME}" \
        --seed "${SEED}" \
        --pipeline_id "${config_ID}" \
        --cluster "${CLUSTER}" &
}

for step_idx in 0 1 2 3; do
    config_ID=$(( BASE_ID + step_idx ))
    run_one "${step_idx}" "${config_ID}"
done

wait

echo "Ended at $(date)"
echo "Job execution complete."
