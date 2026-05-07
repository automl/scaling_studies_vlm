#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -p alldlc_gpu-rtx3080
#SBATCH --time=01:00:00

#SBATCH --array=901-940%6

#SBATCH --job-name=eval_config
#SBATCH --output=evaluator_logs/%x-%A_%a.out
#SBATCH --error=evaluator_logs/%x-%A_%a.err

source ~/miniconda3/bin/activate
conda activate test_scaling_laws

echo "Started at $(date)"
echo "Array task id: ${SLURM_ARRAY_TASK_ID}"

COMMAND_DIR="/work/dlclarge1/sinanid-VLM-scaling-law/repository/scaling_studies_vlm/train"
TRAIN_DIR="/work/dlclarge1/sinanid-VLM-scaling-law/scaling_studies_vlm/train"

export PYTHONPATH="${COMMAND_DIR}:${PYTHONPATH}"
cd "${TRAIN_DIR}"

LOG_DIR="/work/dlclarge1/sinanid-VLM-scaling-law/scaling_studies_vlm/train/results/optimizer=scan_space_final_biased_alpha_0_5/seed=94/evaluation_scripts"
mkdir -p "${LOG_DIR}/array_eval_logs"

CONFIG_ID="${SLURM_ARRAY_TASK_ID}"

echo "Evaluating config ${CONFIG_ID}"

python -u -m eval.upstream_eval \
    --models-cfg-dir /work/dlclarge1/sinanid-VLM-scaling-law/scaling_studies_vlm/train/results/optimizer=scan_space_final_biased_alpha_0_5/seed=94/model_cfg \
    --logs-dir /work/dlclarge1/sinanid-VLM-scaling-law/scaling_studies_vlm/train/results/optimizer=scan_space_final_biased_alpha_0_5/seed=94/individual_training_logs \
    --val-data "/work/dlclarge1/sinanid-VLM-scaling-law/scaling_studies_vlm/pre_training_dataset/laion400m/validation_data/{00000000..00000015}.tar" \
    --test-data "/work/dlclarge1/sinanid-VLM-scaling-law/scaling_studies_vlm/pre_training_dataset/laion400m/test_data/{00000000..00000015}.tar" \
    --seed 94 \
    --train-dir /work/dlclarge1/sinanid-VLM-scaling-law/scaling_studies_vlm/train \
    --command-dir /work/dlclarge1/sinanid-VLM-scaling-law/repository/scaling_studies_vlm/train \
    --config-id "${CONFIG_ID}"

echo "Ended at $(date)"
echo "Job execution complete."
