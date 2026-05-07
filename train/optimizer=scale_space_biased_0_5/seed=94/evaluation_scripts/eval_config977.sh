#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=eval_config977
#SBATCH -p alldlc_gpu-rtx3080
#SBATCH --time=02:00:00

#SBATCH --output evaluator_logs/%x-%A.out
#SBATCH --error evaluator_logs/%x-%A.err

source ~/miniconda3/bin/activate && conda activate test_scaling_laws

echo "Started at $(date)";


COMMAND_DIR="/work/dlclarge1/sinanid-VLM-scaling-law/repository/scaling_studies_vlm/train"
TRAIN_DIR="/work/dlclarge1/sinanid-VLM-scaling-law/scaling_studies_vlm/train"

export PYTHONPATH="${COMMAND_DIR}:${PYTHONPATH}"
cd "${TRAIN_DIR}"


python -u -m eval.upstream_eval \
          --config-id 977 \
          --models-cfg-dir /work/dlclarge1/sinanid-VLM-scaling-law/scaling_studies_vlm/train/results/optimizer=scan_space_final_biased_alpha_0_5/seed=94/model_cfg/config_977 \
          --logs-dir /work/dlclarge1/sinanid-VLM-scaling-law/scaling_studies_vlm/train/results/optimizer=scan_space_final_biased_alpha_0_5/seed=94/individual_training_logs \
          --val-data "/work/dlclarge1/sinanid-VLM-scaling-law/scaling_studies_vlm/pre_training_dataset/laion400m/validation_data/{00000000..00000015}.tar" \
          --test-data "/work/dlclarge1/sinanid-VLM-scaling-law/scaling_studies_vlm/pre_training_dataset/laion400m/test_data/{00000000..00000015}.tar" \
          --seed 94 \
          --train-dir /work/dlclarge1/sinanid-VLM-scaling-law/scaling_studies_vlm/train \
          --command-dir /work/dlclarge1/sinanid-VLM-scaling-law/repository/scaling_studies_vlm/train

echo "Ended at $(date)";
echo "Job execution complete."
