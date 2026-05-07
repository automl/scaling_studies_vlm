#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=train_palldlc_gpu-rtx3080_bs1024_g4_n1
#SBATCH -p alldlc_gpu-rtx3080
#SBATCH --time=24:00:00
#SBATCH --array=1-8

#SBATCH --output array_job_logs/%x-%A_%a.out
#SBATCH --error array_job_logs/%x-%A_%a.err

source ~/miniconda3/bin/activate && conda activate test_scaling_laws

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker,virbr
export NCCL_DEBUG=WARN

echo "Started at $(date)";
start=`date +%s`

export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=$((12000 + SLURM_ARRAY_TASK_ID % 20000))

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr


COMMAND_DIR="/work/dlclarge1/sinanid-VLM-scaling-law/repository/scaling_studies_vlm/train"
TRAIN_DIR="/work/dlclarge1/sinanid-VLM-scaling-law/scaling_studies_vlm/train"

export PYTHONPATH="${COMMAND_DIR}:${PYTHONPATH}"
cd "${TRAIN_DIR}"

export WANDB_DIR=${TRAIN_DIR}/scan_space_biased_space_0_5

CONFIG_TXT="/work/dlclarge1/sinanid-VLM-scaling-law/scaling_studies_vlm/train/results/optimizer=scan_space_final_biased_alpha_0_5/seed=94/train_scripts/train_palldlc_gpu_rtx3080_bs1024_g4_n1.txt"

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${CONFIG_TXT}")
[ -z "$LINE" ] && echo "No line for task $SLURM_ARRAY_TASK_ID" && exit 1

read -r config_ID TRAIN_NUM_SAMPLES LR WD WARMUP BETA1 BETA2 EPS EPOCHS <<< "$LINE"

if ! [[ "$EPOCHS" =~ ^[0-9]+$ ]]; then
    echo "Error: EPOCHS is not an integer: $EPOCHS"
    exit 1
fi

EPOCHS=$((EPOCHS))

TRAIN_NUM_SAMPLES=${TRAIN_NUM_SAMPLES%.*}

srun --cpu-bind=v --accel-bind=gn python -u -m clip_train.main \
    --train-data "/work/dlclarge1/sinanid-VLM-scaling-law/scaling_studies_vlm/pre_training_dataset/laion400m/train_data/{00000000..00012136}.tar" \
    --local-loss \
    --gather-with-grad \
    --dataset-type webdataset \
    --precision amp \
    --report-to wandb \
    --wandb-project-name 'scan_space_biased_space_0_5' \
    --model "local-dir:results/optimizer=scan_space_final_biased_alpha_0_5/seed=94/model_cfg/config_${config_ID}" \
    --train-num-samples "${TRAIN_NUM_SAMPLES}" \
    --epochs ${EPOCHS} \
    --name "train_config${config_ID}" \
    --logs "/work/dlclarge1/sinanid-VLM-scaling-law/scaling_studies_vlm/train/results/optimizer=scan_space_final_biased_alpha_0_5/seed=94/individual_training_logs" \
    --seed 94 \
    --batch-size 1024 \
    --workers 8 \
    --pipeline-id "${config_ID}" \
    --lr "${LR}" \
    --wd "${WD}" \
    --warmup-fraction "${WARMUP}" \
    --beta1 "${BETA1}" \
    --beta2 "${BETA2}" \
    --eps "${EPS}"

end=`date +%s`
echo "Ended at $(date)";
echo "Job execution complete."
