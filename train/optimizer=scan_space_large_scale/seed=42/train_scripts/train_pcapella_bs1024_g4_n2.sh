#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=train_pcapella_bs1024_g4_n2
#SBATCH -p capella
#SBATCH --time=07:00:00
#SBATCH --array=1-80

#SBATCH --output array_job_logs/%x-%A_%a.out
#SBATCH --error array_job_logs/%x-%A_%a.err

module load Anaconda3
source $EBROOTANACONDA3/etc/profile.d/conda.sh
conda activate /data/cat/ws/dosi100i-scan_bench_suite/vlm_env


echo "Started at $(date)";
start=`date +%s`

export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=$((12000 + SLURM_ARRAY_TASK_ID % 20000))

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr


COMMAND_DIR="/data/cat/ws/dosi100i-scan_bench_suite/vlm_repo/scaling_studies_vlm/train"
TRAIN_DIR="/data/cat/ws/dosi100i-scan_bench_suite/runs/train"

export PYTHONPATH="${COMMAND_DIR}:${PYTHONPATH}"
cd "${TRAIN_DIR}"

export WANDB_DIR=${TRAIN_DIR}/scan_space_large_space

CONFIG_TXT="/data/cat/ws/dosi100i-scan_bench_suite/runs/train/results/optimizer=scan_space_large_scale/seed=42/train_scripts/train_pcapella_bs1024_g4_n2.txt"

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
    --train-data "/data/cat/ws/dosi100i-scan_bench_suite/vlm_data/train_data/{00000000..00026286}.tar" \
    --local-loss \
    --gather-with-grad \
    --dataset-type webdataset \
    --precision amp \
    --model "local-dir:results/optimizer=scan_space_large_scale/seed=42/model_cfg/config_${config_ID}" \
    --train-num-samples "${TRAIN_NUM_SAMPLES}" \
    --epochs ${EPOCHS} \
    --name "train_config${config_ID}" \
    --logs "/data/cat/ws/dosi100i-scan_bench_suite/runs/train/results/optimizer=scan_space_large_scale/seed=42/individual_training_logs" \
    --seed 42 \
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
