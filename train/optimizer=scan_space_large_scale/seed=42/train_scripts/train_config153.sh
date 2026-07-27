#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=train_config153
#SBATCH -p capella
#SBATCH --time=05:00:00

#SBATCH --output logs/%x-%A.out
#SBATCH --error logs/%x-%A.err

module load Anaconda3
source $EBROOTANACONDA3/etc/profile.d/conda.sh
conda activate /data/cat/ws/dosi100i-scan_bench_suite/vlm_env


echo "Started at $(date)";

start=`date +%s`

echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=$((12000 + SLURM_ARRAY_TASK_ID % 20000))

# first node to be used as the master node in multi-node distributed setup
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr


COMMAND_DIR="/data/cat/ws/dosi100i-scan_bench_suite/vlm_repo/scaling_studies_vlm/train"
TRAIN_DIR="/data/cat/ws/dosi100i-scan_bench_suite/runs/train"

export PYTHONPATH="${COMMAND_DIR}:${PYTHONPATH}"
cd "${TRAIN_DIR}"


export WANDB_DIR=${TRAIN_DIR}/scan_space_large_scale

srun --cpu-bind=v --accel-bind=gn python -u -m clip_train.main \
    --train-data "/data/cat/ws/dosi100i-scan_bench_suite/vlm_data/train_data/{00000000..00026286}.tar" \
    --local-loss \
    --gather-with-grad \
    --dataset-type webdataset \
    --precision amp \
    --model local-dir:results/optimizer=scan_space_large_scale/seed=42/model_cfg/config_153 \
    --train-num-samples 11248079 \
    --epochs 9 \
    --name train_config153 \
    --logs results/optimizer=scan_space_large_scale/seed=42/individual_training_logs \
    --seed 42 \
    --batch-size 1024 \
    --pipeline-id 153 \
    --lr 0.00020938529632985592 \
    --wd 0.00018436874961480498 \
    --warmup-fraction 0.6940398812294006 \
    --beta1 0.9161475896835327 \
    --beta2 0.9747682809829712 \
    --eps 1.3671243692670032e-08 \
    --workers 8

end=`date +%s`
runtime=$((end-start))

echo "Ended at $(date)";
echo "Job execution complete."
