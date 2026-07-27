#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=train_config28
#SBATCH -p capella
#SBATCH --time=08:00:00

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
    --model local-dir:results/optimizer=scan_space_large_scale/seed=42/model_cfg/config_28 \
    --train-num-samples 11401711 \
    --epochs 6 \
    --name train_config28 \
    --logs results/optimizer=scan_space_large_scale/seed=42/individual_training_logs \
    --seed 42 \
    --batch-size 1024 \
    --pipeline-id 28 \
    --lr 0.0018902275478467345 \
    --wd 0.05218505859375 \
    --warmup-fraction 0.3320428133010864 \
    --beta1 0.918353796005249 \
    --beta2 0.9713658690452576 \
    --eps 1.670760241267999e-08 \
    --workers 8

end=`date +%s`
runtime=$((end-start))

echo "Ended at $(date)";
echo "Job execution complete."
