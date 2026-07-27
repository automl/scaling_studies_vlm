#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=train_config702
#SBATCH -p capella
#SBATCH --time=24:00:00

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
    --val-data "/data/cat/ws/dosi100i-scan_bench_suite/vlm_data/validation_data/{00000000..00000015}.tar" \
    --test-data "/data/cat/ws/dosi100i-scan_bench_suite/vlm_data/test_data/{00000000..00000015}.tar" \
    --local-loss \
    --gather-with-grad \
    --dataset-type webdataset \
    --precision amp \
    --report-to wandb \
    --wandb-project-name "scan_space_large_scale" \
    --model local-dir:results/optimizer=scan_space_large_scale/seed=42/model_cfg/config_702 \
    --train-num-samples 11338608 \
    --epochs 10 \
    --name train_config702 \
    --logs results/optimizer=scan_space_large_scale/seed=42/individual_training_logs \
    --seed 42 \
    --batch-size 1024 \
    --pipeline-id 702 \
    --lr 0.001205988577567041 \
    --wd 0.0004184268764220178 \
    --warmup-fraction 0.5392481088638306 \
    --beta1 0.9123291373252869 \
    --beta2 0.9965324401855469 \
    --eps 1.1325150595098421e-08 \
    --workers 8

end=`date +%s`
runtime=$((end-start))

echo "Ended at $(date)";
echo "Job execution complete."
