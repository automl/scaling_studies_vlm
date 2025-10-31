#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=openclip_pretrain
#SBATCH -p mlhiwidlc_gpu-rtx2080

#SBATCH --output logs/%x-%A.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively. short: -o logs/%x-%A-job_name.out
#SBATCH --error logs/%x-%A.err    # STDERR  short: -e logs/%x-%A-job_name.out

source ~/miniconda3/bin/activate
conda activate vlm_scaling_law

echo "Started at $(date)";

# Running the job
start=`date +%s`

# A few SLURM variables
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=12802

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

cd /shared/open_clip
export PYTHONPATH="$PYTHONPATH:$PWD/src"
srun --cpu_bind=v --accel-bind=gn python -m open_clip_train.main \
    --train-data="../pre_training_dataset/laion400m/shards/{00000000..00000282}.tar" \
    --batch-size=64 \
    --epochs=1 \
    --workers=6 \
    --model ViT-S-16 \
    --seed 0 \
    --train-num-samples 1754784 \
    --local-loss \
    --gather-with-grad

end=`date +%s`
runtime=$((end-start))

echo Job execution complete.
