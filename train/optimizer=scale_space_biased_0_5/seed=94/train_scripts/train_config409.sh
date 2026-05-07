#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=train_config409
#SBATCH -p accelerated
#SBATCH --time=24:00:00

#SBATCH --output logs/%x-%A.out
#SBATCH --error logs/%x-%A.err

source /hkfs/work/workspace/scratch/fr_ds677-vlm_scan/env/miniconda3/etc/profile.d/conda.sh && conda activate test_scaling_laws


echo "Started at $(date)";

start=`date +%s`

echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

export CUDA_VISIBLE_DEVICES=0,1
export MASTER_PORT=$((12000 + SLURM_ARRAY_TASK_ID % 20000))

# first node to be used as the master node in multi-node distributed setup
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr


COMMAND_DIR="/hkfs/work/workspace/scratch/fr_ds677-vlm_scan/repo/scaling_studies_vlm/train"
TRAIN_DIR="/hkfs/work/workspace/scratch/fr_ds677-vlm_scan/study_results/train"

export PYTHONPATH="${COMMAND_DIR}:${PYTHONPATH}"
cd "${TRAIN_DIR}"


export WANDB_DIR=${TRAIN_DIR}/scan_space_final_biased_alpha_0_5

srun --cpu-bind=v --accel-bind=gn python -u -m clip_train.main \
    --train-data "/hkfs/work/workspace/scratch/fr_ds677-vlm_scan/pre_training_data/train_data/{00000000..00012136}.tar" \
    --val-data "/hkfs/work/workspace/scratch/fr_ds677-vlm_scan/pre_training_data/validation_data/{00000000..00000015}.tar" \
    --test-data "/hkfs/work/workspace/scratch/fr_ds677-vlm_scan/pre_training_data/test_data/{00000000..00000015}.tar" \
    --local-loss \
    --gather-with-grad \
    --dataset-type webdataset \
    --precision amp \
    --report-to wandb \
    --wandb-project-name "scan_space_final_biased_alpha_0_5" \
    --model local-dir:results/optimizer=scan_space_final_biased_alpha_0_5/seed=94/model_cfg/config_409 \
    --train-num-samples 5827509 \
    --epochs 2 \
    --name train_config409 \
    --logs results/optimizer=scan_space_final_biased_alpha_0_5/seed=94/individual_training_logs \
    --seed 94 \
    --batch-size 1024 \
    --pipeline-id 409 \
    --lr 0.00026048062136396766 \
    --wd 1.4183568964654114e-05 \
    --warmup-fraction 0.7151464223861694 \
    --beta1 0.9233787655830383 \
    --beta2 0.9977068901062012 \
    --eps 3.482156429868155e-08 \
    --workers 8

end=`date +%s`
runtime=$((end-start))

echo "Ended at $(date)";
echo "Job execution complete."
