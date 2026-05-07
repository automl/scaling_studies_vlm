#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=train_config715
#SBATCH -p alldlc_gpu-rtx3080
#SBATCH --time=24:00:00

#SBATCH --output logs/%x-%A.out
#SBATCH --error logs/%x-%A.err

source ~/miniconda3/bin/activate && conda activate test_scaling_laws
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker,virbr
export NCCL_DEBUG=WARN


echo "Started at $(date)";

start=`date +%s`

echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=$((12000 + SLURM_ARRAY_TASK_ID % 20000))

# first node to be used as the master node in multi-node distributed setup
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr


COMMAND_DIR="/work/dlclarge1/sinanid-VLM-scaling-law/repository/scaling_studies_vlm/train"
TRAIN_DIR="/work/dlclarge1/sinanid-VLM-scaling-law/scaling_studies_vlm/train"

export PYTHONPATH="${COMMAND_DIR}:${PYTHONPATH}"
cd "${TRAIN_DIR}"


export WANDB_DIR=${TRAIN_DIR}/scan_space_final_biased_space_alpha_0_5

srun --cpu-bind=v --accel-bind=gn python -u -m clip_train.main \
    --train-data "/work/dlclarge1/sinanid-VLM-scaling-law/scaling_studies_vlm/pre_training_dataset/laion400m/train_data/{00000000..00012136}.tar" \
    --val-data "/work/dlclarge1/sinanid-VLM-scaling-law/scaling_studies_vlm/pre_training_dataset/laion400m/validation_data/{00000000..00000015}.tar" \
    --test-data "/work/dlclarge1/sinanid-VLM-scaling-law/scaling_studies_vlm/pre_training_dataset/laion400m/test_data/{00000000..00000015}.tar" \
    --local-loss \
    --gather-with-grad \
    --dataset-type webdataset \
    --precision amp \
    --report-to wandb \
    --wandb-project-name "scan_space_final_biased_space_alpha_0_5" \
    --model local-dir:results/optimizer=scan_space_final_biased_alpha_0_5/seed=94/model_cfg/config_715 \
    --train-num-samples 2115604 \
    --epochs 2 \
    --name train_config715 \
    --logs results/optimizer=scan_space_final_biased_alpha_0_5/seed=94/individual_training_logs \
    --seed 94 \
    --batch-size 256 \
    --pipeline-id 715 \
    --lr 0.005875131580978632 \
    --wd 0.00023311439144890755 \
    --warmup-fraction 0.7283106446266174 \
    --beta1 0.9133076667785645 \
    --beta2 0.98757404088974 \
    --eps 6.465618440643084e-08 \
    --workers 8

end=`date +%s`
runtime=$((end-start))

echo "Ended at $(date)";
echo "Job execution complete."
