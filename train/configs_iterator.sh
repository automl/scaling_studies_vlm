#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH -p testdlc2_l40s
#SBATCH --job-name=individual_scripts_builder

export CUDA_VISIBLE_DEVICES=0,1,2,3

srun --gpus-per-task=1 --cpu-bind=cores python -u configs_iterator.py \
    --root_dir results/optimizer=experimental/seed=2 \
    --seed 2 \
    --partition testdlc2_l40s \
    --worker_id $SLURM_PROCID \
    --num_workers $SLURM_NTASKS