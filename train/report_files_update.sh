#!/bin/bash
#SBATCH --job-name=update_reports
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p alldlc_gpu-rtx3080
#SBATCH --gres=gpu:1
#SBATCH --output=report_logs/%x-%j.out
#SBATCH --error=report_logs/%x-%j.err

set -euo pipefail

source ~/miniconda3/bin/activate
conda activate test_scaling_laws

echo "Started at $(date)"

python -u report_files_update.py

echo "Ended at $(date)"
