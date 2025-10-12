#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080
#SBATCH --job-name=download_train_data
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G

#SBATCH --output logs/%x-%A.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively. short: -o logs/%x-%A-job_name.out
#SBATCH --error logs/%x-%A.err    # STDERR  short: -e logs/%x-%A-job_name.out

source ~/miniconda3/bin/activate
conda activate vlm_scaling_law

echo "Started at $(date)";

# Running the job
start=`date +%s`

# A few SLURM variables
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

python main.py --data_dir ../pre_training_dataset/laion400m --download_parquet_files True

end=`date +%s`
runtime=$((end-start))

echo Job execution complete.
