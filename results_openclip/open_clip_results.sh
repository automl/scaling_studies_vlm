#!/bin/bash
#SBATCH --job-name=open_clip_results
#SBATCH --partition=mlhiwidlc_gpu-rtx2080
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1

#SBATCH --output logs/%x-%A.out
#SBATCH --error logs/%x-%A.err

source ~/miniconda3/bin/activate
conda activate vlm_scaling_law

echo "Started at $(date)";

# Running the job
start=`date +%s`

# A few SLURM variables
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

python -m jupyter nbconvert --to notebook --inplace --execute open_clip.ipynb --ExecutePreprocessor.timeout=-1

end=`date +%s`
runtime=$((end-start))

echo Job execution complete.
