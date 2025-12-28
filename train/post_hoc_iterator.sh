#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=post_hoc_iterator
#SBATCH -p testdlc_gpu-rtx2080

#SBATCH --output post_hoc_logs/%x-%A.out
#SBATCH --error post_hoc_logs/%x-%A.err

source ~/miniconda3/bin/activate
conda activate test_scaling_laws

echo "Started at $(date)";

# Running the job
start=`date +%s`

# A few SLURM variables
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

python post_hoc_iterator.py

end=`date +%s`
runtime=$((end-start))

echo "Ended at $(date)";
echo Job execution complete.
