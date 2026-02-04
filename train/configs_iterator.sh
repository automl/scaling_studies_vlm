#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH -p testdlc2_gpu-l40s
#SBATCH --job-name=individual_scripts_builder

#SBATCH --output configs_iterator_logs/%x-%A.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively. short: -o logs/%x-%A-job_name.out
#SBATCH --error configs_iterator_logs/%x-%A.err    # STDERR  short: -e logs/%x-%A-job_name.out

source ~/miniconda3/bin/activate
conda activate test_scaling_laws

echo "Started at $(date)";

# Running the job
start=`date +%s`
cd /work/dlclarge1/sinanid-VLM-scaling-law/scaling_studies_vlm/train/
srun -n $SLURM_NTASKS --gpus-per-task=1 \
     --gpu-bind=single:1 bash -lc '
python -u configs_iterator.py \
  --root_dir results/optimizer=gs_ViT-Tiny-32/seed=42 \
  --seed 42 \
  --worker_id $SLURM_PROCID \
  --num_workers $SLURM_NTASKS
'

end=`date +%s`
runtime=$((end-start))

echo "Ended at $(date)";
echo Job execution complete.
