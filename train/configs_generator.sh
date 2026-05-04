#!/bin/bash
# Usage: sbatch configs_generator.sh [WORKSPACE_DIR] [CODE_DIR] [HPO_OPTIMIZER] [SEED]
# Example: sbatch configs_generator.sh /work/dlclarge1/sinanid-VLM-scaling-law/scaling_studies_vlm /home/alipourn/scaling_studies_vlm/train constrained_bo 42

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --partition develbooster
#SBATCH --account projectnucleus
#SBATCH --job-name=generator

#SBATCH --output configs_iterator_logs/%x-%A.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively.
#SBATCH --error configs_iterator_logs/%x-%A.err    # STDERR

# Set defaults or use provided arguments


WORKSPACE_DIR="${1:-/p/data1/datasets/scan/openclip}"
COMMAND_DIR="${2:-/p/project1/projectnucleus/alipourgougeh1/scaling_studies_vlm/train}"
HPO_OPTIMIZER="${3:-bo_space_expansion_micro}"
SEED="${4:-42}"
VENV_DIR="${5:-/p/project1/projectnucleus/alipourgougeh1/scaling_studies_vlm/venv/bin/activate}"

echo "WORKSPACE_DIR: $WORKSPACE_DIR"
echo "COMMAND_DIR: $COMMAND_DIR"
echo "HPO_OPTIMIZER: $HPO_OPTIMIZER"
echo "SEED: $SEED"

echo "Started at $(date)";

# Running the job
start=`date +%s`



export PYTHONPATH="${COMMAND_DIR}:${PYTHONPATH}"
export NEPS_MAX_RETRIES_GET_NEXT_TRIAL=1 

srun -n $SLURM_NTASKS --gpus-per-task=1 \
     --gpu-bind=single:1 bash -lc '
cd '"${WORKSPACE_DIR}/train"'
source '"${VENV_DIR}"'
python -u /p/project1/projectnucleus/alipourgougeh1/scaling_studies_vlm/train/configs_generator.py --hpo-optimizer primo --seed '"${SEED}"' --workspace_dir /p/data1/datasets/scan/openclip 
'

end=`date +%s`
runtime=$((end-start))

echo "Ended at $(date)";
echo "Job execution complete. Runtime: ${runtime}s"
