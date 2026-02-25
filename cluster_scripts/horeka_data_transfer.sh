#!/bin/bash
#SBATCH --job-name=meta_to_horeka_testdata
#SBATCH --output=transfer_logs/%x_%j.out
#SBATCH --error=transfer_logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition dev_cpuonly

set -euo pipefail

mkdir -p transfer_logs

SRC_DIR="/work/dlclarge1/sinanid-VLM-scaling-law/scaling_studies_vlm/pre_training_dataset/laion400m/train_data"
DEST_DIR="/hkfs/work/workspace/scratch/fr_ds677-vlm_scan/pre_training_data/train_data"

START=0
END=12136

PARALLEL=4

echo "Start: $(date)"
echo "Source: meta:${SRC_DIR}"
echo "Dest:   ${DEST_DIR}"
echo "Range:  $(printf '%08d' "${START}")..$(printf '%08d' "${END}")"
echo "Parallel workers: ${PARALLEL}"

mkdir -p "${DEST_DIR}"

seq "${START}" "${END}" | awk '{printf "%08d.tar\n",$1}' | \
xargs -n 1 -P "${PARALLEL}" -I {} bash -lc '
  set -e
  f="{}"
  rsync -avhP --partial --append-verify "meta:'"${SRC_DIR}"'/${f}" "'"${DEST_DIR}"'/" 2>/dev/null || true
'

echo "Done: $(date)"