#!/bin/bash
set -euo pipefail

SRC_DIR="/work/dlclarge1/sinanid-VLM-scaling-law/scaling_studies_vlm/pre_training_dataset/laion400m/train_data"
DEST_DIR="/p/data1/mmlaion/relaion_400m_subset/67m_samples/train_data"

START_ID="${1:-5}"
END_ID="${2:-15}"
PAR=8

SSH_OPTS="-o BatchMode=yes -o PreferredAuthentications=publickey -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -i $HOME/.ssh/id_ed25519_aadlogin"

stream_one() {
  local fpath="$1"
  local fname
  fname="$(basename "$fpath")"
  echo "Streaming $fname ..."

  ssh $SSH_OPTS aadlogin bash -lc "
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate pyuftp
    export UFTP_USER=sinani1
    export UFTP_AUTH_URL=https://uftp.fz-juelich.de/UFTP_Auth/rest/auth/JUDAC:
    export UFTP_KEY=\$HOME/.uftp/id_uftp
    pyuftp cp -v --user \$UFTP_USER --identity \$UFTP_KEY - \"\$UFTP_AUTH_URL$DEST_DIR/$fname\"
  " < "$fpath"
}

pids=()

for ((i=START_ID; i<=END_ID; i++)); do
  fname="$(printf "%08d.tar" "$i")"
  fpath="$SRC_DIR/$fname"
  [[ -f "$fpath" ]] || { echo "SKIP: $fpath"; continue; }

  stream_one "$fpath" &
  pids+=("$!")

  if (( ${#pids[@]} >= PAR )); then
    wait "${pids[0]}"
    pids=("${pids[@]:1}")
  fi
done

for pid in "${pids[@]}"; do
  wait "$pid"
done
