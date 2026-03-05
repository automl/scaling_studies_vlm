#!/bin/bash
set -euo pipefail

TAR_PATH="/p/data1/datasets/scan/openclip/train/results/optimizer=scan_space_final_biased_alpha_0_5/seed=73/individual_training_logs.tar.gz"
DEST_DIR="/work/dlclarge1/sinanid-VLM-scaling-law/results_from_juwels"

SSH_OPTS="-o BatchMode=yes -o PreferredAuthentications=publickey -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -i $HOME/.ssh/id_ed25519_aadlogin"

mkdir -p "$DEST_DIR"

ssh $SSH_OPTS aadlogin bash -lc "
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate pyuftp
  export UFTP_USER=sinani1
  export UFTP_AUTH_URL=https://uftp.fz-juelich.de/UFTP_Auth/rest/auth/JUDAC:
  export UFTP_KEY=\$HOME/.uftp/id_uftp

  pyuftp cp -v --user \$UFTP_USER --identity \$UFTP_KEY \
    \"\$UFTP_AUTH_URL$TAR_PATH\" -
" > "$DEST_DIR/individual_training_logs.tar.gz"
