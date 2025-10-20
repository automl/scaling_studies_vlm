#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080
#SBATCH --job-name=net_speed_test
#SBATCH --output=logs/%x-%A.out
#SBATCH --error=logs/%x-%A.err

echo "Running on node: $(hostname)"
echo "Started at $(date)"

curl -o /dev/null -w "Time: %{time_total}s, Speed: %{speed_download} bytes/sec\n" http://ipv4.download.thinkbroadband.com/2GB.zip

echo "Finished at $(date)"
