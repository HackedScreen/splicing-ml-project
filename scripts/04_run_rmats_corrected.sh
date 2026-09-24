#!/usr/bin/env bash
set -euo pipefail

source /home/brooklyn/miniforge3/etc/profile.d/conda.sh
conda activate splicing_project
cd /home/brooklyn/splicing-ml-project

B1="results/splicing_corrected/b1.txt"
B2="results/splicing_corrected/b2.txt"
GTF="data/reference/Homo_sapiens.GRCh38.116.gtf"
OD="results/splicing_corrected"
TMP="results/splicing_corrected/tmp"
NTHREADS=4

echo "=== [Stage 4 Corrected] Running rMATS-turbo (Control vs KD1) ==="
echo "Start time: $(date)"
START_SEC=$(date +%s)

rmats.py --b1 "$B1" --b2 "$B2" --gtf "$GTF" -t paired --readLength 72 --nthread "$NTHREADS" --od "$OD" --tmp "$TMP"

END_SEC=$(date +%s)
WALL_TIME=$((END_SEC - START_SEC))
echo "=== rMATS execution completed ==="
echo "End time: $(date)"
echo "Total Wall Time: $WALL_TIME seconds"
