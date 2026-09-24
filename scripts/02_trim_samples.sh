#!/usr/bin/env bash
# =============================================================================
# scripts/02_trim_samples.sh
#
# Stage 2: Quality and adapter trimming using Trimmomatic PE.
# Processes Control (ERR127302, ERR127303) and KD1 (ERR127304, ERR127305).
# =============================================================================

set -euo pipefail

source /home/brooklyn/miniforge3/etc/profile.d/conda.sh
conda activate splicing_project

BASE_DIR="/home/brooklyn/splicing-ml-project"
RAW_DIR="$BASE_DIR/data/raw"
TRIM_DIR="$BASE_DIR/data/trimmed"
QC_DIR="$BASE_DIR/results/qc"
mkdir -p "$TRIM_DIR" "$QC_DIR"

ADAPTER_FA="/home/brooklyn/miniforge3/envs/splicing_project/share/trimmomatic/adapters/TruSeq3-PE.fa"
NTHREADS=$(nproc)

RUNS=("$@")
if [ ${#RUNS[@]} -eq 0 ]; then
  RUNS=("ERR127302" "ERR127303" "ERR127304" "ERR127305")
fi

for RUN in "${RUNS[@]}"; do
  R1_IN="$RAW_DIR/${RUN}_1.fastq.gz"
  R2_IN="$RAW_DIR/${RUN}_2.fastq.gz"

  if [ ! -f "$R1_IN" ] || [ ! -f "$R2_IN" ]; then
    echo "Raw files for $RUN not present in $RAW_DIR, skipping."
    continue
  fi
  R1_PAIRED="$TRIM_DIR/${RUN}_1_paired.fastq.gz"
  R1_UNPAIRED="$TRIM_DIR/${RUN}_1_unpaired.fastq.gz"
  R2_PAIRED="$TRIM_DIR/${RUN}_2_paired.fastq.gz"
  R2_UNPAIRED="$TRIM_DIR/${RUN}_2_unpaired.fastq.gz"
  TRIM_LOG="$QC_DIR/${RUN}_trimmomatic.log"

  echo "=== [Stage 2] Running Trimmomatic on $RUN ==="
  echo "Start: $(date)"
  trimmomatic PE -threads "$NTHREADS" -phred33 \
    "$R1_IN" "$R2_IN" \
    "$R1_PAIRED" "$R1_UNPAIRED" \
    "$R2_PAIRED" "$R2_UNPAIRED" \
    "ILLUMINACLIP:$ADAPTER_FA:2:30:10" \
    LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15 MINLEN:36 2>&1 | tee "$TRIM_LOG"
  echo "Trimming complete for $RUN: $(date)"
done

echo "=== ALL TRIMMING COMPLETE ==="
