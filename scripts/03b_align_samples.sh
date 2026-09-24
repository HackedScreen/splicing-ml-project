#!/usr/bin/env bash
set -euo pipefail

source /home/brooklyn/miniforge3/etc/profile.d/conda.sh
conda activate splicing_project

INDEX="/home/brooklyn/splicing-ml-project/data/reference/grch38_index"
TRIM_DIR="/home/brooklyn/splicing-ml-project/data/trimmed"
ALIGN_DIR="/home/brooklyn/splicing-ml-project/results/alignment"
mkdir -p "$ALIGN_DIR"

NTHREADS=$(nproc)

RUNS=("$@")
if [ ${#RUNS[@]} -eq 0 ]; then
  RUNS=("ERR127302" "ERR127303" "ERR127304" "ERR127305")
fi

SS="/home/brooklyn/splicing-ml-project/data/reference/splicesites.tsv"

for RUN in "${RUNS[@]}"; do
  R1="$TRIM_DIR/${RUN}_1_paired.fastq.gz"
  R2="$TRIM_DIR/${RUN}_2_paired.fastq.gz"

  if [ ! -f "$R1" ] || [ ! -f "$R2" ]; then
    echo "Trimmed input for $RUN not present in $TRIM_DIR, skipping."
    continue
  fi
  SAM="$ALIGN_DIR/${RUN}.sam"
  BAM="$ALIGN_DIR/${RUN}.sorted.bam"
  LOG="$ALIGN_DIR/${RUN}_hisat2.log"

  echo "=== [Stage 3] Aligning $RUN with HISAT2 ==="
  echo "Start: $(date)"
  hisat2 -x "$INDEX" -1 "$R1" -2 "$R2" -p "$NTHREADS" --dta --known-splicesite-infile "$SS" -S "$SAM" 2> "$LOG"
  echo "Alignment log for $RUN:"
  cat "$LOG"

  echo "=== Converting SAM to sorted BAM ==="
  samtools sort -@ "$NTHREADS" -o "$BAM" "$SAM"
  samtools index "$BAM"
  rm -f "$SAM"
  echo "Generated $BAM"
done

echo "=== ALL ALIGNMENTS COMPLETE ==="
