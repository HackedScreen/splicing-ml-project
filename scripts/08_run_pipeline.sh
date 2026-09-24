#!/usr/bin/env bash
# =============================================================================
# scripts/08_run_pipeline.sh
#
# BSc Dissertation Pipeline — Stage 8: KD2 External Validation Pipeline
# Coordinates QC/Trimming, Alignment, rMATS Differential Splicing,
# Feature Extraction, and Robust Model Evaluation on the held-back KD2 replicate.
# =============================================================================

set -euo pipefail

source /home/brooklyn/miniforge3/etc/profile.d/conda.sh
conda activate splicing_project

BASE_DIR="/home/brooklyn/splicing-ml-project"
DATA_DIR="$BASE_DIR/data"
RAW_DIR="$DATA_DIR/raw"
TRIM_DIR="$DATA_DIR/trimmed"
REF_DIR="$DATA_DIR/reference"
RESULTS_DIR="$BASE_DIR/results"
ALIGN_DIR="$RESULTS_DIR/alignment"
RMATS_OD="$RESULTS_DIR/splicing_kd2_validation"
FEAT_DIR="$RESULTS_DIR/features"
MODELS_DIR="$RESULTS_DIR/models"

NTHREADS=2
ADAPTER_FA="/home/brooklyn/miniforge3/envs/splicing_project/share/trimmomatic-0.41-0/adapters/TruSeq3-PE.fa"
INDEX_PREFIX="$REF_DIR/grch38_index"
GTF="$REF_DIR/Homo_sapiens.GRCh38.116.gtf"
FASTA="$REF_DIR/Homo_sapiens.GRCh38.dna.primary_assembly.fa"

mkdir -p "$TRIM_DIR" "$ALIGN_DIR" "$RMATS_OD" "$FEAT_DIR" "$MODELS_DIR"

echo "========================================================"
echo " Starting KD2 External Validation Pipeline"
echo " Date: $(date)"
echo "========================================================"

# -----------------------------------------------------------------------------
# STEP 2: Trimmomatic Quality & Adapter Trimming for KD2
# -----------------------------------------------------------------------------
echo -e "\n=== [Step 2] Running Trimmomatic on KD2 Replicates (ERR127306, ERR127307) ==="
KD2_RUNS=("ERR127306" "ERR127307")

for RUN in "${KD2_RUNS[@]}"; do
  R1_IN="$RAW_DIR/${RUN}_1.fastq.gz"
  R2_IN="$RAW_DIR/${RUN}_2.fastq.gz"
  R1_PAIRED="$TRIM_DIR/${RUN}_1_paired.fastq.gz"
  R1_UNPAIRED="$TRIM_DIR/${RUN}_1_unpaired.fastq.gz"
  R2_PAIRED="$TRIM_DIR/${RUN}_2_paired.fastq.gz"
  R2_UNPAIRED="$TRIM_DIR/${RUN}_2_unpaired.fastq.gz"
  TRIM_LOG="$RESULTS_DIR/qc/${RUN}_trimmomatic.log"

  if [ -f "$R1_PAIRED" ] && [ -f "$R2_PAIRED" ]; then
    echo "Trimmomatic outputs already exist for $RUN, skipping."
  else
    echo "Trimming $RUN..."
    trimmomatic PE -threads "$NTHREADS" -phred33 \
      "$R1_IN" "$R2_IN" \
      "$R1_PAIRED" "$R1_UNPAIRED" \
      "$R2_PAIRED" "$R2_UNPAIRED" \
      "ILLUMINACLIP:$ADAPTER_FA:2:30:10" \
      LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15 MINLEN:36 2>&1 | tee "$TRIM_LOG"
  fi
done

# -----------------------------------------------------------------------------
# STEP 3: HISAT2 Splice-Aware Alignment for KD2
# -----------------------------------------------------------------------------
echo -e "\n=== [Step 3] Running HISAT2 Alignment for KD2 Replicates ==="

for RUN in "${KD2_RUNS[@]}"; do
  R1_PAIRED="$TRIM_DIR/${RUN}_1_paired.fastq.gz"
  R2_PAIRED="$TRIM_DIR/${RUN}_2_paired.fastq.gz"
  SAM="$ALIGN_DIR/${RUN}.sam"
  BAM="$ALIGN_DIR/${RUN}.sorted.bam"
  HISAT_LOG="$ALIGN_DIR/${RUN}_hisat2.log"

  if [ -f "$BAM" ] && [ -f "${BAM}.bai" ]; then
    echo "Sorted BAM already exists for $RUN, skipping."
  else
    echo "Aligning $RUN to $INDEX_PREFIX..."
    hisat2 -x "$INDEX_PREFIX" \
      -1 "$R1_PAIRED" -2 "$R2_PAIRED" \
      -p "$NTHREADS" --dta \
      -S "$SAM" 2> "$HISAT_LOG"

    cat "$HISAT_LOG"

    echo "Sorting and indexing $RUN BAM..."
    samtools sort -@ "$NTHREADS" -o "$BAM" "$SAM"
    samtools index "$BAM"
    rm -f "$SAM"
    echo "Generated $BAM"
  fi
done

# -----------------------------------------------------------------------------
# STEP 4: rMATS Differential Splicing Quantification (Control vs KD2)
# -----------------------------------------------------------------------------
echo -e "\n=== [Step 4] Running rMATS-turbo for Control vs KD2 ==="
B1_FILE="$RMATS_OD/b1.txt"
B2_FILE="$RMATS_OD/b2.txt"

# Group 1: Control (ERR127302, ERR127303)
echo "$ALIGN_DIR/ERR127302.sorted.bam,$ALIGN_DIR/ERR127303.sorted.bam" > "$B1_FILE"
# Group 2: KD2 (ERR127306, ERR127307)
echo "$ALIGN_DIR/ERR127306.sorted.bam,$ALIGN_DIR/ERR127307.sorted.bam" > "$B2_FILE"

if [ -f "$RMATS_OD/SE.MATS.JC.txt" ] && [ -f "$RMATS_OD/RI.MATS.JC.txt" ]; then
  echo "rMATS outputs already exist in $RMATS_OD, skipping."
else
  echo "Executing rmats.py..."
  rmats.py --b1 "$B1_FILE" --b2 "$B2_FILE" \
    --gtf "$GTF" \
    -t paired --readLength 72 \
    --nthread "$NTHREADS" \
    --od "$RMATS_OD" \
    --tmp "$RMATS_OD/tmp"
fi

# -----------------------------------------------------------------------------
# STEP 5: Feature Extraction for KD2 Validation
# -----------------------------------------------------------------------------
echo -e "\n=== [Step 5] Running Feature Extraction (05b_feature_engineering_kd2.py) ==="
python3 "$BASE_DIR/scripts/05b_feature_engineering_kd2.py"

# -----------------------------------------------------------------------------
# STEP 6: Model External Validation Evaluation & Reporting
# -----------------------------------------------------------------------------
echo -e "\n=== [Step 6] Running External Validation Evaluation (08_external_validation_kd2.py) ==="
python3 "$BASE_DIR/scripts/08_external_validation_kd2.py"

echo -e "\n========================================================"
echo " KD2 External Validation Pipeline Successfully Completed!"
echo " Date: $(date)"
echo "========================================================"
