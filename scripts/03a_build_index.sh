#!/usr/bin/env bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate splicing_project

REF_DIR="$(cd "$(dirname "$0")/.." && pwd)/data/reference"
FASTA="$REF_DIR/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
SS="$REF_DIR/splicesites.tsv"
EXON="$REF_DIR/exons.tsv"
INDEX_PREFIX="$REF_DIR/grch38_index"

NTHREADS=$(nproc)

echo "=== Building HISAT2 Index with $NTHREADS threads ==="
echo "Start time: $(date)"
START_SEC=$(date +%s)

hisat2-build -p "$NTHREADS" "$FASTA" "$INDEX_PREFIX"

END_SEC=$(date +%s)
WALL_TIME=$((END_SEC - START_SEC))
echo "=== Index Build Completed ==="
echo "End time: $(date)"
echo "Total Wall Time: $WALL_TIME seconds"