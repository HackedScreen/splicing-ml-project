#!/usr/bin/env bash
# Stage 1c: Download the reference genome + annotation, pinned to a specific
# Ensembl release for reproducibility (never point a script at "latest").
set -euo pipefail

RELEASE=116   # Ensembl release 116, GRCh38.p14 — pinned 2026-08-31
OUTDIR="$(dirname "$0")/../data/reference"
mkdir -p "$OUTDIR"

FASTA="Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz"
GTF="Homo_sapiens.GRCh38.${RELEASE}.gtf.gz"

wget -c "https://ftp.ensembl.org/pub/release-${RELEASE}/fasta/homo_sapiens/dna/${FASTA}" -P "$OUTDIR"
wget -c "https://ftp.ensembl.org/pub/release-${RELEASE}/gtf/homo_sapiens/${GTF}" -P "$OUTDIR"

gunzip -k "$OUTDIR/$FASTA"
gunzip -k "$OUTDIR/$GTF"

echo "Reference genome + GTF ready in $OUTDIR (Ensembl release $RELEASE)"
echo "NOTE: primary assembly FASTA is ~3 GB compressed; HISAT2 index build in"
echo "stage 3 needs roughly 8-16 GB RAM and can take a few hours on a laptop."
