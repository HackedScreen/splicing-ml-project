#!/usr/bin/env bash
# Stage 1a: Fetch the authoritative run table for our selected dataset from ENA.
#
# Dataset: E-MTAB-1147 (Zarnack et al., 2013) — HNRNPC knockdown vs control in
# HeLa cells. Paired-end RNA-seq, 2 conditions (Control, HNRNPC knockdown "KD1")
# x 2 biological replicates each (a second knockdown construct "KD2" is also
# present in the study and can serve as an independent validation set later).
# ENA study accession: ERP001458. Same runs are also fetchable via NCBI SRA's
# prefetch/fasterq-dump using the identical ERR run accessions (ENA and SRA
# mirror each other for INSDC-registered runs).
#
# We pull the table rather than hardcoding run accessions so the run-to-condition
# mapping is independently verifiable before we spend bandwidth downloading.
set -euo pipefail

OUTDIR="$(dirname "$0")/../data/reference"
mkdir -p "$OUTDIR"
OUTFILE="$OUTDIR/E-MTAB-1147_runs.tsv"

curl -sS "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=ERP001458&result=read_run&fields=run_accession,sample_title,library_layout,read_count,fastq_ftp&format=tsv" \
  -o "$OUTFILE"

echo "Saved run table to $OUTFILE"
echo
column -t -s$'\t' "$OUTFILE"
echo
echo "Next: inspect sample_title above, confirm which run_accessions are"
echo "Control rep1/rep2 and KD1 rep1/rep2, then pass them to 01b_download_fastq.sh"
