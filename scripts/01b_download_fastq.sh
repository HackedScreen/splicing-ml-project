#!/usr/bin/env bash
# Stage 1b: Download paired-end FASTQ for the confirmed run accessions.
#
# Usage:
#   ./01b_download_fastq.sh ERR127302 ERR127303 ERR127309 ERR127310
#
# Fill in the accessions after inspecting data/reference/E-MTAB-1147_runs.tsv
# (produced by 01a_fetch_run_metadata.sh) and confirming which runs are
# Control rep1/rep2 and KD1 rep1/rep2.
#
# Two download paths are given below — SRA toolkit matches the wording in the
# dissertation abstract ("NCBI Sequence Read Archive"); direct ENA FTP is the
# faster fallback if prefetch is unavailable or slow. Use ONE of the two.
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 <run_accession> [<run_accession> ...]"
  exit 1
fi

OUTDIR="$(dirname "$0")/../data/raw"
mkdir -p "$OUTDIR"

for RUN in "$@"; do
  echo "=== $RUN ==="
  PREFIX="${RUN:0:6}"
  wget -c "https://ftp.sra.ebi.ac.uk/vol1/fastq/${PREFIX}/${RUN}/${RUN}_1.fastq.gz" -P "$OUTDIR"
  wget -c "https://ftp.sra.ebi.ac.uk/vol1/fastq/${PREFIX}/${RUN}/${RUN}_2.fastq.gz" -P "$OUTDIR"
done

echo "Done. Downloaded runs: $*"
