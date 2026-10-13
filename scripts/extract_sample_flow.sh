#!/bin/bash
set -e
cd /home/brooklyn/splicing-ml-project

echo "================================================================================"
echo "1. ENA READ COUNTS FROM data/reference/E-MTAB-1147_runs.tsv"
echo "--------------------------------------------------------------------------------"
sha256sum data/reference/E-MTAB-1147_runs.tsv
cat data/reference/E-MTAB-1147_runs.tsv

echo ""
echo "================================================================================"
echo "2. TRIMMOMATIC LOGS: INPUT READ PAIRS / BOTH SURVIVING"
echo "--------------------------------------------------------------------------------"
for log in \
  results/alignment/Control_rep1_trimmomatic.log \
  results/alignment/Control_rep2_trimmomatic.log \
  results/alignment/ERR127302_trimmomatic.log \
  results/alignment/ERR127303_trimmomatic.log \
  results/alignment/ERR127304_trimmomatic.log \
  results/alignment/ERR127305_trimmomatic.log; do
    echo "--- LOG: $log ---"
    sha256sum "$log"
    grep -E "Input Read Pairs" "$log" || grep -E "Both Surviving" "$log"
done

echo ""
echo "================================================================================"
echo "3. HISAT2 LOGS: OVERALL ALIGNMENT RATE"
echo "--------------------------------------------------------------------------------"
for log in \
  results/alignment/Control_rep1_hisat2.log \
  results/alignment/Control_rep2_hisat2.log \
  results/alignment/ERR127302_hisat2.log \
  results/alignment/ERR127303_hisat2.log \
  results/alignment/ERR127304_hisat2.log \
  results/alignment/ERR127305_hisat2.log; do
    echo "--- LOG: $log ---"
    sha256sum "$log"
    grep -E "overall alignment rate" "$log"
done
