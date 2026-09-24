#!/usr/bin/env bash
set -euo pipefail

source /home/brooklyn/miniforge3/etc/profile.d/conda.sh
conda activate splicing_project
cd /home/brooklyn/splicing-ml-project

echo "=== Disk Space ==="
df -h /
df -h /mnt/c

echo "=== BAM Read Counts ==="
for bam in results/alignment/ERR127302.sorted.bam results/alignment/ERR127303.sorted.bam results/alignment/Control_rep1.sorted.bam results/alignment/Control_rep2.sorted.bam; do
    echo "--- $bam ---"
    samtools idxstats "$bam" | cut -f 3 | paste -sd+ - | bc
done
