#!/usr/bin/env bash
set -euo pipefail

source /home/brooklyn/miniforge3/etc/profile.d/conda.sh
conda activate splicing_project
cd /home/brooklyn/splicing-ml-project

mkdir -p results/alignment

echo "=== Subsampling Control_rep1 ==="
START1=$(date +%s)
samtools view -@ 4 -s 42.469 -b -o results/alignment/Control_rep1_downsampled.sorted.bam results/alignment/Control_rep1.sorted.bam
samtools index -@ 4 results/alignment/Control_rep1_downsampled.sorted.bam
END1=$(date +%s)
echo "Control_rep1 downsampled in $((END1 - START1)) seconds."

echo "=== Subsampling Control_rep2 ==="
START2=$(date +%s)
samtools view -@ 4 -s 42.469 -b -o results/alignment/Control_rep2_downsampled.sorted.bam results/alignment/Control_rep2.sorted.bam
samtools index -@ 4 results/alignment/Control_rep2_downsampled.sorted.bam
END2=$(date +%s)
echo "Control_rep2 downsampled in $((END2 - START2)) seconds."

echo "=== Downsampled BAM verification ==="
samtools quickcheck -v results/alignment/Control_rep1_downsampled.sorted.bam results/alignment/Control_rep2_downsampled.sorted.bam
echo "Control_rep1_downsampled mapped:"
samtools idxstats results/alignment/Control_rep1_downsampled.sorted.bam | cut -f 3 | paste -sd+ - | bc
echo "Control_rep2_downsampled mapped:"
samtools idxstats results/alignment/Control_rep2_downsampled.sorted.bam | cut -f 3 | paste -sd+ - | bc

ls -lh results/alignment/Control_rep*_downsampled.sorted.bam
