#!/usr/bin/env bash
set -euo pipefail

source /home/brooklyn/miniforge3/etc/profile.d/conda.sh
conda activate splicing_project
cd /home/brooklyn/splicing-ml-project

mkdir -p results/splicing_lane_control/tmp

echo "=== [Step 1] Demultiplexing Control_rep1 into ERR127306 and ERR127307 ==="
START_SPLIT=$(date +%s)
python scripts/split_lanes.py
END_SPLIT=$(date +%s)
echo "Splitting completed in $((END_SPLIT - START_SPLIT)) seconds."

echo "=== [Step 2] Indexing demultiplexed BAMs ==="
samtools index -@ 4 results/alignment/ERR127306.sorted.bam
samtools index -@ 4 results/alignment/ERR127307.sorted.bam

echo "=== [Step 3] Verifying BAM read counts ==="
samtools quickcheck -v results/alignment/ERR127306.sorted.bam results/alignment/ERR127307.sorted.bam
echo "ERR127306 mapped reads:"
samtools idxstats results/alignment/ERR127306.sorted.bam | cut -f 3 | paste -sd+ - | bc
echo "ERR127307 mapped reads:"
samtools idxstats results/alignment/ERR127307.sorted.bam | cut -f 3 | paste -sd+ - | bc

echo "=== [Step 4] Setting up rMATS Lane Comparison ==="
echo -n "results/alignment/ERR127306.sorted.bam" > results/splicing_lane_control/b1.txt
echo -n "results/alignment/ERR127307.sorted.bam" > results/splicing_lane_control/b2.txt

echo "=== [Step 5] Running rMATS (ERR127306 vs ERR127307) ==="
START_RMATS=$(date +%s)
rmats.py --b1 results/splicing_lane_control/b1.txt \
         --b2 results/splicing_lane_control/b2.txt \
         --gtf data/reference/Homo_sapiens.GRCh38.116.gtf \
         -t paired --readLength 72 --nthread 4 \
         --od results/splicing_lane_control/ \
         --tmp results/splicing_lane_control/tmp
END_RMATS=$(date +%s)
echo "=== rMATS Lane Control execution completed in $((END_RMATS - START_RMATS)) seconds ==="
cat results/splicing_lane_control/summary.txt
