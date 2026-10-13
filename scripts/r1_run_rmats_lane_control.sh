#!/usr/bin/env bash
set -euo pipefail

source /home/brooklyn/miniforge3/etc/profile.d/conda.sh
conda activate splicing_project
cd /home/brooklyn/splicing-ml-project

mkdir -p results/splicing_lane_control_vrl/tmp
echo -n "results/alignment/ERR127306.sorted.bam" > results/splicing_lane_control_vrl/b1.txt
echo -n "results/alignment/ERR127307.sorted.bam" > results/splicing_lane_control_vrl/b2.txt

echo "=== [R1 Step 2C] Running rMATS: Lane Control (ERR127306 vs ERR127307, --variable-read-length) ==="
echo "Start time: $(date)"
START_SEC=$(date +%s)

rmats.py --b1 results/splicing_lane_control_vrl/b1.txt \
         --b2 results/splicing_lane_control_vrl/b2.txt \
         --gtf data/reference/Homo_sapiens.GRCh38.116.gtf \
         -t paired \
         --readLength 72 \
         --variable-read-length \
         --nthread 4 \
         --od results/splicing_lane_control_vrl/ \
         --tmp results/splicing_lane_control_vrl/tmp/

END_SEC=$(date +%s)
echo "rMATS Lane Control completed in $((END_SEC - START_SEC)) seconds."
echo "End time: $(date)"

python scripts/summarize_rmats_run.py results/splicing_lane_control_vrl
