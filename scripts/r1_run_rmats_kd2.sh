#!/usr/bin/env bash
set -euo pipefail

source /home/brooklyn/miniforge3/etc/profile.d/conda.sh
conda activate splicing_project
cd /home/brooklyn/splicing-ml-project

mkdir -p results/splicing_kd2_vrl/tmp
echo -n "results/alignment/Control_rep1_downsampled.sorted.bam,results/alignment/Control_rep2_downsampled.sorted.bam" > results/splicing_kd2_vrl/b1.txt
echo -n "results/alignment/ERR127304.sorted.bam,results/alignment/ERR127305.sorted.bam" > results/splicing_kd2_vrl/b2.txt

echo "=== [R1 Step 2B] Running rMATS: Downsampled Control vs KD2 (--variable-read-length) ==="
echo "Start time: $(date)"
START_SEC=$(date +%s)

rmats.py --b1 results/splicing_kd2_vrl/b1.txt \
         --b2 results/splicing_kd2_vrl/b2.txt \
         --gtf data/reference/Homo_sapiens.GRCh38.116.gtf \
         -t paired \
         --readLength 72 \
         --variable-read-length \
         --nthread 4 \
         --od results/splicing_kd2_vrl/ \
         --tmp results/splicing_kd2_vrl/tmp/

END_SEC=$(date +%s)
echo "rMATS KD2 completed in $((END_SEC - START_SEC)) seconds."
echo "End time: $(date)"

python scripts/summarize_rmats_run.py results/splicing_kd2_vrl
