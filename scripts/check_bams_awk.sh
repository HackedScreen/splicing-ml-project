#!/bin/bash
set -e
export PATH=/opt/conda/envs/splicing_project/bin:$PATH
cd "$(cd "$(dirname "$0")/.." && pwd)"
for bam in results/alignment/*.sorted.bam; do
    echo "================================================================================"
    echo "BAM: $bam"
    samtools view -F 0x904 "$bam" | head -1000000 | awk '{n=length($10); if(n==72)a++; if($6~/S|H/)c++; t++} END{print a/t, c/t}'
done
