#!/usr/bin/env bash
set -euo pipefail

cd /home/brooklyn/splicing-ml-project
mkdir -p logs
mkdir -p results/archive_readlength72
mkdir -p results/archive_readlength72/features

echo "=== [R1 Step 1] Archiving readlength-72 results ==="
date

echo "1. Archiving splicing directories..."
if [ -d "results/splicing_corrected_downsampled" ]; then
    cp -r results/splicing_corrected_downsampled results/archive_readlength72/
    echo "  Copied results/splicing_corrected_downsampled -> results/archive_readlength72/"
fi

if [ -d "results/splicing_kd2_corrected" ]; then
    cp -r results/splicing_kd2_corrected results/archive_readlength72/
    echo "  Copied results/splicing_kd2_corrected -> results/archive_readlength72/"
fi

if [ -d "results/splicing_lane_control" ]; then
    cp -r results/splicing_lane_control results/archive_readlength72/
    echo "  Copied results/splicing_lane_control -> results/archive_readlength72/"
fi

echo "2. Archiving v2 feature matrices..."
if [ -f "results/features/feature_matrix_v2.csv" ]; then
    cp results/features/feature_matrix_v2.csv results/archive_readlength72/features/
    cp results/features/feature_matrix_v2.csv results/archive_readlength72/
    echo "  Copied results/features/feature_matrix_v2.csv -> results/archive_readlength72/"
fi

if [ -f "results/features/feature_matrix_kd2_v2.csv" ]; then
    cp results/features/feature_matrix_kd2_v2.csv results/archive_readlength72/features/
    cp results/features/feature_matrix_kd2_v2.csv results/archive_readlength72/
    echo "  Copied results/features/feature_matrix_kd2_v2.csv -> results/archive_readlength72/"
fi

echo "Archive contents verification:"
ls -la results/archive_readlength72/

echo "=== Step 1 Archival Complete ==="
date
