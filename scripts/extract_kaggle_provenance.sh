#!/usr/bin/env bash
set -e
source /home/brooklyn/miniforge3/bin/activate splicing_project
cd /home/brooklyn/splicing-ml-project
mkdir -p logs

{
echo ================================================================================
echo KAGGLE PROVENANCE: SAMTOOLS @PG HEADERS ACROSS ALL BAMS IN results/alignment/
echo ================================================================================
for bam in results/alignment/*.bam; do
    echo --- BAM: ---
    samtools view -H " | grep '^@PG' || echo [No @PG lines]
 echo 
done

echo ================================================================================
echo FIRST 15 LINES OF *_trimmomatic.log IN results/alignment/
echo ================================================================================
for tlog in results/alignment/*_trimmomatic.log; do
 echo --- Trimmomatic Log:  ---
 head -n 15 
 echo 
done

echo ================================================================================
echo FIRST 15 LINES OF *_hisat2.log IN results/alignment/
echo ================================================================================
for hlog in results/alignment/*_hisat2.log; do
 echo --- HISAT2 Log:  ---
 head -n 15 
 echo 
done

echo ================================================================================
echo CONDA SHARE TRIMMOMATIC DIRECTORY NAME IN ADAPTER PATH
echo ================================================================================
grep -o 'ILLUMINACLIP:[^ ]*' results/alignment/*_trimmomatic.log | head -n 5
} | tee logs/kaggle_provenance_from_headers.log
