#!/usr/bin/env bash
set -euo pipefail

source /home/brooklyn/miniforge3/etc/profile.d/conda.sh
conda activate splicing_project

cd /home/brooklyn/splicing-ml-project

INDEX="data/reference/grch38_index"
R1="data/test_subset/test_1.fastq.gz"
R2="data/test_subset/test_2.fastq.gz"
SS="data/reference/splicesites.tsv"

MODE="${1:-no_ss}"

if [ "$MODE" = "no_ss" ]; then
    SAM="data/test_subset/no_ss.sam"
    LOG="data/test_subset/no_ss.log"
    MON="data/test_subset/no_ss_monitor.log"
    rm -f "$SAM" "$LOG" "$MON"

    echo "=== Starting Test 1: WITHOUT --known-splicesite-infile ==="
    echo "Start: $(date +%T)"

    (
        while true; do
            echo "--- $(date +%T) ---"
            ps -o pid,pcpu,pmem,rss,time,comm -C hisat2-align-s 2>/dev/null || echo "hisat2-align-s not running"
            ls -lh "$SAM" 2>/dev/null || echo "SAM not created yet"
            sleep 15
        done
    ) > "$MON" 2>&1 &
    MON_PID=$!

    /usr/bin/time -v hisat2 -x "$INDEX" -1 "$R1" -2 "$R2" -p 4 --dta -S "$SAM" 2> "$LOG" || true

    kill "$MON_PID" 2>/dev/null || true
    echo "End: $(date +%T)"
    echo "=== Alignment Log ==="
    cat "$LOG"
    echo "=== Final SAM Size ==="
    ls -lh "$SAM"

elif [ "$MODE" = "with_ss" ]; then
    SAM="data/test_subset/with_ss.sam"
    LOG="data/test_subset/with_ss.log"
    MON="data/test_subset/with_ss_monitor.log"
    rm -f "$SAM" "$LOG" "$MON"

    echo "=== Starting Test 2: WITH --known-splicesite-infile ==="
    echo "Start: $(date +%T)"

    (
        while true; do
            echo "--- $(date +%T) ---"
            ps -o pid,pcpu,pmem,rss,time,comm -C hisat2-align-s 2>/dev/null || echo "hisat2-align-s not running"
            ls -lh "$SAM" 2>/dev/null || echo "SAM not created yet"
            sleep 15
        done
    ) > "$MON" 2>&1 &
    MON_PID=$!

    /usr/bin/time -v hisat2 -x "$INDEX" -1 "$R1" -2 "$R2" -p 4 --dta --known-splicesite-infile "$SS" -S "$SAM" 2> "$LOG" || true

    kill "$MON_PID" 2>/dev/null || true
    echo "End: $(date +%T)"
    echo "=== Alignment Log ==="
    cat "$LOG"
    echo "=== Final SAM Size ==="
    ls -lh "$SAM"
fi
