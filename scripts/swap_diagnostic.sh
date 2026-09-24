#!/usr/bin/env bash
set -e

source /home/brooklyn/miniforge3/etc/profile.d/conda.sh
conda activate splicing_project

cd /home/brooklyn/splicing-ml-project

rm -f /tmp/wg_diag.sam /tmp/wg_diag.log

echo "=== Launching whole-genome alignment on 200k reads ==="
hisat2 -x data/reference/grch38_index \
  -1 data/test_subset/test_1.fastq.gz \
  -2 data/test_subset/test_2.fastq.gz \
  -p 4 --dta -S /tmp/wg_diag.sam > /tmp/wg_diag.log 2>&1 &
HISAT_PID=$!

echo "hisat2 wrapper PID: $HISAT_PID"

# Wait for hisat2-align-s to start
ALIGN_PID=""
for i in {1..30}; do
  ALIGN_PID=$(pgrep hisat2-align-s | head -1 || true)
  if [ -n "$ALIGN_PID" ]; then break; fi
  sleep 1
done

echo "hisat2-align-s PID: $ALIGN_PID"

echo "=== Recording Swap and Context Switches every 15s for 2 minutes ==="
for i in {1..8}; do
  echo "--- Check $i: $(date +%T) ---"
  free -h | grep -E 'Mem:|Swap:'
  if [ -n "$ALIGN_PID" ] && [ -d "/proc/$ALIGN_PID" ]; then
    cat /proc/$ALIGN_PID/status | grep -E 'VmRSS|VmSwap|voluntary_ctxt_switches'
  else
    ALIGN_PID=$(pgrep hisat2-align-s | head -1 || true)
    if [ -n "$ALIGN_PID" ] && [ -d "/proc/$ALIGN_PID" ]; then
      cat /proc/$ALIGN_PID/status | grep -E 'VmRSS|VmSwap|voluntary_ctxt_switches'
    else
      echo "hisat2-align-s not found"
    fi
  fi
  ls -lh /tmp/wg_diag.sam 2>/dev/null || echo "SAM not created yet"
  sleep 15
done

echo "=== 2-minute monitoring complete, terminating alignment ==="
kill -9 $HISAT_PID 2>/dev/null || true
if [ -n "$ALIGN_PID" ]; then kill -9 $ALIGN_PID 2>/dev/null || true; fi
killall -9 hisat2-align-s hisat2 2>/dev/null || true
rm -f /tmp/wg_diag.sam /tmp/wg_diag.log
echo "=== Cleaned up ==="
