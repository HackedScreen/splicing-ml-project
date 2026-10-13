#!/bin/bash
set -e
cd /home/brooklyn/splicing-ml-project
for f in \
  results/splicing_corrected_downsampled/tmp/*read_outcomes_by_bam.txt \
  results/splicing_kd2_corrected/tmp/*read_outcomes_by_bam.txt \
  results/splicing_lane_control/tmp/*read_outcomes_by_bam.txt; do
    echo "================================================================================"
    echo "PATH: $f"
    echo -n "SHA256: "
    sha256sum "$f"
    echo "--------------------------------------------------------------------------------"
    cat "$f"
    echo ""
done
