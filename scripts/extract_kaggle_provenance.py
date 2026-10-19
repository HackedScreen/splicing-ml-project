#!/usr/bin/env python3
"""
scripts/extract_kaggle_provenance.py

Extracts @PG headers from all BAM files in results/alignment/ and the first 15 lines
of *_trimmomatic.log and *_hisat2.log.
Outputs directly and saves to logs/kaggle_provenance_from_headers.log.
"""

import os
import glob
import subprocess
import sys

def run_cmd(cmd):
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
        return res.stdout
    except Exception as e:
        return f"ERROR: {e}\n"

def main():
    repo_dir = "/home/brooklyn/splicing-ml-project"
    os.chdir(repo_dir)
    os.makedirs("logs", exist_ok=True)

    lines = []
    lines.append("=" * 80)
    lines.append("KAGGLE PROVENANCE: SAMTOOLS @PG HEADERS ACROSS ALL BAMS IN results/alignment/")
    lines.append("=" * 80)

    bam_files = sorted(glob.glob("results/alignment/*.bam"))
    for bam in bam_files:
        lines.append(f"--- BAM: {bam} ---")
        out = run_cmd(f"samtools view -H '{bam}' | grep '^@PG'")
        if not out.strip():
            lines.append("[No @PG lines found]")
        else:
            lines.append(out.strip())
        lines.append("")

    lines.append("=" * 80)
    lines.append("FIRST 15 LINES OF *_trimmomatic.log IN results/alignment/")
    lines.append("=" * 80)
    tlogs = sorted(glob.glob("results/alignment/*_trimmomatic.log"))
    for tlog in tlogs:
        lines.append(f"--- Trimmomatic Log: {tlog} ---")
        try:
            with open(tlog, "r") as f:
                first15 = "".join([f.readline() for _ in range(15)])
                lines.append(first15.strip())
        except Exception as e:
            lines.append(f"ERROR reading {tlog}: {e}")
        lines.append("")

    lines.append("=" * 80)
    lines.append("FIRST 15 LINES OF *_hisat2.log IN results/alignment/")
    lines.append("=" * 80)
    hlogs = sorted(glob.glob("results/alignment/*_hisat2.log"))
    for hlog in hlogs:
        lines.append(f"--- HISAT2 Log: {hlog} ---")
        try:
            with open(hlog, "r") as f:
                first15 = "".join([f.readline() for _ in range(15)])
                lines.append(first15.strip())
        except Exception as e:
            lines.append(f"ERROR reading {hlog}: {e}")
        lines.append("")

    lines.append("=" * 80)
    lines.append("CONDA SHARE TRIMMOMATIC DIRECTORY NAME IN ADAPTER PATH")
    lines.append("=" * 80)
    out_grep = run_cmd("grep -o 'ILLUMINACLIP:[^ ]*' results/alignment/*_trimmomatic.log | head -n 5")
    lines.append(out_grep.strip())
    lines.append("")

    full_output = "\n".join(lines) + "\n"

    # Print to stdout
    sys.stdout.write(full_output)

    # Save to logs
    with open("logs/kaggle_provenance_from_headers.log", "w") as f:
        f.write(full_output)

    print("Saved raw output to logs/kaggle_provenance_from_headers.log")

if __name__ == "__main__":
    main()
