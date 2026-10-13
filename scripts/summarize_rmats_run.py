#!/usr/bin/env python3
import os
import sys
import glob
import pandas as pd

def summarize(od_dir):
    print(f"\n=======================================================")
    print(f"rMATS RUN SUMMARY: {od_dir}")
    print(f"=======================================================\n")

    # 1. Cat read outcomes
    ro_files = glob.glob(os.path.join(od_dir, "tmp", "*read_outcomes_by_bam.txt"))
    if not ro_files:
        ro_files = glob.glob(os.path.join(od_dir, "*read_outcomes_by_bam.txt"))
    
    print("--- Read Outcomes By BAM ---")
    if ro_files:
        for ro in sorted(ro_files):
            print(f"File: {ro}")
            with open(ro, "r") as f:
                print(f.read())
    else:
        print("No read_outcomes_by_bam.txt found.")

    # 2. Print events tested and FDR < 0.05 counts per class
    print("\n--- Splicing Events Summary (JC Counts) ---")
    sum_file = os.path.join(od_dir, "summary.txt")
    if os.path.exists(sum_file):
        df_sum = pd.read_csv(sum_file, sep="\t")
        print(df_sum[["EventType", "EventTypeDescription", "TotalEventsJC", "SignificantEventsJC", "SigEventsJCSample1HigherInclusion", "SigEventsJCSample2HigherInclusion"]].to_string(index=False))
        tot_tested = df_sum["TotalEventsJC"].sum()
        tot_sig = df_sum["SignificantEventsJC"].sum()
        print(f"\nTotal Events Tested (JC): {tot_tested:,}")
        print(f"Total Significant Events (FDR < 0.05, JC): {tot_sig:,} ({tot_sig/tot_tested*100:.2f}%)")
    else:
        # Fallback to individual *.MATS.JC.txt
        jc_files = glob.glob(os.path.join(od_dir, "*.MATS.JC.txt"))
        rows = []
        for jc in sorted(jc_files):
            etype = os.path.basename(jc).split(".")[0]
            d = pd.read_csv(jc, sep="\t")
            n_tot = len(d)
            n_sig = (d["FDR"] < 0.05).sum()
            rows.append({"EventType": etype, "TotalEventsJC": n_tot, "SignificantEventsJC": n_sig})
        df_res = pd.DataFrame(rows)
        print(df_res.to_string(index=False))
        tot_tested = df_res["TotalEventsJC"].sum()
        tot_sig = df_res["SignificantEventsJC"].sum()
        print(f"\nTotal Events Tested (JC): {tot_tested:,}")
        print(f"Total Significant Events (FDR < 0.05, JC): {tot_sig:,} ({tot_sig/tot_tested*100:.2f}%)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: summarize_rmats_run.py <output_directory>")
        sys.exit(1)
    summarize(sys.argv[1])
