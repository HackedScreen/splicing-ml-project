import pandas as pd
import numpy as np

event_types = ["SE", "MXE", "A3SS", "RI", "A5SS"]

lane_dir = "results/splicing_lane_control"
kd_dir = "results/splicing_corrected"
down_dir = "results/splicing_corrected_downsampled"

print("=" * 80)
print("NEGATIVE CONTROL ANALYSIS: ERR127306 vs ERR127307 (TECHNICAL LANES)")
print("=" * 80)

lane_summary = pd.read_csv(f"{lane_dir}/summary.txt", sep="\t")
print("\n--- Summary Table (from summary.txt) ---")
print(lane_summary[["EventType", "TotalEventsJC", "SignificantEventsJC", "SigEventsJCSample1HigherInclusion", "SigEventsJCSample2HigherInclusion"]].to_string(index=False))

total_quant = lane_summary["TotalEventsJC"].sum()
total_sig = lane_summary["SignificantEventsJC"].sum()
print(f"\nTotal Quantified Events (JC): {total_quant:,}")
print(f"Total Significant Events (FDR < 0.05): {total_sig:,} ({total_sig/total_quant*100:.2f}%)")

print("\n" + "=" * 80)
print("EFFECT SIZE (|IncLevelDifference|) COMPARISON ACROSS ALL EVENT TYPES")
print("=" * 80)

for et in event_types:
    lane_f = f"{lane_dir}/{et}.MATS.JC.txt"
    kd_f = f"{kd_dir}/{et}.MATS.JC.txt"
    down_f = f"{down_dir}/{et}.MATS.JC.txt"
    
    df_lane = pd.read_csv(lane_f, sep="\t")
    df_kd = pd.read_csv(kd_f, sep="\t")
    df_down = pd.read_csv(down_f, sep="\t")
    
    sig_lane = df_lane[df_lane["FDR"] < 0.05]
    sig_kd = df_kd[df_kd["FDR"] < 0.05]
    sig_down = df_down[df_down["FDR"] < 0.05]
    
    diff_lane = sig_lane["IncLevelDifference"].abs()
    diff_kd = sig_kd["IncLevelDifference"].abs()
    diff_down = sig_down["IncLevelDifference"].abs()
    
    print(f"\n--- {et} ---")
    print(f"  Lane Control (N=1 vs N=1): {len(sig_lane):,} sig (Median |dPSI| = {diff_lane.median():.4f}, Mean = {diff_lane.mean():.4f})")
    print(f"    |dPSI| >= 0.05: {(diff_lane >= 0.05).sum():,} ({(diff_lane >= 0.05).mean()*100:.1f}%) | >= 0.10: {(diff_lane >= 0.10).sum():,} ({(diff_lane >= 0.10).mean()*100:.1f}%)")
    print(f"  Full-Depth Control vs KD1: {len(sig_kd):,} sig (Median |dPSI| = {diff_kd.median():.4f}, Mean = {diff_kd.mean():.4f})")
    print(f"    |dPSI| >= 0.05: {(diff_kd >= 0.05).sum():,} ({(diff_kd >= 0.05).mean()*100:.1f}%) | >= 0.10: {(diff_kd >= 0.10).sum():,} ({(diff_kd >= 0.10).mean()*100:.1f}%)")
    print(f"  Downsampled Control vs KD1: {len(sig_down):,} sig (Median |dPSI| = {diff_down.median():.4f}, Mean = {diff_down.mean():.4f})")
    print(f"    |dPSI| >= 0.05: {(diff_down >= 0.05).sum():,} ({(diff_down >= 0.05).mean()*100:.1f}%) | >= 0.10: {(diff_down >= 0.10).sum():,} ({(diff_down >= 0.10).mean()*100:.1f}%)")
