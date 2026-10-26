import os
import pandas as pd

results_dir = "results/splicing_corrected"
event_types = ["SE", "RI", "A5SS", "A3SS", "MXE"]

total_events = 0
sig_events = {}
tot_per_type = {}

for et in event_types:
    filepath = os.path.join(results_dir, f"{et}.MATS.JC.txt")
    df = pd.read_csv(filepath, sep="\t")
    tot = len(df)
    tot_per_type[et] = tot
    total_events += tot
    sig = df[df["FDR"] < 0.05]
    sig_events[et] = len(sig)

total_sig = sum(sig_events.values())

print(f"Total quantified events: {total_events}")
print(f"Total significant events (FDR < 0.05): {total_sig}")
print("Per-type counts:")
for et in event_types:
    pct = (sig_events[et] / total_sig * 100) if total_sig > 0 else 0
    print(f"  {et}: {sig_events[et]:,} ({pct:.2f}%) [out of {tot_per_type[et]:,} total]")

summary_str = f"Quantified {total_events:,} total splicing events with {total_sig:,} significant events (FDR < 0.05): " + ", ".join(
    [f"{et}: {sig_events[et]:,} ({sig_events[et]/total_sig*100:.2f}%)" for et in event_types]
)
print("\nFormatted summary:")
print(summary_str)
