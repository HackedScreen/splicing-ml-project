import os
import pandas as pd

runs = {
    "Original (KD1 vs KD2, depth-balanced)": "results/splicing",
    "Corrected Full-Depth (Control vs KD1, ~2.1x depth)": "results/splicing_corrected",
    "Corrected Downsampled (Control vs KD1, depth-balanced)": "results/splicing_corrected_downsampled",
}

event_types = ["SE", "MXE", "A3SS", "RI", "A5SS"]

results = {}

for run_name, run_dir in runs.items():
    if not os.path.exists(run_dir):
        continue
    summary_file = os.path.join(run_dir, "summary.txt")
    if os.path.exists(summary_file):
        summary_df = pd.read_csv(summary_file, sep="\t")
        total_events = summary_df["TotalEventsJC"].sum()
        sig_events = summary_df.set_index("EventType")["SignificantEventsJC"].to_dict()
        total_sig = sum(sig_events.values())
        results[run_name] = {
            "total_quantified": total_events,
            "total_significant": total_sig,
            "sig_per_type": sig_events,
        }
    else:
        # Check if individual JC files exist
        sig_events = {}
        tot_events = {}
        found = False
        for et in event_types:
            jc_path = os.path.join(run_dir, f"{et}.MATS.JC.txt")
            if os.path.exists(jc_path):
                found = True
                df = pd.read_csv(jc_path, sep="\t")
                tot_events[et] = len(df)
                sig_events[et] = len(df[df["FDR"] < 0.05])
        if found:
            total_sig = sum(sig_events.values())
            results[run_name] = {
                "total_quantified": sum(tot_events.values()),
                "total_significant": total_sig,
                "sig_per_type": sig_events,
            }

print("=" * 80)
print("COMPARATIVE SPLICING EVENT ANALYSIS ACROSS RUNS")
print("=" * 80)

for run_name, data in results.items():
    print(f"\n--- {run_name} ---")
    tot_sig = data["total_significant"]
    print(f"Total Quantified Events: {data['total_quantified']:,}")
    print(f"Total Significant Events (FDR < 0.05): {tot_sig:,}")
    print("Event Type Breakdown:")
    for et in event_types:
        cnt = data["sig_per_type"].get(et, 0)
        pct = (cnt / tot_sig * 100) if tot_sig > 0 else 0
        print(f"  {et:5s}: {cnt:6,d} ({pct:6.2f}%)")

print("\n" + "=" * 80)
print("PROPORTION COMPARISON TABLE")
print("=" * 80)
header = f"{'Event':<6} | " + " | ".join([f"{name[:25]:<25}" for name in results.keys()])
print(header)
print("-" * len(header))

for et in event_types:
    row = f"{et:<6} | "
    cols = []
    for run_name, data in results.items():
        tot_sig = data["total_significant"]
        cnt = data["sig_per_type"].get(et, 0)
        pct = (cnt / tot_sig * 100) if tot_sig > 0 else 0
        cols.append(f"{cnt:,} ({pct:.2f}%)".ljust(25))
    row += " | ".join(cols)
    print(row)
