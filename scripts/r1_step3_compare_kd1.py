#!/usr/bin/env python3
import os
import sys
import glob
import numpy as np
import pandas as pd

def get_coord_key(row, etype):
    c = str(row["chr"]).strip()
    s = str(row["strand"]).strip()
    if etype == "SE":
        return (etype, c, s, int(row["exonStart_0base"]), int(row["exonEnd"]),
                int(row["upstreamES"]), int(row["upstreamEE"]),
                int(row["downstreamES"]), int(row["downstreamEE"]))
    elif etype == "MXE":
        return (etype, c, s, int(row["1stExonStart_0base"]), int(row["1stExonEnd"]),
                int(row["2ndExonStart_0base"]), int(row["2ndExonEnd"]),
                int(row["upstreamES"]), int(row["upstreamEE"]),
                int(row["downstreamES"]), int(row["downstreamEE"]))
    elif etype in ["A3SS", "A5SS"]:
        return (etype, c, s, int(row["longExonStart_0base"]), int(row["longExonEnd"]),
                int(row["shortES"]), int(row["shortEE"]),
                int(row["flankingES"]), int(row["flankingEE"]))
    elif etype == "RI":
        return (etype, c, s, int(row["riExonStart_0base"]), int(row["riExonEnd"]),
                int(row["upstreamES"]), int(row["upstreamEE"]),
                int(row["downstreamES"]), int(row["downstreamEE"]))
    return (etype, c, s, row.get("ID"))

def main():
    old_dir = "results/archive_readlength72/splicing_corrected_downsampled"
    new_dir = "results/splicing_kd1_vrl"

    print("=" * 80)
    print("COMPARISON: OLD (readLength 72) VS NEW (--variable-read-length) KD1 EVENT SETS")
    print("=" * 80)
    print(f"Old directory: {old_dir}")
    print(f"New directory: {new_dir}\n")

    event_types = ["SE", "MXE", "A3SS", "A5SS", "RI"]
    results = []

    old_all_sig = []
    new_all_sig = []

    for etype in event_types:
        old_f = os.path.join(old_dir, f"{etype}.MATS.JC.txt")
        new_f = os.path.join(new_dir, f"{etype}.MATS.JC.txt")

        df_old = pd.read_csv(old_f, sep="\t")
        df_new = pd.read_csv(new_f, sep="\t")

        sig_old = df_old[df_old["FDR"] < 0.05].copy()
        sig_new = df_new[df_new["FDR"] < 0.05].copy()

        old_keys = set(sig_old.apply(lambda r: get_coord_key(r, etype), axis=1))
        new_keys = set(sig_new.apply(lambda r: get_coord_key(r, etype), axis=1))

        both_keys = old_keys.intersection(new_keys)
        old_only_keys = old_keys - new_keys
        new_only_keys = new_keys - old_keys

        med_dpsi_old = sig_old["IncLevelDifference"].abs().median()
        med_dpsi_new = sig_new["IncLevelDifference"].abs().median()

        results.append({
            "EventType": etype,
            "Old_Count": len(sig_old),
            "New_Count": len(sig_new),
            "In_Both": len(both_keys),
            "Old_Only": len(old_only_keys),
            "New_Only": len(new_only_keys),
            "Old_Median_dPSI": med_dpsi_old,
            "New_Median_dPSI": med_dpsi_new
        })

        sig_old["coord_key"] = sig_old.apply(lambda r: get_coord_key(r, etype), axis=1)
        sig_new["coord_key"] = sig_new.apply(lambda r: get_coord_key(r, etype), axis=1)
        old_all_sig.append(sig_old)
        new_all_sig.append(sig_new)

    df_old_all = pd.concat(old_all_sig, ignore_index=True)
    df_new_all = pd.concat(new_all_sig, ignore_index=True)

    all_old_keys = set(df_old_all["coord_key"])
    all_new_keys = set(df_new_all["coord_key"])
    all_both = all_old_keys.intersection(all_new_keys)
    all_old_only = all_old_keys - all_new_keys
    all_new_only = all_new_keys - all_old_keys

    results.append({
        "EventType": "Total",
        "Old_Count": len(df_old_all),
        "New_Count": len(df_new_all),
        "In_Both": len(all_both),
        "Old_Only": len(all_old_only),
        "New_Only": len(all_new_only),
        "Old_Median_dPSI": df_old_all["IncLevelDifference"].abs().median(),
        "New_Median_dPSI": df_new_all["IncLevelDifference"].abs().median()
    })

    comp_df = pd.DataFrame(results)
    out_csv = "results/models/r1_kd1_event_comparison.csv"
    comp_df.to_csv(out_csv, index=False)

    print(comp_df.to_string(index=False))
    print(f"\nSaved comparison summary table to {out_csv}")

if __name__ == "__main__":
    main()
