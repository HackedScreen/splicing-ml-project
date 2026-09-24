#!/usr/bin/env python3
"""
scripts/05_feature_engineering_v2.py

BSc Dissertation Pipeline — Stage 5 V2: HNRNPC U-Tract Binding Feature Engineering
Augments depth-balanced feature matrices with 4 new HNRNPC U-tract binding features:
  1. max_utract_length_upstream: Longest consecutive run of T's (poly-U on transcript sense strand) in 100bp upstream window.
  2. max_utract_length_downstream: Longest consecutive run of T's (poly-U on transcript sense strand) in 100bp downstream window.
  3. u_fraction_upstream: Overall U/T nucleotide fraction in 100bp upstream window (sense strand).
  4. u_fraction_downstream: Overall U/T nucleotide fraction in 100bp downstream window (sense strand).

Ensures KD1 and KD2 feature matrices have identical columns (including upstream/downstream intron length coordinates).

Inputs:
  results/features/feature_matrix_corrected.csv (KD1 discovery cohort, N=8,693)
  results/features/feature_matrix_kd2_corrected.csv (KD2 validation cohort, N=7,080)
  results/splicing_kd2_corrected/*.MATS.JC.txt (for coordinate extraction)
  data/reference/Homo_sapiens.GRCh38.dna.primary_assembly.fa

Outputs:
  results/features/feature_matrix_v2.csv
  results/features/feature_matrix_kd2_v2.csv
"""

import os
import sys
import re
import numpy as np
import pandas as pd
import pysam

REV_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")

def reverse_complement(seq: str) -> str:
    return seq.translate(REV_COMP)[::-1]

def get_max_utract(seq: str) -> int:
    runs = re.findall(r"T+", seq.upper())
    return max([len(r) for r in runs], default=0)

def get_u_fraction(seq: str) -> float:
    seq = seq.upper()
    valid_bases = [b for b in seq if b in "ACGT"]
    if not valid_bases:
        return 0.0
    return float(sum(1 for b in valid_bases if b == "T") / len(valid_bases))

def resolve_chrom_name(chrom: str, available_refs: set) -> str:
    if chrom in available_refs:
        return chrom
    clean = chrom.replace("chr", "")
    if clean in available_refs:
        return clean
    with_chr = f"chr{chrom}"
    if with_chr in available_refs:
        return with_chr
    return chrom

def extract_kd2_intron_lengths(splicing_dir: str):
    """Extract upstream and downstream intron lengths for KD2 events to guarantee column parity with KD1."""
    event_types = ["SE", "MXE", "A3SS", "RI", "A5SS"]
    intron_lens = {}
    
    for event_type in event_types:
        fpath = os.path.join(splicing_dir, f"{event_type}.MATS.JC.txt")
        if not os.path.exists(fpath):
            continue
        df = pd.read_csv(fpath, sep="\t")
        sig_df = df[df["FDR"] < 0.05]
        for _, row in sig_df.iterrows():
            eid = f"{event_type}_{row['ID']}"
            strand = str(row["strand"])
            if event_type == "SE":
                target_start, target_end = int(row["exonStart_0base"]), int(row["exonEnd"])
                up_end, dn_start = int(row["upstreamEE"]), int(row["downstreamES"])
                if strand == "+":
                    up_len = max(0, target_start - up_end)
                    dn_len = max(0, dn_start - target_end)
                else:
                    up_len = max(0, dn_start - target_end)
                    dn_len = max(0, target_start - up_end)
            elif event_type == "RI":
                target_start, target_end = int(row["riExonStart_0base"]), int(row["riExonEnd"])
                up_end, dn_start = int(row["upstreamEE"]), int(row["downstreamES"])
                if strand == "+":
                    up_len = max(0, target_start - up_end)
                    dn_len = max(0, dn_start - target_end)
                else:
                    up_len = max(0, dn_start - target_end)
                    dn_len = max(0, target_start - up_end)
            elif event_type in ["A5SS", "A3SS"]:
                target_start, target_end = int(row["longExonStart_0base"]), int(row["longExonEnd"])
                flank_start, flank_end = int(row["flankingES"]), int(row["flankingEE"])
                if strand == "+":
                    up_len = max(0, target_start - flank_end) if flank_end <= target_start else 0
                    dn_len = max(0, flank_start - target_end) if flank_start >= target_end else 0
                else:
                    up_len = max(0, flank_start - target_end) if flank_start >= target_end else 0
                    dn_len = max(0, target_start - flank_end) if flank_end <= target_start else 0
            elif event_type == "MXE":
                target_start, target_end = int(row["1stExonStart_0base"]), int(row["1stExonEnd"])
                ex2_end, up_end, dn_start = int(row["2ndExonEnd"]), int(row["upstreamEE"]), int(row["downstreamES"])
                up_len = max(0, target_start - up_end)
                dn_len = max(0, dn_start - ex2_end)
            intron_lens[eid] = (up_len, dn_len)
            
    return intron_lens

def compute_utract_features_for_df(df: pd.DataFrame, fasta: pysam.FastaFile, fasta_refs: set, dataset_name: str) -> pd.DataFrame:
    print(f"\nProcessing {dataset_name} ({len(df):,} events)...", flush=True)
    
    max_utract_up = []
    max_utract_dn = []
    u_frac_up = []
    u_frac_dn = []
    
    for i, (_, row) in enumerate(df.iterrows(), 1):
        if i % 2000 == 0 or i == len(df):
            print(f"  [{dataset_name}] Processed {i:,}/{len(df):,} events ({(i/len(df))*100:.1f}%)", flush=True)
            
        raw_chr = str(row["chrom"])
        fa_chrom = resolve_chrom_name(raw_chr, fasta_refs)
        ex_s = int(row["exon_start"])
        ex_e = int(row["exon_end"])
        strand = str(row["strand"])
        
        try:
            chrom_len = fasta.get_reference_length(fa_chrom)
            up100_s = max(0, ex_s - 100)
            up100_e = ex_s
            up_raw = fasta.fetch(fa_chrom, up100_s, up100_e)
            
            dn100_s = ex_e
            dn100_e = min(chrom_len, ex_e + 100)
            dn_raw = fasta.fetch(fa_chrom, dn100_s, dn100_e)
            
            # Transcript sense orientation:
            # For + strand: upstream is [ex_s - 100, ex_s], downstream is [ex_e, ex_e + 100]
            # For - strand: transcription is 3' <- 5'. Upstream of exon in pre-mRNA is [ex_e, ex_e + 100],
            #               downstream of exon is [ex_s - 100, ex_s]. Both reverse-complemented to sense RNA.
            if strand == "+":
                sense_up = up_raw
                sense_dn = dn_raw
            else:
                sense_up = reverse_complement(dn_raw)
                sense_dn = reverse_complement(up_raw)
                
            max_utract_up.append(get_max_utract(sense_up))
            max_utract_dn.append(get_max_utract(sense_dn))
            u_frac_up.append(get_u_fraction(sense_up))
            u_frac_dn.append(get_u_fraction(sense_dn))
        except Exception as e:
            print(f"Error fetching sequence for row {i}: {e}")
            max_utract_up.append(np.nan)
            max_utract_dn.append(np.nan)
            u_frac_up.append(np.nan)
            u_frac_dn.append(np.nan)
            
    df_out = df.copy()
    
    # Insert new columns immediately following gc_downstream_100bp
    gc_dn_idx = df_out.columns.get_loc("gc_downstream_100bp")
    
    col_order = list(df_out.columns[:gc_dn_idx + 1]) + [
        "max_utract_length_upstream",
        "max_utract_length_downstream",
        "u_fraction_upstream",
        "u_fraction_downstream"
    ] + list(df_out.columns[gc_dn_idx + 1:])
    
    df_out["max_utract_length_upstream"] = max_utract_up
    df_out["max_utract_length_downstream"] = max_utract_dn
    df_out["u_fraction_upstream"] = u_frac_up
    df_out["u_fraction_downstream"] = u_frac_dn
    
    df_out = df_out[col_order]
    return df_out

def main():
    base_dir = os.path.expanduser("~/splicing-ml-project")
    data_dir = os.path.join(base_dir, "data")
    features_dir = os.path.join(base_dir, "results", "features")
    splicing_kd2_dir = os.path.join(base_dir, "results", "splicing_kd2_corrected")
    
    kd1_in = os.path.join(features_dir, "feature_matrix_corrected.csv")
    kd2_in = os.path.join(features_dir, "feature_matrix_kd2_corrected.csv")
    
    kd1_out = os.path.join(features_dir, "feature_matrix_v2.csv")
    kd2_out = os.path.join(features_dir, "feature_matrix_kd2_v2.csv")
    
    fasta_path = os.path.join(data_dir, "reference", "Homo_sapiens.GRCh38.dna.primary_assembly.fa")
    
    print(f"Opening reference FASTA: {fasta_path}")
    if not os.path.exists(fasta_path):
        print(f"ERROR: Reference FASTA not found at {fasta_path}")
        sys.exit(1)
        
    fasta = pysam.FastaFile(fasta_path)
    fasta_refs = set(fasta.references)
    
    # 1. Process KD1
    print(f"\nLoading KD1 feature matrix: {kd1_in}")
    df_kd1 = pd.read_csv(kd1_in)
    print(f"Loaded {len(df_kd1):,} events and {len(df_kd1.columns)} columns.")
    df_kd1_v2 = compute_utract_features_for_df(df_kd1, fasta, fasta_refs, "KD1 Discovery Cohort")
    
    # 2. Process KD2
    print(f"\nLoading KD2 feature matrix: {kd2_in}")
    df_kd2 = pd.read_csv(kd2_in)
    print(f"Loaded {len(df_kd2):,} events and {len(df_kd2.columns)} columns.")
    
    # Check if upstream_intron_length and downstream_intron_length need extraction for KD2
    if "upstream_intron_length" not in df_kd2.columns:
        print("Extracting KD2 intron length coordinates for exact parity with KD1...")
        intron_lens = extract_kd2_intron_lengths(splicing_kd2_dir)
        up_lens = [intron_lens.get(eid, (0, 0))[0] for eid in df_kd2["event_id"]]
        dn_lens = [intron_lens.get(eid, (0, 0))[1] for eid in df_kd2["event_id"]]
        # Insert right after exon_length
        ex_len_idx = df_kd2.columns.get_loc("exon_length")
        df_kd2.insert(ex_len_idx + 1, "upstream_intron_length", up_lens)
        df_kd2.insert(ex_len_idx + 2, "downstream_intron_length", dn_lens)
        print(f"Added upstream_intron_length and downstream_intron_length to KD2.")
        
    df_kd2_v2 = compute_utract_features_for_df(df_kd2, fasta, fasta_refs, "KD2 Validation Cohort")
    
    # Ensure exact column ordering match
    df_kd2_v2 = df_kd2_v2[df_kd1_v2.columns]
    
    # Save artifacts
    print(f"\nSaving KD1 V2 feature matrix: {kd1_out}")
    df_kd1_v2.to_csv(kd1_out, index=False)
    print(f"Saving KD2 V2 feature matrix: {kd2_out}")
    df_kd2_v2.to_csv(kd2_out, index=False)
    
    # Sync to Windows mount
    win_features_dir = "/mnt/c/Users/Brooklyn/Downloads/splicing-ml-project/results/features"
    if os.path.exists(win_features_dir):
        df_kd1_v2.to_csv(os.path.join(win_features_dir, "feature_matrix_v2.csv"), index=False)
        df_kd2_v2.to_csv(os.path.join(win_features_dir, "feature_matrix_kd2_v2.csv"), index=False)
        print(f"Synced V2 feature matrices to {win_features_dir}")
        
    # Check NaN values
    print("\n=== Validation Check: Missing Values in New Columns ===")
    for name, df in [("KD1", df_kd1_v2), ("KD2", df_kd2_v2)]:
        cols = ["max_utract_length_upstream", "max_utract_length_downstream", "u_fraction_upstream", "u_fraction_downstream"]
        nans = df[cols].isnull().sum().to_dict()
        print(f"  {name} NaN counts: {nans}")
        assert all(v == 0 for v in nans.values()), f"Error: NaNs detected in {name} new features!"
        
    print(f"\nColumns match exactly between KD1 and KD2: {list(df_kd1_v2.columns) == list(df_kd2_v2.columns)}")
    print(f"KD1 dimensions: {df_kd1_v2.shape} (original: {df_kd1.shape})")
    print(f"KD2 dimensions: {df_kd2_v2.shape} (original: {df_kd2.shape})")
    print(f"Total columns in both: {len(df_kd1_v2.columns)}")
    
    # Summary Statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS: U-TRACT MOTIF DENSITY BY SPLICING EVENT TYPE")
    print("="*80)
    
    utract_cols = [
        "max_utract_length_upstream",
        "max_utract_length_downstream",
        "u_fraction_upstream",
        "u_fraction_downstream"
    ]
    
    print("\n--- KD1 DISCOVERY COHORT (N=8,693) ---")
    print("\n[Mean by Event Type]")
    print(df_kd1_v2.groupby("label")[utract_cols].mean().round(4))
    print("\n[Median by Event Type]")
    print(df_kd1_v2.groupby("label")[utract_cols].median().round(4))
    print("\n[Standard Deviation by Event Type]")
    print(df_kd1_v2.groupby("label")[utract_cols].std().round(4))
    
    print("\n--- KD2 INDEPENDENT VALIDATION COHORT (N=7,080) ---")
    print("\n[Mean by Event Type]")
    print(df_kd2_v2.groupby("label")[utract_cols].mean().round(4))
    print("\n[Median by Event Type]")
    print(df_kd2_v2.groupby("label")[utract_cols].median().round(4))
    print("\n[Standard Deviation by Event Type]")
    print(df_kd2_v2.groupby("label")[utract_cols].std().round(4))

if __name__ == "__main__":
    main()
