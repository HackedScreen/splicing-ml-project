#!/usr/bin/env python3
"""
scripts/05_feature_engineering.py

BSc Dissertation Pipeline — Stage 5: Feature Engineering
Extracts genomic, sequence, coverage, splice-site strength, and splicing features
for all significant alternative splicing events (FDR < 0.05) from rMATS-turbo output.

Outputs:
  results/features/feature_matrix.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import pysam

# -----------------------------------------------------------------------------
# 1. Position Weight Matrices (PWM) for Human Splice Sites (Shapiro & Senapathy)
# -----------------------------------------------------------------------------

# 5' Donor Splice Site: 9 bp (-3 to +6) [Exon: 3 bp | Intron: 6 bp]
PWM_5SS = {
    -3: {"A": 0.34, "C": 0.36, "G": 0.18, "T": 0.12},
    -2: {"A": 0.60, "C": 0.13, "G": 0.13, "T": 0.14},
    -1: {"A": 0.10, "C": 0.04, "G": 0.78, "T": 0.08},
     1: {"A": 0.001, "C": 0.001, "G": 0.997, "T": 0.001},
     2: {"A": 0.001, "C": 0.001, "G": 0.001, "T": 0.997},
     3: {"A": 0.59, "C": 0.03, "G": 0.34, "T": 0.04},
     4: {"A": 0.71, "C": 0.08, "G": 0.12, "T": 0.09},
     5: {"A": 0.06, "C": 0.05, "G": 0.84, "T": 0.05},
     6: {"A": 0.16, "C": 0.18, "G": 0.20, "T": 0.46}
}

# 3' Acceptor Splice Site: 23 bp (-20 to +3) [Intron: 20 bp | Exon: 3 bp]
# Polypyrimidine tract + AG dinucleotide + Exon start
PWM_3SS = {
    -20: {"A": 0.09, "C": 0.38, "G": 0.12, "T": 0.41},
    -19: {"A": 0.08, "C": 0.39, "G": 0.11, "T": 0.42},
    -18: {"A": 0.08, "C": 0.40, "G": 0.10, "T": 0.42},
    -17: {"A": 0.08, "C": 0.41, "G": 0.10, "T": 0.41},
    -16: {"A": 0.07, "C": 0.42, "G": 0.09, "T": 0.42},
    -15: {"A": 0.07, "C": 0.43, "G": 0.09, "T": 0.41},
    -14: {"A": 0.07, "C": 0.44, "G": 0.09, "T": 0.40},
    -13: {"A": 0.07, "C": 0.45, "G": 0.08, "T": 0.40},
    -12: {"A": 0.07, "C": 0.46, "G": 0.08, "T": 0.39},
    -11: {"A": 0.07, "C": 0.45, "G": 0.08, "T": 0.40},
    -10: {"A": 0.08, "C": 0.44, "G": 0.08, "T": 0.40},
     -9: {"A": 0.08, "C": 0.43, "G": 0.08, "T": 0.41},
     -8: {"A": 0.08, "C": 0.42, "G": 0.09, "T": 0.41},
     -7: {"A": 0.09, "C": 0.41, "G": 0.09, "T": 0.41},
     -6: {"A": 0.10, "C": 0.39, "G": 0.10, "T": 0.41},
     -5: {"A": 0.11, "C": 0.37, "G": 0.12, "T": 0.40},
     -4: {"A": 0.22, "C": 0.38, "G": 0.15, "T": 0.25},
     -3: {"A": 0.04, "C": 0.65, "G": 0.03, "T": 0.28},
     -2: {"A": 0.997, "C": 0.001, "G": 0.001, "T": 0.001},
     -1: {"A": 0.001, "C": 0.001, "G": 0.997, "T": 0.001},
      1: {"A": 0.26, "C": 0.15, "G": 0.47, "T": 0.12},
      2: {"A": 0.23, "C": 0.28, "G": 0.33, "T": 0.16},
      3: {"A": 0.22, "C": 0.30, "G": 0.25, "T": 0.23}
}

REV_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")

def reverse_complement(seq: str) -> str:
    return seq.translate(REV_COMP)[::-1]

def score_pwm_5ss(seq_9bp: str) -> float:
    seq_9bp = seq_9bp.upper()
    if len(seq_9bp) != 9:
        return np.nan
    score = 0.0
    positions = [-3, -2, -1, 1, 2, 3, 4, 5, 6]
    for i, pos in enumerate(positions):
        b = seq_9bp[i]
        freq = PWM_5SS[pos].get(b, 0.001)
        score += np.log2(freq / 0.25)
    return float(score)

def score_pwm_3ss(seq_23bp: str) -> float:
    seq_23bp = seq_23bp.upper()
    if len(seq_23bp) != 23:
        return np.nan
    score = 0.0
    positions = list(range(-20, 0)) + [1, 2, 3]
    for i, pos in enumerate(positions):
        b = seq_23bp[i]
        freq = PWM_3SS[pos].get(b, 0.001)
        score += np.log2(freq / 0.25)
    return float(score)

# -----------------------------------------------------------------------------
# 2. Helpers for FASTA, GC, and Coverage
# -----------------------------------------------------------------------------

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

def compute_gc_content(seq: str) -> float:
    seq = seq.upper()
    valid_bases = [b for b in seq if b in "ACGT"]
    if not valid_bases:
        return np.nan
    gc_count = sum(1 for b in valid_bases if b in "GC")
    return float(gc_count / len(valid_bases) * 100.0)

def parse_inc_levels(inc_str: str):
    if pd.isna(inc_str) or inc_str == "NA":
        return np.nan, np.nan
    try:
        vals = [float(x) for x in str(inc_str).split(",") if x != "NA" and x.strip() != ""]
        if not vals:
            return np.nan, np.nan
        mean_val = float(np.mean(vals))
        var_val = float(np.var(vals)) if len(vals) > 1 else 0.0
        return mean_val, var_val
    except Exception:
        return np.nan, np.nan

# -----------------------------------------------------------------------------
# 3. Main Feature Engineering Pipeline
# -----------------------------------------------------------------------------

def process_splicing_events():
    base_dir = os.path.expanduser("~/splicing-ml-project")
    data_dir = os.path.join(base_dir, "data")
    results_dir = os.path.join(base_dir, "results")
    splicing_dir = os.path.join(results_dir, "splicing")
    align_dir = os.path.join(results_dir, "alignment")
    out_dir = os.path.join(results_dir, "features")
    os.makedirs(out_dir, exist_ok=True)

    fasta_path = os.path.join(data_dir, "reference", "Homo_sapiens.GRCh38.dna.primary_assembly.fa")
    print(f"Opening reference FASTA: {fasta_path}")
    fasta = pysam.FastaFile(fasta_path)
    fasta_refs = set(fasta.references)

    bam_files = [
        os.path.join(align_dir, "ERR127302.sorted.bam"),
        os.path.join(align_dir, "ERR127303.sorted.bam"),
        os.path.join(align_dir, "ERR127304.sorted.bam"),
        os.path.join(align_dir, "ERR127305.sorted.bam")
    ]
    bams = [pysam.AlignmentFile(b, "rb") for b in bam_files]
    bam_refs = set(bams[0].references)

    event_types = ["SE", "RI", "A5SS", "A3SS", "MXE"]
    rows = []

    print("=== Processing rMATS output files (FDR < 0.05) ===")

    for event_type in event_types:
        file_path = os.path.join(splicing_dir, f"{event_type}.MATS.JC.txt")
        if not os.path.exists(file_path):
            print(f"WARNING: {file_path} not found, skipping.")
            continue

        df = pd.read_csv(file_path, sep="\t")
        total_in_file = len(df)
        sig_df = df[df["FDR"] < 0.05].copy()
        print(f"[{event_type}] Total: {total_in_file} | Significant (FDR < 0.05): {len(sig_df)}")

        for idx, row in sig_df.iterrows():
            event_id = f"{event_type}_{row['ID']}"
            gene_id = str(row.get("GeneID", "")).strip('"')
            gene_symbol = str(row.get("geneSymbol", "")).strip('"')
            raw_chr = str(row["chr"])
            strand = str(row["strand"])
            fdr = float(row["FDR"])
            pvalue = float(row.get("PValue", np.nan))
            dpsi_rmats = float(row.get("IncLevelDifference", np.nan))

            # Replicate PSI metrics
            inc1_mean, inc1_var = parse_inc_levels(row.get("IncLevel1", ""))
            inc2_mean, inc2_var = parse_inc_levels(row.get("IncLevel2", ""))
            
            calc_dpsi = inc1_mean - inc2_mean if (not np.isnan(inc1_mean) and not np.isnan(inc2_mean)) else dpsi_rmats

            # Coordinate extraction per event type
            if event_type == "SE":
                target_start = int(row["exonStart_0base"])
                target_end = int(row["exonEnd"])
                up_start = int(row["upstreamES"])
                up_end = int(row["upstreamEE"])
                dn_start = int(row["downstreamES"])
                dn_end = int(row["downstreamEE"])

                exon_len = target_end - target_start
                if strand == "+":
                    up_intron_len = max(0, target_start - up_end)
                    dn_intron_len = max(0, dn_start - target_end)
                    ss3_pos = target_start
                    ss5_pos = target_end
                else:
                    up_intron_len = max(0, dn_start - target_end)
                    dn_intron_len = max(0, target_start - up_end)
                    ss3_pos = target_end
                    ss5_pos = target_start

                region_start = min(up_start, target_start, dn_start)
                region_end = max(up_end, target_end, dn_end)

            elif event_type == "RI":
                target_start = int(row["riExonStart_0base"])
                target_end = int(row["riExonEnd"])
                up_start = int(row["upstreamES"])
                up_end = int(row["upstreamEE"])
                dn_start = int(row["downstreamES"])
                dn_end = int(row["downstreamEE"])

                exon_len = target_end - target_start
                retained_intron_len = max(0, dn_start - up_end)
                up_intron_len = retained_intron_len
                dn_intron_len = retained_intron_len

                if strand == "+":
                    ss5_pos = up_end
                    ss3_pos = dn_start
                else:
                    ss5_pos = dn_start
                    ss3_pos = up_end

                region_start = min(up_start, target_start, dn_start)
                region_end = max(up_end, target_end, dn_end)

            elif event_type in ["A5SS", "A3SS"]:
                target_start = int(row["longExonStart_0base"])
                target_end = int(row["longExonEnd"])
                short_start = int(row["shortES"])
                short_end = int(row["shortEE"])
                flank_start = int(row["flankingES"])
                flank_end = int(row["flankingEE"])

                exon_len = target_end - target_start
                short_exon_len = short_end - short_start
                alt_region_len = exon_len - short_exon_len

                if strand == "+":
                    up_intron_len = max(0, target_start - flank_end) if flank_end <= target_start else 0
                    dn_intron_len = max(0, flank_start - target_end) if flank_start >= target_end else 0
                    ss5_pos = target_end
                    ss3_pos = target_start
                else:
                    up_intron_len = max(0, flank_start - target_end) if flank_start >= target_end else 0
                    dn_intron_len = max(0, target_start - flank_end) if flank_end <= target_start else 0
                    ss5_pos = target_start
                    ss3_pos = target_end

                region_start = min(target_start, short_start, flank_start)
                region_end = max(target_end, short_end, flank_end)

            elif event_type == "MXE":
                target_start = int(row["1stExonStart_0base"])
                target_end = int(row["1stExonEnd"])
                ex2_start = int(row["2ndExonStart_0base"])
                ex2_end = int(row["2ndExonEnd"])
                up_start = int(row["upstreamES"])
                up_end = int(row["upstreamEE"])
                dn_start = int(row["downstreamES"])
                dn_end = int(row["downstreamEE"])

                exon_len = target_end - target_start
                up_intron_len = max(0, target_start - up_end)
                dn_intron_len = max(0, dn_start - ex2_end)

                if strand == "+":
                    ss3_pos = target_start
                    ss5_pos = target_end
                else:
                    ss3_pos = target_end
                    ss5_pos = target_start

                region_start = min(up_start, target_start, ex2_start, dn_start)
                region_end = max(up_end, target_end, ex2_end, dn_end)

            # Resolve Chromosome
            fa_chrom = resolve_chrom_name(raw_chr, fasta_refs)
            bam_chrom = resolve_chrom_name(raw_chr, bam_refs)

            # Sequence extraction for GC content
            try:
                chrom_len = fasta.get_reference_length(fa_chrom)
                ex_s = max(0, min(target_start, chrom_len))
                ex_e = max(0, min(target_end, chrom_len))
                exon_seq = fasta.fetch(fa_chrom, ex_s, ex_e)
                gc_exon = compute_gc_content(exon_seq)

                up100_s = max(0, ex_s - 100)
                up100_e = ex_s
                up_seq = fasta.fetch(fa_chrom, up100_s, up100_e)
                gc_up_100bp = compute_gc_content(up_seq)

                dn100_s = ex_e
                dn100_e = min(chrom_len, ex_e + 100)
                dn_seq = fasta.fetch(fa_chrom, dn100_s, dn100_e)
                gc_dn_100bp = compute_gc_content(dn_seq)
            except Exception:
                gc_exon = np.nan
                gc_up_100bp = np.nan
                gc_dn_100bp = np.nan

            # Splice Site Strength Scoring
            try:
                if strand == "+":
                    s5_start = max(0, ss5_pos - 3)
                    s5_end = min(chrom_len, ss5_pos + 6)
                    seq_5ss = fasta.fetch(fa_chrom, s5_start, s5_end)
                    score_5ss = score_pwm_5ss(seq_5ss)

                    s3_start = max(0, ss3_pos - 20)
                    s3_end = min(chrom_len, ss3_pos + 3)
                    seq_3ss = fasta.fetch(fa_chrom, s3_start, s3_end)
                    score_3ss = score_pwm_3ss(seq_3ss)
                else:
                    s5_start = max(0, ss5_pos - 6)
                    s5_end = min(chrom_len, ss5_pos + 3)
                    raw_5ss = fasta.fetch(fa_chrom, s5_start, s5_end)
                    seq_5ss = reverse_complement(raw_5ss)
                    score_5ss = score_pwm_5ss(seq_5ss)

                    s3_start = max(0, ss3_pos - 3)
                    s3_end = min(chrom_len, ss3_pos + 20)
                    raw_3ss = fasta.fetch(fa_chrom, s3_start, s3_end)
                    seq_3ss = reverse_complement(raw_3ss)
                    score_3ss = score_pwm_3ss(seq_3ss)
            except Exception:
                score_5ss = np.nan
                score_3ss = np.nan

            # Coverage Extraction Across Event Region from 4 BAMs
            try:
                r_start = max(0, region_start)
                r_end = max(r_start + 1, region_end)
                cov_means = []
                cov_vars = []

                for bam in bams:
                    cov_tuples = bam.count_coverage(bam_chrom, r_start, r_end)
                    cov_array = np.sum(cov_tuples, axis=0)
                    cov_means.append(float(np.mean(cov_array)))
                    cov_vars.append(float(np.var(cov_array)))

                cov_mean_ctrl = float(np.mean(cov_means[:2]))
                cov_var_ctrl = float(np.mean(cov_vars[:2]))
                cov_mean_kd = float(np.mean(cov_means[2:]))
                cov_var_kd = float(np.mean(cov_vars[2:]))
                cov_mean_all = float(np.mean(cov_means))
                cov_var_all = float(np.mean(cov_vars))
            except Exception:
                cov_mean_ctrl = np.nan
                cov_var_ctrl = np.nan
                cov_mean_kd = np.nan
                cov_var_kd = np.nan
                cov_mean_all = np.nan
                cov_var_all = np.nan

            feature_dict = {
                "event_id": event_id,
                "label": event_type,
                "gene_id": gene_id,
                "gene_symbol": gene_symbol,
                "chrom": raw_chr,
                "strand": strand,
                "exon_start": target_start,
                "exon_end": target_end,
                "exon_length": exon_len,
                "upstream_intron_length": up_intron_len,
                "downstream_intron_length": dn_intron_len,
                "gc_exon": gc_exon,
                "gc_upstream_100bp": gc_up_100bp,
                "gc_downstream_100bp": gc_dn_100bp,
                "score_5ss_donor": score_5ss,
                "score_3ss_acceptor": score_3ss,
                "cov_mean_control": cov_mean_ctrl,
                "cov_var_control": cov_var_ctrl,
                "cov_mean_kd": cov_mean_kd,
                "cov_var_kd": cov_var_kd,
                "cov_mean_overall": cov_mean_all,
                "cov_var_overall": cov_var_all,
                "psi_control_mean": inc1_mean,
                "psi_control_var": inc1_var,
                "psi_kd_mean": inc2_mean,
                "psi_kd_var": inc2_var,
                "dpsi": calc_dpsi,
                "pvalue": pvalue,
                "fdr": fdr
            }
            rows.append(feature_dict)

    # Convert to DataFrame
    feat_df = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, "feature_matrix.csv")
    feat_df.to_csv(out_csv, index=False)
    print(f"\nSaved feature matrix to {out_csv}")
    print(f"Total rows: {len(feat_df)}")
    print(f"Total columns: {len(feat_df.columns)}")

    # Class balance summary
    print("\n=== Class Distribution ===")
    class_counts = feat_df["label"].value_counts()
    for lbl, count in class_counts.items():
        pct = count / len(feat_df) * 100.0
        print(f"  {lbl:<6}: {count:>6} ({pct:>5.2f}%)")

    # Missing / NaN analysis
    print("\n=== Missing / NaN Value Analysis ===")
    null_counts = feat_df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if len(null_cols) == 0:
        print("No NaN values found in any column.")
    else:
        for col, cnt in null_cols.items():
            pct = cnt / len(feat_df) * 100.0
            print(f"  {col:<25}: {cnt:>5} NaN values ({pct:>5.2f}%)")

if __name__ == "__main__":
    process_splicing_events()