#!/usr/bin/env python3
"""
scripts/r1_step4_extract_features.py

Extracts the exact 23 features (sequence/GC, splice-site PWMs, coverage, PSI, and U-tract motifs)
for KD1 and KD2 from the variable-read-length rMATS results.
Excludes pvalue and fdr to guarantee zero statistical power leakage.
Includes gene_id for gene-grouped splitting.
"""

import os
import sys
import re
import numpy as np
import pandas as pd
import pysam

# -----------------------------------------------------------------------------
# PWM Splice Site Matrices (Shapiro & Senapathy)
# -----------------------------------------------------------------------------
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

def get_max_utract(seq: str) -> int:
    runs = re.findall(r"T+", seq.upper())
    return max([len(r) for r in runs], default=0)

def get_u_fraction(seq: str) -> float:
    seq = seq.upper()
    valid_bases = [b for b in seq if b in "ACGT"]
    if not valid_bases:
        return 0.0
    return float(sum(1 for b in valid_bases if b == "T") / len(valid_bases))

def extract_features_from_rmats(splicing_dir, bam_files, fasta_path, dataset_name):
    print(f"\n=======================================================")
    print(f"FEATURE EXTRACTION FOR {dataset_name}: {splicing_dir}")
    print(f"=======================================================\n")

    fasta = pysam.FastaFile(fasta_path)
    fasta_refs = set(fasta.references)

    print("Opening BAM files for coverage extraction:")
    for b in bam_files:
        print(f"  {b}")
    bams = [pysam.AlignmentFile(b, "rb") for b in bam_files]
    bam_refs = set(bams[0].references)

    event_types = ["SE", "MXE", "A3SS", "RI", "A5SS"]
    rows = []

    for event_type in event_types:
        file_path = os.path.join(splicing_dir, f"{event_type}.MATS.JC.txt")
        if not os.path.exists(file_path):
            print(f"WARNING: {file_path} not found, skipping.")
            continue

        df = pd.read_csv(file_path, sep="\t")
        total_in_file = len(df)
        sig_df = df[df["FDR"] < 0.05].copy()
        total_sig = len(sig_df)
        print(f"[{event_type}] Total: {total_in_file:,} | Significant (FDR < 0.05): {total_sig:,}", flush=True)

        for i, (idx, row) in enumerate(sig_df.iterrows(), 1):
            if i % 1000 == 0 or i == total_sig:
                print(f"  [{event_type}] Progress: {i:,}/{total_sig:,} ({i/total_sig*100:.1f}%) | Total extracted: {len(rows):,}", flush=True)
            event_id = f"{event_type}_{row['ID']}"
            gene_id = str(row.get("GeneID", "")).strip('"')
            gene_symbol = str(row.get("geneSymbol", "")).strip('"')
            raw_chr = str(row["chr"])
            strand = str(row["strand"])
            dpsi_rmats = float(row.get("IncLevelDifference", np.nan))

            inc1_mean, inc1_var = parse_inc_levels(row.get("IncLevel1", ""))
            inc2_mean, inc2_var = parse_inc_levels(row.get("IncLevel2", ""))
            calc_dpsi = inc1_mean - inc2_mean if (not np.isnan(inc1_mean) and not np.isnan(inc2_mean)) else dpsi_rmats

            # Coordinate handling
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
                if strand == "+":
                    up_intron_len = max(0, target_start - up_end)
                    dn_intron_len = max(0, dn_start - target_end)
                    ss5_pos = up_end
                    ss3_pos = dn_start
                else:
                    up_intron_len = max(0, dn_start - target_end)
                    dn_intron_len = max(0, target_start - up_end)
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

            fa_chrom = resolve_chrom_name(raw_chr, fasta_refs)
            bam_chrom = resolve_chrom_name(raw_chr, bam_refs)

            # Sequence extraction
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

                # Sense orientation for U-tracts
                if strand == "+":
                    sense_up = up_seq
                    sense_dn = dn_seq
                else:
                    sense_up = reverse_complement(dn_seq)
                    sense_dn = reverse_complement(up_seq)

                max_utract_up = get_max_utract(sense_up)
                max_utract_dn = get_max_utract(sense_dn)
                u_frac_up = get_u_fraction(sense_up)
                u_frac_dn = get_u_fraction(sense_dn)
            except Exception:
                gc_exon = np.nan
                gc_up_100bp = np.nan
                gc_dn_100bp = np.nan
                max_utract_up = np.nan
                max_utract_dn = np.nan
                u_frac_up = np.nan
                u_frac_dn = np.nan

            # Splice Site PWM Scoring
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

            # Coverage across event region
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
                "max_utract_length_upstream": max_utract_up,
                "max_utract_length_downstream": max_utract_dn,
                "u_fraction_upstream": u_frac_up,
                "u_fraction_downstream": u_frac_dn,
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
                "dpsi": calc_dpsi
            }
            rows.append(feature_dict)

    df_out = pd.DataFrame(rows)
    print(f"\nExtracted {len(df_out):,} events for {dataset_name}.")
    print(f"Class breakdown:\n{df_out['label'].value_counts()}")
    return df_out

def main():
    fasta_path = "data/reference/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
    os.makedirs("results/features", exist_ok=True)

    # 1. KD1 Discovery
    kd1_bams = [
        "results/alignment/Control_rep1_downsampled.sorted.bam",
        "results/alignment/Control_rep2_downsampled.sorted.bam",
        "results/alignment/ERR127302.sorted.bam",
        "results/alignment/ERR127303.sorted.bam"
    ]
    df_kd1 = extract_features_from_rmats(
        splicing_dir="results/splicing_kd1_vrl",
        bam_files=kd1_bams,
        fasta_path=fasta_path,
        dataset_name="KD1 Discovery"
    )
    kd1_out = "results/features/feature_matrix_v2.csv"
    df_kd1.to_csv(kd1_out, index=False)
    print(f"Saved KD1 feature matrix to {kd1_out} ({len(df_kd1):,} rows, {len(df_kd1.columns)} cols)")

    # 2. KD2 Validation
    kd2_bams = [
        "results/alignment/Control_rep1_downsampled.sorted.bam",
        "results/alignment/Control_rep2_downsampled.sorted.bam",
        "results/alignment/ERR127304.sorted.bam",
        "results/alignment/ERR127305.sorted.bam"
    ]
    df_kd2 = extract_features_from_rmats(
        splicing_dir="results/splicing_kd2_vrl",
        bam_files=kd2_bams,
        fasta_path=fasta_path,
        dataset_name="KD2 Validation"
    )
    kd2_out = "results/features/feature_matrix_kd2_v2.csv"
    df_kd2.to_csv(kd2_out, index=False)
    print(f"Saved KD2 feature matrix to {kd2_out} ({len(df_kd2):,} rows, {len(df_kd2.columns)} cols)")

if __name__ == "__main__":
    main()
