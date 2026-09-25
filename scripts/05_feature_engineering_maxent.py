#!/usr/bin/env python3
"""
scripts/05_feature_engineering_maxent.py

BSc Dissertation Pipeline — Stage 5 V3: Genuine MaxEntScan Splice Site Scoring
Replaces additive PWM splice site scores (score_5ss_donor, score_3ss_acceptor)
with genuine MaxEntScan (Yeo & Burge, 2004) Maximum Entropy model scores:
  - 5' Splice Site (Donor): 9 bp window [-3 exon, +6 intron]
  - 3' Splice Site (Acceptor): 23 bp window [-20 intron, +3 exon]

Inputs:
  results/features/feature_matrix_v2.csv (KD1 discovery cohort, N=8,693)
  results/features/feature_matrix_kd2_v2.csv (KD2 validation cohort, N=7,080)
  results/splicing_corrected_downsampled/*.MATS.JC.txt (KD1 coordinates)
  results/splicing_kd2_corrected/*.MATS.JC.txt (KD2 coordinates)
  data/reference/Homo_sapiens.GRCh38.dna.primary_assembly.fa
  tools/maxentscan/splicemodels/

Outputs:
  results/features/feature_matrix_v3_maxent.csv
  results/features/feature_matrix_kd2_v3_maxent.csv
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import pysam

REV_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")

def reverse_complement(seq: str) -> str:
    return seq.translate(REV_COMP)[::-1]

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

class MaxEntScanner:
    """Exact Python implementation of Yeo & Burge (2004) MaxEntScan models."""
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.bgd = {"A": 0.27, "C": 0.23, "G": 0.23, "T": 0.27}
        
        # 5' Donor parameters
        self.cons1_5 = {"A": 0.004, "C": 0.0032, "G": 0.9896, "T": 0.0032}
        self.cons2_5 = {"A": 0.0034, "C": 0.0039, "G": 0.0042, "T": 0.9884}
        
        self.me2x5 = []
        with open(os.path.join(model_dir, "me2x5")) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.me2x5.append(float(line))

        self.seq_dict_5 = {}
        with open(os.path.join(model_dir, "splice5sequences")) as f:
            for idx, line in enumerate(f):
                self.seq_dict_5[line.strip()] = idx

        # 3' Acceptor parameters
        self.cons1_3 = {"A": 0.9903, "C": 0.0032, "G": 0.0034, "T": 0.0030}
        self.cons2_3 = {"A": 0.0027, "C": 0.0037, "G": 0.9905, "T": 0.0030}
        self.four = [1, 4, 16, 64, 256, 1024, 4096, 16384]
        self.tr = {"A": 0, "C": 1, "G": 2, "T": 3}

        self.metables_3 = []
        for i in range(1, 10):
            tbl_file = os.path.join(model_dir, f"me2x3acc{i}")
            tbl = []
            with open(tbl_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        tbl.append(float(line))
            self.metables_3.append(tbl)

    def hashseq(self, s: str) -> int:
        length = len(s)
        h = 0
        for i, char in enumerate(s):
            h += self.tr.get(char, 0) * self.four[length - i - 1]
        return h

    def score5(self, seq: str) -> float:
        seq = seq.upper()
        if len(seq) != 9 or not all(c in "ACGT" for c in seq):
            return -20.0
        rest = seq[0:3] + seq[5:9]
        c_score = (self.cons1_5[seq[3]] * self.cons2_5[seq[4]]) / (self.bgd[seq[3]] * self.bgd[seq[4]])
        if rest not in self.seq_dict_5:
            return -20.0
        rest_idx = self.seq_dict_5[rest]
        val = c_score * self.me2x5[rest_idx]
        if val <= 0:
            return -20.0
        return math.log2(val)

    def score3(self, seq: str) -> float:
        seq = seq.upper()
        if len(seq) != 23 or not all(c in "ACGT" for c in seq):
            return -20.0
        c_score = (self.cons1_3[seq[18]] * self.cons2_3[seq[19]]) / (self.bgd[seq[18]] * self.bgd[seq[19]])
        seq_rest = seq[:18] + seq[20:23]
        sc = [
            self.metables_3[0][self.hashseq(seq_rest[0:7])],
            self.metables_3[1][self.hashseq(seq_rest[7:14])],
            self.metables_3[2][self.hashseq(seq_rest[14:21])],
            self.metables_3[3][self.hashseq(seq_rest[4:11])],
            self.metables_3[4][self.hashseq(seq_rest[11:18])],
            self.metables_3[5][self.hashseq(seq_rest[4:7])],
            self.metables_3[6][self.hashseq(seq_rest[7:11])],
            self.metables_3[7][self.hashseq(seq_rest[11:14])],
            self.metables_3[8][self.hashseq(seq_rest[14:18])]
        ]
        denom = sc[5] * sc[6] * sc[7] * sc[8]
        if denom == 0:
            return -20.0
        finalscore = (sc[0] * sc[1] * sc[2] * sc[3] * sc[4]) / denom
        val = c_score * finalscore
        if val <= 0:
            return -20.0
        return math.log2(val)

def extract_splice_coordinates(splicing_dir: str):
    """Map event_id -> (chrom, strand, ss5_pos, ss3_pos) from rMATS JC tables."""
    event_types = ["SE", "MXE", "A3SS", "RI", "A5SS"]
    coords = {}

    for event_type in event_types:
        fpath = os.path.join(splicing_dir, f"{event_type}.MATS.JC.txt")
        if not os.path.exists(fpath):
            continue
        df = pd.read_csv(fpath, sep="\t")
        sig_df = df[df["FDR"] < 0.05]
        for _, row in sig_df.iterrows():
            eid = f"{event_type}_{row['ID']}"
            chrom = str(row["chr"])
            strand = str(row["strand"])

            if event_type == "SE":
                target_start = int(row["exonStart_0base"])
                target_end = int(row["exonEnd"])
                if strand == "+":
                    ss3_pos = target_start
                    ss5_pos = target_end
                else:
                    ss3_pos = target_end
                    ss5_pos = target_start

            elif event_type == "RI":
                up_end = int(row["upstreamEE"])
                dn_start = int(row["downstreamES"])
                if strand == "+":
                    ss5_pos = up_end
                    ss3_pos = dn_start
                else:
                    ss5_pos = dn_start
                    ss3_pos = up_end

            elif event_type in ["A5SS", "A3SS"]:
                target_start = int(row["longExonStart_0base"])
                target_end = int(row["longExonEnd"])
                if strand == "+":
                    ss5_pos = target_end
                    ss3_pos = target_start
                else:
                    ss5_pos = target_start
                    ss3_pos = target_end

            elif event_type == "MXE":
                target_start = int(row["1stExonStart_0base"])
                target_end = int(row["1stExonEnd"])
                if strand == "+":
                    ss3_pos = target_start
                    ss5_pos = target_end
                else:
                    ss3_pos = target_end
                    ss5_pos = target_start

            coords[eid] = (chrom, strand, ss5_pos, ss3_pos)

    return coords

def compute_maxent_scores(df_feat: pd.DataFrame, coords: dict, fasta: pysam.FastaFile, scanner: MaxEntScanner):
    available_refs = set(fasta.references)
    scores_5ss = []
    scores_3ss = []

    for _, row in df_feat.iterrows():
        eid = row["event_id"]
        if eid not in coords:
            scores_5ss.append(row["score_5ss_donor"])
            scores_3ss.append(row["score_3ss_acceptor"])
            continue

        chrom, strand, ss5_pos, ss3_pos = coords[eid]
        fa_chrom = resolve_chrom_name(chrom, available_refs)
        chrom_len = fasta.get_reference_length(fa_chrom)

        # 5' Donor 9bp: [-3 exon, +6 intron]
        # 3' Acceptor 23bp: [-20 intron, +3 exon]
        try:
            if strand == "+":
                s5_s = max(0, ss5_pos - 3)
                s5_e = min(chrom_len, ss5_pos + 6)
                seq_5ss = fasta.fetch(fa_chrom, s5_s, s5_e)

                s3_s = max(0, ss3_pos - 20)
                s3_e = min(chrom_len, ss3_pos + 3)
                seq_3ss = fasta.fetch(fa_chrom, s3_s, s3_e)
            else:
                s5_s = max(0, ss5_pos - 6)
                s5_e = min(chrom_len, ss5_pos + 3)
                seq_5ss = reverse_complement(fasta.fetch(fa_chrom, s5_s, s5_e))

                s3_s = max(0, ss3_pos - 3)
                s3_e = min(chrom_len, ss3_pos + 20)
                seq_3ss = reverse_complement(fasta.fetch(fa_chrom, s3_s, s3_e))

            score_5 = scanner.score5(seq_5ss)
            score_3 = scanner.score3(seq_3ss)
        except Exception:
            score_5 = -20.0
            score_3 = -20.0

        scores_5ss.append(score_5)
        scores_3ss.append(score_3)

    return scores_5ss, scores_3ss

def main():
    base_dir = os.path.expanduser("~/splicing-ml-project")
    results_dir = os.path.join(base_dir, "results")
    features_dir = os.path.join(results_dir, "features")
    fasta_path = os.path.join(base_dir, "data", "reference", "Homo_sapiens.GRCh38.dna.primary_assembly.fa")
    model_dir = os.path.join(base_dir, "tools", "maxentscan", "splicemodels")

    splicing_kd1_dir = os.path.join(results_dir, "splicing_corrected_downsampled")
    splicing_kd2_dir = os.path.join(results_dir, "splicing_kd2_corrected")

    kd1_in_path = os.path.join(features_dir, "feature_matrix_v2.csv")
    kd2_in_path = os.path.join(features_dir, "feature_matrix_kd2_v2.csv")

    kd1_out_path = os.path.join(features_dir, "feature_matrix_v3_maxent.csv")
    kd2_out_path = os.path.join(features_dir, "feature_matrix_kd2_v3_maxent.csv")

    print("=" * 80)
    print("STAGE 5 V3: MAXENTSCAN SPLICE SITE FEATURE GENERATION")
    print("=" * 80)

    print("Initializing MaxEntScanner...")
    scanner = MaxEntScanner(model_dir)

    print(f"Opening reference FASTA: {fasta_path}")
    fasta = pysam.FastaFile(fasta_path)

    # 1. Process KD1 Cohort
    print(f"\nProcessing KD1 Discovery Cohort ({kd1_in_path})...")
    df_kd1 = pd.read_csv(kd1_in_path)
    coords_kd1 = extract_splice_coordinates(splicing_kd1_dir)
    print(f"Extracted coordinates for {len(coords_kd1):,} KD1 events.")

    s5_kd1, s3_kd1 = compute_maxent_scores(df_kd1, coords_kd1, fasta, scanner)
    df_kd1_v3 = df_kd1.copy()
    df_kd1_v3["score_5ss_donor"] = s5_kd1
    df_kd1_v3["score_3ss_acceptor"] = s3_kd1
    df_kd1_v3.to_csv(kd1_out_path, index=False)
    print(f"Saved KD1 V3 feature matrix to {kd1_out_path} (N = {len(df_kd1_v3):,})")

    # 2. Process KD2 Cohort
    print(f"\nProcessing KD2 Validation Cohort ({kd2_in_path})...")
    df_kd2 = pd.read_csv(kd2_in_path)
    coords_kd2 = extract_splice_coordinates(splicing_kd2_dir)
    print(f"Extracted coordinates for {len(coords_kd2):,} KD2 events.")

    s5_kd2, s3_kd2 = compute_maxent_scores(df_kd2, coords_kd2, fasta, scanner)
    df_kd2_v3 = df_kd2.copy()
    df_kd2_v3["score_5ss_donor"] = s5_kd2
    df_kd2_v3["score_3ss_acceptor"] = s3_kd2
    df_kd2_v3.to_csv(kd2_out_path, index=False)
    print(f"Saved KD2 V3 feature matrix to {kd2_out_path} (N = {len(df_kd2_v3):,})")

    # Summary Stats Comparison (PWM vs MaxEntScan)
    print("\n--- Splice Site Score Comparison: Additive PWM vs MaxEntScan (Yeo & Burge) ---")
    print(f"KD1 5'ss Donor  : PWM Mean = {df_kd1['score_5ss_donor'].mean():.2f} (std {df_kd1['score_5ss_donor'].std():.2f}) | MaxEnt Mean = {df_kd1_v3['score_5ss_donor'].mean():.2f} (std {df_kd1_v3['score_5ss_donor'].std():.2f})")
    print(f"KD1 3'ss Accept : PWM Mean = {df_kd1['score_3ss_acceptor'].mean():.2f} (std {df_kd1['score_3ss_acceptor'].std():.2f}) | MaxEnt Mean = {df_kd1_v3['score_3ss_acceptor'].mean():.2f} (std {df_kd1_v3['score_3ss_acceptor'].std():.2f})")
    print(f"KD2 5'ss Donor  : PWM Mean = {df_kd2['score_5ss_donor'].mean():.2f} (std {df_kd2['score_5ss_donor'].std():.2f}) | MaxEnt Mean = {df_kd2_v3['score_5ss_donor'].mean():.2f} (std {df_kd2_v3['score_5ss_donor'].std():.2f})")
    print(f"KD2 3'ss Accept : PWM Mean = {df_kd2['score_3ss_acceptor'].mean():.2f} (std {df_kd2['score_3ss_acceptor'].std():.2f}) | MaxEnt Mean = {df_kd2_v3['score_3ss_acceptor'].mean():.2f} (std {df_kd2_v3['score_3ss_acceptor'].std():.2f})")

if __name__ == "__main__":
    main()
