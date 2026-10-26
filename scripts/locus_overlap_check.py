import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

def main():
    df = pd.read_csv("results/features/feature_matrix_v2.csv")
    split_grp = joblib.load("results/models/train_test_data_v2_gene_grouped.joblib")
    rf_grp = joblib.load("results/models/random_forest_v2_gene_grouped.joblib")

    train_idx = split_grp["train_idx"]
    test_idx = split_grp["test_idx"]

    df_train = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()
    X_test = split_grp["X_test"]
    y_test = split_grp["y_test"]
    classes = split_grp["classes"]

    # Predict test set
    y_pred = rf_grp.predict(X_test)
    df_test["y_true"] = y_test.values
    df_test["y_pred"] = y_pred

    print("=" * 80)
    print("DEFINITION 1: EXON-ONLY INTERVAL [exon_start, exon_end]")
    print("=" * 80)
    # Check interval overlap (same chrom, same strand, start1 <= end2 and end1 >= start2)
    # Group train by (chrom, strand)
    train_intervals = {}
    for (chrom, strand), grp in df_train.groupby(["chrom", "strand"]):
        # Store list of (start, end)
        train_intervals[(chrom, strand)] = grp[["exon_start", "exon_end"]].values

    overlap_mask_exon = []
    for idx, row in df_test.iterrows():
        key = (row["chrom"], row["strand"])
        if key not in train_intervals:
            overlap_mask_exon.append(False)
            continue
        s, e = row["exon_start"], row["exon_end"]
        intervals = train_intervals[key]
        # Any overlap: s <= intervals[:, 1] and e >= intervals[:, 0]
        has_overlap = np.any((s <= intervals[:, 1]) & (e >= intervals[:, 0]))
        overlap_mask_exon.append(has_overlap)

    overlap_mask_exon = np.array(overlap_mask_exon)
    eval_strata(df_test, overlap_mask_exon, "Exon-only [exon_start, exon_end]", classes)

    print("\n" + "=" * 80)
    print("DEFINITION 2: OUTER GENOMIC INTERVAL INCLUDING FLANKING INTRONS")
    print("=" * 80)
    # Calculate outer interval:
    # If strand is +, upstream is left, downstream is right:
    # outer_start = min(exon_start, exon_start - upstream_intron_length)
    # outer_end = max(exon_end, exon_end + downstream_intron_length)
    df_train["outer_start"] = df_train["exon_start"] - df_train["upstream_intron_length"].fillna(0)
    df_train["outer_end"] = df_train["exon_end"] + df_train["downstream_intron_length"].fillna(0)
    df_test["outer_start"] = df_test["exon_start"] - df_test["upstream_intron_length"].fillna(0)
    df_test["outer_end"] = df_test["exon_end"] + df_test["downstream_intron_length"].fillna(0)

    train_outer = {}
    for (chrom, strand), grp in df_train.groupby(["chrom", "strand"]):
        train_outer[(chrom, strand)] = grp[["outer_start", "outer_end"]].values

    overlap_mask_outer = []
    for idx, row in df_test.iterrows():
        key = (row["chrom"], row["strand"])
        if key not in train_outer:
            overlap_mask_outer.append(False)
            continue
        s, e = row["outer_start"], row["outer_end"]
        intervals = train_outer[key]
        has_overlap = np.any((s <= intervals[:, 1]) & (e >= intervals[:, 0]))
        overlap_mask_outer.append(has_overlap)

    overlap_mask_outer = np.array(overlap_mask_outer)
    eval_strata(df_test, overlap_mask_outer, "Outer interval with flanking introns", classes)

def eval_strata(df_test, overlap_mask, name, classes):
    n_total = len(df_test)
    n_over = int(overlap_mask.sum())
    n_no = n_total - n_over

    print(f"\nResults for {name}:")
    print(f"Total Test Events: {n_total:,}")
    print(f"  - Overlapping:     {n_over:,} ({n_over/n_total*100:.2f}%)")
    print(f"  - Non-Overlapping: {n_no:,} ({n_no/n_total*100:.2f}%)")

    # Evaluate overlapping
    if n_over > 0:
        y_t_over = df_test.loc[overlap_mask, "y_true"]
        y_p_over = df_test.loc[overlap_mask, "y_pred"]
        acc_over = accuracy_score(y_t_over, y_p_over)
        f1_over = f1_score(y_t_over, y_p_over, average="macro", zero_division=0)
        cr_over = classification_report(y_t_over, y_p_over, labels=classes, output_dict=True, zero_division=0)
        print(f"\nStratum: WITH OVERLAP (N = {n_over}):")
        print(f"  Accuracy: {acc_over*100:.2f}%, Macro-F1: {f1_over:.4f}")
        for c in classes:
            print(f"    - {c}: N={int(cr_over[c]['support'])}, F1={cr_over[c]['f1-score']:.4f}")

    # Evaluate non-overlapping
    if n_no > 0:
        y_t_no = df_test.loc[~overlap_mask, "y_true"]
        y_p_no = df_test.loc[~overlap_mask, "y_pred"]
        acc_no = accuracy_score(y_t_no, y_p_no)
        f1_no = f1_score(y_t_no, y_p_no, average="macro", zero_division=0)
        cr_no = classification_report(y_t_no, y_p_no, labels=classes, output_dict=True, zero_division=0)
        print(f"\nStratum: WITHOUT OVERLAP (N = {n_no}):")
        print(f"  Accuracy: {acc_no*100:.2f}%, Macro-F1: {f1_no:.4f}")
        for c in classes:
            print(f"    - {c}: N={int(cr_no[c]['support'])}, F1={cr_no[c]['f1-score']:.4f}")

if __name__ == "__main__":
    main()
