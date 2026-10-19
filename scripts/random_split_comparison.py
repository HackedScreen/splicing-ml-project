#!/usr/bin/env python3
"""
scripts/random_split_comparison.py

Performs stratified 80/20 random split (random_state=42) of the 9,247 KD1 events.
Reports:
  1. Share of all events in multi-event genes.
  2. Share of test events whose gene also occurs in training.
  3. Evaluates XGBoost (n_estimators=300, max_depth=6, learning_rate=0.05) and
     RandomForest (n_estimators=100, max_depth=None) with fixed hyperparameters and no tuning.
  4. Reports test accuracy and macro-F1 beside the gene-grouped results.
Outputs saved to results/models/random_vs_grouped_split_comparison.csv and logged.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def main():
    repo_dir = "/home/brooklyn/splicing-ml-project"
    if os.path.exists(repo_dir):
        os.chdir(repo_dir)

    print("=" * 80)
    print("RANDOM-SPLIT VS GENE-GROUPED COMPARISON ON CURRENT DATA (KD1: 9,247 EVENTS)")
    print("=" * 80)

    # 1. Load data
    df_kd1 = pd.read_csv("results/features/feature_matrix_v2.csv")
    n_events = len(df_kd1)
    n_genes = df_kd1["gene_id"].nunique()
    print(f"Total KD1 events: {n_events:,}")
    print(f"Unique genes:     {n_genes:,}")

    # Feature preparation
    strand_dummies = pd.get_dummies(df_kd1["strand"], prefix="strand", dtype=float)
    if "strand_+" not in strand_dummies.columns: strand_dummies["strand_+"] = 0.0
    if "strand_-" not in strand_dummies.columns: strand_dummies["strand_-"] = 0.0
    df_kd1 = pd.concat([df_kd1, strand_dummies[["strand_+", "strand_-"]]], axis=1)

    psi_cols = ["psi_control_mean", "psi_control_var", "psi_kd_mean", "psi_kd_var", "dpsi"]
    df_kd1[psi_cols] = df_kd1[psi_cols].fillna(0.0)

    feat_23 = [
        "exon_length", "gc_exon", "gc_upstream_100bp", "gc_downstream_100bp",
        "max_utract_length_upstream", "max_utract_length_downstream",
        "u_fraction_upstream", "u_fraction_downstream",
        "score_5ss_donor", "score_3ss_acceptor",
        "cov_mean_control", "cov_var_control", "cov_mean_kd", "cov_var_kd",
        "cov_mean_overall", "cov_var_overall",
        "psi_control_mean", "psi_control_var", "psi_kd_mean", "psi_kd_var",
        "dpsi", "strand_+", "strand_-"
    ]

    canonical_classes = ["A3SS", "A5SS", "MXE", "RI", "SE"]
    label_to_idx = {c: i for i, c in enumerate(canonical_classes)}
    idx_to_label = {i: c for i, c in enumerate(canonical_classes)}

    X = df_kd1[feat_23].copy()
    y = df_kd1["label"].copy()
    y_idx = y.map(label_to_idx)
    genes = df_kd1["gene_id"].copy()

    # 2. Multi-event gene metrics
    gene_counts = genes.value_counts()
    multi_event_genes = gene_counts[gene_counts > 1].index
    events_in_multi = genes.isin(multi_event_genes).sum()
    pct_events_in_multi = (events_in_multi / n_events) * 100.0

    print("\n" + "-" * 60)
    print("GENE REDUNDANCY & MULTI-EVENT GENE STATISTICS:")
    print("-" * 60)
    print(f"Number of multi-event genes (>1 event): {len(multi_event_genes):,} / {n_genes:,} ({len(multi_event_genes)/n_genes*100:.2f}%)")
    print(f"Events located in multi-event genes:    {events_in_multi:,} / {n_events:,} ({pct_events_in_multi:.2f}%)")

    # 3. Stratified 80/20 random split (random_state=42)
    print("\n" + "-" * 60)
    print("STRATIFIED 80/20 RANDOM SPLIT (random_state=42):")
    print("-" * 60)
    X_train, X_test, y_train, y_test, y_train_idx, y_test_idx, genes_train, genes_test = train_test_split(
        X, y, y_idx, genes,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    n_train = len(X_train)
    n_test = len(X_test)
    train_genes_set = set(genes_train)
    test_genes_set = set(genes_test)

    # Share of test events whose gene also occurs in training
    test_events_in_train_genes = genes_test.isin(train_genes_set).sum()
    pct_test_events_in_train_genes = (test_events_in_train_genes / n_test) * 100.0

    print(f"Training events: {n_train:,} (80.00%) across {len(train_genes_set):,} unique genes")
    print(f"Testing events:  {n_test:,} (20.00%) across {len(test_genes_set):,} unique genes")
    print(f"Test events with gene also in training (Gene Leakage): {test_events_in_train_genes:,} / {n_test:,} ({pct_test_events_in_train_genes:.2f}%)")
    print(f"Shared genes between train and test: {len(train_genes_set.intersection(test_genes_set)):,} genes")

    # 4. Train XGBoost (n_estimators=300, max_depth=6, learning_rate=0.05)
    print("\n" + "-" * 60)
    print("TRAINING XGBOOST ON RANDOM SPLIT:")
    print("-" * 60)
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        eval_metric="mlogloss",
        n_jobs=4
    )
    xgb.fit(X_train, y_train_idx)
    preds_xgb_idx = xgb.predict(X_test)
    acc_xgb_random = accuracy_score(y_test_idx, preds_xgb_idx)
    f1_xgb_random = f1_score(y_test_idx, preds_xgb_idx, average="macro")
    print(f"XGBoost Random-Split Accuracy: {acc_xgb_random*100:.2f}% ({acc_xgb_random:.4f})")
    print(f"XGBoost Random-Split Macro-F1: {f1_xgb_random:.4f}")

    # 5. Train RandomForest (n_estimators=100, max_depth=None)
    print("\n" + "-" * 60)
    print("TRAINING RANDOM FOREST ON RANDOM SPLIT:")
    print("-" * 60)
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=4
    )
    rf.fit(X_train, y_train)
    preds_rf = rf.predict(X_test)
    acc_rf_random = accuracy_score(y_test, preds_rf)
    f1_rf_random = f1_score(y_test, preds_rf, average="macro")
    print(f"Random Forest Random-Split Accuracy: {acc_rf_random*100:.2f}% ({acc_rf_random:.4f})")
    print(f"Random Forest Random-Split Macro-F1: {f1_rf_random:.4f}")

    # 6. Gene-grouped results for comparison
    # From r1_test_evaluation.csv:
    # XGBoost: Acc = 0.676812 (67.68%), Macro-F1 = 0.606037 (0.6060)
    # RF:      Acc = 0.662424 (66.24%), Macro-F1 = 0.565081 (0.5651)
    acc_xgb_grouped = 0.676812
    f1_xgb_grouped = 0.606037
    acc_rf_grouped = 0.662424
    f1_rf_grouped = 0.565081

    print("\n" + "=" * 80)
    print("COMPARISON: RANDOM SPLIT VS GENE-GROUPED SPLIT")
    print("=" * 80)
    print(f"{'Split Scheme':<20} | {'Model':<15} | {'Test Acc':<10} | {'Macro-F1':<10} | {'Test Gene Leakage':<18}")
    print("-" * 80)
    print(f"{'Random (80/20)':<20} | {'XGBoost':<15} | {acc_xgb_random*100:.2f}%     | {f1_xgb_random:.4f}     | {pct_test_events_in_train_genes:.2f}% ({test_events_in_train_genes}/{n_test})")
    print(f"{'Random (80/20)':<20} | {'Random Forest':<15} | {acc_rf_random*100:.2f}%     | {f1_rf_random:.4f}     | {pct_test_events_in_train_genes:.2f}% ({test_events_in_train_genes}/{n_test})")
    print(f"{'Gene-Grouped':<20} | {'XGBoost':<15} | {acc_xgb_grouped*100:.2f}%     | {f1_xgb_grouped:.4f}     | 0.00% (0/{1807})")
    print(f"{'Gene-Grouped':<20} | {'Random Forest':<15} | {acc_rf_grouped*100:.2f}%     | {f1_rf_grouped:.4f}     | 0.00% (0/{1807})")
    print("-" * 80)
    print(f"Optimistic Bias in Random Split (XGBoost): Accuracy +{(acc_xgb_random - acc_xgb_grouped)*100:.2f} pp, Macro-F1 +{(f1_xgb_random - f1_xgb_grouped):.4f}")
    print(f"Optimistic Bias in Random Split (RF):      Accuracy +{(acc_rf_random - acc_rf_grouped)*100:.2f} pp, Macro-F1 +{(f1_rf_random - f1_rf_grouped):.4f}")
    print("=" * 80)

    # Save summary table to CSV
    comp_df = pd.DataFrame([
        {
            "Split_Scheme": "Random (80/20 Stratified)",
            "Model": "XGBoost",
            "Hyperparameters": "n_estimators=300, max_depth=6, learning_rate=0.05",
            "Test_N": n_test,
            "Accuracy": acc_xgb_random,
            "Accuracy_Pct": f"{acc_xgb_random*100:.2f}%",
            "Macro_F1": f1_xgb_random,
            "Test_Gene_Leakage_Count": test_events_in_train_genes,
            "Test_Gene_Leakage_Pct": pct_test_events_in_train_genes,
            "Events_In_Multi_Event_Genes_Pct": pct_events_in_multi
        },
        {
            "Split_Scheme": "Random (80/20 Stratified)",
            "Model": "Random Forest",
            "Hyperparameters": "n_estimators=100, max_depth=None",
            "Test_N": n_test,
            "Accuracy": acc_rf_random,
            "Accuracy_Pct": f"{acc_rf_random*100:.2f}%",
            "Macro_F1": f1_rf_random,
            "Test_Gene_Leakage_Count": test_events_in_train_genes,
            "Test_Gene_Leakage_Pct": pct_test_events_in_train_genes,
            "Events_In_Multi_Event_Genes_Pct": pct_events_in_multi
        },
        {
            "Split_Scheme": "Gene-Grouped (80/20)",
            "Model": "XGBoost",
            "Hyperparameters": "n_estimators=300, max_depth=6, learning_rate=0.05",
            "Test_N": 1807,
            "Accuracy": acc_xgb_grouped,
            "Accuracy_Pct": f"{acc_xgb_grouped*100:.2f}%",
            "Macro_F1": f1_xgb_grouped,
            "Test_Gene_Leakage_Count": 0,
            "Test_Gene_Leakage_Pct": 0.0,
            "Events_In_Multi_Event_Genes_Pct": pct_events_in_multi
        },
        {
            "Split_Scheme": "Gene-Grouped (80/20)",
            "Model": "Random Forest",
            "Hyperparameters": "n_estimators=100, max_depth=None",
            "Test_N": 1807,
            "Accuracy": acc_rf_grouped,
            "Accuracy_Pct": f"{acc_rf_grouped*100:.2f}%",
            "Macro_F1": f1_rf_grouped,
            "Test_Gene_Leakage_Count": 0,
            "Test_Gene_Leakage_Pct": 0.0,
            "Events_In_Multi_Event_Genes_Pct": pct_events_in_multi
        }
    ])
    out_csv = "results/models/random_vs_grouped_split_comparison.csv"
    comp_df.to_csv(out_csv, index=False)
    print(f"\nSaved comparison CSV to: {out_csv}")

if __name__ == "__main__":
    main()
