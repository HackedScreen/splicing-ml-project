#!/usr/bin/env python3
"""
scripts/generate_confusion_matrices.py

Generates and saves the canonical confusion matrices:
1. results/models/r1_confusion_matrix_winner.csv (held-out test set, N=1,807)
2. results/models/r1_confusion_matrix_kd2.csv (KD2 cohort, N=7,279)
Verifies test accuracy = 67.68%, macro-F1 = 0.6060, and KD2 accuracy = 72.73%, macro-F1 = 0.6660.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

def main():
    CANONICAL_CLASSES = ["A3SS", "A5SS", "MXE", "RI", "SE"]
    IDX_TO_LABEL = {i: c for i, c in enumerate(CANONICAL_CLASSES)}
    LABEL_TO_IDX = {c: i for i, c in enumerate(CANONICAL_CLASSES)}

    # Load model
    model = joblib.load("results/models/r1_winner_xgboost.joblib")
    
    # Load KD1 feature matrix and test index
    df_kd1 = pd.read_csv("results/features/feature_matrix_v2.csv")
    split_data = joblib.load("results/models/r1_train_test_data_grouped.joblib")
    test_idx = split_data["test_idx"]

    # Preprocess KD1
    df_kd1_test = df_kd1.iloc[test_idx].copy()
    psi_cols = ["psi_control_mean", "psi_control_var", "psi_kd_mean", "psi_kd_var", "dpsi"]
    df_kd1_test[psi_cols] = df_kd1_test[psi_cols].fillna(0.0)
    df_kd1_test["strand_+"] = (df_kd1_test["strand"].astype(str).str.strip() == "+").astype(float)
    df_kd1_test["strand_-"] = (df_kd1_test["strand"].astype(str).str.strip() == "-").astype(float)

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

    X_test = df_kd1_test[feat_23]
    y_test = df_kd1_test["label"].values

    preds_raw = model.predict(X_test)
    preds_test = [IDX_TO_LABEL[p] for p in preds_raw]

    acc_test = accuracy_score(y_test, preds_test)
    f1_test = f1_score(y_test, preds_test, average="macro")

    cm_test = confusion_matrix(y_test, preds_test, labels=CANONICAL_CLASSES)
    cm_test_df = pd.DataFrame(cm_test, index=CANONICAL_CLASSES, columns=CANONICAL_CLASSES)
    cm_test_df.to_csv("results/models/r1_confusion_matrix_winner.csv")

    print("=" * 70)
    print("HELD-OUT TEST CONFUSION MATRIX (N = 1,807):")
    print(f"Accuracy: {acc_test*100:.2f}% | Macro-F1: {f1_test:.4f}")
    print("=" * 70)
    print(cm_test_df)
    print("\nRow sums (True counts):")
    print(cm_test_df.sum(axis=1))
    print("\nColumn sums (Pred counts):")
    print(cm_test_df.sum(axis=0))

    # KD2 Evaluation
    df_kd2 = pd.read_csv("results/features/feature_matrix_kd2_v2.csv")
    df_kd2[psi_cols] = df_kd2[psi_cols].fillna(0.0)
    df_kd2["strand_+"] = (df_kd2["strand"].astype(str).str.strip() == "+").astype(float)
    df_kd2["strand_-"] = (df_kd2["strand"].astype(str).str.strip() == "-").astype(float)

    X_kd2 = df_kd2[feat_23]
    y_kd2 = df_kd2["label"].values

    preds_kd2_raw = model.predict(X_kd2)
    preds_kd2 = [IDX_TO_LABEL[p] for p in preds_kd2_raw]

    acc_kd2 = accuracy_score(y_kd2, preds_kd2)
    f1_kd2 = f1_score(y_kd2, preds_kd2, average="macro")

    cm_kd2 = confusion_matrix(y_kd2, preds_kd2, labels=CANONICAL_CLASSES)
    cm_kd2_df = pd.DataFrame(cm_kd2, index=CANONICAL_CLASSES, columns=CANONICAL_CLASSES)
    cm_kd2_df.to_csv("results/models/r1_confusion_matrix_kd2.csv")

    print("\n" + "=" * 70)
    print("KD2 CONFUSION MATRIX (N = 7,279):")
    print(f"Accuracy: {acc_kd2*100:.2f}% | Macro-F1: {f1_kd2:.4f}")
    print("=" * 70)
    print(cm_kd2_df)
    print("\nRow sums (True counts):")
    print(cm_kd2_df.sum(axis=1))
    print("\nColumn sums (Pred counts):")
    print(cm_kd2_df.sum(axis=0))

if __name__ == "__main__":
    main()
