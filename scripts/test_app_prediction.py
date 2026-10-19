#!/usr/bin/env python3
"""
scripts/test_app_prediction.py

Tests the app's prediction function on all 1,807 held-out test rows.
Asserts that it reproduces 67.68% accuracy and macro-F1 0.6060.
Prints the integer-to-label mapping used.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def main():
    repo_dir = "/home/brooklyn/splicing-ml-project"
    if os.path.exists(repo_dir):
        os.chdir(repo_dir)

    print("=" * 80)
    print("STREAMLIT APP PREDICTION FUNCTION: EVALUATION ON ALL 1,807 HELD-OUT TEST ROWS")
    print("=" * 80)

    # 1. Load model as done by the Streamlit app
    load_path = "results/models/r1_winner_xgboost.joblib"
    model = joblib.load(load_path)
    print(f"Loaded Model: {load_path}")
    print(f"Model Class: {type(model)}")
    print(f"Model Classes: {model.classes_}")

    # Canonical classes and mapping
    CANONICAL_CLASSES = ["A3SS", "A5SS", "MXE", "RI", "SE"]
    IDX_TO_LABEL = {i: c for i, c in enumerate(CANONICAL_CLASSES)}
    print("\nInteger-to-Label Mapping Used:")
    for idx, lbl in IDX_TO_LABEL.items():
        print(f"  Integer {idx} -> Label '{lbl}'")

    # 2. Load held-out test set
    split_data = joblib.load("results/models/r1_train_test_data_grouped.joblib")
    test_idx = split_data["test_idx"]
    print(f"\nNumber of held-out test rows: {len(test_idx):,}")

    df_kd1 = pd.read_csv("results/features/feature_matrix_v2.csv")

    # 3. Apply the Streamlit app's exact batch preprocessing pipeline
    input_df = df_kd1.iloc[test_idx].copy()
    y_true = input_df["label"].values

    proc_df = input_df.copy()
    psi_cols = ["psi_control_mean", "psi_control_var", "psi_kd_mean", "psi_kd_var", "dpsi"]
    proc_df[psi_cols] = proc_df[psi_cols].fillna(0.0)

    # Strand one-hot encoding exactly as in app
    proc_df["strand_+"] = (proc_df["strand"].astype(str).str.strip() == "+").astype(float)
    proc_df["strand_-"] = (proc_df["strand"].astype(str).str.strip() == "-").astype(float)

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
    X_batch = proc_df[feat_23]

    # 4. App's prediction logic
    raw_preds = model.predict(X_batch)
    preds = [IDX_TO_LABEL[p] for p in raw_preds]
    probs = model.predict_proba(X_batch)

    # 5. Compute metrics
    acc = accuracy_score(y_true, preds)
    macro_f1 = f1_score(y_true, preds, average="macro")

    print("\n" + "-" * 60)
    print("HELD-OUT TEST EVALUATION RESULTS:")
    print("-" * 60)
    print(f"Accuracy:  {acc*100:.2f}% ({acc:.6f})")
    print(f"Macro-F1:  {macro_f1:.4f} ({macro_f1:.6f})")

    # Per-class F1
    for c in CANONICAL_CLASSES:
        f1_c = f1_score(y_true == c, np.array(preds) == c)
        print(f"  Class {c:>5} F1: {f1_c:.4f}")

    # Assertions
    expected_acc = 0.6768123962368566
    expected_f1 = 0.6060372912855134

    assert abs(acc - expected_acc) < 1e-4, f"Accuracy mismatch: got {acc}, expected {expected_acc}"
    assert abs(macro_f1 - expected_f1) < 1e-4, f"Macro-F1 mismatch: got {macro_f1}, expected {expected_f1}"
    assert f"{acc*100:.2f}%" == "67.68%", f"Formatted accuracy is not 67.68%: got {acc*100:.2f}%"
    assert f"{macro_f1:.4f}" == "0.6060", f"Formatted macro-F1 is not 0.6060: got {macro_f1:.4f}"

    print("\n" + "=" * 80)
    print("ALL ASSERTIONS PASSED: 67.68% ACCURACY AND 0.6060 MACRO-F1 EXACTLY REPRODUCED")
    print("=" * 80)

if __name__ == "__main__":
    main()
