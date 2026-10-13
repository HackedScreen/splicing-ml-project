#!/usr/bin/env python3
"""
scripts/test_app_prediction.py
Tests the app's loading of r1_winner_xgboost.joblib and runs one prediction on a test-set row.
"""

import os
import joblib
import pandas as pd
import numpy as np

def main():
    repo_dir = "/home/brooklyn/splicing-ml-project"
    if os.path.exists(repo_dir):
        os.chdir(repo_dir)

    print("=" * 80)
    print("STREAMLIT APP MODEL LOAD & TEST SET ROW PREDICTION VERIFICATION")
    print("=" * 80)

    # 1. Load model using exact candidate line from app/streamlit_app.py
    load_path = "results/models/r1_winner_xgboost.joblib"
    model = joblib.load(load_path)
    print(f"RAW LOAD LINE: model = joblib.load('{load_path}')")
    print(f"MODEL TYPE: {type(model)}")
    print(f"MODEL n_features_in_: {model.n_features_in_}")
    print(f"MODEL classes_: {model.classes_}")

    # 2. Load test set row
    split_data = joblib.load("results/models/r1_train_test_data_grouped.joblib")
    test_idx = split_data["test_idx"]

    df_kd1 = pd.read_csv("results/features/feature_matrix_v2.csv")
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
    idx_to_label = {i: c for i, c in enumerate(canonical_classes)}

    # Pick the first test set event row
    first_test_row_idx = test_idx[0]
    test_row_df = df_kd1.iloc[[first_test_row_idx]][feat_23]
    true_label = df_kd1.iloc[first_test_row_idx]["label"]
    gene_id = df_kd1.iloc[first_test_row_idx]["gene_id"]
    event_id = df_kd1.iloc[first_test_row_idx].get("event_id", f"row_{first_test_row_idx}")

    print(f"\nTest Event Index: {first_test_row_idx} (Gene: {gene_id}, Event: {event_id})")
    print(f"True Label: {true_label}")

    raw_pred = model.predict(test_row_df)[0]
    pred_label = idx_to_label[raw_pred]
    probs = model.predict_proba(test_row_df)[0]

    print(f"Raw Model Output (Integer Index): {raw_pred}")
    print(f"Decoded Predicted Label: {pred_label}")
    print(f"Prediction Confidence: {np.max(probs)*100:.2f}%")
    print("Class Probabilities:")
    for c, p in zip(canonical_classes, probs):
        print(f"  {c:>5}: {p*100:.2f}%")

    print("\n" + "=" * 80)
    print("APP VERIFICATION COMPLETE: SUCCESS")
    print("=" * 80)

if __name__ == "__main__":
    main()
