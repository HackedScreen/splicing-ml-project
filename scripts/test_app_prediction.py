#!/usr/bin/env python3
"""
scripts/test_app_prediction.py

Tests the app's prediction core function (app/predict_core.py::predict_events)
on all 1,807 held-out test rows.
Asserts that it reproduces 67.68% accuracy and macro-F1 0.6060.
Prints the integer-to-label mapping used.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Add repo and app directory to sys.path
repo_dir = "/home/brooklyn/splicing-ml-project"
if not os.path.exists(repo_dir):
    repo_dir = "C:/Users/Brooklyn/Downloads/splicing-ml-project"
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

app_dir = os.path.join(repo_dir, "app")
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

os.chdir(repo_dir)

from app.predict_core import (
    predict_events,
    load_model,
    CANONICAL_CLASSES,
    IDX_TO_LABEL
)

def main():
    print("=" * 80)
    print("STREAMLIT APP PREDICTION FUNCTION: EVALUATION ON ALL 1,807 HELD-OUT TEST ROWS")
    print("=" * 80)

    # 1. Load model via app's predict_core
    model, load_path = load_model()
    print(f"Loaded Model: {load_path}")
    print(f"Model Class: {type(model)}")
    print(f"Model Classes: {model.classes_}")

    print("\nInteger-to-Label Mapping Used:")
    for idx, lbl in IDX_TO_LABEL.items():
        print(f"  Integer {idx} -> Label '{lbl}'")

    # 2. Load held-out test set
    split_data = joblib.load("results/models/r1_train_test_data_grouped.joblib")
    test_idx = split_data["test_idx"]
    print(f"\nNumber of held-out test rows: {len(test_idx):,}")

    df_kd1 = pd.read_csv("results/features/feature_matrix_v2.csv")
    input_df = df_kd1.iloc[test_idx].copy()
    y_true = input_df["label"].values

    # 3. Call app/predict_core.py::predict_events
    results_df = predict_events(input_df, model=model)
    preds = results_df["predicted_class"].values

    # 4. Compute metrics
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
