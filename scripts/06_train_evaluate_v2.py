#!/usr/bin/env python3
"""
scripts/06_train_evaluate_v2.py

BSc Dissertation Pipeline — Stages 6 & 7 V2:
Trains and evaluates XGBoost and Random Forest classifiers on the augmented 23-feature dataset:
  - 17 original numeric features + 2 strand dummy features
  - 4 new HNRNPC U-tract binding motif features:
      * max_utract_length_upstream
      * max_utract_length_downstream
      * u_fraction_upstream
      * u_fraction_downstream
  - Total: 23 features

Uses the exact same held-out train/test split from results/models/train_test_data_corrected.joblib:
  - Training: N = 6,954 (80.0%)
  - Held-out Internal Test: N = 1,739 (20.0%)

Applies sample_weight=compute_sample_weight('balanced', y_train) for XGBoost.
Tunes XGBoost via 5-fold Stratified GridSearchCV over:
  - n_estimators: [100, 300]
  - max_depth: [3, 6, 10]
  - learning_rate: [0.05, 0.1, 0.2]

Also tunes Random Forest V2 (n_estimators: [100, 300], max_depth: [None, 10, 20]) for direct feature ablation.

Evaluates both models on:
  1. Internal KD1 Held-Out Test Set (N = 1,739)
  2. External KD2 Independent Biological Validation Cohort (N = 7,080)

Outputs:
  results/models/xgboost_v2.joblib
  results/models/random_forest_v2.joblib
  results/models/train_test_data_v2.joblib
  results/models/confusion_matrix_xgb_internal_v2.png
  results/models/confusion_matrix_xgb_kd2_v2.png
  results/models/confusion_matrix_rf_internal_v2.png
  results/models/confusion_matrix_rf_kd2_v2.png
  results/models/feature_importance_xgb_v2.png
  results/models/feature_importance_rf_v2.png
  results/models/evaluation_report_v2.md
  results/models/kd2_validation_report_v2.md
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier

def main():
    base_dir = os.path.expanduser("~/splicing-ml-project")
    features_dir = os.path.join(base_dir, "results", "features")
    models_dir = os.path.join(base_dir, "results", "models")
    win_models_dir = "/mnt/c/Users/Brooklyn/Downloads/splicing-ml-project/results/models"
    os.makedirs(models_dir, exist_ok=True)

    kd1_feat_path = os.path.join(features_dir, "feature_matrix_v2.csv")
    kd2_feat_path = os.path.join(features_dir, "feature_matrix_kd2_v2.csv")
    prev_split_path = os.path.join(models_dir, "train_test_data_corrected.joblib")

    print(f"Loading V2 feature matrix: {kd1_feat_path}")
    if not os.path.exists(kd1_feat_path):
        print(f"ERROR: {kd1_feat_path} does not exist.")
        sys.exit(1)

    df_kd1 = pd.read_csv(kd1_feat_path)
    print(f"Loaded KD1: {len(df_kd1):,} events and {len(df_kd1.columns)} columns.")

    print(f"Loading previous train/test split: {prev_split_path}")
    if not os.path.exists(prev_split_path):
        print(f"ERROR: {prev_split_path} does not exist.")
        sys.exit(1)

    prev_split = joblib.load(prev_split_path)
    train_idx = prev_split["X_train"].index
    test_idx = prev_split["X_test"].index

    print(f"Verified train indices: {len(train_idx):,} | test indices: {len(test_idx):,}")

    # 1. Missing Value Imputation
    psi_cols = ["psi_control_mean", "psi_control_var", "psi_kd_mean", "psi_kd_var"]
    df_kd1[psi_cols] = df_kd1[psi_cols].fillna(0.0)

    # 2. Define 23 Feature Standard
    numeric_features = [
        "exon_length",
        "gc_exon",
        "gc_upstream_100bp",
        "gc_downstream_100bp",
        "max_utract_length_upstream",
        "max_utract_length_downstream",
        "u_fraction_upstream",
        "u_fraction_downstream",
        "score_5ss_donor",
        "score_3ss_acceptor",
        "cov_mean_control",
        "cov_var_control",
        "cov_mean_kd",
        "cov_var_kd",
        "cov_mean_overall",
        "cov_var_overall",
        "psi_control_mean",
        "psi_control_var",
        "psi_kd_mean",
        "psi_kd_var",
        "dpsi"
    ]

    strand_dummies = pd.get_dummies(df_kd1["strand"], prefix="strand", dtype=float)
    if "strand_+" not in strand_dummies.columns:
        strand_dummies["strand_+"] = 0.0
    if "strand_-" not in strand_dummies.columns:
        strand_dummies["strand_-"] = 0.0
    strand_dummies = strand_dummies[["strand_+", "strand_-"]]

    X_full = pd.concat([df_kd1[numeric_features], strand_dummies], axis=1)
    y_full = df_kd1["label"]
    feature_names = list(X_full.columns)

    print(f"\nFinal feature set ({len(feature_names)} features):")
    for i, fn in enumerate(feature_names, 1):
        print(f"  {i:>2}. {fn}")

    # 3. Apply exact held-out split
    X_train = X_full.loc[train_idx]
    X_test = X_full.loc[test_idx]
    y_train = y_full.loc[train_idx]
    y_test = y_full.loc[test_idx]

    # Verify target matches exactly
    assert (y_train.values == prev_split["y_train"].values).all(), "Error: y_train labels mismatch!"
    assert (y_test.values == prev_split["y_test"].values).all(), "Error: y_test labels mismatch!"
    print("\nVerified exact identity of sample splits with train_test_data_corrected.joblib.")

    classes = sorted(list(np.unique(y_train)))
    print(f"Classes ({len(classes)}): {classes}")

    le = LabelEncoder()
    le.fit(classes)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)

    # Compute balanced sample weights for training
    sample_weights_train = compute_sample_weight("balanced", y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # -------------------------------------------------------------------------
    # 4. XGBoost & Random Forest Model Loading or Training
    # -------------------------------------------------------------------------
    xgb_model_path = os.path.join(models_dir, "xgboost_v2.joblib")
    rf_v2_model_path = os.path.join(models_dir, "random_forest_v2.joblib")
    split_v2_data_path = os.path.join(models_dir, "train_test_data_v2.joblib")

    if os.path.exists(xgb_model_path) and os.path.exists(rf_v2_model_path):
        print(f"\nExisting trained V2 models detected! Loading directly:")
        print(f"  - XGBoost V2: {xgb_model_path}")
        print(f"  - Random Forest V2: {rf_v2_model_path}")
        best_xgb = joblib.load(xgb_model_path)
        best_rf_v2 = joblib.load(rf_v2_model_path)
        xgb_best_params = {'learning_rate': 0.05, 'max_depth': 6, 'n_estimators': 300}
        rf_best_params = {'max_depth': None, 'n_estimators': 300}
    else:
        print("\n" + "="*80)
        print("TRAINING XGBOOST CLASSIFIER (5-Fold Stratified CV, Balanced Sample Weights)")
        print("="*80)

        xgb_clf = XGBClassifier(
            random_state=42,
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=-1
        )

        xgb_param_grid = {
            "n_estimators": [100, 300],
            "max_depth": [3, 6, 10],
            "learning_rate": [0.05, 0.1, 0.2]
        }

        xgb_grid = GridSearchCV(
            xgb_clf,
            param_grid=xgb_param_grid,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            verbose=1,
            refit=True
        )

        print("Executing XGBoost grid search (18 parameter candidates x 5 folds = 90 fits)...")
        xgb_grid.fit(X_train, y_train_enc, sample_weight=sample_weights_train)

        print("\nXGBoost GridSearch CV Results:")
        for params, mean_score, std_score in zip(
            xgb_grid.cv_results_["params"],
            xgb_grid.cv_results_["mean_test_score"],
            xgb_grid.cv_results_["std_test_score"]
        ):
            print(f"  Params: {params} -> Mean 5-Fold F1-Macro: {mean_score:.4f} (+/- {std_score:.4f})")

        best_xgb = xgb_grid.best_estimator_
        xgb_best_params = xgb_grid.best_params_
        print(f"\nBest XGBoost Parameters: {xgb_best_params}")
        print(f"Best XGBoost 5-Fold CV Score (F1-Macro): {xgb_grid.best_score_:.4f}")

        # -------------------------------------------------------------------------
        # 5. Random Forest V2 Training & GridSearch Tuning (Feature Ablation)
        # -------------------------------------------------------------------------
        print("\n" + "="*80)
        print("TRAINING RANDOM FOREST V2 (5-Fold Stratified CV, Balanced Class Weights, 23 Features)")
        print("="*80)

        rf_clf = RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        rf_param_grid = {
            "n_estimators": [100, 300],
            "max_depth": [None, 10, 20]
        }

        rf_grid = GridSearchCV(
            rf_clf,
            param_grid=rf_param_grid,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            verbose=1,
            refit=True
        )

        rf_grid.fit(X_train, y_train)

        print("\nRandom Forest V2 GridSearch CV Results:")
        for params, mean_score, std_score in zip(
            rf_grid.cv_results_["params"],
            rf_grid.cv_results_["mean_test_score"],
            rf_grid.cv_results_["std_test_score"]
        ):
            print(f"  Params: {params} -> Mean 5-Fold F1-Macro: {mean_score:.4f} (+/- {std_score:.4f})")

        best_rf_v2 = rf_grid.best_estimator_
        rf_best_params = rf_grid.best_params_
        print(f"\nBest Random Forest V2 Parameters: {rf_best_params}")
        print(f"Best Random Forest V2 5-Fold CV Score (F1-Macro): {rf_grid.best_score_:.4f}")

        # Save Model Artifacts
        joblib.dump(best_xgb, xgb_model_path)
        joblib.dump(best_rf_v2, rf_v2_model_path)
        joblib.dump({
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "feature_names": feature_names,
            "label_encoder": le,
            "classes": classes
        }, split_v2_data_path)

        # Sync to Windows mount
        win_models_dir = "/mnt/c/Users/Brooklyn/Downloads/splicing-ml-project/results/models"
        if os.path.exists(win_models_dir):
            joblib.dump(best_xgb, os.path.join(win_models_dir, "xgboost_v2.joblib"))
            joblib.dump(best_rf_v2, os.path.join(win_models_dir, "random_forest_v2.joblib"))
            joblib.dump({
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "feature_names": feature_names,
                "label_encoder": le,
                "classes": classes
            }, os.path.join(win_models_dir, "train_test_data_v2.joblib"))

        print("\nSaved Models:")
        print(f"  - XGBoost V2: {xgb_model_path}")
        print(f"  - Random Forest V2: {rf_v2_model_path}")
        print(f"  - Train/Test Data V2: {split_v2_data_path}")

    # -------------------------------------------------------------------------
    # 7. Helper Evaluation Function
    # -------------------------------------------------------------------------
    def evaluate_model(y_true, y_pred, y_prob, model_name, dataset_name):
        acc = accuracy_score(y_true, y_pred)
        p_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
        r_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)

        p_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        r_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1_wt = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        roc_ovr = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        cr = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)

        return {
            "name": model_name,
            "dataset": dataset_name,
            "accuracy": acc,
            "precision_macro": p_macro,
            "recall_macro": r_macro,
            "f1_macro": f1_mac,
            "precision_weighted": p_weighted,
            "recall_weighted": r_weighted,
            "f1_weighted": f1_wt,
            "roc_auc_ovr": roc_ovr,
            "confusion_matrix": cm,
            "report_dict": cr
        }

    # -------------------------------------------------------------------------
    # 8. Evaluation on Internal KD1 Held-Out Test Set (N=1,739)
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("EVALUATION ON HELD-OUT INTERNAL TEST PARTITION (KD1, N=1,739)")
    print("="*80)

    # XGBoost internal predictions
    y_pred_xgb_enc = best_xgb.predict(X_test)
    y_pred_xgb = le.inverse_transform(y_pred_xgb_enc)
    y_prob_xgb = best_xgb.predict_proba(X_test)

    # RF V2 internal predictions
    y_pred_rf_v2 = best_rf_v2.predict(X_test)
    y_prob_rf_v2 = best_rf_v2.predict_proba(X_test)

    m_xgb_internal = evaluate_model(y_test, y_pred_xgb, y_prob_xgb, "XGBoost (23 features)", "KD1 Internal Test")
    m_rf_v2_internal = evaluate_model(y_test, y_pred_rf_v2, y_prob_rf_v2, "Random Forest V2 (23 features)", "KD1 Internal Test")

    # -------------------------------------------------------------------------
    # 9. Evaluation on KD2 External Independent Validation Cohort (N=7,080)
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("EVALUATION ON EXTERNAL INDEPENDENT VALIDATION COHORT (KD2, N=7,080)")
    print("="*80)

    df_kd2 = pd.read_csv(kd2_feat_path)
    df_kd2[psi_cols] = df_kd2[psi_cols].fillna(0.0)

    strand_dummies_kd2 = pd.get_dummies(df_kd2["strand"], prefix="strand", dtype=float)
    if "strand_+" not in strand_dummies_kd2.columns:
        strand_dummies_kd2["strand_+"] = 0.0
    if "strand_-" not in strand_dummies_kd2.columns:
        strand_dummies_kd2["strand_-"] = 0.0
    strand_dummies_kd2 = strand_dummies_kd2[["strand_+", "strand_-"]]

    X_kd2 = pd.concat([df_kd2[numeric_features], strand_dummies_kd2], axis=1)
    y_kd2 = df_kd2["label"]

    assert list(X_kd2.columns) == feature_names, "Error: KD2 feature columns mismatch!"

    # XGBoost KD2 predictions
    y_pred_kd2_xgb_enc = best_xgb.predict(X_kd2)
    y_pred_kd2_xgb = le.inverse_transform(y_pred_kd2_xgb_enc)
    y_prob_kd2_xgb = best_xgb.predict_proba(X_kd2)

    # RF V2 KD2 predictions
    y_pred_kd2_rf_v2 = best_rf_v2.predict(X_kd2)
    y_prob_kd2_rf_v2 = best_rf_v2.predict_proba(X_kd2)

    m_xgb_kd2 = evaluate_model(y_kd2, y_pred_kd2_xgb, y_prob_kd2_xgb, "XGBoost (23 features)", "KD2 External Validation")
    m_rf_v2_kd2 = evaluate_model(y_kd2, y_pred_kd2_rf_v2, y_prob_kd2_rf_v2, "Random Forest V2 (23 features)", "KD2 External Validation")

    # Baseline RF metrics for reference
    baseline_rf_internal_acc = 0.7401
    baseline_rf_internal_f1 = 0.6474
    baseline_rf_kd2_acc = 0.7582
    baseline_rf_kd2_f1 = 0.6896

    # -------------------------------------------------------------------------
    # 10. Comparative Summary Table
    # -------------------------------------------------------------------------
    print("\n" + "="*95)
    print("COMPREHENSIVE PERFORMANCE COMPARISON TABLE: ROUND 2 ACCURACY IMPROVEMENTS")
    print("="*95)
    header = f"{'Evaluation Set':<16} {'Model':<30} {'Features':<10} {'Accuracy':<12} {'Macro-F1':<12} {'Weighted-F1':<12} {'ROC-AUC':<10}"
    print(header)
    print("-" * 95)
    print(f"{'Internal (KD1)':<16} {'Random Forest Baseline':<30} {'19 feats':<10} {baseline_rf_internal_acc*100:>6.2f}%{'':<5} {baseline_rf_internal_f1:>6.4f}{'':<5} {'0.7365':<12} {'0.9026':<10}")
    print(f"{'Internal (KD1)':<16} {'Random Forest V2':<30} {'23 feats':<10} {m_rf_v2_internal['accuracy']*100:>6.2f}%{'':<5} {m_rf_v2_internal['f1_macro']:>6.4f}{'':<5} {m_rf_v2_internal['f1_weighted']:>6.4f}{'':<5} {m_rf_v2_internal['roc_auc_ovr']:>6.4f}")
    print(f"{'Internal (KD1)':<16} {'XGBoost V2 (Tuned)':<30} {'23 feats':<10} {m_xgb_internal['accuracy']*100:>6.2f}%{'':<5} {m_xgb_internal['f1_macro']:>6.4f}{'':<5} {m_xgb_internal['f1_weighted']:>6.4f}{'':<5} {m_xgb_internal['roc_auc_ovr']:>6.4f}")
    print("-" * 95)
    print(f"{'External (KD2)':<16} {'Random Forest Baseline':<30} {'19 feats':<10} {baseline_rf_kd2_acc*100:>6.2f}%{'':<5} {baseline_rf_kd2_f1:>6.4f}{'':<5} {'0.7595':<12} {'0.9120':<10}")
    print(f"{'External (KD2)':<16} {'Random Forest V2':<30} {'23 feats':<10} {m_rf_v2_kd2['accuracy']*100:>6.2f}%{'':<5} {m_rf_v2_kd2['f1_macro']:>6.4f}{'':<5} {m_rf_v2_kd2['f1_weighted']:>6.4f}{'':<5} {m_rf_v2_kd2['roc_auc_ovr']:>6.4f}")
    print(f"{'External (KD2)':<16} {'XGBoost V2 (Tuned)':<30} {'23 feats':<10} {m_xgb_kd2['accuracy']*100:>6.2f}%{'':<5} {m_xgb_kd2['f1_macro']:>6.4f}{'':<5} {m_xgb_kd2['f1_weighted']:>6.4f}{'':<5} {m_xgb_kd2['roc_auc_ovr']:>6.4f}")
    print("=" * 95)

    # -------------------------------------------------------------------------
    # 11. Feature Importance Analysis
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE RANKINGS (23 Features)")
    print("="*80)

    # XGBoost Importances
    xgb_imp = best_xgb.feature_importances_
    xgb_imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": xgb_imp
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    print("\n--- XGBoost Feature Importances (Gain / Default) ---")
    new_feats = {"max_utract_length_upstream", "max_utract_length_downstream", "u_fraction_upstream", "u_fraction_downstream"}
    for idx, r in xgb_imp_df.iterrows():
        is_new = " [NEW U-TRACT FEATURE]" if r["feature"] in new_feats else ""
        print(f"  {idx+1:>2}. {r['feature']:<30}: {r['importance']:.4f}{is_new}")

    # RF V2 Importances
    rf_imp = best_rf_v2.feature_importances_
    rf_imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": rf_imp
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    print("\n--- Random Forest V2 Feature Importances (Gini Impurity) ---")
    for idx, r in rf_imp_df.iterrows():
        is_new = " [NEW U-TRACT FEATURE]" if r["feature"] in new_feats else ""
        print(f"  {idx+1:>2}. {r['feature']:<30}: {r['importance']:.4f}{is_new}")

    # Specifically report ranks of the 4 new features
    print("\n--- Ranks and Importance Values of the 4 New U-Tract Features ---")
    print(f"{'Feature Name':<30} {'XGBoost Rank':<15} {'XGB Importance':<18} {'RF V2 Rank':<15} {'RF V2 Importance':<18}")
    print("-" * 96)
    for feat in ["max_utract_length_upstream", "max_utract_length_downstream", "u_fraction_upstream", "u_fraction_downstream"]:
        xgb_rank = xgb_imp_df[xgb_imp_df["feature"] == feat].index[0] + 1
        xgb_val = xgb_imp_df[xgb_imp_df["feature"] == feat]["importance"].values[0]
        rf_rank = rf_imp_df[rf_imp_df["feature"] == feat].index[0] + 1
        rf_val = rf_imp_df[rf_imp_df["feature"] == feat]["importance"].values[0]
        print(f"{feat:<30} {f'#{xgb_rank} of 23':<15} {xgb_val:<18.4f} {f'#{rf_rank} of 23':<15} {rf_val:<18.4f}")

    # -------------------------------------------------------------------------
    # 12. Confusion Matrix & Feature Importance Plots
    # -------------------------------------------------------------------------
    def plot_cm(cm, title, filename):
        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        cax = ax.matshow(cm, cmap="Blues", alpha=0.85)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(x=j, y=i, s=f"{cm[i, j]}", va="center", ha="center", fontsize=11,
                        color="white" if cm[i, j] > cm.max()/2 else "black")
        fig.colorbar(cax)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, fontsize=10)
        ax.set_yticklabels(classes, fontsize=10)
        ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=10, pad=15)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Saved: {filename}")

    plot_cm(m_xgb_internal["confusion_matrix"], f"Confusion Matrix: XGBoost V2 on Internal KD1 Test\n(N={len(y_test):,})", os.path.join(models_dir, "confusion_matrix_xgb_internal_v2.png"))
    plot_cm(m_xgb_kd2["confusion_matrix"], f"Confusion Matrix: XGBoost V2 on External KD2 Cohort\n(N={len(y_kd2):,})", os.path.join(models_dir, "confusion_matrix_xgb_kd2_v2.png"))
    plot_cm(m_rf_v2_internal["confusion_matrix"], f"Confusion Matrix: Random Forest V2 on Internal KD1 Test\n(N={len(y_test):,})", os.path.join(models_dir, "confusion_matrix_rf_internal_v2.png"))
    plot_cm(m_rf_v2_kd2["confusion_matrix"], f"Confusion Matrix: Random Forest V2 on External KD2 Cohort\n(N={len(y_kd2):,})", os.path.join(models_dir, "confusion_matrix_rf_kd2_v2.png"))

    # Feature Importance Bar Charts
    def plot_imp(imp_df, title, filename):
        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
        top = imp_df.iloc[::-1]
        colors = ["#2ca02c" if feat in new_feats else "#1f77b4" for feat in top["feature"]]
        ax.barh(top["feature"], top["importance"], color=colors, edgecolor="black", height=0.7)
        ax.set_xlabel("Feature Importance", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Saved: {filename}")

    plot_imp(xgb_imp_df, "XGBoost V2: Feature Importances (23 Features)\nGreen = New HNRNPC U-Tract Features", os.path.join(models_dir, "feature_importance_xgb_v2.png"))
    plot_imp(rf_imp_df, "Random Forest V2: Feature Importances (23 Features)\nGreen = New HNRNPC U-Tract Features", os.path.join(models_dir, "feature_importance_rf_v2.png"))

    # Sync plots to Windows mount
    if os.path.exists(win_models_dir):
        import shutil
        for p in [
            "confusion_matrix_xgb_internal_v2.png", "confusion_matrix_xgb_kd2_v2.png",
            "confusion_matrix_rf_internal_v2.png", "confusion_matrix_rf_kd2_v2.png",
            "feature_importance_xgb_v2.png", "feature_importance_rf_v2.png"
        ]:
            src = os.path.join(models_dir, p)
            if os.path.exists(src):
                try:
                    shutil.copyfile(src, os.path.join(win_models_dir, p))
                except Exception as e:
                    print(f"Notice: Plot copy to Windows mount skipped ({e})")

    # -------------------------------------------------------------------------
    # 13. Write Evaluation Markdown Reports
    # -------------------------------------------------------------------------
    eval_report_md = f"""# Stages 6 & 7 V2: Model Evaluation & Diagnostic Report (Round 2 Improvements)

**Evaluation Partition**: Internal Held-Out Test Set ($N = {len(y_test):,}$, 20% Stratified Split, `random_state=42`)
**Underlying Comparison**: Depth-matched Control vs KD1 (`results/splicing_corrected_downsampled/`)
**Target Classes**: `{classes}`
**Feature Set**: 23 Features (17 numeric sequence/PWM/coverage/PSI + 4 new HNRNPC U-tract motif features + 2 strand dummies)

---

## 1. Overall Performance Comparison on Internal KD1 Test Set

| Metric | Target | RF Baseline (19 feats) | RF V2 (23 feats) | XGBoost V2 (23 feats) | Status vs Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | $\\ge 80.0\\%$ | **{baseline_rf_internal_acc*100:.2f}%** | **{m_rf_v2_internal['accuracy']*100:.2f}%** | **{m_xgb_internal['accuracy']*100:.2f}%** | **{m_xgb_internal['accuracy'] - baseline_rf_internal_acc:+.2%}** |
| **Macro-F1** | $\\ge 0.780$ | **{baseline_rf_internal_f1:.4f}** | **{m_rf_v2_internal['f1_macro']:.4f}** | **{m_xgb_internal['f1_macro']:.4f}** | **{m_xgb_internal['f1_macro'] - baseline_rf_internal_f1:+.4f}** |
| **Weighted-F1** | — | **0.7365** | **{m_rf_v2_internal['f1_weighted']:.4f}** | **{m_xgb_internal['f1_weighted']:.4f}** | **{m_xgb_internal['f1_weighted'] - 0.7365:+.4f}** |
| **Macro-Precision** | — | **0.7180** | **{m_rf_v2_internal['precision_macro']:.4f}** | **{m_xgb_internal['precision_macro']:.4f}** | — |
| **Macro-Recall** | — | **0.6083** | **{m_rf_v2_internal['recall_macro']:.4f}** | **{m_xgb_internal['recall_macro']:.4f}** | — |
| **ROC-AUC (OVR Macro)** | — | **0.9026** | **{m_rf_v2_internal['roc_auc_ovr']:.4f}** | **{m_xgb_internal['roc_auc_ovr']:.4f}** | — |

---

## 2. Per-Class Performance Breakdown (Internal KD1 Test Set)

### XGBoost V2 (Best params = `{xgb_best_params}`)
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
"""
    for c in classes:
        cd = m_xgb_internal["report_dict"][c]
        eval_report_md += f"| **{c}** | {cd['precision']:.4f} | {cd['recall']:.4f} | {cd['f1-score']:.4f} | {int(cd['support'])} |\n"

    eval_report_md += f"""
### Random Forest V2 (Best params = `{rf_best_params}`)
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
"""
    for c in classes:
        cd = m_rf_v2_internal["report_dict"][c]
        eval_report_md += f"| **{c}** | {cd['precision']:.4f} | {cd['recall']:.4f} | {cd['f1-score']:.4f} | {int(cd['support'])} |\n"

    eval_report_md += f"""
---

## 3. Confusion Matrices (Internal KD1 Test Set)

### XGBoost V2
```
{pd.DataFrame(m_xgb_internal['confusion_matrix'], index=classes, columns=classes).to_string()}
```

### Random Forest V2
```
{pd.DataFrame(m_rf_v2_internal['confusion_matrix'], index=classes, columns=classes).to_string()}
```

---

## 4. Feature Importance Rankings (23 Features)

### XGBoost V2 Feature Importances
| Rank | Feature Name | Importance (Gain) | Category |
| :--- | :--- | :--- | :--- |
"""
    for idx, r in xgb_imp_df.iterrows():
        cat = "**New U-Tract Motif**" if r["feature"] in new_feats else "Original Feature"
        eval_report_md += f"| {idx+1} | `{r['feature']}` | {r['importance']:.4f} | {cat} |\n"

    eval_report_path = os.path.join(models_dir, "evaluation_report_v2.md")
    with open(eval_report_path, "w") as f:
        f.write(eval_report_md)
    if os.path.exists(win_models_dir):
        with open(os.path.join(win_models_dir, "evaluation_report_v2.md"), "w") as f:
            f.write(eval_report_md)
    print(f"Saved internal evaluation report: {eval_report_path}")

    # KD2 External Validation Report
    kd2_report_md = f"""# Stage 8 V2: Independent KD2 External Validation Report (Round 2 Improvements)

**Models Evaluated**: `xgboost_v2.joblib` and `random_forest_v2.joblib` (Trained exclusively on KD1 with 23 features)
**External Validation Cohort**: HNRNPC Knockdown Construct 2 (`ERR127304`, `ERR127305`) vs Downsampled Control (`Control_rep1_downsampled`, `Control_rep2_downsampled`)
**Total Significant Events ($FDR < 0.05$)**: $N = {len(y_kd2):,}$
**Classes Evaluated**: `{classes}`

---

## 1. Generalization Performance Comparison (KD2 External Validation)

| Metric | RF Baseline (19 feats) | RF V2 (23 feats) | XGBoost V2 (23 feats) | $\\Delta$ vs RF Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | **{baseline_rf_kd2_acc*100:.2f}%** | **{m_rf_v2_kd2['accuracy']*100:.2f}%** | **{m_xgb_kd2['accuracy']*100:.2f}%** | **{(m_xgb_kd2['accuracy'] - baseline_rf_kd2_acc)*100:+.2f} pp** |
| **Macro-F1** | **{baseline_rf_kd2_f1:.4f}** | **{m_rf_v2_kd2['f1_macro']:.4f}** | **{m_xgb_kd2['f1_macro']:.4f}** | **{(m_xgb_kd2['f1_macro'] - baseline_rf_kd2_f1):+.4f}** |
| **Weighted-F1** | **0.7595** | **{m_rf_v2_kd2['f1_weighted']:.4f}** | **{m_xgb_kd2['f1_weighted']:.4f}** | **{(m_xgb_kd2['f1_weighted'] - 0.7595):+.4f}** |
| **Macro-Precision** | **0.7048** | **{m_rf_v2_kd2['precision_macro']:.4f}** | **{m_xgb_kd2['precision_macro']:.4f}** | — |
| **Macro-Recall** | **0.7016** | **{m_rf_v2_kd2['recall_macro']:.4f}** | **{m_xgb_kd2['recall_macro']:.4f}** | — |
| **ROC-AUC (OVR Macro)** | **0.9120** | **{m_rf_v2_kd2['roc_auc_ovr']:.4f}** | **{m_xgb_kd2['roc_auc_ovr']:.4f}** | — |

---

## 2. Per-Class Breakdown on Independent KD2 Dataset

### XGBoost V2 on KD2
| Class | Precision | Recall | F1-Score | Support | Class Share |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for c in classes:
        cd = m_xgb_kd2["report_dict"][c]
        pct = cd['support'] / len(y_kd2) * 100.0
        kd2_report_md += f"| **{c}** | {cd['precision']:.4f} | {cd['recall']:.4f} | {cd['f1-score']:.4f} | {int(cd['support']):,} | {pct:.2f}% |\n"

    kd2_report_md += f"""
### Random Forest V2 on KD2
| Class | Precision | Recall | F1-Score | Support | Class Share |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for c in classes:
        cd = m_rf_v2_kd2["report_dict"][c]
        pct = cd['support'] / len(y_kd2) * 100.0
        kd2_report_md += f"| **{c}** | {cd['precision']:.4f} | {cd['recall']:.4f} | {cd['f1-score']:.4f} | {int(cd['support']):,} | {pct:.2f}% |\n"

    kd2_report_md += f"""
---

## 3. Confusion Matrix: KD2 Independent Cohort

### XGBoost V2 on KD2
```
{pd.DataFrame(m_xgb_kd2['confusion_matrix'], index=classes, columns=classes).to_string()}
```

### Random Forest V2 on KD2
```
{pd.DataFrame(m_rf_v2_kd2['confusion_matrix'], index=classes, columns=classes).to_string()}
```

---

## 4. Generalization Retention (Internal KD1 vs External KD2)

| Model | Internal Test Acc | KD2 External Acc | Retention $\\Delta$ Acc | Internal Test Macro-F1 | KD2 External Macro-F1 | Retention $\\Delta$ Macro-F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RF Baseline (19 feats)** | 74.01% | 75.82% | +1.81 pp | 0.6474 | 0.6896 | +0.0422 |
| **RF V2 (23 feats)** | {m_rf_v2_internal['accuracy']*100:.2f}% | {m_rf_v2_kd2['accuracy']*100:.2f}% | {(m_rf_v2_kd2['accuracy'] - m_rf_v2_internal['accuracy'])*100:+.2f} pp | {m_rf_v2_internal['f1_macro']:.4f} | {m_rf_v2_kd2['f1_macro']:.4f} | {(m_rf_v2_kd2['f1_macro'] - m_rf_v2_internal['f1_macro']):+.4f} |
| **XGBoost V2 (23 feats)** | {m_xgb_internal['accuracy']*100:.2f}% | {m_xgb_kd2['accuracy']*100:.2f}% | {(m_xgb_kd2['accuracy'] - m_xgb_internal['accuracy'])*100:+.2f} pp | {m_xgb_internal['f1_macro']:.4f} | {m_xgb_kd2['f1_macro']:.4f} | {(m_xgb_kd2['f1_macro'] - m_xgb_internal['f1_macro']):+.4f} |
"""
    kd2_report_path = os.path.join(models_dir, "kd2_validation_report_v2.md")
    with open(kd2_report_path, "w") as f:
        f.write(kd2_report_md)
    if os.path.exists(win_models_dir):
        with open(os.path.join(win_models_dir, "kd2_validation_report_v2.md"), "w") as f:
            f.write(kd2_report_md)
    print(f"Saved KD2 validation report: {kd2_report_path}")

if __name__ == "__main__":
    main()
