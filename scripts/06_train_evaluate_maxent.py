#!/usr/bin/env python3
"""
scripts/06_train_evaluate_maxent.py

BSc Dissertation Pipeline — Stages 6 & 7 V3:
Trains and evaluates Random Forest on the 23-feature dataset upgraded with
genuine Yeo & Burge MaxEntScan splice site scores (score_5ss_donor, score_3ss_acceptor).

Uses the exact same held-out train/test split:
  - Training: N = 6,954 (80.0%)
  - Held-out Internal Test: N = 1,739 (20.0%)
  - External KD2 Independent Validation: N = 7,080

Evaluates:
  1. Internal KD1 Held-Out Test Set (N = 1,739)
  2. External KD2 Independent Biological Validation Cohort (N = 7,080)

Outputs:
  results/models/random_forest_v3_maxent.joblib
  results/models/train_test_data_v3_maxent.joblib
  results/models/confusion_matrix_rf_internal_v3_maxent.png
  results/models/confusion_matrix_rf_kd2_v3_maxent.png
  results/models/feature_importance_rf_v3_maxent.png
  results/models/maxent_evaluation_report.md
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

def plot_confusion_matrix(cm, classes, title, output_path):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )
    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def evaluate_model(y_true, y_pred, y_prob, model_name, cohort_name, classes):
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    try:
        le = LabelEncoder()
        le.fit(classes)
        y_true_enc = le.transform(y_true)
        roc_auc_ovr = roc_auc_score(y_true_enc, y_prob, multi_class="ovr", average="macro")
    except Exception:
        roc_auc_ovr = np.nan

    print(f"\n==================================================================")
    print(f" {model_name} — {cohort_name} ")
    print(f"==================================================================")
    print(f"  Accuracy:         {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Macro-Precision:  {prec_macro:.4f}")
    print(f"  Macro-Recall:     {rec_macro:.4f}")
    print(f"  Macro-F1:         {f1_macro:.4f}")
    print(f"  Weighted-F1:      {f1_weighted:.4f}")
    print(f"  ROC-AUC (OvR):    {roc_auc_ovr:.4f}")
    print("\nDetailed Per-Class Performance:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "roc_auc_ovr": roc_auc_ovr
    }

def main():
    base_dir = os.path.expanduser("~/splicing-ml-project")
    features_dir = os.path.join(base_dir, "results", "features")
    models_dir = os.path.join(base_dir, "results", "models")
    os.makedirs(models_dir, exist_ok=True)

    kd1_feat_path = os.path.join(features_dir, "feature_matrix_v3_maxent.csv")
    kd2_feat_path = os.path.join(features_dir, "feature_matrix_kd2_v3_maxent.csv")
    prev_split_path = os.path.join(models_dir, "train_test_data_corrected.joblib")

    print(f"Loading V3 MaxEntScan feature matrix: {kd1_feat_path}")
    if not os.path.exists(kd1_feat_path):
        print(f"ERROR: {kd1_feat_path} does not exist.")
        sys.exit(1)

    df_kd1 = pd.read_csv(kd1_feat_path)
    print(f"Loaded KD1: {len(df_kd1):,} events and {len(df_kd1.columns)} columns.")

    print(f"Loading previous train/test split: {prev_split_path}")
    prev_split = joblib.load(prev_split_path)
    train_idx = prev_split["X_train"].index
    test_idx = prev_split["X_test"].index

    # Missing Value Imputation
    psi_cols = ["psi_control_mean", "psi_control_var", "psi_kd_mean", "psi_kd_var"]
    df_kd1[psi_cols] = df_kd1[psi_cols].fillna(0.0)

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

    X_train = X_full.loc[train_idx]
    X_test = X_full.loc[test_idx]
    y_train = y_full.loc[train_idx]
    y_test = y_full.loc[test_idx]

    assert (y_train.values == prev_split["y_train"].values).all(), "Target mismatch!"
    assert (y_test.values == prev_split["y_test"].values).all(), "Target mismatch!"

    classes = sorted(list(np.unique(y_train)))
    le = LabelEncoder()
    le.fit(classes)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)

    print("\n" + "="*80)
    print("TRAINING RANDOM FOREST (MAXENTSCAN UPGRADE, 5-FOLD CV)")
    print("="*80)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_clf = RandomForestClassifier(random_state=42, n_jobs=-1)
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
        verbose=1
    )
    rf_grid.fit(X_train, y_train)

    print(f"Best RF Parameters: {rf_grid.best_params_}")
    print(f"Best CV Macro-F1: {rf_grid.best_score_:.4f}")
    best_rf = rf_grid.best_estimator_

    # Save models
    rf_model_path = os.path.join(models_dir, "random_forest_v3_maxent.joblib")
    split_data_path = os.path.join(models_dir, "train_test_data_v3_maxent.joblib")

    joblib.dump(best_rf, rf_model_path)
    joblib.dump({
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feature_names,
        "classes": classes,
        "label_encoder": le
    }, split_data_path)
    print(f"Saved model to {rf_model_path}")
    print(f"Saved split data to {split_data_path}")

    # Evaluate on Internal Test Set
    y_pred_test = best_rf.predict(X_test)
    y_prob_test = best_rf.predict_proba(X_test)
    m_internal = evaluate_model(y_test, y_pred_test, y_prob_test, "Random Forest V3 (MaxEntScan)", "Internal KD1 Test", classes)

    cm_internal = confusion_matrix(y_test, y_pred_test, labels=classes)
    cm_int_path = os.path.join(models_dir, "confusion_matrix_rf_internal_v3_maxent.png")
    plot_confusion_matrix(cm_internal, classes, "Confusion Matrix — RF V3 MaxEnt (Internal Test)", cm_int_path)

    # Evaluate on KD2 External Validation
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
    assert list(X_kd2.columns) == feature_names, "KD2 column mismatch!"

    y_pred_kd2 = best_rf.predict(X_kd2)
    y_prob_kd2 = best_rf.predict_proba(X_kd2)
    m_kd2 = evaluate_model(y_kd2, y_pred_kd2, y_prob_kd2, "Random Forest V3 (MaxEntScan)", "KD2 External Validation", classes)

    cm_kd2 = confusion_matrix(y_kd2, y_pred_kd2, labels=classes)
    cm_kd2_path = os.path.join(models_dir, "confusion_matrix_rf_kd2_v3_maxent.png")
    plot_confusion_matrix(cm_kd2, classes, "Confusion Matrix — RF V3 MaxEnt (KD2 External)", cm_kd2_path)

    # Feature Importances
    rf_imp = best_rf.feature_importances_
    rf_imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": rf_imp
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    print("\n" + "="*80)
    print("FEATURE IMPORTANCE RANKINGS (RF V3 MaxEntScan)")
    print("="*80)
    for idx, r in rf_imp_df.iterrows():
        highlight = ""
        if "score_" in r["feature"]:
            highlight = " [MAXENTSCAN SPLICE SITE]"
        elif "utract" in r["feature"] or "u_fraction" in r["feature"]:
            highlight = " [U-TRACT MOTIF]"
        print(f"  {idx+1:>2}. {r['feature']:<30}: {r['importance']:.4f}{highlight}")

    # Plot feature importances
    plt.figure(figsize=(10, 8), dpi=300)
    colors = []
    for f in rf_imp_df["feature"][::-1]:
        if "score_" in f:
            colors.append("#2ca02c") # green for MaxEntScan
        elif "utract" in f or "u_fraction" in f:
            colors.append("#ff7f0e") # orange for U-tract
        else:
            colors.append("#1f77b4") # blue for original
    plt.barh(rf_imp_df["feature"][::-1], rf_imp_df["importance"][::-1], color=colors)
    plt.xlabel("Gini Importance")
    plt.title("Random Forest V3 Feature Importance (MaxEntScan + U-Tract Augmented)")
    plt.tight_layout()
    feat_imp_path = os.path.join(models_dir, "feature_importance_rf_v3_maxent.png")
    plt.savefig(feat_imp_path, dpi=300)
    plt.close()

    # Save Markdown Report
    report_path = os.path.join(models_dir, "maxent_evaluation_report.md")
    with open(report_path, "w") as f:
        f.write("# MaxEntScan Upgrade Evaluation Report\n\n")
        f.write("## Performance Summary\n\n")
        f.write("| Evaluation Set | Model | Features | Accuracy | Macro-F1 | Weighted-F1 | ROC-AUC |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        f.write(f"| Internal (KD1) | RF V2 (Additive PWM) | 23 | 74.81% | 0.6611 | 0.7423 | 0.9080 |\n")
        f.write(f"| Internal (KD1) | RF V3 (MaxEntScan) | 23 | {m_internal['accuracy']*100:.2f}% | {m_internal['f1_macro']:.4f} | {m_internal['f1_weighted']:.4f} | {m_internal['roc_auc_ovr']:.4f} |\n")
        f.write(f"| External (KD2) | RF V2 (Additive PWM) | 23 | 76.82% | 0.7003 | 0.7672 | 0.9168 |\n")
        f.write(f"| External (KD2) | RF V3 (MaxEntScan) | 23 | {m_kd2['accuracy']*100:.2f}% | {m_kd2['f1_macro']:.4f} | {m_kd2['f1_weighted']:.4f} | {m_kd2['roc_auc_ovr']:.4f} |\n\n")
        f.write("## Best Hyperparameters\n\n")
        f.write(f"```python\n{rf_grid.best_params_}\n```\n")

    print(f"\nReport written to {report_path}")

if __name__ == "__main__":
    main()
