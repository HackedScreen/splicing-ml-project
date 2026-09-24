#!/usr/bin/env python3
"""
scripts/06_train_evaluate_corrected.py

BSc Dissertation Pipeline — Stages 6 & 7: Corrected Model Training, Tuning & Evaluation
Trains and evaluates Logistic Regression and Random Forest classifiers on the corrected,
depth-balanced alternative splicing dataset (results/features/feature_matrix_corrected.csv).

Uses the Stage 7b robust 19-feature standard:
  - 17 numeric features: exon_length, gc_exon, gc_upstream_100bp, gc_downstream_100bp,
    score_5ss_donor, score_3ss_acceptor, cov_mean_control, cov_var_control, cov_mean_kd,
    cov_var_kd, cov_mean_overall, cov_var_overall, psi_control_mean, psi_control_var,
    psi_kd_mean, psi_kd_var, dpsi.
  - 2 strand one-hot features: strand_+, strand_-.
  - EXCLUDES: upstream_intron_length and downstream_intron_length (prevents RI coordinate artifact).
  - EXCLUDES: pvalue and fdr (prevents testing power artifact / label leakage).

Outputs:
  results/models/logistic_regression_corrected.joblib
  results/models/random_forest_corrected.joblib
  results/models/train_test_data_corrected.joblib
  results/models/confusion_matrix_lr_corrected.png
  results/models/confusion_matrix_rf_corrected.png
  results/models/feature_importance_rf_corrected.png
  results/models/evaluation_report_corrected.md
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

def main():
    base_dir = os.path.expanduser("~/splicing-ml-project")
    results_dir = os.path.join(base_dir, "results")
    feat_path = os.path.join(results_dir, "features", "feature_matrix_corrected.csv")
    models_dir = os.path.join(results_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    print(f"Loading corrected feature matrix from: {feat_path}")
    if not os.path.exists(feat_path):
        print(f"ERROR: {feat_path} does not exist.")
        sys.exit(1)

    df = pd.read_csv(feat_path)
    print(f"Loaded {len(df):,} events and {len(df.columns)} columns.")

    # -------------------------------------------------------------------------
    # 1. Missing Value Imputation
    # -------------------------------------------------------------------------
    psi_cols = ["psi_control_mean", "psi_control_var", "psi_kd_mean", "psi_kd_var"]
    nan_before = df[psi_cols].isnull().sum().to_dict()
    print(f"\nMissing values in PSI columns before imputation: {nan_before}")
    df[psi_cols] = df[psi_cols].fillna(0.0)
    print("Imputed all PSI NaN values with 0.0 (biologically: zero observed reads = 0 inclusion).")

    # Verify no remaining NaNs in features
    # -------------------------------------------------------------------------
    # 2. Feature Selection (Robust 19-Feature Standard)
    # -------------------------------------------------------------------------
    numeric_features = [
        "exon_length",
        "gc_exon",
        "gc_upstream_100bp",
        "gc_downstream_100bp",
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

    # One-hot encode strand (+ and -) -> 2 columns, total 19 features
    strand_dummies = pd.get_dummies(df["strand"], prefix="strand", dtype=float)
    if "strand_+" not in strand_dummies.columns:
        strand_dummies["strand_+"] = 0.0
    if "strand_-" not in strand_dummies.columns:
        strand_dummies["strand_-"] = 0.0
    strand_dummies = strand_dummies[["strand_+", "strand_-"]]

    X = pd.concat([df[numeric_features], strand_dummies], axis=1)
    y = df["label"]
    feature_names = list(X.columns)

    print(f"\nFinal feature set ({len(feature_names)} features):")
    for i, fn in enumerate(feature_names, 1):
        print(f"  {i:>2}. {fn}")

    print("\nTarget class distribution across full dataset:")
    class_counts = y.value_counts()
    for lbl, count in class_counts.items():
        pct = count / len(y) * 100.0
        print(f"  {lbl:<6}: {count:>6,} ({pct:>5.2f}%)")

    # -------------------------------------------------------------------------
    # 3. Stratified 80/20 Train/Test Split (fixed random_state=42)
    # -------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    print(f"\nTrain set: {X_train.shape[0]:,} samples (80.0%)")
    print(f"Test set (held-out): {X_test.shape[0]:,} samples (20.0%)")
    print("\nTrain class distribution:\n", y_train.value_counts())
    print("\nTest class distribution:\n", y_test.value_counts())

    classes = sorted(list(np.unique(y)))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # -------------------------------------------------------------------------
    # 4. Logistic Regression Training & GridSearch Tuning
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("TRAINING LOGISTIC REGRESSION (Multinomial, Balanced Class Weights)")
    print("="*70)

    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            class_weight="balanced",
            solver="lbfgs",
            max_iter=2000,
            random_state=42
        ))
    ])

    lr_param_grid = {
        "lr__C": [0.01, 0.1, 1, 10]
    }

    lr_grid = GridSearchCV(
        lr_pipeline,
        param_grid=lr_param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    lr_grid.fit(X_train, y_train)

    print(f"\nLogistic Regression GridSearch Results:")
    for params, mean_score, std_score in zip(
        lr_grid.cv_results_["params"],
        lr_grid.cv_results_["mean_test_score"],
        lr_grid.cv_results_["std_test_score"]
    ):
        print(f"  Params: {params} -> Mean 5-Fold F1-Macro: {mean_score:.4f} (+/- {std_score:.4f})")

    best_lr = lr_grid.best_estimator_
    print(f"\nBest Logistic Regression Params: {lr_grid.best_params_}")
    print(f"Best Logistic Regression 5-Fold CV Score (F1-Macro): {lr_grid.best_score_:.4f}")

    # -------------------------------------------------------------------------
    # 5. Random Forest Training & GridSearch Tuning
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("TRAINING RANDOM FOREST (Balanced Class Weights)")
    print("="*70)

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

    print(f"\nRandom Forest GridSearch Results:")
    for params, mean_score, std_score in zip(
        rf_grid.cv_results_["params"],
        rf_grid.cv_results_["mean_test_score"],
        rf_grid.cv_results_["std_test_score"]
    ):
        print(f"  Params: {params} -> Mean 5-Fold F1-Macro: {mean_score:.4f} (+/- {std_score:.4f})")

    best_rf = rf_grid.best_estimator_
    print(f"\nBest Random Forest Params: {rf_grid.best_params_}")
    print(f"Best Random Forest 5-Fold CV Score (F1-Macro): {rf_grid.best_score_:.4f}")

    # -------------------------------------------------------------------------
    # 6. Save Model Artifacts
    # -------------------------------------------------------------------------
    lr_model_path = os.path.join(models_dir, "logistic_regression_corrected.joblib")
    rf_model_path = os.path.join(models_dir, "random_forest_corrected.joblib")
    split_data_path = os.path.join(models_dir, "train_test_data_corrected.joblib")

    joblib.dump(best_lr, lr_model_path)
    joblib.dump(best_rf, rf_model_path)
    joblib.dump({
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names
    }, split_data_path)

    print("\nSaved Models:")
    print(f"  - LR: {lr_model_path}")
    print(f"  - RF: {rf_model_path}")
    print(f"  - Data: {split_data_path}")

    # -------------------------------------------------------------------------
    # 7. Evaluation on Held-Out Test Set (N=1,739)
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("EVALUATION ON HELD-OUT TEST PARTITION")
    print("="*70)

    y_pred_lr = best_lr.predict(X_test)
    y_prob_lr = best_lr.predict_proba(X_test)

    y_pred_rf = best_rf.predict(X_test)
    y_prob_rf = best_rf.predict_proba(X_test)

    def calc_metrics(y_true, y_pred, y_prob, model_name):
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

    m_lr = calc_metrics(y_test, y_pred_lr, y_prob_lr, "Logistic Regression")
    m_rf = calc_metrics(y_test, y_pred_rf, y_prob_rf, "Random Forest")

    # Feature Importances for RF
    rf_importances = best_rf.feature_importances_
    feat_imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": rf_importances
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    print("\n--- Model Comparison on Held-Out Test Set ---")
    print(f"{'Metric':<25} {'Target':<10} {'Logistic Regression':<22} {'Random Forest':<22}")
    print("-" * 79)
    print(f"{'Accuracy':<25} {'>= 80.0%':<10} {m_lr['accuracy']*100:>6.2f}% ({'PASS' if m_lr['accuracy']>=0.8 else 'FAIL'}){'':<8} {m_rf['accuracy']*100:>6.2f}% ({'PASS' if m_rf['accuracy']>=0.8 else 'FAIL'})")
    print(f"{'Macro-F1':<25} {'>= 0.780':<10} {m_lr['f1_macro']:>6.4f} ({'PASS' if m_lr['f1_macro']>=0.78 else 'FAIL'}){'':<9} {m_rf['f1_macro']:>6.4f} ({'PASS' if m_rf['f1_macro']>=0.78 else 'FAIL'})")
    print(f"{'Weighted-F1':<25} {'—':<10} {m_lr['f1_weighted']:>6.4f}{'':<17} {m_rf['f1_weighted']:>6.4f}")
    print(f"{'Macro-Precision':<25} {'—':<10} {m_lr['precision_macro']:>6.4f}{'':<17} {m_rf['precision_macro']:>6.4f}")
    print(f"{'Macro-Recall':<25} {'—':<10} {m_lr['recall_macro']:>6.4f}{'':<17} {m_rf['recall_macro']:>6.4f}")
    print(f"{'ROC-AUC (OVR Macro)':<25} {'—':<10} {m_lr['roc_auc_ovr']:>6.4f}{'':<17} {m_rf['roc_auc_ovr']:>6.4f}")

    print("\n--- Random Forest Feature Importances ---")
    dominant_features = []
    for idx, r in feat_imp_df.iterrows():
        flag = " [>15% DOMINANT]" if r["importance"] > 0.15 else ""
        if r["importance"] > 0.15:
            dominant_features.append((r["feature"], r["importance"]))
        print(f"  {idx+1:>2}. {r['feature']:<25}: {r['importance']:.4f}{flag}")

    if dominant_features:
        print(f"\nWARNING: Features exceeding 15% threshold: {dominant_features}")
    else:
        print("\nAll feature importances are <= 15% (no single dominant feature artifact).")

    # -------------------------------------------------------------------------
    # 8. Visualizations
    # -------------------------------------------------------------------------
    def plot_confusion_matrix(cm, model_name, filename):
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
        ax.set_title(f"Confusion Matrix: {model_name}\n(Corrected Held-out Test Set, N={len(y_test)})", fontsize=11, pad=15)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Saved: {filename}")

    cm_lr_path = os.path.join(models_dir, "confusion_matrix_lr_corrected.png")
    cm_rf_path = os.path.join(models_dir, "confusion_matrix_rf_corrected.png")
    plot_confusion_matrix(m_lr["confusion_matrix"], "Logistic Regression", cm_lr_path)
    plot_confusion_matrix(m_rf["confusion_matrix"], "Random Forest", cm_rf_path)

    # Feature Importance Plot
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    top_feats = feat_imp_df.iloc[::-1]
    colors = ["#d95f02" if imp > 0.15 else "#2b5c8f" for imp in top_feats["importance"]]
    ax.barh(top_feats["feature"], top_feats["importance"], color=colors, edgecolor="black", height=0.7)
    ax.axvline(0.15, color="red", linestyle="--", linewidth=1.2, label="15% Threshold")
    ax.set_xlabel("Gini Feature Importance", fontsize=11, fontweight="bold")
    ax.set_title("Corrected Random Forest: Feature Importances (19 Features)", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower right")
    plt.tight_layout()
    feat_plot_path = os.path.join(models_dir, "feature_importance_rf_corrected.png")
    plt.savefig(feat_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {feat_plot_path}")

    # -------------------------------------------------------------------------
    # 9. Generate Evaluation Report
    # -------------------------------------------------------------------------
    target_acc = 0.80
    target_f1 = 0.78
    lr_meets_target = (m_lr["accuracy"] >= target_acc) and (m_lr["f1_macro"] >= target_f1)
    rf_meets_target = (m_rf["accuracy"] >= target_acc) and (m_rf["f1_macro"] >= target_f1)

    report_md = f"""# Stages 6 & 7: Corrected Model Evaluation & Diagnostic Report

**Evaluation Dataset**: Corrected Held-out Test Partition ($N = {len(y_test):,}$, 20% Stratified Split, `random_state=42`)
**Underlying Comparison**: Depth-matched Control vs KD1 (`results/splicing_corrected_downsampled/`)
**Target Classes**: `{classes}`
**Feature Standard**: 19 Features (17 numeric + 2 strand dummies; intron lengths & pvalue/fdr excluded)

---

## 1. Overall Model Performance vs Dissertation Target

* **Dissertation Abstract Target**: Accuracy $\\ge 80.0\\%$, Macro-F1 $\\ge 0.780$.

| Metric | Target | Logistic Regression (Baseline) | Random Forest (Ensemble) |
| :--- | :--- | :--- | :--- |
| **Accuracy** | $\\ge 80.0\\%$ | **{m_lr['accuracy']*100:.2f}%** ({'PASS' if m_lr['accuracy']>=0.8 else 'FAIL'}) | **{m_rf['accuracy']*100:.2f}%** ({'PASS' if m_rf['accuracy']>=0.8 else 'FAIL'}) |
| **Macro-F1** | $\\ge 0.780$ | **{m_lr['f1_macro']:.4f}** ({'PASS' if m_lr['f1_macro']>=0.78 else 'FAIL'}) | **{m_rf['f1_macro']:.4f}** ({'PASS' if m_rf['f1_macro']>=0.78 else 'FAIL'}) |
| **Weighted-F1** | — | **{m_lr['f1_weighted']:.4f}** | **{m_rf['f1_weighted']:.4f}** |
| **Macro-Precision** | — | **{m_lr['precision_macro']:.4f}** | **{m_rf['precision_macro']:.4f}** |
| **Macro-Recall** | — | **{m_lr['recall_macro']:.4f}** | **{m_rf['recall_macro']:.4f}** |
| **ROC-AUC (OVR Macro)**| — | **{m_lr['roc_auc_ovr']:.4f}** | **{m_rf['roc_auc_ovr']:.4f}** |
| **Target Status** | — | **{'MET' if lr_meets_target else 'NOT MET'}** | **{'MET' if rf_meets_target else 'NOT MET'}** |

---

## 2. Per-Class Performance Breakdown

### Logistic Regression (Best C = {lr_grid.best_params_['lr__C']})
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
"""
    for c in classes:
        cd = m_lr["report_dict"][c]
        report_md += f"| **{c}** | {cd['precision']:.4f} | {cd['recall']:.4f} | {cd['f1-score']:.4f} | {int(cd['support'])} |\n"

    report_md += f"""
### Random Forest (Best params = {rf_grid.best_params_})
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
"""
    for c in classes:
        cd = m_rf["report_dict"][c]
        report_md += f"| **{c}** | {cd['precision']:.4f} | {cd['recall']:.4f} | {cd['f1-score']:.4f} | {int(cd['support'])} |\n"

    report_md += f"""
---

## 3. Confusion Matrices

### Logistic Regression
```
{pd.DataFrame(m_lr['confusion_matrix'], index=classes, columns=classes).to_string()}
```

### Random Forest
```
{pd.DataFrame(m_rf['confusion_matrix'], index=classes, columns=classes).to_string()}
```

---

## 4. Random Forest Feature Importances (All 19 Features)

| Rank | Feature Name | Gini Importance | Cumulative Importance | Flag |
| :--- | :--- | :--- | :--- | :--- |
"""
    cum_imp = 0.0
    for idx, r in feat_imp_df.iterrows():
        cum_imp += r["importance"]
        flag = "**>15% DOMINANT**" if r["importance"] > 0.15 else "Normal"
        report_md += f"| {idx+1} | `{r['feature']}` | {r['importance']:.4f} | {cum_imp:.4f} | {flag} |\n"

    report_path = os.path.join(models_dir, "evaluation_report_corrected.md")
    with open(report_path, "w") as f:
        f.write(report_md)
    print(f"\nSaved evaluation report: {report_path}")

if __name__ == "__main__":
    main()
