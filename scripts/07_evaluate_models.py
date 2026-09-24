#!/usr/bin/env python3
"""
scripts/07_evaluate_models.py

BSc Dissertation Pipeline — Stage 7: Model Evaluation & Diagnostic Analysis
Evaluates Logistic Regression and Random Forest models on the held-out test set (3,818 samples).
Computes performance metrics, confusion matrices, ROC-AUC, feature importances,
and conducts a diagnostic audit for feature construction artifacts.

Outputs:
  results/models/evaluation_report.md
  results/models/confusion_matrix_lr.png
  results/models/confusion_matrix_rf.png
  results/models/feature_importance_rf.png
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

def main():
    base_dir = os.path.expanduser("~/splicing-ml-project")
    models_dir = os.path.join(base_dir, "results", "models")
    feat_path = os.path.join(base_dir, "results", "features", "feature_matrix.csv")

    data_path = os.path.join(models_dir, "train_test_data.joblib")
    lr_path = os.path.join(models_dir, "logistic_regression.joblib")
    rf_path = os.path.join(models_dir, "random_forest.joblib")

    print(f"Loading data split: {data_path}")
    data = joblib.load(data_path)
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    feature_names = data["feature_names"]

    print(f"Loading models:\n  LR: {lr_path}\n  RF: {rf_path}")
    lr_model = joblib.load(lr_path)
    rf_model = joblib.load(rf_path)

    classes = sorted(list(np.unique(y_test)))
    print(f"Target classes ({len(classes)}): {classes}")
    print(f"Held-out test set size: {len(y_test)} samples")

    # -------------------------------------------------------------------------
    # 1. Predictions & Probability Estimates on Held-Out Test Set
    # -------------------------------------------------------------------------
    y_pred_lr = lr_model.predict(X_test)
    y_prob_lr = lr_model.predict_proba(X_test)

    y_pred_rf = rf_model.predict(X_test)
    y_prob_rf = rf_model.predict_proba(X_test)

    # -------------------------------------------------------------------------
    # 2. Metric Computation
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 3. Diagnostic Audit: Intron Length Symmetry in RI Events
    # -------------------------------------------------------------------------
    df_feat = pd.read_csv(feat_path)
    ri_total = df_feat[df_feat["label"] == "RI"]
    ri_exact_intron_match = (ri_total["upstream_intron_length"] == ri_total["downstream_intron_length"]).sum()
    ri_fraction_match = ri_exact_intron_match / len(ri_total) if len(ri_total) > 0 else 0.0

    # In test set specifically
    ri_test_mask = (y_test == "RI")
    X_test_ri = X_test[ri_test_mask]
    ri_test_match = (X_test_ri["upstream_intron_length"] == X_test_ri["downstream_intron_length"]).sum()
    ri_test_fraction = ri_test_match / len(X_test_ri) if len(X_test_ri) > 0 else 0.0

    ri_prec_lr = m_lr["report_dict"]["RI"]["precision"]
    ri_rec_lr = m_lr["report_dict"]["RI"]["recall"]
    ri_f1_lr = m_lr["report_dict"]["RI"]["f1-score"]

    ri_prec_rf = m_rf["report_dict"]["RI"]["precision"]
    ri_rec_rf = m_rf["report_dict"]["RI"]["recall"]
    ri_f1_rf = m_rf["report_dict"]["RI"]["f1-score"]

    # Extract Feature Importances for RF
    rf_importances = rf_model.feature_importances_
    feat_imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": rf_importances
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    top5_features = feat_imp_df["feature"].head(5).tolist()
    top10_features = feat_imp_df.head(10)

    artifact_flag = (
        ("upstream_intron_length" in top5_features or "downstream_intron_length" in top5_features)
        and (ri_prec_rf > 0.95 and ri_rec_rf > 0.95)
    )

    # -------------------------------------------------------------------------
    # 4. Generate Visualizations (Confusion Matrices & Feature Importance Plot)
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
        ax.set_title(f"Confusion Matrix: {model_name}\n(Held-out Test Set, N={len(y_test)})", fontsize=12, pad=15)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Saved: {filename}")

    cm_lr_path = os.path.join(models_dir, "confusion_matrix_lr.png")
    cm_rf_path = os.path.join(models_dir, "confusion_matrix_rf.png")
    plot_confusion_matrix(m_lr["confusion_matrix"], "Logistic Regression", cm_lr_path)
    plot_confusion_matrix(m_rf["confusion_matrix"], "Random Forest", cm_rf_path)

    # Feature Importance Plot
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    top15 = feat_imp_df.head(15).iloc[::-1]
    ax.barh(top15["feature"], top15["importance"], color="#2b5c8f", edgecolor="black", height=0.7)
    ax.set_xlabel("Gini Feature Importance", fontsize=11, fontweight="bold")
    ax.set_title("Random Forest Classifier: Top-15 Feature Importances", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    feat_plot_path = os.path.join(models_dir, "feature_importance_rf.png")
    plt.savefig(feat_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {feat_plot_path}")

    # -------------------------------------------------------------------------
    # 5. Extract Logistic Regression Coefficients per Class
    # -------------------------------------------------------------------------
    lr_estimator = lr_model.named_steps["lr"]
    lr_coef_df = pd.DataFrame(
        lr_estimator.coef_,
        index=lr_estimator.classes_,
        columns=feature_names
    )

    # -------------------------------------------------------------------------
    # 6. Generate Markdown Evaluation Report
    # -------------------------------------------------------------------------
    target_acc = 0.80
    target_f1 = 0.78

    lr_meets_target = (m_lr["accuracy"] >= target_acc) and (m_lr["f1_macro"] >= target_f1)
    rf_meets_target = (m_rf["accuracy"] >= target_acc) and (m_rf["f1_macro"] >= target_f1)

    report_md = f"""# Stage 7: Model Evaluation & Methodology Diagnostic Report

**Evaluation Dataset**: Held-out Test Partition ($N = 3,818$, 20% Stratified Split, `random_state=42`)
**Classes**: `{classes}`

---

## 1. Overall Model Performance vs Dissertation Target

* **Dissertation Abstract Criteria**: Accuracy $\\ge 80.0\\%$, Macro-F1 $\\ge 0.780$.

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

## 2. Per-Class Breakdown

### Logistic Regression (Multinomial, C=0.1)
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
"""
    for c in classes:
        cd = m_lr["report_dict"][c]
        report_md += f"| **{c}** | {cd['precision']:.4f} | {cd['recall']:.4f} | {cd['f1-score']:.4f} | {int(cd['support'])} |\n"

    report_md += """
### Random Forest (n_estimators=300, max_depth=None)
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

## 4. Random Forest Feature Importances (Top 10)

| Rank | Feature Name | Gini Importance | Cumulative Importance |
| :--- | :--- | :--- | :--- |
"""
    cum_imp = 0.0
    for idx, r in top10_features.iterrows():
        cum_imp += r["importance"]
        report_md += f"| {idx+1} | `{r['feature']}` | {r['importance']:.4f} | {cum_imp:.4f} |\n"

    report_md += f"""
---

## 5. Diagnostic Audit: Feature Construction Artifact Analysis

* **RI Retained Intron Symmetry**:
  * In the full dataset, **{ri_exact_intron_match}/{len(ri_total)}** ({ri_fraction_match*100:.2f}%) of RI events have `upstream_intron_length == downstream_intron_length` by construction in Stage 5.
  * In the held-out test set, **{ri_test_match}/{len(X_test_ri)}** ({ri_test_fraction*100:.2f}%) of RI events exhibit exact length symmetry.
* **RI Classification Performance**:
  * Random Forest achieved **{ri_prec_rf*100:.2f}% Precision** and **{ri_rec_rf*100:.2f}% Recall** on the RI class (F1 = **{ri_f1_rf:.4f}**).
* **Top-5 Importance Presence**:
  * Top-5 ranked features: `{top5_features}`.
* **Artifact Finding & Critical Methodological Discussion (§4.5 / §5)**:
  * **Status**: **{'CONSTRUCTION ARTIFACT FLAGGED' if artifact_flag else 'NO OVERT ARTIFACT'}**
  * **Rationale**: Because Stage 5 defines the retained intron length identically for both upstream and downstream intron attributes in the RI branch (`up_intron_len = dn_intron_len = downstreamES - upstreamEE`), the decision tree ensemble heavily exploits this exact numerical equivalence to isolate RI events with near-perfect accuracy. While this represents a deterministic mathematical property of the dataset's coordinate definition, it must be explicitly cited in the dissertation limitations section as a feature-engineering artifact rather than an autonomous discovery of biological intron kinetics.

---

## 6. Logistic Regression Coefficients (Top Predictors per Class)
```
{lr_coef_df.round(4).to_string()}
```
"""

    report_path = os.path.join(models_dir, "evaluation_report.md")
    with open(report_path, "w") as f:
        f.write(report_md)
    print(f"\nSaved evaluation report: {report_path}")

    # Print summary to stdout
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Logistic Regression: Accuracy={m_lr['accuracy']*100:.2f}%, Macro-F1={m_lr['f1_macro']:.4f} -> Target Met: {lr_meets_target}")
    print(f"Random Forest:       Accuracy={m_rf['accuracy']*100:.2f}%, Macro-F1={m_rf['f1_macro']:.4f} -> Target Met: {rf_meets_target}")
    print("\nTop 10 RF Feature Importances:")
    for idx, r in top10_features.iterrows():
        print(f"  {idx+1:>2}. {r['feature']:<25}: {r['importance']:.4f}")
    print(f"\nRI Diagnostic Check:")
    print(f"  RI Symmetry: {ri_exact_intron_match}/{len(ri_total)} ({ri_fraction_match*100:.1f}%)")
    print(f"  RF RI Precision: {ri_prec_rf*100:.2f}%, Recall: {ri_rec_rf*100:.2f}%")
    print(f"  Artifact Flagged: {artifact_flag}")

if __name__ == "__main__":
    main()