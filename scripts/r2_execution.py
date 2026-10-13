#!/usr/bin/env python3
"""
scripts/r2_execution.py

Execution script for directive R2:
1. Confusion matrices:
   - Compute in-memory test set and KD2 predictions using winning XGBoost.
   - Save to results/models/r1_confusion_matrix_winner.csv and results/models/r1_confusion_matrix_kd2.csv with np.savetxt.
   - Print each with row and column sums.
   - Recompute accuracy and macro-F1 from saved arrays and assert equality with r1_test_evaluation.csv and r1_kd2_evaluation.csv.
   - Regenerate PNGs from the same arrays.
2. Ablations:
   - Header with git rev-parse HEAD.
   - State model, hyperparameters, and SD ddof (ddof=1).
   - Print per-fold values across all 5 subsets.
3. LR baseline:
   - StandardScaler, balanced weights, C tuned over [0.1, 1, 10] by 5-fold GroupKFold on training groups only.
   - Evaluate once on held-out test set and KD2 (full, seen, unseen, each with majority baseline).
   - Explicitly note it took no part in model selection.
4. XGBoost vs RF comparison:
   - Print per-fold CV macro-F1 and accuracy for both winning XGBoost and runner-up RF.
   - Gene-level paired cluster bootstrap (1,000 resamples) on held-out test set for macro-F1 difference with 95% CI.
"""

import os
import sys
import subprocess
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

def get_git_head():
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_HEAD"

def plot_and_save_cm(cm, classes, title, output_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label', xlabel='Predicted label')
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close()

def print_cm_with_sums(cm, classes, title):
    print(f"\n{title}")
    row_title = "True \\ Pred"
    header = f"{row_title:<12}" + "".join([f"{c:>8}" for c in classes]) + f"{'Row Sum':>10}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for i, c in enumerate(classes):
        row_str = f"{c:<12}" + "".join([f"{cm[i, j]:>8d}" for j in range(len(classes))]) + f"{cm[i, :].sum():>10d}"
        print(row_str)
    print("-" * len(header))
    col_sum_str = f"{'Col Sum':<12}" + "".join([f"{cm[:, j].sum():>8d}" for j in range(len(classes))]) + f"{cm.sum():>10d}"
    print(col_sum_str)
    print("-" * len(header))

def main():
    repo_dir = "/home/brooklyn/splicing-ml-project"
    if os.path.exists(repo_dir):
        os.chdir(repo_dir)

    print("=" * 80)
    print("R2 COMPREHENSIVE EXECUTION & VALIDATION BATTERY")
    print(f"Git HEAD: {get_git_head()}")
    print("=" * 80)

    models_dir = "results/models"
    os.makedirs(models_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 0. Load Data & Serialized Artifacts
    # -------------------------------------------------------------------------
    kd1_path = "results/features/feature_matrix_v2.csv"
    kd2_path = "results/features/feature_matrix_kd2_v2.csv"
    
    df_kd1 = pd.read_csv(kd1_path)
    df_kd2 = pd.read_csv(kd2_path)

    # Strand dummy handling
    strand_dummies_kd1 = pd.get_dummies(df_kd1["strand"], prefix="strand", dtype=float)
    if "strand_+" not in strand_dummies_kd1.columns: strand_dummies_kd1["strand_+"] = 0.0
    if "strand_-" not in strand_dummies_kd1.columns: strand_dummies_kd1["strand_-"] = 0.0
    df_kd1 = pd.concat([df_kd1, strand_dummies_kd1[["strand_+", "strand_-"]]], axis=1)

    strand_dummies_kd2 = pd.get_dummies(df_kd2["strand"], prefix="strand", dtype=float)
    if "strand_+" not in strand_dummies_kd2.columns: strand_dummies_kd2["strand_+"] = 0.0
    if "strand_-" not in strand_dummies_kd2.columns: strand_dummies_kd2["strand_-"] = 0.0
    df_kd2 = pd.concat([df_kd2, strand_dummies_kd2[["strand_+", "strand_-"]]], axis=1)

    psi_cols = ["psi_control_mean", "psi_control_var", "psi_kd_mean", "psi_kd_var", "dpsi"]
    df_kd1[psi_cols] = df_kd1[psi_cols].fillna(0.0)
    df_kd2[psi_cols] = df_kd2[psi_cols].fillna(0.0)

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

    X = df_kd1[feat_23].copy()
    y = df_kd1["label"].copy()
    groups = df_kd1["gene_id"].copy()

    classes = sorted(y.unique())
    label_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_label = {i: c for i, c in enumerate(classes)}
    y_idx = y.map(label_to_idx)

    split_data = joblib.load(os.path.join(models_dir, "r1_train_test_data_grouped.joblib"))
    train_idx = split_data["train_idx"]
    test_idx = split_data["test_idx"]

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    y_train_idx, y_test_idx = y_idx.iloc[train_idx], y_idx.iloc[test_idx]
    groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]

    winner_xgb = joblib.load(os.path.join(models_dir, "r1_winner_xgboost.joblib"))
    secondary_rf = joblib.load(os.path.join(models_dir, "r1_secondary_rf.joblib"))

    # -------------------------------------------------------------------------
    # 1. CONFUSION MATRICES (TEST SET & KD2)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ITEM 1: CONFUSION MATRICES (TEST SET & KD2)")
    print("=" * 80)

    # In-memory test predictions
    pred_idx_test = winner_xgb.predict(X_test)
    pred_test = [idx_to_label[i] for i in pred_idx_test]
    cm_test = confusion_matrix(y_test, pred_test, labels=classes)

    # Save test confusion matrix
    cm_test_csv = os.path.join(models_dir, "r1_confusion_matrix_winner.csv")
    np.savetxt(cm_test_csv, cm_test, fmt="%d", delimiter=",")
    print(f"Saved winner test confusion matrix to: {cm_test_csv}")

    # Print with row and column sums
    print_cm_with_sums(cm_test, classes, "Confusion Matrix: Winning XGBoost on Held-Out Test Set (N=1,807)")

    # Recompute accuracy and macro-F1 directly from saved array
    cm_test_loaded = np.loadtxt(cm_test_csv, delimiter=",", dtype=int)
    recomp_acc_test = np.diag(cm_test_loaded).sum() / cm_test_loaded.sum()
    per_cls_f1_test = [
        2.0 * cm_test_loaded[i, i] / (cm_test_loaded[i, :].sum() + cm_test_loaded[:, i].sum())
        for i in range(len(classes))
    ]
    recomp_macro_f1_test = float(np.mean(per_cls_f1_test))

    print(f"Recomputed Test Accuracy from array: {recomp_acc_test:.16f} ({recomp_acc_test*100:.2f}%)")
    print(f"Recomputed Test Macro-F1 from array: {recomp_macro_f1_test:.16f}")
    for c, f1 in zip(classes, per_cls_f1_test):
        print(f"  Class {c} F1: {f1:.4f}")

    # Assert equality with r1_test_evaluation.csv
    test_eval_df = pd.read_csv(os.path.join(models_dir, "r1_test_evaluation.csv"))
    winner_row = test_eval_df[test_eval_df["role"] == "Primary Winner"].iloc[0]
    expected_acc_test = float(winner_row["test_accuracy"])
    expected_f1_test = float(winner_row["test_macro_f1"])

    print(f"Expected in r1_test_evaluation.csv: Acc={expected_acc_test:.16f}, Macro-F1={expected_f1_test:.16f}")
    assert np.isclose(recomp_acc_test, expected_acc_test, atol=1e-12), "Mismatch in test accuracy!"
    assert np.isclose(recomp_macro_f1_test, expected_f1_test, atol=1e-12), "Mismatch in test macro-F1!"
    print("ASSERTION PASSED: Test confusion matrix exactly matches r1_test_evaluation.csv.")

    # Regenerate PNG from array
    plot_and_save_cm(
        cm_test_loaded, classes,
        f"Confusion Matrix: Winner (XGBoost)\nTest Accuracy: {recomp_acc_test*100:.2f}%, Macro-F1: {recomp_macro_f1_test:.4f}",
        os.path.join(models_dir, "r1_confusion_matrix_winner.png")
    )
    print("Regenerated PNG: results/models/r1_confusion_matrix_winner.png")

    # In-memory KD2 predictions
    X_kd2 = df_kd2[feat_23].copy()
    y_kd2 = df_kd2["label"].copy()
    pred_idx_kd2 = winner_xgb.predict(X_kd2)
    pred_kd2 = [idx_to_label[i] for i in pred_idx_kd2]
    cm_kd2 = confusion_matrix(y_kd2, pred_kd2, labels=classes)

    # Save KD2 confusion matrix
    cm_kd2_csv = os.path.join(models_dir, "r1_confusion_matrix_kd2.csv")
    np.savetxt(cm_kd2_csv, cm_kd2, fmt="%d", delimiter=",")
    print(f"\nSaved KD2 confusion matrix to: {cm_kd2_csv}")

    # Print with row and column sums
    print_cm_with_sums(cm_kd2, classes, "Confusion Matrix: Winning XGBoost on Full KD2 Cohort (N=7,279)")

    # Recompute accuracy and macro-F1 directly from saved array
    cm_kd2_loaded = np.loadtxt(cm_kd2_csv, delimiter=",", dtype=int)
    recomp_acc_kd2 = np.diag(cm_kd2_loaded).sum() / cm_kd2_loaded.sum()
    per_cls_f1_kd2 = [
        2.0 * cm_kd2_loaded[i, i] / (cm_kd2_loaded[i, :].sum() + cm_kd2_loaded[:, i].sum())
        for i in range(len(classes))
    ]
    recomp_macro_f1_kd2 = float(np.mean(per_cls_f1_kd2))

    print(f"Recomputed KD2 Accuracy from array: {recomp_acc_kd2:.16f} ({recomp_acc_kd2*100:.2f}%)")
    print(f"Recomputed KD2 Macro-F1 from array: {recomp_macro_f1_kd2:.16f}")
    for c, f1 in zip(classes, per_cls_f1_kd2):
        print(f"  Class {c} F1: {f1:.4f}")

    # Assert equality with r1_kd2_evaluation.csv
    kd2_eval_df = pd.read_csv(os.path.join(models_dir, "r1_kd2_evaluation.csv"))
    full_kd2_row = kd2_eval_df[kd2_eval_df["Stratum"] == "Full KD2 Cohort"].iloc[0]
    expected_acc_kd2 = float(full_kd2_row["Accuracy"])
    expected_f1_kd2 = float(full_kd2_row["Macro_F1"])

    print(f"Expected in r1_kd2_evaluation.csv: Acc={expected_acc_kd2:.16f}, Macro-F1={expected_f1_kd2:.16f}")
    assert np.isclose(recomp_acc_kd2, expected_acc_kd2, atol=1e-12), "Mismatch in KD2 accuracy!"
    assert np.isclose(recomp_macro_f1_kd2, expected_f1_kd2, atol=1e-12), "Mismatch in KD2 macro-F1!"
    for c, f1 in zip(classes, per_cls_f1_kd2):
        assert np.isclose(f1, float(full_kd2_row[f"F1_{c}"]), atol=1e-12), f"Mismatch in KD2 class {c} F1!"
    print("ASSERTION PASSED: KD2 confusion matrix exactly matches r1_kd2_evaluation.csv.")

    # Regenerate PNG from array
    plot_and_save_cm(
        cm_kd2_loaded, classes,
        f"Confusion Matrix: Winner on KD2 Validation Cohort\nAccuracy: {recomp_acc_kd2*100:.2f}%, Macro-F1: {recomp_macro_f1_kd2:.4f}",
        os.path.join(models_dir, "r1_confusion_matrix_kd2.png")
    )
    print("Regenerated PNG: results/models/r1_confusion_matrix_kd2.png")

    # -------------------------------------------------------------------------
    # 2. FEATURE ABLATIONS UNDER 5-FOLD GroupKFold
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ITEM 2: FEATURE ABLATIONS UNDER 5-FOLD GroupKFold")
    print(f"Git HEAD: {get_git_head()}")
    print("Model: XGBClassifier (XGBoost)")
    print("Hyperparameters: n_estimators=300, max_depth=6, learning_rate=0.05, tree_method='hist', random_state=42")
    print("Standard Deviation Convention: Sample SD with ddof=1")
    print("=" * 80)

    ablation_sets = [
        ("(A) All 23 Features", feat_23),
        ("(B) All minus exon_length", [c for c in feat_23 if c != "exon_length"]),
        ("(C) Sequence/Structure Only (12 feats)", [
            "exon_length", "gc_exon", "gc_upstream_100bp", "gc_downstream_100bp",
            "score_5ss_donor", "score_3ss_acceptor",
            "max_utract_length_upstream", "max_utract_length_downstream",
            "u_fraction_upstream", "u_fraction_downstream",
            "strand_+", "strand_-"
        ]),
        ("(D) Sequence/Structure minus exon_length (11 feats)", [
            "gc_exon", "gc_upstream_100bp", "gc_downstream_100bp",
            "score_5ss_donor", "score_3ss_acceptor",
            "max_utract_length_upstream", "max_utract_length_downstream",
            "u_fraction_upstream", "u_fraction_downstream",
            "strand_+", "strand_-"
        ]),
        ("(E) Coverage/PSI Only (11 feats)", [
            "cov_mean_control", "cov_var_control", "cov_mean_kd", "cov_var_kd",
            "cov_mean_overall", "cov_var_overall",
            "psi_control_mean", "psi_control_var", "psi_kd_mean", "psi_kd_var", "dpsi"
        ])
    ]

    gkf = GroupKFold(n_splits=5)
    for a_name, feats_sub in ablation_sets:
        print(f"\n--- Subset: {a_name} (N_Features = {len(feats_sub)}) ---")
        accs, f1s = [], []
        for f, (t_idx, v_idx) in enumerate(gkf.split(X[feats_sub], y, groups=groups), 1):
            clf = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                random_state=42,
                eval_metric="mlogloss",
                tree_method="hist",
                n_jobs=2
            )
            sw = compute_sample_weight("balanced", y_idx.iloc[t_idx])
            clf.fit(X[feats_sub].iloc[t_idx], y_idx.iloc[t_idx], sample_weight=sw)
            p = clf.predict(X[feats_sub].iloc[v_idx])
            f_acc = accuracy_score(y_idx.iloc[v_idx], p)
            f_f1 = f1_score(y_idx.iloc[v_idx], p, average="macro", zero_division=0)
            accs.append(f_acc)
            f1s.append(f_f1)
            print(f"  Fold {f}: Accuracy = {f_acc*100:.2f}% ({f_acc:.4f}), Macro-F1 = {f_f1:.4f}")

        m_a, s_a = np.mean(accs), np.std(accs, ddof=1)
        m_f, s_f = np.mean(f1s), np.std(f1s, ddof=1)
        print(f"  Summary (ddof=1): Mean Accuracy = {m_a*100:.2f}% ± {s_a*100:.2f}%, Mean Macro-F1 = {m_f:.4f} ± {s_f:.4f}")

    # -------------------------------------------------------------------------
    # 3. LOGISTIC REGRESSION BASELINE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ITEM 3: LOGISTIC REGRESSION BASELINE")
    print("Status: Baseline comparator only — TOOK NO PART IN MODEL SELECTION")
    print("Candidates for primary selection were strictly pre-registered RF and XGBoost")
    print("Pipeline: StandardScaler -> LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)")
    print("Tuning grid: C in [0.1, 1, 10] evaluated strictly on training groups via 5-fold GroupKFold")
    print("=" * 80)

    lr_candidates = [0.1, 1.0, 10.0]
    lr_cv_results = []
    for C_val in lr_candidates:
        accs_lr, f1s_lr = [], []
        for f, (tr_fold, val_fold) in enumerate(gkf.split(X_train, y_train, groups=groups_train), 1):
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_train.iloc[tr_fold])
            X_va_sc = scaler.transform(X_train.iloc[val_fold])
            
            lr = LogisticRegression(C=C_val, class_weight="balanced", max_iter=1000, random_state=42)
            lr.fit(X_tr_sc, y_train.iloc[tr_fold])
            p_val = lr.predict(X_va_sc)
            
            accs_lr.append(accuracy_score(y_train.iloc[val_fold], p_val))
            f1s_lr.append(f1_score(y_train.iloc[val_fold], p_val, average="macro", zero_division=0))
        
        m_acc, s_acc = np.mean(accs_lr), np.std(accs_lr, ddof=1)
        m_f1, s_f1 = np.mean(f1s_lr), np.std(f1s_lr, ddof=1)
        lr_cv_results.append({
            "C": C_val,
            "mean_acc": m_acc,
            "std_acc": s_acc,
            "mean_macro_f1": m_f1,
            "std_macro_f1": s_f1
        })
        print(f"  C = {C_val:<5}: CV Mean Acc = {m_acc*100:.2f}% ± {s_acc*100:.2f}%, CV Mean Macro-F1 = {m_f1:.4f} ± {s_f1:.4f} (ddof=1)")

    best_lr_res = max(lr_cv_results, key=lambda x: x["mean_macro_f1"])
    best_C = best_lr_res["C"]
    print(f"\nOptimal LR parameter chosen by training CV: C = {best_C} (Macro-F1 = {best_lr_res['mean_macro_f1']:.4f})")

    # Fit best LR model on full training set
    scaler_full = StandardScaler()
    X_train_scaled = scaler_full.fit_transform(X_train)
    X_test_scaled = scaler_full.transform(X_test)
    X_kd2_scaled = scaler_full.transform(X_kd2)

    best_lr = LogisticRegression(C=best_C, class_weight="balanced", max_iter=1000, random_state=42)
    best_lr.fit(X_train_scaled, y_train)

    # Evaluate ONCE on held-out test set
    p_test_lr = best_lr.predict(X_test_scaled)
    acc_test_lr = accuracy_score(y_test, p_test_lr)
    f1_test_lr = f1_score(y_test, p_test_lr, average="macro", zero_division=0)
    per_cls_f1_test_lr = f1_score(y_test, p_test_lr, average=None, labels=classes, zero_division=0)

    # Test majority baseline
    maj_cls_test = y_test.value_counts().idxmax()
    maj_pred_test = [maj_cls_test] * len(y_test)
    maj_acc_test = accuracy_score(y_test, maj_pred_test)
    maj_f1_test = f1_score(y_test, maj_pred_test, average="macro", zero_division=0)

    print("\n--- Held-Out Test Set Evaluation (N = 1,807) ---")
    print(f"  LR Model: Accuracy = {acc_test_lr*100:.2f}%, Macro-F1 = {f1_test_lr:.4f}")
    print(f"  Majority Baseline ({maj_cls_test}): Accuracy = {maj_acc_test*100:.2f}%, Macro-F1 = {maj_f1_test:.4f}")
    for c, f1 in zip(classes, per_cls_f1_test_lr):
        print(f"    Class {c} F1: {f1:.4f}")

    # Evaluate ONCE on KD2 (full, seen, unseen)
    p_kd2_lr = best_lr.predict(X_kd2_scaled)
    train_genes = set(groups_train)
    kd2_in_train = df_kd2["gene_id"].isin(train_genes)

    strata = [
        ("Full KD2 Cohort", np.ones(len(df_kd2), dtype=bool)),
        ("Seen-Gene Stratum (in KD1 Train)", kd2_in_train.values),
        ("Unseen-Gene Stratum (Absent from KD1 Train)", (~kd2_in_train).values)
    ]

    for s_name, mask in strata:
        y_true_s = y_kd2.iloc[mask]
        y_pred_s = pd.Series(p_kd2_lr).iloc[mask]
        n_s = len(y_true_s)

        acc_s = accuracy_score(y_true_s, y_pred_s)
        f1_s = f1_score(y_true_s, y_pred_s, average="macro", zero_division=0)
        per_cls_f1_s = f1_score(y_true_s, y_pred_s, average=None, labels=classes, zero_division=0)

        maj_cls_s = y_true_s.value_counts().idxmax()
        maj_pred_s = [maj_cls_s] * n_s
        maj_acc_s = accuracy_score(y_true_s, maj_pred_s)
        maj_f1_s = f1_score(y_true_s, maj_pred_s, average="macro", zero_division=0)

        print(f"\n--- KD2: {s_name} (N = {n_s:,}, {n_s/len(df_kd2)*100:.2f}%) ---")
        print(f"  LR Model: Accuracy = {acc_s*100:.2f}%, Macro-F1 = {f1_s:.4f}")
        print(f"  Majority Baseline ({maj_cls_s}): Accuracy = {maj_acc_s*100:.2f}%, Macro-F1 = {maj_f1_s:.4f}")
        for c, f1 in zip(classes, per_cls_f1_s):
            print(f"    Class {c} F1: {f1:.4f}")

    # -------------------------------------------------------------------------
    # 4. XGBOOST VS RF COMPARISON & GENE-LEVEL PAIRED BOOTSTRAP
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ITEM 4: XGBOOST VS RF COMPARISON & GENE-LEVEL PAIRED BOOTSTRAP")
    print("=" * 80)

    # A. Per-fold CV performance on training groups
    print("\nA. Per-Fold 5-Fold GroupKFold Cross-Validation on Training Groups (N=7,440):")
    xgb_cv_accs, xgb_cv_f1s = [], []
    rf_cv_accs, rf_cv_f1s = [], []

    for f, (tr_fold, val_fold) in enumerate(gkf.split(X_train, y_train, groups=groups_train), 1):
        # XGBoost
        clf_xgb = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=2
        )
        sw = compute_sample_weight("balanced", y_train_idx.iloc[tr_fold])
        clf_xgb.fit(X_train.iloc[tr_fold], y_train_idx.iloc[tr_fold], sample_weight=sw)
        p_xgb = clf_xgb.predict(X_train.iloc[val_fold])
        f_acc_xgb = accuracy_score(y_train_idx.iloc[val_fold], p_xgb)
        f_f1_xgb = f1_score(y_train_idx.iloc[val_fold], p_xgb, average="macro", zero_division=0)
        xgb_cv_accs.append(f_acc_xgb)
        xgb_cv_f1s.append(f_f1_xgb)

        # Random Forest
        clf_rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            class_weight="balanced",
            random_state=42,
            n_jobs=2
        )
        clf_rf.fit(X_train.iloc[tr_fold], y_train.iloc[tr_fold])
        p_rf = clf_rf.predict(X_train.iloc[val_fold])
        f_acc_rf = accuracy_score(y_train.iloc[val_fold], p_rf)
        f_f1_rf = f1_score(y_train.iloc[val_fold], p_rf, average="macro", zero_division=0)
        rf_cv_accs.append(f_acc_rf)
        rf_cv_f1s.append(f_f1_rf)

        print(f"  Fold {f}:")
        print(f"    XGBoost Winner:       Acc = {f_acc_xgb*100:.2f}%, Macro-F1 = {f_f1_xgb:.4f}")
        print(f"    Random Forest Runner: Acc = {f_acc_rf*100:.2f}%, Macro-F1 = {f_f1_rf:.4f}")

    print("\nCV Summary (ddof=1):")
    print(f"  XGBoost Winner:       Mean Acc = {np.mean(xgb_cv_accs)*100:.2f}% ± {np.std(xgb_cv_accs, ddof=1)*100:.2f}%, Mean Macro-F1 = {np.mean(xgb_cv_f1s):.4f} ± {np.std(xgb_cv_f1s, ddof=1):.4f}")
    print(f"  Random Forest Runner: Mean Acc = {np.mean(rf_cv_accs)*100:.2f}% ± {np.std(rf_cv_accs, ddof=1)*100:.2f}%, Mean Macro-F1 = {np.mean(rf_cv_f1s):.4f} ± {np.std(rf_cv_f1s, ddof=1):.4f}")

    # B. Gene-level Paired Cluster Bootstrap on Held-Out Test Set
    print("\nB. Gene-Level Paired Cluster Bootstrap (1,000 Resamples on Held-Out Test Set):")
    p_xgb_test = np.array([idx_to_label[i] for i in winner_xgb.predict(X_test)])
    p_rf_test = np.array(secondary_rf.predict(X_test))
    y_test_arr = np.array(y_test)
    groups_test_arr = np.array(groups_test)

    obs_f1_xgb = f1_score(y_test_arr, p_xgb_test, average="macro", zero_division=0)
    obs_f1_rf = f1_score(y_test_arr, p_rf_test, average="macro", zero_division=0)
    obs_diff = obs_f1_xgb - obs_f1_rf

    print(f"  Observed Held-Out Test Metrics:")
    print(f"    XGBoost Macro-F1:       {obs_f1_xgb:.4f}")
    print(f"    Random Forest Macro-F1: {obs_f1_rf:.4f}")
    print(f"    Observed Difference (ΔF1 = XGB - RF): {obs_diff:+.4f} ({obs_diff*100:+.2f} pp)")

    unique_test_genes = np.unique(groups_test_arr)
    n_test_genes = len(unique_test_genes)
    gene_to_test_idx = {g: np.where(groups_test_arr == g)[0] for g in unique_test_genes}

    rng = np.random.RandomState(42)
    boot_diffs = []
    boot_xgb_f1s = []
    boot_rf_f1s = []

    for b in range(1000):
        sampled_genes = rng.choice(unique_test_genes, size=n_test_genes, replace=True)
        sampled_rows = np.concatenate([gene_to_test_idx[g] for g in sampled_genes])
        
        y_b = y_test_arr[sampled_rows]
        p_xgb_b = p_xgb_test[sampled_rows]
        p_rf_b = p_rf_test[sampled_rows]
        
        f1_x = f1_score(y_b, p_xgb_b, average="macro", zero_division=0)
        f1_r = f1_score(y_b, p_rf_b, average="macro", zero_division=0)
        
        boot_xgb_f1s.append(f1_x)
        boot_rf_f1s.append(f1_r)
        boot_diffs.append(f1_x - f1_r)

    boot_diffs = np.array(boot_diffs)
    ci_lower = np.percentile(boot_diffs, 2.5)
    ci_upper = np.percentile(boot_diffs, 97.5)
    p_val_empirical = np.mean(boot_diffs <= 0.0)

    print(f"\n  Bootstrap Results (1,000 Gene Clusters Resampled):")
    print(f"    Bootstrap Mean Difference (ΔF1): {np.mean(boot_diffs):+.4f}")
    print(f"    Bootstrap Std (ddof=1):           {np.std(boot_diffs, ddof=1):.4f}")
    print(f"    95% Empirical Percentile CI:      [{ci_lower:+.4f}, {ci_upper:+.4f}] ([{ci_lower*100:+.2f} pp, {ci_upper*100:+.2f} pp])")
    print(f"    Empirical p-value (ΔF1 <= 0):     {p_val_empirical:.4f}")

    print("\n" + "=" * 80)
    print("R2 BATTERY COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
