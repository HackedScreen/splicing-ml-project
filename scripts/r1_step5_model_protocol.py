#!/usr/bin/env python3
"""
scripts/r1_step5_model_protocol.py

Pre-registered model tuning, evaluation, and supervisor battery pipeline following docs/PROTOCOL.md.
Uses sample standard deviation (ddof=1) everywhere.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import joblib
from scipy.stats import binomtest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, brier_score_loss
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.feature_selection import mutual_info_classif
from sklearn.dummy import DummyClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

def main():
    print("=" * 80)
    print("R1 MODEL PROTOCOL & SUPERVISOR BATTERY EXECUTION")
    print("=" * 80)

    models_dir = "results/models"
    os.makedirs(models_dir, exist_ok=True)

    # 1. Load KD1 and KD2 feature matrices
    kd1_file = "results/features/feature_matrix_v2.csv"
    kd2_file = "results/features/feature_matrix_kd2_v2.csv"
    print(f"Loading KD1 feature matrix: {kd1_file}")
    df_kd1 = pd.read_csv(kd1_file)
    print(f"Loaded KD1: {len(df_kd1):,} events across {df_kd1['gene_id'].nunique():,} unique genes.")

    print(f"Loading KD2 feature matrix: {kd2_file}")
    df_kd2 = pd.read_csv(kd2_file)
    print(f"Loaded KD2: {len(df_kd2):,} events across {df_kd2['gene_id'].nunique():,} unique genes.")

    # Prepare features
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

    # -------------------------------------------------------------------------
    # PART 1: GENE-GROUPED TRAIN/TEST SPLIT (80/20, random_state=42)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("1. EXECUTING GENE-GROUPED SPLIT (GroupShuffleSplit, 80/20, seed=42)")
    print("-" * 70)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    y_train_idx, y_test_idx = y_idx.iloc[train_idx], y_idx.iloc[test_idx]
    groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]

    train_genes = set(groups_train)
    test_genes = set(groups_test)
    overlap = train_genes.intersection(test_genes)
    print(f"Train events: {len(X_train):,} ({len(X_train)/len(X)*100:.2f}%) across {len(train_genes):,} genes")
    print(f"Test events:  {len(X_test):,} ({len(X_test)/len(X)*100:.2f}%) across {len(test_genes):,} genes")
    print(f"Gene overlap between Train and Test: {len(overlap)} (STRICTLY ZERO)")
    assert len(overlap) == 0, "Gene leakage detected!"

    joblib.dump({
        "train_idx": train_idx,
        "test_idx": test_idx,
        "train_genes": train_genes,
        "test_genes": test_genes
    }, os.path.join(models_dir, "r1_train_test_data_grouped.joblib"))

    # -------------------------------------------------------------------------
    # PART 2: CANDIDATE GRID SEARCH ON TRAINING GROUPS ONLY (5-Fold GroupKFold)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("2. HYPERPARAMETER TUNING VIA 5-FOLD GroupKFold ON TRAINING GROUPS ONLY")
    print("-" * 70)
    gkf = GroupKFold(n_splits=5)

    force_rerun = "--force-rerun" in sys.argv
    cv_csv = os.path.join(models_dir, "r1_cv_model_tuning.csv")
    if os.path.exists(cv_csv) and not force_rerun:
        print(f"Loading cached cross-validation tuning results from {cv_csv}...")
        df_cv = pd.read_csv(cv_csv)
        for _, r in df_cv.iterrows():
            print(f"  [{r['model_family'][:3]}] {r['params']}: Acc = {r['cv_mean_acc']*100:.2f}% ± {r['cv_std_acc']*100:.2f}%, Macro-F1 = {r['cv_mean_macro_f1']:.4f} ± {r['cv_std_macro_f1']:.4f} (ddof=1)")
    else:
        cv_results = []

        # A. Random Forest Grid
        rf_grid = [
            {"n_estimators": 100, "max_depth": 10},
            {"n_estimators": 100, "max_depth": 20},
            {"n_estimators": 100, "max_depth": None},
            {"n_estimators": 300, "max_depth": 10},
            {"n_estimators": 300, "max_depth": 20},
            {"n_estimators": 300, "max_depth": None},
        ]

        for params in rf_grid:
            f1_list, acc_list = [], []
            for f, (tr_fold, val_fold) in enumerate(gkf.split(X_train, y_train, groups=groups_train), 1):
                clf = RandomForestClassifier(
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=2
                )
                clf.fit(X_train.iloc[tr_fold], y_train.iloc[tr_fold])
                pred = clf.predict(X_train.iloc[val_fold])
                acc_list.append(accuracy_score(y_train.iloc[val_fold], pred))
                f1_list.append(f1_score(y_train.iloc[val_fold], pred, average="macro", zero_division=0))

            m_acc, s_acc = np.mean(acc_list), np.std(acc_list, ddof=1)
            m_f1, s_f1 = np.mean(f1_list), np.std(f1_list, ddof=1)
            cv_results.append({
                "model_family": "RandomForest",
                "params": f"n_est={params['n_estimators']}, max_depth={params['max_depth']}",
                "n_estimators": params["n_estimators"],
                "max_depth": str(params["max_depth"]),
                "learning_rate": "N/A",
                "cv_mean_acc": m_acc,
                "cv_std_acc": s_acc,
                "cv_mean_macro_f1": m_f1,
                "cv_std_macro_f1": s_f1
            })
            print(f"  [RF] n_est={params['n_estimators']}, max_depth={params['max_depth']}: "
                  f"Acc = {m_acc*100:.2f}% ± {s_acc*100:.2f}%, Macro-F1 = {m_f1:.4f} ± {s_f1:.4f} (ddof=1)")

        # B. XGBoost Grid
        xgb_grid = [
            {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05},
            {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
            {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.05},
            {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1},
            {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05},
            {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.1},
            {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05},
            {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1},
        ]

        for params in xgb_grid:
            f1_list, acc_list = [], []
            for f, (tr_fold, val_fold) in enumerate(gkf.split(X_train, y_train_idx, groups=groups_train), 1):
                clf = XGBClassifier(
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    learning_rate=params["learning_rate"],
                    random_state=42,
                    eval_metric="mlogloss",
                    tree_method="hist",
                    n_jobs=2
                )
                sw = compute_sample_weight("balanced", y_train_idx.iloc[tr_fold])
                clf.fit(X_train.iloc[tr_fold], y_train_idx.iloc[tr_fold], sample_weight=sw)
                pred = clf.predict(X_train.iloc[val_fold])
                acc_list.append(accuracy_score(y_train_idx.iloc[val_fold], pred))
                f1_list.append(f1_score(y_train_idx.iloc[val_fold], pred, average="macro", zero_division=0))

            m_acc, s_acc = np.mean(acc_list), np.std(acc_list, ddof=1)
            m_f1, s_f1 = np.mean(f1_list), np.std(f1_list, ddof=1)
            cv_results.append({
                "model_family": "XGBoost",
                "params": f"n_est={params['n_estimators']}, max_depth={params['max_depth']}, lr={params['learning_rate']}",
                "n_estimators": params["n_estimators"],
                "max_depth": str(params["max_depth"]),
                "learning_rate": str(params["learning_rate"]),
                "cv_mean_acc": m_acc,
                "cv_std_acc": s_acc,
                "cv_mean_macro_f1": m_f1,
                "cv_std_macro_f1": s_f1
            })
            print(f"  [XGB] n_est={params['n_estimators']}, max_depth={params['max_depth']}, lr={params['learning_rate']}: "
                  f"Acc = {m_acc*100:.2f}% ± {s_acc*100:.2f}%, Macro-F1 = {m_f1:.4f} ± {s_f1:.4f} (ddof=1)")

        df_cv = pd.DataFrame(cv_results)
        df_cv = df_cv.sort_values(by="cv_mean_macro_f1", ascending=False).reset_index(drop=True)
        cv_csv = os.path.join(models_dir, "r1_cv_model_tuning.csv")
        df_cv.to_csv(cv_csv, index=False)
        print(f"\nSaved cross-validation results to {cv_csv}")

    winner_row = df_cv.iloc[0]
    runner_up_row = df_cv[df_cv["model_family"] != winner_row["model_family"]].iloc[0]
    print(f"\nWINNER (highest grouped-CV Macro-F1): {winner_row['model_family']} ({winner_row['params']}) with Macro-F1 = {winner_row['cv_mean_macro_f1']:.4f} ± {winner_row['cv_std_macro_f1']:.4f}")
    print(f"RUNNER-UP (best of alternate family): {runner_up_row['model_family']} ({runner_up_row['params']}) with Macro-F1 = {runner_up_row['cv_mean_macro_f1']:.4f} ± {runner_up_row['cv_std_macro_f1']:.4f}")

    # -------------------------------------------------------------------------
    # PART 3: FIT ON FULL TRAINING SET & EVALUATE ON HELD-OUT TEST SET
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("3. EVALUATION ON HELD-OUT GROUPED TEST SET (N = 1,799)")
    print("-" * 70)

    # Train Winner
    if winner_row["model_family"] == "RandomForest":
        m_depth = None if pd.isna(winner_row["max_depth"]) or str(winner_row["max_depth"]) in ["None", "nan"] else int(winner_row["max_depth"])
        winner_clf = RandomForestClassifier(
            n_estimators=int(winner_row["n_estimators"]),
            max_depth=m_depth,
            class_weight="balanced",
            random_state=42,
            n_jobs=2
        )
        winner_clf.fit(X_train, y_train)
        winner_test_pred = winner_clf.predict(X_test)
        winner_test_prob = winner_clf.predict_proba(X_test)
    else:
        winner_clf = XGBClassifier(
            n_estimators=int(winner_row["n_estimators"]),
            max_depth=int(winner_row["max_depth"]),
            learning_rate=float(winner_row["learning_rate"]),
            random_state=42,
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=2
        )
        sw = compute_sample_weight("balanced", y_train_idx)
        winner_clf.fit(X_train, y_train_idx, sample_weight=sw)
        pred_idx = winner_clf.predict(X_test)
        winner_test_pred = [idx_to_label[i] for i in pred_idx]
        winner_test_prob = winner_clf.predict_proba(X_test)

    # Train Secondary (Runner-up)
    if runner_up_row["model_family"] == "RandomForest":
        m_depth = None if pd.isna(runner_up_row["max_depth"]) or str(runner_up_row["max_depth"]) in ["None", "nan"] else int(runner_up_row["max_depth"])
        runner_clf = RandomForestClassifier(
            n_estimators=int(runner_up_row["n_estimators"]),
            max_depth=m_depth,
            class_weight="balanced",
            random_state=42,
            n_jobs=2
        )
        runner_clf.fit(X_train, y_train)
        runner_test_pred = runner_clf.predict(X_test)
    else:
        runner_clf = XGBClassifier(
            n_estimators=int(runner_up_row["n_estimators"]),
            max_depth=int(runner_up_row["max_depth"]),
            learning_rate=float(runner_up_row["learning_rate"]),
            random_state=42,
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=2
        )
        sw = compute_sample_weight("balanced", y_train_idx)
        runner_clf.fit(X_train, y_train_idx, sample_weight=sw)
        pred_idx = runner_clf.predict(X_test)
        runner_test_pred = [idx_to_label[i] for i in pred_idx]

    # Metrics on Test Set
    acc_win = accuracy_score(y_test, winner_test_pred)
    f1_win = f1_score(y_test, winner_test_pred, average="macro", zero_division=0)
    acc_sec = accuracy_score(y_test, runner_test_pred)
    f1_sec = f1_score(y_test, runner_test_pred, average="macro", zero_division=0)

    print(f"\n[WINNER] {winner_row['model_family']} Test Accuracy: {acc_win*100:.2f}%, Test Macro-F1: {f1_win:.4f}")
    print(f"[SECONDARY] {runner_up_row['model_family']} Test Accuracy: {acc_sec*100:.2f}%, Test Macro-F1: {f1_sec:.4f}")
    print("\nWinner Classification Report (Test Set):")
    print(classification_report(y_test, winner_test_pred, digits=4, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, winner_test_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=f"Confusion Matrix: Winner ({winner_row['model_family']})\nTest Accuracy: {acc_win*100:.2f}%, Macro-F1: {f1_win:.4f}",
           ylabel='True label', xlabel='Predicted label')
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(os.path.join(models_dir, "r1_confusion_matrix_winner.png"), dpi=300)
    plt.close()

    # Save test results table
    test_eval_df = pd.DataFrame([
        {
            "role": "Primary Winner",
            "model": winner_row["model_family"],
            "params": winner_row["params"],
            "test_accuracy": acc_win,
            "test_macro_f1": f1_win
        },
        {
            "role": "Secondary Runner-up",
            "model": runner_up_row["model_family"],
            "params": runner_up_row["params"],
            "test_accuracy": acc_sec,
            "test_macro_f1": f1_sec
        }
    ])
    test_eval_df.to_csv(os.path.join(models_dir, "r1_test_evaluation.csv"), index=False)

    # -------------------------------------------------------------------------
    # PART 4: EXTERNAL BIOLOGICAL VALIDATION ON KD2 (OVERALL & STRATIFIED)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("4. EXTERNAL VALIDATION ON INDEPENDENT KD2 COHORT")
    print("-" * 70)

    X_kd2 = df_kd2[feat_23].copy()
    y_kd2 = df_kd2["label"].copy()
    kd2_in_train = df_kd2["gene_id"].isin(train_genes)

    if winner_row["model_family"] == "RandomForest":
        pred_kd2_all = winner_clf.predict(X_kd2)
    else:
        pred_idx_kd2 = winner_clf.predict(X_kd2)
        pred_kd2_all = [idx_to_label[i] for i in pred_idx_kd2]

    strata_kd2 = [
        ("Full KD2 Cohort", np.ones(len(df_kd2), dtype=bool)),
        ("Seen-Gene Stratum (in KD1 Train)", kd2_in_train.values),
        ("Unseen-Gene Stratum (Absent from KD1 Train)", (~kd2_in_train).values)
    ]

    kd2_results = []
    for s_name, mask in strata_kd2:
        y_true_s = y_kd2.iloc[mask]
        y_pred_s = pd.Series(pred_kd2_all).iloc[mask]
        n_s = len(y_true_s)

        acc_s = accuracy_score(y_true_s, y_pred_s)
        f1_s = f1_score(y_true_s, y_pred_s, average="macro", zero_division=0)

        # Majority baseline
        maj_class = y_true_s.value_counts().idxmax()
        maj_count = y_true_s.value_counts().max()
        pred_maj = [maj_class] * n_s
        acc_maj = accuracy_score(y_true_s, pred_maj)
        f1_maj = f1_score(y_true_s, pred_maj, average="macro", zero_division=0)

        per_cls_f1 = f1_score(y_true_s, y_pred_s, average=None, labels=classes, zero_division=0)

        res_row = {
            "Stratum": s_name,
            "N_Events": n_s,
            "Pct_KD2": n_s / len(df_kd2) * 100.0,
            "Accuracy": acc_s,
            "Macro_F1": f1_s,
            "Majority_Class": maj_class,
            "Majority_Baseline_Acc": acc_maj,
            "Majority_Baseline_Macro_F1": f1_maj
        }
        for c_name, c_f1 in zip(classes, per_cls_f1):
            res_row[f"F1_{c_name}"] = c_f1
        kd2_results.append(res_row)

        print(f"\n--- {s_name} (N = {n_s:,}, {n_s/len(df_kd2)*100:.2f}%) ---")
        print(f"  Model Accuracy: {acc_s*100:.2f}%, Model Macro-F1: {f1_s:.4f}")
        print(f"  Majority Baseline ({maj_class}): Acc = {acc_maj*100:.2f}%, Macro-F1 = {f1_maj:.4f}")
        for c_name, c_f1 in zip(classes, per_cls_f1):
            print(f"    F1 ({c_name}): {c_f1:.4f}")

    df_kd2_res = pd.DataFrame(kd2_results)
    df_kd2_res.to_csv(os.path.join(models_dir, "r1_kd2_evaluation.csv"), index=False)

    # -------------------------------------------------------------------------
    # PART 5: SUPERVISOR ROBUSTNESS BATTERY
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("5. SUPERVISOR ROBUSTNESS BATTERY")
    print("-" * 70)

    # A. Gene-Level Non-Parametric Bootstrap (1,000 resamples on test set)
    print("\n[A] Running 1,000 Gene-Level Bootstrap Resamples on Test Set...")
    boot_csv = os.path.join(models_dir, "r1_bootstrap_ci.csv")
    if os.path.exists(boot_csv) and not force_rerun:
        print(f"Loading cached bootstrap confidence intervals from {boot_csv}...")
        df_ci = pd.read_csv(boot_csv)
        print(df_ci.to_string(index=False))
    else:
        gene_to_test_indices = df_kd1.iloc[test_idx].groupby("gene_id").indices
        unique_test_genes = np.array(list(gene_to_test_indices.keys()))
        n_test_genes = len(unique_test_genes)

        rng = np.random.RandomState(42)
        boot_accs, boot_f1s = [], []
        boot_class_f1s = {c: [] for c in classes}

        for b in range(1000):
            sampled_genes = rng.choice(unique_test_genes, size=n_test_genes, replace=True)
            sampled_indices = np.concatenate([gene_to_test_indices[g] for g in sampled_genes])

            b_y_true = y_test.iloc[sampled_indices]
            b_y_pred = pd.Series(winner_test_pred).iloc[sampled_indices]

            boot_accs.append(accuracy_score(b_y_true, b_y_pred))
            boot_f1s.append(f1_score(b_y_true, b_y_pred, average="macro", zero_division=0))
            c_f1s = f1_score(b_y_true, b_y_pred, average=None, labels=classes, zero_division=0)
            for c_name, c_val in zip(classes, c_f1s):
                boot_class_f1s[c_name].append(c_val)

        ci_rows = []
        # Overall Acc
        ci_rows.append({
            "Metric": "Overall Accuracy",
            "Point_Estimate": f"{acc_win*100:.2f}%",
            "Bootstrap_Mean": f"{np.mean(boot_accs)*100:.2f}%",
            "Bootstrap_SD": f"{np.std(boot_accs, ddof=1)*100:.2f}%",
            "CI_95_Lower": f"{np.percentile(boot_accs, 2.5)*100:.2f}%",
            "CI_95_Upper": f"{np.percentile(boot_accs, 97.5)*100:.2f}%",
            "CI_95_String": f"[{np.percentile(boot_accs, 2.5)*100:.2f}%, {np.percentile(boot_accs, 97.5)*100:.2f}%]"
        })
        # Macro-F1
        ci_rows.append({
            "Metric": "Macro-F1",
            "Point_Estimate": f"{f1_win:.4f}",
            "Bootstrap_Mean": f"{np.mean(boot_f1s):.4f}",
            "Bootstrap_SD": f"{np.std(boot_f1s, ddof=1):.4f}",
            "CI_95_Lower": f"{np.percentile(boot_f1s, 2.5):.4f}",
            "CI_95_Upper": f"{np.percentile(boot_f1s, 97.5):.4f}",
            "CI_95_String": f"[{np.percentile(boot_f1s, 2.5):.4f}, {np.percentile(boot_f1s, 97.5):.4f}]"
        })
        # Per-class F1
        per_cls_pt = f1_score(y_test, winner_test_pred, average=None, labels=classes, zero_division=0)
        for c_name, pt_val in zip(classes, per_cls_pt):
            c_arr = boot_class_f1s[c_name]
            ci_rows.append({
                "Metric": f"F1-Score ({c_name})",
                "Point_Estimate": f"{pt_val:.4f}",
                "Bootstrap_Mean": f"{np.mean(c_arr):.4f}",
                "Bootstrap_SD": f"{np.std(c_arr, ddof=1):.4f}",
                "CI_95_Lower": f"{np.percentile(c_arr, 2.5):.4f}",
                "CI_95_Upper": f"{np.percentile(c_arr, 97.5):.4f}",
                "CI_95_String": f"[{np.percentile(c_arr, 2.5):.4f}, {np.percentile(c_arr, 97.5):.4f}]"
            })

        df_ci = pd.DataFrame(ci_rows)
        df_ci.to_csv(boot_csv, index=False)
        print(df_ci.to_string(index=False))

    # B. Probability Calibration Assessment
    print("\n[B] Probability Calibration Assessment...")
    cal_csv = os.path.join(models_dir, "r1_calibration.csv")
    if os.path.exists(cal_csv) and not force_rerun:
        print(f"Loading cached calibration results from {cal_csv}...")
        cal_df = pd.read_csv(cal_csv)
        print(cal_df.to_string(index=False))
    else:
        y_test_onehot = pd.get_dummies(y_test)[classes].values
        brier_uncal = np.mean(np.sum((winner_test_prob - y_test_onehot) ** 2, axis=1))

        # Calibrated Classifier using GroupKFold on Training Groups
        cv_splits_train = list(GroupKFold(n_splits=5).split(X_train, y_train if winner_row["model_family"] == "RandomForest" else y_train_idx, groups=groups_train))
        if winner_row["model_family"] == "RandomForest":
            base_cal = RandomForestClassifier(
                n_estimators=int(winner_row["n_estimators"]),
                max_depth=None if pd.isna(winner_row["max_depth"]) or str(winner_row["max_depth"]) == "None" else int(winner_row["max_depth"]),
                class_weight="balanced",
                random_state=42,
                n_jobs=2
            )
            y_train_cal = y_train
        else:
            base_cal = XGBClassifier(
                n_estimators=int(winner_row["n_estimators"]),
                max_depth=int(winner_row["max_depth"]),
                learning_rate=float(winner_row["learning_rate"]),
                random_state=42,
                n_jobs=2,
                eval_metric="mlogloss",
                tree_method="hist"
            )
            y_train_cal = y_train_idx

        cal_clf = CalibratedClassifierCV(estimator=base_cal, method="sigmoid", cv=cv_splits_train)
        cal_clf.fit(X_train, y_train_cal)
        cal_test_prob = cal_clf.predict_proba(X_test)
        cal_test_pred = [classes[i] for i in np.argmax(cal_test_prob, axis=1)]
        brier_cal = np.mean(np.sum((cal_test_prob - y_test_onehot) ** 2, axis=1))
        acc_cal = accuracy_score(y_test, cal_test_pred)
        f1_cal = f1_score(y_test, cal_test_pred, average="macro", zero_division=0)

        cal_df = pd.DataFrame([
            {"Model": "Uncalibrated Winner", "Multiclass_Brier_Score": brier_uncal, "Accuracy": acc_win, "Macro_F1": f1_win},
            {"Model": "CalibratedClassifierCV (Sigmoid)", "Multiclass_Brier_Score": brier_cal, "Accuracy": acc_cal, "Macro_F1": f1_cal}
        ])
        cal_df.to_csv(cal_csv, index=False)
        print(cal_df.to_string(index=False))

        # Reliability Curves for 5 classes
        fig, axes = plt.subplots(1, 5, figsize=(20, 4), dpi=300)
        for idx, c in enumerate(classes):
            ax = axes[idx]
            y_binary = (y_test.values == c).astype(int)
            frac_pos_uncal, mean_pred_uncal = calibration_curve(y_binary, winner_test_prob[:, idx], n_bins=10, strategy="uniform")
            frac_pos_cal, mean_pred_cal = calibration_curve(y_binary, cal_test_prob[:, idx], n_bins=10, strategy="uniform")

            ax.plot([0, 1], [0, 1], "k--", label="Perfect", alpha=0.7)
            ax.plot(mean_pred_uncal, frac_pos_uncal, "s-", color="#1f77b4", label=f"Uncal (Brier={np.mean((winner_test_prob[:, idx] - y_binary)**2):.3f})")
            ax.plot(mean_pred_cal, frac_pos_cal, "o-", color="#2ca02c", label=f"Cal (Brier={np.mean((cal_test_prob[:, idx] - y_binary)**2):.3f})")
            ax.set_title(f"Class: {c}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Mean Predicted Prob", fontsize=9)
            if idx == 0:
                ax.set_ylabel("Fraction of Positives", fontsize=9)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(fontsize=8, loc="upper left")

        plt.suptitle("Reliability Curves: Uncalibrated vs Calibrated Winning Model", fontsize=13, fontweight="bold", y=1.03)
        plt.tight_layout()
        plt.savefig(os.path.join(models_dir, "r1_reliability_curves.png"), dpi=300, bbox_inches="tight")
        plt.close()
        print("Saved reliability curves: results/models/r1_reliability_curves.png")

    # C. Feature Ablations under 5-Fold GroupKFold (Full Dataset)
    print("\n[C] Feature Ablation Battery under 5-Fold GroupKFold (Full Dataset)...")
    abl_csv = os.path.join(models_dir, "r1_feature_ablations.csv")
    if os.path.exists(abl_csv) and not force_rerun:
        print(f"Loading cached feature ablations from {abl_csv}...")
        df_abl = pd.read_csv(abl_csv)
        for _, r in df_abl.iterrows():
            print(f"  {r['Feature_Subset']} (N={r['N_Features']}): Acc = {r['Accuracy_Summary']} (ddof=1), Macro-F1 = {r['Macro_F1_Summary']} (ddof=1)")
    else:
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

        abl_results = []
        for a_name, feats_sub in ablation_sets:
            accs, f1s = [], []
            for f, (t_idx, v_idx) in enumerate(gkf.split(X[feats_sub], y, groups=groups), 1):
                if winner_row["model_family"] == "RandomForest":
                    m_depth = None if winner_row["max_depth"] == "None" else int(winner_row["max_depth"])
                    clf = RandomForestClassifier(
                        n_estimators=int(winner_row["n_estimators"]),
                        max_depth=m_depth,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=2
                    )
                    clf.fit(X[feats_sub].iloc[t_idx], y.iloc[t_idx])
                    p = clf.predict(X[feats_sub].iloc[v_idx])
                    accs.append(accuracy_score(y.iloc[v_idx], p))
                    f1s.append(f1_score(y.iloc[v_idx], p, average="macro", zero_division=0))
                else:
                    clf = XGBClassifier(
                        n_estimators=int(winner_row["n_estimators"]),
                        max_depth=int(winner_row["max_depth"]),
                        learning_rate=float(winner_row["learning_rate"]),
                        random_state=42,
                        eval_metric="mlogloss",
                        tree_method="hist",
                        n_jobs=2
                    )
                    sw = compute_sample_weight("balanced", y_idx.iloc[t_idx])
                    clf.fit(X[feats_sub].iloc[t_idx], y_idx.iloc[t_idx], sample_weight=sw)
                    p = clf.predict(X[feats_sub].iloc[v_idx])
                    accs.append(accuracy_score(y_idx.iloc[v_idx], p))
                    f1s.append(f1_score(y_idx.iloc[v_idx], p, average="macro", zero_division=0))

            m_a, s_a = np.mean(accs), np.std(accs, ddof=1)
            m_f, s_f = np.mean(f1s), np.std(f1s, ddof=1)
            abl_results.append({
                "Feature_Subset": a_name,
                "N_Features": len(feats_sub),
                "Mean_Accuracy": m_a,
                "Std_Accuracy": s_a,
                "Accuracy_Summary": f"{m_a*100:.2f}% ± {s_a*100:.2f}%",
                "Mean_Macro_F1": m_f,
                "Std_Macro_F1": s_f,
                "Macro_F1_Summary": f"{m_f:.4f} ± {s_f:.4f}"
            })
            print(f"  {a_name} (N={len(feats_sub)}): Acc = {m_a*100:.2f}% ± {s_a*100:.2f}%, Macro-F1 = {m_f:.4f} ± {s_f:.4f} (ddof=1)")

        df_abl = pd.DataFrame(abl_results)
        df_abl.to_csv(abl_csv, index=False)

    # D. Mutual Information & Leakage Diagnostics
    print("\n[D] Information Leakage Diagnostics...")
    mi_csv = os.path.join(models_dir, "r1_mutual_information.csv")
    if os.path.exists(mi_csv) and not force_rerun:
        print(f"Loading cached mutual information from {mi_csv}...")
        df_mi = pd.read_csv(mi_csv)
        print("Top 10 Features by Mutual Information:")
        print(df_mi.head(10).to_string(index=False))
    else:
        mi_scores = mutual_info_classif(X[feat_23], y_idx, random_state=42)
        df_mi = pd.DataFrame({
            "Feature": feat_23,
            "Mutual_Information": mi_scores
        }).sort_values(by="Mutual_Information", ascending=False).reset_index(drop=True)
        df_mi["Rank"] = range(1, len(df_mi) + 1)
        df_mi.to_csv(mi_csv, index=False)
        print("Top 10 Features by Mutual Information:")
        print(df_mi.head(10).to_string(index=False))

    # Exact duplicate feature rows
    n_dups = df_kd1[feat_23].duplicated().sum()
    print(f"\nExact duplicate feature rows in KD1: {n_dups}")

    # E. Duplicates and Overlaps
    print("\n[E] Duplicates and Overlap Analysis...")
    coord_cols = ["chrom", "strand", "exon_start", "exon_end"]
    df_coords = df_kd1[coord_cols + ["label"]].copy()
    coord_dups = df_coords[df_coords.duplicated(subset=coord_cols, keep=False)]
    grouped_coords = coord_dups.groupby(coord_cols)["label"].nunique()
    diff_label_coords = grouped_coords[grouped_coords > 1]
    n_diff_label_loci = len(diff_label_coords)
    n_events_diff_labels = df_coords.set_index(coord_cols).index.isin(diff_label_coords.index).sum()
    print(f"Coordinate loci appearing with DIFFERENT splicing classes: {n_diff_label_loci}")
    print(f"Events with identical coordinates across classes: {n_events_diff_labels}")

    # Outer interval overlap check (same chrom, same strand)
    df_train_sub = df_kd1.iloc[train_idx].copy()
    df_test_sub = df_kd1.iloc[test_idx].copy()

    df_train_sub["outer_start"] = df_train_sub["exon_start"] - df_train_sub["upstream_intron_length"].fillna(0)
    df_train_sub["outer_end"] = df_train_sub["exon_end"] + df_train_sub["downstream_intron_length"].fillna(0)
    df_test_sub["outer_start"] = df_test_sub["exon_start"] - df_test_sub["upstream_intron_length"].fillna(0)
    df_test_sub["outer_end"] = df_test_sub["exon_end"] + df_test_sub["downstream_intron_length"].fillna(0)

    train_outer = {}
    for (chrom, strand), grp in df_train_sub.groupby(["chrom", "strand"]):
        train_outer[(chrom, strand)] = grp[["outer_start", "outer_end"]].values

    overlap_mask_outer = []
    for _, row in df_test_sub.iterrows():
        key = (row["chrom"], row["strand"])
        if key not in train_outer:
            overlap_mask_outer.append(False)
            continue
        s, e = row["outer_start"], row["outer_end"]
        intervals = train_outer[key]
        has_overlap = np.any((s <= intervals[:, 1]) & (e >= intervals[:, 0]))
        overlap_mask_outer.append(has_overlap)
    overlap_mask_outer = np.array(overlap_mask_outer)

    n_test_over = int(overlap_mask_outer.sum())
    n_test_no = len(df_test_sub) - n_test_over
    print(f"\nOuter Genomic Interval Overlap (Test vs Train):")
    print(f"  - Test events overlapping train: {n_test_over} ({n_test_over/len(df_test_sub)*100:.2f}%)")
    print(f"  - Test events NOT overlapping train: {n_test_no} ({n_test_no/len(df_test_sub)*100:.2f}%)")

    # Evaluate performance with vs without outer interval overlap
    y_test_pred_arr = np.array(winner_test_pred)
    overlap_eval_rows = []
    for mask, name in [(overlap_mask_outer, "With Outer Overlap"), (~overlap_mask_outer, "Without Outer Overlap")]:
        y_sub_true = y_test.iloc[mask]
        y_sub_pred = y_test_pred_arr[mask]
        acc_sub = accuracy_score(y_sub_true, y_sub_pred)
        f1_sub = f1_score(y_sub_true, y_sub_pred, average="macro", zero_division=0)
        cr_sub = classification_report(y_sub_true, y_sub_pred, labels=classes, output_dict=True, zero_division=0)
        print(f"\nStratum: {name} (N = {len(y_sub_true)}):")
        print(f"  Accuracy: {acc_sub*100:.2f}%, Macro-F1: {f1_sub:.4f}")
        row_dict = {
            "Stratum": name, "N": len(y_sub_true), "Accuracy": acc_sub, "Macro_F1": f1_sub
        }
        for c in classes:
            print(f"    - {c}: N={int(cr_sub[c]['support'])}, F1={cr_sub[c]['f1-score']:.4f}")
            row_dict[f"F1_{c}"] = cr_sub[c]["f1-score"]
            row_dict[f"N_{c}"] = int(cr_sub[c]["support"])
        overlap_eval_rows.append(row_dict)

    df_overlap = pd.DataFrame(overlap_eval_rows)
    df_overlap.to_csv(os.path.join(models_dir, "r1_locus_overlap_evaluation.csv"), index=False)

    # F. Technical Negative Control (Lane Comparison)
    print("\n[F] Technical Negative Control (Lane Comparison)...")
    lane_dir = "results/splicing_lane_control_vrl"
    lane_dfs = []
    for et in ["SE", "MXE", "A3SS", "A5SS", "RI"]:
        f_p = os.path.join(lane_dir, f"{et}.MATS.JC.txt")
        if os.path.exists(f_p):
            d = pd.read_csv(f_p, sep="\t")
            d["EventType"] = et
            lane_dfs.append(d)
    all_lane = pd.concat(lane_dfs, ignore_index=True)
    sig_lane = all_lane[all_lane["FDR"] < 0.05].copy()

    lane_rows = []
    for et in ["SE", "MXE", "A3SS", "A5SS", "RI"]:
        s_et = sig_lane[sig_lane["EventType"] == et]
        up_c = (s_et["IncLevelDifference"] > 0).sum()
        dn_c = (s_et["IncLevelDifference"] < 0).sum()
        tot_c = len(s_et)
        if tot_c > 0:
            b_test = binomtest(up_c, n=tot_c, p=0.5)
            p_val = b_test.pvalue
            ci = b_test.proportion_ci(confidence_level=0.95)
            ci_low, ci_high = ci.low, ci.high
        else:
            p_val, ci_low, ci_high = 1.0, 0.0, 1.0
        lane_rows.append({
            "event_class": et,
            "total_fdr_under_0_05": tot_c,
            "up_dpsi_gt_0": up_c,
            "down_dpsi_lt_0": dn_c,
            "exact_binomial_p": p_val,
            "ci_95_low": ci_low,
            "ci_95_high": ci_high
        })

    tot_up = (sig_lane["IncLevelDifference"] > 0).sum()
    tot_dn = (sig_lane["IncLevelDifference"] < 0).sum()
    tot_sig_lane = len(sig_lane)
    b_tot = binomtest(tot_up, n=tot_sig_lane, p=0.5)
    ci_t = b_tot.proportion_ci(confidence_level=0.95)
    lane_rows.append({
        "event_class": "Total",
        "total_fdr_under_0_05": tot_sig_lane,
        "up_dpsi_gt_0": tot_up,
        "down_dpsi_lt_0": tot_dn,
        "exact_binomial_p": b_tot.pvalue,
        "ci_95_low": ci_t.low,
        "ci_95_high": ci_t.high
    })
    df_lane_p = pd.DataFrame(lane_rows)
    df_lane_p.to_csv(os.path.join(models_dir, "r1_lane_control_table.csv"), index=False)
    print(df_lane_p.to_string(index=False))

    abs_dpsi_lane = sig_lane["IncLevelDifference"].abs()
    iqr_lane = abs_dpsi_lane.quantile(0.75) - abs_dpsi_lane.quantile(0.25)
    pct_ge_01 = (abs_dpsi_lane >= 0.1).mean() * 100
    pct_ge_05 = (abs_dpsi_lane >= 0.5).mean() * 100

    print(f"\nLane Control |dPSI| Distribution (N = {len(sig_lane):,}):")
    print(f"  Min: {abs_dpsi_lane.min():.4f}")
    print(f"  25%: {abs_dpsi_lane.quantile(0.25):.4f}")
    print(f"  Median: {abs_dpsi_lane.median():.4f}")
    print(f"  75%: {abs_dpsi_lane.quantile(0.75):.4f}")
    print(f"  IQR: {iqr_lane:.4f}")
    print(f"  90%: {abs_dpsi_lane.quantile(0.90):.4f}")
    print(f"  95%: {abs_dpsi_lane.quantile(0.95):.4f}")
    print(f"  Max: {abs_dpsi_lane.max():.4f}")
    print(f"  Mean ± SD: {abs_dpsi_lane.mean():.4f} ± {abs_dpsi_lane.std(ddof=1):.4f} (ddof=1)")
    print(f"  Share with |dPSI| >= 0.1: {pct_ge_01:.2f}%")
    print(f"  Share with |dPSI| >= 0.5: {pct_ge_05:.2f}%")

    plt.figure(figsize=(7, 5), dpi=300)
    plt.hist(abs_dpsi_lane, bins=50, color="#1f77b4", edgecolor="black", alpha=0.7)
    plt.title("Technical Negative Control (|ΔPSI| Distribution at FDR < 0.05)")
    plt.xlabel("|ΔPSI|")
    plt.ylabel("Event Count")
    plt.axvline(abs_dpsi_lane.median(), color="red", linestyle="--", label=f"Median = {abs_dpsi_lane.median():.4f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, "r1_lane_control_abs_dpsi_histogram.png"), dpi=300)
    plt.close()
    print("Saved Lane Control |dPSI| Histogram: results/models/r1_lane_control_abs_dpsi_histogram.png")

    def parse_junc(v):
        if pd.isna(v): return 0
        return sum(int(x) for x in str(v).split(",") if x.strip())

    sig_lane["junc"] = (
        sig_lane["IJC_SAMPLE_1"].apply(parse_junc) + sig_lane["SJC_SAMPLE_1"].apply(parse_junc) +
        sig_lane["IJC_SAMPLE_2"].apply(parse_junc) + sig_lane["SJC_SAMPLE_2"].apply(parse_junc)
    )

    # KD1 Junction reads for comparison
    kd1_dir = "results/splicing_kd1_vrl"
    kd1_dfs = []
    for et in ["SE", "MXE", "A3SS", "A5SS", "RI"]:
        f_p = os.path.join(kd1_dir, f"{et}.MATS.JC.txt")
        if os.path.exists(f_p):
            d = pd.read_csv(f_p, sep="\t")
            d["EventType"] = et
            kd1_dfs.append(d)
    all_kd1 = pd.concat(kd1_dfs, ignore_index=True)
    sig_kd1 = all_kd1[all_kd1["FDR"] < 0.05].copy()
    sig_kd1["junc"] = (
        sig_kd1["IJC_SAMPLE_1"].apply(parse_junc) + sig_kd1["SJC_SAMPLE_1"].apply(parse_junc) +
        sig_kd1["IJC_SAMPLE_2"].apply(parse_junc) + sig_kd1["SJC_SAMPLE_2"].apply(parse_junc)
    )

    junc_comp = pd.DataFrame([
        {
            "Cohort": "Lane Control (ERR127306 vs ERR127307)",
            "N_Events": len(sig_lane),
            "Median_Junction_Reads": sig_lane["junc"].median(),
            "Q25_Junction_Reads": sig_lane["junc"].quantile(0.25),
            "Q75_Junction_Reads": sig_lane["junc"].quantile(0.75),
            "IQR_Junction_Reads": sig_lane["junc"].quantile(0.75) - sig_lane["junc"].quantile(0.25),
            "Mean_Junction_Reads": sig_lane["junc"].mean(),
            "Std_Junction_Reads": sig_lane["junc"].std(ddof=1)
        },
        {
            "Cohort": "KD1 Discovery (KD vs Downsampled Control)",
            "N_Events": len(sig_kd1),
            "Median_Junction_Reads": sig_kd1["junc"].median(),
            "Q25_Junction_Reads": sig_kd1["junc"].quantile(0.25),
            "Q75_Junction_Reads": sig_kd1["junc"].quantile(0.75),
            "IQR_Junction_Reads": sig_kd1["junc"].quantile(0.75) - sig_kd1["junc"].quantile(0.25),
            "Mean_Junction_Reads": sig_kd1["junc"].mean(),
            "Std_Junction_Reads": sig_kd1["junc"].std(ddof=1)
        }
    ])
    junc_comp.to_csv(os.path.join(models_dir, "r1_junction_distribution_comparison.csv"), index=False)
    print("\nJunction Count Distribution Comparison:")
    print(junc_comp.to_string(index=False))

    # F. Dummy Baselines on Held-Out Test Set
    print("\n[F] Dummy Classifier Baselines on Held-Out Test Set...")
    dummy_results = []
    for strat in ["most_frequent", "stratified"]:
        d = DummyClassifier(strategy=strat, random_state=42)
        d.fit(np.zeros((len(y_train), 1)), y_train)
        pred_d = d.predict(np.zeros((len(y_test), 1)))
        acc_d = accuracy_score(y_test, pred_d)
        f1_d = f1_score(y_test, pred_d, average="macro", zero_division=0)
        dummy_results.append({
            "Strategy": strat,
            "Accuracy": acc_d,
            "Macro_F1": f1_d
        })
        print(f"\n--- DummyClassifier strategy='{strat}' ---")
        print(f"  Accuracy: {acc_d*100:.2f}%, Macro-F1: {f1_d:.4f}")
        print(classification_report(y_test, pred_d, digits=4, zero_division=0))

    df_dummy = pd.DataFrame(dummy_results)
    df_dummy.to_csv(os.path.join(models_dir, "r1_dummy_baselines.csv"), index=False)

    # Save trained models
    joblib.dump(winner_clf, os.path.join(models_dir, "r1_winner_xgboost.joblib"))
    joblib.dump(runner_clf, os.path.join(models_dir, "r1_secondary_rf.joblib"))
    print(f"\nSaved models: results/models/r1_winner_xgboost.joblib, results/models/r1_secondary_rf.joblib")

    print("\n" + "=" * 80)
    print("R1 PROTOCOL & SUPERVISOR BATTERY COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main()
