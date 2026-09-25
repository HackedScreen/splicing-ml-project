#!/usr/bin/env python3
"""
scripts/07_ensemble_and_threshold_tuning.py

BSc Dissertation Pipeline — Round 3:
1. Soft-Voting Ensemble (RF V2 + XGBoost V2):
   - Grid searches RF:XGBoost probability-averaging weight w_rf in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
     strictly using 5-fold CV out-of-fold (OOF) predictions on the training set (N = 6,954).
   - Locks winning weight based on CV Macro-F1.
   - Evaluates once on Internal Test Set (N = 1,739) and KD2 External Validation (N = 7,080).
   - Checks if ensemble beats RF V2 on both cohorts.

2. Per-Class Decision Threshold Tuning:
   - Uses ONLY training set 5-fold CV OOF predictions from RF V2.
   - Tunes per-class probability multipliers m = [m_0, ..., m_4] to maximize CV Macro-F1.
   - Locks in optimal multipliers m*.
   - Evaluates once on Internal Test Set (N = 1,739) and KD2 External Validation (N = 7,080).
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier

def main():
    base_dir = os.path.expanduser("~/splicing-ml-project")
    models_dir = os.path.join(base_dir, "results", "models")
    features_dir = os.path.join(base_dir, "results", "features")

    data_path = os.path.join(models_dir, "train_test_data_v2.joblib")
    rf_path = os.path.join(models_dir, "random_forest_v2.joblib")
    xgb_path = os.path.join(models_dir, "xgboost_v2.joblib")
    kd2_feat_path = os.path.join(features_dir, "feature_matrix_kd2_v2.csv")

    print("=" * 80)
    print("ROUND 3: SOFT-VOTING ENSEMBLE & DECISION THRESHOLD TUNING")
    print("=" * 80)

    # 1. Load Data and Models
    print(f"Loading training/test data from {data_path}...")
    data = joblib.load(data_path)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    classes = data["classes"]
    le = data["label_encoder"]
    feature_names = data["feature_names"]

    print(f"Training set: N = {len(X_train):,}, Test set: N = {len(X_test):,}")
    print(f"Classes ({len(classes)}): {classes}")

    print(f"Loading pre-trained RF V2 and XGBoost V2...")
    rf_model = joblib.load(rf_path)
    xgb_model = joblib.load(xgb_path)

    # 2. Load KD2 Dataset
    print(f"Loading KD2 validation data from {kd2_feat_path}...")
    df_kd2 = pd.read_csv(kd2_feat_path)
    psi_cols = ["psi_control_mean", "psi_control_var", "psi_kd_mean", "psi_kd_var"]
    df_kd2[psi_cols] = df_kd2[psi_cols].fillna(0.0)

    numeric_features = [
        "exon_length", "gc_exon", "gc_upstream_100bp", "gc_downstream_100bp",
        "max_utract_length_upstream", "max_utract_length_downstream",
        "u_fraction_upstream", "u_fraction_downstream",
        "score_5ss_donor", "score_3ss_acceptor",
        "cov_mean_control", "cov_var_control", "cov_mean_kd", "cov_var_kd",
        "cov_mean_overall", "cov_var_overall",
        "psi_control_mean", "psi_control_var", "psi_kd_mean", "psi_kd_var",
        "dpsi"
    ]
    strand_dummies_kd2 = pd.get_dummies(df_kd2["strand"], prefix="strand", dtype=float)
    if "strand_+" not in strand_dummies_kd2.columns:
        strand_dummies_kd2["strand_+"] = 0.0
    if "strand_-" not in strand_dummies_kd2.columns:
        strand_dummies_kd2["strand_-"] = 0.0
    strand_dummies_kd2 = strand_dummies_kd2[["strand_+", "strand_-"]]

    X_kd2 = pd.concat([df_kd2[numeric_features], strand_dummies_kd2], axis=1)
    y_kd2 = df_kd2["label"]
    assert list(X_kd2.columns) == feature_names, "KD2 column mismatch!"
    print(f"KD2 set: N = {len(X_kd2):,}")

    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    y_kd2_enc = le.transform(y_kd2)

    # -------------------------------------------------------------------------
    # Baseline RF V2 and XGB V2 Performance Check
    # -------------------------------------------------------------------------
    prob_rf_test = rf_model.predict_proba(X_test)
    prob_xgb_test = xgb_model.predict_proba(X_test)
    prob_rf_kd2 = rf_model.predict_proba(X_kd2)
    prob_xgb_kd2 = xgb_model.predict_proba(X_kd2)

    pred_rf_test = np.argmax(prob_rf_test, axis=1)
    pred_xgb_test = np.argmax(prob_xgb_test, axis=1)
    pred_rf_kd2 = np.argmax(prob_rf_kd2, axis=1)
    pred_xgb_kd2 = np.argmax(prob_xgb_kd2, axis=1)

    print("\n--- Baseline Model Verification ---")
    print(f"RF V2 Internal Test : Acc = {accuracy_score(y_test_enc, pred_rf_test)*100:.2f}%, Macro-F1 = {f1_score(y_test_enc, pred_rf_test, average='macro'):.4f}")
    print(f"RF V2 KD2 Validation: Acc = {accuracy_score(y_kd2_enc, pred_rf_kd2)*100:.2f}%, Macro-F1 = {f1_score(y_kd2_enc, pred_rf_kd2, average='macro'):.4f}")
    print(f"XGB V2 Internal Test : Acc = {accuracy_score(y_test_enc, pred_xgb_test)*100:.2f}%, Macro-F1 = {f1_score(y_test_enc, pred_xgb_test, average='macro'):.4f}")
    print(f"XGB V2 KD2 Validation: Acc = {accuracy_score(y_kd2_enc, pred_xgb_kd2)*100:.2f}%, Macro-F1 = {f1_score(y_kd2_enc, pred_xgb_kd2, average='macro'):.4f}")

    # -------------------------------------------------------------------------
    # STEP 2: SOFT-VOTING ENSEMBLE (RF V2 + XGBoost V2)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 2: SOFT-VOTING ENSEMBLE (5-FOLD CV ON TRAINING SET ONLY)")
    print("=" * 80)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_prob_rf = np.zeros((len(X_train), len(classes)))
    oof_prob_xgb = np.zeros((len(X_train), len(classes)))

    print("Generating 5-fold Out-Of-Fold (OOF) cross-validation probabilities on training set...")
    for fold, (trn_idx, val_idx) in enumerate(cv.split(X_train, y_train_enc), 1):
        X_tr, y_tr = X_train.iloc[trn_idx], y_train_enc[trn_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train_enc[val_idx]

        # Fit RF fold
        rf_fold = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
        rf_fold.fit(X_tr, y_tr)
        oof_prob_rf[val_idx] = rf_fold.predict_proba(X_va)

        # Fit XGB fold with balanced sample weights
        sw_tr = compute_sample_weight("balanced", y_tr)
        xgb_fold = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            random_state=42, eval_metric="mlogloss", tree_method="hist", n_jobs=-1
        )
        xgb_fold.fit(X_tr, y_tr, sample_weight=sw_tr)
        oof_prob_xgb[val_idx] = xgb_fold.predict_proba(X_va)
        print(f"  Fold {fold}/5 complete.")

    # Grid search RF weight in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0] strictly on OOF
    print("\nGrid Search over RF:XGBoost ensemble weights on CV Out-Of-Fold predictions:")
    rf_weights = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    best_w = None
    best_cv_f1 = -1.0
    cv_results = []

    for w_rf in rf_weights:
        w_xgb = 1.0 - w_rf
        ens_oof = w_rf * oof_prob_rf + w_xgb * oof_prob_xgb
        pred_oof = np.argmax(ens_oof, axis=1)
        f1_cv = f1_score(y_train_enc, pred_oof, average="macro")
        acc_cv = accuracy_score(y_train_enc, pred_oof)
        cv_results.append((w_rf, w_xgb, acc_cv, f1_cv))
        print(f"  Weight RF={w_rf:.1f}, XGB={w_xgb:.1f} -> CV Accuracy = {acc_cv*100:.2f}%, CV Macro-F1 = {f1_cv:.4f}")
        if f1_cv > best_cv_f1:
            best_cv_f1 = f1_cv
            best_w = w_rf

    print(f"\nWinning Ensemble Weight in CV: RF={best_w:.1f}, XGB={1.0 - best_w:.1f} (CV Macro-F1: {best_cv_f1:.4f})")
    print("LOCKING IN winning ensemble weight before test-set and KD2 evaluation.")

    # Evaluate locked ensemble on Internal Test and KD2
    w_rf_star = best_w
    w_xgb_star = 1.0 - w_rf_star

    prob_ens_test = w_rf_star * prob_rf_test + w_xgb_star * prob_xgb_test
    pred_ens_test = np.argmax(prob_ens_test, axis=1)
    acc_ens_test = accuracy_score(y_test_enc, pred_ens_test)
    f1_ens_test = f1_score(y_test_enc, pred_ens_test, average="macro")

    prob_ens_kd2 = w_rf_star * prob_rf_kd2 + w_xgb_star * prob_xgb_kd2
    pred_ens_kd2 = np.argmax(prob_ens_kd2, axis=1)
    acc_ens_kd2 = accuracy_score(y_kd2_enc, pred_ens_kd2)
    f1_ens_kd2 = f1_score(y_kd2_enc, pred_ens_kd2, average="macro")

    print("\n--- Locked Soft-Voting Ensemble Results ---")
    print(f"Internal Test : Accuracy = {acc_ens_test*100:.2f}%, Macro-F1 = {f1_ens_test:.4f}")
    print(f"KD2 Validation: Accuracy = {acc_ens_kd2*100:.2f}%, Macro-F1 = {f1_ens_kd2:.4f}")

    beats_test = (acc_ens_test > 0.7481) and (f1_ens_test > 0.6611)
    beats_kd2 = (acc_ens_kd2 > 0.7682) and (f1_ens_kd2 > 0.7003)
    ensemble_beats_both = beats_test and beats_kd2
    print(f"Beats RF V2 on Internal Test? {beats_test}")
    print(f"Beats RF V2 on KD2 Validation? {beats_kd2}")
    print(f"Decision: {'ADOPT ENSEMBLE AS NEW CANDIDATE' if ensemble_beats_both else 'DROP ENSEMBLE (Does not beat RF V2 on both cohorts)'}")

    # -------------------------------------------------------------------------
    # STEP 3: PER-CLASS DECISION THRESHOLD TUNING (ON RF V2 CV OOF)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 3: PER-CLASS DECISION THRESHOLD TUNING (RF V2 CV OOF ONLY)")
    print("=" * 80)

    # Class distribution in training set
    print("Training set class counts:")
    for idx, c in enumerate(classes):
        cnt = np.sum(y_train_enc == idx)
        print(f"  Class {idx} ({c:<5}): {cnt:>5} events ({cnt/len(y_train_enc)*100:>5.2f}%)")

    # Evaluate RF V2 CV baseline with standard argmax
    pred_rf_oof_baseline = np.argmax(oof_prob_rf, axis=1)
    cv_baseline_acc = accuracy_score(y_train_enc, pred_rf_oof_baseline)
    cv_baseline_f1 = f1_score(y_train_enc, pred_rf_oof_baseline, average="macro")
    print(f"\nRF V2 CV OOF Baseline (argmax): Acc = {cv_baseline_acc*100:.2f}%, Macro-F1 = {cv_baseline_f1:.4f}")

    def eval_multipliers(multipliers, probs, y_true):
        scaled = probs * multipliers
        preds = np.argmax(scaled, axis=1)
        return f1_score(y_true, preds, average="macro")

    # Optimization: tune multipliers m = [m0, m1, m2, m3, m4]
    # Bound search, normalizing m[SE] (index 4) = 1.0
    # Search rare class multipliers m0, m1, m2, m3 in range [0.5, 5.0]
    best_multipliers = np.ones(len(classes))
    best_opt_cv_f1 = cv_baseline_f1

    # Grid search for initial warm start over rare classes
    print("\nPerforming coarse grid search over rare-class multipliers on OOF...")
    for m_rare in [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
        for m_mxe in [1.0, 1.25, 1.5, 2.0, 2.5]:
            # A3SS=0, A5SS=1, MXE=2, RI=3, SE=4
            cand = np.array([m_rare, m_rare, m_mxe, 1.0, 1.0])
            score = eval_multipliers(cand, oof_prob_rf, y_train_enc)
            if score > best_opt_cv_f1:
                best_opt_cv_f1 = score
                best_multipliers = cand.copy()

    print(f"Warm start multipliers: {best_multipliers} -> CV Macro-F1 = {best_opt_cv_f1:.4f}")

    # Fine-tune with Powell optimization
    def obj_func(weights):
        # weights are 5 positive numbers
        w = np.maximum(weights, 0.01)
        score = eval_multipliers(w, oof_prob_rf, y_train_enc)
        return -score

    opt_res = minimize(
        obj_func,
        x0=best_multipliers,
        method="Powell",
        options={"maxiter": 200, "ftol": 1e-4}
    )

    refined_multipliers = np.maximum(opt_res.x, 0.01)
    refined_multipliers /= refined_multipliers[4] # normalize relative to SE = 1.0
    tuned_cv_f1 = eval_multipliers(refined_multipliers, oof_prob_rf, y_train_enc)

    if tuned_cv_f1 > best_opt_cv_f1:
        final_multipliers = refined_multipliers
        final_tuned_cv_f1 = tuned_cv_f1
    else:
        final_multipliers = best_multipliers
        final_tuned_cv_f1 = best_opt_cv_f1

    print(f"\nOptimal Locked Class Multipliers (relative to SE=1.0):")
    for idx, c in enumerate(classes):
        print(f"  Multiplier for {c:<5}: {final_multipliers[idx]:.4f}")
    print(f"Optimized Training CV Macro-F1: {final_tuned_cv_f1:.4f} (Baseline: {cv_baseline_f1:.4f}, Gain: +{(final_tuned_cv_f1 - cv_baseline_f1):.4f})")
    print("LOCKING IN tuned decision thresholds before evaluation.")

    # Apply locked multipliers to Test and KD2
    def predict_with_multipliers(probs, multipliers):
        scaled = probs * multipliers
        return np.argmax(scaled, axis=1)

    pred_thresh_test = predict_with_multipliers(prob_rf_test, final_multipliers)
    acc_thresh_test = accuracy_score(y_test_enc, pred_thresh_test)
    f1_thresh_test = f1_score(y_test_enc, pred_thresh_test, average="macro")

    pred_thresh_kd2 = predict_with_multipliers(prob_rf_kd2, final_multipliers)
    acc_thresh_kd2 = accuracy_score(y_kd2_enc, pred_thresh_kd2)
    f1_thresh_kd2 = f1_score(y_kd2_enc, pred_thresh_kd2, average="macro")

    print("\n--- Locked Threshold-Tuned RF V2 Results ---")
    print(f"Internal Test : Accuracy = {acc_thresh_test*100:.2f}%, Macro-F1 = {f1_thresh_test:.4f}")
    print(f"KD2 Validation: Accuracy = {acc_thresh_kd2*100:.2f}%, Macro-F1 = {f1_thresh_kd2:.4f}")

    print("\nClassification Report (Internal Test — Threshold-Tuned):")
    print(classification_report(y_test_enc, pred_thresh_test, target_names=classes, digits=4))

    print("\nClassification Report (KD2 Validation — Threshold-Tuned):")
    print(classification_report(y_kd2_enc, pred_thresh_kd2, target_names=classes, digits=4))

    # Save summary metrics dictionary for consolidation
    results = {
        "rf_baseline": {
            "test_acc": 0.7481, "test_f1": 0.6611,
            "kd2_acc": 0.7682, "kd2_f1": 0.7003
        },
        "ensemble": {
            "best_w_rf": float(w_rf_star),
            "cv_f1": float(best_cv_f1),
            "test_acc": float(acc_ens_test), "test_f1": float(f1_ens_test),
            "kd2_acc": float(acc_ens_kd2), "kd2_f1": float(f1_ens_kd2),
            "decision": "ADOPT" if ensemble_beats_both else "DROP"
        },
        "threshold_tuning": {
            "multipliers": {c: float(final_multipliers[i]) for i, c in enumerate(classes)},
            "cv_baseline_f1": float(cv_baseline_f1),
            "cv_tuned_f1": float(final_tuned_cv_f1),
            "test_acc": float(acc_thresh_test), "test_f1": float(f1_thresh_test),
            "kd2_acc": float(acc_thresh_kd2), "kd2_f1": float(f1_thresh_kd2)
        }
    }
    out_json = os.path.join(models_dir, "round3_step2_step3_results.joblib")
    joblib.dump(results, out_json)
    print(f"\nSaved Step 2 & 3 results to {out_json}")

if __name__ == "__main__":
    main()
