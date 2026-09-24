#!/usr/bin/env python3
"""
scripts/06_train_models.py

BSc Dissertation Pipeline — Stage 6: Model Training & Hyperparameter Tuning
Trains multi-class classification models (Logistic Regression and Random Forest)
to predict alternative splicing event types from structural, sequence, coverage,
and PSI features.

Outputs:
  results/models/logistic_regression.joblib
  results/models/random_forest.joblib
  results/models/train_test_data.joblib
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, balanced_accuracy_score

def main():
    base_dir = os.path.expanduser("~/splicing-ml-project")
    results_dir = os.path.join(base_dir, "results")
    feat_path = os.path.join(results_dir, "features", "feature_matrix.csv")
    models_dir = os.path.join(results_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    print(f"Loading feature matrix from: {feat_path}")
    df = pd.read_csv(feat_path)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns.")

    # -------------------------------------------------------------------------
    # 1. Missing Value Imputation (Zero-read loci imputed with 0.0)
    # -------------------------------------------------------------------------
    psi_cols = ["psi_control_mean", "psi_control_var", "psi_kd_mean", "psi_kd_var"]
    nan_before = df[psi_cols].isnull().sum().to_dict()
    print(f"Missing values before imputation: {nan_before}")
    df[psi_cols] = df[psi_cols].fillna(0.0)
    print("Imputed all PSI NaN values with 0.0 (biologically: zero observed reads = 0 inclusion).")

    # -------------------------------------------------------------------------
    # 2. Feature Selection (21 numeric structural/sequence/coverage/PSI features)
    # Exclude pvalue and fdr to avoid testing power artifact / label leakage
    # -------------------------------------------------------------------------
    numeric_features = [
        "exon_length",
        "upstream_intron_length",
        "downstream_intron_length",
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

    # One-hot encode strand (+ and -) -> 2 columns, total 21 features
    strand_dummies = pd.get_dummies(df["strand"], prefix="strand", dtype=float)
    
    X = pd.concat([df[numeric_features], strand_dummies], axis=1)
    y = df["label"]
    feature_names = list(X.columns)

    print(f"\nFinal feature set ({len(feature_names)} features):")
    print(feature_names)
    print(f"Target class distribution:\n{y.value_counts()}")

    # -------------------------------------------------------------------------
    # 3. Stratified 80/20 Train/Test Split (fixed random_state=42)
    # -------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set (held-out): {X_test.shape[0]} samples")
    print("Train label distribution:\n", y_train.value_counts())
    print("Test label distribution:\n", y_test.value_counts())

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
            multi_class="multinomial",
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

    print(f"\nBest Random Forest Params: {rf_grid.best_params_}")
    print(f"Best Random Forest 5-Fold CV Score (F1-Macro): {rf_grid.best_score_:.4f}")

    # -------------------------------------------------------------------------
    # 6. Save Models and Split Data
    # -------------------------------------------------------------------------
    lr_model_path = os.path.join(models_dir, "logistic_regression.joblib")
    rf_model_path = os.path.join(models_dir, "random_forest.joblib")
    split_data_path = os.path.join(models_dir, "train_test_data.joblib")

    joblib.dump(lr_grid.best_estimator_, lr_model_path)
    joblib.dump(rf_grid.best_estimator_, rf_model_path)
    joblib.dump({
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names
    }, split_data_path)

    print("\n" + "="*70)
    print("SAVED ARTIFACTS:")
    print(f"  - Logistic Regression Model: {lr_model_path}")
    print(f"  - Random Forest Model:        {rf_model_path}")
    print(f"  - Train/Test Data Split:      {split_data_path}")
    print("="*70)

if __name__ == "__main__":
    main()