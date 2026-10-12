# Model Checksums & Binary Artifact Audit

**Date**: 2026-10-09  
**Status**: All `.joblib` serialized model and partition files are explicitly excluded from Git version control via `.gitignore` (`*.joblib`) due to GitHub file size limits and binary immutability standards.

This document serves as the immutable cryptographic register (SHA256) of all trained machine learning models, baseline pipelines, and partition dictionaries used across the dissertation. Checksums reflect authentic binary contents on disk in `results/models/`.

---

## 1. Primary and Robustness Machine Learning Models

| Artifact Filename | Description | Size (bytes) | SHA256 Checksum |
| :--- | :--- | :---: | :--- |
| `random_forest_v2_gene_grouped.joblib` | Gene-grouped retrained Random Forest V2 (frozen parameters: `n_est=300, max_depth=None`, 23 features) | 72,228,785 | `931a2963a01837b9455b0f1c5a4d082e8675be94bd2b2046d03441328973f5ab` |
| `random_forest_v2.joblib` | Random Forest V2 (incorporating 4 HNRNPC U-tract motif features, 23 features total, random split) | 73,268,049 | `518d033184c355af9de82a932655e0e716906db25f39ecfcf246af9ef749836f` |
| `random_forest_corrected.joblib` | Promoted primary model (identical binary to `random_forest_v2.joblib`, 23 features) | 73,268,049 | `518d033184c355af9de82a932655e0e716906db25f39ecfcf246af9ef749836f` |
| `random_forest_corrected_baseline_backup.joblib` | Archived 19-feature Random Forest baseline (Stage 6 corrected, prior to U-tract addition) | 72,548,065 | `9dfaf20894fdbcfa1d2e365fcedd0fe48120393b2d24eb36a3a384009024135a` |
| `random_forest_v3_maxent.joblib` | Random Forest V3 with genuine MaxEntScan splice site scores (training CV winner: `max_depth=20, n_est=100`, 23 features) | 26,336,081 | `237277508704c573ce95f8e133c5b7ed1cba20156d9cee9a7e2bec10f305a04b` |
| `xgboost_v2.joblib` | Extreme Gradient Boosting V2 model (incorporating U-tract features, `lr=0.05, max_depth=6, n_est=300`) | 4,561,797 | `43428687eb13e48a9ffedd20343cef74fe3adb41f6947bba5d1e1192e498627b` |
| `logistic_regression_corrected.joblib` | Corrected multinomial Logistic Regression baseline ($C=10$, 19 features) | 3,233 | `baa474e4a8de62b1beb8ae7af8d18f17b7edf23dda574f05189dda7b8f0522f4` |
| `round3_step2_step3_results.joblib` | Soft-voting ensemble weight and threshold search CV results | 434 | `80308fe2785221c053a837446e5c99eece67f7bd27a578ba1da47889dd5083d9` |

---

## 2. Serialized Train/Test Split Dictionaries

| Artifact Filename | Description | Size (bytes) | SHA256 Checksum |
| :--- | :--- | :---: | :--- |
| `train_test_data_v2_gene_grouped.joblib` | Gene-grouped split dictionary (`GroupShuffleSplit`, `random_state=42`, $N_{\text{train}}=6,894$, $N_{\text{test}}=1,799$, 0 gene overlap) | 2,129,464 | `69980753ed88b31c28d5632be0c44642df33380ec064a3b52227f24108e0a5bf` |
| `train_test_data_v2.joblib` | Random stratified split dictionary for V2 23-feature dataset ($N_{\text{train}}=6,954$, $N_{\text{test}}=1,739$) | 1,832,292 | `deb032bb939ae60ec95b02a5a7cc4146104e5bc057255f0351529010aa2266cc` |
| `train_test_data_corrected.joblib` | Random stratified split dictionary for corrected 19-feature dataset ($N_{\text{train}}=6,954$, $N_{\text{test}}=1,739$) | 1,552,978 | `f3f656fe457fe804c0620cd422fc256572fbf548c389e2d333da9dde71380d51` |
| `train_test_data_v3_maxent.joblib` | Random stratified split dictionary for MaxEntScan dataset ($N_{\text{train}}=6,954$, $N_{\text{test}}=1,739$) | 1,832,280 | `114cd1ea9fd4682b2ff8f01221c075ef02ba8e391322a5c27c2a3a381ac01677` |
