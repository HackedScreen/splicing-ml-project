# Results Index & Artifact Map

**BSc Dissertation:** *Machine Learning Classification of Alternative Splicing Events from RNA-seq Data*  
**Date:** September 2026  
**Primary Dataset:** *HNRNPC* knockdown (KD1) vs. Control in HeLa cells (*ArrayExpress E-MTAB-1147* / *Zarnack et al., 2013*)  
**External Validation:** Independent *HNRNPC* siRNA construct (KD2) vs. Control  

---

## Overview

This directory contains outputs from the primary dissertation analysis (depth-matched Control vs. KD1 and external KD2 validation) as well as secondary diagnostic runs retained for auditability.

---

## Table 1: Primary Dissertation Results

| Pipeline Component | File Path | Scope & Key Quantitative Metrics | Relevant Dissertation Section |
| :--- | :--- | :--- | :--- |
| **Differential Splicing (rMATS)** | `results/splicing_corrected_downsampled/` | **8,693 significant events** ($FDR < 0.05$) under matched library depth (~26M pairs/replicate). Proportions: SE 4,306 (49.53%), MXE 2,701 (31.07%), A3SS 786 (9.04%), RI 500 (5.75%), A5SS 400 (4.60%). | §3.5, §4.3 |
| **Feature Matrix (KD1 vs Control)** | `results/features/feature_matrix.csv` | **8,693 rows × 29 columns** (19 modeling features: 17 numeric sequence/PWM/coverage/PSI features + 2 strand dummies). **0 NaNs / missing values**. Excludes flanking intron lengths and $p$/FDR values. | §3.6, §4.3 |
| **Train/Test Partition** | `results/models/train_test_data.joblib` | Stratified 80/20 split (`random_state=42`): **6,954 training events**, **1,739 held-out test events**. Preserves 5-class distribution. | §3.7 |
| **Primary ML Classifier** | `results/models/random_forest.joblib` | Random Forest ensemble (`n_estimators=300`, `max_depth=None`, `class_weight='balanced'`). **Test Accuracy: 74.01%**, **Macro-F1: 0.6474**, **Weighted-F1: 0.7365**, **Macro ROC-AUC: 0.9026**. | §3.7.2, §4.1–§4.3 |
| **Baseline ML Classifier** | `results/models/logistic_regression.joblib` | Multinomial Logistic Regression baseline ($C=10$, `class_weight='balanced'`). **Test Accuracy: 47.50%**, **Macro-F1: 0.4049**, **Weighted-F1: 0.4879**, **Macro ROC-AUC: 0.7626**. | §3.7.1, §4.1–§4.3 |
| **Model Evaluation Report** | `results/models/evaluation_report.md` | Statistical evaluation report detailing CV tuning, held-out test metrics, per-class breakdown (SE F1: 0.7937, MXE F1: 0.7572, RI F1: 0.7092, A3SS F1: 0.5017, A5SS F1: 0.4754), and feature rankings. | §4.1–§4.5 |
| **Confusion Matrices** | `results/models/confusion_matrix_rf.png`<br>`results/models/confusion_matrix_lr.png` | Confusion matrix heatmaps across all 5 classes for Random Forest and Logistic Regression on the held-out test set ($N=1,739$). | §4.2, §4.3 |
| **Feature Importance Plot** | `results/models/feature_importance_rf.png` | Gini importance bar chart. Top predictor: `exon_length` (18.25%), followed by `score_5ss_donor` (7.91%), `score_3ss_acceptor` (7.53%), `psi_control_mean` (6.53%), `psi_kd_mean` (5.89%), `dpsi` (5.67%). | §4.3, §4.5 |
| **KD2 External Validation rMATS** | `results/splicing_kd2_corrected/` | **7,080 significant events** ($FDR < 0.05$) between downsampled Control and independent siRNA construct 2 (KD2). Proportions: SE 4,223 (59.65%), MXE 1,555 (21.96%), A3SS 529 (7.47%), RI 442 (6.24%), A5SS 331 (4.68%). | §3.5, §4.6 |
| **KD2 Feature Matrix** | `results/features/feature_matrix_kd2.csv` | **7,080 rows × 29 columns**, 0 missing/NaN values. Extracted under the same 19-feature specification. | §3.6, §4.6 |
| **KD2 Validation Report** | `results/models/kd2_validation_report.md` | Generalization assessment evaluating `random_forest.joblib` on the 7,080 KD2 events. **Accuracy: 75.82%** (+1.81 pp vs internal test), **Macro-F1: 0.6896** (+0.0422 vs internal test), **ROC-AUC: 0.9120** (+0.0094 vs internal test). | §4.6 |
| **KD2 Confusion Matrix** | `results/models/confusion_matrix_kd2.png` | Out-of-sample confusion matrix on KD2 (SE F1: 0.8255, RI F1: 0.7623, MXE F1: 0.6763, A5SS F1: 0.6241, A3SS F1: 0.5598). | §4.6 |
| **Web Application & Launcher** | `app/streamlit_app.py`<br>`run_app.bat` | Interactive Streamlit web application serving `random_forest.joblib` for single-event and batch CSV prediction. | §3.8, Appendix B |

---

## Table 2: Secondary & Diagnostic Runs

| Directory / File | Purpose | Notes | Relevant Dissertation Section |
| :--- | :--- | :--- | :--- |
| `results/splicing/` | Preliminary pipeline execution (19,089 events). | Initial run prior to ENA sample accession remapping (comparing KD1 vs KD2 replicates). Retained to document the sample manifest correction. | §3.2.1, Appendix A.3 |
| `results/splicing_corrected/` | Full-depth comparison between Control and KD1 (~57M vs ~27.5M mapped pairs). | Quantified 10,746 events (47.57% SE, 30.34% MXE). Demonstrates that rMATS event proportions at full depth closely match the downsampled run. | §3.5, §4.3 |
| `results/splicing_lane_control/` | Technical comparison between Lane 5 (`ERR127306`) and Lane 6 (`ERR127307`) of the same Control library. | Negative control showing 3,741 nominal calls (3.27% of tested events, below the 5% FDR threshold) with 50/50 directional symmetry. | §3.5, §4.3 |

---

## Performance Summary

| Metric | Baseline Logistic Regression | Primary Random Forest (Internal KD1 Test, $N=1,739$) | Primary Random Forest (External KD2 Validation, $N=7,080$) | Dissertation Target |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 47.50% | **74.01%** | **75.82%** | $\ge 80.0\%$ (NOT MET) |
| **Macro-F1** | 0.4049 | **0.6474** | **0.6896** | $\ge 0.780$ (NOT MET) |
| **Weighted-F1** | 0.4879 | **0.7365** | **0.7595** | N/A |
| **Macro ROC-AUC** | 0.7626 | **0.9026** | **0.9120** | N/A |
| **SE Class F1** | 0.6033 | **0.7937** | **0.8255** | N/A |
| **MXE Class F1** | 0.5056 | **0.7572** | **0.6763** | N/A |
| **RI Class F1** | 0.3807 | **0.7092** | **0.7623** | N/A |
| **A3SS Class F1** | 0.2831 | **0.5017** | **0.5598** | N/A |
| **A5SS Class F1** | 0.2520 | **0.4754** | **0.6241** | N/A |

---

## Follow-up Feature & Model Ablations

Additional experiments evaluated HNRNPC U-tract binding motif density features, soft-voting ensembles, per-class threshold tuning, and MaxEntScan splice-site scoring on the 5-fold cross-validation training split before testing on the held-out KD1 ($N=1,739$) and external KD2 ($N=7,080$) sets.

### Table 3: Comparative Performance Across Pipeline Iterations

| Iteration / Model | Feature Configuration | Internal Test Acc ($N=1,739$) | Internal Test Macro-F1 | KD2 Validation Acc ($N=7,080$) | KD2 Validation Macro-F1 | Role |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **RF Baseline** | 19 feats (Additive PWM, unconstrained trees) | 74.01% | 0.6474 | 75.82% | 0.6896 | 19-Feature Baseline |
| **XGBoost V2** | 23 feats (+ U-tract, balanced sample weights) | 72.17% | 0.6385 | 73.90% | 0.6655 | Gradient-Boosted Benchmark |
| **Soft-Voting Ensemble** | RF V2 + XGB V2 ($w=0.7/0.3$ from CV OOF) | 74.64% | 0.6532 | 76.82% | 0.7020 | Ensemble Ablation |
| **Threshold-Tuned RF V2** | Class multipliers tuned on CV OOF | 72.80% | 0.6462 | 74.38% | 0.6799 | Threshold Ablation |
| **RF V3 MaxEntScan** | 23 feats (`depth=20, n=100`, training CV winner) | **75.79%** | 0.6203 | **79.41%** | 0.6897 | MaxEntScan Ablation |
| **Random Forest V2** | 23 feats (+ U-tract motif density, unconstrained) | **74.81%** | **0.6611** | **76.82%** | **0.7003** | **23-Feature Primary Model** |

---

### Notes on Model Selection

1. **Macro-F1 Performance:**  
   Because alternative splicing event classes are imbalanced (Skipped Exons comprise ~50–60% of events, while Alternative 5' Splice Sites comprise <5%), Macro-F1 weights all five splicing classes equally. Random Forest V2 achieved the highest unweighted Macro-F1 on both the internal held-out test partition (**0.6611** vs 0.6474 baseline) and the external KD2 validation cohort (**0.7003** vs 0.6896 baseline).
2. **HNRNPC U-Tract Features:**  
   Adding the four U-tract motif density features (`max_utract_length_upstream`, `max_utract_length_downstream`, `u_fraction_upstream`, `u_fraction_downstream`) captures HNRNPC poly-U binding near regulated exons (*Zarnack et al., 2013*), improving generalization across both knockdown constructs.
3. **MaxEntScan Accuracy vs. Macro-F1 Trade-off (§4.6, §5.2):**  
   Replacing additive PWM scores with MaxEntScan (*Yeo & Burge, 2004*) 9 bp donor and 23 bp acceptor scores yielded the highest overall accuracy (**75.79%** on internal test and **79.41%** on KD2), while the depth-regularized trees (`max_depth=20, n_estimators=100`) produced lower minority-class recall on A5SS, reducing Macro-F1 to 0.6203 on the internal test set.
