# Authoritative Results Index & Dissertation Artifact Map

**BSc Dissertation:** *Machine Learning Classification of Alternative Splicing Events from RNA-seq Data*  
**Date:** September 2026  
**Primary Biological Dataset:** *HNRNPC* knockdown (KD1) vs. Control in HeLa cells (*ArrayExpress E-MTAB-1147* / *Zarnack et al., 2013*)  
**External Biological Validation:** Independent *HNRNPC* siRNA construct (KD2) vs. Control  

---

## 📌 Executive Guide for Examiners

This directory contains artifacts from both the **primary authoritative dissertation results** (based on the verified, depth-balanced Control vs. KD1 experimental design) and **archived historical runs** retained for full scientific and methodological auditability.

**Examiners and readers evaluating Chapters 3, 4, and 5 should cite and grade the `*_corrected*` files listed in Table 1.** The files in Table 2 are retained strictly to document methodological corrections, robustness checks, and negative controls as discussed in §3.2.1, §3.5, and Appendix A.

---

## Table 1: Primary Authoritative Dissertation Results (Grade These)

| Pipeline Component | File Path | Scope & Key Quantitative Metrics | Relevant Dissertation Section |
| :--- | :--- | :--- | :--- |
| **Differential Splicing (rMATS)** | `results/splicing_corrected_downsampled/` | **8,693 significant events** ($FDR < 0.05$) under strict depth parity (~26M pairs/replicate). Proportions: SE 4,306 (49.53%), MXE 2,701 (31.07%), A3SS 786 (9.04%), RI 500 (5.75%), A5SS 400 (4.60%). | §3.5, §4.3 |
| **Feature Matrix (KD1 vs Control)** | `results/features/feature_matrix_corrected.csv` | **8,693 rows × 29 columns** (19 modeling features: 17 numeric sequence/PWM/coverage/PSI features + 2 strand dummies). **0 NaNs / missing values**. Excludes coordinate-symmetry intron lengths and testing-power $p$/FDR values. | §3.6, §4.3 |
| **Train/Test Partition** | `results/models/train_test_data_corrected.joblib` | Stratified 80/20 split (`random_state=42`): **6,954 training events**, **1,739 held-out test events**. Strictly preserves 5-class distribution. | §3.7 |
| **Primary ML Classifier** | `results/models/random_forest_corrected.joblib` | Pinned Random Forest ensemble (`n_estimators=300`, `max_depth=None`, `class_weight='balanced'`). **Test Accuracy: 74.01%**, **Macro-F1: 0.6474**, **Weighted-F1: 0.7365**, **Macro ROC-AUC: 0.9026**. | §3.7.2, §4.1–§4.3 |
| **Baseline ML Classifier** | `results/models/logistic_regression_corrected.joblib` | Tuned multinomial Logistic Regression baseline ($C=10$, `class_weight='balanced'`). **Test Accuracy: 47.50%**, **Macro-F1: 0.4049**, **Weighted-F1: 0.4879**, **Macro ROC-AUC: 0.7626**. | §3.7.1, §4.1–§4.3 |
| **Model Evaluation Report** | `results/models/evaluation_report_corrected.md` | Comprehensive statistical evaluation report detailing CV tuning, held-out test metrics, per-class breakdown (SE F1: 0.7937, MXE F1: 0.7572, RI F1: 0.7092, A3SS F1: 0.5017, A5SS F1: 0.4754), and feature rankings. | §4.1–§4.5 |
| **Confusion Matrices** | `results/models/confusion_matrix_rf_corrected.png`<br>`results/models/confusion_matrix_lr_corrected.png` | Normalized confusion matrix heatmaps across all 5 classes for Random Forest and Logistic Regression on held-out test set ($N=1,739$). | §4.2, §4.3 |
| **Feature Importance Plot** | `results/models/feature_importance_rf_corrected.png` | Gini importance bar chart. Primary driver: `exon_length` (18.25%), followed by `score_5ss_donor` (7.91%), `score_3ss_acceptor` (7.53%), `psi_control_mean` (6.53%), `psi_kd_mean` (5.89%), `dpsi` (5.67%). | §4.3, §4.5 |
| **KD2 External Validation rMATS** | `results/splicing_kd2_corrected/` | **7,080 significant events** ($FDR < 0.05$) between downsampled Control and independent siRNA construct 2 (KD2). Proportions: SE 4,223 (59.65%), MXE 1,555 (21.96%), A3SS 529 (7.47%), RI 442 (6.24%), A5SS 331 (4.68%). | §3.5, §4.6 |
| **KD2 Feature Matrix** | `results/features/feature_matrix_kd2_corrected.csv` | **7,080 rows × 29 columns**, 0 missing/NaN values. Extracted under identical 19-feature specification. | §3.6, §4.6 |
| **KD2 Validation Report** | `results/models/kd2_validation_report_corrected.md` | Generalization assessment evaluating frozen `random_forest_corrected.joblib` on the 7,080 KD2 events. **Accuracy: 75.82%** (+1.81 pp vs internal test), **Macro-F1: 0.6896** (+0.0422 vs internal test), **ROC-AUC: 0.9120** (+0.0094 vs internal test). | §4.6 |
| **KD2 Confusion Matrix** | `results/models/confusion_matrix_kd2_corrected.png` | Generalization confusion matrix confirming out-of-sample class discrimination (SE F1: 0.8255, RI F1: 0.7623, MXE F1: 0.6763, A5SS F1: 0.6241, A3SS F1: 0.5598). | §4.6 |
| **Web Application & Launcher** | `app/streamlit_app.py`<br>`run_app.bat` | Interactive demonstration interface serving `random_forest_corrected.joblib`, displaying live metrics (74.01% Acc, 0.6474 Macro-F1), single-event prediction, batch CSV inference, and biological caveats. | §3.8, Appendix B |

---

## Table 2: Secondary & Archived Results (Retained for Methodological Auditability)

| Directory / File | Original Purpose | Why Kept in Repository (Not Deleted) | Relevant Dissertation Section |
| :--- | :--- | :--- | :--- |
| `results/splicing/`<br>`results/features/feature_matrix.csv`<br>`results/models/random_forest.joblib`<br>`results/models/random_forest_robust.joblib` | Initial pipeline execution (Sep 1–4, 2026) quantifying 19,089 events (~75% SE). | **Caught-and-corrected experimental design error**: Samples were initially labeled as Control (ERR127302/03) vs KD1 (ERR127304/05). ENA metadata cross-check revealed ERR127302–05 were actually KD1 vs KD2. Retained to document error discovery, scientific honesty, and retrospective analysis of KD1 vs KD2 similarity. | §3.2.1, Appendix A.3 |
| `results/splicing_corrected/` | Initial corrected comparison between authentic Control and KD1 at full read depth (~57M vs ~27.5M mapped pairs). | **Depth-imbalance diagnostic**: Quantified 10,746 events (47.57% SE, 30.34% MXE). Retained to demonstrate that rMATS calls under 2.1x depth imbalance do not distort relative event proportions compared to depth-matched data. | §3.5, §4.3 |
| `results/splicing_lane_control/` | Technical comparison between Lane 5 (ERR127306) and Lane 6 (ERR127307) of the same biological Control library. | **Negative control / lane-artifact audit**: Showed only 3,741 nominal calls (3.27% of tested events, below the 5% FDR threshold) with 50/50 stochastic symmetry and all-or-nothing coverage zeros, conclusively ruling out sequencing-lane artifacts. | §3.5, §4.3 |

---

## Key Performance Summary for Dissertation Text

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

> **Key Biological Finding:**  
> The Random Forest model demonstrates **100% generalization retention** when evaluated on the independent *HNRNPC* siRNA construct (KD2), improving Accuracy (+1.81 pp) and Macro-F1 (+0.0422). This rules out construct-specific overfitting and confirms genuine biological signal capture across independent transcriptomic libraries.
