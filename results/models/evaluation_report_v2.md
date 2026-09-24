# Stages 6 & 7 V2: Model Evaluation & Diagnostic Report (Round 2 Improvements)

**Evaluation Partition**: Internal Held-Out Test Set ($N = 1,739$, 20% Stratified Split, `random_state=42`)
**Underlying Comparison**: Depth-matched Control vs KD1 (`results/splicing_corrected_downsampled/`)
**Target Classes**: `['A3SS', 'A5SS', 'MXE', 'RI', 'SE']`
**Feature Set**: 23 Features (17 numeric sequence/PWM/coverage/PSI + 4 new HNRNPC U-tract motif features + 2 strand dummies)

---

## 1. Overall Performance Comparison on Internal KD1 Test Set

| Metric | Target | RF Baseline (19 feats) | RF V2 (23 feats) | XGBoost V2 (23 feats) | Status vs Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | $\ge 80.0\%$ | **74.01%** | **74.81%** | **72.17%** | **-1.84%** |
| **Macro-F1** | $\ge 0.780$ | **0.6474** | **0.6611** | **0.6385** | **-0.0089** |
| **Weighted-F1** | — | **0.7365** | **0.7450** | **0.7215** | **-0.0150** |
| **Macro-Precision** | — | **0.7180** | **0.6876** | **0.6290** | — |
| **Macro-Recall** | — | **0.6083** | **0.6694** | **0.6568** | — |
| **ROC-AUC (OVR Macro)** | — | **0.9026** | **0.9077** | **0.9058** | — |

---

## 2. Per-Class Performance Breakdown (Internal KD1 Test Set)

### XGBoost V2 (Best params = `{'learning_rate': 0.05, 'max_depth': 6, 'n_estimators': 300}`)
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **A3SS** | 0.4724 | 0.4904 | 0.4813 | 157 |
| **A5SS** | 0.5000 | 0.4125 | 0.4521 | 80 |
| **MXE** | 0.7193 | 0.7593 | 0.7387 | 540 |
| **RI** | 0.6493 | 0.8700 | 0.7436 | 100 |
| **SE** | 0.8040 | 0.7517 | 0.7770 | 862 |

### Random Forest V2 (Best params = `{'max_depth': None, 'n_estimators': 300}`)
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **A3SS** | 0.5379 | 0.4968 | 0.5166 | 157 |
| **A5SS** | 0.7317 | 0.3750 | 0.4959 | 80 |
| **MXE** | 0.7426 | 0.7907 | 0.7659 | 540 |
| **RI** | 0.6122 | 0.9000 | 0.7287 | 100 |
| **SE** | 0.8135 | 0.7842 | 0.7986 | 862 |

---

## 3. Confusion Matrices (Internal KD1 Test Set)

### XGBoost V2
```
      A3SS  A5SS  MXE  RI   SE
A3SS    77     8   13  21   38
A5SS    14    33    6  12   15
MXE     18     3  410   4  105
RI       8     5    0  87    0
SE      46    17  141  10  648
```

### Random Forest V2
```
      A3SS  A5SS  MXE  RI   SE
A3SS    78     3   15  21   40
A5SS    16    30    5  12   17
MXE      9     1  427   5   98
RI       8     2    0  90    0
SE      34     5  128  19  676
```

---

## 4. Feature Importance Rankings (23 Features)

### XGBoost V2 Feature Importances
| Rank | Feature Name | Importance (Gain) | Category |
| :--- | :--- | :--- | :--- |
| 1 | `exon_length` | 0.1612 | Original Feature |
| 2 | `score_5ss_donor` | 0.0651 | Original Feature |
| 3 | `score_3ss_acceptor` | 0.0643 | Original Feature |
| 4 | `psi_kd_mean` | 0.0631 | Original Feature |
| 5 | `max_utract_length_upstream` | 0.0504 | **New U-Tract Motif** |
| 6 | `dpsi` | 0.0502 | Original Feature |
| 7 | `psi_control_mean` | 0.0500 | Original Feature |
| 8 | `cov_mean_kd` | 0.0454 | Original Feature |
| 9 | `cov_mean_overall` | 0.0449 | Original Feature |
| 10 | `cov_mean_control` | 0.0391 | Original Feature |
| 11 | `cov_var_control` | 0.0380 | Original Feature |
| 12 | `cov_var_overall` | 0.0372 | Original Feature |
| 13 | `max_utract_length_downstream` | 0.0364 | **New U-Tract Motif** |
| 14 | `psi_control_var` | 0.0328 | Original Feature |
| 15 | `cov_var_kd` | 0.0318 | Original Feature |
| 16 | `gc_exon` | 0.0316 | Original Feature |
| 17 | `psi_kd_var` | 0.0301 | Original Feature |
| 18 | `u_fraction_downstream` | 0.0299 | **New U-Tract Motif** |
| 19 | `gc_downstream_100bp` | 0.0294 | Original Feature |
| 20 | `u_fraction_upstream` | 0.0259 | **New U-Tract Motif** |
| 21 | `gc_upstream_100bp` | 0.0259 | Original Feature |
| 22 | `strand_+` | 0.0171 | Original Feature |
| 23 | `strand_-` | 0.0000 | Original Feature |
