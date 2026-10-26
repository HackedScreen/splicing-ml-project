# Stages 6 & 7: Corrected Model Evaluation & Diagnostic Report

**Evaluation Dataset**: Corrected Held-out Test Partition ($N = 1,739$, 20% Stratified Split, `random_state=42`)
**Underlying Comparison**: Depth-matched Control vs KD1 (`results/splicing_corrected_downsampled/`)
**Target Classes**: `['A3SS', 'A5SS', 'MXE', 'RI', 'SE']`
**Feature Standard**: 19 Features (17 numeric + 2 strand dummies; intron lengths & pvalue/fdr excluded)

---

## 1. Overall Model Performance vs Dissertation Target

* **Dissertation Abstract Target**: Accuracy $\ge 80.0\%$, Macro-F1 $\ge 0.780$.

| Metric | Target | Logistic Regression (Baseline) | Random Forest (Ensemble) |
| :--- | :--- | :--- | :--- |
| **Accuracy** | $\ge 80.0\%$ | **47.50%** (FAIL) | **74.01%** (FAIL) |
| **Macro-F1** | $\ge 0.780$ | **0.4049** (FAIL) | **0.6474** (FAIL) |
| **Weighted-F1** | — | **0.4806** | **0.7365** |
| **Macro-Precision** | — | **0.3941** | **0.6719** |
| **Macro-Recall** | — | **0.4397** | **0.6567** |
| **ROC-AUC (OVR Macro)**| — | **0.7626** | **0.9026** |
| **Target Status** | — | **NOT MET** | **NOT MET** |

---

## 2. Per-Class Performance Breakdown

### Logistic Regression (Best C = 10)
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **A3SS** | 0.2789 | 0.2611 | 0.2697 | 157 |
| **A5SS** | 0.1046 | 0.2000 | 0.1373 | 80 |
| **MXE** | 0.4645 | 0.6056 | 0.5257 | 540 |
| **RI** | 0.4965 | 0.7000 | 0.5809 | 100 |
| **SE** | 0.6263 | 0.4316 | 0.5110 | 862 |

### Random Forest (Best params = {'max_depth': None, 'n_estimators': 300})
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **A3SS** | 0.5362 | 0.4713 | 0.5017 | 157 |
| **A5SS** | 0.6905 | 0.3625 | 0.4754 | 80 |
| **MXE** | 0.7411 | 0.7741 | 0.7572 | 540 |
| **RI** | 0.5894 | 0.8900 | 0.7092 | 100 |
| **SE** | 0.8021 | 0.7854 | 0.7937 | 862 |

---

## 3. Confusion Matrices

### Logistic Regression
```
      A3SS  A5SS  MXE  RI   SE
A3SS    41    28   25  25   38
A5SS    24    16   10  10   20
MXE     17    21  327  11  164
RI      20    10    0  70    0
SE      45    78  342  25  372
```

### Random Forest
```
      A3SS  A5SS  MXE  RI   SE
A3SS    74     4   13  22   44
A5SS    16    29    5  11   19
MXE     11     0  418   7  104
RI      10     1    0  89    0
SE      27     8  128  22  677
```

---

## 4. Random Forest Feature Importances (All 19 Features)

| Rank | Feature Name | Gini Importance | Cumulative Importance | Flag |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `exon_length` | 0.1825 | 0.1825 | **>15% DOMINANT** |
| 2 | `score_5ss_donor` | 0.0791 | 0.2616 | Normal |
| 3 | `score_3ss_acceptor` | 0.0753 | 0.3369 | Normal |
| 4 | `psi_control_mean` | 0.0653 | 0.4022 | Normal |
| 5 | `psi_kd_mean` | 0.0589 | 0.4611 | Normal |
| 6 | `dpsi` | 0.0567 | 0.5178 | Normal |
| 7 | `cov_mean_control` | 0.0509 | 0.5687 | Normal |
| 8 | `cov_mean_overall` | 0.0474 | 0.6162 | Normal |
| 9 | `gc_exon` | 0.0466 | 0.6628 | Normal |
| 10 | `cov_mean_kd` | 0.0455 | 0.7082 | Normal |
| 11 | `gc_downstream_100bp` | 0.0420 | 0.7503 | Normal |
| 12 | `cov_var_control` | 0.0409 | 0.7912 | Normal |
| 13 | `cov_var_kd` | 0.0406 | 0.8318 | Normal |
| 14 | `cov_var_overall` | 0.0403 | 0.8721 | Normal |
| 15 | `psi_kd_var` | 0.0399 | 0.9121 | Normal |
| 16 | `gc_upstream_100bp` | 0.0397 | 0.9518 | Normal |
| 17 | `psi_control_var` | 0.0362 | 0.9880 | Normal |
| 18 | `strand_+` | 0.0062 | 0.9942 | Normal |
| 19 | `strand_-` | 0.0058 | 1.0000 | Normal |
