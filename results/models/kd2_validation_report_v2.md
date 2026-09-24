# Stage 8 V2: Independent KD2 External Validation Report (Round 2 Improvements)

**Models Evaluated**: `xgboost_v2.joblib` and `random_forest_v2.joblib` (Trained exclusively on KD1 with 23 features)
**External Validation Cohort**: HNRNPC Knockdown Construct 2 (`ERR127304`, `ERR127305`) vs Downsampled Control (`Control_rep1_downsampled`, `Control_rep2_downsampled`)
**Total Significant Events ($FDR < 0.05$)**: $N = 7,080$
**Classes Evaluated**: `['A3SS', 'A5SS', 'MXE', 'RI', 'SE']`

---

## 1. Generalization Performance Comparison (KD2 External Validation)

| Metric | RF Baseline (19 feats) | RF V2 (23 feats) | XGBoost V2 (23 feats) | $\Delta$ vs RF Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | **75.82%** | **76.82%** | **73.90%** | **-1.92 pp** |
| **Macro-F1** | **0.6896** | **0.7003** | **0.6655** | **-0.0241** |
| **Weighted-F1** | **0.7595** | **0.7698** | **0.7434** | **-0.0161** |
| **Macro-Precision** | **0.7048** | **0.7225** | **0.6463** | — |
| **Macro-Recall** | **0.7016** | **0.7089** | **0.6917** | — |
| **ROC-AUC (OVR Macro)** | **0.9120** | **0.9180** | **0.9078** | — |

---

## 2. Per-Class Breakdown on Independent KD2 Dataset

### XGBoost V2 on KD2
| Class | Precision | Recall | F1-Score | Support | Class Share |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A3SS** | 0.5091 | 0.5841 | 0.5440 | 529 | 7.47% |
| **A5SS** | 0.5489 | 0.5257 | 0.5370 | 331 | 4.68% |
| **MXE** | 0.6050 | 0.7061 | 0.6516 | 1,555 | 21.96% |
| **RI** | 0.7085 | 0.8688 | 0.7805 | 442 | 6.24% |
| **SE** | 0.8600 | 0.7736 | 0.8145 | 4,223 | 59.65% |

### Random Forest V2 on KD2
| Class | Precision | Recall | F1-Score | Support | Class Share |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A3SS** | 0.5709 | 0.5784 | 0.5746 | 529 | 7.47% |
| **A5SS** | 0.8670 | 0.4924 | 0.6281 | 331 | 4.68% |
| **MXE** | 0.6381 | 0.7505 | 0.6897 | 1,555 | 21.96% |
| **RI** | 0.6700 | 0.9186 | 0.7748 | 442 | 6.24% |
| **SE** | 0.8664 | 0.8044 | 0.8342 | 4,223 | 59.65% |

---

## 3. Confusion Matrix: KD2 Independent Cohort

### XGBoost V2 on KD2
```
      A3SS  A5SS   MXE   RI    SE
A3SS   309    22    42   56   100
A5SS    22   174    32   37    66
MXE     55    27  1098   11   364
RI      35    19     2  384     2
SE     186    75   641   54  3267
```

### Random Forest V2 on KD2
```
      A3SS  A5SS   MXE   RI    SE
A3SS   306     2    43   61   117
A5SS    24   163    24   45    75
MXE     46     3  1167   14   325
RI      23     3     3  406     7
SE     137    17   592   80  3397
```

---

## 4. Generalization Retention (Internal KD1 vs External KD2)

| Model | Internal Test Acc | KD2 External Acc | Retention $\Delta$ Acc | Internal Test Macro-F1 | KD2 External Macro-F1 | Retention $\Delta$ Macro-F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RF Baseline (19 feats)** | 74.01% | 75.82% | +1.81 pp | 0.6474 | 0.6896 | +0.0422 |
| **RF V2 (23 feats)** | 74.81% | 76.82% | +2.01 pp | 0.6611 | 0.7003 | +0.0392 |
| **XGBoost V2 (23 feats)** | 72.17% | 73.90% | +1.73 pp | 0.6385 | 0.6655 | +0.0270 |
