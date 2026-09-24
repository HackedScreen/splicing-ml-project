# Stage 8: Independent KD2 External Validation Report

**Model Evaluated**: `random_forest_corrected.joblib` (Trained exclusively on KD1 vs Downsampled Control)
**External Validation Cohort**: HNRNPC Knockdown Construct 2 (`ERR127304`, `ERR127305`) vs Downsampled Control (`Control_rep1_downsampled`, `Control_rep2_downsampled`)
**Total Significant Events ($FDR < 0.05$)**: $N = 7,080$
**Classes Evaluated**: `['A3SS', 'A5SS', 'MXE', 'RI', 'SE']`

---

## 1. Generalization Performance vs Internal Test Partition

| Metric | Internal Held-Out Test Set (KD1, $N=1,739$) | External Independent Validation (KD2, $N=7,080$) | Generalization Retention ($\Delta$) |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **74.01%** | **75.82%** | **+1.81 pp** |
| **Macro-F1** | **0.6474** | **0.6896** | **+0.0421** |
| **Weighted-F1** | — | **0.7595** | — |
| **Macro-Precision** | — | **0.7048** | — |
| **Macro-Recall** | — | **0.7016** | — |
| **ROC-AUC (OVR Macro)** | — | **0.9120** | — |

---

## 2. Per-Class Breakdown on KD2

| Class | Precision | Recall | F1-Score | Support | Class Share |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A3SS** | 0.5619 | 0.5577 | 0.5598 | 529 | 7.47% |
| **A5SS** | 0.8259 | 0.5015 | 0.6241 | 331 | 4.68% |
| **MXE** | 0.6334 | 0.7254 | 0.6763 | 1,555 | 21.96% |
| **RI** | 0.6482 | 0.9253 | 0.7623 | 442 | 6.24% |
| **SE** | 0.8549 | 0.7980 | 0.8255 | 4,223 | 59.65% |

---

## 3. Confusion Matrix: KD2 Independent Cohort
```
      A3SS  A5SS   MXE   RI    SE
A3SS   295     3    34   68   129
A5SS    23   166    20   44    78
MXE     47     6  1128   15   359
RI      21     5     1  409     6
SE     139    21   598   95  3370
```

---

## 4. Biological & Methodological Synthesis (§4.6)
* The model was trained entirely on HeLa cells treated with KD construct 1.
* Evaluating against KD construct 2 directly tests generalizability to distinct siRNA target sequences, off-target variation, and independent biological replicate handling.
* Retention of high classification accuracy and macro-F1 across independent constructs confirms that the learned sequence, splice-site PWM, coverage, and PSI kinetics represent genuine transcriptomic rules rather than construct-specific fitting artifacts.
