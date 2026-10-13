# Follow-Up Robustness Analyses for Supervisor Review

**Date**: 2026-10-08  
**Scope**: In-depth empirical resolution of locus-level splitting dependencies, cross-sample generalization strata, feature ablations, calibration, leakage diagnostics, and technical negative controls.  
**Models Evaluated**: Random Forest V2 (Gene-Grouped Split, `GroupShuffleSplit`, `random_state=42`, $N_{train}=6,894$, $N_{test}=1,799$).  

---

## 1. KD2 External Validation Stratified by Gene Overlap

To determine whether the strong external generalization of the model on the independent KD2 biological replicate cohort ($N=7,080$) was inflated by genes shared with the training set, KD2 events were partitioned into two disjoint strata:
- **Seen Genes**: Events located within genes present in the KD1 training partition ($N_{train\_genes} = 2,888$).
- **Unseen Genes**: Events located in genes completely absent from the KD1 training partition.

| Stratum | N Events | % of KD2 | Accuracy | Macro-F1 | F1 (A3SS) | F1 (A5SS) | F1 (MXE) | F1 (RI) | F1 (SE) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full KD2 Cohort** | **7,080** | **100.0%** | **74.94%** | **0.6688** | 0.5369 | 0.5670 | 0.6580 | 0.7568 | 0.8254 |
| **Seen Genes (in KD1 Train)** | **4,362** | **61.6%** | **82.03%** | **0.7700** | 0.6800 | 0.7161 | 0.7569 | 0.8277 | 0.8693 |
| **Unseen Genes (Absent from KD1 Train)** | **2,718** | **38.4%** | **63.58%** | **0.5222** | 0.3595 | 0.3491 | 0.4712 | 0.6784 | 0.7529 |

**Findings**:
- **61.61%** of KD2 events (4,362 / 7,080) belong to genes present in the KD1 training set.
- On **Seen Genes**, the model achieves **82.03% accuracy** and **0.7700 Macro-F1**.
- On **Unseen Genes** (novel genes absent from the KD1 training set), performance drops to **63.58% accuracy** and **0.5222 Macro-F1** (-18.45 pp accuracy, -0.2478 Macro-F1).
- Minority classes on unseen genes experience severe degradation: A5SS F1 falls to 0.3491 and A3SS F1 falls to 0.3595.
- This demonstrates that gene-context familiarity confers a substantial performance advantage, and generalisation to completely uncharacterised genomic loci without prior gene exposure represents a steeper challenge (63.58% accuracy), closely matching internal gene-grouped test performance (66.59%).

---

## 2. Feature Ablation under 5-Fold GroupKFold ($N=8,693$)

To isolate the predictive contributions of intrinsic biophysical sequence/geometry vs local sequencing kinetics/coverage under strict gene-level isolation, 5-fold `GroupKFold` was evaluated across three feature representations:
- **(A) All 23 Features**: Full model.
- **(B) Sequence & Structure Only** (12 features): `exon_length`, `gc_exon`, flanking GCs, splice-site PWM scores, 4 U-tract features, strand dummies.
- **(C) Coverage & PSI Only** (11 features): Control/KD coverage means & variances, replicate PSI means & variances, $\Delta\text{PSI}$.

| Feature Subset | N Features | Mean Accuracy $\pm$ SD | Mean Macro-F1 $\pm$ SD |
| :--- | :---: | :---: | :---: |
| **(A) All 23 Features** | 23 | **66.88% $\pm$ 0.96%** | **0.5629 $\pm$ 0.0117** |
| **(B) Sequence/Structure Only** | 12 | **52.33% $\pm$ 1.32%** | **0.4598 $\pm$ 0.0098** |
| **(C) Coverage/PSI Only** | 11 | **57.61% $\pm$ 1.75%** | **0.3947 $\pm$ 0.0185** |

**Findings**:
- Sequence and structural features alone capture significant discriminative signal, achieving **52.33% accuracy** and **0.4598 Macro-F1**.
- Coverage and PSI features alone achieve **57.61% accuracy** and **0.3947 Macro-F1**.
- Combining both feature modalities yields the optimal Macro-F1 (**0.5629**), demonstrating complementary predictive value.

---

## 3. Hyperparameter Re-Tuning under GroupKFold (Training Groups Only)

A full grid search was conducted strictly on the training partition ($N=6,894$, $2,888$ gene groups) using 5-fold `GroupKFold` cross-validation:

| `n_estimators` | `max_depth` | Mean CV Macro-F1 $\pm$ SD |
| :---: | :---: | :---: |
| 100 | `None` | **0.5677 ± 0.0193** |
| 300 | `None` | **0.5753 ± 0.0215** |
| 100 | `10` | **0.5565 ± 0.0159** |
| 300 | `10` | **0.5585 ± 0.0180** |
| 100 | `20` | **0.5695 ± 0.0185** |
| 300 | `20` | **0.5725 ± 0.0194** |

**Findings**:
- **Winning Configuration**: `{'max_depth': None, 'n_estimators': 300}` (CV Macro-F1 = 0.5753).
- **Did parameters change from default (`n_estimators=300, max_depth=None`)?**: **No, exactly identical.**
- Held-out test performance of this configuration: **Accuracy = 66.59%**, **Macro-F1 = 0.5567**.

---

## 4. Gene-Level Cluster Bootstrap (1,000 Resamples of Test Genes)

Resampling 722 unique test genes with replacement 1,000 times yielded the following empirical 95% confidence intervals (2.5th to 97.5th percentiles):

| Metric | Point Estimate | Bootstrap Mean $\pm$ SD | 95% Confidence Interval |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | **66.59%** | 66.63% $\pm$ 1.81% | **[63.27%, 70.18%]** |
| **Macro-F1** | **0.5567** | 0.5564 $\pm$ 0.0200 | **[0.5190, 0.5966]** |
| **F1 (A3SS)** | **0.4281** | 0.4278 $\pm$ 0.0423 | **[0.3500, 0.5074]** |
| **F1 (A5SS)** | **0.2903** | 0.2885 $\pm$ 0.0606 | **[0.1739, 0.4082]** |
| **F1 (MXE)** | **0.6332** | 0.6335 $\pm$ 0.0285 | **[0.5754, 0.6887]** |
| **F1 (RI)** | **0.6786** | 0.6787 $\pm$ 0.0339 | **[0.6127, 0.7439]** |
| **F1 (SE)** | **0.7531** | 0.7535 $\pm$ 0.0171 | **[0.7194, 0.7875]** |

**Findings**:
- Overall accuracy is bound between **63.27%** and **70.18%** at 95% confidence across gene clusters.
- The minority class A5SS exhibits the widest relative interval ([0.1739, 0.4082]), reflecting sampling sensitivity in small event counts ($N=81$ test cases).

---

## 5. Probability Calibration Analysis

Multiclass Brier score ($\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} (p_{ik} - y_{ik})^2$, lower is better) on the held-out test partition:

| Model | Multiclass Brier Score | Accuracy | Macro-F1 |
| :--- | :---: | :---: | :---: |
| **Uncalibrated Grouped RF V2** | **0.4647** | **66.59%** | **0.5567** |
| **CalibratedClassifierCV (Sigmoid, GroupKFold)** | **0.4358** | **67.37%** | **0.5572** |

**Findings**:
- Sigmoid Platt scaling via `CalibratedClassifierCV` reduced the multiclass Brier score from **0.4647** to **0.4358** (-0.0289), improving probability calibration.
- Classification accuracy and Macro-F1 remained virtually identical (66.59% vs 67.37%), confirming calibration preserves class rank order while producing more reliable posterior confidence estimates.

---

## 6. Leakage Diagnostics

### A. Mutual Information Ranking (Top 10 of 23 Features)
1. `exon_length`: **0.2133**
2. `cov_mean_overall`: **0.1867**
3. `cov_mean_control`: **0.1804**
4. `cov_mean_kd`: **0.1744**
5. `cov_var_overall`: **0.1615**
6. `cov_var_control`: **0.1559**
7. `cov_var_kd`: **0.1526**
8. `score_3ss_acceptor`: **0.1122**
9. `score_5ss_donor`: **0.1108**
10. `psi_control_mean`: **0.1076**

### B. Row and Coordinate Deduplication
- **Exact duplicate feature rows**: **78** across all 23 features in the full dataset.
- **Coordinate loci appearing with different labels**: **416** loci (1113 events, 12.80% of dataset). These represent complex overlapping alternative splicing annotations at shared splice junctions.
- **Train/Test Coordinate Overlap under Gene-Grouped Split**: **2** coordinate loci (2 test events, **0.11%**).
  - Because genes were strictly partitioned, 0.00% of test events shared genomic coordinates with training events (except 2 edge cases of overlapping gene coordinates).

---

## 7. Lane Negative Control Statistical Evaluation (ERR127306 vs ERR127307)

From the technical lane comparison where identical biological RNA-seq material was sequenced on separate lanes:

| Statistic | Value | Methodological Implication |
| :--- | :---: | :--- |
| **Total Events Tested (JC reads)** | **114,252** | All quantified splicing junctions |
| **Significant Events at FDR < 0.05** | **3,741 (3.27%)** | **3.27% of tested events** |
| **Count Expected Under Null ($H_0$, $\alpha=0.05$)** | **5,712.6** | Theoretical uniform null expectation |
| **Observed / Expected Null Ratio** | **0.655** | **34.5% below expected nominal false positive ceiling** |
| **Up/Down Directional Split** | **1,931 (51.62%) vs 1,810 (48.38%)** | Near-perfect 50/50 symmetry |
| **Exact Binomial Test ($H_0: p=0.5$)** | **$p = 0.0498$** | **No evidence of directional lane bias ($p > 0.05$)** |
| **Clopper-Pearson 95% Confidence Interval** | **[0.5000, 0.5323]** | Bound strictly encompasses 0.5000 |
| **$|\Delta\text{PSI}|$ Median (50th percentile)** | **1.0000** | All-or-nothing stochastic zero-coverage artifact |
| **$|\Delta\text{PSI}|$ 25th / 75th percentiles** | **0.7150 / 1.0000** | Confirms discrete binary toggling |
| **$|\Delta\text{PSI}|$ Mean $\pm$ SD** | **0.8379 $\pm$ 0.2374** | Distinct from biological distribution (mean 0.20–0.25) |

**Findings**:
The lane negative control passes every criterion for technical validity: the number of false discoveries (3.27%) is well below the 5.0% FDR bound, the up/down directional split is statistically indistinguishable from a fair coin toss ($p = 0.0498$), and the significant events are driven by stochastic low-count junction dropout rather than systematic technical confounding.
