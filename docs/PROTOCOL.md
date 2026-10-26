# Machine Learning Evaluation Protocol & Analysis Plan

**Document**: `docs/PROTOCOL.md`  
**Date**: 2026-10-13  
**Status**: Pre-registered protocol locked prior to model training on variable-read-length feature matrices.

---

## 1. Objective and Methodological Rationale

Following supervisor review and methodological audit (R1), all RNA-seq differential splicing quantifications are executed with rMATS v4.4.0 incorporating `--variable-read-length` to eliminate read-truncation filtering artifacts. This document pre-registers the exact modeling pipeline, hyperparameter tuning grid, evaluation standards, and acceptance thresholds before any candidate model is trained or evaluated on the updated matrices.

---

## 2. Data Preparation & Grouped Partitioning

1. **Feature Matrices**:
   - KD1 Discovery Matrix: `results/features/feature_matrix_v2.csv`
   - KD2 External Validation Matrix: `results/features/feature_matrix_kd2_v2.csv`
   - Feature count: Exactly 23 features (17 numeric sequence/PWM/coverage/PSI features + 4 U-tract binding features + 2 one-hot strand dummies).
   - Statistical leakage exclusion: No rMATS $p$-values, FDR, or coordinate symmetry proxy features.
   - Grouping identifier: `gene_id` (derived from Ensembl 116 GTF annotation).

2. **Train/Test Splitting**:
   - Splitting method: `sklearn.model_selection.GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)`
   - Grouping column: `gene_id`
   - Acceptance criterion: Exactly zero overlapping `gene_id` values between the training set ($X_{\text{train}}, y_{\text{train}}$) and test set ($X_{\text{test}}, y_{\text{test}}$).

---

## 3. Candidate Models and Hyperparameter Grid Search

Hyperparameter tuning is conducted strictly on the training partition ($N_{\text{train}}$) using 5-fold cross-validation grouped by gene: `GroupKFold(n_splits=5)`. Held-out test data is never accessed during grid search.

### Candidate 1: Random Forest Classifier
- Base Estimator: `sklearn.ensemble.RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=2)`
- Grid Search Space:
  - `n_estimators`: `[100, 300]`
  - `max_depth`: `[None, 10, 20]`
  - Total configurations: 6

### Candidate 2: XGBoost Classifier
- Base Estimator: `xgboost.XGBClassifier(random_state=42, eval_metric='mlogloss', tree_method='hist', n_jobs=2)`
- Sample Weights: `compute_sample_weight('balanced', y_train)` computed fold-wise.
- Grid Search Space:
  - `n_estimators`: `[100, 300]`
  - `max_depth`: `[3, 6]`
  - `learning_rate`: `[0.05, 0.1]`
  - Total configurations: 8

---

## 4. Model Selection Criterion

1. **Selection Metric**: Mean 5-fold cross-validation Macro-F1:
   $$\overline{\text{Macro-F1}} = \frac{1}{5} \sum_{k=1}^5 \text{Macro-F1}_k$$
2. **Variance Reporting**: Standard deviation across the 5 validation folds must use sample degrees of freedom (`ddof=1`):
   $$s = \sqrt{\frac{1}{4} \sum_{k=1}^5 (\text{Macro-F1}_k - \overline{\text{Macro-F1}})^2}$$
3. **Winner Rule**: The candidate configuration with the highest mean grouped-CV Macro-F1 is selected as the primary dissertation model.
4. **Secondary Model**: The runner-up model configuration will also be evaluated on the test set and reported transparently.

---

## 5. Held-Out Evaluation Protocol

1. **Internal Generalization (KD1 Test Set)**:
   - Evaluated ONCE on the group-held-out test partition ($N_{\text{test}}$).
   - Metrics: Overall Accuracy, Macro-F1, Weighted-F1, per-class breakdown (Precision, Recall, F1 for A3SS, A5SS, MXE, RI, SE), and Confusion Matrix.
   - Baselines: Evaluated alongside Majority-Class (`DummyClassifier(strategy='most_frequent')`) and Stratified Chance (`DummyClassifier(strategy='stratified', random_state=42)`).

2. **External Biological Generalization (KD2 Cohort)**:
   - Primary model applied directly to the independent KD2 dataset ($N_{\text{KD2}}$).
   - Stratified evaluation across three predefined strata:
     - Full KD2 Cohort
     - Seen-Gene Stratum: KD2 events originating from genes observed in the KD1 training set.
     - Unseen-Gene Stratum: KD2 events originating from genes completely absent from the KD1 training set.
   - Each stratum reported with its own empirical Majority-Class baseline.

---

## 6. Supervisor Robustness Battery

1. **Gene-Level Non-Parametric Bootstrap**:
   - 1,000 bootstrap iterations on the test set resampling cluster groups (`gene_id`) with replacement.
   - Report point estimate, bootstrap mean, bootstrap SD, and 95% empirical percentile confidence intervals for Accuracy, Macro-F1, and per-class F1 scores.

2. **Probability Calibration**:
   - Compute Multiclass Brier Score for the uncalibrated primary model.
   - Fit `CalibratedClassifierCV(method='sigmoid', cv='prefit')` on the training set and evaluate test Brier score and reliability curves.

3. **Feature Ablation Battery (5-Fold GroupKFold)**:
   - (A) All 23 features
   - (B) All minus `exon_length` (22 features)
   - (C) Sequence/Structure only (12 features)
   - (D) Sequence/Structure only minus `exon_length` (11 features)
   - (E) Read Coverage & PSI kinetics only (11 features)

4. **Information Leakage Diagnostics**:
   - Mutual information (`mutual_info_classif`, `random_state=42`) ranked across all 23 features.
   - Identification of exact duplicate feature rows.
   - Genomic locus coordinate overlap audit between training and test sets.

5. **Technical Negative Control (Lane Comparison)**:
   - Exact two-sided binomial test of $H_0: p = 0.5$ on directional shifts (Sample 1 higher vs Sample 2 higher) across all five splicing classes.
   - 95% Clopper-Pearson confidence intervals.
   - $|\Delta\Psi|$ distribution summary (median, IQR, mean $\pm$ SD) and total junction read depth ($IJC + SJC$).
