# Audit Documentation: Retraction of Post-Hoc MaxEntScan Exploration

**Directory**: `audit/exploratory_not_for_inference/`  
**Date of Retraction**: 2026-09-24  
**Logged Milestone**: `docs/COMMAND_LOG.md` (§3.7.2, §4.6, Chapter 5 Methodological Integrity)

---

## 1. Context and Rationale for Retraction

During Stage 7 V3 modeling (`scripts/06_train_evaluate_maxent.py`), 5-fold cross-validation on the KD1 training set ($N=6,954$) selected `max_depth=20, n_estimators=100` as the optimal hyperparameter configuration (CV Macro-F1 = 0.6210).

Subsequently, an exploratory scratch script (`scripts/check_maxent_configs.py`) evaluated an unconstrained configuration (`max_depth=None, n_estimators=300`) post-hoc against both the locked internal test partition ($N=1,739$) and the external biological validation cohort (KD2, $N=7,080$). Because that unconstrained configuration yielded higher raw accuracy (76.19% test acc, 80.34% KD2 acc), it was initially considered for reporting.

However, selecting or highlighting a model variant based on performance against a held-out test or validation set violates strict machine learning methodology:
1. **Information Leakage**: Evaluating multiple hyperparameter configurations on locked test data turns the test set into a tuning set.
2. **Post-Hoc Optimization Bias**: Promoting a model based on out-of-sample metrics inflates reported generalization.

To preserve strict scientific integrity for the BSc dissertation:
- The exploratory script was immediately removed (`rm -f scripts/check_maxent_configs.py`).
- All post-hoc comparisons were formally retracted in `docs/COMMAND_LOG.md` (lines 503–508).
- The official MaxEntScan results were locked strictly to the training-CV winner (`max_depth=20, n_estimators=100`).

---

## 2. Recovery Status

Because `scripts/check_maxent_configs.py` was an untracked scratch script deleted before git staging, it was never committed to the git tree and is **unrecoverable from git history or filesystem blocks**. This audit file permanently records the rationale, parameters tested, and formal retraction.
