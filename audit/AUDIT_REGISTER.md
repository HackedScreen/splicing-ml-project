# Dissertation Pipeline Audit Register

**Audit Tag**: `thesis-results-v3`  
**Date**: 2026-10-12  
**Purpose**: Rigorous methodological and provenance audit for dissertation defense and examiners.

---

## 1. Target Gene & Protein Nomenclature

- **Gene Symbol**: *HNRNPC* (Ensembl ID: ENSG00000092199)
- **Protein**: Heterogeneous Nuclear Ribonucleoprotein C (HNRNPC, isoforms C1/C2)
- **Standardization**: All pipeline documentation, scripts, and logs standardize strictly on **HNRNPC** (correcting any colloquial references such as hnRNP C, hnRNPC, or HNRPC).

---

## 2. Control Downsampling Parameters

- **Tool**: `seqtk sample` (version 1.4-r122)
- **Target Fraction**: `0.469` (derived from sequencing depth balancing between Control and KD1 libraries: Control library ~42.3M pairs vs KD1 library ~19.8M pairs; ratio $19.8 / 42.3 \approx 0.4687 \approx 0.469$)
- **Random Seed**: `42` (`-s42`)
- **Exact Invocation**:
  ```bash
  seqtk sample -s42 Control_rep1_1.fastq.gz 0.469 | gzip > Control_rep1_downsampled_1.fastq.gz
  seqtk sample -s42 Control_rep1_2.fastq.gz 0.469 | gzip > Control_rep1_downsampled_2.fastq.gz
  seqtk sample -s42 Control_rep2_1.fastq.gz 0.469 | gzip > Control_rep2_downsampled_1.fastq.gz
  seqtk sample -s42 Control_rep2_2.fastq.gz 0.469 | gzip > Control_rep2_downsampled_2.fastq.gz
  ```
- **Provenance**: Logged in `docs/COMMAND_LOG.md` (§3.4) and executed in `scripts/04_downsample_controls.sh`.

---

## 3. Sample Flow & Read Metrics Provenance

| Run Accession | Sample Role | ENA Raw Pairs (read_count) | Trimmed Pairs Survived (%) | HISAT2 Alignment Rate (%) | Mapped Pairs (Derived) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `ERR127302` | KD1 Rep 1 | 20,447,883 | 18,211,889 (89.06%) | 94.30% | 17,173,811 |
| `ERR127303` | KD1 Rep 2 | 22,125,745 | 19,836,367 (89.65%) | 94.02% | 18,650,152 |
| `ERR127304` | KD2 Rep 1 | 24,000,562 | 21,797,117 (90.82%) | 94.55% | 20,609,174 |
| `ERR127305` | KD2 Rep 2 | 21,807,819 | 19,742,912 (90.53%) | 94.22% | 18,601,772 |
| `ERR127306` | Control Rep 1 (Lane 1) | 44,795,849 | 40,323,398 (90.02%) | 94.88% | 38,258,840 |
| `ERR127307` | Control Rep 1 (Lane 2) | 44,228,881 | 39,812,059 (90.01%) | 94.86% | 37,765,719 |
| `ERR127308` | Control Rep 2 (Lane 1) | 41,200,942 | 37,125,483 (90.11%) | 94.61% | 35,124,419 |
| `ERR127309` | Control Rep 2 (Lane 2) | 40,789,124 | 36,750,919 (90.10%) | 94.60% | 34,766,369 |

*Note on Mapped Pairs*: The metric "Mapped Pairs" is explicitly labeled **Derived** because it represents the mathematical product of trimmed surviving pairs multiplied by the overall alignment rate reported by HISAT2 (`Surviving_Pairs * Overall_Alignment_Rate`), rather than a direct raw count from a primary sequencing summary.

---

## 4. Stratified Dummy Classifier Recomputed Metrics

Evaluated on the Gene-Grouped Held-Out Test Set ($N=1,799$, strictly zero gene overlap with training set):

- **Baseline 1: Majority Class (`DummyClassifier(strategy='most_frequent')`)**
  - Predicting dominant class (`SE`, $N=892$, 49.58%):
  - Accuracy: `0.4958` (49.58%)
  - Macro-F1: `0.1326`
  - Per-class F1: SE = `0.6630`, A3SS = `0.0000`, A5SS = `0.0000`, MXE = `0.0000`, RI = `0.0000`

- **Baseline 2: Stratified Chance (`DummyClassifier(strategy='stratified', random_state=42)`)**
  - Accuracy: `0.3535` (35.35%)
  - Macro-F1: `0.1912`
  - Per-class breakdown:
    - **A3SS** ($N=153$): Precision = `0.0844`, Recall = `0.0850`, F1 = `0.0847`
    - **A5SS** ($N=81$): Precision = `0.0000`, Recall = `0.0000`, F1 = `0.0000`
    - **MXE** ($N=566$): Precision = `0.3037`, Recall = `0.3021`, F1 = `0.3029`
    - **RI** ($N=107$): Precision = `0.0787`, Recall = `0.0654`, F1 = `0.0714`
    - **SE** ($N=892$): Precision = `0.4950`, Recall = `0.4989`, F1 = `0.4969`

---

## 5. Execution Environment by Pipeline Step

| Pipeline Step | Script / Mission | Execution Environment | Hardware Context |
| :--- | :--- | :--- | :--- |
| FASTQ Download | `scripts/01_download_fastq.sh` | Local WSL (Ubuntu 22.04 LTS) | Local NVMe SSD |
| Trimming | `scripts/02_trim_reads.sh` | Local WSL (Ubuntu 22.04 LTS) | 8 vCPUs, 16 GB RAM |
| Alignment & Sorting | `scripts/03_align_reads.sh` | Local WSL (Ubuntu 22.04 LTS) | 8 vCPUs, 16 GB RAM |
| Control Downsampling | `scripts/04_downsample_controls.sh` | Local WSL (Ubuntu 22.04 LTS) | 8 vCPUs, 16 GB RAM |
| Differential Splicing | `scripts/05_run_rmats.sh` | Local WSL (Ubuntu 22.04 LTS) | 8 vCPUs, 16 GB RAM |
| Feature Extraction | `scripts/05_feature_extraction.py` | Local WSL (Ubuntu 22.04 LTS) | Python 3.11 conda env |
| Baseline ML Models | `scripts/06_train_evaluate_baseline.py`| Local WSL (Ubuntu 22.04 LTS) | Python 3.11 conda env |
| Model Iteration (V2) | `scripts/07_model_iteration_v2.py` | Local WSL (Ubuntu 22.04 LTS) | Python 3.11 conda env |
| MaxEntScan RF (V3) | `scripts/06_train_evaluate_maxent.py` | Local WSL (Ubuntu 22.04 LTS) | Python 3.11 conda env |
| Gene-Grouped Split (07c) | `scripts/07c_gene_grouped_split_robustness.py` | Local WSL (Ubuntu 22.04 LTS) | Python 3.11 conda env |
| Supervisor Battery (07d) | `scripts/07d_supervisor_followup_analyses.py` | Local WSL (Ubuntu 22.04 LTS) | Python 3.11 conda env |

*Note*: No pipeline steps were executed on Kaggle kernels or external cloud instances. The entire end-to-end pipeline was executed in a local WSL2 environment under Ubuntu 22.04.

---

## 6. Software Versions (Logged vs Verified)

| Software / Library | Version Recorded in Logs | Verification Source | Status |
| :--- | :---: | :--- | :---: |
| Trimmomatic | `0.39` | `logs/02_trimming.log` | Logged |
| HISAT2 | `2.2.1` | `logs/03_alignment.log` | Logged |
| samtools | `1.19.2` | `logs/03_alignment.log` | Logged |
| seqtk | `1.4-r122` | `logs/04_downsample.log` | Logged |
| rMATS | `v4.4.0` | `rmats.py --version` (CLI output) | Logged |
| Python | `3.11.9` | `python --version` (Miniforge3) | Logged |
| scikit-learn | `1.4.2` | `sklearn.__version__` | Logged |
| XGBoost | `2.0.3` | `xgboost.__version__` | Logged |
| pandas | `2.2.2` | `pandas.__version__` | Logged |
| numpy | `1.26.4` | `numpy.__version__` | Logged |
| FastQC | `not recorded` | Not captured in Stage 01 stdout log | Not Recorded |
| STAR (if benchmarked) | `not recorded` | Not utilized in final primary pipeline | Not Recorded |
