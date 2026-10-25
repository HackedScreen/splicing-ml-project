# Dissertation Pipeline Audit Register

**Audit tag at first writing**: `thesis-results-v3` (2026-10-12)  
**Corrected**: 2026-10-25  

> **CORRECTION NOTICE (2026-10-25).** The version of this register dated 2026-10-12 described the superseded pipeline (tag `thesis-results-v3`) and contained statements that conflict with the project record: the execution environment (it said nothing ran on Kaggle), the downsampling method (it said `seqtk` on FASTQ files), the per-run read counts, the held-out test set (N = 1,799), and several software versions. Sections 2-6 below replace the old ones. The original text is kept unchanged at the end of this file under "Superseded text" for audit history. Every value below names its source. "Not recorded" means no record was found; it does not mean the item is correct.

---

## 1. Target gene and protein nomenclature (unchanged)

- **Gene symbol**: *HNRNPC* (Ensembl ID: ENSG00000092199)
- **Protein**: Heterogeneous Nuclear Ribonucleoprotein C (HNRNPC, isoforms C1/C2)

---

## 2. Control downsampling (corrected)

- **Method**: `samtools view -@ 4 -s 42.469 -b` applied to the coordinate-sorted Control BAMs (seed 42, fraction 0.469). Source: `docs/COMMAND_LOG.md` (2026-09-22 entries); dissertation Section 3.5.3. The previous statement that `seqtk sample` was run on FASTQ files is incorrect.
- **Rationale**: pooled Control trimmed pairs 112,640,171 (55,453,742 + 57,186,429) against KD1 52,827,938 (26,069,721 + 26,758,217); 0.469 x 112,640,171 = 52,828,240, which matches the KD1 total. Source: Trimmomatic logs `results/alignment/*_trimmomatic.log` and arithmetic.

---

## 3. Sample flow and read metrics (corrected)

Raw pairs are the ENA `read_count` values (ENA portal API, re-checked 2026-10-24) and equal the "Input Read Pairs" in the Trimmomatic logs. The Control runs were concatenated by replicate before trimming, so trimming statistics exist per Control replicate, not per run.

| Run / library | Role | Raw pairs | Trimmed pairs (survival) | Primary alignment rate (samtools flagstat) |
| :--- | :--- | :---: | :---: | :---: |
| `ERR127302` | KD1 rep 1 | 29,741,852 | 26,069,721 (87.65%) | 95.29% |
| `ERR127303` | KD1 rep 2 | 32,665,567 | 26,758,217 (81.92%) | 95.55% |
| `ERR127304` | KD2 rep 1 | 31,876,181 | 26,945,447 (84.53%) | 96.25% |
| `ERR127305` | KD2 rep 2 | 32,345,742 | 28,042,418 (86.70%) | 94.74% |
| `Control_rep1` (`ERR127306` + `ERR127307`) | Control rep 1 | 29,038,455 + 32,442,539 = 61,480,994 | 55,453,742 (90.20%) | 94.52% |
| `Control_rep2` (`ERR127308` + `ERR127309`) | Control rep 2 | 31,096,102 + 31,168,942 = 62,265,044 | 57,186,429 (91.84%) | 89.49% |
| **Total** | | **250,375,380** | **220,455,974 (88.05%)** | |

After `scripts/split_lanes.py` separated `Control_rep1` by read-name prefix, the surviving pairs were 26,304,501 (`ERR127306`) and 29,149,241 (`ERR127307`), which sum to 55,453,742. Source: `samtools flagstat` of the split BAMs. The old per-run values in the superseded table (for example 20,447,883 for `ERR127302`) do not match ENA and are withdrawn. "Mapped pairs (derived)" is reported only in the dissertation (Table 4.2).

---

## 4. Dummy classifier baselines on the held-out test set (corrected)

Held-out test set: N = 1,807 events from 748 genes, zero genes shared with the 7,440 training events (2,989 genes). Class counts: A3SS 216, A5SS 114, MXE 431, RI 132, SE 914.

| Baseline | Accuracy | Macro-F1 | Weighted-F1 |
| :--- | :---: | :---: | :---: |
| Majority class (`most_frequent`, predicts SE, 914 / 1,807 = 50.58%) | 0.5058 | 0.1344 | 0.3398 |
| Stratified (`random_state=42`, fitted on training labels) | 0.3464 | 0.1963 | 0.3377 |

These values were recomputed independently on 2026-10-25 from `results/models/r1_train_test_data_grouped.joblib` and `results/features/feature_matrix_v2.csv`. The N = 1,799 test set in the superseded section belongs to an earlier pipeline.

**Independent recomputation of the primary results (2026-10-25).** Using the saved models and the saved gene-grouped split (recomputation environment: scikit-learn 1.8.0, XGBoost 3.4.1): XGBoost accuracy 0.6768 (1,223 / 1,807), macro-F1 0.6060, weighted-F1 0.6741, with a confusion matrix identical to dissertation Figure 4.1; Random Forest accuracy 0.6624 (1,197 / 1,807), macro-F1 0.5651, weighted-F1 0.6522. The KD2 confusion matrix (7,279 events) is identical to dissertation Table B.1.

---

## 5. Execution environment by pipeline stage (corrected)

| Stage | Environment | Source |
| :--- | :--- | :--- |
| ENA download, Trimmomatic, linear HISAT2 index build (49.9 min), HISAT2 alignment, samtools sort and index | Kaggle notebooks (notebooks not retained; commands reconstructed from BAM `@PG` headers and Trimmomatic logs) | Project record; BAM headers; `docs/COMMAND_LOG.md` |
| Control downsampling, `scripts/split_lanes.py`, rMATS, feature extraction (`scripts/r1_step4_extract_features.py`), modelling (`scripts/r1_step5_model_protocol.py`), Streamlit app | Local WSL2 (`.wslconfig`: 4 processors, 3.8 GiB RAM, 8-16 GiB swap over the project) | `docs/COMMAND_LOG.md` |

- **Graph-index attempts**: `docs/COMMAND_LOG.md` lists four local builds (one on 2026-09-01 that completed on a truncated GTF; three on 2026-09-07, of which two hit a memory-allocation rejection and one was killed by a WSL update; see the 2026-10-24 correction entry), and one Kaggle attempt (MemTotal 32,870,492 kB, recorded in the project chat and logged on 2026-10-24) ended with Kaggle's memory error after 6 of 8 index files. The linear index plus known splice sites was adopted.
- **Ubuntu release**: not recorded consistently (22.04 in the superseded text, 24.04 in the dissertation). Resolve from the WSL environment (`lsb_release -a`).
- The superseded statement "8 vCPUs, 16 GB RAM, no pipeline step ran on Kaggle" is incorrect.

---

## 6. Software versions

| Software | Version | Source | Status |
| :--- | :---: | :--- | :---: |
| HISAT2 | 2.2.1 | BAM `@PG` header (`ERR127302`, `ERR127305`) | Verified |
| samtools | 1.24 | BAM `@PG` header (`VN:1.24`) | Verified |
| Trimmomatic | 0.41 | Trimmomatic logs (`trimmomatic-0.41-0` path) | Verified |
| rMATS | 4.4.0 | `rmats.py --version`, project record | Reported |
| scikit-learn | **conflict** | The saved Random Forest was pickled with scikit-learn **1.9.0** (detected on loading, 2026-10-25); the superseded text says 1.4.2 and dissertation Table 3.1 says 1.7.1 | **Unresolved** |
| XGBoost | **conflict** | Not recoverable from the saved model; superseded text says 2.0.3, dissertation Table 3.1 says 2.1.4 | **Unresolved** |
| Python, pandas, numpy | **conflict** | Superseded text: 3.11.9, 2.2.2, 1.26.4; dissertation: 3.11.16, 2.2.3, 2.2.1 | **Unresolved** |
| FastQC | 0.12.1 in the dissertation; not recorded here | | Unresolved |
| seqtk | Not used | The superseded text was wrong | n/a |

Open item: resolve every "Unresolved" row from `docs/environment_audit_2026-10-13.txt` and `docs/pip_audit_2026-10-13.txt`, then correct dissertation Table 3.1 if needed.

---

## Superseded text (kept unchanged for audit history; do not rely on it)

> # Dissertation Pipeline Audit Register
>
> **Audit Tag**: `thesis-results-v3`  
> **Date**: 2026-10-12  
> **Purpose**: Rigorous methodological and provenance audit for dissertation defense and examiners.
>
> ---
>
> ## 1. Target Gene & Protein Nomenclature
>
> - **Gene Symbol**: *HNRNPC* (Ensembl ID: ENSG00000092199)
> - **Protein**: Heterogeneous Nuclear Ribonucleoprotein C (HNRNPC, isoforms C1/C2)
> - **Standardization**: All pipeline documentation, scripts, and logs standardize strictly on **HNRNPC** (correcting any colloquial references such as hnRNP C, hnRNPC, or HNRPC).
>
> ---
>
> ## 2. Control Downsampling Parameters
>
> - **Tool**: `seqtk sample` (version 1.4-r122)
> - **Target Fraction**: `0.469` (derived from sequencing depth balancing between Control and KD1 libraries: Control library ~42.3M pairs vs KD1 library ~19.8M pairs; ratio $19.8 / 42.3 \approx 0.4687 \approx 0.469$)
> - **Random Seed**: `42` (`-s42`)
> - **Exact Invocation**:
>   ```bash
>   seqtk sample -s42 Control_rep1_1.fastq.gz 0.469 | gzip > Control_rep1_downsampled_1.fastq.gz
>   seqtk sample -s42 Control_rep1_2.fastq.gz 0.469 | gzip > Control_rep1_downsampled_2.fastq.gz
>   seqtk sample -s42 Control_rep2_1.fastq.gz 0.469 | gzip > Control_rep2_downsampled_1.fastq.gz
>   seqtk sample -s42 Control_rep2_2.fastq.gz 0.469 | gzip > Control_rep2_downsampled_2.fastq.gz
>   ```
> - **Provenance**: Logged in `docs/COMMAND_LOG.md` (§3.4) and executed in `scripts/04_downsample_controls.sh`.
>
> ---
>
> ## 3. Sample Flow & Read Metrics Provenance
>
> | Run Accession | Sample Role | ENA Raw Pairs (read_count) | Trimmed Pairs Survived (%) | HISAT2 Alignment Rate (%) | Mapped Pairs (Derived) |
> | :--- | :--- | :---: | :---: | :---: | :---: |
> | `ERR127302` | KD1 Rep 1 | 20,447,883 | 18,211,889 (89.06%) | 94.30% | 17,173,811 |
> | `ERR127303` | KD1 Rep 2 | 22,125,745 | 19,836,367 (89.65%) | 94.02% | 18,650,152 |
> | `ERR127304` | KD2 Rep 1 | 24,000,562 | 21,797,117 (90.82%) | 94.55% | 20,609,174 |
> | `ERR127305` | KD2 Rep 2 | 21,807,819 | 19,742,912 (90.53%) | 94.22% | 18,601,772 |
> | `ERR127306` | Control Rep 1 (Lane 1) | 44,795,849 | 40,323,398 (90.02%) | 94.88% | 38,258,840 |
> | `ERR127307` | Control Rep 1 (Lane 2) | 44,228,881 | 39,812,059 (90.01%) | 94.86% | 37,765,719 |
> | `ERR127308` | Control Rep 2 (Lane 1) | 41,200,942 | 37,125,483 (90.11%) | 94.61% | 35,124,419 |
> | `ERR127309` | Control Rep 2 (Lane 2) | 40,789,124 | 36,750,919 (90.10%) | 94.60% | 34,766,369 |
>
> *Note on Mapped Pairs*: The metric "Mapped Pairs" is explicitly labeled **Derived** because it represents the mathematical product of trimmed surviving pairs multiplied by the overall alignment rate reported by HISAT2 (`Surviving_Pairs * Overall_Alignment_Rate`), rather than a direct raw count from a primary sequencing summary.
>
> ---
>
> ## 4. Stratified Dummy Classifier Recomputed Metrics
>
> Evaluated on the Gene-Grouped Held-Out Test Set ($N=1,799$, strictly zero gene overlap with training set):
>
> - **Baseline 1: Majority Class (`DummyClassifier(strategy='most_frequent')`)**
>   - Predicting dominant class (`SE`, $N=892$, 49.58%):
>   - Accuracy: `0.4958` (49.58%)
>   - Macro-F1: `0.1326`
>   - Per-class F1: SE = `0.6630`, A3SS = `0.0000`, A5SS = `0.0000`, MXE = `0.0000`, RI = `0.0000`
>
> - **Baseline 2: Stratified Chance (`DummyClassifier(strategy='stratified', random_state=42)`)**
>   - Accuracy: `0.3535` (35.35%)
>   - Macro-F1: `0.1912`
>   - Per-class breakdown:
>     - **A3SS** ($N=153$): Precision = `0.0844`, Recall = `0.0850`, F1 = `0.0847`
>     - **A5SS** ($N=81$): Precision = `0.0000`, Recall = `0.0000`, F1 = `0.0000`
>     - **MXE** ($N=566$): Precision = `0.3037`, Recall = `0.3021`, F1 = `0.3029`
>     - **RI** ($N=107$): Precision = `0.0787`, Recall = `0.0654`, F1 = `0.0714`
>     - **SE** ($N=892$): Precision = `0.4950`, Recall = `0.4989`, F1 = `0.4969`
>
> ---
>
> ## 5. Execution Environment by Pipeline Step
>
> | Pipeline Step | Script / Mission | Execution Environment | Hardware Context |
> | :--- | :--- | :--- | :--- |
> | FASTQ Download | `scripts/01_download_fastq.sh` | Local WSL (Ubuntu 22.04 LTS) | Local NVMe SSD |
> | Trimming | `scripts/02_trim_reads.sh` | Local WSL (Ubuntu 22.04 LTS) | 8 vCPUs, 16 GB RAM |
> | Alignment & Sorting | `scripts/03_align_reads.sh` | Local WSL (Ubuntu 22.04 LTS) | 8 vCPUs, 16 GB RAM |
> | Control Downsampling | `scripts/04_downsample_controls.sh` | Local WSL (Ubuntu 22.04 LTS) | 8 vCPUs, 16 GB RAM |
> | Differential Splicing | `scripts/05_run_rmats.sh` | Local WSL (Ubuntu 22.04 LTS) | 8 vCPUs, 16 GB RAM |
> | Feature Extraction | `scripts/05_feature_extraction.py` | Local WSL (Ubuntu 22.04 LTS) | Python 3.11 conda env |
> | Baseline ML Models | `scripts/06_train_evaluate_baseline.py`| Local WSL (Ubuntu 22.04 LTS) | Python 3.11 conda env |
> | Model Iteration (V2) | `scripts/07_model_iteration_v2.py` | Local WSL (Ubuntu 22.04 LTS) | Python 3.11 conda env |
> | MaxEntScan RF (V3) | `scripts/06_train_evaluate_maxent.py` | Local WSL (Ubuntu 22.04 LTS) | Python 3.11 conda env |
> | Gene-Grouped Split (07c) | `scripts/07c_gene_grouped_split_robustness.py` | Local WSL (Ubuntu 22.04 LTS) | Python 3.11 conda env |
> | Supervisor Battery (07d) | `scripts/07d_supervisor_followup_analyses.py` | Local WSL (Ubuntu 22.04 LTS) | Python 3.11 conda env |
>
> *Note*: No pipeline steps were executed on Kaggle kernels or external cloud instances. The entire end-to-end pipeline was executed in a local WSL2 environment under Ubuntu 22.04.
>
> ---
>
> ## 6. Software Versions (Logged vs Verified)
>
> | Software / Library | Version Recorded in Logs | Verification Source | Status |
> | :--- | :---: | :--- | :---: |
> | Trimmomatic | `0.39` | `logs/02_trimming.log` | Logged |
> | HISAT2 | `2.2.1` | `logs/03_alignment.log` | Logged |
> | samtools | `1.19.2` | `logs/03_alignment.log` | Logged |
> | seqtk | `1.4-r122` | `logs/04_downsample.log` | Logged |
> | rMATS | `v4.4.0` | `rmats.py --version` (CLI output) | Logged |
> | Python | `3.11.9` | `python --version` (Miniforge3) | Logged |
> | scikit-learn | `1.4.2` | `sklearn.__version__` | Logged |
> | XGBoost | `2.0.3` | `xgboost.__version__` | Logged |
> | pandas | `2.2.2` | `pandas.__version__` | Logged |
> | numpy | `1.26.4` | `numpy.__version__` | Logged |
> | FastQC | `not recorded` | Not captured in Stage 01 stdout log | Not Recorded |
> | STAR (if benchmarked) | `not recorded` | Not utilized in final primary pipeline | Not Recorded |
>
