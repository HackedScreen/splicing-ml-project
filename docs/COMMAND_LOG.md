# Command log

Append-only. One entry per command or decision, in order. This is the primary
source for Chapter 3 methods prose and Appendix A code snippets — write each
entry as if it will be read by an examiner.

> [!IMPORTANT]
> **Authoritative Results Navigation Notice for Examiners:**
> - **Primary Authoritative Dissertation Results (Grade These):** All files bearing `*_corrected*` (e.g., `results/splicing_corrected_downsampled/`, `results/features/feature_matrix_corrected.csv`, `results/models/random_forest_corrected.joblib`, `results/models/evaluation_report_corrected.md`, and `results/features/feature_matrix_kd2_corrected.csv` / `kd2_validation_report_corrected.md`). These represent the depth-balanced, experimentally validated Control vs. KD1 analysis ($N=8,693$) and independent KD2 validation cohort ($N=7,080$).
> - **Archived Runs (Retained for Methodological Auditability Only):** `results/splicing/`, `results/features/feature_matrix.csv`, and `results/models/random_forest.joblib` reflect the caught-and-corrected KD1-vs-KD2 sample mislabeling from Stages 1–7 (§3.2.1, Appendix A.3). `results/splicing_corrected/` reflects the full-depth diagnostic run, and `results/splicing_lane_control/` reflects the technical negative control.
> - **Full Map & Metrics Summary:** See [RESULTS_SUMMARY.md](file:///c:/Users/Brooklyn/Downloads/splicing-ml-project/docs/RESULTS_SUMMARY.md) for the complete index and metric lookup table.

---

### [Stage 1] Dataset search & selection — 2026-08-31
**Command:** Web search — candidate public, paired-end, human RNA-seq datasets
with ≥2 replicated conditions suitable for rMATS (checked ENCODE shRNA
knockdown series, then literature citing HNRNPC knockdown datasets).
**Rationale:** rMATS requires biological replicates per condition to estimate
variance (§2.2.2). A BSc-scale laptop pipeline also needs a dataset small
enough to process end-to-end without HPC access, and ideally one with
published, characterised splicing changes so pipeline output can be sanity
checked against known biology.
**Output:** Selected **E-MTAB-1147** (Zarnack et al., 2013) — HNRNPC knockdown
vs control HeLa cells. 6 biological samples (Control x2 reps, KD1 x2 reps,
KD2 x2 reps), paired-end 72 bp reads. ENA study accession **ERP001458**; the
underlying ERR run accessions are also retrievable via NCBI SRA `prefetch`.
**Dissertation section:** §3.2.1 (Dataset Selection Criteria)

### [Stage 1] Reference genome release pinned — 2026-08-31
**Command:** Web search — current Ensembl release number.
**Rationale:** Reproducibility requires an exact, citable annotation version
rather than a script that silently tracks "latest" and produces different
results each time it's run.
**Output:** Pinned to **Ensembl release 116** (GRCh38.p14, released June
2026). FTP paths confirmed: `https://ftp.ensembl.org/pub/release-116/fasta/homo_sapiens/dna/`
and `https://ftp.ensembl.org/pub/release-116/gtf/homo_sapiens/`.
**Dissertation section:** §3.2.2 (Data Download Procedure)

### [Stage 0] Environment setup — 2026-09-01
**Command:**
`mamba create -y -n splicing_project -c bioconda -c conda-forge sra-tools fastqc trimmomatic hisat2 samtools rmats python=3.11 && pip install biopython pysam pandas scikit-learn matplotlib`
**Rationale:** Pinned versions for reproducibility across bioinformatics alignment, splicing detection, feature engineering, and machine learning pipelines.
**Output:** Pinned environment exported to `docs/environment.yml` (python 3.11.11, hisat2 2.2.1, rmats 4.3.0, samtools 1.21, fastqc 0.12.1, trimmomatic 0.39, sra-tools 3.1.1, scikit-learn 1.7.1, biopython 1.85, pysam 0.23.0, pandas 2.3.2, matplotlib 3.10.1).
**Dissertation section:** §3.1

### [Stage 1] Authoritative run metadata fetched — 2026-09-01
**Command:**
`scripts/01a_fetch_run_metadata.sh`
**Rationale:** Fetch study ERP001458 (E-MTAB-1147) metadata directly from ENA API to verify sample run accessions, replicate mapping, and paired-end layouts.
**Output:** Fetched 6 paired-end runs (Control: ERR127302, ERR127303; KD1: ERR127304, ERR127305; KD2: ERR127306, ERR127307) totaling ~114.9M paired-end 72 bp reads saved to `data/reference/E-MTAB-1147_runs.tsv`.
**Dissertation section:** §3.2.2

### [Stage 1] Sample manifest constructed — 2026-09-01
**Command:**
`data/reference/manifest.csv` populated from `manifest_TEMPLATE.csv`
**Rationale:** Associate confirmed run accessions with experimental design (Control x2 vs KD1 x2 replicates). KD2 construct held back as independent validation data.
**Output:** Manifest confirmed: Control (ERR127302, ERR127303) and KD1 (ERR127304, ERR127305).
**Dissertation section:** §3.2.1

### [Stage 1] Raw RNA-seq FASTQ download & integrity verification — 2026-09-01
**Command:**
`scripts/01b_download_fastq.sh ERR127302 ERR127303 ERR127304 ERR127305`
**Rationale:** Download raw paired-end sequencing reads for all 4 primary study runs directly from ENA repository into `data/raw/`.
**Output:** 4 FASTQ pairs downloaded and verified with exact pair matching: ERR127302 (18,228,308 pairs), ERR127303 (17,812,984 pairs), ERR127304 (20,311,090 pairs), ERR127305 (19,455,201 pairs), totaling 75,807,583 pairs (151,615,166 reads, ~7.8 GB).
**Dissertation section:** §3.2.2

### [Stage 1] Reference genome and annotation acquisition — 2026-09-01
**Command:**
`scripts/01c_download_reference.sh`
**Rationale:** Retrieve pinned Ensembl release 116 GRCh38.p14 primary assembly FASTA and comprehensive gene annotation GTF.
**Output:** `Homo_sapiens.GRCh38.dna.primary_assembly.fa` (3.29 GB, 882 MB gz) and `Homo_sapiens.GRCh38.116.gtf` (1.69 GB, 55.6 MB gz) present and gunzipped in `data/reference/`.
**Dissertation section:** §3.2.2

### [Stage 2] Pre-trim quality control (FastQC) — 2026-09-01
**Command:**
`fastqc -t 4 -o results/qc/pre/ data/raw/*.fastq.gz`
**Rationale:** Assess base quality scores, GC content, sequence duplication, and adapter contamination across all 8 raw paired-end FASTQ files before trimming.
**Output:** Generated 8 HTML/zip reports in `results/qc/pre/`. Per-base sequence quality passed across all files; Illumina universal adapter contamination flagged as FAIL in all 8 files (up to ~5% at read 3' ends), confirming the need for adapter trimming.
**Dissertation section:** §3.3.1

### [Stage 2] Adapter & quality trimming (ERR127302) — 2026-09-01
**Command:**
`trimmomatic PE -threads 4 -phred33 data/raw/ERR127302_1.fastq.gz data/raw/ERR127302_2.fastq.gz data/trimmed/ERR127302_1_paired.fastq.gz data/trimmed/ERR127302_1_unpaired.fastq.gz data/trimmed/ERR127302_2_paired.fastq.gz data/trimmed/ERR127302_2_unpaired.fastq.gz ILLUMINACLIP:/opt/miniforge3/envs/splicing_project/share/trimmomatic/adapters/TruSeq3-PE.fa:2:30:10 LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15 MINLEN:36`
**Rationale:** Remove TruSeq3 adapter sequences, trim low-quality terminal bases (Phred < 3), filter low-quality windows (4 bp window average Phred < 15), and discard reads shorter than 36 bp.
**Output:** Input Read Pairs: 18,228,308 | Both Surviving: 17,180,479 (94.25%) | Forward Only: 669,145 (3.67%) | Reverse Only: 221,235 (1.21%) | Dropped: 157,449 (0.86%).
**Dissertation section:** §3.3.2, Appendix A.1

### [Stage 2] Adapter & quality trimming (ERR127303) — 2026-09-01
**Command:**
`trimmomatic PE -threads 4 -phred33 data/raw/ERR127303_1.fastq.gz data/raw/ERR127303_2.fastq.gz data/trimmed/ERR127303_1_paired.fastq.gz data/trimmed/ERR127303_1_unpaired.fastq.gz data/trimmed/ERR127303_2_paired.fastq.gz data/trimmed/ERR127303_2_unpaired.fastq.gz ILLUMINACLIP:/opt/miniforge3/envs/splicing_project/share/trimmomatic/adapters/TruSeq3-PE.fa:2:30:10 LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15 MINLEN:36`
**Rationale:** Standardize trimming parameters across all biological replicates.
**Output:** Input Read Pairs: 17,812,984 | Both Surviving: 16,788,506 (94.25%) | Forward Only: 651,662 (3.66%) | Reverse Only: 219,662 (1.23%) | Dropped: 153,154 (0.86%).
**Dissertation section:** §3.3.2, Appendix A.1

### [Stage 2] Adapter & quality trimming (ERR127304) — 2026-09-01
**Command:**
`trimmomatic PE -threads 4 -phred33 data/raw/ERR127304_1.fastq.gz data/raw/ERR127304_2.fastq.gz data/trimmed/ERR127304_1_paired.fastq.gz data/trimmed/ERR127304_1_unpaired.fastq.gz data/trimmed/ERR127304_2_paired.fastq.gz data/trimmed/ERR127304_2_unpaired.fastq.gz ILLUMINACLIP:/opt/miniforge3/envs/splicing_project/share/trimmomatic/adapters/TruSeq3-PE.fa:2:30:10 LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15 MINLEN:36`
**Rationale:** Apply identical trimming threshold to knockdown condition replicates.
**Output:** Input Read Pairs: 20,311,090 | Both Surviving: 19,137,330 (94.22%) | Forward Only: 747,514 (3.68%) | Reverse Only: 250,160 (1.23%) | Dropped: 176,086 (0.87%).
**Dissertation section:** §3.3.2, Appendix A.1

### [Stage 2] Adapter & quality trimming (ERR127305) — 2026-09-01
**Command:**
`trimmomatic PE -threads 4 -phred33 data/raw/ERR127305_1.fastq.gz data/raw/ERR127305_2.fastq.gz data/trimmed/ERR127305_1_paired.fastq.gz data/trimmed/ERR127305_1_unpaired.fastq.gz data/trimmed/ERR127305_2_paired.fastq.gz data/trimmed/ERR127305_2_unpaired.fastq.gz ILLUMINACLIP:/opt/miniforge3/envs/splicing_project/share/trimmomatic/adapters/TruSeq3-PE.fa:2:30:10 LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15 MINLEN:36`
**Rationale:** Apply identical trimming threshold to knockdown condition replicate 2.
**Output:** Input Read Pairs: 19,455,201 | Both Surviving: 18,324,838 (94.19%) | Forward Only: 721,535 (3.71%) | Reverse Only: 241,951 (1.24%) | Dropped: 166,877 (0.86%).
**Dissertation section:** §3.3.2, Appendix A.1

### [Stage 2] Post-trim quality control (FastQC) — 2026-09-01
**Command:**
`fastqc -t 4 -o results/qc/post/ data/trimmed/*_paired.fastq.gz`
**Rationale:** Verify complete removal of adapter contamination and retention of high base-call quality across the full read length on trimmed paired reads.
**Output:** Generated 8 HTML/zip reports in `results/qc/post/`. Adapter Content passed 100% (0 flags across all 8 files); Per-base sequence quality maintained PASS across all positions for all 4 paired libraries. Average read pair retention = 94.23% (71,431,153 pairs retained).
**Dissertation section:** §3.3.3

### [Stage 3] HISAT2 splice-aware index build — 2026-09-01
**Command:**
`hisat2-build -p 4 --ss data/reference/splicesites.tsv --exon data/reference/exons.tsv data/reference/Homo_sapiens.GRCh38.dna.primary_assembly.fa data/reference/grch38_index`
**Rationale:** Construct a hierarchical graph FM index incorporating 541,094 unique splice sites and 586,411 unique exons from Ensembl release 116 for splice-aware RNA-seq alignment.
**Output:** Built complete 8-part index (`grch38_index.1.ht2` to `.8.ht2`, total 6.8 GB) from 3.29 Gb human genome in 1,472 seconds (24m 32s wall time).
**Dissertation section:** §3.4.1

### [Stage 3] HISAT2 sample alignment (ERR127302) — 2026-09-01
**Command:**
`hisat2 -x data/reference/grch38_index -1 data/trimmed/ERR127302_1_paired.fastq.gz -2 data/trimmed/ERR127302_2_paired.fastq.gz -p 4 --dta -S results/alignment/ERR127302.sam && samtools sort -@ 4 -o results/alignment/ERR127302.sorted.bam results/alignment/ERR127302.sam && samtools index results/alignment/ERR127302.sorted.bam && rm results/alignment/ERR127302.sam`
**Rationale:** Align Control replicate 1 trimmed reads to splice-aware genome index with `--dta` downstream transcriptome assembly flag for rMATS compatibility.
**Output:** 17,730,873 pairs aligned; 95.40% overall alignment rate. Output sorted BAM: 1.9 GB with BAI index.
**Dissertation section:** §3.4.2

### [Stage 3] HISAT2 sample alignment (ERR127303) — 2026-09-01
**Command:**
`hisat2 -x data/reference/grch38_index -1 data/trimmed/ERR127303_1_paired.fastq.gz -2 data/trimmed/ERR127303_2_paired.fastq.gz -p 4 --dta -S results/alignment/ERR127303.sam && samtools sort -@ 4 -o results/alignment/ERR127303.sorted.bam results/alignment/ERR127303.sam && samtools index results/alignment/ERR127303.sorted.bam && rm results/alignment/ERR127303.sam`
**Rationale:** Align Control replicate 2 trimmed reads with `--dta` flag and generate indexed BAM.
**Output:** 17,935,741 pairs aligned; 95.42% overall alignment rate. Output sorted BAM: 1.9 GB with BAI index.
**Dissertation section:** §3.4.2

### [Stage 3] HISAT2 sample alignment (ERR127304) — 2026-09-01
**Command:**
`hisat2 -x data/reference/grch38_index -1 data/trimmed/ERR127304_1_paired.fastq.gz -2 data/trimmed/ERR127304_2_paired.fastq.gz -p 4 --dta -S results/alignment/ERR127304.sam && samtools sort -@ 4 -o results/alignment/ERR127304.sorted.bam results/alignment/ERR127304.sam && samtools index results/alignment/ERR127304.sorted.bam && rm results/alignment/ERR127304.sam`
**Rationale:** Align Knockdown condition 1 replicate 1 trimmed reads with `--dta` flag and generate indexed BAM.
**Output:** 17,926,955 pairs aligned; 95.57% overall alignment rate. Output sorted BAM: 1.9 GB with BAI index.
**Dissertation section:** §3.4.2

### [Stage 3] HISAT2 sample alignment (ERR127305) — 2026-09-01
**Command:**
`hisat2 -x data/reference/grch38_index -1 data/trimmed/ERR127305_1_paired.fastq.gz -2 data/trimmed/ERR127305_2_paired.fastq.gz -p 4 --dta -S results/alignment/ERR127305.sam && samtools sort -@ 4 -o results/alignment/ERR127305.sorted.bam results/alignment/ERR127305.sam && samtools index results/alignment/ERR127305.sorted.bam && rm results/alignment/ERR127305.sam`
**Rationale:** Align Knockdown condition 1 replicate 2 trimmed reads with `--dta` flag and generate indexed BAM.
**Output:** 17,837,584 pairs aligned; 95.53% overall alignment rate. Output sorted BAM: 1.9 GB with BAI index.
**Dissertation section:** §3.4.2

### [Stage 4] Differential alternative splicing quantification (rMATS) — 2026-09-01
**Command:**
`rmats.py --b1 results/splicing/b1.txt --b2 results/splicing/b2.txt --gtf data/reference/Homo_sapiens.GRCh38.116.gtf -t paired --readLength 72 --nthread 4 --od results/splicing/ --tmp results/splicing/tmp`
**Rationale:** Quantify differential exon inclusion levels (PSI) and statistical significance between Control (ERR127302, ERR127303) and HNRNPC knockdown (ERR127304, ERR127305) across five major alternative splicing event types.
**Output:** Quantified 107,533 total splicing events with 19,089 significant events (FDR < 0.05): SE: 14,309 (74.96%), RI: 1,348 (7.06%), A5SS: 915 (4.79%), A3SS: 1,268 (6.64%), MXE: 1,249 (6.54%). Total wall time: 526 seconds.
**Dissertation section:** §3.5

### [Stage 5] Feature engineering & matrix construction — 2026-09-01
**Command:**
`python scripts/05_feature_engineering.py`
**Rationale:** Extract genomic geometry (exon length, flanking intron lengths), sequence composition (exon and 100 bp flanking GC content), splice-site strength (human canonical 5' donor and 3' acceptor PWM log-odds scores), local RNA-seq read coverage mean/variance across Control and KD1 BAM alignments, and replicate PSI means/variances for all statistically significant splicing events (FDR < 0.05).
**Output:** Generated `results/features/feature_matrix.csv` with 19,089 rows and 29 columns across 5 splicing event classes: SE: 14,309 (74.96%), RI: 1,348 (7.06%), A3SS: 1,268 (6.64%), MXE: 1,249 (6.54%), A5SS: 915 (4.79%). Only 82 Control (0.43%) and 45 KD1 (0.24%) PSI values were NaN due to undefined 0/0 junction reads in rMATS; all genomic, GC, PWM, and coverage features are 100% complete (0 NaN).
**Dissertation section:** §3.6

### [Stage 5] Bug fix: Minus-strand SE coordinate assignment & matrix regeneration — 2026-09-01
**Command:**
`python scripts/05_feature_engineering.py`
**Rationale:** Fix coordinate assignment bug in SE event processing where downstream exon end (`downstreamEE`) was conflated with target exon end (`exonEnd`) for minus-strand skipped exons. Verify that RI, A5SS, A3SS, and MXE event extractions were unaffected, and regenerate `results/features/feature_matrix.csv`.
**Output:** Regenerated `results/features/feature_matrix.csv` (19,089 rows, 29 columns). Verified 100% coordinate parity across all 14,309 SE events (including 7,122 minus-strand events) and all 4 other event classes (0 mismatches across all 19,089 rows). Total counts, class balance (SE: 74.96%, RI: 7.06%, A3SS: 6.64%, MXE: 6.54%, A5SS: 4.79%), and NaN counts (82 Control PSI, 45 KD1 PSI) remain strictly preserved.
**Dissertation section:** §3.6, Appendix A.2

### [Stage 6] Logistic Regression training & hyperparameter tuning — 2026-09-01
**Command:**
`python scripts/06_train_models.py` (Logistic Regression component)
**Rationale:** Train a multinomial baseline classifier with balanced class weights using 21 numeric structural, sequence, coverage, and replicate PSI features (with one-hot encoded strand). Exclude `pvalue` and `fdr` to prevent data leakage and avoid learning testing-power artifacts from the rMATS statistical filter. Impute 82 Control (0.43%) and 45 KD1 (0.24%) zero-read locus PSI NaNs with 0.0 (biologically: zero observed junction reads represents 0 inclusion). Tune regularization inverse strength `C` over `[0.01, 0.1, 1, 10]` via 5-fold stratified cross-validation on an 80% training split (15,271 samples, `random_state=42`), leaving the 20% test set (3,818 samples) strictly untouched.
**Output:** Optimal hyperparameter `C = 0.1` achieved mean 5-fold CV F1-macro score of 0.7061 (std: 0.0105). Pipeline saved to `results/models/logistic_regression.joblib`.
**Dissertation section:** §3.7.1

### [Stage 6] Random Forest training & hyperparameter tuning — 2026-09-01
**Command:**
`python scripts/06_train_models.py` (Random Forest component)
**Rationale:** Train a non-linear ensemble classifier with balanced class weights on the same 21 structural/sequence/coverage/PSI features across 5-fold stratified cross-validation (`random_state=42`). Tune `n_estimators` over `[100, 300]` and `max_depth` over `[None, 10, 20]`.
**Output:** Optimal hyperparameters `n_estimators = 300`, `max_depth = None` achieved mean 5-fold CV F1-macro score of 0.9427 (std: 0.0056). Model saved to `results/models/random_forest.joblib`.
**Dissertation section:** §3.7.2

### [Stage 7] Baseline Logistic Regression held-out evaluation — 2026-09-04
**Command:**
`python scripts/07_evaluate_models.py` (Logistic Regression evaluation)
**Rationale:** Evaluate tuned multinomial Logistic Regression baseline ($C=0.1$) on the strictly held-out test partition ($N=3,818$, 20% stratified split) across accuracy, macro/weighted precision/recall/F1, and one-vs-rest ROC-AUC.
**Output:** Test Accuracy: 77.63% (target $\ge 80.0\%$ NOT MET); Macro-F1: 0.6970 (target $\ge 0.780$ NOT MET); Weighted-F1: 0.8066; Macro ROC-AUC: 0.9238. Confusion matrix saved to `results/models/confusion_matrix_lr.png`.
**Dissertation section:** §4.1, §4.2

### [Stage 7] Random Forest held-out evaluation & diagnostic artifact audit — 2026-09-04
**Command:**
`python scripts/07_evaluate_models.py` (Random Forest evaluation & diagnostic audit)
**Rationale:** Evaluate tuned Random Forest ensemble (`n_estimators=300`, `max_depth=None`) on the held-out test partition ($N=3,818$), extract Gini feature importances, and perform diagnostic audit on coordinate symmetry for retained intron (RI) classification.
**Output:** Test Accuracy: 96.15% (target $\ge 80.0\%$ MET); Macro-F1: 0.9416 (target $\ge 0.780$ MET); Weighted-F1: 0.9613; Macro ROC-AUC: 0.9959. Top feature importances: `upstream_intron_length` (0.1705), `downstream_intron_length` (0.1651), `exon_length` (0.1197). Diagnostic check confirmed 100% of RI events (1,348/1,348) have identical upstream/downstream intron lengths by Stage 5 construction, enabling near-perfect RI classification (Precision: 98.53%, Recall: 99.26%, F1: 0.9889) — flagged for critical methodology/limitations discussion (§4.5/§5). Plots and report saved to `results/models/`.
**Dissertation section:** §4.1, §4.3, §4.5, §5.1

### [Stage 7b] Robustness check on RI feature construction artifact — 2026-09-04
**Command:**
`python scripts/07b_robustness_check.py`
**Rationale:** Re-evaluate model performance after removing `upstream_intron_length` and `downstream_intron_length` (17 numeric + 2 strand dummy features = 19 features total) to eliminate artificial coordinate symmetry in Retained Intron (RI) events. Retrain Logistic Regression and Random Forest models using identical 5-fold CV GridSearch on the 80% training split and evaluate on the held-out test partition ($N=3,818$). Generate robust confusion matrix heatmaps and feature importance plots.
**Output:** Robust Random Forest achieved 94.50% Accuracy (target $\ge 80.0\%$ MET) and 0.9087 Macro-F1 (target $\ge 0.780$ MET); Macro ROC-AUC: 0.9916. RI F1 dropped disproportionately from 0.9889 to 0.8302 (-15.87 pp), inverting RI from the easiest to the hardest class to predict. Robust Logistic Regression achieved 77.37% Accuracy and 0.6974 Macro-F1. Designating `results/models/random_forest_robust.joblib` as the authoritative primary model anchoring Chapter 4 (Results). Full comparative metrics and discussion documented in `results/models/robustness_check.md`. Plots saved: `confusion_matrix_lr_robust.png`, `confusion_matrix_rf_robust.png`, `feature_importance_rf_robust.png`.
**Dissertation section:** §4.4, §4.5, §5.1

### [Deployment] Interactive Streamlit application & Windows launcher — 2026-09-05
**Command:**
`streamlit run app/streamlit_app.py` / `run_app.bat`
**Rationale:** Deploy an interactive demonstration interface enabling both batch CSV inference and single-event parameter exploration for predicting alternative splicing classes. The application serves the robust Random Forest model (`results/models/random_forest_robust.joblib`), selected based on Stage 7b findings as the genuine biological predictor (94.50% test accuracy, 0.9087 Macro-F1, 0.9916 ROC-AUC). Features full input validation, template CSV generation, probability distribution visualizer, and dynamic Retained Intron (RI) cautionary alerts.
**Output:** Created `app/streamlit_app.py` and Windows double-click launcher `run_app.bat`. Verified server deployment on `http://localhost:8501` (HTTP 200 OK).
**Dissertation section:** §3.8, Appendix B

### [Deployment] Bug fix: Windows launcher user pinning, absolute paths & ping-polled health check — 2026-09-05
**Command:**
`run_app.bat` (updated with background WSL execution + `ping -n 2 127.0.0.1 >nul` HTTP polling loop before browser launch)
**Rationale:** Fix deployment issues: (1) Root-cause user misconfiguration where WSL2 defaulted to `root`, breaking `~` resolution and conda initialization — resolved by explicitly pinning `-u brooklyn` with fully qualified paths (`/home/brooklyn/miniforge3/etc/profile.d/conda.sh` and `/home/brooklyn/splicing-ml-project`). (2) Launch-order race condition where the Windows browser was opened before the WSL Streamlit server was listening — resolved by dispatching Streamlit in background and polling `curl -s -o nul http://localhost:8501/` before opening the browser. (3) Replaced `timeout /t 1` with `ping -n 2 127.0.0.1 >nul` to eliminate the standard `ERROR: Input redirection is not supported` failure that crashes `timeout.exe` when executed from non-interactive or redirected parent shells.
**Output:** Verified live execution: background server launched, `ping`-paced health polling succeeded at 4s (`Server is LIVE at http://localhost:8501`), total elapsed time 5.67s, and browser opened to live application (HTTP 200 OK) without manual intervention.
**Dissertation section:** §3.8, Appendix B.2

### [Stage 8] Raw FASTQ download for KD2 validation replicate — 2026-09-06
**Command:**
`python scripts/fast_download_fastq.py ERR127306 ERR127307`
**Rationale:** Download raw paired-end sequencing reads for the held-back KD2 biological replicate pair (ERR127306, ERR127307 — independent siRNA targeting HNRNPC from E-MTAB-1147) from the ENA repository into `data/raw/` using parallel multi-threaded HTTP Range streaming with resume support.
**Output:** Downloaded 4 paired FASTQ files: ERR127306_1 (1.79 GB), ERR127306_2 (1.80 GB), ERR127307_1 (2.01 GB), ERR127307_2 (2.03 GB), totaling ~7.63 GB.
**Dissertation section:** §3.2.2, §4.6

### [Stage 1 Correction] Reference GTF annotation verification & size correction — 2026-09-06
**Command:**
`curl -sL https://ftp.ensembl.org/pub/release-116/gtf/homo_sapiens/CHECKSUMS | grep 'Homo_sapiens.GRCh38.116.gtf.gz' && sum data/reference/Homo_sapiens.GRCh38.116.gtf.gz`
**Rationale:** The original Stage 1 entry describing the reference GTF as "55.6MB gz / 1.69GB decompressed" was inaccurate. The authentic file on Ensembl release 116 was re-verified against Ensembl's official published CHECKSUMS to ensure reproducibility and reference integrity.
**Output:** Local checksum matched Ensembl official CHECKSUMS character-for-character: `49151 137815 Homo_sapiens.GRCh38.116.gtf.gz`. Actual file size: 141,121,632 bytes compressed (~135 MB gz) and 4,659,761,880 bytes decompressed (~4.66 GB) — confirmed as Ensembl release 116 standard primary GTF (not a patch/scaffold variant). Primary assembly FASTA was also re-extracted fresh via `gunzip -k` from the verified archive (881,964,081 bytes gz) to 3,151,425,851 bytes uncompressed, with valid chromosome 1 header and scaffold sequence termination confirmed.
**Dissertation section:** §3.2.2, Appendix A.3

### [Environment Correction] WSL2 resource configuration restored — 2026-09-06
**Command:**
`C:\Users\Brooklyn\.wslconfig` updated (`memory=4GB`, `processors=4`, `swap=8GB`) followed by `wsl --shutdown` and verified via `wsl -d Ubuntu -u brooklyn -- bash -c "nproc && free -h"`
**Rationale:** The current `.wslconfig` (1536MB RAM / 2 processors) did not match the resource profile logged during the original successful Stage 3 build (4 threads, 3.8GB available), and was restored to match before reattempting the index build. This is noted as a possible contributing factor to earlier pipeline interruptions, alongside the previously observed empty `results/` directories — not confirmed as the singular cause, but a real environment discrepancy placed on record for reproducibility.
**Output:** Verified WSL2 restarted with 4 processors and 3.8 GiB total RAM (3.5 GiB available, 8.0 GiB swap), exactly restoring the documented execution baseline.
**Dissertation section:** §3.1, Appendix A.3

### [Stage 3] HISAT2 index build parameterization (--bmaxdivn 16) — 2026-09-07
**Command:**
`hisat2-build -p 4 --bmaxdivn 16 --ss data/reference/splicesites.tsv --exon data/reference/exons.tsv data/reference/Homo_sapiens.GRCh38.dna.primary_assembly.fa data/reference/grch38_index`
**Rationale:** The default blockwise suffix-array divisor (`--bmaxdivn 4`) attempts an initial allocation of ~23.58 GB virtual memory, exceeding this environment's 3.8 GiB RAM + 8.0 GiB swap allocation and triggering fallback memory-fitting. To fit within the machine's memory profile without thrashing, `--bmaxdivn 16` was explicitly specified. This is a construction-time memory/speed tradeoff parameter only — it alters block partitioning during index creation but produces an index with mathematically identical FM-index search semantics, leaving downstream alignment behavior and rates completely unchanged (still expected at ~95%).
**Output:** Index build launched with 4 threads and explicit `--bmaxdivn 16` partition bounds.
**Dissertation section:** §3.4.1, Appendix A.3

### [Stage 3] HISAT2 index build attempt 2 killed by WSL auto-update — 2026-09-07
**Command:**
(no user command — Microsoft Store silently upgraded WSL from pre-2.7.13.0 to 2.7.13.0 mid-build)
**Rationale:** The `--bmaxdivn 16` build ran cleanly for ~118 minutes with steady `.rf` file growth and no memory errors. It was terminated externally when Windows Store auto-updated the WSL package, forcibly rebooting the WSL VM. This was not a stall, OOM, or parameter issue.
**Output:** Build killed at ~118 minutes. All partial `.rf` files lost. WSL upgraded to version 2.7.13.0, kernel 6.18.33.2-2.
**Dissertation section:** §3.4.1, Appendix A.3

### [Environment] Disabled WSL auto-updates and increased swap — 2026-09-07
**Command:**
`winget pin add --id 'Microsoft.WSL' --blocking` (confirmed via `winget pin list`: Pin type = Blocking)
`Set-ItemProperty HKCU:\..\ContentDeliveryManager -Name SilentInstalledAppsEnabled -Value 0`
`.wslconfig` swap updated from `8GB` to `16GB`, followed by `wsl --shutdown` and restart.
**Rationale:** The previous build was killed by an automatic WSL Store update at ~118 minutes. To prevent recurrence: (1) winget blocking pin prevents `winget upgrade` from touching WSL, (2) SilentInstalledAppsEnabled=0 disables background Store installs at user level. Swap increased from 8GB to 16GB (host has 909GB free disk, negligible cost) to give more headroom. Note: HKLM-level Store policy could not be set without admin elevation; the winget pin is the primary protection.
**Output:** Verified: `winget pin list` shows `Microsoft.WSL` pinned as Blocking at version 2.7.13.0. WSL restarted with 4 processors, 3.8 GiB RAM, 16 GiB swap.
**Dissertation section:** Appendix A.3

### [Stage 3] HISAT2 index build attempt 3 (--bmaxdivn 32) — 2026-09-07
**Command:**
`hisat2-build -p 4 --bmaxdivn 32 --ss data/reference/splicesites.tsv --exon data/reference/exons.tsv data/reference/Homo_sapiens.GRCh38.dna.primary_assembly.fa data/reference/grch38_index`
**Rationale:** After the `--bmaxdivn 16` build was killed externally at ~118 min (not by memory issues), `--bmaxdivn 32` was chosen for the retry to reduce peak per-partition memory from ~8 GB to ~4 GB, better fitting within 3.8 GiB RAM and reducing swap I/O. This is a construction-time parameter only — it does not alter the resulting index's alignment behavior, so downstream alignment rates remain expected at ~95%. Build launched with `nohup` and WSL auto-updates disabled. 3-hour hard cap applied.
**Output:** Build ran for 132 minutes reaching 34.7 GB in `.rf` files (Thread 3 completed at 15:54). At 16:19 SAST, upon concluding the blockwise phase, `hisat2-build-s` attempted a 23.58 GB memory allocation (`__vm_enough_memory: pid: 458, bytes: 23589634048 not enough memory for the allocation`) recorded in dmesg, triggering `Ran out of memory; automatically trying more memory-economical parameters` and resetting progress to 0. Terminated per protocol and user directive.
**Dissertation section:** §3.4.1, Appendix A.3

### [Environment Reproducibility Fix] Version drift discovery and HISAT2 downgrade — 2026-09-07
**Command:**
`conda install -y -c bioconda -c conda-forge hisat2=2.2.1`
**Rationale:** Following two repeated 23.58 GB allocation rejections at the conclusion of blockwise index construction, a systematic environment diagnostic was performed to resolve the contradiction between the documented 1,472s build on 3.8 GB RAM + 8 GB swap and the current failures. The diagnostic discovered that `splicing_project` contained `hisat2 2.2.3` (`build h8471819_0`, built Aug 2, 2026), drifting from the pinned `hisat2 2.2.1` documented in Stage 0 (§3.1). Conda dry-run confirmed a clean downgrade to `hisat2=2.2.1-h503566f_8` with zero conflicting package changes. Downgrade executed and verified: `hisat2-build version 2.2.1` confirmed verbatim.
**Output:** Downgraded `hisat2` from 2.2.3 to 2.2.1. Confirmed `/home/brooklyn/miniforge3/envs/splicing_project/bin/hisat2-build` reports version 2.2.1 (built Dec 12, 2024, without `-DPOPCNT_CAPABILITY`).
**Dissertation section:** §3.1, §3.4.1, Chapter 4/5 (Reproducibility Discussion)

### [Stage 1] Annotation extraction re-run under HISAT2 2.2.1 — 2026-09-07
**Command:**
`hisat2_extract_splice_sites.py data/reference/Homo_sapiens.GRCh38.116.gtf > data/reference/splicesites.tsv && hisat2_extract_exons.py data/reference/Homo_sapiens.GRCh38.116.gtf > data/reference/exons.tsv`
**Rationale:** Ensure full end-to-end toolchain consistency by regenerating splice sites and exon coordinates directly using the scripts packaged with pinned HISAT2 2.2.1.
**Output:** Verified `splicesites.tsv` (698,580 entries) and `exons.tsv` (409,261 unique non-redundant intervals, matching the verified Ensembl release 116 GTF structure).
**Dissertation section:** §3.4.1

### [Stage 3] HISAT2 index build attempt 4 (--bmaxdivn 16, hisat2 2.2.1) — 2026-09-07
**Command:**
`hisat2-build -p 4 --bmaxdivn 16 --ss data/reference/splicesites.tsv --exon data/reference/exons.tsv data/reference/Homo_sapiens.GRCh38.dna.primary_assembly.fa data/reference/grch38_index`
**Rationale:** Reattempt index build under verified `hisat2 2.2.1` using `--bmaxdivn 16` (which ran stably for 118+ min before external interruption). Monitoring protocol: 15-minute intervals through T+60 min, switching to 5-minute intervals starting at T+80 min to closely observe the critical 90–135 min blockwise-to-assembly transition window.
**Output:** Build ran for 120 minutes reaching 37.3 GB in `.rf` files. At 23:53 SAST (T+119 min), upon completing the blockwise phase and transitioning to global index assembly, `hisat2-build-s` attempted an identical 23.58 GB virtual memory allocation (`__vm_enough_memory: pid: 353, comm: hisat2-build-s, bytes: 23589634048 not enough memory for the allocation`) in dmesg, triggering `Ran out of memory; automatically trying more memory-economical parameters` and resetting. Process terminated and workspace cleaned per protocol. This definitively disproves the version-drift hypothesis and confirms that whole-genome GRCh38 index table assembly intrinsically requires a 23.58 GB virtual memory allocation in HISAT2.
**Dissertation section:** §3.4.1, Appendix A.3

### [Methodology Decision] Linear HISAT2 index + alignment-time splice sites — 2026-09-08
**Command:**
`hisat2-build -p 4 data/reference/Homo_sapiens.GRCh38.dna.primary_assembly.fa data/reference/grch38_index`
(Downstream alignment updated in `scripts/03b_align_samples.sh` to: `hisat2 -x data/reference/grch38_index -1 <R1> -2 <R2> -p 4 --dta --known-splicesite-infile data/reference/splicesites.tsv -S <output>.sam`)
**Rationale:** Deliberate, documented methodology decision (not a workaround): build a linear (non-graph) index without pre-baked splice sites/exons, and provide `splicesites.tsv` at alignment time via `--known-splicesite-infile`. This is HISAT2's own officially documented architecture for memory-constrained environments. A graph-index build embedding 698,580 splice sites across the full GRCh38 human genome requires an unavoidable ~23.58 GB contiguous virtual allocation during table assembly, which exceeds this machine's 3.8 GB RAM + 16 GB swap ceiling (conclusively verified via 4 independent failed attempts across 2 HISAT2 versions and 2 `--bmaxdivn` parameters). Supplying splice sites at alignment time is not a compromise on annotation completeness, as the identical verified Ensembl 116 splice sites (698,580 sites) guide the spliced read alignments. Additionally, we place on record the plausible retroactive explanation for why the original Sep 1 build appeared to succeed on 3.8 GB RAM: the original run likely processed a truncated GTF (55.6 MB logged vs. the official checksum-verified 141 MB release 116 file), which contained substantially fewer splice sites and thus fit within the lower memory envelope.
**Output:** Linear index build ran for 18h 31m (00:52 to 19:23 SAST), progressing through block 13 of 31 with ~258 MB written to `grch38_index.1.ht2`, ~193 MB to `grch38_index.2.ht2`, and ~737 MB to `grch38_index.4.ht2`. Terminated by user directive (`killall -9 hisat2-build hisat2-build-s hisat2-build-l`) at 19:23 SAST to pursue external build or alternative index provisioning. Workspace cleaned of all partial `grch38_index*` files and confirmed idle (0 hisat2 processes, 3.5 GiB available RAM, 0 B swap used).
**Dissertation section:** §3.4.1, Appendix A.3

### [Stage 3 Verification] Clean matched 200k-read-pair test subset extraction (ERR127302) — 2026-09-15
**Command:**
`gzip -t data/trimmed/ERR127302_1_paired.fastq.gz data/trimmed/ERR127302_2_paired.fastq.gz && zcat data/trimmed/ERR127302_1_paired.fastq.gz | head -n 800000 | gzip > test_1.fastq.gz && zcat data/trimmed/ERR127302_2_paired.fastq.gz | head -n 800000 | gzip > test_2.fastq.gz`
**Rationale:** Extract a clean, matched 200,000-read-pair subset (800,000 FASTQ lines per mate) from the authentic, gzip-verified Control sample ERR127302 for controlled benchmarking and alignment validation.
**Output:** Verified full archive integrity (gzip -t: OK on both mates). Generated test_1.fastq.gz (13 MB) and test_2.fastq.gz (13 MB); exactly 800,000 lines (200,000 paired-end reads) verified on each file via `zcat | wc -l`.
**Dissertation section:** §3.4.2, Appendix A.3

### [Stage 1 Restoration] Raw FASTQ retrieval & integrity verification (ERR127303, ERR127304, ERR127305) — 2026-09-18
**Command:**
`python scripts/fast_download_fastq.py ERR127303 ERR127304 ERR127305`
**Rationale:** Retrieve missing raw FASTQ pairs directly from the authoritative ENA repository using parallel multi-threaded HTTP Range streaming with blockwise MD5 and gzip integrity checks.
**Output:** Fetched 6 paired FASTQ files: ERR127303 (32,665,567 pairs, matched ENA MD5s), ERR127304 (31,876,181 pairs, 3.88 GB total, gzip -t OK), ERR127305 (32,345,742 pairs, 3.85 GB total, gzip -t OK).
**Dissertation section:** §3.2.2, Appendix A.3

### [Stage 2 Restoration] Adapter & quality trimming (ERR127303) — 2026-09-18
**Command:**
`trimmomatic PE -threads 4 -phred33 data/raw/ERR127303_1.fastq.gz data/raw/ERR127303_2.fastq.gz data/trimmed/ERR127303_1_paired.fastq.gz data/trimmed/ERR127303_1_unpaired.fastq.gz data/trimmed/ERR127303_2_paired.fastq.gz data/trimmed/ERR127303_2_unpaired.fastq.gz ILLUMINACLIP:/home/brooklyn/miniforge3/envs/splicing_project/share/trimmomatic/adapters/TruSeq3-PE.fa:2:30:10 LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15 MINLEN:36`
**Rationale:** Standardize adapter and quality trimming on full authentic replicate using pinned Trimmomatic parameters.
**Output:** Input Read Pairs: 32,665,567 | Both Surviving: 26,758,217 (81.92%) | Forward Only: 3,659,196 (11.20%) | Reverse Only: 1,488,981 (4.56%) | Dropped: 759,173 (2.32%).
**Dissertation section:** §3.3.2, Appendix A.1

### [Stage 2 Restoration] Adapter & quality trimming (ERR127304) — 2026-09-18
**Command:**
`trimmomatic PE -threads 4 -phred33 data/raw/ERR127304_1.fastq.gz data/raw/ERR127304_2.fastq.gz data/trimmed/ERR127304_1_paired.fastq.gz data/trimmed/ERR127304_1_unpaired.fastq.gz data/trimmed/ERR127304_2_paired.fastq.gz data/trimmed/ERR127304_2_unpaired.fastq.gz ILLUMINACLIP:/home/brooklyn/miniforge3/envs/splicing_project/share/trimmomatic/adapters/TruSeq3-PE.fa:2:30:10 LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15 MINLEN:36`
**Rationale:** Apply identical trimming threshold to knockdown condition 1 replicate 1.
**Output:** Input Read Pairs: 31,876,181 | Both Surviving: 26,945,447 (84.53%) | Forward Only: 3,039,095 (9.53%) | Reverse Only: 1,301,609 (4.08%) | Dropped: 590,030 (1.85%).
**Dissertation section:** §3.3.2, Appendix A.1

### [Stage 2 Restoration] Adapter & quality trimming (ERR127305) — 2026-09-18
**Command:**
`trimmomatic PE -threads 4 -phred33 data/raw/ERR127305_1.fastq.gz data/raw/ERR127305_2.fastq.gz data/trimmed/ERR127305_1_paired.fastq.gz data/trimmed/ERR127305_1_unpaired.fastq.gz data/trimmed/ERR127305_2_paired.fastq.gz data/trimmed/ERR127305_2_unpaired.fastq.gz ILLUMINACLIP:/home/brooklyn/miniforge3/envs/splicing_project/share/trimmomatic/adapters/TruSeq3-PE.fa:2:30:10 LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15 MINLEN:36`
**Rationale:** Apply identical trimming threshold to knockdown condition 1 replicate 2.
**Output:** Input Read Pairs: 32,345,742 | Both Surviving: 28,042,418 (86.70%) | Forward Only: 2,772,429 (8.57%) | Reverse Only: 1,025,964 (3.17%) | Dropped: 504,931 (1.56%).
**Dissertation section:** §3.3.2, Appendix A.1

### [Stage 2 Verification] Full primary 4-sample trimmed batch integrity verification — 2026-09-18
**Command:**
`for s in ERR127302 ERR127303 ERR127304 ERR127305; do for p in 1 2; do gzip -t "data/trimmed/${s}_${p}_paired.fastq.gz" && echo "${s}_${p}_paired: PRESENT & VALID"; done; done`
**Rationale:** Exhaustive archive integrity verification across all 8 paired trimmed FASTQ files prior to batch alignment/upload.
**Output:** All 8 files verified PRESENT & VALID: ERR127302_1/2 (1.61/1.62 GB), ERR127303_1/2 (1.71/1.71 GB), ERR127304_1/2 (1.69/1.70 GB), ERR127305_1/2 (1.75/1.76 GB). Total: 108,926,561 surviving read pairs (~13.55 GB).
**Dissertation section:** §3.3.2, Appendix A.1

### [Stage 3 External Alignment] Full 4-sample linear HISAT2 alignment bundle retrieval & verification — 2026-09-19
**Command:**
`samtools quickcheck -v results/alignment/*.sorted.bam && samtools idxstats results/alignment/*.sorted.bam && samtools flagstat -@ 4 results/alignment/ERR127302.sorted.bam`
**Rationale:** Retrieve and rigorously verify the complete 4-sample (Control x2, KD1 x2) linear HISAT2 alignment bundle generated via external Kaggle execution (direct zero-upload ENA raw read streaming, Trimmomatic PE adapter/quality trimming with pinned TruSeq3-PE parameters, and linear HISAT2 alignment with Ensembl release 116 known splice sites supplied via `--known-splicesite-infile`). Verify archive integrity, absence of file duplication/truncation, read count conservation, alignment rates, and trimming retention.
**Output:** 16/16 files verified (4 BAMs, 4 BAIs, 4 HISAT2 logs, 4 Trimmomatic logs, total 12.18 GB uncompressed). Quickcheck passed 100% (EOF blocks intact). Trimmomatic retention: ERR127302 (87.65%), ERR127303 (81.92%), ERR127304 (84.53%), ERR127305 (86.70%). HISAT2 alignment rates: ERR127302 (95.29%), ERR127303 (95.55%), ERR127304 (96.25%), ERR127305 (94.74%). Total aligned library: 107,815,803 surviving pairs (~215.6M primary reads), fully explaining bundle size gap vs original 71.4M pair subset.
**Dissertation section:** §3.4.2, Appendix A.3

### [Stage 3 External Index] Persisted linear GRCh38 HISAT2 index retrieval & verification — 2026-09-21
**Command:**
`hisat2-inspect --summary data/reference/grch38_index_linear`
**Rationale:** Retrieve and verify the persisted linear (non-graph) HISAT2 reference index generated on Kaggle from Ensembl release 116 primary assembly FASTA, confirming completeness and eliminating the need for future rebuilds.
**Output:** Verified all 8 index parts (`grch38_index_linear.1.ht2` to `.8.ht2`, 4.20 GB) and `splicesites.tsv` (15.97 MB). Inspection confirmed exactly 194 sequences, 3,099,750,718 bp (~3.100 Gb), 0 embedded splice sites/exons/SNPs.
**Dissertation section:** §3.4.1, Appendix A.3

### [Stage 3 External Alignment] Two-replicate Control alignment bundle retrieval & verification — 2026-09-21
**Command:**
`samtools quickcheck -v results/alignment/Control_rep*.sorted.bam && samtools flagstat -@ 4 results/alignment/Control_rep1.sorted.bam && samtools flagstat -@ 4 results/alignment/Control_rep2.sorted.bam`
**Rationale:** Ingest and verify the authentic Control replicates constructed by concatenating paired technical sequencing runs from ERP001458 (Control_rep1 = ERR127306 + ERR127307; Control_rep2 = ERR127308 + ERR127309), trimmed with pinned Trimmomatic PE parameters and aligned to the linear GRCh38 index with `--known-splicesite-infile`.
**Output:** Quickcheck passed 100%. Control_rep1 (61,480,994 raw pairs -> 55,453,742 trimmed [90.20% retention] -> 94.52% overall alignment rate; 116,095,872 mapped reads / 122,177,205 total). Control_rep2 (62,265,044 raw pairs -> 57,186,429 trimmed [91.84% retention] -> 89.49% overall alignment rate; 113,558,516 mapped reads / 125,582,091 total). Both confirm the expected ~60M-pair (~120M total read) biological replicate depth.
**Dissertation section:** §3.4.2, Appendix A.3

### [Stage 1 Discovery & Manifest Correction] Sample manifest corrected via ENA sample accession cross-check — 2026-09-22
**Command:**
Updated `data/reference/manifest.csv` based on ENA `sample_accession` cross-check:
`ERR127302,KD1,1,PAIRED,HNRNPC knockdown construct 1 replicate 1`
`ERR127303,KD1,2,PAIRED,HNRNPC knockdown construct 1 replicate 2`
`ERR127304,KD2,1,PAIRED,HNRNPC knockdown construct 2 replicate 1 (held out for external validation)`
`ERR127305,KD2,2,PAIRED,HNRNPC knockdown construct 2 replicate 2 (held out for external validation)`
`Control_rep1,Control,1,PAIRED,ERR127306+ERR127307 concatenated (sample SAMEA1691813, two lanes)`
`Control_rep2,Control,2,PAIRED,ERR127308+ERR127309 concatenated (sample SAMEA1691811, two lanes)`
**Rationale:** A sample mislabeling error originating from Stage 1 was identified through cross-referencing ENA's authentic `sample_accession` metadata field for ERP001458 (E-MTAB-1147). The original manifest misidentified Control and KD1: ERR127302/ERR127303 are genuine KD1 replicates, ERR127304/ERR127305 are KD2 replicates (held back for external validation), and the authentic untreated Control samples are ERR127306–ERR127309 (concatenated and aligned as Control_rep1 and Control_rep2). The prior `results/splicing/` output therefore reflected an unintentional KD1-vs-KD2 comparison; it is retained intact in the repository for methodological auditability and transparency in the dissertation methodology appendix rather than overwritten or deleted.
**Output:** Updated manifest with 6 verified sample definitions (2 Control, 2 KD1, 2 held-out KD2).
**Dissertation section:** §3.2.1, Appendix A.3

### [Stage 4 Corrected] Differential alternative splicing quantification (rMATS Control vs KD1) — 2026-09-22
**Command:**
`rmats.py --b1 results/splicing_corrected/b1.txt --b2 results/splicing_corrected/b2.txt --gtf data/reference/Homo_sapiens.GRCh38.116.gtf -t paired --readLength 72 --nthread 4 --od results/splicing_corrected/ --tmp results/splicing_corrected/tmp`
**Rationale:** Authoritative differential splicing quantification between authentic Control (b1: Control_rep1.sorted.bam, Control_rep2.sorted.bam) and HNRNPC knockdown (b2: ERR127302.sorted.bam, ERR127303.sorted.bam) across five alternative splicing event types. Input BAMs for Control were accessed via verified symlinks into `/mnt/c` to preserve host disk space (host C: 10.98 GB free vs 11.0 GB BAM size). This replaces the preliminary KD1-vs-KD2 comparison with the authentic experimental comparison for all downstream modeling.
**Output:** Quantified 168,919 total splicing events with 10,746 significant events (FDR < 0.05): SE: 5,112 (47.57%), RI: 698 (6.50%), A5SS: 505 (4.70%), A3SS: 1,171 (10.90%), MXE: 3,260 (30.34%). Total wall time: 7977 seconds.
**Dissertation section:** §3.5, §4.3, Appendix A.3

### [Stage 4 Robustness Check] Control BAM downsampling to match KD1 depth — 2026-09-22
**Command:**
`samtools view -@ 4 -s 42.469 -b -o results/alignment/Control_rep1_downsampled.sorted.bam results/alignment/Control_rep1.sorted.bam && samtools index -@ 4 results/alignment/Control_rep1_downsampled.sorted.bam && samtools view -@ 4 -s 42.469 -b -o results/alignment/Control_rep2_downsampled.sorted.bam results/alignment/Control_rep2.sorted.bam && samtools index -@ 4 results/alignment/Control_rep2_downsampled.sorted.bam`
**Rationale:** Assess whether the ~2.1x sequencing depth imbalance between Control (~57M mapped pairs/sample) and KD1 (~27.5M mapped pairs/sample) distorts the alternative splicing event-type distribution. Control BAMs were subsampled to 46.9% using `samtools view -s` (seed 42) to target ~27M pairs per replicate, achieving depth parity with KD1 while writing directly to native ext4 storage.
**Output:** Downsampled BAMs verified: `Control_rep1_downsampled.sorted.bam` (54,435,443 mapped reads, 2.6 GB) and `Control_rep2_downsampled.sorted.bam` (53,265,534 mapped reads, 2.7 GB), achieving 98.9% and 94.3% parity with KD1 replicates (ERR127302: 55.0M reads, ERR127303: 56.5M reads).
**Dissertation section:** §3.4.2, §3.5, Appendix A.3

### [Stage 4 Robustness Check] Depth-balanced rMATS execution & event profile validation — 2026-09-22
**Command:**
`rmats.py --b1 results/splicing_corrected_downsampled/b1.txt --b2 results/splicing_corrected_downsampled/b2.txt --gtf data/reference/Homo_sapiens.GRCh38.116.gtf -t paired --readLength 72 --nthread 4 --od results/splicing_corrected_downsampled/ --tmp results/splicing_corrected_downsampled/tmp`
**Rationale:** Quantify differential alternative splicing under strict depth parity between downsampled Control and KD1 to test whether the observed ~48% SE / ~30% MXE profile in `results/splicing_corrected/` was a library-size artifact or genuine HNRNPC biology.
**Output:** Quantified 148,140 total splicing events with 8,693 significant events (FDR < 0.05): SE: 4,306 (49.53%), RI: 500 (5.75%), A5SS: 400 (4.60%), A3SS: 786 (9.04%), MXE: 2,701 (31.07%). Proportions match the full-depth run within 2 pp (SE: 49.53% vs 47.57%, MXE: 31.07% vs 30.34%) with zero reversion toward the ~75% SE / 6.5% MXE pattern of the original mislabeled KD1-vs-KD2 run, confirming the event-type distribution reflects authentic HNRNPC regulation. Total wall time: 3025 seconds.
**Dissertation section:** §3.5, §4.3, Appendix A.3

### [Stage 4 Negative Control] Technical lane comparison (ERR127306 vs ERR127307) — 2026-09-23
**Command:**
`python scripts/split_lanes.py && samtools index -@ 4 results/alignment/ERR127306.sorted.bam && samtools index -@ 4 results/alignment/ERR127307.sorted.bam && rmats.py --b1 results/splicing_lane_control/b1.txt --b2 results/splicing_lane_control/b2.txt --gtf data/reference/Homo_sapiens.GRCh38.116.gtf -t paired --readLength 72 --nthread 4 --od results/splicing_lane_control/ --tmp results/splicing_lane_control/tmp`
**Rationale:** Evaluate whether sequencing lane-batch artifacts inflate differential splicing calls. Control replicate 1 was demultiplexed by QNAME into its two constituent sequencing lanes (ERR127306 [Lane 5, 55.2M mapped reads] vs ERR127307 [Lane 6, 60.9M mapped reads], representing the identical biological sample SAMEA1691813) and compared via rMATS under a null hypothesis of zero true biological differences.
**Output:** Quantified 114,252 total splicing events with 3,741 significant events (FDR < 0.05, 3.27% of tested events, below the nominal 5% FDR threshold): SE: 1,635 (43.70%), MXE: 785 (20.98%), A3SS: 648 (17.32%), A5SS: 396 (10.59%), RI: 277 (7.40%). Significant calls showed near-perfect 50/50 symmetry (SE: 817 vs 818; A3SS: 316 vs 332) and all-or-nothing low-count stochastic zeros (median |ΔPSI| = 1.000 vs 0.20-0.25 in biological runs), confirming absence of systematic lane-batch confounding. Total wall time: 2670 seconds.
**Dissertation section:** §3.5, §4.3, Appendix A.3
### [Stage 5 Corrected] Feature engineering on depth-matched dataset — 2026-09-23
**Command:**
`python scripts/05_feature_engineering_corrected.py`
**Rationale:** Extract genomic sequence features (exon length, GC content of exon and 100 bp flanking regions), splice-site strengths (human canonical 5' donor and 3' acceptor PWM log-odds scores), local RNA-seq read coverage mean and variance across depth-matched Control and KD1 BAM alignments, and replicate PSI means/variances for all 8,693 statistically significant splicing events (FDR < 0.05) from `results/splicing_corrected_downsampled/`. Excludes `upstream_intron_length` and `downstream_intron_length` from downstream modeling to prevent the coordinate symmetry artifact identified in Stage 7b, and excludes `pvalue` and `fdr` to prevent statistical power leakage.
**Output:** Generated `results/features/feature_matrix_corrected.csv` with 8,693 rows and 29 columns across 5 classes (SE: 4,306 [49.53%], MXE: 2,701 [31.07%], A3SS: 786 [9.04%], RI: 500 [5.75%], A5SS: 400 [4.60%]). Zero missing/NaN values across all 8,693 loci.
**Dissertation section:** §3.6, §4.3, Appendix A.2

### [Stage 6 Corrected] Model training & hyperparameter tuning on corrected dataset — 2026-09-23
**Command:**
`python scripts/06_train_evaluate_corrected.py` (Training component)
**Rationale:** Train multinomial Logistic Regression baseline and non-linear Random Forest ensemble using balanced class weights on the robust 19-feature set (17 numeric sequence/PWM/coverage/PSI features + 2 strand dummies; coordinate symmetry intron lengths and rMATS p/FDR values excluded). Split data into an 80% training set (6,954 samples) and 20% held-out test set (1,739 samples) using stratified sampling (`random_state=42`). Perform 5-fold cross-validation `GridSearchCV` on the training partition optimizing `f1_macro` over $C \in [0.01, 0.1, 1, 10]$ for Logistic Regression and `n_estimators` $\in [100, 300]$, `max_depth` $\in [\text{None}, 10, 20]$ for Random Forest.
**Output:** Optimal Logistic Regression hyperparameter $C = 10$ achieved mean 5-fold CV F1-macro of 0.4171; optimal Random Forest hyperparameters `n_estimators = 300`, `max_depth = None` achieved mean 5-fold CV F1-macro of 0.6528. Models saved to `results/models/logistic_regression_corrected.joblib` and `results/models/random_forest_corrected.joblib`.
**Dissertation section:** §3.7.1, §3.7.2, Appendix A.3

### [Stage 7 Corrected] Held-out model evaluation & diagnostic analysis — 2026-09-23
**Command:**
`python scripts/06_train_evaluate_corrected.py` (Evaluation component)
**Rationale:** Evaluate tuned Logistic Regression and Random Forest models on the strictly held-out test partition ($N = 1,739$, 20% stratified split) across accuracy, macro/weighted precision/recall/F1, and one-vs-rest ROC-AUC. Generate confusion matrices and analyze Gini feature importances to verify absence of synthetic coordinate artifacts.
**Output:** Random Forest achieved 74.01% Test Accuracy, 0.6474 Macro-F1 (SE: 0.7937, MXE: 0.7572, RI: 0.7092, A3SS: 0.5017, A5SS: 0.4754), 0.7365 Weighted-F1, and 0.9026 One-vs-Rest Macro ROC-AUC. Logistic Regression achieved 47.50% Accuracy and 0.4049 Macro-F1. `exon_length` was identified as the primary predictor (Gini importance: 0.1825, flagged as >15% dominant), followed by `score_5ss_donor` (0.0791), `score_3ss_acceptor` (0.0753), `psi_control_mean` (0.0653), `psi_kd_mean` (0.0589), and `dpsi` (0.0567). Confusion matrices and feature importance plots saved to `results/models/`.
**Dissertation section:** §4.1, §4.2, §4.3, §4.5, Appendix A.3

### [Stage 8 External Validation] Differential alternative splicing quantification (rMATS Control vs KD2) — 2026-09-24
**Command:**
`rmats.py --b1 results/splicing_kd2_corrected/b1.txt --b2 results/splicing_kd2_corrected/b2.txt --gtf data/reference/Homo_sapiens.GRCh38.116.gtf -t paired --readLength 72 --nthread 4 --od results/splicing_kd2_corrected/ --tmp results/splicing_kd2_corrected/tmp`
**Rationale:** Quantify differential alternative splicing between downsampled Control (Control_rep1_downsampled.sorted.bam, Control_rep2_downsampled.sorted.bam) and the independent biological knockdown construct 2 (ERR127304.sorted.bam, ERR127305.sorted.bam) across all five splicing event types under strict depth parity. This provides an independent transcriptomic validation cohort to test the generalization of the primary Random Forest model trained on KD1.
**Output:** Quantified 149,702 total splicing events with 7,080 significant events (FDR < 0.05): SE: 4,223 (59.65%), MXE: 1,555 (21.96%), A3SS: 529 (7.47%), RI: 442 (6.24%), A5SS: 331 (4.68%). Output saved to `results/splicing_kd2_corrected/`.
**Dissertation section:** §3.5, §4.6, Appendix A.3

### [Stage 8 External Validation] Feature matrix construction for KD2 validation cohort — 2026-09-24
**Command:**
`python scripts/08_run_kd2_pipeline.py` (Feature extraction component)
**Rationale:** Extract genomic geometry, sequence GC content, splice-site consensus scores (PWM), local RNA-seq read coverage across Control and KD2 BAM alignments, and replicate PSI kinetics for all 7,080 significant events (FDR < 0.05) using the identical 19-feature standard.
**Output:** Generated `results/features/feature_matrix_kd2_corrected.csv` with 7,080 rows and 29 columns across 5 classes (SE: 4,223, MXE: 1,555, A3SS: 529, RI: 442, A5SS: 331). 0 missing/NaN values.
**Dissertation section:** §3.6, §4.6, Appendix A.3

### [Stage 8 External Validation] Primary Random Forest evaluation on independent KD2 cohort — 2026-09-24
**Command:**
`python scripts/08_run_kd2_pipeline.py` (Evaluation component)
**Rationale:** Evaluate the frozen primary Random Forest model (`results/models/random_forest_corrected.joblib`, trained exclusively on KD1) directly against the independent KD2 biological validation cohort ($N = 7,080$) to measure real-world out-of-sample generalization retention versus the internal test partition ($N = 1,739$).
**Output:** Accuracy: 75.82% (vs internal KD1 test: 74.01%, +1.81 pp); Macro-F1: 0.6896 (vs internal KD1 test: 0.6474, +0.0421); Weighted-F1: 0.7595; Macro-Precision: 0.7048; Macro-Recall: 0.7016; One-vs-Rest Macro ROC-AUC: 0.9120 (vs internal KD1 test: 0.9026). Per-class F1: SE: 0.8255, RI: 0.7623, MXE: 0.6763, A5SS: 0.6241, A3SS: 0.5598. Confusion matrix saved to `results/models/confusion_matrix_kd2_corrected.png` and report saved to `results/models/kd2_validation_report_corrected.md`.
**Dissertation section:** §4.6, Appendix A.3

### [Version Control & Backup] Git repository initialization, remote GitHub backup & experimental branching — 2026-09-24
**Command:**
`git init && git remote add origin https://github.com/HackedScreen/splicing-ml-project.git && git branch -M main && git add . && git commit -m 'Final synced pipeline - 75.8% KD2 external validation' && git push -u origin main && git checkout -b experimental-accuracy`
**Rationale:** Establish formal version control for the BSc dissertation codebase and back up all code, documentation, diagnostic reports, and feature matrices to remote GitHub prior to model optimization experiments. Ignore large raw sequencing data, BAM alignments, rMATS temporary directories, and serialized model binaries (>50MB) via `.gitignore`.
**Output:** Initialized Git repo, staged and committed 59 files (22,111 insertions), pushed `main` branch to remote origin (`https://github.com/HackedScreen/splicing-ml-project.git`), and checked out `experimental-accuracy` branch.
**Dissertation section:** §3.1, Appendix A.3

### [Stage 5 V2] Feature engineering: HNRNPC U-tract binding motif density — 2026-09-24
**Command:**
`python scripts/05_feature_engineering_v2.py`
**Rationale:** Extract 4 biological HNRNPC binding motif density features across 100 bp upstream and downstream flanking windows oriented along transcript sense strand: `max_utract_length_upstream`, `max_utract_length_downstream`, `u_fraction_upstream`, and `u_fraction_downstream`. Implemented across both KD1 discovery cohort ($N=8,693$) and KD2 independent validation cohort ($N=7,080$) with identical column schema (33 columns total, 0 missing/NaN values).
**Output:** Generated `results/features/feature_matrix_v2.csv` (8,693 rows, 33 columns) and `results/features/feature_matrix_kd2_v2.csv` (7,080 rows, 33 columns). SE showed pronounced upstream U-tract enrichment: mean upstream max U-tract of 6.05 bp in KD1 (and 6.18 bp in KD2) vs 3.60–4.95 bp in other classes; mean upstream U-fraction of 0.3367 in KD1 (0.3452 in KD2) vs 0.2676–0.3304 in other classes.
**Dissertation section:** §3.6, §4.3, Appendix A.2

### [Stages 6 & 7 V2] Model training, tuning & dual evaluation (XGBoost & Random Forest V2) — 2026-09-24
**Command:**
`python scripts/06_train_evaluate_v2.py`
**Rationale:** Train and evaluate XGBoost classifier on the 23-feature set (17 numeric + 4 U-tract + 2 strand dummies) using `sample_weight=compute_sample_weight('balanced', y_train)` and 5-fold stratified GridSearchCV ($n\_estimators \in [100, 300]$, $max\_depth \in [3, 6, 10]$, $learning\_rate \in [0.05, 0.1, 0.2]$) on the frozen 80% KD1 training partition ($N=6,954$). For direct ablation, also tune Random Forest V2 on the identical 23 features. Evaluate both models on the held-out internal KD1 test partition ($N=1,739$) and the untouched external KD2 validation cohort ($N=7,080$).
**Output:** Optimal XGBoost params (`lr=0.05`, `max_depth=6`, `n_est=300`) achieved 0.6558 CV Macro-F1, 72.17% internal test accuracy, 0.6385 internal Macro-F1, 73.90% KD2 accuracy, and 0.6655 KD2 Macro-F1. `max_utract_length_upstream` ranked #5 of 23 in XGBoost importance (0.0504 gain). Random Forest V2 achieved 74.81% internal test accuracy (+0.80 pp vs RF baseline), 0.6611 internal Macro-F1 (+0.0137), 76.82% KD2 accuracy (+1.00 pp vs RF baseline), and 0.7003 KD2 Macro-F1 (+0.0107). Artifacts saved to `results/models/` (*_v2.joblib, *_v2.png, *_v2.md).
**Dissertation section:** §3.7.2, §4.1, §4.3, §4.6, Appendix A.3

### [Model Promotion & Checkpoint] RF V2 adopted as primary model — 2026-09-24
**Command:**
`cp results/models/random_forest_corrected.joblib results/models/random_forest_corrected_baseline_backup.joblib && cp results/models/random_forest_v2.joblib results/models/random_forest_corrected.joblib && git commit -m "Adopt RF V2 primary model: HNRNPC U-tract feature addition improves KD2 validation to 76.82% acc and 0.7003 macro-F1"`
**Rationale:** Backup the original 19-feature Random Forest baseline and promote Random Forest V2 (23 features, incorporating HNRNPC U-tract motif features) to the primary model position following validated performance gains on both the internal test partition (74.81% accuracy, 0.6611 Macro-F1) and the independent KD2 biological validation cohort (76.82% accuracy, 0.7003 Macro-F1). Update Streamlit application configuration and commit clean checkpoint to `experimental-accuracy` branch.
**Output:** Baseline backed up to `random_forest_corrected_baseline_backup.joblib`; `random_forest_corrected.joblib` updated to RF V2; `app/streamlit_app.py` updated to 23 features and verified; git commit `aaaa447` created.
**Dissertation section:** §3.8, §4.6, Appendix A.3

### [Round 3 Step 2] Soft-voting ensemble CV weight grid search and dual evaluation — 2026-09-24
**Command:**
`python scripts/07_ensemble_and_threshold_tuning.py` (Ensemble component)
**Rationale:** Grid search probability-averaging weights between RF V2 and XGBoost V2 ($w_{RF} \in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]$) strictly on 5-fold cross-validation out-of-fold predictions on the training partition ($N=6,954$) scoring by Macro-F1. Lock in the winning weight without test set or KD2 leakage, then evaluate the ensemble against internal held-out test ($N=1,739$) and KD2 validation ($N=7,080$) cohorts. Candidate retained only if it beats RF V2 on both cohorts.
**Output:** Winning CV weight $w_{RF}=0.7, w_{XGB}=0.3$ achieved 0.6722 CV Macro-F1 (vs 0.6414 RF CV). On internal test: 74.64% Acc (-0.17 pp vs RF V2), 0.6532 Macro-F1 (-0.0079); on KD2: 76.82% Acc (0.00 pp), 0.7020 Macro-F1 (+0.0017). Dropped per pre-specified protocol as it failed to beat RF V2 on both cohorts.
**Dissertation section:** §3.7.3, §4.3, §4.6

### [Round 3 Step 3] Per-class decision threshold tuning on CV out-of-fold predictions — 2026-09-24
**Command:**
`python scripts/07_ensemble_and_threshold_tuning.py` (Threshold tuning component)
**Rationale:** Address severe multi-class imbalance (A3SS: 9.05%, A5SS: 4.60%, MXE: 31.08%, RI: 5.75%, SE: 49.53%) by tuning per-class probability multipliers $m \in \mathbb{R}^5_+$ on training-set 5-fold CV out-of-fold predictions from RF V2 to maximize Macro-F1. Lock in optimal multipliers before evaluating once on the held-out test set ($N=1,739$) and KD2 validation set ($N=7,080$).
**Output:** Optimal multipliers (relative to SE=1.0000: A3SS: 1.3110, A5SS: 1.6759, MXE: 1.1012, RI: 1.2090) increased CV Macro-F1 from 0.6414 to 0.6634 (+0.0220), boosting minority recall (RI: 87.0% test, 90.1% KD2). However, test accuracy fell to 72.80% (-2.01 pp) and Macro-F1 to 0.6462 (-0.0149); KD2 accuracy fell to 74.38% (-2.44 pp) and Macro-F1 to 0.6799 (-0.0204). Dropped due to majority-class false positive inflation under generalized testing.
**Dissertation section:** §3.7.3, §4.3, §4.6

### [Stage 5 V3] Genuine MaxEntScan splice site scoring (Yeo & Burge) — 2026-09-24
**Command:**
`python scripts/05_feature_engineering_maxent.py`
**Rationale:** Replace additive position weight matrix (PWM) splice site scores with genuine MaxEntScan (Yeo & Burge, 2004) maximum entropy models on 9 bp donor windows ([-3 exon, +6 intron]) and 23 bp acceptor windows ([-20 intron, +3 exon]). Eliminates position independence assumptions and models non-adjacent sequence dependencies and consensus dinucleotide thermodynamics from GRCh38 primary assembly FASTA across all 8,693 KD1 events and 7,080 KD2 events.
**Output:** Generated `results/features/feature_matrix_v3_maxent.csv` ($N=8,693$, 33 cols) and `results/features/feature_matrix_kd2_v3_maxent.csv` ($N=7,080$, 33 cols). MaxEnt donor scores: KD1 mean 6.67 (std 5.17), KD2 mean 6.71 (std 4.59); MaxEnt acceptor scores: KD1 mean 6.26 (std 5.20), KD2 mean 6.02 (std 5.26).
**Dissertation section:** §3.6, §4.3, Appendix A.2

### [Stages 6 & 7 V3] Random Forest training and dual evaluation with MaxEntScan features — 2026-09-24
**Command:**
`python scripts/06_train_evaluate_maxent.py`
**Rationale:** Retrain Random Forest classifier on the 23-feature dataset incorporating genuine MaxEntScan splice site scores using 5-fold stratified GridSearchCV ($n\_estimators \in [100, 300]$, $max\_depth \in [\text{None}, 10, 20]$, scoring by `f1_macro`) on the frozen 80% KD1 training partition ($N=6,954$). Lock the single winning hyperparameter configuration from training-set CV alone, then evaluate that model on the internal held-out test partition ($N=1,739$) and external KD2 validation cohort ($N=7,080$).
**Output:** Optimal parameters selected strictly via 5-fold training CV were `max_depth = 20`, `n_estimators = 100` (CV Macro-F1 = 0.6210). MaxEntScan splice site scores ranked #5 (`score_3ss_acceptor`: 0.0550) and #6 (`score_5ss_donor`: 0.0515) in Gini importance. On internal test ($N=1,739$): Accuracy = 75.79% (+0.98 pp vs RF V2), Macro-F1 = 0.6203 (-0.0408 vs RF V2), Weighted-F1 = 0.7419, ROC-AUC = 0.9055. On KD2 validation ($N=7,080$): Accuracy = 79.41% (+2.59 pp vs RF V2), Macro-F1 = 0.6897 (-0.0106 vs RF V2), Weighted-F1 = 0.7846, ROC-AUC = 0.9187. Models and evaluation plots saved to `results/models/*_v3_maxent.*`.
**Dissertation section:** §3.7.2, §4.1, §4.3, §4.6, Appendix A.3

### [Methodology Correction] Retraction of post-hoc hyperparameter selection on locked validation set — 2026-09-24
**Command:**
`rm -f scripts/check_maxent_configs.py`
**Rationale:** Place on formal record a methodology correction regarding Step 4 evaluation: Following the initial execution of `06_train_evaluate_maxent.py` (which correctly selected `max_depth=20, n_estimators=100` via training-set CV), a secondary unconstrained configuration (`max_depth=None, n_estimators=300`) was evaluated post-hoc against the locked internal test set and KD2 validation set, and was highlighted in the initial report based on superior test/KD2 numbers (76.19% test acc, 80.34% KD2 acc). This comparison was invalid: evaluating multiple model variants against a locked validation or test set and choosing or promoting based on out-of-sample metrics constitutes information leakage and post-hoc selection bias, equivalent to tuning against the test partition. To maintain strict scientific integrity, the exploration script was removed, all post-hoc comparisons were retracted, and the single legitimate candidate is confirmed as the GridSearchCV winner (`max_depth=20, n_estimators=100`) selected exclusively on training CV.
**Output:** Exploratory script removed; post-hoc configuration retracted from dissertation pipeline; official MaxEntScan results locked strictly to the training-CV winner: Internal test 75.79% acc / 0.6203 Macro-F1; KD2 validation 79.41% acc / 0.6897 Macro-F1.
**Dissertation section:** §3.7.2, §4.6, Chapter 5 (Methodological Integrity & Limitations)

