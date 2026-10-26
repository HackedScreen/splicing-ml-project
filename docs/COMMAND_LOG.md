# Command Log

Chronological execution log of commands, parameter choices, and outputs across the pipeline stages.

---

### [Stage 1] Reference genome release pinned — 2026-08-31
**Command:**
`curl -sL https://ftp.ensembl.org/pub/release-116/gtf/homo_sapiens/CHECKSUMS`
**Rationale:** Pin the human reference genome and annotation to Ensembl release 116 (GRCh38.p14) for deterministic reproducibility across alignment and splicing quantification stages.
**Output:** Confirmed FTP paths under `https://ftp.ensembl.org/pub/release-116/fasta/homo_sapiens/dna/` and `https://ftp.ensembl.org/pub/release-116/gtf/homo_sapiens/`.
**Dissertation section:** §3.2.2

### [Stage 0] Environment setup — 2026-09-01
**Command:**
`mamba create -y -n splicing_project -c bioconda -c conda-forge sra-tools fastqc trimmomatic hisat2 samtools rmats python=3.11 && pip install biopython pysam pandas scikit-learn matplotlib`
**Rationale:** Install pinned versions for bioinformatics alignment, splicing detection, feature engineering, and machine learning pipelines.
**Output:** Environment `splicing_project` configured (python 3.11.11, hisat2 2.2.1, rmats 4.3.0, samtools 1.21, fastqc 0.12.1, trimmomatic 0.39, sra-tools 3.1.1, scikit-learn 1.7.1, biopython 1.85, pysam 0.23.0, pandas 2.3.2, matplotlib 3.10.1).
**Dissertation section:** §3.1

### [Stage 1] Run metadata fetched — 2026-09-01
**Command:**
`bash scripts/01a_fetch_run_metadata.sh`
**Rationale:** Fetch study ERP001458 (E-MTAB-1147, Zarnack et al., 2013) metadata from the ENA API to verify sample run accessions, replicate mapping, and paired-end layouts.
**Output:** Fetched paired-end run table saved to `data/reference/E-MTAB-1147_runs.tsv`.
**Dissertation section:** §3.2.2

### [Stage 1] Sample manifest constructed — 2026-09-01
**Command:**
`cp data/reference/manifest_TEMPLATE.csv data/reference/manifest.csv`
**Rationale:** Map run accessions to experimental conditions (Control vs HNRNPC knockdown replicates).
**Output:** Saved initial sample manifest to `data/reference/manifest.csv`.
**Dissertation section:** §3.2.1

### [Stage 1] Raw RNA-seq FASTQ download & integrity verification — 2026-09-01
**Command:**
`bash scripts/01b_download_fastq.sh ERR127302 ERR127303 ERR127304 ERR127305`
**Rationale:** Download raw paired-end sequencing reads for the 4 primary study runs from ENA into `data/raw/`.
**Output:** 4 FASTQ pairs downloaded into `data/raw/`: ERR127302, ERR127303, ERR127304, ERR127305.
**Dissertation section:** §3.2.2

### [Stage 1] Reference genome and annotation acquisition — 2026-09-01
**Command:**
`bash scripts/01c_download_reference.sh`
**Rationale:** Retrieve Ensembl release 116 GRCh38.p14 primary assembly FASTA and gene annotation GTF.
**Output:** `Homo_sapiens.GRCh38.dna.primary_assembly.fa` and `Homo_sapiens.GRCh38.116.gtf` downloaded and extracted in `data/reference/`.
**Dissertation section:** §3.2.2

### [Stage 2] Pre-trim quality control (FastQC) — 2026-09-01
**Command:**
`fastqc -t 4 -o results/qc/pre/ data/raw/*.fastq.gz`
**Rationale:** Assess base quality scores, GC content, sequence duplication, and adapter contamination across raw paired-end FASTQ files before trimming.
**Output:** Generated 8 HTML/zip reports in `results/qc/pre/`. Per-base sequence quality passed across all files; Illumina universal adapter contamination flagged at read 3' ends.
**Dissertation section:** §3.3.1

### [Stage 2] Adapter & quality trimming (ERR127302) — 2026-09-01
**Command:**
`trimmomatic PE -threads 4 -phred33 data/raw/ERR127302_1.fastq.gz data/raw/ERR127302_2.fastq.gz data/trimmed/ERR127302_1_paired.fastq.gz data/trimmed/ERR127302_1_unpaired.fastq.gz data/trimmed/ERR127302_2_paired.fastq.gz data/trimmed/ERR127302_2_unpaired.fastq.gz ILLUMINACLIP:/opt/miniforge3/envs/splicing_project/share/trimmomatic/adapters/TruSeq3-PE.fa:2:30:10 LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15 MINLEN:36`
**Rationale:** Remove TruSeq3 adapter sequences, trim low-quality terminal bases (Phred < 3), filter low-quality windows (4 bp window average Phred < 15), and discard reads shorter than 36 bp.
**Output:** Trimmed paired and unpaired FASTQ outputs written to `data/trimmed/`.
**Dissertation section:** §3.3.2, Appendix A.1

### [Stage 2] Adapter & quality trimming (ERR127303) — 2026-09-01
**Command:**
`trimmomatic PE -threads 4 -phred33 data/raw/ERR127303_1.fastq.gz data/raw/ERR127303_2.fastq.gz data/trimmed/ERR127303_1_paired.fastq.gz data/trimmed/ERR127303_1_unpaired.fastq.gz data/trimmed/ERR127303_2_paired.fastq.gz data/trimmed/ERR127303_2_unpaired.fastq.gz ILLUMINACLIP:/opt/miniforge3/envs/splicing_project/share/trimmomatic/adapters/TruSeq3-PE.fa:2:30:10 LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15 MINLEN:36`
**Rationale:** Standardize trimming parameters across biological replicates.
**Output:** Input Read Pairs: 32,665,567 | Both Surviving: 26,758,217 (81.92%) | Forward Only: 3,659,196 (11.20%) | Reverse Only: 1,488,981 (4.56%) | Dropped: 759,173 (2.32%).
**Dissertation section:** §3.3.2, Appendix A.1

### [Stage 2] Adapter & quality trimming (ERR127304) — 2026-09-01
**Command:**
`trimmomatic PE -threads 4 -phred33 data/raw/ERR127304_1.fastq.gz data/raw/ERR127304_2.fastq.gz data/trimmed/ERR127304_1_paired.fastq.gz data/trimmed/ERR127304_1_unpaired.fastq.gz data/trimmed/ERR127304_2_paired.fastq.gz data/trimmed/ERR127304_2_unpaired.fastq.gz ILLUMINACLIP:/opt/miniforge3/envs/splicing_project/share/trimmomatic/adapters/TruSeq3-PE.fa:2:30:10 LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15 MINLEN:36`
**Rationale:** Apply identical trimming thresholds to knockdown replicate 1.
**Output:** Input Read Pairs: 31,876,181 | Both Surviving: 26,945,447 (84.53%) | Forward Only: 3,039,095 (9.53%) | Reverse Only: 1,301,609 (4.08%) | Dropped: 590,030 (1.85%).
**Dissertation section:** §3.3.2, Appendix A.1

### [Stage 2] Adapter & quality trimming (ERR127305) — 2026-09-01
**Command:**
`trimmomatic PE -threads 4 -phred33 data/raw/ERR127305_1.fastq.gz data/raw/ERR127305_2.fastq.gz data/trimmed/ERR127305_1_paired.fastq.gz data/trimmed/ERR127305_1_unpaired.fastq.gz data/trimmed/ERR127305_2_paired.fastq.gz data/trimmed/ERR127305_2_unpaired.fastq.gz ILLUMINACLIP:/opt/miniforge3/envs/splicing_project/share/trimmomatic/adapters/TruSeq3-PE.fa:2:30:10 LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15 MINLEN:36`
**Rationale:** Apply identical trimming thresholds to knockdown replicate 2.
**Output:** Input Read Pairs: 32,345,742 | Both Surviving: 28,042,418 (86.70%) | Forward Only: 2,772,429 (8.57%) | Reverse Only: 1,025,964 (3.17%) | Dropped: 504,931 (1.56%).
**Dissertation section:** §3.3.2, Appendix A.1

### [Stage 2] Post-trim quality control (FastQC) — 2026-09-01
**Command:**
`fastqc -t 4 -o results/qc/post/ data/trimmed/*_paired.fastq.gz`
**Rationale:** Verify removal of adapter contamination and retention of base-call quality across trimmed paired reads.
**Output:** Generated 8 HTML/zip reports in `results/qc/post/`. Adapter Content passed across all files; Per-base sequence quality maintained PASS across all positions.
**Dissertation section:** §3.3.3

### [Stage 1] Reference GTF annotation verification — 2026-09-06
**Command:**
`curl -sL https://ftp.ensembl.org/pub/release-116/gtf/homo_sapiens/CHECKSUMS | grep 'Homo_sapiens.GRCh38.116.gtf.gz' && sum data/reference/Homo_sapiens.GRCh38.116.gtf.gz`
**Rationale:** Verify the Ensembl release 116 GTF against published Ensembl CHECKSUMS.
**Output:** Checksum matched: `49151 137815 Homo_sapiens.GRCh38.116.gtf.gz` (141,121,632 bytes compressed, 4,659,761,880 bytes uncompressed).
**Dissertation section:** §3.2.2

### [Stage 1] Annotation splice-site and exon coordinate extraction — 2026-09-07
**Command:**
`hisat2_extract_splice_sites.py data/reference/Homo_sapiens.GRCh38.116.gtf > data/reference/splicesites.tsv && hisat2_extract_exons.py data/reference/Homo_sapiens.GRCh38.116.gtf > data/reference/exons.tsv`
**Rationale:** Extract known splice sites and exon coordinates from Ensembl release 116 GTF using HISAT2 helper utilities.
**Output:** Generated `splicesites.tsv` (698,580 entries) and `exons.tsv` (409,261 intervals).
**Dissertation section:** §3.4.1

### [Stage 3] Linear HISAT2 index construction & verification — 2026-09-21
**Command:**
`hisat2-build -p 4 data/reference/Homo_sapiens.GRCh38.dna.primary_assembly.fa data/reference/grch38_index && hisat2-inspect --summary data/reference/grch38_index`
**Rationale:** Build a linear GRCh38 reference index and supply known splice sites (`splicesites.tsv`) at alignment time via `--known-splicesite-infile`.
**Output:** Verified all 8 index parts (`grch38_index.1.ht2` to `.8.ht2`, 4.20 GB) covering 194 sequences (3,099,750,718 bp).
**Dissertation section:** §3.4.1

### [Stage 3] HISAT2 sample alignment and BAM indexing (ERR127302–ERR127305) — 2026-09-19
**Command:**
`bash scripts/03b_align_samples.sh ERR127302 ERR127303 ERR127304 ERR127305 && samtools quickcheck -v results/alignment/*.sorted.bam`
**Rationale:** Align trimmed paired-end reads to the GRCh38 index with `--dta` and `--known-splicesite-infile data/reference/splicesites.tsv`, then sort and index BAM files via samtools.
**Output:** Alignment rates: ERR127302 (95.29%), ERR127303 (95.55%), ERR127304 (96.25%), ERR127305 (94.74%). All BAM EOF blocks verified via `samtools quickcheck`.
**Dissertation section:** §3.4.2

### [Stage 3] Control replicate alignment verification (Control_rep1, Control_rep2) — 2026-09-21
**Command:**
`samtools quickcheck -v results/alignment/Control_rep*.sorted.bam && samtools flagstat -@ 4 results/alignment/Control_rep1.sorted.bam && samtools flagstat -@ 4 results/alignment/Control_rep2.sorted.bam`
**Rationale:** Verify Control biological replicates formed by concatenating paired technical sequencing lanes from ERP001458 (Control_rep1 = ERR127306 + ERR127307; Control_rep2 = ERR127308 + ERR127309).
**Output:** Control_rep1: 55,453,742 trimmed pairs (90.20% retention), 94.52% overall alignment rate (116,095,872 mapped reads). Control_rep2: 57,186,429 trimmed pairs (91.84% retention), 89.49% overall alignment rate (113,558,516 mapped reads).
**Dissertation section:** §3.4.2

### [Stage 1] Sample manifest updated via ENA sample accession cross-check — 2026-09-22
**Command:**
`python scripts/validate_stage1.py`
**Rationale:** Cross-referencing ENA `sample_accession` fields confirmed that ERR127302/ERR127303 correspond to HNRNPC knockdown construct 1 (KD1), ERR127304/ERR127305 correspond to knockdown construct 2 (KD2), and ERR127306–ERR127309 correspond to untreated Control replicates 1 and 2. Updated `data/reference/manifest.csv` accordingly.
**Output:** Confirmed manifest with 2 Control replicates (`Control_rep1`, `Control_rep2`), 2 KD1 replicates (`ERR127302`, `ERR127303`), and 2 held-out KD2 replicates (`ERR127304`, `ERR127305`).
**Dissertation section:** §3.2.1

### [Stage 4] Differential alternative splicing quantification (Control vs KD1 full-depth) — 2026-09-22
**Command:**
`bash scripts/04_run_rmats.sh`
**Rationale:** Quantify differential splicing between Control (`Control_rep1.sorted.bam`, `Control_rep2.sorted.bam`) and HNRNPC KD1 (`ERR127302.sorted.bam`, `ERR127303.sorted.bam`) across five canonical alternative splicing event types.
**Output:** Quantified 168,919 total splicing events with 10,746 significant events (FDR < 0.05): SE: 5,112 (47.57%), RI: 698 (6.50%), A5SS: 505 (4.70%), A3SS: 1,171 (10.90%), MXE: 3,260 (30.34%).
**Dissertation section:** §3.5, §4.3

### [Stage 4] Control BAM subsampling to match KD1 library depth — 2026-09-22
**Command:**
`bash scripts/04b_downsample_controls.sh`
**Rationale:** Subsample Control BAMs to 46.9% using `samtools view -s 42.469` to match the ~27M mapped read-pair depth of KD1 replicates and test whether sequencing depth differences affect event-type proportions.
**Output:** Generated `Control_rep1_downsampled.sorted.bam` (54,435,443 mapped reads) and `Control_rep2_downsampled.sorted.bam` (53,265,534 mapped reads).
**Dissertation section:** §3.4.2, §3.5

### [Stage 4] Depth-matched rMATS execution (Downsampled Control vs KD1) — 2026-09-22
**Command:**
`bash scripts/04c_run_rmats_downsampled.sh`
**Rationale:** Run rMATS on depth-matched Control vs KD1 BAMs (`--readLength 72 --nthread 4`).
**Output:** Quantified 148,140 total splicing events with 8,693 significant events (FDR < 0.05): SE: 4,306 (49.53%), RI: 500 (5.75%), A5SS: 400 (4.60%), A3SS: 786 (9.04%), MXE: 2,701 (31.07%).
**Dissertation section:** §3.5, §4.3

### [Stage 4] Technical lane negative control (ERR127306 vs ERR127307) — 2026-09-23
**Command:**
`bash scripts/04d_run_lane_control.sh`
**Rationale:** Split `Control_rep1.sorted.bam` by read name prefix into its two constituent sequencing lanes (`ERR127306` vs `ERR127307`, same biological library `SAMEA1691813`) and run rMATS to evaluate false-positive lane effects.
**Output:** Quantified 114,252 total splicing events with 3,741 nominal calls (FDR < 0.05, 3.27% of tested events): SE: 1,635, MXE: 785, A3SS: 648, A5SS: 396, RI: 277, with median |ΔPSI| = 1.000.
**Dissertation section:** §3.5, §4.3

### [Stage 5] Feature engineering on depth-matched dataset — 2026-09-23
**Command:**
`python scripts/05_feature_engineering.py`
**Rationale:** Extract exon length, exon and flanking 100 bp GC content, 5' donor and 3' acceptor PWM log-odds scores, local BAM read coverage statistics, and replicate PSI means/variances across all 8,693 significant events (FDR < 0.05).
**Output:** Generated `results/features/feature_matrix_corrected.csv` (8,693 rows, 29 columns; 0 missing values).
**Dissertation section:** §3.6, §4.3

### [Stages 6 & 7] Model training, cross-validation tuning & held-out evaluation — 2026-09-23
**Command:**
`python scripts/06_train_models.py`
**Rationale:** Train multinomial Logistic Regression (`C` in `[0.01, 0.1, 1, 10]`) and Random Forest (`n_estimators` in `[100, 300]`, `max_depth` in `[None, 10, 20]`) with balanced class weights using 5-fold stratified cross-validation on the 80% training split ($N=6,954$) and evaluate on the 20% held-out test set ($N=1,739$).
**Output:** Random Forest (`n_estimators=300, max_depth=None`, CV Macro-F1 = 0.6528) achieved 74.01% Test Accuracy, 0.6474 Macro-F1, 0.7365 Weighted-F1, and 0.9026 Macro ROC-AUC. Logistic Regression (`C=10`, CV Macro-F1 = 0.4171) achieved 47.50% Test Accuracy and 0.4049 Macro-F1.
**Dissertation section:** §3.7, §4.1–§4.5

### [Stage 8] External validation on independent KD2 construct — 2026-09-24
**Command:**
`python scripts/08_run_kd2_pipeline.py`
**Rationale:** Run rMATS comparing downsampled Control vs KD2 (`ERR127304`, `ERR127305`), extract the 19 modeling features across all significant events (FDR < 0.05), and evaluate the trained Random Forest classifier on the independent KD2 cohort.
**Output:** Extracted 7,080 significant KD2 events (`results/features/feature_matrix_kd2_corrected.csv`). Random Forest achieved 75.82% Accuracy, 0.6896 Macro-F1, 0.7595 Weighted-F1, and 0.9120 Macro ROC-AUC.
**Dissertation section:** §3.5, §3.6, §4.6

### [Version Control] Repository initialization and remote push — 2026-09-24
**Command:**
`git init && git remote add origin https://github.com/HackedScreen/splicing-ml-project.git && git branch -M main && git add . && git commit -m "Final synced pipeline - 75.8% KD2 external validation" && git push -u origin main`
**Rationale:** Initialize Git repository and push pipeline scripts, feature matrices, and evaluation reports to GitHub.
**Output:** Pushed `main` branch to `origin`.
**Dissertation section:** §3.1

### [Stage 5 V2] HNRNPC U-tract binding motif feature extraction — 2026-09-24
**Command:**
`python scripts/05_feature_engineering_v2.py`
**Rationale:** Extract 4 HNRNPC U-tract motif features (`max_utract_length_upstream`, `max_utract_length_downstream`, `u_fraction_upstream`, `u_fraction_downstream`) across 100 bp flanking intronic windows for KD1 ($N=8,693$) and KD2 ($N=7,080$).
**Output:** Saved `results/features/feature_matrix_v2.csv` (8,693 rows, 33 columns) and `results/features/feature_matrix_kd2_v2.csv` (7,080 rows, 33 columns).
**Dissertation section:** §3.6, §4.3

### [Stages 6 & 7 V2] XGBoost and Random Forest V2 training and evaluation — 2026-09-24
**Command:**
`python scripts/06_train_evaluate_v2.py`
**Rationale:** Train and evaluate XGBoost and Random Forest V2 on the 23-feature set (19 baseline features + 4 U-tract features) using 5-fold stratified CV on the 80% KD1 training partition ($N=6,954$), then evaluate on the internal test set ($N=1,739$) and external KD2 cohort ($N=7,080$).
**Output:** Random Forest V2 achieved 74.81% internal test accuracy (0.6611 Macro-F1) and 76.82% KD2 validation accuracy (0.7003 Macro-F1). XGBoost (`lr=0.05, max_depth=6, n_estimators=300`) achieved 72.17% internal test accuracy (0.6385 Macro-F1) and 73.90% KD2 accuracy (0.6655 Macro-F1).
**Dissertation section:** §3.7.2, §4.1, §4.3, §4.6

### [Stage 7] Soft-voting ensemble and threshold tuning experiments — 2026-09-24
**Command:**
`python scripts/07_ensemble_and_threshold_tuning.py`
**Rationale:** Evaluate out-of-fold soft-voting weights combining RF V2 and XGBoost V2 as well as per-class probability threshold multipliers on 5-fold training CV.
**Output:** Soft-voting ($w_{RF}=0.7, w_{XGB}=0.3$) achieved 74.64% internal test accuracy (0.6532 Macro-F1) and 76.82% KD2 accuracy (0.7020 Macro-F1). Threshold tuning reduced internal test accuracy to 72.80% (0.6462 Macro-F1). Retained standalone RF V2 model.
**Dissertation section:** §3.7.3, §4.3, §4.6

### [Stages 5–7 V3] MaxEntScan splice site scoring and evaluation — 2026-09-24
**Command:**
`python scripts/05_feature_engineering_maxent.py && python scripts/06_train_evaluate_maxent.py`
**Rationale:** Score 9 bp 5' donor and 23 bp 3' acceptor windows using MaxEntScan (Yeo & Burge, 2004) maximum entropy models and evaluate 5-fold CV tuned Random Forest (`max_depth=20, n_estimators=100`).
**Output:** Generated `feature_matrix_v3_maxent.csv` and `feature_matrix_kd2_v3_maxent.csv`. Internal test accuracy reached 75.79% (Macro-F1 = 0.6203, ROC-AUC = 0.9055); external KD2 accuracy reached 79.41% (Macro-F1 = 0.6897, ROC-AUC = 0.9187).
**Dissertation section:** §3.6, §3.7.2, §4.3, §4.6

### [Stage 7c] Gene-grouped split robustness check (GroupShuffleSplit & GroupKFold) — 2026-10-08
**Command:**
`python scripts/07c_gene_grouped_split_robustness.py`
**Rationale:** Quantify multi-event gene overlap and evaluate model performance under `GroupShuffleSplit` (80/20 grouped by `gene_id`, zero shared genes between train and test) and 5-fold `GroupKFold`.
**Output:** 79.01% of events reside in multi-event genes. Under gene-grouped splitting, RF V2 test accuracy was 66.59% (Macro-F1 = 0.5567; 5-fold GroupKFold mean accuracy = 66.88%, Macro-F1 = 0.5629), and external KD2 accuracy was 74.94%.
**Dissertation section:** §4.5, §5.1

### [Stage 7d] Gene-overlap stratification, feature ablations, calibration & bootstrap CIs — 2026-10-08
**Command:**
`python scripts/07d_supervisor_followup_analyses.py`
**Rationale:** Stratify KD2 performance by seen vs unseen training genes, run 5-fold GroupKFold feature family ablations, compute 1,000-iteration gene-cluster bootstrap 95% confidence intervals, and test sigmoid/isotonic probability calibration (`CalibratedClassifierCV`).
**Output:** KD2 seen genes ($N=4,362$): 82.03% accuracy, 0.7700 Macro-F1; unseen genes ($N=2,718$): 63.58% accuracy, 0.5222 Macro-F1. 1,000-iteration gene-cluster bootstrap 95% CIs on test set: Accuracy [63.27%, 70.18%], Macro-F1 [0.5190, 0.5966]. Calibration reduced multiclass Brier score from 0.4647 to 0.4358.
**Dissertation section:** §4.5, §5.1

### [Stage R1] Variable-read-length rMATS re-quantification and grouped model evaluation — 2026-10-13
**Command:**
`bash scripts/r1_step1_archive.sh && bash scripts/r1_run_rmats_kd1.sh && bash scripts/r1_run_rmats_kd2.sh && bash scripts/r1_run_rmats_lane_control.sh && python scripts/r1_step3_compare_kd1.py && python scripts/r1_step4_extract_features.py && python scripts/r1_step5_model_protocol.py`
**Rationale:** Re-run rMATS with `--variable-read-length` so trimmed reads shorter than 72 bp are retained, regenerate the 23-feature matrices for KD1 and KD2, split KD1 by `gene_id` (80/20 `GroupShuffleSplit`), and tune XGBoost and Random Forest via 5-fold `GroupKFold`.
**Output:** KD1 significant events: 9,247 (`7,440` train / `1,807` test, 0 shared genes); KD2 significant events: 7,279; lane control nominal calls: 3,640 / 119,797 (3.04%). GroupKFold winner XGBoost (`n_estimators=300, max_depth=6, learning_rate=0.05`, CV Macro-F1 = 0.5832) achieved 67.68% test accuracy and 0.6060 Macro-F1 (95% bootstrap CI: [65.25%, 70.09%] acc, [0.5703, 0.6382] Macro-F1) and 72.73% KD2 accuracy / 0.6660 Macro-F1.
**Dissertation section:** §3.5–§3.7, §4.1–§4.6

### [Stage R2] Confusion matrix verification, Logistic Regression baseline & paired bootstrap — 2026-10-13
**Command:**
`python scripts/r2_execution.py && python scripts/test_app_prediction.py`
**Rationale:** Export confusion matrix arrays, run 5-fold GroupKFold across 5 feature ablation subsets, evaluate the tuned Logistic Regression baseline (`C=0.1`) under gene-grouped splitting, and compute a 1,000-iteration paired gene-cluster bootstrap comparing XGBoost and Random Forest on the 1,807 held-out test events.
**Output:** Logistic Regression baseline achieved 47.59% test accuracy (0.4228 Macro-F1) and 45.01% KD2 accuracy (0.3972 Macro-F1). Paired gene-cluster bootstrap Macro-F1 difference (XGBoost - RF) = +0.0410 (95% CI: [+0.0119, +0.0697], p = 0.0010).
**Dissertation section:** §3.7, §4.1–§4.6

### [Stage 7e] Mutual information and missingness sensitivity checks — 2026-10-24
**Command:**
`python scripts/run_r5_checks.py && python scripts/r6_weighted_f1.py && python scripts/random_split_comparison.py`
**Rationale:** Compute feature-label mutual information (`mutual_info_classif`), verify test-set weighted-F1 scores across trained models and dummy baselines, and compare random 80/20 splitting against gene-grouped splitting on the 9,247 KD1 dataset.
**Output:** Weighted-F1 on held-out test set ($N=1,807$): XGBoost = 0.6741, Random Forest = 0.6522 (1,197 / 1,807 correct, 66.24%), Majority Dummy = 0.3398, Stratified Dummy = 0.3377. Random 80/20 split resulted in 75.89% test gene leakage and inflated XGBoost accuracy from 67.68% to 76.97% (+9.29 pp) and Macro-F1 from 0.6060 to 0.6633 (+0.0573).
**Dissertation section:** §4.1, §4.3, §4.5

### [Stage 9] Reproducing dissertation tables and figures — 2026-10-26
**Command:**
`python scripts/fold_composition.py && python scripts/perm_importance.py && python scripts/generate_dissertation_figures.py && python scripts/generate_confusion_matrices.py`
**Rationale:** Compute 5-fold GroupKFold class distributions (Table 4.2), 30-repeat permutation feature importance on the 1,807 held-out test events (Table B.4), and generate publication figures and confusion matrices (Figures 4.1–4.5).
**Output:** Verified 5-fold class distributions across 7,440 training events (2,989 unique genes), permutation importance rankings led by `exon_length` (0.1936 ± 0.0098 macro-F1 drop), and saved updated figure assets.
**Dissertation section:** §4.1, §4.5–§4.7, Appendix B.4
