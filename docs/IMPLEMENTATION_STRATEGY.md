# Implementation Strategy

Stage-by-stage technical workflow and validation criteria for the alternative splicing classification pipeline. Each stage defines the execution steps, expected outputs, and verification checks required before proceeding to the next stage.

---

## Stage 0 — Environment setup

**Procedure:**
Set up a conda environment (`splicing_project`) with `sra-tools`, `fastqc`, `trimmomatic`, `hisat2`, `samtools`, `rmats` (bioconda), `python=3.11`, and `biopython`, `pysam`, `pandas`, `scikit-learn`, `matplotlib` via pip. Verify each binary with `--version`.

**Verification criteria:**
- Environment packages are installed with pinned versions.
- Every tool responds to `--version` without error.

---

## Stage 1 — Data acquisition

**Procedure:**
1. Run `scripts/01a_fetch_run_metadata.sh` and inspect the resulting run table in `data/reference/E-MTAB-1147_runs.tsv`.
2. Confirm the run accession to condition/replicate mapping and populate `data/reference/manifest.csv`.
3. Run `scripts/01b_download_fastq.sh` with the confirmed accessions and `scripts/01c_download_reference.sh` to fetch the GRCh38 primary assembly FASTA and Ensembl release 116 GTF.

**Verification criteria:**
- `data/reference/manifest.csv` maps each run accession to its biological condition and replicate number.
- Paired FASTQ files are present in `data/raw/` with matching read counts between `_1` and `_2` mates.
- Reference FASTA and GTF are extracted in `data/reference/`.

---

## Stage 2 — Quality control

**Procedure:**
Run FastQC on each raw FASTQ pair into `results/qc/pre/`. Trim adapters and low-quality bases with Trimmomatic in paired-end mode (`ILLUMINACLIP` against `TruSeq3-PE.fa`, `LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15 MINLEN:36`), writing outputs to `data/trimmed/`. Re-run FastQC on the trimmed paired files into `results/qc/post/`.

**Verification criteria:**
- Post-trim FastQC shows no adapter contamination flags and per-base sequence quality passing across the read length.
- Read pair retention after trimming exceeds 80%.

---

## Stage 3 — Genome alignment (HISAT2)

**Procedure:**
Extract known splice sites and exons from the Ensembl 116 GTF using `hisat2_extract_splice_sites.py` and `hisat2_extract_exons.py`. Build the GRCh38 HISAT2 index (`scripts/03a_build_index.sh`) and align each trimmed FASTQ pair with `hisat2 -x <index> -1 <R1> -2 <R2> -p <threads> --dta --known-splicesite-infile <ss> -S <sam>` (`scripts/03b_align_samples.sh`). Sort and index BAM files using `samtools`.

**Verification criteria:**
- Overall alignment rate >85% for every replicate.
- Sorted and indexed `.sorted.bam` and `.bai` files are present in `results/alignment/` and pass `samtools quickcheck`.

---

## Stage 4 — Splicing detection (rMATS)

**Procedure:**
Prepare `b1.txt` (Control BAM paths) and `b2.txt` (knockdown BAM paths) and run `rmats.py` (`scripts/04_run_rmats.sh`) with `--gtf <gtf> -t paired --readLength 72 --nthread 4`. Also run the depth-matched comparison (`scripts/04b_downsample_controls.sh`, `scripts/04c_run_rmats_downsampled.sh`) and technical lane negative control (`scripts/04d_run_lane_control.sh`).

**Verification criteria:**
- All five `*.MATS.JC.txt` event tables (SE, RI, A5SS, A3SS, MXE) are generated and non-empty.
- Significant event counts ($FDR < 0.05$) and class proportions are recorded across all five canonical splicing categories.

---

## Stage 5 — Feature engineering

**Procedure:**
Run `scripts/05_feature_engineering.py` to parse significant events ($FDR < 0.05$) across all five rMATS output files and extract:
1. Replicate PSI means, variances, and $\Delta\text{PSI}$ from rMATS inclusion levels.
2. Target exon length from genomic coordinates.
3. Exon and 100 bp flanking intronic GC content from the GRCh38 FASTA via `pysam`.
4. Local RNA-seq read coverage mean and variance across the event locus from the sorted BAMs.
5. 5' donor (9 bp) and 3' acceptor (23 bp) splice-site consensus PWM log-odds scores.
6. Event type class label (`SE`, `RI`, `A5SS`, `A3SS`, `MXE`).

**Verification criteria:**
- `results/features/feature_matrix.csv` is written with one row per significant event.
- Feature columns are checked for missing or NaN values and class distribution is summarized.

---

## Stage 6 — Model development

**Procedure:**
Run `scripts/06_train_models.py` to split `feature_matrix.csv` into an 80/20 stratified train/test partition (`random_state=42`) and train:
1. Multinomial Logistic Regression (`class_weight='balanced'` with `StandardScaler`).
2. Random Forest Classifier (`class_weight='balanced'`).
Tune hyperparameters on the training split using 5-fold stratified `GridSearchCV` (`f1_macro` scoring) and serialize the fitted models to `results/models/`.

**Verification criteria:**
- Both classifiers are tuned via cross-validation on the training partition only.
- The 20% test partition remains untouched until final evaluation.

---

## Stage 7 — Evaluation & Robustness Checks

**Procedure:**
Run `scripts/07_evaluate_models.py` and `scripts/07b_robustness_check.py` to compute held-out test accuracy, macro/weighted precision, recall, F1, one-vs-rest ROC-AUC, confusion matrices, and feature importances. Save evaluation reports and figures to `results/models/`.

**Verification criteria:**
- Both models are evaluated on the held-out test set.
- Per-class F1 scores, confusion matrices, and feature importance rankings are saved to `results/models/`.

---

## Stage 8 — External validation (KD2)

**Procedure:**
Run `scripts/08_run_kd2_pipeline.py` to quantify differential splicing on the held-back HNRNPC knockdown construct 2 (KD2: `ERR127304`, `ERR127305`), extract the corresponding feature matrix (`results/features/feature_matrix_kd2.csv`), and evaluate out-of-sample generalization of the trained Random Forest classifier (`results/models/kd2_validation_report.md`).
