# splicing-ml-project

Machine learning classification of human alternative splicing events from RNA-seq data — implementation repository for Brooklyn Moyo's BSc dissertation (Chapters 3–5).

## Status

All pipeline stages (Stages 1–8) are complete:
- **Dataset:** ArrayExpress `E-MTAB-1147` / ENA `ERP001458` (*Zarnack et al., 2013* — *HNRNPC* knockdown vs. control in HeLa cells, paired-end RNA-seq).
- **Discovery Cohort (Control vs. KD1):** 8,693 depth-matched significant events (9,247 under variable-read-length quantification).
- **External Validation Cohort (Control vs. KD2):** 7,080 depth-matched significant events (7,279 under variable-read-length quantification).
- **Models & Evaluation:** Multinomial Logistic Regression, Random Forest, and XGBoost classifiers trained and evaluated across stratified and gene-grouped (`GroupShuffleSplit` / `GroupKFold`) partitions, with external validation on KD2 and an interactive Streamlit web application (`app/streamlit_app.py`).

## Pipeline Execution

```bash
cd scripts

# 1. Fetch ENA run metadata
./01a_fetch_run_metadata.sh

# 2. Download raw paired-end FASTQ files and Ensembl release 116 GRCh38 reference
./01b_download_fastq.sh ERR127302 ERR127303 ERR127304 ERR127305 ERR127306 ERR127307 ERR127308 ERR127309
./01c_download_reference.sh

# 3. Adapter & quality trimming (FastQC + Trimmomatic)
./02_trim_samples.sh

# 4. Build HISAT2 index and align reads
./03a_build_index.sh
./03b_align_samples.sh

# 5. Quantify differential splicing (rMATS) and run depth/lane controls
./04_run_rmats.sh
./04b_downsample_controls.sh
./04c_run_rmats_downsampled.sh
./04d_run_lane_control.sh

# 6. Extract features, train/evaluate models, and run external KD2 validation
python 05_feature_engineering.py
python 06_train_models.py
python 07_evaluate_models.py
python 08_run_kd2_pipeline.py
```

Requires: `curl`, `wget`, `fastqc`, `trimmomatic`, `hisat2`, `samtools`, `rmats`, and Python 3.11 (`pysam`, `biopython`, `pandas`, `scikit-learn`, `xgboost`, `matplotlib`, `streamlit`).

## Reproducing Dissertation Tables & Figures

### 1. GroupKFold Cross-Validation Composition (Table 4.2)
Computes validation event counts and class distributions across all 5 gene-grouped folds (0 shared genes):
```bash
python scripts/fold_composition.py
```

### 2. Primary XGBoost Permutation Feature Importance (Table B.4)
Computes the mean decrease and sample standard deviation in macro-F1 across 30 random permutations on the 1,807 held-out test events:
```bash
python scripts/perm_importance.py
```

### 3. Dissertation Figures & Confusion Matrices (Figures 4.1–4.5)
Generates the publication figures, ROC curves, calibration plots, and confusion matrix heatmaps:
```bash
python scripts/generate_dissertation_figures.py
python scripts/generate_confusion_matrices.py
```

### 4. Interactive Application & Demo Event Validation
To launch the interactive Streamlit classifier application on Windows:
```cmd
run_app.bat
```
Or directly from a Python environment with Streamlit installed:
```bash
streamlit run app/streamlit_app.py
```
To validate inference across all five canonical alternative splicing classes (`SE`, `MXE`, `A3SS`, `RI`, `A5SS`), open the **"Batch Predict (CSV)"** tab and upload `demo/demo_events.csv` (ground truth labels in `demo/demo_events_truth.csv`).
