# splicing-ml-project

AI-based prediction of alternative splicing — implementation repo for Brooklyn
Moyo's dissertation (Chapters 3–4). See `docs/RESEARCH_ARCHITECTURE.md` for the
full pipeline design and `docs/COMMAND_LOG.md` for the running, append-only
reproducibility log.

## Status

Stage 1 (data acquisition) in progress. Dataset selected: E-MTAB-1147
(HNRNPC knockdown vs control, HeLa, paired-end, 2 conditions x 2 replicates).

## Quickstart (run on your own machine — needs internet access)

```bash
cd scripts

# 1. Pull the authoritative run table and inspect it
./01a_fetch_run_metadata.sh

# 2. Fill in ../data/reference/manifest_TEMPLATE.csv with the confirmed
#    run_accession -> condition/replicate mapping, save as manifest.csv

# 3. Download the confirmed runs (edit accessions to match your manifest)
./01b_download_fastq.sh ERR127302 ERR127303 ERR127309 ERR127310

# 4. Download the pinned reference genome + annotation (~3 GB, takes a while)
./01c_download_reference.sh
```

Requires: `curl`, `wget`, SRA Toolkit (`prefetch`, `fasterq-dump`) on PATH.

## Next stages (not yet built — see architecture doc)

2. Quality control (FastQC, Trimmomatic)
3. Genome alignment (HISAT2)
4. Splicing detection (rMATS)
5. Feature engineering
6. Model training (scikit-learn)
7. Evaluation


## Reproducing the thesis tables

To reproduce the core empirical tables, figures, and interactive model demonstrations reported in the dissertation:

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
Alternatively, in any Python environment with Streamlit installed:
```bash
streamlit run app/streamlit_app.py
```
To validate inference across all five canonical alternative splicing classes (SE, MXE, A3SS, RI, A5SS):
1. Navigate to the **"Batch Prediction from CSV File"** tab.
2. Upload `demo/demo_events.csv`.
3. Verify predicted classifications and probabilities against ground truth in `demo/demo_events_truth.csv` (100% agreement on high-confidence held-out events).
