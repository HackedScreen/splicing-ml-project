# Research Architecture & Reproducibility Protocol
**Project:** AI-Based Prediction of Alternative Splicing of the Human Genome
**Author:** Brooklyn Moyo
**Covers:** Implementation of Chapter 3 (Methodology), feeding Chapter 4 (Results) and Appendix A (Code Snippets)
**Status:** Complete (Stages 1–8)

---

## 1. Purpose

Chapters 1–2 define *what* is being tested (whether machine learning can classify splicing event types from
RNA-seq-derived features) and *why* (Section 2.5's research gap: a genuinely
reproducible, end-to-end pipeline is rare). This document defines *how* the pipeline is built —
the system architecture, the tools per stage, the project layout, and the reproducibility protocol
followed so every execution step is traceable back to the dissertation text.

Everything here maps directly onto the Chapter 3 table of contents, so as
each stage is implemented, the outputs and execution records slot straight into the corresponding
subsection.

## 2. Pipeline architecture

Seven sequential stages, each with a single tool responsibility and a well-defined
input/output contract:

| # | Stage | Tool(s) | Input | Output |
|---|-------|---------|-------|--------|
| 1 | Data acquisition | SRA Toolkit (`prefetch`, `fasterq-dump`), Ensembl FTP | Accession list (§3.2.1 criteria) | Raw paired-end FASTQ + `manifest.csv` |
| 2 | Quality control | FastQC, Trimmomatic | Raw FASTQ | Trimmed FASTQ + QC reports (pre/post) |
| 3 | Genome alignment | HISAT2, samtools | Trimmed FASTQ + GRCh38 + GTF | Sorted, indexed BAM |
| 4 | Splicing detection | rMATS-turbo | BAM (≥2 replicated conditions) | Per-event-type PSI tables (SE, RI, A5SS, A3SS, MXE) with FDR |
| 5 | Feature engineering | Python (biopython, pysam, pandas) | rMATS output + reference | `feature_matrix.csv` (PWM scores, exon/intron length, GC%, conservation, ESE/ESS density, coverage stats, PSI + variance, label) |
| 6 | Model development | scikit-learn | `feature_matrix.csv` | Trained models (Logistic Regression, Random Forest) + tuned hyperparameters |
| 7 | Evaluation | scikit-learn metrics | Held-out test set + models | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, feature importances |

**Design rationale (detailed in Chapter 2):** HISAT2 over STAR for memory
efficiency (§2.3.1); rMATS over SUPPA2 for its explicit replicate model (§2.3.2);
hand-engineered features + classical scikit-learn models over deep learning, for
interpretability given available compute (§2.4.2).

## 2a. Implementation workflow

Development, execution, and validation are carried out locally on the primary
workstation. This document (alongside `docs/IMPLEMENTATION_STRATEGY.md`)
defines the stage-by-stage engineering specification, including exact command
invocations and validation gates per stage.

## 3. Reproducibility protocol

This protocol is followed at every stage so that full methodological reproducibility
is maintained by construction rather than reconstructed after the fact:

1. **Define the stage objective** (e.g., Stage 3 — building the HISAT2 index).
2. **Specify the exact command(s) and scripts**, with parameters justified against the
   literature or the dataset's empirical characteristics.
3. **Execute and validate** — running initial verification checks on subset data where appropriate before committing to full-scale execution (see §5).
4. **Record parameters and outputs** immediately upon completion of each stage.
5. **Document decision points** (threshold choices, model selection, replicate design) with a clear methodological justification for inclusion in Chapter 3 and Appendix A.

## 4. Project directory structure

```
splicing-ml-project/
├── data/
│   ├── raw/              # untouched FASTQ from SRA/Ensembl
│   ├── trimmed/           # Trimmomatic output
│   └── reference/         # GRCh38 FASTA, GTF, HISAT2 index
├── results/
│   ├── qc/                 # FastQC reports, pre and post trim
│   ├── alignment/          # sorted BAM + HISAT2 summary logs
│   ├── splicing/            # rMATS output per event type
│   ├── features/            # feature_matrix.csv
│   └── models/               # trained model artifacts, metrics, plots
├── scripts/
│   ├── 01_download_data.sh
│   ├── 02_run_fastqc.sh
│   ├── 03_trim_reads.sh
│   ├── 04_build_hisat2_index.sh
│   ├── 05_align_reads.sh
│   ├── 06_run_rmats.sh
│   ├── 07_feature_engineering.py
│   ├── 08_train_models.py
│   └── 09_evaluate_models.py
├── docs/
│   ├── RESEARCH_ARCHITECTURE.md   # this file
│   └── environment.yml             # conda environment spec
└── README.md
```

Numbered scripts double as the pipeline's execution order and as a natural table of
contents for Appendix A.

## 5. Compute & execution note

Resource-intensive stages require dedicated local or high-performance compute capacity:

- Stages 1 (raw FASTQ download) and 3 (HISAT2 index build against full GRCh38, ~4 GB index) are executed directly on the local workstation / compute environment with full network and storage access.
- Stages 2, 5, 6, and 7 can be validated first on chromosome-subset data to verify pipeline logic end-to-end before committing compute time to full-scale runs.
- Every stage invocation and parameter set is version-controlled in the numbered scripts to ensure deterministic reproduction across environments.

## 6. Execution record template

Each pipeline stage is documented using the following structured format:

```
### [Stage N] <short description> — YYYY-MM-DD
**Command:**
`<exact command>`
**Rationale:** <why these parameters/tool>
**Output:** <one-line summary, e.g. "18.2M read pairs, 94.3% overall alignment rate">
**Dissertation section:** §3.X
```

## 7. Decisions log (resolved)

- **Execution environment:** Local workstation.
- **Starting point:** Stage 1, dataset selection & download.
- **Dataset selected:** ArrayExpress/ENA **E-MTAB-1147** (Zarnack et al., 2013) —
  HNRNPC knockdown vs control in HeLa cells. Paired-end, human, 2 conditions
  (Control, HNRNPC knockdown "KD1") x 2 biological replicates each — satisfies
  rMATS's replicate requirement out of the box, and the splicing changes it
  produces are well characterised in the literature, providing a reliable benchmark to
  validate the pipeline against known biology. Same runs are also fetchable via NCBI SRA's `prefetch` under
  the identical ERR accessions.
- **Reference genome:** Pinned to **Ensembl release 116** (GRCh38.p14) rather than "latest," so the exact annotation is always reproducible.

## 8. Open decisions still ahead

- Confirm the exact run accession → condition/replicate mapping from the ENA
  table (script `01a_fetch_run_metadata.sh` — first thing to run).
- Whether to also use the study's second knockdown construct (KD2) as an
  independent validation set later, or keep it out of the training data
  entirely.
