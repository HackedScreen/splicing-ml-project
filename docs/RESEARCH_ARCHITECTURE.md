# Research Architecture & Reproducibility Protocol
**Project:** AI-Based Prediction of Alternative Splicing of the Human Genome
**Author:** Brooklyn Moyo | **Co-lead technician:** Claude
**Covers:** Implementation of Chapter 3 (Methodology), feeding Chapter 4 (Results) and Appendix A (Code Snippets)
**Status:** Living document — append to the Command Log (Section 6) every session

---

## 1. Purpose

Chapters 1–2 define *what* we're testing (can ML classify splicing event type from
RNA-seq-derived features?) and *why* (Section 2.5's research gap: a genuinely
reproducible, end-to-end pipeline is rare). This document defines *how* we build it —
the system architecture, the tools per stage, the project layout, and the protocol we
follow so every command we run is traceable back into your dissertation text.

Everything here maps directly onto your existing Chapter 3 table of contents, so as we
implement each stage, the outputs and command log slot straight into the corresponding
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

**Design rationale (already argued in your Chapter 2):** HISAT2 over STAR for memory
efficiency (§2.3.1); rMATS over SUPPA2 for its explicit replicate model (§2.3.2);
hand-engineered features + classical scikit-learn models over deep learning, for
interpretability given available compute (§2.4.2).

## 2a. Division of labour

Antigravity IDE handles implementation, execution, and validation on
Brooklyn's laptop. This document (and `docs/IMPLEMENTATION_STRATEGY.md`)
is the strategy Antigravity follows — see that file for the exact,
mission-by-mission playbook with commands and validation gates per stage.
`AGENTS.md` at the repo root encodes the documentation protocol so it
applies automatically to every agent session, not just the ones where
someone remembers to ask for it.

## 3. Agent-in-the-loop workflow

This is the protocol we follow at every stage, so "well documented for reproducibility"
is true by construction rather than reconstructed after the fact:

1. **You** set the objective for a stage (e.g. "let's do stage 3 — build the HISAT2 index").
2. **I** propose the exact command(s)/script, with parameters justified against the
   literature or your dataset's characteristics.
3. We **run** it — either directly here for validation on small/toy data, or as a
   ready-to-execute script for your real-scale environment (see §5).
4. **Every command, its rationale, and a summary of its output** gets appended to the
   Command Log (§6) immediately — not reconstructed later from memory.
5. Any decision point (threshold choice, model choice, which replicate design) gets a
   one-line justification in the log, written so it can be lifted almost directly into
   Chapter 3 prose or an Appendix A snippet.

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
│   ├── COMMAND_LOG.md              # append-only, see §6
│   └── environment.yml             # conda environment spec
└── README.md
```

Numbered scripts double as the pipeline's execution order and as a natural table of
contents for Appendix A.

## 5. Compute & execution note

This chat session's sandbox has no network access to NCBI SRA, Ensembl, or an HPC
scheduler — only package registries (PyPI, npm, GitHub). That means:

- Stages 1 (download) and 3 (HISAT2 index build against full GRCh38, ~4 GB index) need
  to run on **your machine or university HPC/cloud** — I can write and validate every
  script here, but the actual execution of those two stages happens on your end.
- I *can* execute and verify stages 2, 5, 6, and 7 directly in this sandbox, and can
  dry-run the full pipeline logic end-to-end on a small synthetic or chromosome-subset
  dataset to catch bugs before you commit real compute time to the full-scale run.
- Whichever machine actually runs each command, its exact invocation still goes in the
  Command Log — reproducibility doesn't care where a command executed, only that it's
  recorded.

## 6. Command log

Append one entry per command, in order, as we work. Template:

```
### [Stage N] <short description> — YYYY-MM-DD
**Command:**
`<exact command>`
**Rationale:** <why these parameters/tool>
**Output:** <one-line summary, e.g. "18.2M read pairs, 94.3% overall alignment rate">
**Dissertation section:** §3.X
```

*(Log starts empty — first entry gets added the moment we run stage 1.)*

## 7. Decisions log (resolved)

- **Execution environment:** your own laptop/desktop.
- **Starting point:** stage 1, dataset selection & download.
- **Dataset selected:** ArrayExpress/ENA **E-MTAB-1147** (Zarnack et al., 2013) —
  HNRNPC knockdown vs control in HeLa cells. Paired-end, human, 2 conditions
  (Control, HNRNPC knockdown "KD1") x 2 biological replicates each — satisfies
  rMATS's replicate requirement out of the box, and the splicing changes it
  produces are well characterised in the literature, which gives us a way to
  sanity-check the pipeline against known biology before trusting it on
  anything novel. Same runs are also fetchable via NCBI SRA's `prefetch` under
  the identical ERR accessions. Full rationale and search trail in
  `COMMAND_LOG.md`.
- **Reference genome:** pinned to **Ensembl release 116** (GRCh38.p14, June
  2026) rather than "latest," so the exact annotation is always reproducible.

## 8. Open decisions still ahead

- Confirm the exact run accession → condition/replicate mapping from the ENA
  table (script `01a_fetch_run_metadata.sh` — first thing to run).
- Whether to also use the study's second knockdown construct (KD2) as an
  independent validation set later, or keep it out of the training data
  entirely.
