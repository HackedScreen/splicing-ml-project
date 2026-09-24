# Implementation strategy

How to drive Antigravity through this pipeline. Each stage below is a
**mission**: a prompt you paste into Antigravity's Manager view, a
definition of done the agent checks against before stopping, and a note on
what should land in `docs/COMMAND_LOG.md`. Do them in order — each one
gates the next.

**Recommended Antigravity mode:** Review-driven. This pipeline downloads
real data, runs multi-hour jobs, and produces numbers that go in your
dissertation — you want to approve terminal commands before they run, not
just review a diff afterward.

**Recommended workflow:** run one stage's mission in the Manager view, let
it work (HISAT2 indexing and rMATS in particular can run 1–4 hours — good
candidates for Antigravity's background/async task handling so you're not
watching a terminal), review the artifact it produces, check the
Definition of Done, then move to the next mission.

---

## Stage 0 — Environment setup

**Mission prompt:**
> Set up a conda environment named `splicing-ml` with: sra-tools, fastqc,
> trimmomatic, hisat2, samtools, rmats (bioconda), python 3.11, and
> biopython, pysam, pandas, scikit-learn, matplotlib via pip. Write the
> resolved versions to `docs/environment.yml`. Verify each tool runs with
> `--version` and paste the output back to me.

**Definition of done:**
- `docs/environment.yml` exists with exact pinned versions (not `*` or
  unpinned).
- Every tool responds to `--version` without error.

**Log:** one entry — command used to create the env, rationale ("pinned
versions for reproducibility"), and the full version list as output.

---

## Stage 1 — Data acquisition *(already scaffolded)*

**Mission prompt:**
> Run `scripts/01a_fetch_run_metadata.sh`. Show me the resulting run table
> from `data/reference/E-MTAB-1147_runs.tsv`. Do not download anything yet.

Then, after you confirm the run accession → condition mapping by eye:

> Fill `data/reference/manifest_TEMPLATE.csv` with the confirmed mapping,
> save as `data/reference/manifest.csv`. Then run
> `scripts/01b_download_fastq.sh` with those accessions, and
> `scripts/01c_download_reference.sh`.

**Definition of done:**
- `data/reference/manifest.csv` has real run accessions, no blanks.
- 4 FASTQ pairs present in `data/raw/` (Control ×2, KD1 ×2), each with
  matching read counts between `_1` and `_2` files (`zcat file_1.fastq.gz
  | wc -l` should equal the `_2` count — mismatched pairs mean a corrupt
  download).
- Reference FASTA and GTF present in `data/reference/`, gunzipped.

**Log:** entries for the metadata fetch, the manifest confirmation, and
the actual download (with real read counts — not the placeholder text
currently in `COMMAND_LOG.md`).

---

## Stage 2 — Quality control

**Mission prompt:**
> For each FASTQ pair in `data/raw/`, run FastQC into `results/qc/pre/`.
> Then trim with Trimmomatic (paired-end mode): `ILLUMINACLIP` against the
> TruSeq3-PE adapter file bundled with Trimmomatic,
> `LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15 MINLEN:36`. Write trimmed output
> to `data/trimmed/`. Run FastQC again on the trimmed output into
> `results/qc/post/`. Summarise per-base quality and adapter content
> before/after in a short table.

**Definition of done:**
- Post-trim FastQC shows no adapter content flagged and per-base quality
  "pass" or "warn" (not "fail") across the read length.
- Read retention after trimming >85% (if much lower, flag it — may
  indicate a library prep issue worth noting in Chapter 5 Limitations).

**Log:** one entry per Trimmomatic invocation, with exact retention counts
in the output field. This is Appendix A.1 material — save the exact
command verbatim, it's the dissertation's "A.1 Trimmomatic Command."

---

## Stage 3 — Genome alignment (HISAT2)

**Mission prompt:**
> Extract splice sites and exons from the GTF using HISAT2's
> `hisat2_extract_splice_sites.py` and `hisat2_extract_exons.py`. Build a
> HISAT2 index from the primary assembly FASTA incorporating those splice
> sites (`hisat2-build --ss ... --exon ...`). This will take a while and
> use significant RAM — run it as a background task and let me know when
> it's done. Then align each trimmed FASTQ pair with
> `hisat2 -x <index> -1 <R1> -2 <R2> -p <threads> --dta -S <sam>` (`--dta`
> is required — rMATS needs it downstream). Convert to sorted, indexed BAM
> with samtools. Report the HISAT2 alignment summary (overall alignment
> rate) for each sample.

**Definition of done:**
- Overall alignment rate >85% for every sample (human RNA-seq against
  GRCh38 should comfortably clear 90%; investigate anything below 80%
  before proceeding — likely a trimming or index problem, not just
  biological noise).
- 4 sorted, indexed BAM files in `results/alignment/`.

**Log:** index-build command + wall time, then one entry per sample
alignment with its exact overall alignment rate. This is Appendix A.2 and
feeds Chapter 4.2 (Genome Alignment Statistics) directly — those
per-sample rates are a results table, not just a log line.

---

## Stage 4 — Splicing detection (rMATS)

**Mission prompt:**
> Build `b1.txt` (Control BAM paths, comma-separated) and `b2.txt` (KD1 BAM
> paths). Run rMATS-turbo comparing b1 vs b2: `--gtf <gtf> -t paired
> --readLength 72 --nthread <threads> --od results/splicing/ --tmp
> results/splicing/tmp`. Report how many significant events (FDR < 0.05)
> were found per event type (SE, RI, A5SS, A3SS, MXE) from the
> `*.MATS.JC.txt` output files.

**Definition of done:**
- All five `*.MATS.JC.txt` output files present and non-empty.
- At least a few hundred significant events total (if near-zero, something
  upstream is wrong — check alignment rate and BAM sort order before
  assuming the biology is just quiet; this knockdown is published to
  produce strong splicing changes).
- Class distribution across the five event types recorded — you'll need
  this to judge how imbalanced the ML labels are in stage 6.

**Log:** the exact rMATS command, plus the significant-event counts per
type as output. This is Appendix A.3 and Chapter 4.3 (Splicing Event
Detection) verbatim.

---

## Stage 5 — Feature engineering

**Mission prompt:**
> Write `scripts/07_feature_engineering.py`. For each significant event
> (FDR < 0.05) across all five rMATS output files, build one row of a
> feature matrix with: (a) PSI value and PSI variance across replicates
> from the rMATS output directly; (b) exon/intron length from the event's
> genomic coordinates; (c) GC content of the exon and 100 bp flanking
> regions, computed from the reference FASTA with pysam/biopython; (d)
> local RNA-seq coverage mean and variance across the event region from
> the BAM, via pysam; (e) splice-site strength — if MaxEntScan or a
> comparable PWM tool isn't available, use a simple position weight matrix
> scored against the canonical splice-site consensus as a documented
> fallback, and say so explicitly in the log; (f) label = event type (SE/
> RI/A5SS/A3SS/MXE), taken from which rMATS output file the row came from.
> Save as `results/features/feature_matrix.csv`. Report class balance
> (row count per label) and check for and report any missing/NaN feature
> values.

**Definition of done:**
- `feature_matrix.csv` exists, one row per event, columns matching (a)–(f)
  above plus the label.
- No unexplained NaNs (any that exist are documented with a reason, not
  silently dropped).
- Class balance reported — expect SE to dominate (~40% is typical per your
  own Chapter 2 lit review); this matters for stage 6's evaluation choices.

**Log:** the feature engineering script logic summarised in one entry
(script itself lives in `scripts/`, referenced not pasted), plus the
resulting class balance numbers. Feeds Chapter 4.4 (Feature Matrix) and
Appendix A.4.

---

## Stage 6 — Model development

**Mission prompt:**
> Write `scripts/08_train_models.py`. Load `feature_matrix.csv`. Stratified
> train/test split (80/20) on the label column. Train two models: (1)
> Logistic Regression (multinomial, `class_weight='balanced'` given the
> expected class imbalance), (2) Random Forest. Tune both with
> `GridSearchCV` (5-fold, stratified) over a small, documented
> hyperparameter grid — don't grid-search everything, pick 2-3 parameters
> per model and say why in a comment. Save the tuned models and the
> chosen hyperparameters.

**Definition of done:**
- Both models trained, hyperparameters chosen via CV (not defaults, and
  not hand-picked without justification).
- Test set was never touched during tuning.

**Log:** one entry per model with the final chosen hyperparameters and the
CV score that selected them. Feeds Chapter 3.7 and Appendix A.4.

---

## Stage 7 — Evaluation

**Mission prompt:**
> Write `scripts/09_evaluate_models.py`. On the held-out test set, compute
> for each model: accuracy, precision/recall/F1 (both macro and weighted,
> given class imbalance), ROC-AUC (one-vs-rest), and a confusion matrix.
> For Random Forest, report feature importances; for Logistic Regression,
> report coefficients. Save all metrics to
> `results/models/evaluation_report.md` and plots (confusion matrix,
> feature importance) to `results/models/`.

**Definition of done:**
- Both models evaluated on the same untouched test set.
- Report explicitly states whether the dissertation's target (accuracy
  >80%, F1 >0.78, per the Abstract) was met — and if not, that's a
  results-section finding, not a failure to fix by re-tuning against the
  test set.

**Log:** final metrics for both models, verbatim. This is the core of
Chapter 4.5 and directly answers your three research questions from
§1.3 — worth cross-referencing explicitly in the log entry.

---

## After stage 7

`docs/COMMAND_LOG.md` should now be a near-complete draft of Chapter 3's
method narrative and Chapter 4's results tables. Chapter 3 prose is
largely "turn each log entry's rationale into a paragraph"; Chapter 4 is
largely "turn each log entry's output into a table or figure." That's the
payoff of logging as you go instead of reconstructing it afterward.
