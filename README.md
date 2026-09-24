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
