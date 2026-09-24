# AGENTS.md

Instructions for any agent (Antigravity or otherwise) working in this repo.
This is a BSc dissertation pipeline — every command that touches data,
alignments, or models must be traceable back to this log for examiners.

## Non-negotiable rules

1. **Follow stage order.** Stages are numbered in `docs/IMPLEMENTATION_STRATEGY.md`
   and `scripts/` (01, 02, 03, ...). Do not start stage N+1 until stage N's
   "Definition of done" checklist in that file is fully satisfied.

2. **Log before you claim done.** Before marking any milestone complete,
   append an entry to `docs/COMMAND_LOG.md` using this exact template:

   ```
   ### [Stage N] <short description> — YYYY-MM-DD
   **Command:**
   `<exact command, verbatim>`
   **Rationale:** <why these parameters/tool>
   **Output:** <one-line factual summary, e.g. "18.2M read pairs, 94.3% alignment rate">
   **Dissertation section:** §3.X
   ```

   One entry per command that produces or transforms data — not one entry
   per stage. If you run five commands to complete a stage, that's five
   log entries.

3. **Never silently change a parameter from what's documented.** If a
   command in `docs/IMPLEMENTATION_STRATEGY.md` needs to change (tool
   version mismatch, memory limit, etc.), say so explicitly, log the
   original attempt AND the change with a one-line reason, in that order.

4. **Pin, don't chase "latest."** Reference genome, tool versions,
   package versions — all pinned in `docs/RESEARCH_ARCHITECTURE.md`. If a
   pinned version is unavailable, stop and ask Brooklyn rather than
   silently substituting.

5. **Flag before expensive operations.** Anything downloading >1 GB,
   running >30 minutes, or using >8 GB RAM: state what you're about to do
   and why before running it, even in review-driven mode.

6. **Validate before advancing.** Every stage has numeric acceptance
   thresholds in `docs/IMPLEMENTATION_STRATEGY.md` (alignment rate,
   FDR cutoffs, accuracy/F1 targets). If a run falls outside them, stop,
   log what happened, and report rather than adjusting thresholds to make
   it pass.

## Where things live

- `docs/RESEARCH_ARCHITECTURE.md` — full pipeline design and rationale.
- `docs/IMPLEMENTATION_STRATEGY.md` — the mission-by-mission playbook, one
  mission per stage, with exact commands and definitions of done.
- `docs/COMMAND_LOG.md` — the append-only reproducibility log (see rule 2).
- `scripts/` — numbered, one file per pipeline step.
- `data/`, `results/` — pipeline inputs/outputs, gitignored if this becomes
  a git repo (raw FASTQ and BAMs don't belong in version control).
