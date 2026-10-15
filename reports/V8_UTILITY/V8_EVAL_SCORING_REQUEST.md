# V8 evaluation — scoring request (DRAFT by the V8-eval fork, 2026-10-15; the main session confirms and commits it before the read)

**What is read.** The sealed GenAI-Rank human-rating file (`/projects/EEG-foundation-model/yinghao/data/genai_rank/closed/genai_rank_labels_sealed.jsonl`,
sha256 `0eb955b4…` = the seal report), a second time after V7C-04 (2026-10-15).  Nothing else: GenAI-Bench, TIFA160, RichHF and EvalMuse test labels
stay sealed / unread.

**Why it is allowed.** V8-03 (`V8_03_PROTOCOL_FROZEN.yaml`, frozen 2026-10-14T23:21:07Z, before V7's first GenAI-Rank read) pre-specifies
`evaluation.primary_external` on GenAI-Rank as a follow-up comparison on the same benchmark (package docs/06 §1 case 1) and registers the GenAI-Bench
non-Rank prompts as the untouched confirmation source.  No V8 unit, lr, epoch or readout was chosen with any GenAI label; the lr rule and the epoch
selection use CAL_SELECT only.

**What is scored (each family = one call of the frozen `scripts/v7/v7_score.py rank`, sha256 `e86af47e…`, identical to the V7 worktree copy).**
- primary: U-CS (seeds 0–2, seed-paired) vs B0 FGA released, B1 VQAScore, B4T TIFA, B4D DSG × DALLE_3 / SDXL_Base, N = 9, official first-index ties,
  prompt-cluster bootstrap 5,000 (seed 2718), Holm over 8.  External predictions = the V7X-50 v3 frozen files (sha256 checked against the manifest).
- same_conditions: U-CS vs U-RPS, U-MSE, U-RANK, U-LIST × 2 generators, Holm over 8.
- attribution: U-CS vs U-BCE, 2 contrasts.
- full_rows: U-CS vs U-CE, U-L1 (4; descriptive), plus every method's N1 / N3 / N9 selected and regret.
- sensitivities (not pre-specified in V8-03; labelled as such): the V7X-50 v2 common-defined subset (prompts with a flagged failure in an external method
  removed) for the primary and same-conditions families.
- The distribution readouts (CAL_PROB / DEV) and the lr / epoch table read no GenAI label.

**Procedure.** `sbatch slurm/v8/v8_eval.sbatch run <new OUT dir>`: the job appends the ledger row (`LABEL_ACCESS_LEDGER.csv`, V7's columns), verifies
the label-file sha256, the external-file sha256s and the scorer sha256, runs the families, deletes its filtered label copy, and writes
`<OUT>/v8_eval.{json,md}`; outputs are never overwritten.  Afterwards: results-only commit of `<OUT>` into `reports/V8_UTILITY/V8_EVAL_results/`, then
the report with every number taken from `v8_eval.json`.  No prediction, readout or denominator changes after the read.

**Dry run before the read.** `sbatch slurm/v8/v8_eval.sbatch gate` (tests + `dry-run`) must report all 24 units present and `ready_for_run=True`.
