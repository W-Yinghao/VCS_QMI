# V8-05 — confirmation read request — CONFIRMED 2026-10-15T17:31:20Z by the main session on the owner's explicit approval of 2026-10-15 (verbatim: 「批准读取 GenAI-Bench 850 条非 Rank 提示的人工标签，作为 V8 的一次确认评价（V7 不使用）。」); one read, frozen models and selection rule, all seeds, full reporting, no tuning feedback. (Drafted by the V8-confirmation fork; original header replaced.)

**What would be read.** The `HumanRatings` column of the 12 GenAI-Bench parquet shards (`/projects/EEG-foundation-model/yinghao/data/genai_bench/data`,
sha256 of each shard recorded by the dry-run; concatenated sha256 in the ledger row), restricted to the 850 prompts without a GenAI-Rank text match
(ids sha256 `ee8cb0460bc52fd6…`, 5,100 images).  Nothing else: the 750 overlap prompts, TIFA160 and RichHF stay unread.

**Why it is allowed once approved.** V8-03 (`V8_03_PROTOCOL_FROZEN.yaml`, frozen 2026-10-14T23:21:07Z) registered these prompts as the untouched
confirmation source before V7's first GenAI-Rank read and before any V8 result.  No V8 unit, lr, epoch, aggregation or readout was chosen with any
GenAI label (lr / epoch by CAL_SELECT); V7 never scored on these prompts.  The owner's review (2026-10-15) recommends approving this read as V8's one
confirmation evaluation and not using it for V7 selection or for any further V8 tuning.

**What is scored** (`V8_05_CONFIRMATION_PROTOCOL_*.yaml`): the 24 frozen V8 units (seed means of raw μ; all seeds) and the four frozen external
systems (FGA released, VQAScore, TIFA local, DSG local) on pooled image-level Spearman with the mean human rating (primary family, Holm over 4) and
within-prompt pairwise preference accuracy (secondary family, Holm over 4); U-CS vs U-BCE (attribution, 2); the other arms descriptive; prompt-cluster
bootstrap 5,000, seed 2718; images missing in any method dropped for all and listed.

**Procedure.** `sbatch slurm/v8/v8_confirm.sbatch gate` (tests + dry-run, no label read) → owner go → this file marked CONFIRMED and the protocol frozen
→ `sbatch slurm/v8/v8_confirm.sbatch run <new OUT>` (ledger row, one read, `confirm.{json,md}`) → results-only commit → report section.  No prediction,
readout or denominator changes after the read; the result is reported in full whatever its sign and feeds no further tuning.

**TIFA160** (secondary, separate approval): not covered by this request; its images are not on the server (see the protocol §B).
