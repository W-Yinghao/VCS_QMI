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

## TIFA160 CONFIRMED 2026-10-15T19:33:12Z — secondary external evaluation (protocol §B), separate from the GenAI-Bench read above

Confirmed by the main session's TIFA160 fork on the owner's explicit approval of 2026-10-15 (verbatim: 「2. 批准读取 TIFA160 标签，作为冻结 V8 模型的次级外部评价。」;
recorded in `reports/OWNER_DECISIONS_20261015.md` on main).  One read, frozen V8 models and selection rule, all seeds, full reporting, no tuning feedback.

**What would be read, once.** DSG's `tifa160-likert-anns.csv` (sha256 `61c9ff41b9f9fc26…`) and TIFA's `human_annotations/human_annotations.json`
(sha256 `1e41c66e78b7f1e7…`), both sealed under `third_party/closed/` and equal to `SHA256SUMS_v7x_20261014.txt` (job 1042215).  Nothing else in the sealed
folders is opened (`human_annotations_with_scores.json`, the folder `README.md` and `compute_correlation.ipynb` stay unread).

**What is scored.** The 24 frozen V8 units (seed means of raw μ; all seeds) and the external routes present on all 800 images — B0 FGA released (native
harness), B1 VQAScore CLIP-FlanT5-XXL, B4T TIFA local — on the 800 TIFA160 images (inputs sha256 `2160cf9ce1d22b5c…`).  B4D DSG local is not run (exposed
through DSG's TIFA160 in-context examples; prepared, outside the GPU budget) and is reported as absent.  Correlations: Spearman (primary), Pearson, Kendall
tau-b with the rater mean (TIFA 2-rater and, separately, DSG 5-rater); families `tifa_primary` (U-CS vs the 3 comparators present, Holm 3),
`tifa_attribution` (U-CS vs U-BCE), same-data arms descriptive; bootstrap over the 160 text inputs (5,000, seed 2718).

**Reading rule (protocol §B).** 160 prompts → intervals reported, no claim language in either direction; a secondary evaluation added 2026-10-15, not the
registered confirmation source; no V7 method is scored.

**Gate.** Job 1042215 on commit 1793e62 (clean tree): tests/v8 56/56; `tifa160-dry-run --hash` ready_for_run=True (24/24 units and B0 / B1 / B4T at 800/800
images, B4D absent, seals matched).  Synthetic-label smoke job 1042216 on the same commit (real predictions, synthetic labels in both public formats; no
sealed file opened): all 12 checks passed (both label formats parsed for 800/800 images; families, Holm and levels present; 193 s for the full run).

**Procedure.** `sbatch slurm/v8/v8_confirm.sbatch tifa160-run /projects/EEG-foundation-model/yinghao/outputs_v8/confirm_v1/tifa160/run_v1` (ledger row with
both sealed hashes, then the one read) → results-only commit → report section labelled "secondary external evaluation added 2026-10-15".
