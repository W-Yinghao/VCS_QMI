# V81_00_INTAKE — V8.1 server diagnostics, intake and inventory (D00)

Package `VCS_V8_1_Server_Diagnostics_v1_20261015` (copied to `v8_1/package_v81_20261015/`, commit 8aaefea on branch `v8.1/utility-diagnostics`,
worktree `/home/infres/yinwang/CS_QMI/ssl_pilot_v81`, branched from V8 b8ac563).  Package reference tests: 66 passed in the CPU env before any
diagnostic was run.  This file follows `templates/INTAKE_CN.md`; machine inventory in `results/inventory.csv` (48 rows) and `results/fork1_facts.json`,
written by `scripts/v81/v81_fork1_audit.py` (reads log.json / done.json / pred row counts / export.json / file sizes and the 12 SLURM logs; no
checkpoint is deserialised, no label file is opened).

- **实际UTC**：audit run 2026-10-15T18:32Z–18:49Z on `nodecpu10` (CPU).  Training env `envs/v7_fga`: torch 2.9.0+cu128, numpy 1.26.4, scipy 1.15.3,
  transformers 4.33.2 (as used by the 48 units).  Diagnostic env `CS_QMI/env`: python 3.12.12, numpy 2.5.2, scipy 1.18.1 (package reference code).
  SLURM accounting (`sacct`) was unreachable at audit time ("failed to open persistent connection"); job facts below come from the job logs.
- **main/V8/V9 HEAD与新工作树**：main `55d47c9` (= package baseline), V7 `2dc39b5` (= package baseline), V8 `b8ac563` (package delivery note says
  `580781f`; two later commits, both TIFA160 secondary tooling / execution note, no trainer or evaluator change), V9 `953421d` (moved on from the
  package's `1ce0e84`; V9 is not used by V8.1).  New worktree `ssl_pilot_v81`, branch `v8.1/utility-diagnostics` at 8aaefea + this work.
- **对本包来源截点的增量**：V8 `a87b2d5` → `580781f` (V8-05 confirmation, already in docs/09) → `544ce9b` (TIFA160 dry-run / run tooling, no
  label read) → `b8ac563` (execution note B1: TIFA160 images verified, prediction jobs queued).  The trainer blob `67f9dd5d…` (scripts/v8/v8_train.py)
  named in docs/09 is unchanged at b8ac563; `scripts/v8/v8_core.py` blob `0542e843…` and `v8_labels.py` `4e5c0ce8…` likewise (V81_01).
- **已有同类诊断，可复用的路径**：V8-04 §1 (selected epochs / lr edges), §4 (CAL_PROB / DEV RPS, histogram log loss, mean-MSE per arm) and Table C
  (`outputs_v8/eval_v1/v8_eval.json`, sha256 `1949fb32…`); V8-05 confirmation summary (`V8_05_CONFIRMATION_results/confirm.json`, `f4094a8a…`);
  V7C DEV secondary table with the external systems' DEV predictions (`ssl_pilot_v7/reports/V7_CLOSEOUT/V7C_DEV_results/x40_dev_external.json`,
  `f02f1bfe…`; prediction files under `outputs_v7/x40_*` / `x30_dev`); untrained BLIP-2 features for FIT / CAL / DEV / GenAI-Rank
  (`outputs_v8/features/`, V7X-20 construction) and the kernel-route outputs (`outputs_v8/kernels/`); per-unit predictions with logits / q / u / mu
  for CAL_SELECT, CAL_PROB, DEV in every unit's `pred/`.  Nothing is re-scored here; D02–D04 reuse these files.
- **V8-05状态（只据作业/报告，不重提）**：**completed**.  Run 1042053 (submitted 2026-10-15T17:31:21Z after the owner's approval; dry-run 1042040),
  results-only commit 1aaf61e, report section V8-04 §7 in 580781f, ledger row in `LABEL_ACCESS_LEDGER.csv`.  Not re-submitted.  Secondary §B (TIFA160,
  separately approved): V8 export job 1042056 finished 18:33:13Z (exit 0; `pred/tifa160.jsonl` 800 rows in all 24 formal units, export epoch =
  selected epoch 24 / 24); comparator job 1042057 (VQAScore + TIFA questions on the 800 images) RUNNING at 18:38Z; no TIFA160 label has been read.
  V8.1 does not use GenAI-Bench, GenAI-Rank or TIFA160 labels in any diagnostic (roles table docs/07 §2; reference CLI refuses those roles).
- **48原单元和24正式预测完整性**：48 / 48 units have `done.json`, `log.json` with 5 logged epochs, `best.pt` (802 MiB) and `resume.pt` (2,219 MiB).
  Formal units (24): `images_CAL_SELECT` 1,653, `images_CAL_PROB` 1,630, `images_DEV` 3,257, `genai_rank` 14,400, `genai_bench` 9,600, `tifa160` 800
  rows each — complete 24 / 24.  Tuning units (24): CAL_SELECT / CAL_PROB / DEV / genai_rank complete 24 / 24 (no GenAI-Bench export by design).
  `selected_epoch` agrees between log.json, done.json and `pred/export.json` in 48 / 48.  No unit was resumed (`resumed_at` empty 48 / 48); none is a
  smoke unit.  Job → unit map, host, GPU, start / end UTC in `inventory.csv`.
- **每epoch checkpoint实际保存情况**：only two files per unit — `best.pt` (trainable state at the selected epoch, written when the CAL_SELECT MSE
  strictly improves) and `resume.pt` (trainable state + optimiser + scaler after the last epoch, index 4), both overwritten each epoch (trainer lines
  211–215).  No per-epoch weights, no initial (pre-update) weights, no RNG state.  Consequences: D05 level 2 (checkpoint-level re-selection) is limited
  to best vs final; per-epoch per-image curves are `not_saved`; the true initial state can only be RECONSTRUCTED_INIT (D06).
- **当前已有数据权限决定已沿用**：owner decision 2026-10-15 ("all data already on the server is cleared") recorded in main `OWNER_DECISIONS_20261015.md`;
  no licence check is a gate here.  Label isolation unchanged: GenAI-Bench 850 and GenAI-Rank reads stay one-shot and ledgered; V8.1 opens no label
  file outside the development roles.
- **本轮使用的开发角色与确认隔离**：FIT (fixed FIT-AUDIT subset by `sha256('v81-fit-audit-2718|'+prompt_id)`, D06 only), CAL_SELECT (re-check of the
  frozen selection, log level), CAL_PROB (secondary), DEV (development diagnostics, already exposed in V7 / V8-04).  GenAI-Rank: published aggregates
  only.  GenAI-Bench 850 / TIFA160: published summary only (docs/09); never per-image.  Design-card times are recorded in each report; the V8-05
  confirmation summary was known before D01–D07 ran (docs/09 says the designs are not changed by it; exposure stated, not claimed independent).
- **设备、配额、标准batch实测与推理预算**：queue at 18:38Z: 20 running + 7 pending jobs on the account across all lines (REV10, V9-G1 / G2 / N2,
  TIFA160); partitions L40S / H100 / RTX PRO 6000 all `mixed`.  The standard-batch measurement on the real GPU type is done by the D06 job before any
  re-inference (V81_05); historical, comparable timings from the 48 job logs (not a standard workload): training 33–53 min per unit on RTX PRO 6000 /
  34 min on H100 NVL / 63–71 min on L40S (5 epochs × 2,379 grouped steps + 5 CAL_SELECT evaluations), CAL_SELECT evaluation 17–31 images / s, export
  10–37 images / s (batch 32, 364 px, fp16 autocast), peak training memory 7.8 GB.  Re-inference budget cap per docs/07 §3: ≤ 24 formal models ×
  {best, resume} on FIT-AUDIT (≤ 2,048 images) + CAL / DEV where needed; new full training 0.
- **不能完成的项目及原因（不猜测）**：(1) successful optimiser updates per epoch — unknown: the AMP GradScaler skip count is not logged; the scaler
  state exists only inside `resume.pt`, which D00 / D01 do not deserialise (D06 may report it if it loads resume.pt).  (2) Exact SLURM elapsed /
  requeue history — `sacct` unreachable; start / end from the job logs.  (3) Per-epoch checkpoints — not saved (above).  (4) B0 FGA native-eval
  timing — not recorded separately in V7.  (5) B2 / B5 / B3 external systems — predictions exist in the V7 freeze but were never scored in a frozen
  V8 family; listed as `not_scored`, not scored now.  (6) V8 arms' DEV SRCC / agreement — produced by D02 / D03 (V81_02, V81_03), not duplicated here.
