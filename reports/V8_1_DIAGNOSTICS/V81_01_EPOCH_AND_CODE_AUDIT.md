# V81_01_EPOCH_AND_CODE_AUDIT — epoch semantics, weight state and source audit (D01)

Scope: the 48 V8 units (24 tuning + 24 formal).  Evidence: the trainer source at the four commits the jobs actually ran at (`git show`,
snapshots under `/projects/EEG-foundation-model/yinghao/outputs_v81/source_snapshots/`), the 48 `log.json`, the 12 SLURM job logs, `done.json`,
`pred/export.json` and `pred/tifa160_export.json`.  No checkpoint was deserialised (`results/audit_logs.json`: `checkpoint_bytes_read: false`).
Per-unit rows with every docs/03 §2 field: `results/epoch_audit.csv`; the package tool's own output: `results/audit_logs.json`
(`reference/audit_logs.py`, 48 / 48 `after_logged_epoch_if_snapshot_matches`, `not_equivalent_to_preupdate_init: true` 48 / 48).  Numbers below are
copied from `results/fork1_facts.json` / `epoch_audit.csv`.

## 1. Which code ran (source_commit / snapshot_hash)

Every job prints `commit=<HEAD> code=<sha256 of the snapshotted scripts/v8/*.py, 16 hex>` at start and trains from that snapshot (wrappers
`slurm/v8/v8_formal.sbatch`, `slurm/v8/v8_group.sbatch`).  The four code hashes were **recomputed from the committed trees and all four match**:

| jobs | units | HEAD at job start | code snapshot hash (logged = recomputed) | trainer blob `scripts/v8/v8_train.py` | `cmd_train` (lines 164–220) sha256 |
|---|---|---|---|---|---|
| 1040713 (TUNE U-CS, U-BCE × 3 lr) | 6 | 4b215f3 (2026-10-15 01:28 CEST) | 744672e56d1ede09 | 5c4cba6d… | 974560c26fee… |
| 1040714 / 1040715 / 1040716 (TUNE RPS, CE / MSE, L1 / RANK, LIST) | 18 | ca06687 (02:19 CEST) | 37bcdfa770eb121a | 67f9dd5d… | 974560c26fee… |
| 1040811–1040817 (FIT CS, BCE, RPS, CE, MSE, L1, RANK × seeds 0–2) | 21 | cc6299f (06:12 CEST) | e9532e115d2769e2 | 67f9dd5d… | 974560c26fee… |
| 1040818 (FIT LIST × seeds 0–2) | 3 | a59c833 (14:17 CEST) | 605aea42c2cbd141 | 67f9dd5d… | 974560c26fee… |

- The only trainer difference across the four commits is 4b215f3 → ca06687: `cmd_features` (Table C feature export) and its sub-parser were added
  *after* `cmd_train`; the `cmd_train` function is byte-identical at all four commits (same sha256, same line span 164–220), and `v8_core.py`
  (blob 0542e843…, losses, batch plan, schedule) and `v8_labels.py` (4e5c0ce8…) are identical at all four.  The code hashes differ only because
  other files in `scripts/v8/` (evaluator, kernels, lr rule) changed between commits.  **Conclusion: all 48 units were trained by one and the same
  training loop; the V8-04 numbers need no code-version caveat.**
- Job 1040818 (U-LIST) started at 12:40:35Z, after its sibling jobs (10:23–10:57Z), at the later commit a59c833; its three units show no resume
  (`resumed_at` empty) and five fresh epochs, i.e. they trained from scratch in that run.  Whether the job id had an earlier aborted start cannot be
  read from the logs (the log file holds one start) and `sacct` was unreachable; the main session's own record says the pending job was requeued
  before it started.  No consequence for the audit: same trainer blob, same batch plans (seeded by (seed, epoch), not by time).
- The package's docs/09 statement that the trainer blob is `67f9dd5d…` is confirmed for 42 / 48 units; the remaining 6 (tuning U-CS / U-BCE) ran the
  earlier blob `5c4cba6d…` whose `cmd_train` is identical.

## 2. What the training loop does, in source order (`v8_train.py` @ cc6299f, line numbers)

| lines | step |
|---|---|
| 85 | freeze mask: `requires_grad` only for `Qformer.*`, `ln_vision.*`, `query_tokens`; the new `v8_head` (Linear 768 → 5, created at 86–89) is trainable by construction; 186,314,048 trainable parameters in every unit |
| 87–89 | head weight ~ N(0, 0.02) from `torch.Generator().manual_seed(seed·7919 + 5)`, bias = log FIT natural histogram (identical in all 48 logs: `head_bias_init_shared: true`) |
| 100 | the frozen EVA ViT-g runs under `torch.no_grad()`; 106: the head runs in fp32 outside autocast |
| 111 | `set_train`: `model.train()` then `model.visual_encoder.eval()` (the Q-Former's dropout is active in training; the ViT stays in eval) |
| 125 | `predict`: `model.eval()` for every CAL / export pass |
| 181 | `plans = [batch_manifest(fit, seed, e) for e in range(5)]` — the batch plan of every epoch is fixed before training (5 plan hashes in every log) |
| 186–189 | resume block (never taken: no `resume.pt` existed at start in any unit) |
| 191 | `for epoch in range(start, E)` |
| 194 | `set_train(model)` at the start of every epoch |
| 196–204 | per batch: lr = `warmup_cosine(step, total, lr, 100)`; fp16 autocast forward; `batch_loss`; `scaler.scale(loss).backward(); scaler.step(opt); scaler.update()`; `step += 1`, `seen += batch size` |
| 207 | **after the full pass**: `cal_metrics(predict(model, CAL_SELECT))` |
| 208–210 | epoch record `{epoch, steps, images_seen, train_s, eval_s, train_means, cal_select, peak_mem_gb}` appended to `epochs_log` |
| 211–213 | if CAL_SELECT mean-rating MSE **strictly** improves: `best = (mse, epoch)`; `best.pt` ← trainable state (ties keep the earlier epoch) |
| 214–215 | `resume.pt` ← trainable state + optimiser + scaler + `best` + log (overwritten every epoch) |
| 216–219 | `selected_epoch = best[1]`; `log.json`; `done.json` (only for full runs: `max_steps` 0 and 5 epochs) |

Therefore **`epochs_log[k]` / `selected_epoch = k` describes the model after k + 1 complete passes over FIT**.  There is no evaluation and no
checkpoint before the first optimiser update anywhere in the loop.

## 3. Evidence table (docs/03 §2 fields; aggregate over the 48 units, per unit in `epoch_audit.csv`)

| field | value |
|---|---|
| source_commit / snapshot_hash | §1; reproduced 48 / 48 |
| epoch_index (0-based, as logged) | formal: {0: 8 units, 1: 3, 2: 10, 3: 2, 4: 1}; tuning: {0: 2, 1: 15, 2: 6, 4: 1}; per unit in the CSV (V8-04 §1 table unchanged) |
| completed_epoch_count | epoch_index + 1 (1 to 5 full passes) |
| planned_batches / attempted_steps | 2,379 per epoch in every unit and epoch (`steps_per_epoch_set: [2379]`); `total_steps` 11,895 = Σ logged steps 48 / 48; 5 batch-plan hashes 48 / 48 |
| successful_updates | **unknown** — the AMP GradScaler's skipped-step count is not logged; `scaler.step` is called once per batch (line 203); the scaler state lives only in `resume.pt` (not read here) |
| images_seen | 26,177 = n_fit in every epoch of every unit (`all_images_complete: true`); no smoke flag, no truncation (`max_steps` 0) |
| resume_history | empty 48 / 48 (no unit ever resumed) |
| checkpoint_stage — `best.pt` | AFTER_EPOCH (index = selected_epoch); `export.json` and `tifa160_export.json` report the same epoch 48 / 48 and 24 / 24 |
| checkpoint_stage — `resume.pt` | AFTER_EPOCH (index 4, the last pass; inferred from the loop, file not opened) |
| INIT_PREUPDATE | not saved; cannot be recovered bitwise |
| RECONSTRUCTED_INIT | possible per seed: `build(seed, fit_hist)` is deterministic given the upstream checkpoint bytes and the seed (head from the seeded generator, bias from the FIT histogram, Q-Former from the fixed upstream file); the eight arms share it at a seed (D06, labelled RECONSTRUCTED_INIT) |
| lr at the selected checkpoint | schedule: linear warm-up over 100 steps, then cosine to 0 over the remaining 11,795 steps; at the end of epoch index 0 the lr is 0.91 × peak, index 1: 0.66, index 2: 0.35, index 3: 0.10, index 4: 0 (`curves.csv`, `lr_end_of_epoch`) |
| cross-checks | the `[v8_train] epoch k: CAL_SELECT mean-rating MSE …` lines printed in the SLURM logs match `epochs_log` (48 / 48); the step prints every 200 steps count 12 per epoch = ⌈2,379 / 200⌉ (48 / 48) |

Module modes and AMP (docs/03 §4): training = Q-Former / ln_vision / query_tokens / head in `train()` with the ViT in `eval()`; prediction fully
`eval()`; the forward is fp16 under autocast with the head and all losses in fp32.  This is the intended V8 design; D1's phase-B dropout problem has
no counterpart here (the Q-Former is *meant* to train).  The D06 gradient checks (V81_05) use eval-mode autograd on this same construction.

Ranking arms (docs/03 §5 facts, from `train_means`): U-RANK / U-LIST see on average 2.0 scored subgroups and 25.5 unordered pairs per optimiser
step (GROUP_MAX 7, BATCH_MAX 14, tau 1, lambda 1); the point term and the ranking term are logged separately (`curves.csv`).  Their gradient
magnitudes / angles are D06's job; nothing here supports changing lambda.

## 4. Correction to V8-04 (interpretation only; no number, selected epoch or conclusion changes)

V8-04 §1 (line 24) reads: *"Seeds 1 of several arms selected epoch 0 (the initialised head with the natural-histogram bias)"*.  With the loop above
this is wrong: **epoch index 0 is the checkpoint after the first complete pass over the 26,177 FIT images (2,379 grouped optimiser steps, warm-up
finished at step 100, lr still at 0.91 × peak when the checkpoint was written)**, not the initialised head.  The eight formal units that selected
index 0 (U-CS-s1, U-RPS-s1, U-MSE-s1, U-L1-s1, U-CE-s2, U-RANK-s0 / s1 / s2) therefore are trained models whose CAL_SELECT mean-rating MSE was
lowest after one pass; their later epochs are worse on that rule by 0.0003–0.018 (trained arms) and by 0.03–0.38 (U-RANK) — see `results/selection_gap.md`.
Nothing in V8-04 §2–§7 used the "initialised head" reading, so the tables, the Holm decisions and the V8-05 confirmation stand.

Proposed text for the V8-04 **Corrections** list (to be applied by the main session as item 3, numbers unchanged):

> 3. §1 — "epoch 0 (the initialised head with the natural-histogram bias)" is wrong.  The trainer (`scripts/v8/v8_train.py`, blob 67f9dd5d, lines
> 191–215; identical `cmd_train` in every job) trains a full pass, then evaluates CAL_SELECT and saves; `selected_epoch = 0` is the checkpoint after
> one complete pass (2,379 steps over all 26,177 FIT images, lr 0.91 × peak at the save point), i.e. a trained model.  The initial head was never
> evaluated or saved.  The same applies to U-RANK's "epoch 0 selected in all three seeds" (§1): after one pass, with its CAL mean-rating MSE rising
> in every later epoch.  Audit: `reports/V8_1_DIAGNOSTICS/V81_01_EPOCH_AND_CODE_AUDIT.md` (branch v8.1/utility-diagnostics).

V8-03 (`V8_03_PROTOCOL_FROZEN.yaml`) needs no change: its selection rule ("CAL_SELECT prompt-macro mean-rating MSE of raw mu; ties → earlier epoch")
is what the code does; it never promised an epoch-0 = initialisation reading.

## 5. What this audit does not establish
- Whether every one of the 11,895 `scaler.step` calls applied an update (fp16 overflow steps are skipped silently by the GradScaler); the count is
  unknown, not zero.  The monotone decrease of every native training loss over the five epochs (48 / 48, `selection_gap.md`) shows that updates
  happened in every epoch, not that none was skipped.
- Bitwise reproducibility of any unit (RNG state not saved; `persistent_workers=False`, 8 workers).  RECONSTRUCTED_INIT reproduces the *initial*
  state, not the trajectory.
- Anything about DEV / GenAI behaviour of the epoch-0 units; that is D02–D04's and the frozen reports' domain.
