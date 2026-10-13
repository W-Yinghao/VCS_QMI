# V7X-00 — v2 intake delta (external benchmarks) — 2026-10-14

Package: owner's `VCS_CVPR_V7_Server_v2_External_Benchmarks_20261014` (read in full: delta, README, agent prompt, docs 02-15, configs, literature,
reference code and tests, schema, templates).  This file records where the server stands against it; nothing here is a result.

## 1. HEAD, branches, and the 404 the package saw
- `main` at 4a51e52 (R7-50 phase F report); V7 lives on branch **`v7/t2i-alignment`** (git worktree `ssl_pilot_v7`), pushed as
  `origin/v7/t2i-alignment` (b40ab0a at this writing).  The package's 404 for `reports/V7` was on `main`; the V7 reports are on the branch.
- Since aeb56b5 (the package's G09): 14 commits on the branch — V7-00 intake, V7-01 audit, the parser, B0 / B1 plumbing, the evaluators, the
  owner-approved sealed split of GenAI-Rank, V7-03 frozen (c346a9e), note n1 (ab797e2, b40ab0a).
- v2 does not change the v1 losses, inputs or initial weights: the 12 formal + 9 tuning units continue unchanged (nothing is rerun).

## 2. Label and test exposure register (2026-10-14)
| set | status | who may read it |
|---|---|---|
| EvalMuse `test.json` | closed: downloaded, sha256 only, never opened | nobody (no labels published for test) |
| EvalMuse DEV (prompt-disjoint holdout of public TRAIN) | labels inside the derived manifest; read by no decision so far | `v7_score.py table-a` after all 12 formal units and every export exist (V7-03); the M export writes DEV metrics as a by-product, unopened (note n1) |
| GenAI-Rank `human_rating.json` | sealed 0400 after the owner-approved split (prompts only extracted); source sha256 b4d812ec…, sealed file sha256 0eb955b4… (V7_OWNER_DECISIONS; 800 prompts, 14,400 rows) | `v7_score.py rank --allow-scoring` after the owner's separate scoring approval |
| GenAI-Bench ratings, TIFA160, RichHF, NegBench, Winoground | not downloaded | would need the owner's decision (download closed + a later read approval) |

## 3. Running and queued work (2026-10-14 ~00:30 UTC)
- V7 v1: N0 seeds 0-2 running (1038695-97); tuning M0 lr 5e-6 / 1e-5 running (1038698-99), 7 tuning units queued; the driver then selects lr
  on CAL_SELECT and submits the 9 formal M units; B0 (1038708 / 1038709), B1 (1038710) and the second parser run (1038724) queued.
- R7 / REV8 (main): REV8 Qwen diagnostics done (12 / 12); LLaVA diagnostics running; E2 selectors running; R7-50 phase F scored and reported.
- Authorisations in force: V7-03 protocol; owner-approved sealed split (scoring needs a separate approval); standing GPU autonomy
  (2026-10-11) for useful experiments on free GPUs; R7-50 official-test approvals (unaffected).

## 4. What v2 adds and where it is planned
| task | file | state |
|---|---|---|
| X00 intake delta | this file | done |
| X10 legacy coverage | `V7X_10_EVIDENCE_COVERAGE.csv` (32 rows) + `scripts/v7x/x10_legacy_inventory.py` (recomputes the cited P85 counts from the raw JSONs, job 1038759) | done; gaps below |
| X11 classical CS / KDE / RuLSIF / uLSIF reference checks | reference estimators adopted into `scripts/v7x/` with the package tests; new controlled-data anchor cells | code next |
| X12 native non-CS table | reuse corrected P85 / P151_r4 / P152 rows; new cells only where V7X_10 says missing | after X11 |
| X20 common features | `V7X_20_COMMON_FEATURE_PROTOCOL.yaml` (draft; freeze after the extraction smoke) | code next |
| X21 15 common-feature fits | same yaml | after X20 |
| X30 public scorers | `V7X_30_EXTERNAL_BASELINES.md`; primary family by `V7_03_AMENDMENT_V2_20261014.yaml` (frozen) | B0 / B1 queued; TIFA / DSG / PN-VQA / B2 / B5 / B6 to install |
| X40 tables | `reports/V7/templates/V7X_TABLE_{A_CS,B_NONCS,C_SYSTEMS,D_ATTRIBUTION}_TEMPLATE.csv` (empty, status planned_not_run) | templates done |

## 5. Legacy coverage: what the old work does and does not cover (facts from V7X_10; recomputed counts from job 1038759)
- **Classical CS-KDE (A0)** exists only in P85 / P86 (4 bandwidth multiples), with its own truth only on the 55 Gaussian cells (truths
  1.07-6.56).  Against it, multiples 0.5, 1 and 2 are far below the truth in every cell (value / truth at most 0.17, 0.03 and 0.003); the
  smallest multiple 0.25 spans 0.06-1.84 times the truth (over- and underestimates); no SE exists.  New anchor cells are needed.
- **Kernel same-target rows (S-KDE, RuLSIF-tanh, RFF)** were selected at a grid edge: S-KDE at the smallest bandwidth on both axes in 85 of 85
  cells, RuLSIF with the largest basis in 83 of 85 and the smallest ridge in 76 of 85, RFF with the largest basis in 81 of 85.  These rows may be
  under-tuned; v2's one pre-declared boundary extension was never run.
- **Never run:** uLSIF / LSMI; MMD and HSIC on the controlled generators; RuLSIF with its own basis and CV (script committed, not run); any trained
  estimator at independence (I = 0).
- **Corrected artefacts to use:** SMILE only from the smile_fix rows; JS-family / S-KDE J fields only from P151_r4 (r3 superseded); InfoNCE bound
  log 1024 (not log 256).  One stale public figure (`reports/figures/fig_decisive_estimator.*`, main) still reads the superseded P151_E0 fields:
  to be regenerated from P151_r4 with a Corrections note (my fix, next).

## 6. Resources
Measured (V7-03 smokes, L40S): M step 0.40-0.45 s (batch 14), peak 8.3 GB; N0 0.63 s / step, peak 8.5 GB, checkpoints 5.6 GB.  Estimates for v2:
X20 ~0.5-1 GPU-h (one job); X21 24 small MLP fits + 125 kernel configurations in 1-2 GPU jobs; X11 anchors < 2 GPU-h; TIFA / DSG / PN-VQA
~2-5 GPU-h each; downloads ~120 GB (1.9 TB free on /projects; guard 1.2 TB).  Every estimate is replaced by a measured smoke before a freeze.

## 7. For the owner (only what the data and the existing approvals cannot settle)
1. **Paid APIs for the native question-generation routes** (TIFA GPT-3.5, DSG GPT-3.5 / PaLM 2, PN-VQA GPT-4).  Default without approval: TIFA's
   released local LLaMA-2 generator (an official route), DSG and PN-VQA with a local LLM (Qwen2.5-32B-Instruct), labelled as adaptations.
2. **GenAI-Rank scoring approval** after every prediction of the v2 primary family is in the manifest (unchanged from v1).
3. **Optional extra terminal sets** (GenAI-Bench ratings, TIFA160, RichHF): not downloaded; each would need a download decision and a later read
   approval.
