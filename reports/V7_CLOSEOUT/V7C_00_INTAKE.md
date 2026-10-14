# V7C-00 — Intake of the V7 supplement v3 closeout package (2026-10-14)

## 1. Branches and package
| item | value |
|---|---|
| main HEAD | `2d38ea35e8e6e178f580e5bbb052fc37e05230c7` (the package's last-read main, unchanged since) |
| v7/t2i-alignment HEAD at intake | `f44624edbc723f1c82badfc669ff32d9dfe338e6` (later than the package's `4f90d6d`; adds V7X-50, the GenAI-Rank prediction freeze v1 / v2, see §3) |
| package | `VCS_V7_Supplement_v3_Closeout_20261014.zip`, sha256 `b7a87a0c30483653a57e73dd989832b23631e31b3f960c1de4618bcda9c45e06`; unpacked read-only to `outputs/V7/package_ref_v3` (every file matches its SHA256SUMS) |
| output root | `reports/V7_CLOSEOUT/` (the original `reports/V7/` reports are not overwritten) |

## 2. Task status (by files, schema and logs; a plan is not counted as a result)
| ID | status | evidence |
|---|---|---|
| Formal units | **complete, reused**: M0 / M1 / M2 × seeds 0–2, N0 × seeds 0–2; 9 TUNE units | `outputs_v7/units/*` (checkpoints, CAL_SELECT logs, DEV and GenAI-Rank exports) |
| Table A | **complete** | `reports/V7/V7_TABLE_A_results/` |
| X11 / X21 (main, PCA64, PCA256, N0-s0 features) | **complete** | `reports/V7/V7X_X11_results/`, `V7X_X21_results/` |
| GenAI-Rank predictions, all methods | **complete and frozen (v2)**: 30 method files incl. the primary family (M2-E × 3, B0, B1, B4T, B4D) | `reports/V7/V7X_50_RANK_PREDICTION_FREEZE_v2.md` |
| DEV external rows | partly: B0 / N0 native, B2, B5 HPSv3, PN-VQA (own / EvalMuse elements), CLIPScore done; **B1, TIFA, DSG on DEV pending** in group `v7x_grp_dev_A` (H100) | `outputs_v7/x30_dev/`, `native_eval/*/native_eval_DEV.json` |
| GenAI-Bench predictions (labels sealed) | all done except **DSG** (group `v7x_grp_bench_dsg`, H100) | `outputs_v7/x30_bench/` |
| V7C-01 linkage | not started: needs E_gt / E_auto / O / H per CAL / DEV candidate; E_auto needs a GPU re-export with the fixed parser's elements | — |
| V7C-02 tail / parser | not started (same exports) | — |
| V7C-03 readiness | **in progress**: completeness (freeze v2) and TIFA / DSG structure checks done; source-fidelity generation queued (job 1040534) | `V7C_03_EXTERNAL_READINESS.md` (to come) |
| V7C-04 scoring request | to be written after C03; the owner's scoring approval was given 2026-10-14 (§3) | — |
| V7C-05 estimator panel | inputs complete; checks to do | X21 / X11 results |
| V7C-06 external generalisation | inventory only: GenAI-Bench labels sealed (not approved), TIFA160 Likert / human annotations sealed in `third_party/closed/`, RichHF not downloaded | — |
| V7C-07 ledger | after C01–C06 | — |

## 3. Data, access and approvals
- **EvalMuse:** FIT / CAL_SELECT / CAL_PROB / DEV from public TRAIN (32,717 images; prompt-disjoint split); the EvalMuse test labels stay closed.
- **GenAI-Rank human ratings:** sealed and unread.  **Owner approval for scoring, 2026-10-14**, in reply to the scoring request (freeze record,
  hashes, frozen commands): "上面那个我也approve".  Per the package, the formal request with all hashes, the reproduction review and the exact commands
  (V7C-04) is written and committed before the first read; the read is logged in `LABEL_ACCESS_LEDGER.csv`.
- **Prediction freeze v1 → v2 (before any read):** v1 had tie-filled the M-arm E readouts on 16 decomposition-failed pools.  The frozen deployment rule
  (`scripts/v7/v7_parser.py`, `scripts/v7/v7_train.py`) falls back to O there, as the package's docs/03 §4 noted; v2 applies it.  v1 is never scored.
- **Not approved:** GenAI-Bench, TIFA160, RichHF label reads; paid APIs; new human annotation.

## 4. Order of work
1. C03 (label-free): fidelity generation → comparison and blind review table → `V7C_03_EXTERNAL_READINESS.md`.
2. C04: scoring request document + ledger entry → frozen scorer runs (primary 8, attribution 2, v1 6, pre-declared sensitivities).
3. C01 / C02: one GPU re-export job (E_gt / E_auto / O / H on CAL_SELECT, CAL_PROB, DEV for the 9 M units) → CPU analyses.
4. C05, C06, C07 on CPU.
5. V8 code is prepared on its own branch in parallel; no V8 training before V7 is complete (owner, 2026-10-14).
