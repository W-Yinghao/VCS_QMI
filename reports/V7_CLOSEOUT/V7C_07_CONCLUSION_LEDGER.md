# V7C-07 — V7 conclusion ledger (2026-10-15)

Every row comes from a committed result file; the report column says where.  Wording follows the closeout package (docs/02 §4, docs/05 §3).

## 1. The four closeout questions
| question | answer | evidence |
|---|---|---|
| **A** Does the full V7 system choose better images than the public systems on the same candidates? | **Level with the released FGA-BLIP2**: −0.023 / +0.001. **Ahead of TIFA and DSG on SDXL_Base**: +0.091 / +0.108, Holm 0.003; level on DALLE_3. **Behind VQAScore on both generators**: −0.119 / −0.085, Holm 0.003. The modern-host VQAScore (Qwen2.5-VL) is the strongest system. | `V7C_04_SCORING_REPORT.md` |
| **B** With the same representation, is the learned route better or cheaper than KDE / RuLSIF / Nyström? | Better than KDE on every representation, on DEV Brier and on GenAI-Rank selection (SDXL_Base +0.074). Level with RuLSIF and Nyström; Nyström is the best single method on 3 of 4 DEV feature sets. The kernels are, if anything, under-tuned (grid edges). | `V7C_05_ESTIMATOR_PANEL.md`, `V7X_X21_results/` |
| **C** Non-CS estimators (DV, NWJ) on the same features | On native scales the KL-bridge critics are unstable at strong dependence (X11). On the common posterior they match VCS on DEV Brier; VCS selects slightly better than DV on GenAI-Rank DALLE_3 (+0.036, Holm 0.022). | `V7X_X11_results/`, `V7C_04_results/x21.json` |
| **D** Attribution: does the quadratic (VCS) objective differ from logistic? | **No.** Table A: M2 − M1 DEV Brier +0.0005 / +0.0005 / +0.0003, every interval containing 0. GenAI-Rank: M2-E − M1-E +0.003 / −0.001, Holm 1.0. Common features: VCS-F − BCE-F on DEV and GenAI-Rank, not significant. | `V7_TABLE_A_results/`, `V7C_04_results/attribution.json` |

## 2. Mechanism findings (CAL / DEV)
- **The human element mean orders images within a prompt 82 % of the time** (τ-b 0.63). The element-to-utility link has a ceiling.
- **Model readouts reach 0.71–0.74.** E over the human elements is 0.09 below the human reference (estimation loss), and the parser costs about 0.01.
- **The frozen mixture H is the best model readout.** The overall score O equals or beats E, on DEV and in GenAI-Rank selection.
- **6-vote images order almost at random** between halves of their votes. They are the disputed, re-annotated images, and every method faces them equally.
- **Evidence:** `V7C_01_02_LINKAGE_TAIL_PARSER_REPORT.md`.

## 3. Readiness and fidelity of the external rows
- All primary comparators are complete and frozen. TIFA (official local route) and DSG (labelled LLM-substitution adaptation) were checked against the authors' released question files:
  - DSG: tuple content precision 0.86, recall 0.81; the category labels differ.
  - TIFA: 5.92 vs 7.02 kept questions per prompt; element precision 0.91, recall 0.78.
- Paid-API routes were not run.
- **Evidence:** `V7C_03_EXTERNAL_READINESS.md`.

## 4. What the paper may say, and what not (package docs/05 §3)
- **May say.**
  - The V7 visual-evidence system (element posteriors on a fine-tuned Q-former) is competitive with the released FGA-BLIP2.
  - It improves on question-answering pipelines (TIFA, DSG) on one generator, but does not reach VQAScore.
  - On fixed features, learned critics and kernel squared regression estimate the posterior equally well, and density-estimation KDE is worse.
- **May not say.**
  - Any accuracy or selection advantage specific to the quadratic / CS objective.
  - That traditional CS methods fail.
  - That mean element satisfaction is a sufficient estimate of overall human quality.
  - That the 6-vote images or any other subset were filtered.
  - Anything about GenAI-Bench, TIFA160 or RichHF (not read).

## 5. Status and open items
| item | status |
|---|---|
| Formal units, Table A, X11, X21 (+ sensitivities), GenAI-Rank freeze and scoring | complete |
| DEV external secondary rows (B1 / TIFA / DSG on DEV) | waiting for an H100 (`v7x_grp_dev_A`); the DEV secondary table follows (descriptive; DEV is already exposed) |
| GenAI-Bench DSG predictions | waiting for an H100 (`v7x_grp_bench_dsg`); labels sealed (V7C-06 decision) |
| External generalisation | owner decision (V7C-06): GenAI-Bench is currently V8's confirmation source |
| Kernel edge extensions | not run: post-hoc for GenAI-Rank; owner decision |
| V8 | protocol frozen before this read; tuning running (`v8/utility-aligned-cs`) |
