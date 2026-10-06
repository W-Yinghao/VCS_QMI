# MV6 intake — CVPR-MV6-HANDOFF-20261006-r2 (first full intake; r1 was never ingested) — 2026-10-06

Package read in full outside the repo (scratchpad copy); `reference_v6/` untouched.  Package checks: `verify_package.py` PASS (72 package + 31
reference files), `check_contract.py` PASS, `check_r2_sync.py` PASS; `intake_repo.py` read-only output kept with this report's working notes.
These checks cover file integrity and toy algebra only; the science checks below are from the repository's raw records.

## 1. Baseline

| field | value |
|---|---|
| V6 evidence snapshot | `bc28bf89c94d5331de811a83aa8bcbfb77d7d155` (= the P141 report commit, 2026-10-06 05:58 UTC); available locally |
| current HEAD / branch | `e7b4e22` (P136 report) on `main`, pushed to `W-Yinghao/VCS_QMI` |
| evidence reports in `contracts/evidence_sources.json` | all 16 paths present; **byte-identical at snapshot and HEAD** (sha256 equal) |
| dirty tree | one tracked deletion (`reports/P126_V6_CURVE_PREREG_DRAFT_20261003.md`, the superseded draft of the frozen P126 prereg) and untracked `*.partial.json` run journals; no parallel agent, no uncommitted manuscript edits |
| manuscript | the owner writes the paper; this server keeps no edited copy, so no three-way merge is needed |
| authorisation | standing programme rules (pre-register → freeze → run; owner go per package).  This package grants no new training; measurement-only units below are DRAFT until the owner says go |

## 2. V6 → HEAD reconciliation (numbers recomputed from per-run files)

| claim / table | V6 value | check | status |
|---|---|---|---|
| C10 five-seed linear | VCS 89.02 ± .04, SimCLR 88.31 ± .21 | 89.024 ± .043 / 88.308 ± .212 from `evaluation_epoch_800.json` | unchanged |
| C100 selected-score | VCS 60.00 ± .25, JS 60.49 ± .21, SimCLR 58.25 ± .35 | 60.000 ± .250 / 60.487 ± .214 / 58.253 ± .353 | unchanged; **lr-sensitivity pending** (P129 addendum 4: at lr × 0.5 the gap is +0.16 vs −0.52 at lr 1e-3, seed 0; lr × 2 cell queued) |
| C100 shared-score | VCS 60.15 ± .38, JS 59.35 ± .42 | 60.152 ± .383 / 59.348 ± .423 | unchanged |
| Table 3 (P116) | 0.75/0.82, 0.99/1.00, 0.97/0.94; s 0.04/0.05, 0.04/0.06, 0.05/0.07 | `tables/measurement_comparison.csv` from `P116_results.md` | unchanged; see W1, W2 |
| Table 4 (P118) | 4 colour-0.1 rows | `tables/ssl_retention_response.csv` from the raw P118/P109 JSON | all 4 rows + 4 blur rows match exactly; see W3–W5 |
| new since snapshot | — | P133 (momentum keys, closed: no gain), P136 (schedule / views / batch, closed: no cell promising) | supplement candidates only; no main-table change |
| running | — | P129 add. 4 (lr × 2), P135 add. 1, P130 (ResNet-50 C100), P137 (projector 512), P138 (PAIR K16) | reported when done |

### Writing reconciliation items (for the owner; nothing edited here)
- **W1 — Table 3's "fixed external model" is the in-repo SimCLR control `P5_simclr_seed0` (2 views, 200 epochs, epoch_200.pt)**, not an
  external download and **not the Table 4 SimCLR** (`P41_simclr_views4_800ep`, 4 views / 800 epochs).  The feature in φ = [H(2N−1),1] is the
  FIT-standardised 512-d backbone output h (the manuscript's H is correct; the handoff's worry about "z" is resolved).  Option: P142 block B
  repeats the Table 3 protocol on the Table 4 SimCLR (new fixture), see §5.
- **W2 — "Applicable null rejection rates are 0.04–0.07"** (Table 3 text): on the SimCLR fixture the nulls are 0.04–0.06 (A1 and A3, colour
  `null_label_only` / `null_all_planted` at n 2000, R 200); 0.07 comes from the P35-VCS fixture (`level_vcs`), which Table 3 does not use.  No null
  cell exists at n 1000 (P142 adds one).
- **W3 — Table 4 nulls**: VCS column, no flag on H or O.  Colour H nulls 0.040–0.065, O nulls 0.025–0.080 (the manuscript's 0.04–0.07 / 0.03–0.08
  round from these).  Two flags at 0.10 fired in the same families on **the JS statistic**: A-P3 seed 1 colour `null_label_only` logits/JS, and
  SimCLR seed 1 colour `null_all_planted` z/JS (a layer not shown).  `null_all_planted` uses the family's largest strength (colour 0.2 / blur 1.0),
  not 0.1 / 0.25; `null_label_only` uses the recorded versions colour_s0.05 / blur_s0.25.  Supplement F should state both.
- **W4 — erratum for `reports/P118_V5_NEST_REPORT_20261003.md:27`**: "0.58–0.60 (VCS/JS)" for A-P3 colour h is wrong for JS seed 2 (0.44); VCS
  0.60 / 0.58 is right.  Recorded here; the frozen report is not rewritten.
- **W5 — Table 4 analysis head**: the classifier weights lived in `outputs/P109_I1_features/`, which the v4 cleanup (rule R6) removed; the head
  hash is therefore null in the ledger.  The extraction is deterministic code; P143 stage 0 regenerates it and checks the recorded effects.

## 3. Fixture registry (`fixture_registry.json`)

| fixture | encoder(s) | site / features | data | status |
|---|---|---|---|---|
| F-SOLVER-P116 | P5_simclr_seed0 epoch_200 (sha `f180f84ccef5ea7c`); VCS cells: P35_vcs_a5_views4_800ep_seed0 | standardised h (512); φ = [h(2N−1),1] | 45k FIT; per repeat disjoint FIT / EVAL / POOL of n; one stored Bernoulli(½) N; colour R×(1+s), B×(1−s) on uint8; blur PIL GaussianBlur(radius = s px) | **features on disk** (`outputs/P45_precheck_D_simclr{,_blur}/`); per-repeat JSON in `reports/P116/` |
| F-PAIRED-P109/P118 | P107_AP3 seeds 1/2, P41_simclr seeds 1/2 (+ P35, P104 G2/U2, P107 seed 0 in P109) | layer3 (pooled 256), h (512), z (L2-normalised projector 128), logits (10-way logistic on clean standardised h, L-BFGS, FIT) | 20k base images, FIT 10k / VAL 2k / EVAL 8k, clean + colour .05/.1/.2 + blur .25/.5/1 | **feature cache deleted**; results JSON kept (`reports/P118_i1_*`, `P109_i1_*_effects.json`) |
| F-NEST-P118 visual | same 4 encoders | h vs logit_h (separate probe on a 4k PROBE subset) and h ⊇ pca64 ⊇ pca16 ⊇ logit16 | n 3000 × 5 repeats | JSON kept; **no h→r→z, no h→fixed head** |

## 4. Unit status and P-number mapping (`task_to_p_number.json`)

| logical | decision | P / existing | reason (verifiable) |
|---|---|---|---|
| MV6-00 | DONE | this report | — |
| MV6-E0 | REUSE | P86, P118 truth | reports unchanged since snapshot; figure source export is a table task, no compute |
| MV6-S0 | REUSE + CONTINUE_EXISTING | P107, P111, P120, P129, P115; P140 tables | refresh after P129 add. 4, P135 add. 1, P130, P137, P138 |
| MV6-S1 | CONTINUE_EXISTING / BLOCKED on budget | P132 (preflight done), P139 (GPU-augmentation pipeline: bf16 check pending, 2/4-GPU DDP parked) | ImageNet-1k full training needs the owner's GPU budget; nothing is auto-submitted |
| MV6-I0 | DONE | §3 | both fixtures registered; hashes recorded where recoverable |
| MV6-R0 | REUSE (done) | P118 / P109 | 8 rows rebuilt from raw JSON, exact match; W3–W5 |
| MV6-I1 | DRAFT_NEW | **P142** | P116 has no kernel / HSIC rows; same-target kernel exists only for synthetic data (`src/vcs_estim/kernel_cs.py`), conditional HSIC exists with a permutation null (`scripts/cond_test_t1.py`, P74) but on different draws |
| MV6-I2 | DRAFT_NEW | **P143** | VICReg (4 views / 800 ep, seeds 0–2, C10 + C100, documented in P41 add. 2 / P91) and selected JS (P129 cells, seeds 0–2) exist; CIFAR-100 needs a runner change; P109 cache must be re-extracted |
| MV6-C1 | DRAFT_NEW | **P144** | no h→r, r→z or h→fixed-head measurement exists; r was never cached |
| MV6-W0 | owner | — | writing stays with the owner; W1–W5 are the change list |

P142–P144 are the next free numbers (v7's STRESS unit, not yet built, moves to P145).

## 5. Planned units (DRAFT pre-registrations, measurement only — no encoder training)

| unit | what | compute (estimate; smoke will replace) |
|---|---|---|
| **P142 (Table 3)** | A: same P116 fixture and **same draws** (original seeds, reproduction gate on A1/A3) + same-target kernel critics (RFF two-stage like A1; RuLSIF α = ½, whose ratio gives η = r − 1) + conditional HSIC (class-wise median kernel; deep kernel) with the shared within-class permutations; new null cell at n 1000.  B (decision D2): the same protocol on the Table 4 SimCLR (P41 seed 1) as a new fixture | A: CPU ≈ 4–8 h on 8 cores; B: GPU ≈ 0.2 h + CPU ≈ 6 h |
| **P143 (Table 4 axis)** | stage 0: re-extract A-P3 seed 1, rerun its colour-0.1 cell, compare with the recorded row (gate); stage 1 C10: VICReg and selected JS, seeds 1–2, colour 0.1 + blur 0.25 + nulls, same protocol; stage 2 C100: A-P3, SimCLR, VICReg, selected JS (3, 0.5), seeds 1–2 (runner gets a dataset argument and a 100-way head) | GPU ≈ 1–1.5 h extraction (13 encoders); CPU ≈ 15–22 h on 16 cores (25 audit units) |
| **P144 (compression)** | h→r (projector, eval mode), r→z, h→O (fixed clean-h head), (Y, N) kept; P118 nested machinery; A-P3 seed 1 first (map tolerance gate), then SimCLR seed 1, then seeds 2; colour 0.1, blur 0.25, null; signed ΔJ, D_T, ΔJ − D_T, repeat spread, paired bootstrap; one fit-budget sensitivity check | shares P143 extraction (+ r, fp32); CPU ≈ 4–6 h on 16 cores |

## 6. Decisions for the owner (`decisions_and_budget.md`)
- **D1** go to freeze and submit P142–P144 (measurement only; ≈ 1.5 GPU-h + ≈ 35 CPU-node-h).
- **D2** add P142 block B (Table 3 protocol on the Table 4 SimCLR) — removes W1's mismatch at small cost.
- **D3** include P143 stage 2 (CIFAR-100) — needs a reviewed runner change.
- **D4** ImageNet budget (unchanged; after the P139 report).
