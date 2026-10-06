# Pre-registration — P142 (MV6-I1): Table 3 completion — same-target kernel and conditional HSIC on the fixed-encoder fixture — FROZEN 2026-10-06

Source: CVPR-MV6-HANDOFF-20261006-r2 (docs/04 §5, docs/05 §C), intake `reports/manuscript_v6_handoff/intake_report.md`.  Axis
`fixed_encoder_vary_estimator` only; nothing here is pooled with Table 4.  No encoder training.  Owner go 2026-10-06 (D1, D2).

## 1. Question
On the fixed encoder of Table 3, with the same data, splits, N | Y draws and within-class permutations as P116, how do a same-target kernel
critic and conditional HSIC compare with the VCS two-stage solver (A1) and matched logistic (A3 at 50 / 200 / 800 closures) in detection and
cost?  Estimator usability only; no statement about encoders.

## 2. Block A — the P116 fixture, the same draws (primary)
- Fixture F-SOLVER-P116: `outputs/P45_precheck_D_simclr{,_blur}/` (P5_simclr_seed0 epoch_200, sha f180f84ccef5ea7c; FIT-standardised h, 512-d).
- Cells (as P116): blur 0.5 / n 1000, blur 0.5 / n 2000, colour 0.2 / n 2000, R = 100; nulls colour `null_label_only` and
  `null_all_planted` (0.2) at n 2000, R = 200; **new**: `null_label_only` at n 1000, R = 200 (gives the n-1000 cell an applicable null).
- Same seeds as P116 (614 blur, 612 colour, 616 level; 617 for the new null cell), same `run_repeat` order.  New methods run after the original
  ones inside the repeat and draw randomness only from their own generators seeded by `split_seed`, so the numpy stream of data draws, POOL
  negatives and permutations is unchanged.
- **Reproduction gate**: A1 and A3@{50,200,800} per-repeat own / common statistics and reject decisions must equal the stored
  `reports/P116/*.json` instances (statistics to 1e-6).  Failure = technical failure; the new rows are not reported until it is resolved.

## 3. Methods added (fixed here, before any EVAL)
| id | method | definition | selection (FIT 80/20 split only) | readout |
|---|---|---|---|---|
| K1 | same-target kernel critic (RFF) | A1's two-stage solver on φ_K = [ψ(h)(2N−1), 1], ψ = random Fourier features of standardised h (D = 1024, Gaussian, own generator) | bandwidth σ ∈ {0.5, 1, 2} × FIT median pairwise distance, ridge grid and 25-point tanh scale as A1; σ by VAL own risk | own statistic J and common J (same target S as A1); permutation power with the shared permutations |
| H1 | conditional HSIC, class-wise | `scripts/cond_test_t1.py::hsic_class_stat` (Σ_c n_c/n · biased HSIC within class; median bandwidth within class; delta kernel on N) | none (median heuristic) | native statistic; permutation power with the same within-class permutations |
| H2 | conditional HSIC, deep kernel | `cond_test_t1.py` deep-kernel variant (MLP 512→128→32 trained on FIT-train, selected on FIT-VAL) | as implemented in P74 | native statistic; permutation power, shared permutations |

A1 and A3 rows are the reproduced P116 rows.  The same-target kernel K1 is the RFF comparator family of P86 (critic on J) adapted to the
conditional construction.  HSIC is reported on its native scale and is never placed in the J / S column.

## 4. Readouts and rules
Per method and cell: rejection rate with a 95 % Wilson interval; paired difference to A1 on the shared repeats (McNemar exact test,
descriptive); common J on EVAL for A1 / A3 / K1 only; null rejection at R = 200 with the P116 flag (> 0.09); fit + selection seconds,
permutation seconds, CPU threads.  No posterior MSE (no oracle).  A method whose null flag fires keeps its numbers, marked; its power is not
read as a stronger valid test.  Non-rejection is reported as "not detected at this n, critic class and budget".

## 5. Block B — Table 4's SimCLR as a second fixed model (new fixture F-SOLVER-P142B; decision D2)
`P41_simclr_views4_800ep_seed1` features extracted with the P45 constructor (`scripts/precheck_d_features.py`, 45k FIT, same planting seed):
colour s ∈ {0, 0.1, 0.2}, blur s ∈ {0.25, 0.5}, cond-label null files.  Cells: the three Table 3 cells + colour 0.1 / n 2000 (Table 4's
strength) + the same three nulls; all methods (A1, A3@50/200/800, K1, H1, H2); seeds 621–625.  Reported as a separate block; Block A rows are
not replaced.

## 6. Cost (measured in the smoke, 16 CPU cores, n 2000)
Per repeat: A1 0.03 s, A3@50/200/800 0.06/0.35/1.39 s, K1 0.25 s (3 bandwidths), H1 0.74 s (200 permutations), H2 ≈ 65 s (300 steps; the P74
setting) — H2 dominates.  Block A 900 repeats ≈ 15 h, block B 1 000 repeats ≈ 16 h, in four CPU bundles (no GPU).  Block B extraction ≈ 15 min
on CPU inside the P143 extraction job.

## 7. Not claimed
Encoder properties; ranking across native targets; any S error on image data; results at other strengths, n or encoders.

## Decisions at the freeze (main session, 2026-10-06; owner go: "Submit all", "Add block B")
- Owner chose block B; both blocks run.  P116 seeds and unit grouping are reused exactly (colour 612; blur 614 with n 1000 then n 2000; level
  616 with label-only then all-planted); new cells: 617 (n-1000 null), 621 / 622 / 623 (block B).
- Smoke (job 1024513): reproduction gate on the first two repeats of colour 0.2 / n 2000 passed (max |Δstat| 1.1e-13, 0 decision mismatches).
  The gate needs the CPU device (P116 ran on CPU), so all P142 bundles run on the CPU partition.
- K1's same-target kernel is the RFF-critic family of P86 (critic on J); RuLSIF / S-KDE are not added (one same-target kernel row, fixed
  here).  H1 / H2 are the existing P74 implementations, with H2's base permutations drawn from a separate generator so that the main stream stays
  identical to P116.
- Launch: `slurm/mv6/p142_A1.txt`, `p142_A2.txt` (immediately), `p142_B1.txt`, `p142_B2.txt` (after the extraction job), via
  `slurm/mv6_bundle.sbatch`; aggregation by a results-only commit, then the report.
