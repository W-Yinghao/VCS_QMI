# Pre-registration — P119: v5 NEXT-V-PROBE — label efficiency paired by encoder seed, symmetric transfer readouts, float16 cache check (no encoder training), 2026-10-03 — FROZEN 2026-10-03T11:57:40Z before the full GPU jobs (CPU gate 1019931, GPU smoke 1019932)

Status: FROZEN 2026-10-03T11:57:40Z (main session; owner 2026-10-03: execute the v5 plan).  The main session freezes it (rename to `*_FROZEN_*`, freeze time in the title) before the full GPU jobs.  Source:
`VCS_Server_Next_Round_v5_CN.md` §7 (NEXT-V-PROBE) and `VCS_Results_Review_and_Next_Plan_v5_CN.md` §3.1–3.2.  Evaluation only on saved epoch-800
checkpoints; official CIFAR-10 test set untouched; labels are used only by the probes and kNN.
Code: `src/vcs_vtask/probe5.py` (new; the P110 modules are unchanged), `scripts/p119_v.py`, `scripts/p119_aggregate.py`, `slurm/p119_unit.sbatch`,
`slurm/p119_gate.sbatch`, `slurm/p119_gpu_smoke.sbatch`, `tests/test_p119.py`.

## Encoders (identical for every part)
recipe VCS `P35_vcs_a5_views4_800ep`, tuned SimCLR `P41_simclr_views4_800ep`, `P104_G2_views4_800ep`, `P104_U2_views4_800ep`, `P107_AP3_views4_800ep`;
seeds 0–2 each (15 encoders).  Only COMPLETED runs with `epoch_800.pt` are read.

## Part 1 — CIFAR-10 label efficiency, paired by encoder seed
- **Primary (re-summary of already reported P110 data; no decision depends on it):** P110 V1 rows (FIT-selected probe on raw h; draws 0–2 at 1 % and
  10 %, one 100 % row) are kept per seed × draw; per encoder seed the draws are averaged; differences are paired **by encoder seed** (family − SimCLR,
  A-P3 − G2): per-seed values, mean, sd over seeds, 95 % t interval (df 2).  Draws are never counted as pretraining seeds.
- **Secondary (new compute):** three further independent class-stratified subset draws (3, 4, 5) at 1 % and 10 % with exactly the P110 readout
  (`subset_seed("cifar10", frac, draw)`, FIT-selected and recipe-fixed probe), to estimate the within-seed readout variability (reported as the mean
  within-seed draw sd) and a more precise per-seed mean.  Same pairing.
- **Pre-stated reading (descriptive):** "A ahead of B at fraction f" only if all three seed-paired differences are positive **and** the 95 % t interval
  excludes 0; otherwise the per-seed values are reported without an ordering claim.  This is downstream label efficiency, not the estimator's
  unlabelled sample efficiency.

## Part 2 — CIFAR-10 → CIFAR-100 frozen transfer, identical readouts for every method
- Data: local CIFAR-100 train partition, P91 dev45k / val5k manifest (FIT 45 000, selection 5 000); labels 100 % and 10 % (draws 0–2), the **P110
  subset seeds** (`subset_seed("cifar100", frac, draw)`), identical index sets for every encoder.
- Readouts (same map for every method; `*_std` = per-dimension z-score with statistics of the labelled training subset only):
  `raw_unstd` (= the P110 readout, anchor; must reproduce P110 within GPU noise), `raw_std`, `l2_std` (h/‖h‖), `lognorm_std` (log‖h‖ alone, 1-d),
  `l2_lognorm_std` ([h/‖h‖, log‖h‖], 513-d).  Every readout: the P110 probe — (lr, wd) on an inner 80 / 20 split of the labelled subset (grid
  lr {0.03, 0.1, 0.3} × wd {0, 5e-4}), recipe stopping (100 epochs, cosine), refit on the subset.  kNN: recipe hyper-parameters (k 200, T 0.1, cosine on h),
  bank = the same labelled subset.
- **Norm readouts are diagnostics** and never replace the standard transfer table (`raw_unstd`).
- **Pre-stated reading (descriptive):**
  1. "Readout-robust ordering" of two families on transfer: the seed-paired mean difference keeps its sign and |Δ| ≥ 1.0 under all four of
     `raw_unstd`, `raw_std`, `l2_std` and kNN (P98's robustness criterion); otherwise "readout-dependent", reported with the per-readout values.
  2. Norm diagnostics: `lognorm_std` accuracy per family (chance = 1 %) = class information carried by ‖h‖ alone; `l2_lognorm_std` − `l2_std` = what adding
     the norm back restores.  Reported per family; no causal statement.
  3. The chosen probe lr is tabulated per family and readout; a grid-edge choice (0.3) in most rows is reported as "the probe grid bound binds".
  4. Not claimed: any explanation of the transfer reversal (colour retention, probe under-fitting) beyond what 1–3 show.
- The P110 numbers stay as reported; P119 adds readouts, it does not replace the P110 table.

## Part 3 — float16 cache check (decides the feature precision of Parts 1–2 before they run)
- P110 caches h in float16.  On a small subset — CIFAR-100 10 % draw-0 FIT subset (4 500 images) + the 5 000 selection images, and CIFAR-10 1 % draw-0
  FIT subset (450) + selection (5 000) — h is re-extracted in float32 on a GPU, for all 15 encoders, and compared with the cache: relative Frobenius and max
  abs difference of h; kNN, `raw_unstd` and `raw_std` accuracy on the identical subset.
- **Frozen rule:** the float16 cache "matters" if any encoder × readout has |Δacc| > 0.3 points, or the family-mean ordering of any readout changes.  If it
  matters, Parts 1–2 run with `--fp32` (float32 re-extraction for **all** methods, cache `outputs/P119_features`); otherwise they use the P110 cache.
- Disclosure: the cache was extracted on the P110 / P113 GPUs (A100 nodes); the check runs on whatever GPU the smoke gets, so the comparison includes
  GPU-model differences on top of the float16 rounding (it is an upper bound on the precision effect).

## Smoke (disclosed) — filled in from the jobs
- **CPU gate, job 1019931 (gate_rc 0; `reports/P119_GATE_1019931/`):** `tests/test_p119.py` 5 passed; P110 regression `tests/test_p110.py` 10 passed;
  the Part-1 primary re-summary ran on the committed P110 JSONs (`label_efficiency_paired_p110.md`); every runner module ran at smoke size on
  P41 seed 0 on CPU, including the float32 path.  On CPU, the float32 re-extraction reproduced the P110 `raw_unstd` value of SimCLR seed 0, CIFAR-100
  10 % draw 0 exactly (27.24).
- **GPU smoke, job 1019932 (L40S, gate_rc 0, ≈ 6 min; `reports/P119_GATE_SMOKE_1019932/`):** Part 3 on all 15 encoders.  h relative Frobenius
  difference fp16 vs fp32 ≤ 1.37e-3.  Max |Δacc| per readout over 15 encoders × 2 datasets: **kNN 0.08, `raw_unstd` 0.08, `raw_std` 1.52** (SimCLR seed 1,
  CIFAR-100: 25.84 fp16 vs 27.36 fp32; next U2 seed 2: 0.38; all other cells ≤ 0.26).  No family-order change.
  **Frozen rule outcome: the float16 cache MATTERS** (via the standardised readout only) → Parts 1–2 run with `--fp32` for **all** methods.
  The P110 readouts themselves (`raw_unstd`, kNN) moved ≤ 0.08, so the reported P110 numbers stand.  The cause of the `raw_std` jump is not diagnosed
  (candidates: a flip of the discrete inner (lr, wd) choice, or near-dead ReLU dimensions — SimCLR h is 72 % zeros — amplified by the z-score); the
  comparison also includes the A100 → L40S device difference.  Readout smoke (P41 seed 0, smoke sizes): all five readouts and kNN ran; values are
  code-path evidence only.
- **Disclosure for Part 1 secondary:** with `--fp32`, the extra draws 3–5 use float32 CIFAR-10 features while P110's draws 0–2 used the float16
  cache; on CIFAR-10 the P110 readout differed ≤ 0.06 between precisions (all 15 encoders), so the pooled per-seed means are reported with this note.

## Cost
Measured: fp32 re-extraction ≈ 25 s per 15 k images per encoder on L40S → ≈ 3 min per encoder for CIFAR-10 + CIFAR-100 (100 k images).
P110 timing for one FIT-selected probe on 45 k CIFAR-100 features: ≈ 90 s (A100); per encoder 5 readouts × (100 % ≈ 90 s + 3 × 10 % ≈ 28 s) ≈ 10 min,
kNN and the CIFAR-10 extra draws ≈ 1 min → ≈ 14 min per encoder, ≈ 3.5 GPU-h for 15 encoders, in **2 bundle jobs** (8 / 7 encoders, ≈ 1.8 h each).
Feature cache: 15 × 100 k × 512 float32 ≈ 3.1 GB under `outputs/P119_features` (the P110 cache is not modified).

## Launch (after freezing)
`slurm/p119_lines.txt`: two GPU bundle jobs (`slurm/p119_unit.sbatch`, normal QOS, RTX6000PRO / H100 / A100 / L40S, node60 allowed), modules
`transfer,c10extra`, `EXTRA=--fp32` (Part 3 outcome), encoders split 8 / 7; then `python scripts/p119_aggregate.py` → `reports/P119_v5_probe_results.{md,json}` → report.

## Delivery
```yaml
experiment_family: evaluation_only (readout robustness)
protocol_id: P119 (v5 NEXT-V-PROBE)
estimator: none (frozen encoders; linear probes and kNN)
evaluation_readout: selection-split top-1 (CIFAR-10 label fractions; CIFAR-100 frozen transfer), seed-paired differences
n_independent_units: 3 encoder seeds per family (subset draws are not seeds)
status: draft
```

## Decisions at the freeze (main session)
1. Feature precision: float32 for every method (`--fp32`), by the drafted 0.3-point rule (part 3: the standardised raw-h readout moved 1.52 for SimCLR seed 1).
2. The standardisation floor stays at the drafted 1e-6 (not changed after the smoke).  The fragility of the standardised readout (one SimCLR seed moved 1.52
   between float16 and float32; candidate causes: a flipped discrete lr choice, or near-empty dimensions — SimCLR h is ~72 % zeros — amplified by
   standardisation; the comparison also mixes an A100 → L40S device change) is a disclosed property of that diagnostic readout; the chosen lr per row is
   recorded in the outputs; the standardised readout is not a primary result.
3. Launch: the two bundle jobs of `slurm/p119_lines.txt` now (normal QOS, any of RTX6000PRO / H100 / A100 / L40S).
