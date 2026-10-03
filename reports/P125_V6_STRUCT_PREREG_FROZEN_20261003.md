# Pre-registration — P125: v6 V6-STRUCT — structured pair features for the measurement critic (synthetic), 2026-10-03 — FROZEN 2026-10-03T20:11:30Z before the confirmation runs (CPU gate 1020290, pilot 1020291 disclosed)

Status: FROZEN 2026-10-03T20:11:30Z (main session; owner 2026-10-03: organise and submit the v6 plan).  The main session freezes it (rename to `*_FROZEN_*`, freeze time in the title) before the confirmation jobs.
Source: v6 package `VCS_Server_Tasks_v6_CN.md` §9 (P117 already tested output calibration; the question now is how to keep dependence **before** the
critic output becomes one number).  Code: `src/vcs_estim/p125.py` (new; imports the P85 / P108 builders, oracles, readouts and rotations unchanged),
`scripts/p125_struct.py`, `scripts/p125_aggregate.py`, `slurm/p125_gate.sbatch`, `slurm/p125_unit.sbatch`, `tests/test_p125.py`.

## Question
On the P108 / P117 synthetic generators, do structured pair features — an explicit interaction input or a quadratic log-ratio form — reduce the
posterior error of a fitted measurement critic relative to the JointMLP reference at the same data, budget and selection rule, for VCS and matched JS?

## Design (all named before the confirmation run)
- **Conditions** (P108 / P117, unchanged): C1 Gaussian (d 20, I 4), C2 Gaussian 2 signal of 100 dims (I 1.5), C3 xor mixture (d 10, I 4).
  N ∈ {4096, 16384} independent FIT pairs; roles FIT / SELECT / EVAL (32 768) / TRUTH (200 000) from the P85 builder (same total budget for every candidate).
- **Candidates** (one shared trainer = the P85 `train_neural` with a critic factory; proven bit-identical to it for JointMLP):
  `joint` JointMLP concat → 256 → 256 → 1 (reference, unchanged); `inter` Interaction-MLP on [x, y, x⊙y, |x − y|] with width matched to the
  reference parameter count (±2 %); `quad_full` f = b + uᵀx + vᵀy + xᵀAx + yᵀBy + xᵀCy (full A, B, C, small random init); in the large-dimensional
  condition (C2, d 100) also `quad_r4`, `quad_r16` with low-rank factors (non-zero random init so no factor product starts at zero).
- **Losses**: VCS (−J, T = tanh f) and matched JS (Deep-InfoMax logit f, posterior T = tanh(f/2)), identical feature class, indices, critic-init RNG
  role (torch.manual_seed(seed·7919 + 17) before the critic is built), batch stream, budget 1000 updates, batch 256, Adam lr ∈ {1e-4, 5e-4, 2e-3}
  chosen by SELECT native risk (checked every 100 updates; best-SELECT state kept), exactly the P108 / P117 rule.
- **Rotation**: P108 E2 fixed Haar orthogonal maps (Rx ≠ Ry), applied to `joint` and `inter` only (the quadratic class is closed under separate
  orthogonal maps; the interaction features are not; no oracle signal coordinates are used).
- **Readouts** on EVAL: posterior MSE vs the oracle η, J and S_plug bias / RMSE, eval SE of J, fit / tuning seconds, parameter count, selected update.
- **Seeds**: pilot = fit seeds 0, 1 (disclosed below, not pooled); confirmation = fresh fit seeds 2, 3, 4, 5, 6 (5 seeds), all cells × orig / rot.

## Pre-stated reading (descriptive; per condition × N × loss)
1. Primary: paired (same seed, same roles) Δ posterior MSE of each candidate vs `joint`, mean ± sd over the 5 confirmation seeds.  A candidate
   "reduces posterior error" in a cell if all 5 paired differences are negative; "increases" if all 5 are positive; otherwise "no consistent difference".
2. Gaussian-structure advantage (C1, C2) and cross-condition generalisation (C3 xor) are stated separately; no single "best critic" across conditions.
3. Rotation: for `inter`, Δ(rot − orig) posterior MSE is reported per cell; a gain in orig coordinates that disappears under rotation is reported as
   "coordinate-dependent inductive bias", not as a structural advantage.
4. VCS vs JS: side by side, no ranking.  J and S_plug errors reported per candidate.  If added expressiveness does not help, that is reported as the
   result (v6 §9: no unbounded network growth).  Not claimed: anything about image data or online SSL (a measurement critic is not added to SSL).

## Pilot (disclosed; job 1020291, CPU, exit 0; fit seeds 0, 1; 24 cells; `reports/P125_pilot/P125_pilot_results.md`)
Gate first: CPU job 1020290 (gate_rc 0): `tests/test_p125.py` 8/8 + P108 / P117 suites (29 passed in total), smoke cells (orig + rot), aggregator.
Selected-lr posterior MSE (mean of seeds 0–1), JointMLP → candidate (VCS; JS similar):
- C1 Gaussian: N 4096 joint 0.096 → inter 0.048, quad_full 0.088; N 16384 joint 0.044 → inter 0.034, quad_full 0.038.
- C2 Gaussian 2-of-100: N 4096 joint 0.546 (≈ S: fit has not started) → inter 0.114; quad_full / low-rank 0.59–0.71 (no fit); N 16384 joint 0.147 →
  inter 0.049, quad_r4 0.119, quad_r16 0.236, quad_full 0.583.
- C3 xor: N 4096 joint 0.595 → inter 0.289; N 16384 joint 0.159 → inter 0.171 (no gain); quad_full 0.84–0.86 (fails, as expected for a non-Gaussian
  log-ratio).
- Rotation removes most of the Interaction-MLP gain (C1 N 4096 0.048 → 0.203; C2 N 4096 0.114 → 0.548; C3 N 4096 0.289 → 0.823) while the JointMLP is
  rotation-insensitive (|Δ| ≤ 0.15, mostly ≤ 0.02).  I.e. in the pilot the interaction features' advantage is largely coordinate-aligned with the
  generators (signal coordinates are paired x_i ↔ y_i), which is exactly what reading 3 is designed to flag.
These pilot numbers fixed nothing in the design (candidate set, ranks, budget, lr grid and readings are as drafted before the pilot).

## Cost (measured, pilot, 16 CPU threads)
82–156 s per cell (C2 orig with five candidates the slowest).  Confirmation: 3 conditions × 2 N × 5 seeds × {orig, rot} = 60 cells ≈ 1.9 CPU-node-hours,
in 2 CPU bundle jobs (`slurm/p125_lines.txt`: C1 + C3, 40 cells ≈ 1.1 h; C2, 20 cells ≈ 0.7 h).  GPU not measured: the GPU quota is saturated by SSL
runs and the cells are small; CPU starts immediately.

## Decisions at the freeze (main session)
1. Confirmation on fresh fit seeds 2–6 (60 cells), 2 CPU bundle jobs, as drafted; candidates fixed from the pilot as drafted.
2. The rotation control is part of the reading (frozen rule 3): an Interaction-MLP gain that disappears under rotation is reported as a coordinate-dependent
   inductive bias, not as a general structural advantage.
