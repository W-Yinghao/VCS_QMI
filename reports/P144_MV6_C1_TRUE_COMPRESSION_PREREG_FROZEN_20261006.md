# Pre-registration — P144 (MV6-C1): dependence change under real deterministic maps h→r→z and h→O — FROZEN 2026-10-06

Source: CVPR-MV6-HANDOFF-20261006-r2 (docs/05 §G), Proposition 3; intake `reports/manuscript_v6_handoff/intake_report.md`.  Existing
checkpoints only.  Owner go 2026-10-06 (D1).

## 1. Maps and observation
For one realised input set (the P143 base images and versions), observations (Y, U, N) with U ∈ {h, r, z, O}: r = projector(h) (eval mode, BN
frozen, no dropout), z = r / max(‖r‖, 1e-8), O = fixed clean-h logistic head (the P109 / P143 head, fitted once on FIT clean features).  Maps
tested: h→r, r→z, h→z, h→O.  Y, N and the P / Q construction are identical on both sides.  pooled layer3→h is not a deterministic map and is
excluded.

## 2. Map definition and gate (per encoder)
The measured observations are computed at measurement time on CPU in fp32 from the cached fp32 h: r = projector(h) (the run's projector, eval
mode), z = r / max(‖r‖, eps), O = stored head(h).  A = g(B) therefore holds exactly.  Cross-checks recorded per version: cached r vs recomputed
r (≤ 1e-4), stored fp16 z vs recomputed z (≤ 2e-3, fp16 rounding), stored logits vs recomputed O (≤ 1e-4).  A failed cross-check is reported as
an extraction discrepancy; the map used is always the recomputed one.

## 3. Estimation (P118 machinery, unchanged)
`src/vcs_measure/nested.py` as in P118 visual (n 3000, 5 repeats, 300 steps, seed 20261004): independent and nested fits, exact Q (primary
readout fixed now: **nested, exact Q, VCS objective**), JS objective and sampled Q secondary, constant-zero control.  For each map, condition
and repeat: J_B, J_A, signed ΔJ = J_B − J_A, D_T = mean_M (T_B − T_A)², ΔJ − D_T, repeat spread, paired bootstrap over base images.  No
clipping, no monotone correction, no response noise.  Image data carry no oracle, so posterior-MSE fields are null.

## 4. Scope and order (fixed; by reusability, not by effect)
CIFAR-10; conditions colour 0.1, blur 0.25, null.  Order: A-P3 seed 1 (full map validation), SimCLR seed 1, then A-P3 seed 2 and SimCLR
seed 2.  Sensitivity on one registered condition (A-P3 seed 1, colour 0.1, h→r): steps {300, 1000} × 2 independent fit seeds; if signed ΔJ and
D_T disagree beyond the repeat spread the readout is labelled estimation-sensitive.

## 5. Cost
Shares the P143 extraction (+ r, fp32).  CPU ≈ 4–6 h on 16 cores (P118 visual: 2.9 h for 4 encoders × 3 cells), sensitivity ≈ 1 h.

## 6. Not claimed
That any encoder keeps the least nuisance; monotone loss along the head; differences between encoders as compression; probe accuracy as
information.

## Decisions at the freeze (main session, 2026-10-06; owner go: "Submit all")
- Observations are standardised with PROBE-set (4 000 FIT images) mean / sd before fitting — an affine bijection, used as in P118.
- Chain h ⊇ r ⊇ z nested from z upwards (r on z, then h on r); pair h → O nested on the independent O critic.  Primary readout: VCS,
  nested, exact Q.
- Smoke (jobs 1024513 / 1024518) exercised the extraction with `--fp32-maps` and the runner at smoke sizes.
- Launch: `slurm/mv6/p144.txt` (main run in the registered order, then the three sensitivity runs) after the extraction job.
