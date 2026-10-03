# Pre-registration — P115: strong augmentation split into crop and colour-jitter factors for A-P3 and tuned SimCLR (v5 NEXT-A-AUG), four seed-0 800-epoch units, 2026-10-03 — FROZEN 2026-10-03T13:30:56Z before 800-epoch compute (CPU gate 1019936, GPU smoke 1019937)

Status: FROZEN 2026-10-03T13:30:56Z (main session; owner 2026-10-03: execute the v5 plan).  Source: `VCS_Server_Next_Round_v5_CN.md` §3, review §6 P2.  **Launch timing (v5 §3):** after
P107 layer 2 (A-P3 seeds 3–4, strong aug, CIFAR-100) and the P112 strong-aug follow-up seeds have been collected.

## Question
Which part of the P89 strong augmentation block — the stronger random-resized crop (scale min 0.20 → 0.08) or the stronger colour jitter (0.4/0.4/0.4/0.1 →
0.8/0.8/0.8/0.2) — produces A-P3's loss under strong augmentation (std 89.06 → strong 88.66 at seed 0; 3 seeds −0.78) while SimCLR gains (seed 0 88.20 →
89.74; 3 seeds +1.17)?

## Cells: a 2 × 2 per method (seed 0, 800 epochs); 4 new, 4 reused
| cell | crop scale min | jitter B/C/S/H | A-P3 | SimCLR (P41 recipe) |
|---|---|---|---|---|
| standard | 0.20 | 0.4/0.4/0.4/0.1 | P107_AP3_views4_800ep_seed0 (89.06 / 87.30) — reuse | P41_simclr_views4_800ep_seed0 (88.20 / 87.20) — reuse |
| crop-only strong | 0.08 | 0.4/0.4/0.4/0.1 | **P115_AP3_croponly_views4_800ep_seed0** | **P115_simclr_croponly_views4_800ep_seed0** |
| jitter-only strong | 0.20 | 0.8/0.8/0.8/0.2 | **P115_AP3_jitteronly_views4_800ep_seed0** | **P115_simclr_jitteronly_views4_800ep_seed0** |
| both strong | 0.08 | 0.8/0.8/0.8/0.2 | P107_AP3_augstrong_views4_800ep_seed0 (88.66 / 86.20) — reuse | P89_simclr_views4_800ep_augstrong_seed0 (89.74 / 88.86) — reuse |
Each new YAML (`configs/cifar10_hpAF_{AP3,simclr}_{croponly,jitteronly}_views4_800ep_seed0.yaml`, `configs/make_p115_configs.py`, `configs/P115_SHA256.json`)
= the method's standard seed-0 config with only the crop scale or the four jitter strengths set to the values of the existing both-strong config (verified
by diff); flip, grayscale p, jitter apply-p unchanged; checkpoint epochs 20 / 100 / 400 / 800 added to the existing list (A-P3 already has them).
The P112 thresholds are **not** mixed in: A-P3 keeps (2, −1).  Existing G2 / A-P3 strong results are kept as they are.

## Reading (pre-stated, descriptive; one seed per new cell)
Per method: the four cells' linear / kNN; main effects (crop: mean of the two crop-strong cells − mean of the two crop-standard cells; jitter likewise)
and the interaction (both − crop-only − jitter-only + standard); the method contrast of each effect (A-P3 − SimCLR).  No verdict on one seed.
- A factor whose A-P3 effect is ≤ −0.5 linear while SimCLR's is ≥ 0 is named "the factor carrying A-P3's strong-augmentation loss" (descriptive).
- **Follow-up (separate addendum, submitted immediately):** a cell with a positive candidate (an A-P3 single-factor cell ≥ its standard cell, or the
  method contrast of a factor ≥ +0.5 in A-P3's favour) gets seeds 1–2; no simultaneous scans of backbone, projector, noise or thresholds (v5 §3).
- Diagnostics (positive / negative s, T, residuals, 1 − T², per-pair-type loss-gradient norms, crop-IoU-grouped positive-gradient shares, multi-view
  gradient angles, h norm / spectrum / linear / kNN at checkpoints 20 / 100 / 400 / 800) are a separate read-only diagnostic unit (p115_diag, other fork);
  they never filter or reweight training pairs.

## Gate
CPU gate job 1019936 (shared with P114): the 4 P115 configs resolve under the policy, 6-step smokes and stop/resume COMPLETED (CPU step-time
ratio vs A-P3 0.97–1.02); config-hash invariance 0 changes; evidence `reports/P114_GATE_1019936/gate_*{croponly,jitteronly}*`.

## Cost
A-P3 ≈ 0.114 s/step, SimCLR ≈ 0.115 s/step on RTX6000PRO → ≈ 4.4 h per run unshared; 4 runs; normal QOS, RTX6000PRO / H100 (node60 allowed).
Launch: `slurm/p115_lines.txt`.

## Decisions at the freeze (main session)
1. Launch via the orchestrator when P107 layer 2 (A-P3 CIFAR-100 seeds 0–2) and the P112 strong-augmentation follow-up seeds (4 units) have their epoch-800
   evaluations (v5 §3).  Normal QOS; node60 allowed.
2. P115 diagnostics job 2 (the four runs here) is registered with the orchestrator, condition: all four P115 runs have their epoch-800 evaluation.

## Execution note (2026-10-03T18:49:58Z, main session; no design change)
Submitted directly instead of waiting for the orchestrator condition: the last P112 follow-up run (strong κ 0.25 seed 2) was at epoch 697 / 800, all
other conditions (P107 layer 2, three of the four P112 follow-ups) were complete, and the owner asked why GPU quota sat idle.  The P115 design does not
depend on the pending P112 run (P112 thresholds are not mixed into P115 — §1 of this prereg), so queueing now only gains SLURM age priority.  The
orchestrator marker `P115_v5_aug.submitted` was created to prevent a duplicate submission.
