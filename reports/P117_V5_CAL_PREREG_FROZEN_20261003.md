# Pre-registration — P117: independent regression calibration of a fixed fitted critic on the P108 generators (v5 NEXT-E-CAL), 2026-10-03 — FROZEN 2026-10-03T12:09:33Z before the full runs (CPU gates 1019950 / 1019951)

Status: FROZEN 2026-10-03T12:09:33Z (main session; owner 2026-10-03: execute the v5 plan).  The main session freezes it (rename to `*_FROZEN_*`, freeze time in the title) before the GPU job.  Source:
`VCS_Server_Next_Round_v5_CN.md` §5, `VCS_Results_Review_and_Next_Plan_v5_CN.md` §4.2.  Code: `src/vcs_estim/p117.py` (new; imports the P85 builder /
trainer and P108's conditions, readouts and T maps unchanged), `scripts/p117_cal.py`, `scripts/p117_aggregate.py`, `slurm/p117_cal.sbatch`,
`slurm/p117_gate.sbatch`, `tests/test_p117.py` (8 tests).  Synthetic data only; no SSL; the training estimator's definition is not changed.

## Question
P108 found the squared plug-in closer to S than Ĵ for realistic fitted critics.  Is the readout error mostly an output-calibration problem — repairable
by a bounded map of the frozen score — or has the score U = T0(W) already lost dependence?  For any bounded g and m(U) = E[C | U] = E_M[η | U]
(C = ±1 balanced), S − J(g(U)) = E_M[(η − m(U))²] + E_M[(g(U) − m(U))²] =: A + B (orthogonality of the conditional expectation; the original S and J,
no new divergence).  A is the floor no calibrator can beat.

## Design
- **Conditions (P108, unchanged):** C1 Gaussian d 20, I 4; C2 Gaussian 2 signal + 98 irrelevant coordinates, I 1.5; C3 xor mixture, I 4.  N ∈ {4096, 16384}
  FIT pairs, 5 fit seeds each (30 cells); no rotation grid.
- **T0:** P108's JointMLP critic, VCS loss (T = tanh f) and matched JS loss (T = tanh(f/2)), product negatives, one fixed small budget of 1 000 updates,
  lr ∈ {1e-4, 5e-4, 2e-3} chosen by SELECT native risk (P108 procedure).
- **Roles:** FIT N; SELECT N/4 (T0 lr / early stopping); **CAL = the P85 TUNE role** (N/4, its own named stream, unused by every neural fit of P108);
  EVAL 32 768 per side (readouts only, selects nothing); DIAG 100 000 per side (new named stream, decomposition only).  Disjointness of CAL vs EVAL /
  FIT / SELECT and DIAG vs EVAL is asserted per cell.
- **Calibrators, all bounded in [−1, 1]:** identity; latent-affine tanh(α·atanh(clip(U, 1 − 1e-6)) + β) by L-BFGS on CAL squared risk; bounded
  bins = K equal-frequency bins of the pooled CAL U with the balanced (p̂ − q̂)/(p̂ + q̂) value per bin, K ∈ {4, 8, 16, 32} by 5-fold CV inside CAL.
  Selection among identity / latent-affine / bins(K*) by 5-fold CV squared risk inside CAL (squared risk = 1 − J on the same units; bins' CV risk
  is the minimum over K — slight optimism, disclosed).  Every candidate is reported; the selected one is a separate row.
- **Two comparisons, reported separately:** (i) *mechanism* — same frozen T0, calibrators on CAL; (ii) *end-to-end at equal total independent sample
  budget* — identity critic trained on FIT ∪ CAL (N + N/4, same SELECT, lr grid, budget) vs T0 on FIT + selected calibrator on CAL.
- **Decomposition (DIAG):** m(U) estimated by equal-frequency bins of U with the oracle η averaged per bin (balanced pool), at 100 / 400 / 1 600 bins;
  A = E_M(η − m̂)², B_g = E_M(g − m̂)², gap_g = E_M(η − g)²; residual_g = gap − A − B (= 2 E_M[(η − m̂)(m̂ − g)], → 0 as the bins refine) is reported.

## Endpoints (one table per candidate)
Posterior MSE E_M(g − η)², Ĵ bias and RMSE, Ŝ_plug bias and RMSE (over the 5 fit seeds; S from the P85 TRUTH sample), evaluation SE of Ĵ, fit time
(T0 chosen / total tuning) and calibration time; selected-candidate counts.

## Pre-stated reading (descriptive; per condition × N × fit loss)
1. **"Calibration repairs most of the error"** if the selected calibrator's posterior MSE ≤ 0.5 × identity's.
2. **"The score has lost most of the dependence"** if A ≥ 0.5 × (A + B_identity) on DIAG (A share).  Otherwise **"mixed"**.  The fixable share
   (B_identity − B_selected) / (A + B_identity) is reported for every cell.
3. End-to-end: the paired (same seed) difference of posterior MSE and the Ĵ RMSE of the two arms, mean ± SE; no claim of sample-efficiency gain if the
   calibrated arm is not better at equal total budget.
4. **If calibration does not help (reading 1 fails broadly and A dominates), that is reported as the result and the next step is representation
   compression / the critic function class — not further monotone maps.**  No new calibrator families are added after seeing EVAL.
5. Ĵ on EVAL is a finite-sample readout; it is not a guaranteed lower bound per run (only J(T) ≤ S in population).  Not claimed: anything about
   image data, SSL, or training the estimator with a calibrator.
6. Decomposition QC: max |residual| at 1 600 bins is reported; it should shrink with the bin count (it is an estimation artefact of m̂, not a test).

## Gate and smoke (filled in from the jobs; numbers are not results)
- Local smoke (login node, C1 N 4096 seed 0, smoke sizes, 100 updates): VCS posterior MSE identity 0.1975 → latent-affine 0.1678, A 0.1399,
  B_identity 0.0536, residual −1.3e-6; JS 0.1965 → 0.1730; end-to-end identity (FIT ∪ CAL) Ĵ error −0.1948 vs calibrated −0.1816.
- CPU gate `slurm/p117_gate.sbatch` (job 1019950, gate_rc 0): 8 P117 + 13 P108 tests pass; smokes of C1 / C2 / C3 (N 4096, smoke sizes) end to end
  (decomposition residual ≤ 2e-5 at 1 600 bins); full-size timing cell C2_gauss_pad N 16384 seed 0: 59 s on 16 CPUs (VCS posterior MSE identity 0.1438 →
  bins 0.1107, A 0.1070, B_identity 0.0349, residual 9e-7).  Logs: `reports/P117_GATE/`.

## Cost
Measured on CPU: ≤ 59 s per cell (largest cell) on 16 CPUs → 30 cells ≈ 0.5 h in **one CPU job** (`slurm/p117_lines.txt`).  GPU was not
measured (the GPU timing job could not start under the saturated 8-GPU quota and was cancelled); CPU is chosen to stay off the GPU quota.

## Delivery (v5 §8 fields)
```yaml
experiment_family: estimator (synthetic)
protocol_id: P117 (v5 NEXT-E-CAL)
estimator: P108 neural VCS / JS critics, frozen; calibrators identity / latent-affine / bounded bins (CAL only)
evaluation_readout: posterior MSE, J and S_plug bias / RMSE, eval SE, A + B decomposition on DIAG
n_independent_units: FIT N, SELECT N/4, CAL N/4, EVAL 32768 / side, DIAG 100000 / side; 5 fit seeds per cell
status: draft
```

## Decisions at the freeze (main session)
1. CPU partition as drafted: the 8-GPU quota is saturated by 800-epoch SSL runs, these fits are small and CPU-bound, and CPU starts now (GPU speed
   not measured; disclosed).  Normal QOS; never runfill.
2. Conditions, rules and controls as drafted (named before any full run); disclosures above kept.
