# P137 — projector output 128 → 512 for VCS (A-P3), matched JS and SimCLR, CIFAR-10 / CIFAR-100, seed 0 — report — 2026-10-07

Pre-registration `P137_V7_HEAD_PROJECTOR512_PREREG_FROZEN_20261005.md`; results-only commit `472f271` (`P137_results.json`).  Each cell = its seed-0 parent
with only `model.projector.output_dim` 128 → 512 (+ marker); development split; frozen-h linear primary, kNN alongside.

| method | dataset | 512: linear / kNN | 128 parent | Δ linear | Δ kNN |
|---|---|---|---|---|---|
| VCS (A-P3) | CIFAR-10 | 88.42 / 86.96 | 89.06 / 87.30 | **−0.64** | −0.34 |
| JS-AP3 | CIFAR-10 | 88.90 / 87.44 | 88.72 / 87.50 | +0.18 | −0.06 |
| SimCLR | CIFAR-10 | 87.56 / 87.98 | 88.20 / 87.20 | −0.64 | +0.78 |
| VCS (A-P3) | CIFAR-100 | 59.88 / 55.86 | 60.20 / 55.92 | **−0.32** | −0.06 |
| JS-AP3 | CIFAR-100 | 59.82 / 55.22 | 58.76 / 54.80 | +1.06 | +0.42 |
| SimCLR | CIFAR-100 | 58.02 / 56.58 | 58.66 / 57.38 | −0.64 | −0.80 |

Interaction (attribution rule): Δ_HEAD(VCS) − Δ_HEAD(JS) = −0.82 (CIFAR-10), −1.38 (CIFAR-100); Δ_HEAD(VCS) − Δ_HEAD(SimCLR) = 0.00 / +0.32.
Wall time varies with the GPU type the job landed on (L40S ≈ 2× a healthy RTX6000PRO per epoch); it is listed in the results file, not compared.

## Reading (frozen)
- **Trigger not met** on either dataset (VCS needs ≥ +0.30 / +0.50; observed −0.64 / −0.32) → no seeds 1–2, no candidate; per the prereg the
  single dimension ablation ends (no 256 / 1024 / depth / new regularisers).  The 512/512/128 head stays the VCS configuration.
- Descriptive, one seed: the wider head helps matched JS (+0.18 / +1.06) and not VCS or SimCLR, so the JS-vs-VCS shared-scorer gap narrows at
  512 on CIFAR-100 (VCS 59.88 vs JS 59.82).  This is a seed-0 observation and is not used to change any comparison (the plan forbids new-VCS vs
  old-JS readings; both 128 parents remain the reference pair).
- The pre-stated CIFAR-100 coarse / fine / conditional readout (P141 protocol, evaluation only) of the three 512 encoders runs as job 1025487;
  it is appended when done.

## Not claimed
Multi-seed effects; other widths or depths; any change of the main VCS configuration.
