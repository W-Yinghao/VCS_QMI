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
- The pre-stated CIFAR-100 coarse / fine / conditional readout (P141 protocol, evaluation only; job 1025487, CPU, exit 0) is in the addendum below.

## Not claimed
Multi-seed effects; other widths or depths; any change of the main VCS configuration.

## Addendum — CIFAR-100 four-site readout of the 512 encoders (results-only commit `fb3eee2`, `reports/P137_readout/`)
raw_std probe, 512 encoder (Δ vs its 128 parent, seed 0, from `reports/P141/`); r and z are 512-d in the new encoders, 128-d in the parents.

| method | site | coarse | fine | conditional |
|---|---|---|---|---|
| VCS | h | 71.22 (−0.72) | 59.58 (+0.00) | 72.82 (−0.20) |
| VCS | r | 67.92 (−0.78) | 55.04 (−0.18) | 71.66 (−0.56) |
| VCS | z | 68.44 (−0.40) | 55.52 (−0.56) | 72.26 (−0.06) |
| JS | h | 70.06 (−0.08) | 57.84 (+0.10) | 71.46 (−0.32) |
| JS | r | 67.58 (+0.08) | 53.16 (+0.48) | 70.64 (+0.30) |
| JS | z | 67.30 (−0.30) | 53.60 (−0.44) | 71.12 (+0.90) |
| SimCLR | h | 70.02 (+0.28) | 57.14 (+0.32) | 70.50 (+0.86) |
| SimCLR | r | 69.46 (+1.54) | 55.56 (+0.90) | 70.72 (−0.60) |
| SimCLR | z | 70.46 (+0.96) | 56.58 (−0.66) | 71.98 (−0.80) |

Reading (descriptive, one seed; P141 seed sd ≈ 0.2–0.7): the wider head does **not** recover the fine-grained information that P141 found lost
between h and the 128-d head — fine at z changes by −0.56 (VCS), −0.44 (JS), −0.66 (SimCLR), and the h → z fine gap stays (VCS 59.58 → 55.52).
kNN at every site is lower or unchanged for VCS and SimCLR.  Consistent with the main result: no reason to widen the head.
