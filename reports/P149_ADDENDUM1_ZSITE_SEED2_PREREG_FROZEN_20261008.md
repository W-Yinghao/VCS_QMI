# P149 addendum 1 — the projector site z and a second encoder seed — FROZEN 2026-10-08

Source: P149 prereg §5 deferred "z-site and other encoder seeds (later protocol)"; P149 report §4 (model observation on one encoder seed each);
owner rule: keep ≥ 15 experiment jobs queued (the queue falls to 14 when P138 add. 1 closes; no triggered follow-up, the SSL exploration axes are
closed, and r3 authorises no new pretraining — this is the cheapest open measurement in r3's scope).  CPU only; frozen after the gate.

## 1. Units (4 CPU jobs, one per encoder family; `slurm/p149a1_run.sbatch <dataset> <seed-1 run> <seed-2 run>`)
- **z-site on the four P149 fixtures** (seed 1; projector output z, 128-d): `measure --site z` on the existing fixtures (no new views).
- **Seed-2 encoders** P107_AP3_views4_800ep_seed2, P41_simclr_views4_800ep_seed2, P107_AP3_c100_views4_800ep_seed2,
  P91_c100_simclr_views4_800ep_seed2: new fixtures with the identical P149 builder (same 20 000 base images, roles, views and EVAL pools — the
  augmentation and pool seeds do not depend on the encoder), then `measure` at h and at z.
Everything else is P149's frozen protocol unchanged (standardisation, 4 settings, 3 estimators, 3 init seeds, TUNE 500 / 500, 1 000 steps,
derangement nulls, re-pairing, transport check, paired bootstrap 1 000).  The only code change is the `--site` option (default h).

## 2. Pre-stated reading (as P149 §4, per site and seed)
Labels per fixture × estimator × site: numerically resolvable / not resolvable / estimation-sensitive, orthogonal drift with its paired interval.
Additional descriptive comparisons: (i) z vs h on the same fixture (z is the site the SSL objective is applied to; a lower J at z than at h is the
expected data-processing direction, not a requirement); (ii) seed 1 vs seed 2 for the P149 §4 model observation (VCS vs SimCLR level at t 0 and
loss under the channel) — the observation is called **seed-stable** only if its sign holds at both seeds for both MLP routes, otherwise
**seed-dependent**.  No encoder or estimator is required to win; S_plugin stays context only (P149 §3).

## 3. Gate (`slurm/p149a1_gate.sbatch`)
Unit tests unchanged; the default h path on the P149 smoke fixture must reproduce the P149 gate smoke J values exactly (max |ΔJ| < 1e-6);
smoke of `--site z`.

## 4. Cost
P149: 21–43 min per fixture-measurement → per job ≈ 3 measurements + 1 fixture build ≈ 1.5–2.5 h; 16 h limit (no resume; skips finished outputs).

## 5. Decisions at the freeze (2026-10-08)
Gate 1028346: `tests/test_p149.py` 6 / 6; the default h path on the P149 smoke fixture reproduces the P149 gate smoke exactly (max |ΔJ| = 0.0);
`--site z` smoke runs (128-d).  Submitted as four CPU jobs (one per encoder family); nothing else changed.
