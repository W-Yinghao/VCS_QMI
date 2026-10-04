# P134 — R1: estimator-level synthetic contamination on the P85 staircase cell (VCS / JS / InfoNCE / NWJ) — report — 2026-10-05

Pre-registration `P134_R1_CONTAMINATION_PREREG_FROZEN_20261004.md` (spec: Next-Round Plan v2 "R1"); jobs 1021198–1021200 exit 0; results-only commit
`650b66c` (`P134_r1_aggregate.{md,json}`, 51 cell files in `reports/P134/`).  Base cell = P85's Gaussian staircase cell (d = 20, I = 4 nats, N = 4096,
2000 updates, P85 roles / trainer / lr grid), seeds 0–2; ε = 0 reproduces P85's I = 4 values (|diff| ≤ 6e-5 except NWJ in-batch 4e-3, GPU
non-determinism).  Shift = estimate under contamination − clean estimate, in each estimator's own resolution unit u (half the I = 2 → 6 gap on the
clean P85 staircase); normalized shift = |shift| / u per 0.1 of ε, maximised over ε ∈ {0.01, 0.05, 0.1, 0.2}.

## 1. Primary reading (frozen): **supported**

| max normalized shift (bound 1) | independent pairs | outlier pairs (x + 5σ, y from I = 10) | heavy-tailed y (t₂) |
|---|---|---|---|
| **VCS (product)** | **0.88** | **0.18** | **0.62** |
| JS (product) | 1.40 | 0.13 | 1.64 |
| InfoNCE (in-batch), descriptive | 0.49 | 0.21 | 0.48 |
| NWJ (in-batch), descriptive | 0.48 | 0.21 | 0.48 |

VCS stays within 1 resolution unit per 0.1 of ε in all three primary types; learned JS exceeds it in two (independent, heavy), so the spec's
falsification condition ("JS bounded alike, within 20 %") is not met.  JS's excess comes from small ε (the per-0.1-ε normalization weights ε = 0.01
by 10): at ε = 0.01 JS moves −0.14 units vs VCS −0.09 (independent); at ε = 0.2 both are near 0.8.

## 2. What moves: the target, not the estimator (decomposition, ε = 0.2)
shift (units) = target change (units) + error change: the contaminated problem has a different true S / JS / MI, and most of every shift is that
target change (e.g. independent ε = 0.2: VCS −1.56 = −1.39 target + error −0.03; JS −1.62 = −1.40 + error −0.07).  The change of the *error* against
the contaminated truth is smallest for VCS in every primary type (−0.03 / −0.006 / −0.05 native units), larger for JS (−0.07 / −0.03 / −0.13), and
large for InfoNCE / NWJ under outliers (≈ −1.0) and heavy tails (−0.27 / −0.49) — descriptive, not a pre-stated criterion.

## 3. Caveats (frozen not-claimed list + reading)
- In their own resolution units InfoNCE / NWJ in-batch move *less* than VCS (≈ 0.5 vs 0.88): their units are large (u ≈ 1.6–1.8 nats on the MI
  staircase).  The primary claim is VCS vs learned JS only (spec); the unit-normalised comparison across different targets is descriptive (P85 rule 5).
- One setting (Gaussian, d = 20, I = 4, one critic class and budget); nothing about SSL training, image data, or recovering the clean target.
