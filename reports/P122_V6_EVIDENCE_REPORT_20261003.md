# P122 — v6 V6-EVIDENCE: evidence recovered from frozen representations beyond the training scorer — report — 2026-10-03

Pre-registration `P122_V6_EVIDENCE_PREREG_FROZEN_20261003.md` (frozen 2026-10-03T19:31:46Z).  Jobs 1020304 (CIFAR-10, 5 encoders × 2 measurement laws) and
1020305 (CIFAR-100, 3 encoders × common-standard), both exit 0.  Aggregate `P122_v6_evidence_results.{md,json}` (13 cell files, 26 rows); per-cell JSONs in
`reports/P122/`.  Rule script reproduced at the end.

All increments are **additional score recovered in these function classes and budgets** (1000 updates, lr {1e-4, 5e-4, 2e-3} × 3 inits, TUNE selection, parent
always a candidate), with 95 % bootstrap intervals over EVAL pair units.  They are not true values of nested information terms.  Seed-0 encoders only.
Labels (v6 §11.2): **[observed]** result, **[identity]** verified by definition, **[hypothesis]** to be tested.

## 1. Training scorer vs best scalar critic of the same s (reading 2)

| encoder (train aug) | law | J train | J s (picked s critic) | s − train |
|---|---|---|---|---|
| A-P3 std (C10) | standard | 0.903 | 0.975 (affine VCS / 1-D MLP JS) | **+0.072** |
| A-P3 strong (C10) | standard | 0.877 | 0.970 | **+0.093** |
| recipe VCS (C10) | standard | 0.987 | 0.994 | +0.007 |
| A-P3 (C100) | standard | 0.902 | 0.978 | **+0.076** |
| recipe VCS (C100) | standard | 0.987 | 0.993 | +0.007 |
| A-P3 std / strong (C10) | strong | 0.710 / 0.822 | 0.782 / 0.897 | +0.072 / +0.076 |
| recipe VCS (C10) | strong | 0.787 | 0.818–0.819 | +0.031 |

**[observed]** For the fixed-scale scorer (A-P3, a = 2, κ = 0.5), a scalar re-reading of the very same s recovers ≈ 0.07–0.09 of score; for the learned-scale
recipe only ≈ 0.007 (standard law).  **[identity]** For a fixed (a, κ) this gap is the expected consequence of T = tanh(a(s − κ)) not being a calibrated
posterior of s (v6 §2.2): the fixed scorer is a training device, not a measurement critic.  SimCLR has no training scorer (n/a).

## 2. Structural increments (Z − s, H − Z), VCS and JS side by side; share = increment / (1 − Ĵ_s)

### CIFAR-10, common-standard measurement law

| encoder | Z − s VCS | Z − s JS | H − Z VCS | H − Z JS | positive gain (rule) |
|---|---|---|---|---|---|
| A-P3 std | +0.0003 [−0.0004, +0.0010] | +0.0008 [+0.0001, +0.0016] | +0.0003 [−0.0007, +0.0015] | +0.0005 [+0.0001, +0.0010] | no |
| A-P3 strong | **+0.0041 [+0.0022, +0.0059]** (14 %) | **+0.0045 [+0.0028, +0.0063]** (15 %) | +0.0009 [−0.0007, +0.0026] | +0.0004 [−0.0013, +0.0023] | **yes (Z)** |
| recipe VCS | −0.0000 [−0.0003, +0.0002] | −0.0001 [−0.0004, +0.0003] | −0.0005 [−0.0008, −0.0002] | −0.0002 [−0.0007, +0.0003] | no |
| SimCLR std | +0.0000 [−0.0000, +0.0000] | +0.0000 [−0.0000, +0.0000] | **+0.0006 [+0.0000, +0.0012]** (3 %) | **+0.0008 [+0.0003, +0.0015]** (4 %) | **yes (H), borderline** |
| SimCLR strong | **+0.0010 [+0.0005, +0.0015]** (4 %) | **+0.0019 [+0.0008, +0.0030]** (7 %) | −0.0010 [−0.0028, +0.0006] | −0.0008 [−0.0027, +0.0010] | **yes (Z)** |

### CIFAR-10, common-strong measurement law (cross-evaluation)

| encoder | J s VCS | Z − s VCS | Z − s JS | H − Z VCS | H − Z JS | positive gain (rule) |
|---|---|---|---|---|---|---|
| A-P3 std | 0.782 | **+0.0091** (4 %) | **+0.0090** (4 %) | **+0.0149** (7 %) | **+0.0146** (7 %) | **yes (Z, H)** |
| A-P3 strong | 0.897 | **+0.0031** (3 %) | **+0.0046** (5 %) | **+0.0149** (15 %) | **+0.0187** (18 %) | **yes (Z, H)** |
| recipe VCS | 0.818 | **+0.0138** (8 %) | **+0.0208** (12 %) | **+0.0169** (9 %) | **+0.0042** (2 %) | **yes (Z, H)** |
| SimCLR std | 0.776 | **+0.0073** (3 %) | **+0.0090** (4 %) | −0.0022 [−0.0066, +0.0017] | −0.0010 [−0.0042, +0.0021] | **yes (Z)** |
| SimCLR strong | 0.903 | +0.0004 [−0.0000, +0.0009] | +0.0014 [+0.0002, +0.0027] | **+0.0078** (8 %) | **+0.0090** (9 %) | **yes (H)** |

(Intervals of all bold entries exclude 0; full intervals in `P122_v6_evidence_results.md`.)

### CIFAR-100, common-standard measurement law

| encoder | J s VCS | Z − s VCS | Z − s JS | H − Z VCS | H − Z JS | positive gain (rule) |
|---|---|---|---|---|---|---|
| A-P3 | 0.978 | −0.0044 [−0.0077, −0.0012] | −0.0000 [−0.0013, +0.0013] | +0.0014 [−0.0020, +0.0049] | +0.0003 [−0.0009, +0.0016] | no |
| recipe VCS | 0.993 | −0.0002 [−0.0005, +0.0000] | −0.0013 [−0.0026, −0.0002] | +0.0000 (no H residual selected) | +0.0000 (no H residual selected) | no |
| SimCLR | 0.975 | +0.0000 [−0.0000, +0.0000] | −0.0000 [−0.0000, +0.0000] | −0.0007 [−0.0011, −0.0002] | −0.0014 [−0.0023, −0.0004] | no |

## 3. Frozen reading 1 — which cells meet "positive structural gain" (interval > 0 for both losses)
- **CIFAR-10 standard law:** A-P3 strong-trained (Z), SimCLR strong-trained (Z), SimCLR std (H; VCS lower bound ≈ +0.00001, borderline).
- **CIFAR-10 strong law:** all five encoders (A-P3 std: Z and H; A-P3 strong: Z and H; recipe VCS: Z and H; SimCLR std: Z; SimCLR strong: H).
- **CIFAR-100:** none.
- **Not met:** A-P3 std and recipe VCS under the standard law (CIFAR-10), and every CIFAR-100 cell → "no additional score recovered at this budget"
  (not "the similarity is sufficient").
By the frozen rule, the encoders with a qualifying cell (A-P3 std / strong, recipe VCS, SimCLR std / strong on CIFAR-10) get encoder seeds 1–2 in a later
addendum (the main session decides and submits; nothing submitted here).

## 4. Cross-evaluation and method comparison (seed 0; [observed])
- **The measurement law dominates the structural increments.**  Under the standard law, the best scalar critic of s already reaches J ≈ 0.970–0.994 and
  full-pair structure adds ≤ 0.0045 (≤ 15 % of the small remaining headroom); under the strong law J_s falls to 0.78–0.90 and Z / H recover 0.003–0.021
  (up to 18 % of the headroom), for every encoder.  Harder (strongly augmented) pairs are where s stops summarising the recoverable pair evidence.
- **Augmentation match:** encoders trained with strong augmentation read strong pairs much better (J_s 0.897 A-P3 strong vs 0.782 A-P3 std; 0.903 SimCLR
  strong vs 0.776 SimCLR std) and lose little on standard pairs (0.970 vs 0.975; 0.975 vs 0.980).  Strong-trained encoders are the ones with a positive Z
  increment under the standard law.
- **Methods (standard law):** recipe VCS has the highest J at every level (0.987 train, 0.994 s) and no structural increment; A-P3 has the largest scorer-to-s
  gap (+0.072) and essentially no structural increment; SimCLR (0.980 s) has a small positive H increment.  This repeats P109's finding that a higher
  measured J does not track downstream accuracy (A-P3 has the best linear accuracy of the three).
- **[hypothesis]** That a fixed-scale training scorer leaves pair evidence that only a non-scalar critic can read under hard pairs (A-P3 std: Z + H ≈ +0.024
  under the strong law) is a hypothesis for the curvature / scale experiments (P126 / P112), not a demonstrated mechanism.

## 5. Density-ratio diagnostic (necessary-condition only; f = 0 also passes)
- **Training scorers:** global log E_Q e^{2f} ≈ −1.5 to −1.8 for A-P3 and −3.5 / −2.3 for the recipe (standard / strong law), with high per-anchor ESS (158–195
  for A-P3; 19–27 for the recipe) — the training scorers are far from normalised density ratios, as expected for fixed / training-device scorers.
- **Fitted critics:** values scatter around −3 to +1 with per-anchor ESS of 2–20 (tail-dominated); a few cells are extreme and unreliable (CIFAR-100 A-P3 VCS
  Z +13.6 and H +41.1, ESS 1.5–1.6; CIFAR-10 strong-law recipe VCS s / Z / H +7.3 / +9.0 / +9.1).  No conclusion about correctness is drawn from these values.

## 6. Anomalies and caveats
- **Negative EVAL increments where a residual was selected on TUNE** (CIFAR-100 A-P3 VCS Z − s −0.0044, CIFAR-100 recipe JS Z − s −0.0013, CIFAR-10 recipe VCS
  H − Z −0.0005, SimCLR C100 H − Z): the parent is always a candidate, so these reflect TUNE-selection noise / over-fit with TUNE = 1024 pairs, not a defect in
  the nesting; kept as is (negative values are reported, per the prereg).
- CIFAR-100 recipe VCS: no H residual selected for either loss (H − Z exactly 0).
- SimCLR standard-law H increment: VCS lower bound rounds to +0.0000 (positive by a hair); flagged as borderline.
- One seed per encoder; no claim about other seeds, datasets, or any causal link to linear / kNN accuracy.

## Rule script
```python
import json, collections
d = json.load(open("reports/P122_v6_evidence_results.json"))["rows"]
g = collections.defaultdict(dict)
for r in d: g[(r["dataset"], r["law"], r["run"])][r["loss"]] = r
for k, v in sorted(g.items()):
    for inc in ("Z_minus_s", "H_minus_Z"):
        ok = all(v[l][inc]["ci95"][0] > 0 for l in ("vcs", "js"))
        shares = {l: v[l][inc]["mean"] / (1 - v[l]["J_s"]) for l in ("vcs", "js")}
        print(k, inc, ok, shares)
```
