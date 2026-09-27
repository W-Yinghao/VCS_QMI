# Pre-check A — calibration and threshold transfer (P49/P50), setting 1 (exact image–caption pairs): result and reading

Pre-registration: `P49_PRECHECK_A_PREREG_FROZEN_20260927.md` (+ addenda 1–4).  Table: `P50_precheck_A.md/.json` (GPU job 1010623; smoke outputs
`P50_precheck_A_smoke*.md` disclosed).  Frozen CLIP ViT-B/32 towers, identity-initialised 512×512 adapters, COCO source (no `animal`) →
target (`animal`), balanced joint/product pairs on held-out captions; 3 seeds (spread ≤ 0.004 ECE, ≤ 0.004 R@1 — omitted below).

## Numbers (mean of 3 seeds; selected configurations: VCS lr 1e-3 / 40 ep, InfoNCE 1e-3 / 15, logistic 1e-3 / 40)
| probability | SRC-EVAL ECE / Brier | TGT-EVAL ECE / Brier |
|---|---|---|
| VCS native (1+T)/2 | 0.046 / 0.018 | 0.089 / 0.052 |
| VCS embedding, cosine + Platt(CAL) | 0.004 / 0.013 | **0.036** / 0.045 |
| VCS score T + Platt(CAL) | 0.007 / 0.014 | **0.032** / 0.044 |
| InfoNCE cosine + Platt(CAL) | 0.003 / 0.013 | 0.064 / 0.054 |
| logistic native σ(a·cos + b) | 0.337 / 0.248 | 0.357 / 0.279 |
| logistic cosine + Platt(CAL) | 0.004 / 0.015 | 0.064 / 0.066 |
| raw CLIP cosine + Platt(CAL) (no adapter) | 0.004 / — | 0.034 / — |

| threshold rule | SRC-EVAL FNR / FPR | TGT-EVAL FNR / FPR | drift |ΔFNR| / |ΔFPR| |
|---|---|---|---|
| VCS absolute T ≥ 0 (no calibration data) | 0.018 / 0.017 | 0.027 / 0.085 | 0.009 / 0.068 |
| cosine on the VCS embedding, CAL threshold matched to the same source FNR | 0.019 / 0.017 | 0.027 / 0.084 | 0.009 / 0.067 |
| InfoNCE cosine, CAL quantile at FNR 5 % | 0.048 / 0.008 | 0.034 / 0.061 | 0.014 / 0.054 |
| logistic absolute logit ≥ 0 | 0.778 / 0.000 | 0.869 / 0.000 | 0.092 / 0.000 |
Retrieval R@1 (secondary): raw CLIP 0.368 / 0.334 (source / target); InfoNCE 0.362 / 0.284; logistic 0.300 / 0.204; **VCS 0.269 / 0.212**.
Held-out J of the VCS model: **0.923 on SRC-EVAL**, 0.797 on TGT-EVAL; learned a = 6.15, b = −0.50 (threshold cos* = 0.08).

## Reading (pre-committed grid)
1. **Mid-dependence check fails on the source**: held-out J = 0.92 > 0.9.  Exact image–caption pairs on CLIP features are close to the
   degenerate regime the brief warns about (P1); by the frozen rule the source-side calibration claim is *not decidable* in this setting.  The
   target (J = 0.80) is inside the regime and is read.
2. **Claim 1 (native calibration) does not hold.**  (1+T)/2 has ECE 0.046 / 0.089 (source / target) with max deviation 0.13 / 0.23 —
   over-confident (a = 6.2) — and is worse than Platt-scaled cosine on the *same* embedding (0.004 / 0.036) and worse than the InfoNCE
   competitor on the target (0.064).  The pairwise Brier objective did not deliver a calibrated probability after finite training with a
   two-parameter cosine critic; this is "binarisation", the failure mode the brief names.
3. **Claim 2 (threshold transfer) does not hold.**  The absolute rule T ≥ 0 transfers exactly as well as a cosine threshold placed at the same
   source operating point (drift 0.068 vs 0.067 in FPR): T = tanh(a·cos + b) is monotone in the cosine, so the only thing the absolute rule
   buys is that no calibration data are needed to *find* the threshold — not a smaller drift.  The logistic model's absolute rule is unusable
   (FNR 0.78: its learned bias encodes the 1 : 255 in-batch prior).
4. **Observation (not a pre-registered claim):** after a one-dimensional Platt fit on the source, the VCS-trained embedding transfers its
   calibration to the shifted target better than the InfoNCE- and logistic-trained embeddings (ECE 0.032–0.036 vs 0.064) and as well as raw
   CLIP (0.034) — i.e. the VCS adapter preserved CLIP's calibration under shift while InfoNCE/logistic adapters degraded it; the VCS adapter
   also *lost* retrieval (R@1 0.27 vs 0.37 raw), consistent with the SSL finding that the VCS code is a separability code rather than a ranking
   code.  Embedding calibration ≠ critic calibration; the brief asked to keep the two apart.

## Verdict for setting 1
**Does not hold** (claims 1 and 2), with the source at the degenerate edge.  Consequence for the brief's family table (row A): mismatch
detection / open-set rejection via the *native* T are not supported by this setting.

## Follow-up inside the pre-check (pre-registered as addendum 5 before running)
Setting 2, mid-dependence by construction: positives = (image, caption of a *different* image with the same supercategory set) — a topic-level
relation whose S is well below 1 and whose truth is known by construction; product = random caption.  Same towers, adapters, grid, seeds and
metrics.  This answers whether the native calibration holds when the regime is not degenerate, and it is the setting pre-check C will use.

---
# Setting 2 (topic pairing, mid-dependence by construction) — result and final reading of pre-check A

Table: `P50_precheck_A2.md/.json` (GPU job 1010742; prereg addendum 5).  Positives = caption of a different same-topic image; 96 % of images
have a partner.  Held-out J = **0.572 (source) / 0.157 (target)** — inside the mid-dependence regime.  Selected: VCS lr 1e-3 / 15 ep, InfoNCE and
logistic 1e-3 / 40 ep; seed spread ≤ 0.005 ECE.

| probability | SRC-EVAL ECE / Brier | TGT-EVAL ECE / Brier |
|---|---|---|
| **VCS native (1+T)/2 — no calibration data** | **0.023** / 0.104 | **0.088** / 0.208 |
| VCS embedding, cosine + Platt(CAL) | 0.022 / 0.104 | 0.089 / 0.209 |
| InfoNCE cosine + Platt(CAL) | 0.013 / 0.099 | 0.092 / 0.208 |
| logistic native σ(a·cos + b) | 0.483 / 0.470 | 0.488 / 0.483 |
| logistic cosine + Platt(CAL) | 0.011 / 0.098 | 0.079 / 0.204 |
| raw CLIP cosine + Platt(CAL) | 0.012 / — | 0.087 / — |
(InfoNCE "score + Platt" is NaN: the Platt fit on the τ-scaled logit overflowed; the cosine + Platt row is the intended competitor.)

| threshold rule | SRC-EVAL FNR / FPR | TGT-EVAL FNR / FPR | drift |ΔFNR| / |ΔFPR| |
|---|---|---|---|
| VCS absolute T ≥ 0 | 0.091 / 0.178 | 0.350 / 0.283 | 0.259 / 0.105 |
| cosine on the VCS embedding, matched to the same source FNR | 0.091 / 0.178 | 0.351 / 0.282 | 0.260 / 0.104 |
| InfoNCE cosine, CAL quantile at FNR 5 % | 0.039 / 0.247 | 0.091 / 0.564 | 0.052 / 0.317 |
| logistic absolute logit ≥ 0 | 1.000 / 0.000 | 1.000 / 0.000 | — |
Learned a = 5.26, b = −0.13 (threshold cos* = 0.025).  Retrieval R@1 of exact captions collapses for every adapter trained on topic pairs (0.05–0.08
vs raw 0.37) — expected and secondary.

## Reading (pre-committed grid, setting 2)
- Mid-dependence check: **passes** (J 0.57 / 0.16).
- **Claim 1 (native calibration): holds conditionally.**  In this regime (1+T)/2 is as calibrated as a post-hoc Platt fit on the same embedding
  (0.023 vs 0.022 source; 0.088 vs 0.089 target) and meets the frozen thresholds (≤ 0.05 on source; on target ≤ the Platt-scaled InfoNCE cosine
  0.092) — i.e. the VCS critic delivers probabilities *without a calibration set* of the quality a calibration set would give.  It is not better
  than post-hoc calibration of the competitors (logistic + Platt 0.079 on the target), and every method's calibration degrades under the
  shift by the same amount (≈ 0.06–0.08 ECE): the shift, not the objective, dominates the target error.  Setting 1 showed the same statement
  fails when the pair problem is near-degenerate (over-confidence); so the property is conditional on the regime the brief's P1 asks for.
- **Claim 2 (threshold transfer): does not hold**, structurally: T = tanh(a·cos + b) is monotone in the cosine, so the absolute rule T ≥ 0 is a
  cosine threshold in disguise and transfers exactly like one placed at the same source operating point (drift 0.259 / 0.105 for both).  What
  the absolute rule offers is that the operating point needs no calibration data, not a smaller drift; a non-monotone (multi-dimensional)
  critic would be needed for the claim to be even testable.

## Final verdict for pre-check A
**Holds conditionally** (calibration without a calibration set, only in the mid-dependence regime; no transfer advantage).  Families
(brief appendix A, row A): mismatch detection / open-set rejection / pointwise PMI maps move to *conditional* candidates — usable where the pair
problem is genuinely uncertain (S ≪ 1) and where "no calibration data" is the value proposition; data re-weighting by T stays unsupported
(no advantage over a calibrated cosine).  Evidence category: completed (3 seeds, single split, CLIP ViT-B/32 features, COCO topic shift).
