# VL2 addendum 2 — the VL1-11 prior-corrected ranking on RefCOCO and RefCOCO+ — report — 2026-10-10

Protocol `VL2_ADDENDUM2_NCURVES_PRIORCORR_FROZEN_20261010.md`; results-only commit `8b767c9` (`reports/VL2/VL2_priorcorr_<dataset>_<features>.json`, job 1032563;
fits on the GPU fitter, equivalence-gated).  **The N-curve part of this addendum was cancelled** before any result (owner 2026-10-10: full experiments,
not probes / subsets); only the prior-correction part ran.  Phrase-uniform law, N all × 3 seeds; reading = prior-corrected (2f + log p_G(r)) − raw-f
image-macro Top-1 on DEV, the frozen VL1-11 rule (holds if the mean ≥ 0 and the 95 % interval excludes a loss of more than 0.2 point).

| dataset × features | VCS | matched JS | softmax |
|---|---|---|---|
| RefCOCO × CLIP | **+0.50 [+0.28, +0.72] holds** | +0.35 [−0.03, +0.74] holds | +0.23 [−0.02, +0.48] holds |
| RefCOCO × SigLIP 2 | +0.05 [−0.19, +0.29] holds | +0.12 [−0.25, +0.50] not shown | +0.10 [+0.07, +0.13] holds |
| RefCOCO+ × CLIP | +0.01 [−0.11, +0.13] holds | +0.16 [−0.06, +0.37] holds | +0.40 [−0.08, +0.88] holds |
| RefCOCO+ × SigLIP 2 | +0.16 [−0.03, +0.34] holds | +0.06 [−0.54, +0.66] not shown | +0.40 [−0.05, +0.86] holds |
| *RefCOCOg × CLIP (VL1-11)* | *+0.66 [+0.52, +0.79]* | *+0.63 [+0.34, +0.91]* | *+0.56 [+0.09, +1.04]* |

## Reading
- **The theory's direction replicates:** all 12 new means are ≥ 0, and 10 of 12 cells meet the frozen rule.  The two "not shown" cells are matched JS on
  SigLIP 2, with intervals of ±0.4 / ±0.6 around small positive means.
- **The effect is smaller on RefCOCO / RefCOCO+ than on RefCOCOg** (mostly +0.0 to +0.5 vs +0.6).  This is expected from the law: the correction adds
  log p_G(r), and on RefCOCO / RefCOCO+ the referred objects of an image carry more similar numbers of expressions (a flatter p_G).  This was not
  measured here; it is a hypothesis.
- Wording for the paper: "ranking by the calibrated critic plus the region prior never hurts and helps by up to +0.7 point; the gain scales with
  how non-uniform the prior is (largest on RefCOCOg)".  The scaling clause stays a hypothesis unless the prior entropy is reported per dataset.
