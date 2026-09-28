# P94 — D-line report: D1 (P72 adapter drift decomposition) and D2 (same-class negative-pair gradient share) — 2026-09-28

Pre-registration `P93_D_DIAGNOSTICS_PREREG_FROZEN_20260928.md` (frozen 2026-09-28T18:1x UTC, D-line CPU gate 1013249); results-only commit 256a5d9
(`P94_d1_p72_adapters.{json,md}` job 1013256, `P94_d2_same_class_negatives.{json,md}` job 1013288).  Reading rules applied by their letter first, then
the descriptive findings.  Both units read completed models only; nothing was selected or re-trained beyond the disclosed D1 reproduction.

## D1 — why did the P72 adapters drift by different amounts?

**Reproduction.**  All 16 (shift, m, method) cells reproduce P72 seed 0 exactly (SRC R@1, TGT R@1, R3 AUROC identical to `P72_robust_r2r3.json`); the
instrumented copy's QC sentinel max |Δparam| vs the original code path is 0.0 in every cell.  P72's selected (lr, epochs) are the inputs.

**Net displacement and path length (m = 0.4, final checkpoint).**

| shift | quantity | VCS | InfoNCE | logistic | JS |
|---|---|---|---|---|---|
| animal | ‖W_img − I‖_F | 6.82 | 7.05 | 11.13 | 22.91 |
| animal | Σ‖dθ‖ (path) / steps / mean ‖dθ‖ per step | 32.8 / 390 / 0.084 | 33.1 / 390 / 0.085 | 76.9 / 3120 / 0.025 | 151.4 / 3120 / 0.049 |
| animal | selected lr / epochs | 1e-3 / 5 | 1e-3 / 5 | 3e-4 / 40 | 1e-3 / 40 |
| indoor→outdoor | ‖W_img − I‖_F | 3.85 | 5.63 | 10.30 | 18.68 |
| indoor→outdoor | Σ‖dθ‖ / steps / mean ‖dθ‖ per step | 17.9 / 2080 / 0.0086 | 22.1 / 260 / 0.085 | 57.6 / 780 / 0.074 | 83.3 / 2080 / 0.040 |
| indoor→outdoor | selected lr / epochs | 1e-4 / 40 | 1e-3 / 5 | 1e-3 / 15 | 1e-3 / 40 |

**Pre-stated attributions (m = 0.4).**
- *Budget*: the displacement ordering JS > logistic > InfoNCE > VCS coincides with the path-length ordering in both shifts, but the per-step clause fails:
  JS's mean step is 0.58× VCS's on animal (1.7× apart) and 4.6× VCS's on indoor→outdoor.  By the letter: not met.
- *Injected pairs*: the injected share of the positive-term projection is 0.607 for JS on animal (≥ 1.5 × 0.4) but 0.623 for VCS as well, so the exclusivity
  clause fails; on indoor→outdoor JS's share is 0.529 (< 0.6).  Not met.
- *Gate*: the injected / clean gate ratio at the final checkpoint is 1.26 (VCS) vs 1.06 (JS) on animal and 1.13 vs 0.98 on indoor→outdoor — below the 2×
  threshold.  Not met.

**Verdict (letter): "not separated by these measurements" in both shifts.**

**What the measurements do show (descriptive).**  The drift ordering is the ordering of the training budget that P72's CAL selection assigned to each
method, through two different channels: on animal JS ran 8× the steps of VCS (40 vs 5 selected epochs) at a smaller step size (0.58×), giving 4.6× the path
and 3.4× the displacement; on indoor→outdoor both ran 2080 steps but JS's selected lr was 10× VCS's (1e-3 vs 1e-4), giving 4.6× the step size, 4.6× the
path and 4.9× the displacement.  So "VCS's adapters moved less" in P72 is a statement about the selected (lr, epochs), not about a smaller per-pair gradient
of the loss: at VCS's own operating point the injected pairs sit near T ≈ 0 (score −0.05, gate 0.85 vs 0.67 for clean pairs) and their per-pair
posterior-logit gradient is 0.45 — larger than on clean pairs (0.20) and of the same size as JS's on its injected pairs (0.57 at gate 0.22).  The local swap
table (VCS-form vs JS-form at one state) is reported and not read as a cause.  In every method the injected-pair gradient opposes the clean-pair gradient
(cos −0.62 VCS, −0.34 JS, −0.56 InfoNCE, −0.36 logistic on animal; −0.50 / −0.11 / −0.51 / −0.37 on indoor→outdoor).  The m = 0 cells (no injection) show the
same displacement ordering, i.e. the topic-pairing drift of P80 §1 is present without mismatch; on indoor→outdoor at equal steps VCS's path is 1.7× JS's
while its net displacement is 0.62× (path / net 7.8 vs 2.8): VCS's updates cancel more along the trajectory, JS's are more directed.

**Consequence for the wording.**  P72's smaller VCS drift cannot be cited as a property of the quadratic loss; it follows from the selected training budget.
The R line stays closed (P80).

## D2 — do same-class negatives carry less of VCS's gradient?

Identical 4 096 FIT-role images, seeded views, six view pairs and K = 8 partners for both methods; labels split the report only.

| checkpoint (epoch) | variant | sim same / diff | gate same / diff | F_same z [CI] | mass ratio z [CI] | relative weighting (÷ count share 0.100) | cancel z | cos(G_diff, G_pos) |
|---|---|---|---|---|---|---|---|---|
| VCS 100 | K = 8 | 0.675 / 0.586 | 0.378 / 0.147 | 0.246 [0.240, 0.251] | 0.322 [0.313, 0.330] | 3.2 | 0.98 | −0.74 |
| VCS 400 | K = 8 | 0.802 / 0.752 | 0.209 / 0.057 | 0.255 [0.249, 0.260] | 0.417 [0.403, 0.429] | 4.15 [4.01, 4.27] | 0.99 | −0.84 |
| VCS 800 | K = 8 | 0.805 / 0.759 | 0.165 / 0.040 | 0.255 [0.250, 0.261] | 0.451 [0.435, 0.465] | 4.49 [4.33, 4.63] | 0.99 | −0.84 |
| SimCLR 100 | native (2B − 2) | 0.078 / −0.007 | 0.0040 / 0.0015 (softmax w) | 0.231 [0.228, 0.234] | 0.231 [0.228, 0.234] | 2.32 | 0.80 | −0.38 |
| SimCLR 400 | native | 0.049 / −0.005 | 0.0037 / 0.0014 | 0.223 [0.220, 0.226] | 0.224 [0.221, 0.227] | 2.25 [2.22, 2.28] | 0.78 | −0.28 |
| SimCLR 800 | native | 0.038 / −0.004 | 0.0036 / 0.0014 | 0.216 [0.213, 0.219] | 0.217 [0.214, 0.221] | 2.18 [2.16, 2.22] | 0.78 | −0.26 |

Initial checkpoints: relative weighting 1.00 for both (no learned structure).  The K-matched SimCLR row (relative weighting ≈ 2.0) agrees with the native
row and is not used for the verdict.  h-space shares equal the z-space shares (0.256 vs 0.255 for VCS at 800).

**Verdict.**  The three pre-stated outcomes do not cover the observed configuration: at epochs 400 and 800 both methods' relative weightings exceed 1 with
CIs excluding 1, and the two CIs are disjoint — VCS's is twice SimCLR-native's.  Read descriptively: **VCS does not reduce the semantic-conflict gradient
share; it concentrates it**, putting 4.2–4.5× the count share of its negative gradient mass on same-class negatives against 2.2–2.3× for SimCLR-native.
The mechanism is visible in the table: same-class negatives are the most similar negatives (cos 0.81 vs 0.76 under VCS), therefore the least saturated
under the bounded critic (gate 0.165 vs 0.040), and the quadratic loss's gate (1 − T²) weights exactly those pairs; SimCLR's softmax also up-weights hard
negatives, by a smaller factor.  No cancellation flag (index 0.99 VCS, 0.78 SimCLR; threshold 0.5).  A further descriptive difference: under VCS the
different-class negative gradient sum is nearly anti-parallel to the positive term (cos −0.84 at 400/800; SimCLR −0.26 to −0.28), i.e. VCS's positive and
negative pulls largely oppose each other in direction on the projector output — consistent with the narrow-cone geometry recorded in `GEOMETRY_DIAG`.

**Consequence for the wording.**  The supplement's hypothesis that VCS "weakens the semantic conflict of same-class negatives" is not supported at the
frozen-checkpoint level; it must not appear as a mechanism claim.  If a same-class effect is discussed at all, the measured direction is the opposite.

## Delivery (Spec v2 §13.2)
```yaml
experiment_family: diagnostic
protocol_id: P93_D1_p72_adapter_drift | P93_D2_same_class_negatives
source_commit: 36771f5 (code, frozen prereg); results 256a5d9
source_basis: supplement_af7172c_plus_v2
estimator: vcs_neural (plus js / infonce / logistic adapters in D1; simclr_native in D2)
estimand: none (mechanism diagnostics on completed models)
evaluation_readout: native scores, gates, per-pair gradients, displacement (D1); per-anchor gradient mass by class of the negative (D2)
n_independent_units: D1 16 cells x 1 seed (P72 seed 0, disclosed retraining); D2 4 096 images x 4 checkpoints x 2 methods
n_positive_pairs: D1 per cell = n_fit x epochs; D2 4 096 x 6 view pairs per checkpoint
n_negative_pairs: D1 K = 8 per anchor; D2 K = 8 per anchor (VCS, K-matched) / 2B - 2 (SimCLR native)
status: complete
```
