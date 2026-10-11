# R5-QN — query-normalized critic for VCS and matched JS — report — 2026-10-12

Protocol `reports/R5/R5_QNORM_PROTOCOL_FROZEN_20261011.yaml` (frozen before any result; the owner chose "VCS + JS, 12 runs").  Results-only commit
`dcdf6a0` (`results_qnorm.json`, `qnorm_comparison.md`, `qnorm_run_manifest.json`; aggregate job 1034387).
- **Runs:** 12 full two-tower fine-tuning runs: VQ (VCS loss on T = tanh g) and JQ (balanced logistic on 2g) × RefCOCOg, RefCOCO+ × SigLIP 2 B/16 ×
  seeds 0–2, with the critic normalized within each query: g = f − ½ log Σ_r ν(r) exp(2f), ν uniform over the image's proposals (KLIEP constraint
  E_ν exp(2g) = 1; estimand, S, η and the optimal critic unchanged).
- **Fixed from R4-VL1:** recipe, roles, proposals and selection (CAL all-proposal Acc@0.5, ties earliest).
- **Gates:** step-0 given-box DEV Top-1 within 0.3 of the frozen value, 12 / 12 pass.  No run missing.
- **Controls reused:** R4-VL1 VCS / JS / set-event softmax E (18 runs) and R5-30 distribution softmax D (6 runs), all with the same seeds.
- **Status:** DEV has been read many times (exploratory); official val / test untouched.

## 1. Results (DEV all-proposal image-macro Acc@0.5, %, mean ± sd over 3 seeds)
| arm | RefCOCOg | RefCOCO+ |
|---|---|---|
| **VQ: VCS, query-normalized** | **72.86 ± 0.36** | **75.25 ± 0.09** |
| JQ: matched JS, query-normalized | 72.53 ± 0.26 | 75.04 ± 0.24 |
| VCS (R4-VL1) | 71.73 ± 0.21 | 73.03 ± 0.34 |
| JS (R4-VL1) | 71.36 ± 0.27 | 72.86 ± 0.16 |
| E: set-event softmax (R4-VL1) | 73.58 ± 0.36 | 76.25 ± 0.45 |
| D: distribution softmax (R5-30) | 73.85 ± 0.23 | 76.03 ± 0.14 |

Paired contrasts (by seed, 95 % t, df 2):
| contrast | RefCOCOg | RefCOCO+ |
|---|---|---|
| **VQ − VCS (primary)** | **+1.13 [+0.62, +1.63]** | **+2.22 [+1.29, +3.15]** |
| **VQ − E (primary)** | **−0.72 [−0.81, −0.63]** | **−1.00 [−2.22, +0.21]** |
| JQ − JS | +1.18 [+1.15, +1.21] | +2.19 [+1.98, +2.40] |
| JQ − E | −1.05 [−2.02, −0.08] | −1.21 [−2.92, +0.51] |
| VQ − JQ | +0.33 [−0.70, +1.35] | +0.20 [−0.36, +0.76] |
| VQ − D | −0.99 [−1.66, −0.32] | −0.78 [−1.33, −0.23] |

## 2. Pre-fixed readings (frozen wording)
1. **"Within-query normalization improves VCS by 1.13 (RefCOCOg) and 2.22 (RefCOCO+) points."**  Matched JS improves by the same amount (+1.18 /
   +2.19).
2. **RefCOCOg: "softmax still leads by 0.72."  RefCOCO+: "the interval contains 0 (no equivalence claim)."**
3. **"A gap closed by normalization supports the within-query competition explanation; it is not a new estimator (KLIEP normalization of the
   critic)."**  Normalization closes 61 % (1.13 of 1.85) and 69 % (2.22 of 3.22) of the R4-VL1 VCS − E gap.

## 3. Where the remaining gap sits (exact decomposition of means; descriptive)
With ν uniform over the image's n real proposals, 2g = log softmax_r(2f) + log n, so exp(2g) = ρ_f(r | w) / ν(r) with ρ_f = softmax over the
image's proposals.
- VQ's (and JQ's) loss therefore depends on f only through ρ_f.  That is the same model class as D, whose loss is −Σ_r p log ρ_f.
- Both losses are minimised at ρ_f = p, the uniform-over-valid-boxes target: for VQ, T* = tanh(½ log(p / ν)), i.e. exp(2g*) = p / ν.
- So VQ and D differ only in the criterion: VCS's bounded criterion on T = tanh g versus log-likelihood.

The R4-VL1 gap splits exactly:

  VCS − E = (VCS − VQ) + (VQ − D) + (D − E):  RefCOCOg −1.85 = −1.13 − 0.99 + 0.27;  RefCOCO+ −3.22 = −2.22 − 0.78 − 0.22

| piece | what differs (everything else equal) | RefCOCOg | RefCOCO+ |
|---|---|---|---|
| VCS − VQ: critic class | independent pairwise critic vs within-query normalized critic; same loss, same estimand | −1.13 [−1.63, −0.62] | −2.22 [−3.15, −1.29] |
| VQ − D: criterion | same normalized model class and target; VCS criterion vs log-likelihood | −0.99 [−1.66, −0.32] | −0.78 [−1.33, −0.23] |
| D − E: positive target | uniform over valid boxes vs set event (R5-30) | +0.27 [−0.35, +0.88] | −0.22 [−1.46, +1.02] |

**Reading:**
- About 60–70 % of the VCS deficit comes from the pairwise critic class: it has no coupling among a query's candidates.
- The rest, 0.8–1.0 points with the interval below 0 on both datasets, is the price of the VCS criterion relative to log-likelihood within the same
  normalized model class.
- The choice of positive target contributes nothing detectable.
- The pieces are paired contrasts of runs that share seeds, recipe and data; this is an arithmetic decomposition, not a mediation analysis.

## 4. Secondary observations (descriptive)
- **Same posterior:** VQ ≈ JQ (+0.33, +0.20; intervals contain 0), as VCS ≈ JS before.
- **Box quality:** normalization recovers about half of the Acc@0.75 deficit to softmax.
  - VQ 57.15 / 60.64; VCS 55.13 / 58.61; E 58.87 / 62.91 (RefCOCOg / RefCOCO+).
  - Recovered share: 54 % (RefCOCOg), 47 % (RefCOCO+).
- **Critic-fittable dependence:** VQ encoders sit with the softmax family on J_recal, not with VCS.
  - J_recal ×100 (common affine refit, same p / q): VQ 56.57 / 60.60; VCS 60.64 / 64.03; E 56.39 / 59.32; D 56.24 / 60.63.
  - The extra pairwise-separable dependence of the R4-VL1 VCS / JS encoders disappears as within-query ranking improves.
  - Across the six arms, J_recal and accuracy move in opposite directions.
- **Given-box Top-1:** VQ 81.40 / 83.07; VCS 81.47 / 81.66; E 82.16 / 83.23.
- **Selected epochs:** VQ 6–8 / 4–5, about as late as VCS (5–8 / 4–7); E 2–3 / 4; D 2 / 3–5.
- **Fitted critic (VQ):** a ≈ 10.8–11.1 (b ≈ −2.5 is irrelevant after normalization), similar across seeds.

## 5. Interpretation (hypothesis, not tested)
- **The deficit is mainly a critic-class effect.**  The normalization constrains the critic class, and the constrained class contains the optimum.
  That the same loss ranks better with it says most of the deficit was an inductive-bias / optimization property of the unconstrained pairwise
  critic, not of the VCS estimand.
  - With independent per-pair terms, raising one candidate's score does not lower the others'.
  - The pairwise critic therefore also rewards global, cross-query separation, which shows up as the higher J_recal.
- **The remaining 0.8–1.0 points look like a criterion effect.**  A candidate explanation is the boundedness of T = tanh g.
  - The VCS score gradient carries the factor (1 − T²), which vanishes as |g| grows.  That weakens the push on candidates that are confidently
    scored but still mis-ranked.
  - The log-likelihood gradient does not vanish this way.
- **Pending test:** the R5-20 score-gradient diagnostics for the new checkpoints (gradients through the normalizer; job 1034390, pending) will show
  whether VQ's gradient concentrates on the top self-distractor like the softmaxes' (52–67 %) or stays at VCS's 45–50 %.

## 6. Caveats
- Two datasets, one backbone, three seeds (df 2); pooled values descriptive.
- Three of the 12 runs ran on L40S with smaller crop chunks (160 vs 384 crops).  Accumulation is exact over whole images, so the loss is the same.
- The decomposition uses uniform ν (the R4-VL1 reference); with a non-uniform ν, exp(2g) is still ρ_f / ν, but VQ's target is then compared with a
  different D.
- RefCOCO (R5-X1, third dataset) has no VQ / JQ arms yet.

## 7. Next
- **Addendum:** R5-20 diagnostics of the VQ / JQ checkpoints (nested candidate curves, multiplicity strata, gradient shares through the normalizer)
  when jobs 1034389 / 1034390 finish.
- **Owner's choice for the paper:** whether the normalized critic belongs in the method (a KLIEP-normalized VCS critic that keeps the estimand and
  recovers 61–69 % of the ranking gap) or only in the analysis (the decomposition in §3).
