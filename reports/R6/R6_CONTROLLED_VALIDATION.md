# R6-02 — controlled validation of the two priority routes (REG, GAIN) — 2026-10-11

- **Package:** r6 (`r6/package_r6_20261011`; docs/03 §2–6, docs/04, docs/05).
- **Compute:** CPU only, no training on natural data, no downloads.
- **Labels:**
  - **S**: inherited (collaborator plan, prior work);
  - **D**: direct derivation, standard;
  - **P**: new design or hypothesis.
- **Nothing here is a natural-image result.**

| item | location |
|---|---|
| code | `scripts/r6_reg_controlled.py`, `scripts/r6_gain_controlled.py`, `scripts/r6_controlled_summary.py`, `slurm/r6_controlled_cpu.sbatch` |
| tests | `tests/test_r6_controlled.py`, 12 / 12 (job 1034541).  The package's own reference tests also pass on the server (job 1034434). |
| raw outputs | `outputs/R6_controlled/reg_controlled.json` (27 condition-seeds), `outputs/R6_controlled/gain_controlled.json` (500 cells) |
| tables | `reports/R6/R6_controlled_tables.md` (job 1034542) |

---

## 1. REG — does a frozen local-dependence critic give a usable registration objective?

### What was done (P on S)
- **Scenes.**  Synthetic two-modality scenes with a known sub-pixel shift φ* ∈ [−5, 5]² px:
  - U = L + noise;
  - V = g(L shifted by φ*) + noise;
  - L is a smooth random field (random Fourier features).
- **Relations:**
  - monotone (g = −l);
  - non-monotone (g = l² − 1: zero linear correlation, still dependent).
- **Laws, exactly as specified in docs/04 §3.**
  - **Fixed interior support Ω:** every candidate shift in [−8, 8]² keeps all samples inside the image; the bilinear-sampling guard refuses
    otherwise.
  - **The same samples for every candidate:** P_φ = (U(x), V(x + φ)); Q_φ = (U(x), V(x′ + φ)) with a fixed within-scene permutation.
  - **Q is recomputed under each φ.**
  - **The critic sees only (u, v).**
- **Critics.**  Fitted on 20 independent training scenes at φ = id, then frozen:
  - VCS (maximise J, T = tanh(w·poly₄(u, v)));
  - balanced logistic (damped Newton);
  - generator oracle (exact pixel-level η by quadrature).
- **Matching energy:** F_T(φ) = J(T; P_φ, Q_φ).
- **Baselines on the same samples:** histogram MI, NMI, |Pearson correlation|.
- **One shared optimiser for every method:** integer grid search, then multi-scale compass refinement.  Capture range = compass search
  from an initial offset r ∈ {1, 2, 4, 6} px.
- **Scale:** 3 seeds × 20 test scenes per condition.  Scenes are the units.

### Results (success = final |error|∞ ≤ 0.5 px; 60 scenes per cell)
| condition | VCS | logistic | oracle | MI | NMI | corr | reading |
|---|---|---|---|---|---|---|---|
| non-monotone (main) | 1.00 (med 0.12 px; capture@6 px 0.98) | 1.00 (0.11; 0.97) | 1.00 (0.16; 0.97) | 1.00 (0.17; 0.88) | 1.00 (0.19; 0.93) | 0.02 | learned critics work; slightly more precise, larger capture than histogram MI; correlation fails by design |
| monotone | 1.00 | 1.00 | 1.00 | 0.97 | 0.95 | 1.00 | sanity: every method |
| **weak relation** (σ_b = 3) | 0.32 (med 0.71) | 0.38 | 0.38 | **0.13** (0.88) | 0.15 | 0.07 | all degrade; critics and oracle degrade less than MI |
| **periodic** structure (period 6 px) | 0.03 | 0.05 | — | 0.07 | 0.03 | 0.00 | **counterexample: repeated structure makes the objective ambiguous for every method** (median error ≈ one period) |
| **scene offsets**, within-scene Q | **0.85** | **0.87** | — | **1.00** | 1.00 | 0.18 | **negative for the learned critic:** a fixed critic fitted across scenes on raw values cannot know each scene's offset; per-scene MI is invariant to it |
| scene offsets, **cross-scene** Q in training | 0.78 | 0.72 | — | 1.00 | 1.00 | 0.12 | the scene-identity shortcut costs a further 7–15 points |
| **coordinate leak** (critic sees the squared coordinate difference) | **0.17** (sharpness 0.04) | 0.18 | — | 1.00 | 1.00 | 0.13 | **guard confirmed:** the critic learns "same position" instead of content; the objective collapses |
| **border frame, shrinking overlap** (support varies with φ) | 0.53 | 0.17 | — | 0.55 (p90 err 4.7 px) | 0.48 | 0.00 | **guard confirmed:** frames align at φ = 0 and attract every method |
| border frame, **fixed interior Ω** | 1.00 | 1.00 | — | 1.00 | 0.98 | 0.10 | the fixed support removes the artefact completely |

### Other findings
- **Differentiability (D):** analytic dF/dφ = finite differences to 9.7 × 10⁻⁹ (72 checks at non-integer shifts).  The reference term's
  dependence on φ is included.  The matching energy is a valid smooth optimisation target in this setting.
- **VCS ≈ logistic** in every condition (same posterior; S).
- **An implementation failure, found and fixed before the full run:** the first logistic fit (undamped Newton, ridge 1e-4) diverged on one
  smoke condition.  T ≈ −1 everywhere, so F ≡ −1.  It now uses damped Newton with backtracking (ridge 1e-3).  The smoke output is kept as
  `outputs/R6_controlled/reg_controlled_smoke.json`.

### Reading
- **In clean controlled conditions, a learned local-dependence critic gives a correct, smooth registration objective.**  Its precision and
  capture range are at least as good as histogram MI's.  It also degrades more gracefully than MI under heavy noise.
- **The guards matter:** coordinates and shrinking supports produce false alignment.
- **The weakness is photometric nuisance across scenes.**  A critic fitted once across scenes loses 13–15 points of success to per-scene
  offsets.  Per-scene MI / NMI does not, because it re-estimates the joint histogram in every image.  On natural data, inter-image appearance
  variation is the norm, so this is the key risk for REG.  It points to per-image normalisation or learned invariant encoders (the CoMIR
  direction), not to a better estimator.
- **Repeated structure** defeats every method.

---

## 2. GAIN — does the evidence increment decide when to compute V, and is the squared disagreement a safe proxy?

### What was done (D on S; one added construction is P)
- **Exact enumeration:** S_U, S_UV, Δ = S_UV − S_U = R*_U − R*_UV and d(u) for:
  - the four package constructions (redundant, irrelevant, XOR, context);
  - **context_noise** (added, P): U = (K, X); K = 0: C is pure noise; K = 1: C = X·V.  Cheap uncertainty is maximal in both strata, but only
    K = 1 gains from V.
- **Fitted predictors** (FIT_TASK n = 2000) in two families × two classes:
  - families: square / Brier vs balanced logistic;
  - classes: linear (no U × V interaction: misspecified for XOR / context) vs interaction (well specified).
- **Policy targets** on FIT_POLICY (n = 2000): z = (b − a)², h = (C − a)² − (C − b)², residual = h − z.  Policies read U only; the
  reference guard is enforced.
- **Policies compared:**
  - never / always / random at the same rate;
  - cheap uncertainty 1 − a²;
  - GAIN-square (predict z);
  - GAIN-utility (predict h);
  - oracle d(u) (upper bound).
- **Evaluation:** DEV task risk (n = 4000) at budgets 0 / .25 / .5 / .75 / 1; 20 seeds; label noise ε = 0.1.
- **Negative checks:** a worse full predictor; XOR with constant d(u).

### Results (DEV, mean over 20 seeds; risk at budget 0.5 unless noted)
| construction (Δ at ε = 0.1) | fitted class | mean z | mean h | random | cheap uncertainty | GAIN-square | GAIN-utility | oracle d |
|---|---|---|---|---|---|---|---|---|
| redundant (0) | both | 0.000 | 0.000 | 0.357 | 0.357 | 0.357 | 0.357 | 0.357 |
| irrelevant (0) | both | 0.000 | 0.000 | 0.355 | 0.355 | 0.355 | 0.355 | 0.355 |
| XOR (0.64) | interaction | 0.643 | +0.638 | 0.681 | 0.680 | 0.684 | 0.683 | 0.678 |
| XOR | **linear** | 0.000 | 0.000 | 1.001 | 1.002 | 1.000 | 0.988 | 1.001 |
| context (0.32) | interaction | 0.321 | +0.320 | 0.523 | **0.366** | 0.365 | 0.365 | 0.365 |
| context | **linear** | 0.000 | 0.000 | 0.684 | 0.684 | 0.682 | 0.678 | 0.684 |
| **context_noise** (0.32) | interaction | 0.330 | +0.321 | 0.840 | **0.818** | **0.684** | **0.684** | 0.684 |
| context, **worse full model** | interaction (b ≡ 0) | **0.327** | **−0.319** | 0.837 | 0.682 | **0.998** | 0.682 | 0.682 |
| redundant / irrelevant, worse full | (b ≡ 0) | 0.64 | −0.64 | — | — | — | — | — |

Square and logistic families give the same numbers to three decimals in every well-specified cell (same posterior; S).

### Reading
1. **Δ and its local version d(u) behave exactly as derived (D).**  The values are 0 / 0 / 1 / 0.5 / 0.5, scaled by (1 − 2ε)² under label
   noise, with S_UV − S_U = R*_U − R*_UV to machine precision.
2. **The squared disagreement z matches the realised gain h only when the full model is right** (residual within 0.01 in well-specified
   cells).
   - With a worse full model, z stays large (0.33–0.65) while the realised gain is negative (−0.32 to −0.64).
   - **The z-policy is then the worst policy of all:** 0.998 at budget 0.5, worse than random (0.837).  It acquires exactly where the bad
     model hurts most.
   - The h-policy correctly declines to acquire (0.682).
   - **The package's warning is confirmed at scale: z is not a safe policy target unless the full predictor is validated; the h (utility)
     target is.**
3. **Misspecified full models (no interaction) realise none of the Bayes gain** (z ≈ h ≈ 0 while Δ = 0.64 / 0.32).  No policy can recover it.
   Evidence gain is only actionable through a model class that can use V.
4. **Gain prediction beats cheap uncertainty only when uncertainty is partly irreducible.**
   - In context, low cheap confidence coincides with gain, so the plain uncertainty policy is as good as the gain policies (0.366 vs 0.365).
   - In context_noise (P), half of the uncertain inputs are pure noise.  There the gain policies (0.684) clearly beat uncertainty (0.818)
     and random (0.840), and reach the oracle.
5. **XOR negative check:** d(u) is constant, and no policy beats random at a fixed budget (all 0.678–0.684), as required.

---

## 3. What bears on the REG vs GAIN route choice (P)
| criterion | REG | GAIN |
|---|---|---|
| controlled evidence that the estimator readout drives a decision | yes: smooth, correct objective; precision and capture ≥ histogram MI in clean conditions | yes: d(u) gives the optimal acquisition rule; h-prediction realises it with finite models |
| clear negative controls / failure modes found | periodic ambiguity; coordinate and support leakage (guards work); **cross-scene photometric nuisance hurts the learned critic but not per-scene MI** | **z-policy failure under a bad full model**; misspecification removes all gain; uncertainty is as good as gain prediction unless part of the uncertainty is irreducible |
| where it can beat strong simple baselines | heavy noise (vs MI); otherwise MI / NMI is equal or better and adapts per image | inputs whose uncertainty is partly irreducible (context_noise type); otherwise uncertainty / entropy matches it |
| main natural-data risk | appearance variation across images (needs normalisation / invariant encoders, CoMIR-like); MI / NMI and CoMIR are strong baselines | the full model must realise gain on a subset predictable from U; the direct-h policy is standard risk regression, so the CS-specific contribution must be shown beyond it |

## 4. Next
- **REG:** any natural pilot must report per-image photometric normalisation (or invariant encoders) explicitly, and compare against
  per-image MI / NMI and CoMIR under the same optimiser, with fixed interior support and no coordinates.
- **GAIN:** a natural pilot (AV-MNIST first) must:
  - include the direct h-policy and cheap-uncertainty baselines;
  - report z vs h residuals;
  - test whether the data contain a context_noise-type regime (gain predictable from U and not aligned with uncertainty).
- Neither result is a natural-data claim; the route choice is R6-04.
