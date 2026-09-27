# Pre-registration — pre-check B, question 2: closed-form J* as a registration energy (P51), frozen 2026-09-27 before launch

**Property tested.** With a fixed pixel-intensity feature class the VCS objective has a closed-form optimum, so J*(ω) can be evaluated exactly
for every candidate transform ω without training — a registration similarity with the bounded, quadratic structure of the estimator.  Question 2
of the brief: compared with histogram MI / NMI on the same pixel pairs, does the J* energy surface have a wider capture range, fewer local
maxima and no divergence?

**Data (constructed; truth known).** 60 COCO val2017 images (identity-disjoint from the train2017 images used by A/C; seed 20260927; list in the
output JSON), 256×256 centre crops in grey [0, 1] = modality A.  Modality B = tent map b = 1 − |2 a^0.7 − 1| (non-monotone: correlation ≈ 0,
dependence high) × smooth sinusoidal gain field (±15 %) + 3 % Gaussian noise, aligned at ω* = identity.  Rigid transforms ω = (tx, ty, θ).

**Measures** (`scripts/precheck_b2_registration.py`; identical pixel sample and overlap mask per ω for all measures; 20 000 samples):
- VCS closed form: φ(a, b) = tensor product of Fourier features {1, cos/sin 2πka}_{k≤3} ⊗ {1, cos/sin 2πlb}_{l≤3} (49-d); w* = ½(A_M + λI)⁻¹d with
  λ = 1e-3·mean diag(A_M); J*(ω) = w*ᵀd − w*ᵀA_M w*.  No tanh (measurement, not training, as the brief allows).  No spatial coordinates in φ.
- MI, NMI (Studholme) from a 32×32 joint histogram of the same pairs.
Energy surfaces: translations ±24 px (step 2, θ = 0) and rotations ±30° (step 1, t = 0) per image → peak-at-truth rate, number of local maxima on
the translation grid, basin half-width (monotone descent from the truth).  Optimisation: Nelder–Mead on (tx, ty, θ) with ≤ 150 function
evaluations from 10 random initial offsets per radius R ∈ {5, 10, 20, 30} (|t| ≤ R px, |θ| ≤ R°), same initial points for all measures;
success = final error < 1 px and < 1°; divergence = |t| > 64 px.  Probe: --smoke (3 images, 2 inits) must run cleanly first.

**Pre-committed reading.**
- Property holds: success rate at R = 20 and R = 30 for VCS ≥ NMI + 0.10, mean number of local maxima ≤ NMI's, divergence rate 0, and peak at
  truth in ≥ 95 % of images.  Holds conditionally: advantage only at one radius or only in translation/rotation.  Does not hold: no success-rate
  difference (|Δ| < 0.05) or more local maxima than NMI.
- Value at truth vs at the corners is reported (contrast); J* at the truth is also an S lower bound for the constructed dependence.
- Not claimed: real cross-modal images (RGB-NIR, MRI) — proposed follow-up once a licensed source is confirmed; non-rigid transforms.

Outputs: `reports/P52_precheck_B2.md/.json` (surfaces and profiles per image kept).  Compute: one GPU job (feature/histogram evaluation is batched
on the device; the optimiser loop is Python).
