# P121 — v6 V6-THEORY: exact finite-model checks — report — 2026-10-03

Status: DRAFT (P121 fork; CPU only, no encoder training, no data).  Source: v6 package `VCS_Server_Tasks_v6_CN.md` §3, `VCS_Theory_to_Experiments_v6_CN.md`
§2–7.  Code: `src/vcs_theory/geometry_evidence_core.py` (v6 support core, copied verbatim with attribution), `src/vcs_theory/p121.py`,
`scripts/p121_theory.py`, `tests/test_p121.py` (13 tests incl. the package's own 27 reference checks run against the copied core).
Job: CPU 1020300 (exit 0, 607 s; an earlier run 1020282 without the two-tower context was superseded).  Outputs: `reports/P121_theory_checks.json`,
`reports/P121_gram_fit.csv`, `reports/P121_conditions.md`; derivations in `reports/P121_derivation_note.md`.
Labels (v6 §11.2): **[identity]** verified by definition / exact algebra; **[observed]** numerical result on the finite model; **[hypothesis]** untested.

## 1. Conditions (K = 4 latent classes, L = 12 states; strictly positive emissions; A ∝ U^γ, U ~ Unif(0.02, 1))
| condition | seed | γ | S | max |f*| |
|---|---|---|---|---|
| core_seed73 (= the v6 core's latent_model) | 73 | 1 | 0.01870 | 0.51 |
| weak | 11 | 0.5 | 0.00062 | 0.13 |
| moderate | 11 | 2 | 0.04293 | 1.11 |
| strong | 11 | 4 | 0.15967 | 2.84 |

## 2. Identities (float64; tolerance 1e-10 relative)  [identity]
- Posterior-kernel identity P_uv/(p_u p_v) = Σ_y P(y|u)P(y|v)/π_y: max relative error 4.3e-16 – 6.0e-16 (all conditions).  The kernel is PSD (rank ≤ 4);
  log(P/Q) is not (minimum eigenvalue −0.04 / −0.79 / −2.63 / −8.91 for weak / core / moderate / strong).
- Three-term nested decomposition S_W − J(T) = E(η−η_Z)² + E(η_Z−η_s)² + E(η_s−T)² over 50 random two-level compressions and arbitrary bounded
  scorers on s per condition: max relative residual ≤ 6.9e-16.  Data processing S_W − S_Z = E(η − η_Z)²: |difference| ≤ 1.4e-17.
- Oracle: J(η) = S exactly; constant-zero critic J = 0; global / per-anchor normalisation hold (and hold for f = 0 — necessary, not sufficient).
- Gradient formulas vs autograd: ∂A_c/∂s max abs error 2.7e-15 (affine and λ = ±¼); affine peak location error 1.6e-6 (grid spacing) and peak value
  32a/27 relative error 7.7e-12; tangent gradient ∇_r A_c error 2.8e-17; at s = 1 the encoder-side gradient is exactly 0 while the scalar gate is 0.20;
  matched-JS dL/df = T − c and VCS (T − c)(1 − T²) both exact (2.2e-16) and equal at f = 0 (−1 / +1).  Curvature scorer: f(0) = −1, f(1) = 1 for
  a = 2, κ = 0.5 at every λ; f′/a on [−1, 1] within [0.25, 1.25] (λ = −¼) and [0.75, 1.75] (λ = +¼) ⊂ [¼, 7/4]; actual zero 0.562 / 0.5 / 0.438 for
  λ = −¼ / 0 / +¼ (≠ κ when λ ≠ 0).

## 3. Gram realisability  [observed]
- G* = κ + log(P/Q)/(2a), five (a, κ) settings: symmetric to 1e-16; diagonal far from 1 (max |G*_uu − 1| 0.20–0.75 everywhere); range excess > 0 only
  for moderate / strong at small a or large |κ − ½| (up to 1.34); negative-eigenvalue mass 0.005 (weak, a = 3) to 4.3 (strong, a = 1); 42–50 % of
  eigenvalues negative.
- Direct unit-vector fits (all 3 starts reported, 2000 Adam steps, d ∈ {2, 4, 8}; best J/S over the five (a, κ) settings):

| condition | shared unit Gram d = 2 / 4 / 8 | two-tower cross-Gram d = 2 / 4 / 8 |
|---|---|---|
| core_seed73 | −2.98 / −0.55 / 0.27 | −2.07 / 0.99 / 1.00 |
| weak | −166 / −79 / −38 | −41 / 0.94 / 1.00 |
| moderate | −0.56 / 0.50 / 0.70 | −0.73 / 0.97 / 1.00 |
| strong | 0.73 / 0.91 / 0.91 | 0.78 / 0.97 / 1.00 |

  With one unit vector per state shared by both views the forced unit diagonal (T_uu = tanh(a(1 − κ)) against G*_uu ≈ κ) dominates: fits are often
  worse than the constant-zero critic, and larger κ helps.  With separate towers (no diagonal coupling) the same rank budget reaches J/S ≈ 1 at d = 8.
  Starts agree closely (identical to 3 decimals in most d = 8 cells).  These are feasible values, not optimality certificates.
- Two-sided Lipschitz bound: holds on every fit (both geometries); the lower side is ≈ 0 everywhere (sech⁴(B) negligible once |f| is large), so it
  gives no usable lower control.

## 4. Reading
- [identity] All exact relations the v6 plan relies on (posterior kernel, nested three-term decomposition, data processing, gradient formulas, the
  curvature scorer's anchors / slope bounds / moved zero, JS vs VCS matching at f = 0) hold to float64 precision on the finite model.
- [observed] In this toy the binding geometric constraint is the unit diagonal of a shared-embedding Gram, not the embedding dimension: a two-tower
  cross-Gram of rank 8 represents the bounded half-log-ratio almost exactly.
- [hypothesis] Whether the shared-diagonal effect matters for real SSL is untested: on continuous image domains the identical-view diagonal has
  measure zero; nothing here is a CIFAR result, and no scorer recommendation follows from this toy.

## Delivery
```yaml
logical_id: V6-THEORY
protocol_id: P121
experiment_family: theory check (finite exact model)
estimand: S of finite latent models; exact decompositions
status: complete (draft report; main session reviews / renames)
compute: CPU job 1020300, 607 s, 8 CPUs
```

## Main-session note (final)
- The two-tower cross-Gram fits go beyond v6 §3 (which specifies one shared unit-vector Gram); they are kept as **context**, clearly separated from the v6
  shared-Gram table.  Within this finite toy they indicate that the forced unit diagonal of a shared embedding — not the rank — is the binding constraint;
  this is a property of the toy (finite diagonal events), not a statement about real images (v6 theory note §4).
- Optimisation values are feasible upper bounds, not optimality certificates; the Lipschitz lower bound is ≈ 0 throughout, so geometry error and
  estimation error are not shown equivalent.
