# P121 — derivation note (v6 V6-THEORY)

Labels as v6 §11.2: **[identity]** = verified by definition / exact algebra; **[observed]** = numerical result on the finite model; **[hypothesis]** = not tested here.

## 1. Population objects  [identity]
Balanced label C ∈ {−1, +1}, M = (P + Q)/2, η = E[C | W] = (p − q)/(p + q), S = E_M η².  For any bounded T:
J(T) = E_P[T − T²/2] + E_Q[−T − T²/2] = 2 E_M[ηT] − E_M T², hence S − J(T) = E_M (T − η)² and J(T) = 1 − E(C − T)².
At the oracle T = η (the free per-pair critic of the finite model) J = S; the constant-zero critic gives J = 0.

## 2. Posterior kernel of the latent model  [identity]
Two views independent given Y:  P(u, v) = Σ_y π_y p(u | y) p(v | y),  p(u) = Σ_y π_y p(u | y).  Bayes: p(u | y) = P(y | u) p(u) / π_y, so
P(u, v) / (p(u) p(v)) = Σ_y P(y | u) P(y | v) / π_y = ⟨q(u), q(v)⟩ with q(u)_y = P(y | u) / √π_y.
The density-ratio kernel is therefore PSD of rank ≤ K; its logarithm (the optimal half-logit times 2) is in general **not** PSD (observed: minimum
eigenvalue of log(P/Q) −0.04 … −8.9 across the four conditions).  Global and per-anchor normalisation E_Q[P/Q] = 1, E_v[P(u, v)/(p(u)p(v))] = 1 hold
exactly; f = 0 satisfies them too, so normalisation is necessary, not sufficient.

## 3. Nested decomposition  [identity]
For deterministic W → Z → s and T = T(s): with η_Z = E[η | Z], η_s = E[η | s] (M-weighted conditional means),
η − T = (η − η_Z) + (η_Z − η_s) + (η_s − T); the cross terms vanish because η_s and T are Z-measurable (resp. s-measurable) and
E[η − η_Z | Z] = 0, E[η_Z − η_s | s] = 0.  Hence S_W − J(T) = E(η − η_Z)² + E(η_Z − η_s)² + E(η_s − T)².
The last term is a **scoring (calibration) error**, not an information loss: an injective T(s) keeps all information in s.
Data processing on the pushed-forward joint: S_W − S_Z = E(η − η_Z)² exactly.

## 4. Gram target and Lipschitz sandwich  [identity + observed]
With T = tanh(a(G − κ)) and f* = ½ log(P/Q) = atanh η, the element-wise ideal Gram is G* = κ + log(P/Q)/(2a).  If |f*|, |a(G − κ)| ≤ B, the
mean-value theorem gives sech²(B)|x − y| ≤ |tanh x − tanh y| ≤ |x − y|, hence a² sech⁴(B) ‖G − G*‖²_M ≤ S − J(G) ≤ a² ‖G − G*‖²_M.
Observed: both inequalities hold on every fit; the lower bound is numerically ≈ 0 in all conditions (sech⁴(B) is tiny once fits saturate), i.e. the
sandwich gives no usable lower control — as the theory note anticipated.

## 5. Gradient formulas  [identity, checked against autograd]
A_c(T) = cT − T²/2, T = tanh f(s):  ∂A_c/∂s = (c − T)(1 − T²) f′(s).  Affine f = a(s − κ): as a function of T, |(c − T)(1 − T²)| peaks at T = −c/3,
i.e. s₊ = κ − log2/(2a) (positives), s₋ = κ + log2/(2a) (negatives), peak value 32a/27; if a peak lies outside [−1, 1] the restricted maximum is
smaller (a = 0.5, κ = 0.5: s₋ = 1.19, domain maximum 0.585 vs 0.593).  With z = r/‖r‖: ∇_r s = (z_j − s z)/‖r‖, so
∇_r A_c = (c − T)(1 − T²) f′(s) (z_j − s z)/‖r‖; at s = 1 this vanishes although the scalar gate is non-zero (observed 0.20).  Matched JS with
logistic logit 2f has dL/df = T − c; VCS (−A_c) has (T − c)(1 − T²); equal at f = 0.
Curvature scorer f = a(s − κ) + aλ s(1 − s), |λ| ≤ ¼: f(0) = −aκ, f(1) = a(1 − κ), f′/a = 1 + λ(1 − 2s) ∈ [¼, 7/4] on [−1, 1]; the zero moves off κ
(a = 2, κ = 0.5: λ = −¼ → 0.562, λ = +¼ → 0.438).

## 6. What the realisability fits do and do not show
- [observed] With **one unit vector per state shared by both views** (symmetric Gram, unit diagonal), the best fitted J/S over the five (a, κ) settings
  is far below 1 and often negative (worse than the constant-zero critic): at d = 8, 0.27 (core), −38 (weak), 0.70 (moderate), 0.91 (strong).  The
  diagonal is forced to G_uu = 1, i.e. T_uu = tanh(a(1 − κ)), while the target G*_uu is ≈ κ + small (diag error 0.20–0.75 in every setting); with weak
  dependence (S = 6e-4) this forced mismatch dominates S.  Larger κ (0.75) reduces it.
- [observed] With **two towers** (separate unit vectors per side, cross-Gram, no unit-diagonal coupling) the fits reach J/S ≥ 0.96 at d = 8 in every
  condition and setting, every start (≥ 0.99 for the best start in 18 of 20), and the best setting reaches ≥ 0.94 already at d = 4; at d = 2 they fail in the weak / moderate conditions.
- [hypothesis, not tested] On continuous image domains the "same view twice" diagonal event has measure zero, so the finite-model diagonal penalty is
  not a statement about real SSL embeddings; the shared-vs-two-tower contrast only shows that the unit-diagonal constraint, not the rank, is the
  binding one in this toy.  Optimised values are feasible upper bounds, not optimality certificates; G* eigen-diagnostics are not solutions of the
  PSD / unit-diagonal / rank-constrained problem.
