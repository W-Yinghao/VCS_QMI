# Pre-registration — P156 (SEP-1): P134 extended into a separation-axis experiment — quadratic vs logistic scoring rule on the same posterior under contamination — DRAFT 2026-10-14

Status: **DRAFT for the owner's review** (owner 2026-10-14: "把P134扩展成分离轴实验，先写协议").  Nothing below has been run.  The freeze (rename to
`*_FROZEN_*`, freeze time, "Decisions at the freeze") happens after the CPU gate and a GPU timing smoke, before any full unit.
Sources: `reports/P134_R1_CONTAMINATION_PREREG_FROZEN_20261004.md` and its report; the R4-E0 same-target re-read (`reports/R4/P134_audit/`,
`reports/R4/R4_02_EVIDENCE_LEDGER_20261011.md` l.25: "a local degradation advantage, not robustness"); `r4/package_r4_20261010/docs/03_THEORY_RPC_AND_SAMPLING_CN.md`
§3 and §7; `docs/05_REPAIR_AND_SMALL_CONFIRMATIONS_CN.md` (the common-posterior readout rules); `sources/SOURCE_ANALYSIS_WITH_APPENDIX4_CN.md` §C
(contamination as the separation axis) and Appendix 4 §4.1 items 3–4, §4.2 C (corrections and the withdrawn 0.7× prediction);
`reports/PROJECT_MAP_20261014.md` §4.4 and §11.B.

## 0. Why this unit
Matched JS (balanced logistic) and VCS (quadratic) estimate the **same posterior** η = (p − q)/(p + q); on every natural-data line and on the clean P85
ladder JS is equal or 5–25 % better on posterior MSE (42 / 45 cells).  The only reading on VCS's side is P134 under contamination, re-read by R4-E0 on
the common contaminated posterior: JS keeps the lower absolute error almost everywhere, but VCS degrades less from its own clean fit under
independent-pair contamination (degradation ratio 0.53–0.70 at ε 0.1 / 0.2), and at ε = 0.2 its absolute error is lower in all three negative
constructions (product −0.0086 [−0.0219, +0.0047], cyclic8 −0.0048, in-batch −0.0018; 3 seeds, every interval contains 0).  This unit asks, with
enough seeds, fresh data, a wider ε range and the controls the re-read lacked: **is there a region where the quadratic rule fits the same posterior
better than the logistic rule, and is that region defined by contamination, or only by the dependence level / saturation of the target?**

## 1. Theory the design rests on (each item is algebra or a cited fact; none is a claim of this unit)
- **T1 Same target.**  For any data law (P′, Q′) every proper scoring rule below has the same population minimiser η′ = (p′ − q′)/(p′ + q′).  Under
  contamination the target moves (P134 §2: most of every shift is the target change); the comparison is therefore the fit error against the
  contaminated posterior, never "which estimand".
- **T2 Within-class ordering.**  Inside one function class F, the population quadratic risk equals E_M(T − η)² + const, so the population VCS
  minimiser over F has posterior MSE ≤ that of the population logistic minimiser over F, by definition (Appendix 4 §4.1 item 3).  JS's empirical
  edge is therefore a finite-sample / optimisation effect (R4-E1: mostly the optimisation path), and the testable prediction is that it shrinks
  with N and with the update budget (R4-E1: +0.0082 → +0.0027 from N 4 096 to 16 384, no reversal yet).
- **T3 Gradient weight (the Beta family).**  With T = tanh g, the members ℓ_γ of the symmetric Beta family of proper scoring rules (Buja, Stuetzle &
  Shen 2005, α = β = γ) have ∂ℓ_γ/∂g ∝ (1 − T²)^γ (T − c), c = ±1:
  γ = 0 → logistic (matched JS, softplus(−2cg)); γ = 1 → quadratic (VCS, ½(T − c)²); γ = ½ → ℓ = −sech g − c·gd(g) (gd = Gudermannian,
  2 arctan tanh(g/2)); γ = 2 → ℓ = T²/2 − T⁴/4 − c(T − T³/3).  For γ > 0 a confidently wrong pair (|T| → 1, sign opposite to c) gets weight → 0
  (redescending); for γ = 0 its pull stays |T − c| ≈ 2.  All four share T1's minimiser, so γ is a pure scoring-rule dose axis.
- **T4 What contamination does to the target.**  Independent-pair contamination P_ε = (1 − ε)P + εQ gives η_ε = (1 − ε)η / (1 − εη) (docs/03 §7):
  the P side keeps its saturation (η → 1 ⇒ η_ε → 1), the Q side loses it (η → −1 ⇒ η_ε → −(1 − ε)/(1 + ε)).  The mirror, false-negative
  contamination Q_ε = (1 − ε)Q + εP (dependent pairs among the negatives, the SSL / VL failure mode), gives η_ε = (1 − ε)η / (1 + εη): the P side is
  capped at (1 − ε)/(1 + ε), the Q side keeps its saturation.  Inverse maps (known ε): η = η_ε / (1 − ε + εη_ε) and η = η_ε / (1 − ε − εη_ε).
- **Three explanations a VCS advantage under contamination could have, and the readout that separates them.**
  - **E-level:** the VCS − JS gap is a function of the dependence level S alone; contamination only lowers S.  → S-matched clean controls (§3 block B).
  - **E-desat:** VCS's clean handicap is slow fitting of saturated regions (vanishing (1 − T²)); contamination removes saturation on one side.  →
    the gain sits in the de-saturated region (Q-confident pairs for the independent type, P-confident pairs for the false-negative type), mirrors
    between the two types, and shows on the contaminated-target error at least as much as on the clean-target diagnostic.
  - **E-robust:** the redescending weight down-weights contaminating pairs, pulling VCS's fit toward the clean η.  → VCS's relative advantage is
    larger on the clean-target diagnostic (oracle-ε inversion) than on the contaminated-target error, which it can even worsen.

## 2. Fixed parts (unchanged from P85 / P134)
Gaussian generator (`benchmark.build_roles`; per-coordinate correlation ρ from the total MI I over d coordinates); roles FIT = N per distribution,
TUNE = SELECT = N / 4, EVAL = 32 768 per side, TRUTH = 200 000 per side; critic `JointMLP` concat → 256 → 256 → 1; Adam, batch 256; selection by each
arm's own (native) risk on the **contaminated** SELECT role every 100 updates (the estimator never knows which pairs are contaminated); the same
seed gives every arm the same roles, the same critic initialisation and the same minibatch stream (`train_neural` seeding), so arms are paired;
contamination indices recorded with a SHA-256 per role; Q is the product of the actual contaminated marginals (spec v1 §11.1) except in the
false-negative type, whose negative law is the contaminated Q_ε by definition.  Truths of every contaminated problem in closed form (Gaussian
mixtures) or by the P134 t₂ quadrature, with the bounded identity E_M η = 0 as the sampler / density QC.  P85 / P134 modules are not edited; the new
code lives in `src/vcs_estim/sep.py`, `scripts/p156_sep.py`, `scripts/p156_aggregate.py`, `slurm/p156_sep.sbatch`, `tests/test_p156.py`.

## 3. Design
**Arms (product negative construction unless stated):** γ = 0 (matched JS, the P85 Deep-InfoMax parameterisation T = tanh(f/2), i.e. g = f/2 — the
same function class up to a factor 2 in the last layer), γ = 1 (VCS, P85 form), γ = ½ and γ = 2 (new, T = tanh g).
**Learning-rate grid:** {3e-5, 1e-4, 5e-4, 2e-3}, all arms.  Reason: in P134 matched JS selected the lowest P85 value 1e-4 in 47 / 51 cells (VCS in
29 / 51), i.e. at the grid edge; the extra lower point is added before any new run.  Edge rule: if an arm selects the lowest point in more than half
of the cells of a block, one further point (1e-5) is run for all arms of that block in an addendum frozen before the contrasts are computed.
**Seeds:** 8 fresh seeds (100–107; P85 / P134 used 0–2).  One reproduction cell: the P134 clean cell, seed 0, P85 grid, VCS / JS product — must
match P134's selected posterior MSE to GPU non-determinism (QC, not analysed).

| block | contamination | ε | d | I (nats) | N / updates | arms | cells per seed |
|---|---|---|---|---|---|---|---|
| **A (primary)** | independent; false-negative | 0, 0.05, 0.1, 0.2, 0.3 | 20, 64 | 2, 4, 8 | 4 096 / 2 000 | γ 0, ½, 1, 2 | 6 clean + 48 |
| **B (S-matched clean controls)** | none: a clean Gaussian at the same (d, N) whose ρ′ gives S(ρ′) = S_ε of the matched block-A cell | matched to ε 0.1, 0.3 | 20, 64 | from S_ε | 4 096 / 2 000 | γ 0, ½, 1, 2 | 24 |
| C (budget) | independent | 0, 0.1, 0.2, 0.3 | 20 | 4 | 4 096 / 8 000; 16 384 / 2 000; 16 384 / 8 000 | γ 0, 1 | 12 |
| D (secondary types) | outlier, heavy (P134 laws, rescaled to d) | 0.1, 0.2, 0.3 | 20, 64 | 4 | 4 096 / 2 000 | γ 0, 1 | 12 |
| D′ (construction) | independent, cyclic8 negatives | 0, 0.1, 0.2, 0.3 | 20 | 4 | 4 096 / 2 000 | γ 0, 1 | 4 |

S-matching: ρ′ solves S_clean(ρ′; d) = S_ε by bisection on a 200 000-pair Monte Carlo of S_clean (the matching table, with Monte Carlo s.e., is
written and committed before block B runs); the matched clean cell also records the saturation mass P_M(|η| > 0.9) of both laws, so the shape
difference that S-matching leaves is visible.  ε ≤ 0.3 keeps the independent type inside the positive RPC family (ρ < ½, docs/03 §7).

## 4. Readouts (docs/05 rules: absolute error first; degradation second; ratios only with same-sign denominators far from 0)
For arm k in a cell: **e_k = E_{M_ε}(T_k − η_ε)²** on EVAL (contaminated problem, common posterior; the existing `posterior_mse` field) — primary.
**Δe = e_VCS − e_JS** (paired by seed; negative = VCS better).  Degradation d_k = e_k(ε) − e_k(0) (same seed, same (d, I)) and Δd — secondary.
Also per cell: Ĵ_eval − S_ε (existing `J_eval_minus_S`); the logistic regret E_{M_ε} KL(Ber((1 + η_ε)/2) ‖ Ber((1 + T)/2)) (the JS-native counterpart,
so a quadratic-score win is never shown without the log-score reading); region contributions of e_k in clean-η bins (η ≤ −0.5, |η| < 0.5, η ≥ 0.5);
the **clean-target diagnostic** e_k^clean = E_M(clip(T̃_k) − η)² on the clean EVAL role, T̃ = the known-ε inverse map of T (independent and
false-negative types only; oracle ε, a diagnostic for E-robust vs E-desat, not a method; the inverse map's maximal slope (1 + ε)²/(1 − ε) is printed
next to it); selection-rule sensitivity: the same readouts with the state and lr chosen by the common quadratic score J(tanh g) on SELECT.
QC per cell as P134 (exact contaminated counts, recorded indices, E_M η = 0 within 4 s.e., finiteness, non-finite steps counted, lr-edge counts).

## 5. Pre-stated contrasts and reading
Unit of inference = seed (8 paired values); 95 % paired t intervals; Holm within each family; two-sided.
- **Family 1 (primary, 12 tests):** Δe at ε = 0.3, independent and false-negative types × (d, I) ∈ {20, 64} × {2, 4, 8}.
- **Family 2 (attribution, 12 tests):** Δe(ε = 0.3) − Δe(S-matched clean), same 12 cells (paired by seed).
- **Descriptive (pre-stated, no family):** Δe at ε 0.05 / 0.1 / 0.2 and its per-seed least-squares slope in ε; Δd; the γ dose ordering of e at every ε;
  region contributions; e^clean vs e; blocks C, D, D′; the selection-rule sensitivity; the historical SOURCE_ANALYSIS §C ratio e_VCS / e_JS at
  ε = 0.3, d = 64, I = 4 (its 0.7× prediction was withdrawn in Appendix 4 §4.2 C; quoted only).
- **Reading (words fixed now):**
  - **R1 absolute separation:** Family 1 has Holm-significant Δe < 0 cells → "under {type} contamination at ε = 0.3, in cells {list}, the
    quadratic rule fits the contaminated posterior more accurately than matched logistic"; Holm-significant Δe > 0 cells are reported with the same
    weight ("logistic fits more accurately").
  - **R2 attribution, only for R1 cells:** Family 2 significant < 0 → "beyond the dependence level: contamination is a separation axis"; otherwise
    → "explained by the dependence level: the ordering tracks S, contamination is one route to a less saturated target" (and the axis to report is S
    / saturation, not contamination).
  - **R3 mechanism (descriptive):** E-desat is favoured if the R1 gain is concentrated in the de-saturated bin, mirrors between the two types, and
    e^clean does not favour VCS more than e; E-robust is favoured if VCS's advantage is clearly larger on e^clean than on e.  Neither pattern → "mechanism
    not identified".
  - **R4 budget (descriptive, prediction from T2):** the clean gap Δe(0) shrinks with N and updates (block C); whether the ε at which Δe crosses 0
    moves with the budget.
  - **Kill condition (SOURCE_ANALYSIS §C, kept):** no Holm-significant Δe < 0 cell in Family 1 → "contamination is not a separation axis"; the
    paper then writes VCS as "a member of the balanced proper-scoring family with an exact gap identity and a bounded target" and removes every
    "better than JS" (that source's own wording).
- **Not claimed:** a robustness theorem (Appendix 4 §4.2 C); recovery of the clean target as a method (ε is an oracle input); anything about SSL,
  VL or image features (the label-channel truth on natural features is only an upper bound, Appendix 4 §4.2 D); other critic classes, generators or
  budgets.
- **Prior art to cite with any result:** bounded / redescending and Beta-family losses under label noise (Buja, Stuetzle & Shen 2005; Masnadi-Shirazi &
  Vasconcelos 2008; Manwani & Sastry 2013; Ghosh, Kumar & Sastry 2017; Zhang & Sabuncu 2018), known-rate noise correction (Natarajan et al. 2013;
  Patrini et al. 2017), the mixture structure of P_ε (positive-unlabelled learning, Elkan & Noto 2008; du Plessis et al. 2014), false negatives in
  contrastive learning (Chuang et al. 2020).  The unit's question is narrow: where, with exact contaminated truth, the quadratic member separates from
  the logistic member of the same family.

## 6. Gate before the freeze (CPU, then a GPU timing smoke)
Tests: Beta-family gradients equal (1 − T²)^γ (T − c) to float64 precision against autograd, including |g| up to 20 (no NaN at saturation);
properness (on a finite toy every γ's population minimiser equals η); γ = 1 and γ = 0 equal the P85 VCS / JS per-pair losses up to constants and the
g = f/2 reparameterisation; false-negative closed-form log-ratio vs Monte Carlo, the bounded identity E_M η = 0, and the identity rejecting a wrong
density (P134's power check); both inverse maps (round trip to 1e-12, clipping only outside [−1, 1]); S-matching bisection reproduces S_ε within 2
Monte Carlo s.e.; ε = 0 roles bit-for-bit P85 for every (d, I); determinism; the aggregate on toy cells.  CPU smoke: 4 tiny cells end to end.  GPU
timing: 2 full cells (d 64 / I 8 / ε 0.3, and block C 16 384 / 8 000), written to `outputs/P156_TIMING/`, not reused.

## 7. Cost and submission (estimate; replaced by the timing smoke)
From P134 (≈ 4.3 s per fit at d 20, N 4 096, 2 000 updates on RTX6000PRO): blocks A + B ≈ 78 cells × 16 fits, C ≈ 12 cells × 8 fits at 1–4× the
updates, D + D′ ≈ 16 cells × 8 fits → **≈ 2.5–3 GPU-h per seed, ≈ 20–24 GPU-h for 8 seeds**, one resumable job per seed (≈ 3 h each; short work grouped
by seed as the owner asked).  Partitions RTX6000PRO / H100 / L40S, node51 and node60 excluded.  No image data, no official test set, no SSL training.

## 8. Open decisions for the owner (defaults in bold)
1. Arms: **γ ∈ {0, ½, 1, 2}** (the dose axis) vs only γ ∈ {0, 1} (halves blocks A–B).
2. Seeds: **8** vs 5 (5 makes Holm over 12 hard to pass with paired t on 4 df).
3. ε range: **up to 0.3** vs adding 0.4 (still inside the positive RPC family; +12 cells per seed).
4. S-matched controls at **ε 0.1 and 0.3** vs every ε.
5. GPU priority relative to the V7 external-baseline queue and REV9: **fill free slots after the current V7X / REV9 groups**, or put it ahead.
