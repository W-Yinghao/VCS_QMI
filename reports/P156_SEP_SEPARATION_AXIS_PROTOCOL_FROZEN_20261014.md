# Pre-registration — P156 (SEP-1): P134 extended into a separation-axis experiment — where on the same posterior the quadratic and the logistic scoring rule separate — FROZEN 2026-10-14T16:16:27Z

Status: **FROZEN 2026-10-14T16:16:27Z** (main session), after the CPU gate, the smoke and the GPU timing / reproduction smoke, and before any full unit.  The owner confirmed the rev. 1 defaults on 2026-10-14 ("确认，按默认值写代码跑测试和smoke").  The owner asked for it on 2026-10-14 ("把P134扩展成分离轴实验，先写协议").  Nothing below has been run.

The freeze comes after the CPU gate and a GPU timing smoke, and before any full unit. Freezing means renaming the file to `*_FROZEN_*`, adding the freeze time and a "Decisions at the freeze" section.

**rev. 1 (2026-10-14).** The owner pasted a review of rev. 0 and these changes take it in:
- **Mechanism corrected (§1).** The review withdraws its Appendix-4 story that the redescending weight protects VCS from contamination.
- **Capacity axis added (block E).** It includes the application's own critic form.
- **Lr grid widened at both ends.** A converged-budget arm is added on a second cell.
- **Saturation mass is a recorded covariate in every cell.**
- **Four outcomes are pre-written,** each with the sentence the paper may then use.
- **Pooled primary tests,** with per-cell Holm tests as secondary.
- **Predictions and falsifiers written out (§6).**

**rev. 2 (2026-10-14, after the CPU gate).** The block-B matching table (`reports/P156_S_MATCHING.md`, written before any block-B cell) shows that S-matching leaves the saturated mass unmatched by about a factor of 2, in opposite directions:
- At ε = 0.3 the contaminated law is **more** saturated than its S-matched clean Gaussian: 0.175 vs 0.091 at d 20, I 4, all of it on one side.
- At ε = 0.1 it is **less** saturated: 0.244 vs 0.362.

The S-matched contrast P2 therefore cannot by itself separate "contamination-specific" from "a different saturation profile at equal S". So rev. 2 adds:
- **Block B2:** a clean Gaussian whose ρ″ matches the contaminated law's saturated mass P_M(|η| > 0.95), at ε = 0.3, for 12 cells per seed and all four arms.
- **Secondary pooled test P2-sat,** plus per-cell Holm tests. The primary family is unchanged.
- **The interpretive rule in §6:** O2 is called contamination-specific only if P2-sat has the same sign; otherwise it is "not separable from the saturation profile".

One recommendation was checked and **not** adopted: dropping most false-negative cells under product negatives as an exact mirror of the independent type. §1 T4 gives the check. The posterior map is a mirror, but the two laws are not.

Sources:
- `reports/P134_R1_CONTAMINATION_PREREG_FROZEN_20261004.md` and its report.
- The R4-E0 same-target re-read: `reports/R4/P134_audit/`, and `reports/R4/R4_02_EVIDENCE_LEDGER_20261011.md` line 25 ("a local degradation advantage, not robustness").
- `r4/package_r4_20261010/docs/03_THEORY_RPC_AND_SAMPLING_CN.md` §3 and §7.
- `docs/05_REPAIR_AND_SMALL_CONFIRMATIONS_CN.md`: the common-posterior readout rules.
- `sources/SOURCE_ANALYSIS_WITH_APPENDIX4_CN.md` §C, and Appendix 4 §4.1 items 3–4 and §4.2 C.
- `reports/PROJECT_MAP_20261014.md` §4.4 and §11.B.

## 0. Why this unit
Matched JS (balanced logistic) and VCS (quadratic) estimate the **same posterior** η = (p − q)/(p + q). On every natural-data line and on the clean P85 ladder, JS is equal to VCS or 5–25 % better on posterior MSE (42 of 45 ladder cells).

The only reading on VCS's side is P134 under contamination, re-read by R4-E0 on the common contaminated posterior:
- JS keeps the lower absolute error almost everywhere.
- VCS degrades less from its own clean fit under independent-pair contamination: degradation ratio 0.53–0.70 at ε 0.1 and 0.2.
- At ε = 0.2, VCS's absolute error is lower in all three negative constructions, but every interval contains 0 (3 seeds): product −0.0086 [−0.0219, +0.0047], cyclic8 −0.0048, in-batch −0.0018.

The question is no longer "which rule wins". It is **in which region of (target saturation, contamination, critic capacity, budget) each rule is the better posterior estimator, and whether contamination defines a region of its own.**

## 1. Theory the design rests on
Each item below is algebra or a cited fact. None is a claim of this unit.

- **T1 — Same target.** For any data law (P′, Q′), every rule below has the same population minimiser η′ = (p′ − q′)/(p′ + q′). Under contamination the target moves to η_ε; P134 §2 shows that most of every shift is this target change. **No pair is "mislabelled" from the estimator's point of view.** The comparison is therefore always the fit error against η_ε.
- **T2 — Which rule should be more accurate.**
  - **Expressive (well-specified) class.** The logistic rule is the maximum-likelihood fit. Asymptotically it is the more efficient estimator of the posterior, including on posterior MSE, and its edge is expected to grow with saturation.
  - **Restricted (misspecified) class F.** The population quadratic risk is E_M(T − η)² + const. So the quadratic minimiser over F is the L²(M) projection of η onto F, and it has the lower posterior MSE by definition (Appendix 4 §4.1 item 3).
  - **Consequence.** The only sign reversal with theoretical support is in restricted classes.
  - **Budget.** At finite budget, JS's empirical edge may be an optimisation-speed effect. R4-E1 measured +0.0082 → +0.0027 from N 4 096 to 16 384, with no reversal.
- **T3 — The Beta family.** Let T = tanh g. The symmetric Beta family of proper scoring rules (Buja, Stuetzle & Shen 2005, a = b = γ) has weight function ∝ (1 − T²)^(γ − 1) and ∂ℓ_γ/∂g ∝ (1 − T²)^γ (T − c), with c = ±1.
  - γ = 0: logistic (matched JS), softplus(−2cg).
  - γ = 1: quadratic (VCS), ½(T − c)².
  - γ = ½: −sech g − c·gd(g), where gd(g) = 2 arctan tanh(g/2).
  - γ = 2: T²/2 − T⁴/4 − c(T − T³/3).
  - The four share T1's minimiser. Gradients and properness were checked numerically on 2026-10-14: gradient error ≤ 2e-8, finite at |g| = 20, and the pointwise minimiser equals η exactly.
  - **Any γ = 1 result is a Brier-vs-log result.** VCS inherits it; it is not a property of the CS divergence. If γ = 2 beats γ = 1 somewhere, VCS is not the best member of its own family there, and the paper must know that before writing a conclusion.
- **T4 — What contamination does to the target.**
  - **Independent-pair type (a).** P_ε = (1 − ε)P + εQ gives η_ε = (1 − ε)η/(1 − εη). The P side keeps its saturation. The Q side loses it: η → −1 gives η_ε → −(1 − ε)/(1 + ε), which is −0.54 at ε = 0.3.
  - **False-negative type (b).** Q_ε = (1 − ε)Q + εP gives η_ε = (1 − ε)η/(1 + εη) = −η_a(−η).
  - **Inverse maps.** Type (a): η = η_ε/(1 − ε + εη_ε). Type (b): η = η_ε/(1 − ε − εη_ε). The maximal slope is (1 + ε)²/(1 − ε): 1.34, 1.80 and 2.41 at ε = 0.1, 0.2 and 0.3.
  - **(b) is not a distributional mirror of (a).** Swapping P ↔ Q and T → −T leaves every rule invariant, but the Gaussian joint P and the product Q are not exchangeable: KL(P‖Q) = 4.00 nats vs KL(Q‖P) = 5.84 at d 20, I 4. A 200 000-pair Monte Carlo at ε = 0.3 (2026-10-14) gives:
    - S = 0.4252 for (a) vs 0.4311 for (b);
    - saturated mass P_M(|η_ε| > 0.95) = 0.176 vs 0.221, all of it on the P side for (a) and on the Q side for (b);
    - clean: S 0.8025 and saturated mass 0.613, of which 0.279 on the P side and 0.333 on the Q side.

    So (b) is run in full under product negatives too. It gives a second, different one-sided de-saturation profile. It is also run under cyclic negatives (block D′), where negatives are derived from the positives — the false-negative setting of SSL and VL.
- **Corrected mechanism (the review's correction of its own Appendix 4).** Take a pair whose true contaminated posterior is −0.54 (labelled P, Q-typical). The logistic gradient pulls a saturated T toward it, which is the correct direction. The quadratic gradient is ≈ 0 there. So the redescending weight is expected to **slow** VCS where the contamination lands, not to protect it. The smaller P134 degradation is more plausibly de-saturation and a lower S. Block B (S-matched) and the saturation covariate test this; the clean-target diagnostic (§4) is where a redescending pull toward the clean η would show.

## 2. Fixed parts (unchanged from P85 / P134)
- **Data.**
  - Gaussian generator (`benchmark.build_roles`); the per-coordinate correlation ρ comes from the total MI I over d coordinates.
  - Roles per distribution: FIT = N, TUNE = SELECT = N/4, EVAL = 32 768, TRUTH = 200 000.
- **Default critic and optimiser.** `JointMLP` concat → 256 → 256 → 1; Adam, batch 256.
- **Selection.** By each arm's own (native) risk on the **contaminated** SELECT role, every 100 updates. The estimator never knows which pairs are contaminated.
- **Pairing.** A seed gives every arm the same roles, the same initialisation and the same minibatch stream (`train_neural` seeding), so arms are paired.
- **Contamination records.** Contamination indices are recorded with a SHA-256 per role.
- **Negative law.** Q is the product of the actual contaminated marginals (spec v1 §11.1). The exception is type (b), whose negative law is Q_ε by definition.
- **Truths.** Closed form, or the P134 t₂ quadrature. The sampler/density QC is the bounded identity E_M η = 0.
- **Code.** P85 / P134 modules are not edited. New code goes in `src/vcs_estim/sep.py`, `scripts/p156_sep.py`, `scripts/p156_aggregate.py`, `slurm/p156_sep.sbatch` and `tests/test_p156.py`.

## 3. Design
**Arms.**
- γ = 0: matched JS, in the P85 Deep-InfoMax parameterisation T = tanh(f/2), i.e. g = f/2. This is the same class up to a factor 2 in the last layer.
- γ = ½: new, T = tanh g.
- γ = 1: VCS, the P85 form.
- γ = 2: new, T = tanh g.

**Learning-rate grids, widened at both ends.** In P134, JS chose P85's lowest value, 1e-4, in 47 of 51 cells (VCS in 29 of 51).
- MLP classes: {3e-5, 1e-4, 5e-4, 2e-3, 8e-3}.
- Parametric classes: {1e-3, 3e-3, 1e-2, 3e-2, 1e-1}.

For every arm and cell the record says whether the chosen lr is at a grid edge, and whether the chosen update is the last checkpoint (not converged). Edge rule: if an arm sits at an edge in more than half of a block's cells, one further point beyond that edge is run for all arms of the block. That addendum is frozen before any contrast is computed.

**Seeds.** 8 fresh seeds, 100–107 (P85 / P134 used 0–2). One reproduction cell — P134 clean, seed 0, P85 grid, VCS / JS product — must match P134 to GPU non-determinism. It is a QC check only.

**Restricted critic classes (block E).**
- **cos-2:** g = a·cos(x, y) + b. This is the application's critic form (SSL's fixed scorer, VL's a·cos + b). It is misspecified for the Gaussian, because log r depends on x·y and on |x|² + |y|².
- **quad-3:** g = a·x·y/d + b·(|x|² + |y|²)/d + c. It contains log r/2 exactly at ε = 0, so it is a well-specified parametric anchor. It is misspecified under contamination, because the mixture log-ratio is not quadratic.
- **MLP-8:** concat → 8 → 8 → 1, ReLU.
- The parametric classes start from a = b = c = 0 (T = 0) for every arm.

**Blocks.** All use 4 096 / 2 000 updates (N / updates) and product negatives unless stated otherwise.

| block | critic | contamination | ε | d | I (nats) | N / updates | arms | cells per seed |
|---|---|---|---|---|---|---|---|---|
| **A** | MLP-256 | (a) independent; (b) false-negative | 0, 0.05, 0.1, 0.2, 0.3 | 20, 64 | 2, 4, 8 | 4 096 / 2 000 | γ 0, ½, 1, 2 | 6 clean + 48 |
| **B** (S-matched clean controls) | MLP-256 | none: a clean Gaussian whose ρ′ gives S(ρ′) = S_ε of the matched A cell | matched to 0.1 and 0.3 | 20, 64 | from S_ε | 4 096 / 2 000 | γ 0, ½, 1, 2 | 24 |
| **E** (capacity) | cos-2, quad-3, MLP-8 | (a) | 0, 0.1, 0.3 | 20, 64 | 4, 8 | 4 096 / 2 000 | γ 0, ½, 1, 2 | 36 |
| C (budget / convergence) | MLP-256 | (a) | 0, 0.1, 0.2, 0.3 at (d 20, I 4); 0, 0.3 at (d 64, I 8) | — | — | 4 096 / 8 000; 16 384 / 2 000; 16 384 / 8 000 at (20, 4); 4 096 / 8 000 at (64, 8) | γ 0, 1 | 14 |
| D (secondary types) | MLP-256 | outlier, heavy (P134 laws rescaled to d) | 0.1, 0.2, 0.3 | 20, 64 | 4 | 4 096 / 2 000 | γ 0, 1 | 12 |
| **B2** (saturation-matched clean controls, rev. 2) | MLP-256 | none: a clean Gaussian whose ρ″ gives sat95(ρ″) = sat95_ε of the matched A cell | matched to 0.3 | 20, 64 | from sat95_ε | 4 096 / 2 000 | γ 0, ½, 1, 2 | 12 |
| D′ (negatives derived from positives) | MLP-256 | (a) and (b), cyclic8 negatives; in (b) each cyclic negative is replaced with probability ε by a fresh conditional draw y ~ P(y∣x_anchor) | 0, 0.1, 0.2, 0.3 | 20 | 4 | 4 096 / 2 000 | γ 0, 1 | 7 |

**S-matching.** ρ′ solves S_clean(ρ′; d) = S_ε by bisection on a 200 000-pair Monte Carlo of S_clean. The matching table, with Monte Carlo standard errors, is committed before block B runs.

**Saturation covariates, every cell, from the TRUTH role.** P_M(|η| > 0.95), split into the P-side mass (η > 0.95) and the Q-side mass (η < −0.95); P_M(|η| > 0.9); and S.

**No ε = 0.4.** The inverse-map amplification at 0.4 is 3.27. It stays a trigger-based extension (§8).

## 4. Readouts
The docs/05 rules apply: absolute error first, degradation second, and ratios only when both denominators have the same sign and are far from 0.

- **Primary:** e_k = E_{M_ε}(T_k − η_ε)², the posterior MSE against the contaminated posterior, under the contaminated mixture M_ε, on EVAL.
- **Δe = e_VCS − e_JS,** paired by seed; negative means VCS is more accurate. Also e for γ ½ and 2.
- **Secondary:**
  - **Clean-target diagnostic** (types (a) and (b) only): e_k^clean = E_M(clip(T̃_k) − η)² on the clean EVAL role. T̃ is the known-ε inverse map of T, and clip only acts outside [−1, 1]. ε is an oracle input, so this is a mechanism diagnostic, not a method. The amplification factor (1 + ε)²/(1 − ε) is printed next to it.
  - Degradation d_k = e_k(ε) − e_k(0), and Δd.
  - Ĵ_eval − S_ε.
  - The logistic regret E_{M_ε} KL(Ber((1 + η_ε)/2) ‖ Ber((1 + T)/2)): the log-score counterpart, so that a quadratic-score result is never shown without it.
  - Region contributions to e in clean-η bins: η ≤ −0.5, |η| < 0.5, η ≥ 0.5.
  - Selection-rule sensitivity: the state and lr chosen by the common quadratic score on SELECT.
- **QC:** as in P134, plus the lr-edge and not-converged flags.

## 5. Tests
The unit of inference is the seed (8 paired values). Intervals are paired 95 % t intervals, and tests are two-sided.

**Standardisation.** Each cell's paired effect is divided by its clean JS error ē_JS,c(0), taken from the same (class, d, I) at ε = 0, mean over seeds. Effects are then fractions of JS's clean error and comparable across cells. The version standardised by the across-seed sd is a sensitivity analysis.

**Primary family: three pooled tests, Holm over 3.** Each is the per-seed mean over its cells, then a t test over the 8 seeds.
- **P1 — Does contamination narrow the gap?** G_c = Δe_c(0.3) − Δe_c(0), over the 12 A cells: 2 types × (d, I) ∈ {20, 64} × {2, 4, 8}.
- **P2 — Is the narrowing contamination-specific?** H_c = Δe_c(0.3) − Δe_c(S-matched to 0.3), over the same 12 cells.  (Secondary companion, rev. 2: **P2-sat**, the same with the saturation-matched B2 cell; pooled and per cell.)
- **P3 — Do the restricted classes reverse the ordering?** Δe at ε = 0 over the 8 restricted cells: cos-2 and MLP-8 × {20, 64} × {4, 8}.

**Secondary, per cell, Holm within each list.**
- Δe(0.3) in the 12 A cells (absolute separation).
- G_c and H_c per cell.
- Δe in the restricted classes at ε 0.1 and 0.3.
- quad-3 at ε = 0, which tests the MLE-efficiency prediction.

**Descriptive.**
- The γ dose ordering of e at every cell.
- Region contributions; e^clean vs e.
- Blocks C, D and D′; the selection-rule sensitivity.
- The **saturation model:** per-cell relative Δe regressed on the saturated mass over clean, S-matched and contaminated cells, with an indicator for contaminated cells.
- The historical SOURCE_ANALYSIS §C ratio e_VCS/e_JS at ε 0.3, d 64, I 4 is quoted only. Its 0.7× prediction was withdrawn in Appendix 4 §4.2 C.

## 6. Outcomes, the paper sentence each permits, predictions and falsifiers (fixed now)
**Outcomes.** They can combine; the permitted sentences then combine.

| outcome | pattern | permitted sentence |
|---|---|---|
| **O1 saturation region** | P1 < 0; P2 n.s.; contaminated cells on the clean cells' Δe-vs-saturation line | "The logistic rule's advantage over the quadratic rule grows with target saturation; contamination narrows it only by de-saturating the target." |
| **O2 contamination-specific** | P1 < 0 and P2 < 0, and P2-sat < 0 (rev. 2; if P2-sat disagrees: "not separable from the saturation profile") | "Beyond its effect on the dependence level and saturation, mixture contamination narrows (or reverses) the logistic advantage"; reported with the clean-target diagnostic and the mechanism left open |
| **O3 no region** | P1 n.s. and P3 n.s. | "In this design no region was found where the quadratic rule estimates the posterior more accurately than matched logistic." |
| **O4 misspecification region** | P3 < 0 (VCS more accurate), with or without O1 | "When the critic class cannot represent the posterior, the quadratic (Brier) rule is the more accurate posterior estimator — the L² projection property of the Brier score, which VCS inherits; it is not a property of the CS divergence." |

**Predictions** (written before any run, from T2–T4 and the review):
- Clean cells: JS ≤ VCS at every I, with the gap growing with I.
- S-matched and contaminated cells: similar gaps at the same S.
- Restricted classes: VCS ≤ JS.
- quad-3 at ε = 0: JS ≤ VCS (MLE efficiency).
- The γ axis: in saturated clean cells, error rises monotonically with γ; in de-saturated or restricted cells, the minimum moves to γ ≥ 1.
- The clean-target diagnostic does not favour VCS more than e does.

**Falsifiers.**
- If the restricted classes leave JS no worse (P3 n.s. or > 0), the "Brier projection advantage" does not exist at this scale, and every "better than JS" sentence is deleted.
- If contaminated cells narrow but S-matched cells do not (O2), the saturation explanation is wrong and another mechanism is needed before anything is written.
- If P1, P2 and P3 are all n.s., the SOURCE_ANALYSIS §C kill condition applies. The paper writes VCS as "a member of the balanced proper-scoring family with an exact gap identity and a bounded target" and removes every "better than JS".

**Not claimed.**
- A robustness theorem.
- Recovery of the clean target as a method.
- Anything about SSL, VL or image features. The label-channel truth on natural features is only an upper bound (Appendix 4 §4.2 D).
- Other generators, classes or budgets.

**Prior art cited with any result.**
- Buja, Stuetzle & Shen 2005.
- Bounded and redescending losses under label noise: Masnadi-Shirazi & Vasconcelos 2008; Manwani & Sastry 2013; Ghosh, Kumar & Sastry 2017; Zhang & Sabuncu 2018.
- Known-rate noise correction: Natarajan et al. 2013; Patrini et al. 2017.
- The mixture structure of P_ε (positive-unlabelled learning): Elkan & Noto 2008; du Plessis, Niu & Sugiyama 2014.
- False negatives in contrastive learning: Chuang et al. 2020.

## 7. Gate before the freeze (CPU tests, CPU smoke, GPU timing)
**CPU tests.**
- Beta-family gradients against autograd up to |g| = 20.
- Properness on a finite toy.
- γ = 0 and γ = 1 equal the P85 JS / VCS per-pair losses, up to constants and the g = f/2 reparameterisation.
- The (b) closed-form log-ratio against Monte Carlo; E_M η = 0; and the identity rejecting a wrong density, as in P134.
- Both inverse maps: round trip to 1e-12, with clipping only outside [−1, 1].
- S-matching reproduces S_ε within 2 Monte Carlo standard errors.
- The saturation fields against direct counts.
- quad-3 represents log r/2 exactly at ε = 0 (oracle parameters reach posterior MSE ≈ 0).
- cos-2 and quad-3 for both parameterisations.
- The cyclic false-negative injection: exact counts, and the per-pair law equals Q_ε.
- ε = 0 roles bit-for-bit P85 for every (d, I); determinism; the aggregate on toy cells.

**CPU smoke.** 5 tiny cells end to end, one per block type.

**GPU timing.** Two full cells, not reused: (d 64, I 8, ε 0.3, all four arms) and (16 384 / 8 000). Written to `outputs/P156_TIMING/`.

## 7b. Gate record (filled from the jobs)
**Code.** `src/vcs_estim/sep.py`, `src/vcs_estim/sep_agg.py`, `scripts/p156_sep.py`, `scripts/p156_aggregate.py`, `slurm/p156_sep.sbatch`, `slurm/p156_cpu.sbatch` and `tests/test_p156.py`. P85 and P134 modules are untouched.

**CPU gate, job 1040214 (first run).** 77 of 78 passed; the matching table, the smoke cells and the toy aggregate ran. The one failure was the S-matching test's tolerance, not the code. It compared a fresh re-estimate with the target using two Monte Carlo errors and omitted the third: the error of the bisection's own sample, which ρ′ inherits. A production-size check (n 200 000, three fresh draws in each of three rows) put every re-estimate within about 1.5 three-source s.e. of S_ε, so the matching is accurate to about 0.001 in S. The test now uses n 200 000 and the full error budget.

**CPU gate, job 1040227 (rev. 2 code).**
- **80 of 80** tests passed: `tests/test_p156.py` (23 test functions, parametrised) plus the P134 suite.
- The matching table, with the S- and saturation-matched columns, was committed with this revision.
- Smoke cells, two per block, ran for A, E, C, D, D′, B and B2.
- The aggregate ran on the smoke cells.

**GPU gate, job 1040236 (L40S node39; exit 0).**
- **Reproduction (QC).** The P134 clean cell, seed 0, P85 grid:
  - JS chose lr 1e-4 at update 1000, posterior MSE 0.09024, identical to P134 to 5 decimals.
  - VCS chose lr 5e-4 at update 300, 0.09496 vs P134's 0.09502. The difference of 6e-5 is GPU non-determinism (L40S here, RTX6000PRO in P134).
  - Both choices are the same as P134's.
- **Timing.**
  - (d 64, I 8, ε 0.3, four arms × five lrs): 72 s, i.e. 3.5 s per fit.
  - (d 20, I 4, ε 0.3, N 16 384 / 8 000 updates, two arms × five lrs): 130 s, i.e. 12.9 s per fit.
  - Written to `outputs/P156_TIMING/` and not reused.
- **Disclosure.** That job's log and the JSONs print the selected lr, the selected update and the read-outs of its two timing cells; they were seen before the freeze. The full runs recompute every cell under its own block name. The timing cells use the "T" block and are never aggregated.

## 8. Cost, priority, decisions
**Cost estimate,** to be replaced by the timing smoke. P134's figure was ≈ 4.3 s per fit at d 20, N 4 096 and 2 000 updates on RTX6000PRO; the small classes are faster.

| blocks | fits per seed | estimated time per seed |
|---|---|---|
| A + B | 78 cells × 20 | ≈ 2.1 h |
| E | 36 × 20 | ≈ 0.4 h |
| C | 14 × 10, at 1–4× the updates | ≈ 0.5 h |
| D + D′ | 19 × 10 | ≈ 0.25 h |

Block B2 (rev. 2) adds ≈ 0.35 h per seed. That was the estimate: **≈ 3.6 GPU-h per seed and ≈ 29 GPU-h for 8 seeds** (the timing smoke gives ≈ 3 GPU-h per seed on L40S, ≈ 24 GPU-h in all): one resumable job per seed of about 3.5 h, so short work stays grouped. Partitions RTX6000PRO, H100 and L40S, with node51 and node60 excluded. No image data, no official test set, no SSL training.

**Decisions.** The review's recommendations are listed below; the owner confirms them at the freeze.
1. γ arms: **all four** (γ 0, ½, 1, 2). The review's reason is to see the trend and whether γ = 2 beats γ = 1.
2. Seeds: **8.** P134's three seeds suggest ≈ 5 are needed for 80 % power at the Holm threshold in the VCS-favoured independent-type cells, and a three-seed sd is itself unstable.
3. ε: **up to 0.3,** with no 0.4. Adding 0.4 is a trigger-based extension, because its amplification is 3.27.
4. S-matched controls: **at ε 0.1 and 0.3,** with the saturation fields in every cell.
5. Priority: the review says "Table C first, this unit second, add. 7 stays paused". Both of those are already complete: VL1-22 Table C was reported on 2026-10-10 and VL3 addendum 7 on 2026-10-11. The proposal is therefore to run P156 on free GPU slots after the current V7X, REV9 and P155 jobs. **The owner confirms which "Table C" is meant.**

## Decisions at the freeze (main session, 2026-10-14T16:16:27Z)
1. **Defaults as confirmed by the owner (2026-10-14):** four γ arms, 8 seeds (100–107), ε up to 0.3, S-matched controls at ε 0.1 and 0.3, and saturation fields in every cell.
2. **Block B2 (rev. 2)**, the saturation-matched controls at ε 0.3, is included as a secondary contrast; the primary family is unchanged. The owner was informed on 2026-10-14. B2 cells run last in every seed, so an owner objection can remove them by an addendum before any B2 cell runs.
3. **Submission:** 4 jobs of 2 seeds each — `SEED=100+101`, `102+103`, `104+105`, `106+107` — as one job per seed would push the normal-QOS count past 27. Each job is about 6 h, with `--time=12:00:00`. Partitions RTX6000PRO, H100 and L40S, with node51 and node60 excluded. Nice 100, so the V7X and REV9 groups already pending start first; the owner asked for this unit after them.
4. **Aggregation:** `scripts/p156_aggregate.py --dir outputs/P156_sep --out reports/P156_SEP_results/p156_aggregate` once all 8 seeds are in. Results-only commit first, then the report. Any lr-edge addendum (§3) is frozen before the aggregate is computed.
