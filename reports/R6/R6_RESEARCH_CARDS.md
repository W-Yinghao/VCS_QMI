# R6-01 — five research question cards (科学问题卡) — 2026-10-11

- **Package:** r6 (`r6/package_r6_20261011`).
- **Template:** `templates/RESEARCH_CARD.md`; drafts in `docs/09`; specs in `docs/03–07`.
- **Status:** candidates only.  No card is a training queue.
  - REG and GAIN get controlled checks (R6-02).
  - Exactly one of them goes to a natural-data pilot (R6-04).
  - NUISANCE / COMPRESS / AMORTIZE stay cards until their trigger holds.

**Labels:**
- **S:** from a source (collaborator plan O1, repository reports, published papers).
- **D:** direct derivation this round; standard, not claimed as new.
- **P:** proposal or hypothesis; untested on natural data.

## 0. Shared objects (used by every card; never mixed)

**(i) Original CS-QMI [S, O1].**
- P = joint law of (X, Y), Q = global product P_X ⊗ P_Y, M = (P + Q)/2, η = (p − q)/(p + q).
- S = E_M η²; D_CS^(M) = 2 log[(1 + S)/(1 − S)].
- Only this object is a "QMI".

**(ii) General two-distribution CS discrepancy / binary squared risk [S + D].**
- The same S for any P, Q on a common space.
- If C = ±1 is balanced, with P = Law(W | C = 1), Q = Law(W | C = −1) and M = Law(W):
  - η = E[C | W];
  - S = E η² = 1 − R*_W, where R*_W = min_T E(C − T)² is the squared Bayes risk [D].
- Not Shannon MI or conditional MI unless Q is the product of P's marginals.

**(iii) Multi-class Brier extension [D, standard].**
- q_W = P(Y | W) and R*_W = E‖e_Y − q_W‖².
- For nested observations U ⊂ (U, V): R*_U − R*_UV = E‖q_UV − q_U‖².
  - This is the Brier resolution increment (Murphy 1973; Bröcker 2009, verified).
- It is not a sum of per-class binary S values, and not conditional MI.

**Readout and repository facts [S].**
- J(T) = E_P(T − T²/2) + E_Q(−T − T²/2) and S − J(T) = E_M(T − η)².
- J ≤ S holds in population only; empirical Ĵ can exceed S or be negative.
- **Balanced RPC F₁,₁,₁ equals J** (Tsai et al., ICLR 2021, verified).  Repo R4-T0: value to 1e-12, gradient to 1e-14
  (`reports/R4/R4_RPC_DELTA_20261011.md`).
- **RuLSIF.**  RuLSIF with α = ½ estimates the relative ratio 1 + η (Yamada et al., NeurIPS 2011 / arXiv 1106.4729, verified).
- **Logistic posteriors.**  A matched logistic critic reaches the same posterior and is legitimately read through J; that is not a weakness of
  either method (R4-VL1/VL2, R5).
- **Statistical tools are standard.**  U-statistic / image-level intervals are standard instances (R4-T2).

---

## Card REG — can learned local cross-modal dependence drive correct geometric alignment?

**一个问题 / The decision.**  For a pair of images of one scene in two modalities with an unknown 2-D translation (then rigid) offset, choose the
transform φ that maximises the energy of a fixed, independently trained critic.  Success is the recovered geometry, not a pairing score.

**对象 / Object [P].**
- **P_φ:** draw a scene G uniformly, then a location x in a fixed support Ω; return (U_G(x), V_G(φ(x))).
- **Q_φ:** the same scene G, with independent locations x, x′; return (U_G(x), V_G(φ(x′))).
- **Type:** (ii), a *within-scene-reference* CS discrepancy.  Not (i): Q_φ is a mixture of within-scene products, not the global product.
- **W:** a local patch pair, with no coordinates, scene id or border mask.
- **Independent unit:** the scene / tile (Zurich: 20 tiles).  Patches and perturbations are conditional repeats.
- **Target:** F_T(φ) = J(T; P_φ, Q_φ) with T trained on training scenes at φ = id and then frozen.
  - This is a witness-function energy.  It is **not** S_φ = sup_T J, and no envelope argument is claimed.
- **Training:** encoders and critic on independent training scenes.
- **Deployment:** argmax_φ F_T(φ) with one shared multi-scale optimiser for every method.

**为什么需要这个 estimator / Structure used.**
- F_T(φ) = S_φ − E_{M_φ}(T − η_φ)² [D].  A matched critic loses both when misalignment destroys local dependence and when its posterior
  stops fitting.  That is the candidate reason for a bounded, quadratic matching energy.
- **Deterministic-observation trap [D, docs/03 §2]:** if V = g(U) is non-atomic, P ⟂ Q and S = 1 whatever the geometry.  Only local,
  scene-independent observations can carry geometric information.
- **What the estimator does not supply:** a plain aligned-vs-shuffled classifier *is* this critic.  The only estimator-specific choice is
  reading it through J instead of its native log-loss or InfoNCE score.

**已有与候选新增 / Prior art and candidate addition.**
- **Prior art (all verified):**
  - CoMIR (Pielawski et al., NeurIPS 2020): InfoNCE dense per-modality representations with rotation equivariance; then mono-modal
    registration; RGB–NIR and BF–SHG.
  - MINE-guided registration (Snaauw et al., arXiv 2201.10305; EMBC 2022 pp. 3510–3513 per its IEEE citation).
  - Classical MI / NMI: Viola & Wells, IJCV 1997; Maes et al., TMI 1997; Studholme et al., Pattern Recognition 1999.
  - CS point-set registration: Hasanbelliu et al., TPAMI 2014.
  - MIND descriptor: Heinrich et al., MedIA 2012.
- **Consequence:** "first CS / MI / neural-dependence registration" is **not** a candidate.
- **Hypothesis [P].**  Under identical local observations, scene-level training and one optimiser, F_T has a wider capture range or fewer
  false optima than:
  - the same critic's native energy;
  - InfoNCE;
  - MI / NMI.
  It also predicts that the critic's posterior error E_M(T − η)² tracks landscape quality.
- **Unproven:** no theory predicts a J-energy advantage over the same critic's log-loss energy.

**完整验证 / Validation.**
- **Data:**
  - Zurich Summer: 20 RGB-NIR tiles; tile-level splits; known synthetic rigid perturbations.
  - BF–SHG: Zenodo 4550300.
  - EPFL RGB-NIR: 477 pairs.
  - Access, licence and alignment checks are R6-03's job.
- **Two tables:**
  - Target isolation (same optimiser): VCS; matched logistic → J; logistic native energy; InfoNCE native; same-target kernel; MI / NMI; local
    correlation.
  - Full systems: CoMIR native pipeline; classical MI / NMI registration; MIND.
- **Evaluation:**
  - Transform error and TRE where landmarks exist.
  - Success rate vs initial offset (capture range); failure tail.
  - Registration time plus training cost.
  - ≥ 3 seeds; aggregated by scene.
- **Predictions:**
  - (D) REG-C0: on the binary sensor field P(u, v) = (1 + ρuv)/4, S = ρ²/(4 − ρ²) at the true shift and 0 off it.  argmax F_T must recover
    integer shifts exactly.
  - (P) Repetitive or flat textures must produce false optima for *every* energy.
  - (Threshold) The J-energy must match MI / NMI success at equal compute on held-out scenes, or REG has no application value.

**反例与替代 / Counterexamples and closure.**
- **Constructions that must fail:**
  - independent modalities;
  - periodic textures;
  - constant regions;
  - rotational symmetry;
  - coordinate / border leakage, which gives spurious peaks;
  - a cross-scene Q without scene-identity control.
- **If VCS ≈ logistic ≈ MI:** report the learned-relation effect only, with no VCS-specific claim.
- **If all methods fail:** fix geometry, support or interpolation first.  Then change the observation scale once, or close the pilot.

---

## Card GAIN — can the value of an extra modality be predicted before computing it?

**一个问题 / The decision.**  After observing only the cheap input U, compute the expensive branch V for this sample iff the predicted gain
ĝ(u) exceeds the cost c(u).  The threshold or budget is set on CAL.

**对象 / Object.**
- **Binary balanced task — type (ii):**
  - S_W = E η_W² = 1 − R*_W.
  - **Δ = S_UV − S_U = E(η_UV − η_U)² = R*_U − R*_UV ≥ 0** [D; nested conditional expectations].
  - Local gain d(u) = E[(η_UV − η_U)² | U = u].
  - Single-step Bayes rule under squared loss: acquire iff d(u) > c(u) [D, not a multi-step optimality claim].
- **Natural multi-class tasks — type (iii):** E‖q_UV − q_U‖² [D].
- **Imbalance [D]:** the balanced η is not the natural E[C | W].  Pick one target and keep it throughout.
- **Finite predictors a(U), b(U, V) — the central caveat [D]:**
  - z = (b − a)².
  - **h = (C − a)² − (C − b)² = z + 2(b − a)(C − b).**
  - E[h | U = u] is the realised gain of this model pair and can be negative.  z equals d only if a and b are the conditional means.
  - h − z is the diagnostic.
- **Independent unit:**
  - AV-MNIST: the source digit image and the FSDD recording.  Audio reuse across pairs must be tracked.
  - CREMA-D: the actor.
- **Roles:** FIT_TASK (predictors) / FIT_POLICY (out-of-fold z, h) / CAL (budget) / DEV.
- **Cost model:** cost = cost_U + cost_policy + overhead + 1[acquire]·(cost_V + fusion).

**为什么需要 / Structure used.**
- The quadratic geometry links an evidence increment (a difference of S values) exactly to squared-risk improvement, and splits a
  per-sample disagreement z from the realised gain h.
- Entropy of a(U) cannot tell "uncertain, V would help" (XOR) from "uncertain, irreducible noise" [P].
- **The strongest baseline is direct regression of h**, plain risk prediction with no CS ingredient.

**已有与候选新增 / Prior art and candidate addition.**
- **Prior art (all verified):**
  - EDDI (Ma et al., ICML 2019): expected information gain with a partial VAE.
  - AdaMML (Panda et al., ICCV 2021): learned per-segment modality policy, 35–55 % less compute.
  - Covert et al., ICML 2023: amortised greedy conditional-MI feature selection.
  - Norcliffe et al., ICML 2025: stochastic latent encodings beyond greedy conditional MI.
  - Guney et al., ICML 2025: explainability-driven acquisition ranking.
  - Kobayashi et al., "Learning-To-Measure": arXiv 2510.12624; ICML 2026 poster page.
  - Brier resolution decompositions: Murphy 1973; Bröcker 2009.
- **Consequence:** on-demand, information-driven acquisition is **not new**.  The Δ = R*_U − R*_UV identity is standard.
- **Hypothesis [P].**  In a full raw-input pipeline with measured costs, compare three gain targets at the same budget: z from a Brier/VCS
  predictor pair, h, and conditional-MI-style targets.  Which predicts realised risk improvement best?  Does h − z expose where
  "disagreement ≠ gain"?
- **Unproven:** if direct h-regression wins or ties, there is no CS-specific contribution.

**完整验证 / Validation.**
- **Data:**
  - The four controlled constructions with exact d(u): redundant, irrelevant, XOR, visible-context.
  - AV-MNIST (MultiBench loader; official 10 classes).
  - Then CREMA-D: 7 442 clips, 91 actors, actor-disjoint, raw video.
  - Access checks are R6-03's job; the official test is untouched.
- **Baselines:**
  - never / always acquire;
  - random at matched rate (repeated);
  - cheap entropy / confidence;
  - direct h-regression;
  - EDDI-style information gain;
  - an AdaMML-type policy where the task matches;
  - an oracle, as an upper bound only;
  - a 2 × 2 attribution: predictor family (Brier / VCS vs cross-entropy) × policy target (z vs h).
- **Evaluation:**
  - Brier and accuracy vs acquisition fraction (0 / .25 / .5 / .75 / 1).
  - Measured ms per sample, FLOPs and memory.
  - Calibration of predicted vs realised h.
- **Predictions (D, controlled):**
  - Redundant / irrelevant: d ≡ 0, so no policy beats never-acquire.
  - XOR: d is constant, so no policy beats random at the same budget.
  - Visible context: d(U) = K, so a correct policy acquires exactly where K = 1.
  - b worse than a: z > 0 while h ≤ 0, so a z-policy must lose to an h-policy there.
- **Threshold:** on AV-MNIST, the z-policy must beat random and entropy at matched budget.

**反例与替代 / Counterexamples and closure.**
- **Full model not better than cheap:** nothing to acquire; check training, not the theory.
- **Value not predictable from U:** static or random budgets are the correct answer.
- **Selector sees V, full logits or Y:** the policy is invalid.
- **FLOPs saved but not wall-clock:** report the measurement.
- **No difference:** report the residual h − z and stop.  Change the task (one where V's value varies with visible context), not the
  thresholds.

---

## Card NUISANCE (去干扰) — can an observable nuisance be made unusable without mistaking critic failure for success?

**一个问题 / The decision.**  Which representation (or penalty weight) to deploy when the nuisance–label correlation shifts at test time.

**对象 / Object.**
- P = P_Y P_{Z,N | Y} versus Q = P_Y P_{Z | Y} P_{N | Y}, with Z the representation and N the observable nuisance.
- **Type:** a class-conditional-product CS discrepancy (type (ii) on a Y-stratified space).  It measures Z ⟂ N | Y; it is not Shannon
  conditional MI [D].
- **Unit:** the image, grouped by (Y, N) cell.
- **Training:** task loss + λ · Ĵ, minimised over the encoder and maximised over the critic.
- **Readout:** a re-trained critic and independent probes on held-out data.

**为什么需要 / Structure used.**
- The bounded J penalty.
- **Caveat [S]:** J(T) ≤ S lower-bounds only.  A low training J under a non-converged maximisation can mean a fooled critic, not
  independence.

**已有与候选新增 / Prior art and candidate addition.**
- **Prior art (all verified):**
  - DANN (Ganin et al., JMLR 2016).
  - ReBias (Bahng et al., ICML 2020, HSIC).
  - Learning Not to Learn (Kim et al., CVPR 2019, MI with an adversarial bias predictor).
  - Adversarial debiasing with a label-conditioned adversary (Zhang, Lemoine & Mitchell, AIES 2018).
  - Group DRO (Sagawa et al., ICLR 2020), as the evaluation reference.
- **Hypothesis [P]:** the bounded quadratic penalty gives a min-max whose training readout *agrees* with independent strong probes and
  improves OOD accuracy.  Penalties by CS, HSIC or MI already exist.

**完整验证 / Validation.**
- **Data:** a Waterbirds-type controlled bias (version, licence and pretraining exposure to verify) and biased MNIST variants, each with a
  flipped-correlation OOD test.
- **Baselines:** ERM, DANN, ReBias, Learning Not to Learn, group DRO, and a matched logistic-critic penalty.
- **Metrics:** worst-group / OOD accuracy, and post-hoc probe accuracy for N given (Z, Y).
- **Falsifier:** a low training J alongside a high independent-probe accuracy refutes "J certifies removal".  No OOD gain over DANN or
  ReBias at equal task accuracy means no value.

**反例与替代 / Counterexamples and closure.**
- **Unobservable nuisance:** the card does not apply.
- **Nuisance truly predictive in the target domain:** removal must hurt.
- **Non-converged min-max:** a low J proves nothing.
- **Trigger:** an observable nuisance with a defined OOD test, *and* REG / GAIN not selected.  Card only.

---

## Card COMPRESS (任务相关压缩) — how much task evidence does a given budget keep?

**一个问题 / The decision.**  Allocate tokens / channels / bit-rate (g and budget) so that task risk stays within tolerance at the lowest
*measured* cost.

**对象 / Object.**
- **Not** teacher–student CS-QMI: for a deterministic non-atomic A = g(B), P ⟂ Q and S = 1 whatever the task [D, docs/03 §2].
- **Use the task objects:**
  - evidence loss Δ = S_B − S_A = R*_A − R*_B (type (ii));
  - or E‖q_B − q_A‖² (type (iii)), with A measurable w.r.t. B [D].
  - The structure is identical to GAIN, with the same finite-predictor caveat (z vs h).
- **Unit:** the input sample.

**为什么需要 / Structure used.**  Per-unit predicted evidence loss gives an allocation signal in units of task risk.

**已有与候选新增 / Prior art and candidate addition.**
- **Prior art (all verified):**
  - VIB (Alemi et al., ICLR 2017).
  - VID (Ahn et al., CVPR 2019).
  - CRD (Tian et al., ICLR 2020).
  - **CS-divergence distillation already exists:** Xu, Yang & Zhang, ICCV 2023, multi-task dense prediction.
  - Standard task-only pruning / compression.
- **Hypothesis [P]:** allocating by predicted evidence loss beats magnitude- or attention-based allocation at equal measured cost.
- **Unproven:** a gain over direct risk-increase prediction (h-type).

**完整验证 / Validation.**
- An existing model with a real compute or storage bottleneck.
- Error–compute and error–bit-rate curves.
- **Falsifier:** if evidence-loss allocation ≈ magnitude pruning at equal cost, the card has no value.

**反例与替代 / Counterexamples and closure.**
- **Saturation (S = 1) objects:** excluded.
- **Uniform importance across units:** allocation is moot.
- **Trigger:** GAIN's finite-risk tool independently validated, *and* a real budget bottleneck.  Card only.

---

## Card AMORTIZE (摊销估计) — can one conditioned critic replace per-pair fitting, and fail detectably?

**一个问题 / The decision.**  When many (P, Q) pairs need measuring (scenes in REG, modality pairs in GAIN), decide whether to use an
amortized critic or fall back to per-pair training.

**对象 / Object.**
- Per-pair S_{P,Q} (type (ii)).
- Critic T_ψ(w; D_ctx) conditioned on a small context set from (P, Q).
- **Readout:** J on an independent evaluation sample, kept separate from any directly regressed number.
- **Target:** S, not Shannon MI.

**为什么需要 / Structure used [S/D].**
- J(T) ≤ S holds in population for *any* T.
- An amortized critic's J on fresh data therefore stays a population lower readout even out of family.
- The gap E_M(T − η)² is exactly the amortization error.  A direct-number regressor has no such guarantee.

**已有与候选新增 / Prior art and candidate addition.**
- **Prior art (all verified):**
  - InfoNet (Hu et al., ICML 2024): MI without test-time optimisation.
  - Amortized ratio estimators (Hermans et al., ICML 2020).
  - Amortized CMI policies (Covert et al., 2023).
- **Hypothesis [P]:** reading through J makes out-of-family failure *detectable* (J falls far below a per-pair refit), where InfoNet-style
  outputs stay plausible but wrong.
- Amortization itself is not new.

**完整验证 / Validation.**
- Families with an oracle S (Gaussian / mixtures), plus out-of-family shifts.
- **Metrics:** oracle error, OOD-failure detection rate, and adaptation or fallback cost against per-pair training.
- **Falsifier:** J-readouts flag OOD failure no better than direct outputs.

**反例与替代 / Counterexamples and closure.**
- Contexts too small.
- Heavy-tailed shifts.
- **Trigger:** REG / GAIN shows per-pair fitting is the cost bottleneck.  Card only.

---

## Citation verification (2026-10-11 web check)

**Verified (title / venue / authors):**
- CoMIR (NeurIPS 2020).
- Snaauw et al. (arXiv 2201.10305; EMBC 2022 per its IEEE citation; formal venue taken from that citation only).
- Viola & Wells (IJCV 1997).
- Maes et al. (TMI 1997).
- Studholme et al. (Pattern Recognition 1999).
- Hasanbelliu et al. (TPAMI 2014).
- MIND (MedIA 2012).
- EDDI (ICML 2019).
- AdaMML (ICCV 2021).
- Covert et al. (ICML 2023).
- Norcliffe et al. (ICML 2025).
- Guney et al. (ICML 2025).
- Kobayashi et al. (arXiv 2510.12624; ICML 2026 poster page).
- InfoNet (ICML 2024).
- Hermans et al. (ICML 2020).
- RPC (ICLR 2021).
- RuLSIF (NeurIPS 2011; the package cites the arXiv version).
- Reid & Williamson (JMLR 2010).
- Murphy (1973).
- Bröcker (QJRMS 2009).
- DANN (JMLR 2016).
- ReBias (ICML 2020).
- Kim et al. (CVPR 2019).
- Zhang, Lemoine & Mitchell (AIES 2018).
- Sagawa et al. (ICLR 2020).
- VIB (ICLR 2017).
- VID (CVPR 2019).
- CRD (ICLR 2020).
- Xu, Yang & Zhang (ICCV 2023).

**Not re-verified here:**
- Dataset facts (Zurich, BF–SHG, EPFL, AV-MNIST, CREMA-D) are package statements; R6-03 audits them.
- The KLIEP-style normalisation used by R5-QN is cited in its own protocol, not here.
