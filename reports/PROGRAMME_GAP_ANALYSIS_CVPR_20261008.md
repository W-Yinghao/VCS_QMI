# Where the programme stands against a CVPR manuscript — gap analysis and remaining plan — 2026-10-08

Written from the theory outward: for each claim the paper needs, what the theory says, what the evidence now supports (development split, raw run
files), what is missing, and what to run.  Status sources: `STATUS_20261008.md`, `r3_intake/R3_PHASE_REPORT_20261008.md`,
`VL1/VL1_BASELINES_AND_REPRODUCTION_PLAN_20261008.md`, `status_v7.md` (197 finished runs).

## 0. The claim chain the paper rests on
The object is S = sup_T J(T) over bounded T, with the exact gap S − J(T) = E_M (T − η)², η = (p − q)/(p + q), M = (P + Q)/2.  Everything else is
a consequence of estimating or optimising J with a bounded critic:
- **T1 estimation**: J is a fit-limited lower estimate of S; bounded, no exp / log, same posterior as the balanced logistic (matched JS).
- **T2 learning**: an encoder trained to maximise the all-view-token J with the fixed scorer learns the view-relation; the objective has no
  temperature, no log-sum-exp, a built-in "balanced posterior" target.
- **T3 measurement**: the same critic measures dependence in frozen representations (two-view relation, planted attributes), with nulls.
- **T4 application (new)**: with few region–phrase annotations, the same estimator learns a calibrated relation score on frozen vision-language
  features; task accuracy is read on the same object.
A CVPR paper needs T1 + T2 at scale, T4 as the vision application, T3 as diagnostics.  The table below is the status per claim.

## 1. Status per claim (what theory predicts → what we have → gap)
| claim | theory says | evidence (dev split) | gap / honest limit |
|---|---|---|---|
| **T1a precision on the shared target S** | matched JS has the same η → equal at the optimum; differences are finite-sample / optimisation | P151: Gaussian ladder posterior MSE VCS .065→.027, JS .061→.018, kernels .08–.70; xor and cubic ladders same order | **JS ≥ VCS on S at every level** — the paper must not claim better precision; the defensible claim is "bounded quadratic fit = same-target precision as JS, far above kernel routes" |
| **T1b stability (O1 §12.2)** | bounded T ⇒ no exponential moments ⇒ bounded gradients | P152 (54 recorded fits): 0 non-finite steps for every in-batch method; gradient CV InfoNCE ≤ JS ≤ VCS ≪ NWJ / DV; VCS least saturated, no |f| > 10 overflow at its lr; product-sampled DV / NWJ diverge (P86) | **VCS is not uniquely the most stable** (InfoNCE lowest gradient CV); claim = "bounded objectives + InfoNCE stable, exponential-family not; VCS least saturated". W7 |
| **T1c J vs plug-in (O1 §4, §10)** | S_plugin = (E T_P² + E T_Q²)/2 is exact only for calibrated T | P108: plug-in smaller bias in 70/72 synthetic cells; **P149: on real images plug-in inflates under weak signal (0.80 vs J 0.35)** | the paper reports J on real data and plug-in as context (W8); must state both results |
| **T1d resolution / invariance / DPI (O1 §8)** | S invariant under invertible maps; non-increasing under channels | P149 (+add. 1): 59 / 60 encoder × estimator × site cells resolve the channel ladder; orthogonal-refit drift ≤ ~2 SE; seed-stable | complete; **W11**: fitted J across sites (z > h by 0.10–0.17) is not a DPI statement |
| **T1e decisive figure (O1 §17.5)** | estimator / gradient variance vs dependence strength | P151 + P152 panels exist as JSON / tables | **figure not drawn** — same-target precision panel + P152 stability fields, MI methods on their own targets, log 1024 bound (W6) |
| **T2a VCS vs SimCLR, R18** | — | C10 89.02 ± .04 (5 s) vs 88.31 ± .21; C100 60.15 ± .38 vs 58.25 ± .35; linear > kNN for VCS, reverse for SimCLR | solid at R18; **CIFAR-only** |
| **T2b VCS vs matched JS, R18** | same posterior ⇒ the gap is an optimisation / finite-sample effect, must be explained not assumed | C10 +0.21 (shared scorer; close); C100 +0.80 clear vs JS-AP3, but selected-JS (3, 0.5) = 60.49 (−0.49 clear) and lr-dependent (P129a4: +0.16 / −0.52 / +0.26) | **not a robust headline**; present as "within ~0.5 under tuned scorers; VCS has fewer knobs (no temperature) and lower saturation" |
| **T2c scale: ResNet-50** | — | C100 R50 seed 0: VCS 64.66 / JS 62.56 / SimCLR 58.54; seed 1 VCS 65.04 (replicates); JS / SimCLR seeds 1–2 running (done ≈ 2026-10-09/10); P141: equal at layer3, the gain arises in layer3 → h | **wait for the chains**; then the strongest SSL result (+6 over SimCLR) is 3-seed |
| **T2d scale: beyond CIFAR** | — | P132 preflight, P139 pipeline, **P154 ImageNet-100 frozen + gate passed (924 img/s, 7.5 h / run)**, runs cancelled at the owner's request | **largest credibility gap for a CVPR vision paper**: no non-CIFAR SSL number.  Cost to close: 3 runs ≈ 22 GPU-h (1 seed) |
| **T2e mechanism** | balanced posterior; all pairs as negatives | P114 / P123 (pair-weight diagnostics), P115 (crop factor: SimCLR gains from small crops, VCS not), P145 (contamination: no difference at ε 0.10), P141 (layer profile), P149 add. 1 (trained z saturates J ≈ 0.97) | exists in pieces; **one mechanism figure** needed (pair-weight share + layer profile + saturation at z) |
| **T2f ablations** | — | scorer (a, κ) P112 / P129; views, epochs, batch (P136 / P147), lr (P135), wd (P148), projector (P137), momentum keys (P133), curvature (P126), K = 16 sampling (P138: non-inferior for VCS C10 only), crop (P115, P153 −2.2 C10), STRESS (P145; ε 0.20 running) | **complete** — a compact ablation table from `status_v7.md`; no new cells |
| **T3 measurement** | planted-attribute detection with nulls; detection ≠ estimation | P142 (logistic ≈ VCS > RFF > HSIC), P143 (colour detectable in balanced-posterior encoders), P150 (detection at power 0.60 with J ≈ 0) | complete; **present as detection tables (W9)**, compression claim dropped (P144) |
| **T4a VL same-feature table (A)** | on fixed features positive affine rescaling cannot change ranking (O1 §3.5) ⇒ gains need the joint MLP; VCS and JS share η ⇒ expect ≈ equal Top-1 at large N; the difference, if any, is at small N and in calibration | infrastructure done today: RefCOCOg-UMD audit (25,799 images, roles, no leakage), pair-law port (48 tests), fitter for 6 rows; raw-CLIP job queued | **no result yet**; 36 selected cells + lr grid ≈ 2–3 CPU/GPU days |
| **T4b VL public baselines (B)** | — | ReCLIP official code + rebuilt inputs (sentence counts = paper), 2 val runs queued; FG-CLIP 2 Base downloaded; SigLIP 2 / FLAIR planned | ReCLIP check against 68.08 / 65.32 (UMD val) first |
| **T4c full grounding (C)** | — | plan only (MDETR, Grounding DINO Swin-T OGC, proposer + VCS) | needs a clean proposer (retrain on COCO-2014-train minus RefCOCO val/test, ≈ 1 GPU-day) and two more envs |
| **T4d second VL dataset** | — | Flickr30k Entities annotations downloaded; images need the official form | **owner action** |
| **final numbers** | — | **every number is development-split**; official CIFAR test and RefCOCOg UMD test closed | one final pass on frozen recipes (`evaluate_final_official_test`, VL test file) — a scheduled decision |

## 2. What is genuinely missing (ranked by effect on acceptance)
1. **A non-CIFAR SSL result** (T2d).  P154 is frozen and gated; 22 GPU-h buys VCS / JS / SimCLR at ResNet-18 / 200 epochs on ImageNet-100
   with the same readout.  Without it the learning claim is "CIFAR + R50-CIFAR".
2. **The VL application with results** (T4a–b): Table A on CLIP features and the ReCLIP / FG-CLIP 2 / SigLIP 2 rows; the paper's vision
   contribution is empty until these exist.  Table C is the "feasibility" row; with Grounding DINO Swin-T (zero-shot, no RefCOCO) and MDETR it
   is a 1–2 week item and can go to the appendix if time is short.
3. **The decisive estimator figure and the reframed §5.1 text** (T1e + W6–W11): all numbers exist; this is drawing + wording.  The honest
   framing — same-target precision equal to matched JS, bounded / unsaturated / no failures, exact gap identity, J (not plug-in) on real
   data — is defensible; "VCS is more precise / more stable than MI estimators" is not.
4. **R50 seeds 1–2** (running) and the final **official-test pass**.
5. One **mechanism figure** (T2e) and the **ablation table** (T2f) — assembly only.
6. Flickr30k Entities (owner form) for a second VL dataset; RefCOCO+ as the conditional extension.

## 3. Remaining plan (three blocks, ≈ 3 weeks of server time; owner writes the paper in parallel)
**Block 1 — this week (closers + VL Table A).**
- GPU: P130 R50 chains (finish 10-09/10), P145 add. 2 (ε 0.20), P138 add. 2, P153 C100 — all submitted; reports as they land.
- VL: raw-CLIP DEV table → VL1-10 gate → freeze → Table A (6 routes × N {1k, 4k, all} × 3 seeds; CPU-bound, 4–6 jobs); ReCLIP val
  reproduction (queued) then ReCLIP on DEV; writing: ablation table from `status_v7.md`, decisive-figure script (P151 + P152 JSON).
- **Decision D1 (owner): re-enable P154 (ImageNet-100, 22 GPU-h) as soon as the R50 chains free GPUs** — recommended.
**Block 2 — week 2 (public baselines + strong features).**
- FG-CLIP 2 Base (official region API) and SigLIP 2 Base (crop / blur adaptation) on the same candidates; Table A re-run on FG-CLIP 2 region
  features; Flickr30k Entities given-box matching if the images have arrived; mechanism figure.
- **Decision D2 (owner): budget for the clean proposer (≈ 1 GPU-day)** for Table C; MDETR / Grounding DINO envs built meanwhile.
**Block 3 — week 3 (grounding feasibility + final numbers).**
- Table C rows (proposer + VCS / matched controls, MDETR fine-tuned + pretrained, Grounding DINO Swin-T OGC) on RefCOCOg val; freeze all
  recipes; **one official-test pass** (CIFAR official test for the frozen SSL recipes; RefCOCOg UMD test for the frozen VL configurations).
- **Decision D3 (owner): the date of the official-test pass** (nothing is tuned after it) and whether Table C is main-paper or appendix.

## 4. Suggested paper skeleton (so every section has its evidence)
1. Theory: S, exact gap, bounded critic, matched-JS posterior equivalence, J vs plug-in (T1c).
2. Estimator benchmark: decisive figure (T1e), same-target precision (T1a), stability fields (T1b), real two-view relation (T1d).
3. SSL: CIFAR R18 (T2a/b), R50 (T2c), ImageNet-100 (T2d, if run), mechanism figure (T2e), ablations (T2f, appendix).
4. Vision-language: Table A (same features), Table B (public models on given boxes), Table C (grounding feasibility, appendix if short).
5. Measurement diagnostics (T3) in the appendix; limitations: JS ≥ VCS on S; VCS vs tuned JS within ±0.5 on CIFAR; saturation at z; S_plugin on
   real data; detection ≠ estimation.

## 5. Design recommendation (owner 2026-10-08: the paper is about the estimator / critic; JS ≈ VCS on CIFAR is expected, not a target)
**Thesis:** one bounded quadratic critic with an exact gap identity, used three ways — estimate a dependence (S), train a representation by
maximising it, and probe frozen representations with it.  Matched JS / logistic is the *same-posterior control*: equality with it confirms the
theory (same η) and is reported as such; no further VCS-vs-JS tuning units.
**Experiment design per section**
1. Estimator (centrepiece): the decisive figure — precision / stability / resolution vs dependence on the oracle ladders, VCS and JS as the
   bounded pair against MINE/DV, NWJ, InfoNCE, SMILE and the kernel routes; the real two-view relation (P149) as the no-oracle panel
   (invariance, channel ordering, saturation).  Done except the figure.
2. Learning: "maximising J trains an encoder as well as the best contrastive recipe, with no temperature and no log-sum-exp" — a sufficiency
   result (CIFAR R18, R50, ImageNet-100), plus the link objective-value ↔ measurement (J of the trained z ≈ 0.97, P149 add. 1).  No SimCLR
   tuning, no JS tuning; ablations in the appendix.
3. Probing: detection tables (P142 / P143) with the detection ≠ estimation caveat (P150).
4. Vision-language (the application): **keep the given-box design** — it isolates the critic and is the one setting where the theory's object
   (conditional joint vs product with explicit multi-positive laws) is exactly realised.  Primary readouts: common J / calibration on CAL and
   the sample-efficiency curve N ∈ {1k, 4k, all}; task Top-1 as the sanity check against the same-backbone zero-shot method (ReCLIP).  Two
   theory-driven additions, both cheap: (a) **prior-corrected ranking** — with non-uniform candidate priors (referred + distractor objects)
   rank by 2f + log p(r) and test that it beats raw f (O1 §3.4 prediction); (b) **estimator as a probe of foundation features** — the same
   P/Q laws on CLIP, FG-CLIP 2 and SigLIP 2 features: J measures how much region–phrase relation each feature set carries.
   **Drop Table C (full grounding vs MDETR / Grounding DINO) from the main paper**: those systems are trained on 1.3M grounding pairs and answer a
   different question; at most one appendix row (Grounding DINO Swin-T zero-shot vs proposer + VCS) if time allows.  SigLIP 2 / FLAIR rows
   optional.
**Consequences for the queue:** finish the running closers (P130 R50, P145 add. 2, P138 add. 2, P153) and add no SSL exploration or robustness
cells; re-enable P154 (ImageNet-100) when GPUs free; VL order = raw CLIP → Table A → ReCLIP → FG-CLIP 2 probe → prior-correction test.
