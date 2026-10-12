# R4-02 — complete evidence restatement (evidence ledger) — 2026-10-11

Package task R4-02 (`r4/package_r4_20261010/docs/02_SERVER_PLAN_CN.md` §R4-02; docs/05 §4–5; docs/06 §2–4).  Snapshot HEAD `77b6a40`.
- **Read-only.**  No raw JSON, results table or old report was edited.  Nothing was refitted.  Sources: committed reports and aggregates,
  pre-registrations, the estimator code (for the negative-construction check) and directory listings of `outputs/`.
- **Old items** get a `SUPERSEDED_BY_*.md` note next to them (§5).
- **Companion files:**
  - `R4_02_claim_ledger.csv` (columns of the package template `claim_ledger.csv`).
  - `R4_02_result_rows.csv` (columns of `result_columns.csv`; the registered same-target comparison at the reference cell).
- This ledger replaces `reports/CVPR_EVIDENCE_LEDGER_20260928.md` as the evidence ledger.

**Claim types** (package prompt: theory inference / exploratory prediction / practical threshold; docs/05 §5):
- **T** theory or direct algebra: an identity, numerically checked where stated.
- **E-reg** measured and read by a pre-registered rule.
- **E-exp** measured, exploratory or descriptive: development split reused, post-hoc grouping, oracle-assisted diagnostic.
- **H** explanatory hypothesis or exploratory prediction, not tested.
- **G** practical threshold: a decision rule fixed in advance.  It decides what runs next; it is not a finding.

## 0. Research sentences that change
| # | old sentence (where) | restated | type |
|---|---|---|---|
| 1 | "VCS-N beats every same-target kernel estimator on all axes" (P86 §1 / §9, CVPR ledger, r3 analysis) | On the **registered partner** (product negatives, the same N joint + N product units as the kernel rows), VCS is better than S-KDE, Nyström and RFF on every axis by rule 1.  Against RuLSIF-tanh it is better on four of five axes and **on par on sample size** (RuLSIF-tanh lower at N ≤ 1 024).  K = 8 cyclic or in-batch negatives lower the neural error by a further 30–60 %, at 3× time or 17× memory and with no extra data (§2). | E-reg |
| 2 | "Ĵ remains a valid / guaranteed lower readout" (P108 §2 / §6, withdrawn in its addendum); "J value is a valid lower bound on S" (P47); "as a lower bound on S the raw closed form …" (P48); "measurement J is a lower readout" (P109) | J(T) ≤ S holds **in population** for every bounded T.  An evaluated Ĵ is a finite-sample estimate with its own SE, not a per-run bound.  The squared plug-in of the same fitted critic reads S more accurately in 70 / 72 cells (\|bias\|) and 69 / 72 (RMSE) (§3). | T + E-reg |
| 3 | "tuned SimCLR", "under equal tuning budgets both controls exceed …" (52 report lines; CVPR ledger §A2 / §E) | "SimCLR recipe (τ 0.2, AdamW 1e-3) at VCS's view / batch / epoch knobs (P41)".  Since 2026-10-04 SimCLR receives no tuning.  Every VCS-vs-SimCLR statement compares the **tuned VCS family** with the **SimCLR recipe** (§4.1). | label |
| 4 | P134 headline "R1 supported" | This is the result of P134's normalised-shift rule only.  On the common contaminated posterior:<br>• JS has the lower absolute error in 19 / 48 cells (VCS in none).<br>• VCS degrades less from its own clean seed in 9 / 48 (JS in none), mostly under independent-pair contamination at ε 0.10–0.20.<br>This is a local degradation advantage, not robustness (R4-E0). | E-reg (re-read) |
| 5 | P151 §2–3: JS stability and ordering read on native values | On the common J (R4-01): seed spread comparable (JS smaller in 24 / 45 ladder cells).  Common-J ordering probabilities are not computable: the draws were never saved. | E-reg |
| 6 | VL3 B16 / interim pooled readings | Complete 12-cell, 3-seed table: VCS − softmax +0.01 [−0.35, +0.36].  The seed-0 rule had biased the pooled view by +0.3. | E-exp (DEV) |
| 7 | multi-seed intervals on cells seeded after a seed-0 screen (most SSL rows; VL3 before add. 3) | Conditional on the screen.  The screening seed is part of every mean (§6.3). | exposure |
| 8 | "InfoNCE bound log 256" (P86 §5) | log 1 024 = 6.93 nats on the EVAL blocks of 1 024 anchors (P151 W6) | T |

Unchanged:
- VCS = balanced RPC F₁,₁,₁ (R4-T0).
- On S, matched JS ≥ VCS in posterior MSE in 42 / 45 ladder cells (R4-01).
- Posterior MSE values (R4-01).
- The VL results of VL3, Table C, R4-VL0 and R4-VL1 as reported.
- The second-application verdicts (CVPR ledger §B).
- The P68 test numbers.

## 1. Scope of the restatement
The package names five items: (a) the registered kernel comparison, (b) P108, (c) external baselines, (d) supersession and (e) exposure.  Each is
§2–§6 below.  The claim ledger CSV lists every claim-able statement with its type, status, assumptions, allowed conclusion and what it does not imply.

## 2. (a) Same-target kernel comparison: the registered partner, and the multi-negative effect with its cost

### 2.1 What was registered
- **Registration.**  P85's methods table reads: "negatives — vcs, js: product ∣ cyclic8 ∣ inbatch … **product = the same-target comparison partner of
  the kernel rows**".  FIT = N joint pairs plus an independent product sample of N.
- **Code check** (`src/vcs_estim/benchmark.py`):
  - The product branch draws its Q batch from the FIT product sample (`Fqx[iq], Fqy[iq]`).
  - The cyclic-8 and in-batch branches build their negatives from the joint pairs only.
  - The RFF, Nyström and RuLSIF features use the FIT joint and product samples (`Fp, Fq`).
  - S-KDE uses the joint pairs.
  - So the product rows and the kernel rows see the **same independent data**.
- **What P86 did instead.**  P86 §1 read rule 1 on the cyclic-8 rows ("VCS-N below means the K = 8 cyclic-shift construction (the recipe's)").
- **This re-read.**  Rule 1 applied unchanged to the registered rows: "better" if lower on ≥ ¾ of an axis's values by more than 2 pooled seed sd.
  Source: committed `P86_p85_aggregate.md` (3 seeds, selected rows).  Rule 1 is stated for VCS-N; JS(product) is given descriptively.

### 2.2 Rule 1 with the registered partner: VCS(product) against each kernel, per axis
| axis (values) | vs S-Kernel RFF | vs RuLSIF-tanh | vs S-KDE | vs Nyström |
|---|---|---|---|---|
| irrelevant dimensions d_total 2 / 10 / 20 / 50 / 100 | better (4 / 5; worse at 100) | better (4 / 5; worse at 100) | better (4 / 5; worse at 100) | better (4 / 5; worse at 100) |
| sample size N 256 / 1 024 / 4 096 / 16 384 | better (3 / 4; worse at 256) | **on par** (better at 4 096, 16 384; worse at 256, 1 024) | better (4 / 4) | better (4 / 4) |
| batch, equal updates 64 / 256 / 1 024 | better (3 / 3) | better (3 / 3) | better (3 / 3) | better (3 / 3) |
| batch, equal exposure 64 / 256 / 1 024 | better (3 / 3) | better (3 / 3) | better (3 / 3) | better (3 / 3) |
| dependence staircase (gaussian, cubic, xor × I 2–10; 15 cells) | better (14 / 15; worse at xor I 2) | better (13 / 15; worse at gaussian I 2, xor I 2) | better (14 / 15; worse at xor I 2) | better (14 / 15; worse at xor I 2) |

P86's cyclic-8 reading was "better on every value of every axis".  The cells where the registered partner loses (posterior MSE, mean ± sd over 3
seeds; the trivial estimate T ≡ 0 has posterior MSE = S):

| cell | S | VCS(product) | JS(product) | RFF (sel.) | RuLSIF-tanh | S-KDE | Nyström | VCS cyclic-8 |
|---|---|---|---|---|---|---|---|---|
| d_total 100 | 0.540 | 0.546 ± 0.001 | 0.542 ± 0.001 | 0.540–0.542 | 0.541 ± 0.002 | 0.537 ± 0.001 | 0.541 ± 0.001 | 0.354 ± 0.048 |
| N 256 | 0.802 | 0.563 ± 0.024 | 0.592 ± 0.032 | 0.508 | 0.491 ± 0.004 | 0.624 ± 0.007 | 0.676 ± 0.029 | 0.364 ± 0.031 |
| N 1 024 | 0.802 | 0.223 ± 0.012 | 0.207 ± 0.005 | 0.273 ± 0.006 | 0.179 ± 0.008 | 0.607 ± 0.006 | 0.550 ± 0.011 | 0.146 ± 0.010 |
| gaussian I 2 | 0.568 | 0.112 ± 0.005 | 0.113 ± 0.010 | 0.175 | 0.095 ± 0.002 | 0.441 ± 0.001 | 0.404 ± 0.001 | 0.066 ± 0.004 |
| xor I 2 | 0.580 | 0.588 ± 0.002 | 0.582 ± 0.002 | 0.576 ± 0.002 | 0.571 ± 0.002 | 0.524 ± 0.001 | 0.580 ± 0.003 | 0.400 ± 0.042 |

Reading (E-reg):
- **Every kernel and every axis.**  With the registered partner, rule 1 gives "better" against S-KDE, Nyström and RFF on every axis, and against
  RuLSIF-tanh on four of five axes.
- **Sample size.**  VCS(product) and RuLSIF-tanh are on par: the closed-form ridge read-out is more accurate at N ≤ 1 024.
- **Where the losses sit:**
  - the near-trivial corner (d_total = 100, xor I 2), where every product or kernel estimate is within 0.06 of the trivial value S;
  - the weakest Gaussian level (vs RuLSIF-tanh);
  - small N.
- **JS(product)** shows the same picture.

### 2.3 The multi-negative effect and its cost, listed separately
Reference cell: gaussian, d 20, I 4 nats (S 0.802), N 4 096, B 256, 2 000 updates.  Fit seconds and peak memory from the P85 cost table on one GPU.
| estimator / construction | independent FIT units | pair evaluations per update, pos / neg (total over 2 000 updates) | candidates per anchor | registered selection grid | fit s / peak MB | posterior MSE |
|---|---|---|---|---|---|---|
| VCS, product | 4 096 joint + 4 096 product | 256 / 256 (0.51 M / 0.51 M) | 1 (not anchored) | lr × 3; SELECT every 100 updates | 5.1 / 140 | 0.095 ± 0.002 |
| VCS, cyclic-8 | 4 096 joint (product sample unused) | 256 / 2 048 (0.51 M / 4.1 M) | 8 | lr × 3 | 16.8 / 141 | 0.059 ± 0.0003 |
| VCS, in-batch | 4 096 joint | 256 / 65 280 (0.51 M / 131 M) | 255 | lr × 3 | 8.7 / 2 445 | 0.060 ± 0.001 |
| JS, product / cyclic-8 / in-batch | as VCS | as VCS | 1 / 8 / 255 | lr × 3 | 4.8 / 145; 16.3 / 147; 8.4 / 2 451 | 0.089 / 0.052 / 0.052 |
| S-Kernel RFF | 4 096 + 4 096 | 256 / 256 (product) | 1 | m × bandwidth × lr = 27 | 2.9 / 619 | 0.185 ± 0.001 |
| Nyström (512 centres) | 4 096 + 4 096 | 256 / 256 (product) | 1 | lr × 3 | 3.5 / 331 | 0.533 ± 0.009 |
| RuLSIF-tanh | 4 096 + 4 096 (all moments) | closed form | — | m × bandwidth × λ = 36, + Nelder–Mead tanh wrap | 0.4 / 2 342 | 0.163 ± 0.002 |
| S-KDE | 4 096 joint (marginals from the same pairs) | closed form, O(N²) kernel sums | — | (h, b) = 16 | 0.1 / 421 | 0.602 ± 0.003 |

Every method selects on the same SELECT role (max(64, N / 4) = 1 024 units); EVAL (32 768 per side) is never used for a choice.

Reading (descriptive comparison of E-reg rows):
- **Size of the effect.**  Multi-negative constructions lower the neural posterior MSE by 30–60 % against product negatives for both losses: 38 %
  (VCS) and 41 % (JS) at this cell.
- **No extra data.**  They use *fewer* independent units than the product rows (the product sample goes unused), and 8× or 255× more negative
  pairs per anchor.
- **Cost:** 3.3× fit time (cyclic-8) or 17× peak memory (in-batch).
- **Attribution.**  This is a pair-construction effect shared by VCS and JS, not a property of the VCS loss.
- **Open.**  Whether the extra pairs behave like extra independent samples is the R4-T1 / T2 question.

### 2.4 Restated sentence (for the owner's text)
"On the same N joint and N product-of-marginals units, the neural VCS critic has a lower posterior MSE on S than kernel-density, RFF and Nyström
same-target estimators on every tested axis.  It is also lower than a RuLSIF-type ridge read-out on every axis but sample size, where the two are
on par (the ridge read-out is more accurate at N ≤ 1 024).  Building 8 cyclic or in-batch negatives per anchor from the joint pairs lowers the
neural error by a further 30–60 % for both VCS and matched JS, at 3× the fit time or 17× the memory and without additional data.  Matched JS is
equal or 5–25 % better than VCS on S in most cells."

## 3. (b) P108: the plug-in advantage is kept; the population bound is not a per-run guarantee
| statement | evidence | type |
|---|---|---|
| J(T) − S = −E_M(T − η)², so J(T) ≤ S for every bounded T (population) | Algebra.  E1(a): \|bias\| slope 2.000 in the critic error vs 0.965–0.980 for the plug-in; identity residual ≤ 2.1e-16 | T, verified numerically |
| The squared plug-in Ŝ_plug of the same fitted critic is the more accurate point readout | Smaller \|bias\| in 70 / 72 cells, smaller RMSE in 69 / 72.  Covers VCS and JS fit losses, 3 generators, N ≤ 16 384, ≤ 4 000 updates.  Ĵ's evaluation SE is ≈ 3.2× Ŝ_plug's (median). | E-reg |
| "Ĵ on EVAL is a lower readout in every run" | **Not claimed.**  Ĵ carries sampling error.  The negative seed-mean bias in 72 / 72 cells is an observation for these generators and budgets.  A certified lower bound needs a held-out concentration interval (second-application D's Hoeffding test needed 20–50× more samples). | — |
| No cell reaches \|J − S\| ≤ 0.02 within 4 000 updates (best ≈ 0.038–0.041 at C1, N 16 384) | E3 | G (the pre-set tolerance) |

Old sentences that state a per-run bound (files not edited).  Read them as population statements:
- **P47 prereg:** "J value is a valid lower bound on S".
- **P48 report §1:** "as a lower bound on S the raw closed form is still informative (≥ 0.81 …)".  The 0.81 values are finite-sample estimates of
  J(T) for the closed-form critic, each with its evaluation SE.
- **P109 prereg / report:** "measurement J is a lower readout over a finite function class" (J(T) ≤ S in population).

P108's own wording was withdrawn in its 2026-10-03 addendum.  P117's prereg already states the population-only reading.

## 4. (c) Protocol identity of every external baseline

### 4.1 SSL
| baseline | recipe actually run | tuned? | used as | sets read | label to use |
|---|---|---|---|---|---|
| SimCLR, P5 | NT-Xent τ 0.2, AdamW 1e-3, 2 views, 200 ep, B 256.  VCS's trunk, augmentation and probe. | no | 1× frozen control | selection | "SimCLR, P5 recipe" |
| SimCLR, P41 (the "88.32" of every later comparison) | The P5 recipe given VCS's knobs:<br>• 4 views (loss averaged over the 6 pairs);<br>• B 128 / 100 ep at 1×, 2 views × 800 ep at 4×, 4 views × 800 ep at 8×.<br>τ ∈ {0.1, 0.5} checked at 1× without gain. | Only the views / batch / schedule knobs that mattered for VCS (≈ 1.1× VCS's search on those knobs).  No lr, wd, τ-at-8×, projector or augmentation search. | equal-compute control 1×–8×; reference in P104–P145 | selection; official test once (P68) | "SimCLR recipe at VCS's view / epoch budget (P41)", not "tuned SimCLR" |
| SimCLR seeds 3–4 (P111) | as P41 8× | no | 5-seed comparison with A-P3 (SimCLR 88.31 ± 0.21) | selection | as P41 |
| SimCLR, strong augmentation (P89) | P41 8× + crop 0.08, jitter 0.8 / 0.8 / 0.8 / 0.2 | no | Δ_int reference (89.49) | selection | "SimCLR recipe, strong augmentation" |
| VICReg, P5 / P41 | (25, 25, 1).  P41 checked (25, 25, 0.1) and (10, 10, 1) at 1×; views / batch / schedule as SimCLR. | as SimCLR P41 | equal-compute control | selection; test once (P68) | "VICReg recipe at VCS's view / epoch budget (P41)" |
| SimCLR / VICReg 16× (P97) | P41 configuration at 16×, seed 0 | no | scaling check | selection | as P41 |
| SimCLR on CIFAR-100 (P91), ResNet-50 (P130), IN-100 (P154) | the CIFAR recipe (τ 0.2, AdamW 1e-3); IN-100 at 224 px, not an ImageNet recipe | no (P128 grid withdrawn by the owner, 2026-10-04) | dataset / width / scale checks | selection splits | "SimCLR recipe (untuned)".  CIFAR-100 8× linear is probe-limited (P98): read with kNN. |
| SimCLR in solo-learn (P75 / P76) | solo-learn's own CIFAR-10 SimCLR config (LARS, 1 000 ep).  VCS ported into the same framework: critic scalars LARS-excluded in P75; AdamW in S2b (P77 / P99). | external recipe | external-protocol check | official test once per run (owner exemption) | "SimCLR, solo-learn recipe" |

Losing rows, all kept:
- **P41:** both controls lead VCS's recipe at every budget (SimCLR on the test set: +3.96 at 1× → +1.48 at 8×).
- **P76 / P99:** gap 3.91 / 3.28; refuted by the pre-committed 1.5-point rule.
- **P89 / P111:** A-P3 Δ_int −1.95 under strong augmentation.
- **P92 / P98:** SimCLR ahead in kNN on CIFAR-100.
- **P130 add. 1:** kNN −1.59, inconclusive.
- **P154:** −3.08 linear at 200 epochs; the trigger fired and was not followed (owner order).

Search-cost disclosure (counts of `outputs/P*seed*` directories by name; approximate):
- ≈ 409 VCS-family runs (VCS and matched JS).
- 70 SimCLR runs, including 13 second-application batch-study runs and 4 withdrawn P128 partial runs that were never evaluated.
- 39 VICReg runs.

### 4.2 Estimator baselines (synthetic: P85, P86, P151)
| baseline | identity as run | read on | kept |
|---|---|---|---|
| matched JS | Our family's control: same JointMLP, pairs, lr grid and SELECT; Deep-InfoMax softplus form | S (posterior) and its own JS2 | JS ≥ VCS on S in 42 / 45 ladder cells (R4-01) |
| InfoNCE (in-batch, cyclic-8), NWJ, DV / MINE (EMA denominator), SMILE (τ 5) | Our implementations in `src/vcs_estim`: shared FIT / TUNE / SELECT / EVAL roles, JointMLP 256-256, lr ∈ {1e-4, 5e-4, 2e-3}, 2 000 updates.  Not the original papers' architectures or schedules. | Own MI target only; cross-target comparisons only through resolution and native error | See the list below the table |
| CS-K-native (classical kernel CS plug-in) | CS-IB median-heuristic bandwidth, pre-declared multiple 1 | its own Lebesgue truth | reads ≤ 13 % of its truth at d = 20 |
| same-target kernels (S-KDE, RFF, Nyström, RuLSIF-tanh) | §2 | S | §2 |

Kept for the MI-estimator row:
- In-batch InfoNCE resolves adjacent levels 1.3–6× more finely than VCS / JS.  Its EVAL bound is log 1 024 = 6.93 nats; P86 §5's "log 256" is
  corrected.
- Failure rows kept:
  - single-negative DV (1 981 non-finite steps);
  - product NWJ / DV divergence;
  - single-negative SMILE bias.
- The first SMILE rows were void (implementation fault).  They were re-run with the reference implementation (P85 addendum 2).

### 4.3 VL
| baseline | identity | where read | reproduction vs published | label |
|---|---|---|---|---|
| Grounding DINO Swin-T OGC | Zero-shot one-stage grounder, no RefCOCO training.  The Swin-B "cogcoor" model saw RefCOCO and is not used. | official val (reproduction); DEV (Table C: 56.75 / 53.86 / 54.41) | val 60.52 vs third-party 60.4 | "Clean zero-shot external row".  "+4.5 to +7.2 over Grounding DINO-T" compares a two-stage system (COCO detector trained without CAL / DEV / val / test images + RefCOCO-fine-tuned crop ranker) with a zero-shot grounder, not two objectives. |
| MDETR R101, RefCOCOg-fine-tuned | supervised on RefCOCOg train, which contains our DEV images | official val only | 81.94 vs paper 81.64 (81.21 with raw captions) | "supervised reference, val only"; no DEV row |
| MDETR EB3 | — | — | not reproducible under timm 1.0 (NaN outputs) | failed reproduction, kept |
| ReCLIP IPS-only / official | zero-shot; the official heuristics were tuned on val | val reproduction; DEV 70.78 / 73.18 | 65.26 vs 65.32; 68.20 vs 68.08 | DEV held out for it |
| FG-CLIP 2 Base (official region API) | released checkpoint | DEV 72.29 | 10 points below its paper on COCO box (64.8 vs 74.9), traced to the released checkpoint | row kept, gap stated |
| CLIP / SigLIP 2 / FG-CLIP zero-shot cosine | frozen public encoders | DEV | step-0 gates reproduce the raw values | "zero-shot" |
| softmax and matched JS (VL3, Table C, R4-VL1; R4-VL2 reuses R4-VL1's softmax and runs no new softmax) | matched controls on the shared recipe (not external) | DEV | — | "matched controls" |

## 5. (d) Supersession
| old item | replaced by | what the replacement changes | note |
|---|---|---|---|
| `CVPR_EVIDENCE_LEDGER_20260928.md` | this ledger | Numbers:<br>• §A unchanged.<br>Wording:<br>• §A2 / §E "tuned" / "equal tuning" → §4.1 labels.<br>Resolved rows:<br>• §C "pending / running" rows → P86 (+ §2 here), P74 / P106, P76 / P99, P88, P90, P92 / P98. | `reports/SUPERSEDED_BY_R4_02_EVIDENCE_LEDGER_20261011.md` |
| P86 report §1 / §9 (rule 1 read on cyclic-8) | §2 here | registered partner; RuLSIF-tanh on par on N | same note |
| `P151_E0/*` | `R4/P151_r4/` | common readouts | existing `P151_E0/SUPERSEDED_BY_R4.md` |
| `P86_p85_aggregate.{md,json}`, `P86_p85_aggregate_yaml.md` (SMILE rows) | `P86_smile_fix_aggregate.{md,json}` + P86 addendum | SMILE rows only | `reports/SUPERSEDED_BY_P86_smile_fix_aggregate.md` |
| `P27_negdetach_base_results_table_interim.{md,json}` | `P27_negdetach_base_results_table.{md,json}`, `P27_NEGDETACH_BASE_REPORT_20260926.md` | final rows | `reports/SUPERSEDED_BY_P27_negdetach_base_results_table.md` |
| `P36_A5_BASE_REPORT_interim_20260926.md`, `P36_a5_base_results_table_interim.{md,json}` | `P36_A5_BASE_REPORT_20260928.md`, `P36_a5_base_results_table.{md,json}` | 800-epoch rows completed | `reports/SUPERSEDED_BY_P36_A5_BASE_REPORT_20260928.md` |
| `P48_precheck_B1_attempt1_partial.{md,json}` | `P48_precheck_B1.{md,json}`, `P48_PRECHECK_B1_REPORT_20260927.md` | completed attempt | `reports/SUPERSEDED_BY_P48_precheck_B1.md` |
| `P68_FINAL_OFFICIAL_TEST_standin.{md,json}` (selection-split smoke of the pipeline) | `P68_FINAL_OFFICIAL_TEST.{md,json}`, `P68_FINAL_OFFICIAL_TEST_REPORT_20260928.md` | the stand-in is not a test result | `reports/SUPERSEDED_BY_P68_FINAL_OFFICIAL_TEST_REPORT_20260928.md` |
| `P115_DIAG_INTERIM_REPORT_20261003.md` | `P115_DIAG_FINAL_REPORT_20261004.md` | final | `reports/SUPERSEDED_BY_P115_DIAG_FINAL_REPORT_20261004.md` |
| `NEXT_ROUND_EXECUTION_PLAN_DRAFT_20260928.md` (plan) | package v2 + `V2_RECONCILIATION_20260928.md` | plan only | `reports/SUPERSEDED_BY_V2_RECONCILIATION_20260928.md` |
| `VL3/VL3_INTERIM_REFCOCOG_CLIP_REPORT_20261010.md`, `VL3/VL3_ADDENDUM1_LARGE_INTERIM_REPORT_20261010.md`, pooled readings of `VL3/VL3_B16_REPORT_20261010.md` | `VL3/VL3_COMPLETE_12CELL_REPORT_20261010.md` | three seeds in every cell; pooled +0.01 instead of the seed-rule view | `reports/VL3/SUPERSEDED_BY_VL3_COMPLETE_12CELL_REPORT_20261010.md` |
| `r3_intake/*` estimator statements | R4-01 (`R4/P151_r4/`) and §2 here | native-J columns; cyclic-8 reading | `reports/r3_intake/SUPERSEDED_BY_R4.md` |
| `R4/R4_VL_TRAINING_PROTOCOL_DRAFT_20261011.yaml`, `R4/R4_VL2_NU_PROTOCOL_DRAFT_20261011.yaml`, `R4/R4_SAMPLING_PROTOCOL_DRAFT_20261011.yaml` | their FROZEN versions (the sampling draft → `R4_T2_SAMPLING_PROTOCOL_FROZEN_20261011.yaml` + `R4_T1_STATISTICAL_OBJECT_20261011.md`, frozen during this restatement) | as run | `reports/R4/SUPERSEDED_BY_R4_FROZEN_PROTOCOLS.md` |
| tracked gate `*.partial.json` (15 files in `P109_GATE_1017246/`, `P116_GATE_1019934/`, `P116_GATE_1019942/`, `P116_GATE_1019945/`, `P118_GATE/`) | the final `.json` next to each (all present) | in-run checkpoints | `SUPERSEDED_BY_FINAL_JSON.md` in each folder |
| untracked `*.partial.json` (74 SSL evaluation checkpoints, never committed) | each unit's final JSON | not evidence | listed here only (the owner deletes) |
| `P128_*` | withdrawn | 4 partial run directories, never evaluated | existing `P128_WITHDRAWN_20261004.md` |

Not superseded:
- `P139_V7_IMAGENET_PIPELINE_DRAFT_20261005.md` (not run).
- P132, which ran under its DRAFT protocol: engineering only, no model trained.  That is an exposure fact, not a supersession.

## 6. (e) Exposure accounting

### 6.1 Official evaluation sets (access ledger)
| set | read by | count | status |
|---|---|---|---|
| CIFAR-10 official test, clean | P67 → P68: one pass over the 52 frozen units (VCS 1×–8× and the P41 controls, 3 seeds), 2026-09-28 | 1 pass | read |
| CIFAR-10 official test, clean | P75 / P76 S2 solo-learn: online linear Acc@1 at epoch 1 000, once per run (owner exemption 2026-09-28, "给S2也豁免") | 6 runs | read |
| CIFAR-10 official test, clean | P99 S2b continuation (same rule) | 3 runs | read |
| CIFAR-10 official test, clean | P77 / P78 S2b seed 0: stopped at epoch ≈ 272 by its probe gate | 0 | not read |
| CIFAR-10 official test images, corrupted only | P131 CIFAR-10-C-style regeneration (owner 2026-10-04, "全部提交"); clean images never scored; reporting only | 1 pass | corrupted versions read |
| CIFAR-10-C released files | not used | 0 | — |
| CIFAR-100 official test | Never opened.  The file is hashed for provenance only; `final_official_test` refuses non-CIFAR-10 runs. | 0 | unread |
| ImageNet val | not used to evaluate any of our models (P154 pilot, P132 preflight).  **Correction 2026-10-12:** 10 000 images (the first 10 per class by file name) were read as implementation gates for public encoders — VL1-12 addendum 2 (SigLIP 2 B zero-shot, job 1030280, `reports/VL1/vl1_12_siglip2_check.json`) and VL1-15 (SigLIP 2 L, `vl1_12_siglip2_check_large.json`); no selection used them | 0 for our models | partly read (10 per class, public encoders only) |
| RefCOCO / RefCOCO+ / RefCOCOg official test splits | Never evaluated.  Image ids were used only to exclude those images from detector training. | 0 | unread |
| RefCOCO / RefCOCO+ / RefCOCOg official val | external-baseline reproductions:<br>• Grounding DINO-T;<br>• MDETR R101 on 3 datasets + raw-caption variant;<br>• MDETR EB3 attempt;<br>• ReCLIP IPS / official.<br>Also the VL1-21 detector's val proposals and box recall. | — | read for external rows and detector recall only; **none of our rankers or critics was scored on val** |

Consequences:
- **No unread CIFAR-10 confirmation set remains.**  The clean test was read in three protocols, and the corrupted pass reused its images.
- **No v3–v7 recipe was ever confirmed on held-out data.**  This covers G2, A-P3, JS-AP3 and others; P68 predates them.
- **Unread sets:** the CIFAR-100 test, the VL test splits, and the VL val split for our models.  ImageNet val is not unread: 10 images per class
  were read for public-encoder gates (correction 2026-10-12; disclose if ImageNet val is ever used for confirmation).  Any of these would need a
  one-shot protocol approved by the owner.

### 6.2 Development splits (repeated use)
| split | role | reads (record counts from `outputs/`) | consequence |
|---|---|---|---|
| CIFAR-10 selection (5 000 images of the official training set) | every SSL decision: selection, seeding, rules | ≈ 400 run directories.  `outputs/P*seed*` holds 518 SSL run directories; 114 are CIFAR-100, 3 are IN-100. | every SSL comparison after P68 is a development-split result |
| CIFAR-100 selection (5 000 of the training set) | every CIFAR-100 decision | 114 run directories | same |
| VL DEV (image-level role carved from the UMD train split) | every VL reading | • VL1-10…15 and VL2 frozen-feature fits: ≈ 695 record files<br>• VL3 run records: 211 (107 × 10 epochs, 104 × 20 epochs; each read at step 0 and at its selected epoch)<br>• add. 5 strata: 111<br>• add. 6 transfer: 240<br>• Table C: 108 checkpoints<br>• R4-VL0: 18 rescorings<br>• R4-VL1: 18 runs + 18 diagnostic rescorings<br>• R4-VL2: 12 runs, running | exploratory status for every VL claim; a new DEV split would not erase these selections |
| VL CAL | selection only: epoch, critic, K for top-K | — | — |
| synthetic estimator roles (FIT / TUNE / SELECT / EVAL / TRUTH) | fresh seeded streams per cell; EVAL never used for a choice | — | no reuse issue |

### 6.3 Data-driven seed additions
| unit | screen | seeds added after the screen | rule frozen before the added seeds were read? |
|---|---|---|---|
| P29 / P30, P35 / P36, P37 / P38 (VCS recipe search) | seed 0 on the selection split | seeds 1–2 for the base / final recipe (owner rule: single-seed exploration, seeds only for the base / final recipe) | yes |
| P41 | 12 single-seed units | seeds 1–2 for the best cell per compute point | yes |
| P95 stage B | stage-A seed 0 per family | seeds 1–2 for families ≥ 85.42; JS always | yes |
| P104 add. 1 / add. 3 | seed-0 batch (G2 88.66) | G2 s1–2 + G2F controls; U2 s1–2 + U2F | yes |
| P107 add. 1 / add. 2 | seed-0 screen (A-P3 89.06, A-P2 88.98) | • A-P3 / A-P2 s1–2 + F controls<br>• then, after the layer-1 rule held: A-P3 s3–4, strong ×3, CIFAR-100 ×3 | yes |
| P111 | G2 layer-1 rule ("extra" branch) | G2 s3–4, SimCLR s3–4, strong augmentation, CIFAR-100 | yes |
| P112 add. 1 | κ screen | strong-augmentation seeds 1–2 | yes |
| P120 add. 1 | inconclusive 3-seed JS contrast (+0.80) | seeds 3–4 (continuation *because* the result was inconclusive) | yes |
| P129 add. 1–4 | JS grid, seed 0 | seeds 1–2 of the selected cells (+ one extra CIFAR-100 cell); lr check | yes |
| P130 add. 1 | ResNet-50 seed 0 | seeds 1–2, all three methods | yes |
| P135 add. 1 | lr screen | JS lr 2e-3 seeds 1–2 | yes |
| P138 add. 1 / add. 2 | seed 0 | seeds 1–2, then n = 5 | yes |
| P145 add. 1–3 | STRESS trigger (\|dV − dm\| ≥ 1.0 on CIFAR-100) | seeds 1–2 × 3 methods; add. 2's trigger → owner chose seeds → add. 3 | yes |
| P154 | trigger fired (\|Δ\| ≥ 1.0) | none (owner stop, 2026-10-09) | — |
| VL3 | seed-0 rule: seeds only where VCS was within 0.5 of the best | 7 cells seeded, then add. 3 completed all 12 cells to 3 seeds | yes.  The rule's pooled bias was +0.29 (vs +0.01 on all cells), disclosed. |
| VL3 add. 2 → add. 4 → add. 7 | 20 epochs, 1 seed | 3 seeds for B/16 (add. 4) and Large (add. 7) | yes |
| VL1-12 add. 2 / 3, VL1-15 add. 1 | after the CLIP results | further backbones (SigLIP 2, FG-CLIP v1, Large / SO400M) | yes (frozen addenda) |

Fixed in advance, with no data-driven additions: P85 (3 seeds), P108 (5 / 3), P134 (3), P151 / P152, R4-VL1 (3) and R4-VL2 (3).

Consequence:
- **How the seeds were added.**  The added seeds ran under rules frozen before they were read, but the decision to add them followed the screen.
- **Effect on the means.**  Every multi-seed mean includes the screening seed.  It is optimistic in the direction of the screen (winner's curse).
- **The one measured case** is VL3: +0.29 → +0.01.
- **A-P3.**  The screening seed (89.06) is close to the 5-seed mean (89.02), so the bias is small there, but not zero by construction.

### 6.4 Pretraining and detector exposure
- **Public encoders.**  CLIP B/16 and L/14@336, SigLIP 2 B/16, L/16 and SO400M, FG-CLIP v1 / 2 were pretrained on web-scale data that may contain
  COCO images (every RefCOCO image is a COCO train2014 image).  Not audited.
- **The VL1-21 detector** was trained on COCO train2017 minus every CAL / DEV / val / test image of the three datasets (14 781 excluded).  It
  includes the FIT images, so FIT proposals are slightly cleaner: recall 98.0–98.9 % vs DEV 97.2–98.7 %.

## 7. Claim ledger
`R4_02_claim_ledger.csv`: 37 rows in the template's columns.  By type:
- algebra / theory: 3
- standard instantiation: 1
- E-reg: 19
- E-exp: 10
- hypothesis: 2
- practical threshold: 1
- protocol label: 1

By status:
- verified: 2
- supported: 10
- supported with scope: 10
- mixed: 4
- not supported / refuted / threshold not met: 6
- corrected: 2
- untested / running: 3

Each row says what may be concluded and what is not implied.  The `owner_decision` column is left for the owner.

## 8. Open questions for the owner
1. **Kernel comparison in the paper.**  Primary panel = the registered product partner (§2.2); the multi-negative effect = a separate
   benefit-and-cost panel (§2.3)?  This is my recommendation.  Alternatively, keep the cyclic-8 rows as primary with a disclosure that they
   are not the registered partner.
2. **SimCLR label.**  Adopt "SimCLR recipe at VCS's view / epoch budget (P41)" everywhere the manuscript says "tuned SimCLR"?
3. **Confirmation data.**  Which unread set, if any, should take a one-shot confirmation of the v3–v7 SSL recipes or the VL results, and when?
   Candidates: CIFAR-100 test, ImageNet val, the VL test splits.  The CIFAR-10 test is spent.
4. **Pretraining exposure.**  Accept the unknown COCO exposure of the public encoders as a stated caveat, or audit it?
5. **Cleanup.**  The 74 untracked `*.partial.json` files and the `outputs/P128_*` partial runs can go in the next cleanup; the owner runs
   deletions.
