# Pre-registration DRAFT — pre-check A, wave 2 (P57): A-S1 calibration-vs-dependence curve and A-T mismatch detection without a target calibration set — FROZEN 2026-09-27T14:48:57Z before GPU compute (CPU smoke 1011085 disclosed in the addendum)

Status: DRAFT written by the implementing fork on 2026-09-27; to be frozen (renamed `*_FROZEN_*`) by the main session before the GPU run.
Parent: `P49_PRECHECK_A_PREREG_FROZEN_20260927.md` (+ addenda 1–5) and its result `P50_PRECHECK_A_REPORT_20260927.md` (holds conditionally: native
(1+T)/2 as calibrated as post-hoc Platt only in the mid-dependence regime; no threshold-transfer advantage — monotone critic).  Plan row:
`SECOND_APP_WAVE2_PLAN_20260927.md`, row A.  Code: `scripts/precheck_a_wave2.py` (imports the P50 helpers unchanged), `slurm/precheck_a_wave2_features.sbatch`,
`slurm/precheck_a_wave2.sbatch`.

## Questions
- **A-S1.** P50 gave two points (J 0.92: native worse than Platt; J 0.57 / 0.16: native ≈ Platt).  Is "native calibration ≈ Platt-on-cosine" a
  property of a *regime* of the dependence strength, measured as a curve rather than two points, and does it hold on a second shift?
- **A-T (task).** In the use the family row A promises — mismatch detection / open-set verification *without a target calibration set* — does the
  native rule "accept if (1+T)/2 ≥ 0.8" keep its nominal precision under a distribution shift and under hard mismatches, compared with a cosine
  threshold calibrated on the source (which needs a labelled source calibration set) and with an oracle calibrated on the target?

## Unchanged from P49/P50 (frozen protocol)
Frozen open CLIP ViT-B/32 `laion2b_s34b_b79k` towers (provenance file under `models/open_clip/`); identity-initialised 512×512 linear adapters per
tower + L2; methods vcs (T = tanh(a·cos + b), a₀ = 5, K = 8 independent-pool negatives, P and Q averaged separately), infonce (learnable τ),
logistic (SigLIP-style, learnable a, b); grid lr ∈ {1e-3, 3e-4, 1e-4} × epochs ∈ {5, 15, 40} per method, selected on SRC-CAL by the method's own
loss (equal budgets, 9 configurations each; budgets tabled); 3 seeds of the selected configuration; captions 0–3 train, caption 4 held out;
balanced joint/product evaluation pairs (product = caption 4 of a random other image, fixed derangement); ECE with 15 equal-mass bins; Platt =
1-D logistic fit on SRC-CAL; no official test set; identity = COCO image id; a caption is only ever paired through its own image's topic.

## Data
- **Shift 1 (animal)**: the P49 splits and cached features (`outputs/P49_precheck_A/`, sha256 of the id lists in its `splits.json`: SRC-FIT 587163f0…,
  SRC-CAL 00e25ade…, SRC-EVAL fa12a552…, TGT-EVAL cc5b8408…).
- **Shift 2 (indoor → outdoor)**, built by `precheck_a_wave2.py features` from the local COCO index (train2017, ≥ 5 captions): source pool = images
  whose supercategory set is non-empty and ⊆ {furniture, appliance, indoor, kitchen, electronic, food} (**23 327** images); target pool = non-empty
  and ⊆ {vehicle, outdoor, sports} (**10 325**).  Images containing `person`, `animal` or `accessory` are in neither pool, so this is a scene-type
  shift with people removed.  Sizes: the source pool cannot supply 30 000, so **SRC-FIT 13 327 (= pool − 10 000)**, SRC-CAL 5 000, SRC-EVAL 5 000,
  TGT-EVAL 5 000; seed 20260927; identity-disjoint; id lists and sha256 written to `outputs/P57_precheck_A_wave2/features_indoor_outdoor/splits.json`
  (filled in by the feature job; the hashes are appended to the frozen prereg before the adapter run).  Overlap with the P49 id lists is allowed
  (a different unit; splits are disjoint *within* each unit).  Every image in both pools has a same-topic partner (checked on the index).
- **Pairing difficulties** (the dependence knob; partner drawn per training step, one fixed evaluation partner per image, seed 20260927):
  exact (own caption) · topic (caption of another image with the same supercategory set — P50 setting 2) · coarse (another image sharing ≥ 1
  supercategory but with a *different* set; 64 pre-drawn candidates per image) · random (any other image; 64 pre-drawn candidates).
  Product pairs are always a random image's caption.  Expected J ordering: exact > topic > coarse > random ≈ 0.

## A-S1 design
For each shift × pairing × method: grid → selected configuration → 3 seeds; on SRC-EVAL and TGT-EVAL record held-out J (vcs), native ECE
((1+T)/2 for vcs; σ(a·cos + b) for logistic; none for infonce), Platt-on-cosine ECE (Platt fitted on the method's own SRC-CAL cosine), Brier,
R@1 (secondary).  The **curve** is the set of points (J of the vcs model on that split, native ECE − Platt ECE of the *same* vcs embedding) over
(shift, pairing, split); the competitors' Platt-on-cosine ECE is tabled alongside at each point.

**Reading (pre-committed, copied from the plan):** native ECE vs Platt-cosine ECE on the target as a function of held-out J.  "Native calibration
holds in a regime" if native ≤ Platt + 0.01 in every setting with J ∈ [0.1, 0.7] on both shifts; "does not hold" if any such setting has native
> Platt + 0.02; else conditional (regime narrower than [0.1, 0.7]).  A "setting" is a (shift, pairing, split) point whose vcs held-out J lies in
[0.1, 0.7]; source-side points are read too and reported separately.  P50's two points are expected to reappear (exact ≈ 0.9 → native worse;
topic ≈ 0.6 / 0.16 → native ≈ Platt).

## A-T design (task-level pre-check; adapters = the A-S1 *topic* models of each shift, 3 seeds; raw CLIP as reference)
- **Pairs** on SRC-EVAL and TGT-EVAL (and SRC-CAL for fitting source rules): matched = (image, caption 4 of its fixed same-topic partner) — the
  positive relation the adapters were trained on; **hard mismatch** = caption 4 of the image's fixed *coarse* partner (shares ≥ 1 supercategory,
  different set); **easy mismatch** = caption 4 of a random image.  50 / 50 balanced; images lacking a partner are dropped; evaluation only —
  no hard-negative *training* selection (brief §5 prohibition respected).  Note: under topic-pairing training a "topic-matched wrong caption" *is*
  the positive class, so the plan's "topic-matched wrong caption" is implemented as the coarse-topic caption (the hardest mismatch that is still
  a mismatch under the trained relation).
- **Rules** (nominal precision 0.8; all fixed without target labels except the oracle): native accept if (1+T)/2 ≥ 0.8 (vcs) / σ ≥ 0.8 (logistic;
  infonce has no native probability); source-Platt-on-cosine (Platt fitted on SRC-CAL task pairs of the same kind, accept if ≥ 0.8) for every
  method's embedding and for raw CLIP; source-Platt on the native score (vcs, logistic; diagnostic); **oracle** target Platt on the cosine,
  cross-fitted on the target task pairs (upper reference, marked *not deployable*).
- **Metrics** per rule: realized precision among accepted vs nominal 0.8 (primary: |precision − 0.8|), acceptance rate, mean p̂ among accepted,
  FNR / FPR / balanced error at the rule, AUROC of the underlying score (ranking; identical for T and cosine of the same embedding by
  monotonicity — reported to make that visible), ECE and Brier on the task pairs.

**Reading (pre-committed, copied from the plan):** primary: |realized precision − 0.8| on target; native holds if within 0.05 and not worse than
source-Platt-cosine by more than 0.02; does not hold if worse by ≥ 0.05.  Secondary: balanced error, AUROC (ranking; expected equal for monotone
critics), ECE.  The primary reading is taken on the **hard** mismatch pairs of TGT-EVAL for both shifts (both must pass for "holds"; one → conditional);
the easy pairs and SRC-EVAL are reported as context.  If the native rule accepts fewer than 5 % of pairs its precision is reported but marked
"degenerate acceptance" and the reading falls to conditional.

## Not claimed / limits
Only CLIP ViT-B/32 features; COCO captions (5 per image); one seed of the towers (frozen); adapters are linear; topic granularity = COCO
supercategory sets; the shifts are constructed within COCO (no external target — Flickr8k/30k would need a licence check); nothing about retrieval
quality; no task is selected by this unit.

## Compute and provenance
Features: 1 GPU job (~10 min).  Main run: 1 GPU job — 2 shifts × 4 pairings × 3 methods × (9 grid + 3 seeds) = 288 adapter trainings ≈ 30–40 min
(P50 setting 2: 36 trainings in 160 s).  Results-only commit of `P58_precheck_A_wave2.{json,md}` first, then the report.  Smoke: CPU job (see
addendum below).

## Addendum (smoke probe, disclosed) — filled in after the CPU smoke job

**Smoke probe (CPU job 1011085, `slurm_logs/precheck_a_wave2_pa2_smoke_1011085.out`, 2 000 images per split, 1 epoch, 1 seed, pairings exact + topic, shift = animal; ~13 s compute).**
End-to-end run produced `P58_precheck_A_wave2_smoke.{json,md}`.  Partners: coarse 2000/2000 in every split, topic 1909–1942/2000.  Held-out J of the 1-epoch VCS adapter: exact 0.68 / 0.62, topic 0.32 / 0.09 (source / target) — the pairing knob moves J as intended (P50's converged values were 0.92 and 0.57 / 0.16).
1-epoch native ECE (not a result: single probe epoch): exact 0.197 / 0.186 vs cosine+Platt 0.011 / 0.115; topic 0.105 / 0.043 vs 0.052 / 0.105.
A-T rows from the probe (TGT-EVAL, hard mismatches; numbers only show the pipeline works):
- hard/TGT-EVAL native_(1+T)/2: accept 0.092 prec 0.846 |dev| 0.046 AUROC 0.730
- hard/TGT-EVAL cosine+Platt(SRC-CAL): accept 0.135 prec 0.833 |dev| 0.033 AUROC 0.730
- hard/TGT-EVAL cosine+Platt(TGT oracle, cross-fitted): accept 0.087 prec 0.855 |dev| 0.055 AUROC 0.730
- raw CLIP hard/TGT-EVAL cosine+Platt(SRC-CAL): accept 0.076 prec 0.855 |dev| 0.055 AUROC 0.709
- easy mismatches, VCS native: accept 0.109 prec 0.715 |dev| 0.085 AUROC 0.685
No design change follows from the probe.  Full run: `features` job (indoor → outdoor) then the main job with all four pairings, 9-point grid, 3 seeds.

**Freezing note (main session).** Primary nominal precision stays 0.8.  Because the probe showed acceptance of only 7–15 % at 0.8 on the topic
relation, a *secondary* run with nominal 0.7 (`--nominal 0.7`, separate job, otherwise identical) is added now; it is read with the same rules
and reported as secondary — it cannot rescue a failed primary, only qualify a "degenerate acceptance" outcome.  The indoor → outdoor split
hashes are appended below when the features job lands (before the main job starts, which depends on it).

**Indoor → outdoor split hashes (appended when the features job 1011090 landed).** rule: source = train2017 images whose supercategory set is non-empty and a subset of {furniture, appliance, indoor, kitchen, electronic, food}; target = non-empty subset of {vehicle, outdoor, sports}; >= 5 captions; images with 'person' or 'animal' or 'accessory' are in neither pool; pool sizes: {'source': 23327, 'target': 10325}; sizes: {'SRC-FIT': 13327, 'SRC-CAL': 5000, 'SRC-EVAL': 5000, 'TGT-EVAL': 5000}; sha256: SRC-FIT f26372c6901b8428…, SRC-CAL d1ce7582059c5e18…, SRC-EVAL d1397b2b016ca080…, TGT-EVAL 5a194efe61c99cad….
