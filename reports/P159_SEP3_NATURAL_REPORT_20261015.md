# P159 (SEP-3) report — the quadratic vs logistic rule in parametric critic classes under contamination, on frozen natural features (2026-10-15)

Protocol `P159_SEP3_NATURAL_PREREG_FROZEN_20261015.md` (frozen 2026-10-15T10:16:18Z before any formal cell; outcome rule and readouts fixed there).
Jobs 1041369 / 1041370 (RTX PRO 6000, 4 seeds each, exit 0), aggregate 1041399 (CPU).  Cells: 320 (40 per seed × seeds 300–307, missing 0) in
`outputs/P159_sep3`.  Results-only commit e22de56: `P159_SEP3_results/p159_aggregate.{md,json}`; every number below is from that JSON (tables by
`scripts/p159_aggregate.py`, counts by the summary snippet in this commit's message).  Sources: COCO-2017 image–caption pairs through frozen CLIP ViT-B/32
(P49 roles, image-disjoint) and A-P3 CIFAR-10 FIT-split (h_clean, h_blur_s1) pairs (P150 cache; a nuisance view, not an augmentation pair — disclosed in the
protocol).  Autonomous unit (owner's idle-GPU rule, 2026-10-11); no official test split; no truth — all readouts on a clean, image-disjoint EVAL set.

## 1. Outcome by the frozen rule: **FIT-ONLY**
| test (Holm over 3; paired t over 8 seeds) | effect | p Holm | cells |
|---|---|---|---|
| Q1 — dJ_rel at ε > 0 (both sources, classes, types, ε, N) | **+0.0562 [+0.0548, +0.0576]** | 1.3e-11 | 32 |
| Q2 — dJ_rel at ε = 0 | **+0.0020 [+0.0014, +0.0026]** | 7.6e-5 | 8 |
| Q3 — dA_rel at ε > 0 | +0.0485 [+0.0455, +0.0515] | 4.2e-9 | 32 |

Q1 > 0 and Q2 > 0 under Holm → the FIT-ONLY branch.  Its permitted sentence: **"On frozen natural features (CLIP image–caption pairs; A-P3 CIFAR view
pairs), parametric critics fitted on contaminated pairs reach a higher clean-set squared score under the quadratic rule than under matched logistic; the
quadratic rule also leads on clean fitting in these classes (a property of the fit, not of contamination)."**  As in P157 this is reported as a property of the
scoring rule's fit (the Brier-score L² projection), not of the CS divergence, not as robustness, and with no SSL / VL implication.

**Honest nuance the branch does not capture (pre-registered secondary numbers):** the clean-fitting lead is +0.2 % of J — 28× smaller than the +5.6 % under
contamination — and it is confined to the CIFAR source: on COCO the four clean cells are n.s. (dJ_rel +0.0003 / +0.0002 / +0.0002 / −0.0005), while on
CIFAR three of four clean cells are significant (quad-3 +0.0070 / +0.0060, cos-2 at N 8 192 +0.0017).  Read per source, COCO shows the SUPPORTED pattern
(contamination-specific, no clean difference) and CIFAR the FIT-ONLY pattern; the pooled rule returns FIT-ONLY, which is what is written.

## 2. Secondary readouts (pre-registered)
- **Per-cell dJ_rel (Holm over 40):** VCS higher in **32 / 32** contaminated cells (COCO +0.026 to +0.059; CIFAR +0.008 to +0.173); 3 / 8 clean cells
  significant (all CIFAR, listed above).  Largest effects: CIFAR quad-3 under false negatives (+0.113 to +0.173); smallest: CIFAR cos-2 under independent
  pairs (+0.008 to +0.012).
- **Degradation ratios J(ε)/J(0) (R4-E0 reading, beside the absolute effect):** the VCS ratio exceeds the JS ratio in **32 / 32** contaminated cells with CIs
  excluding 0 (e.g. COCO cos-2 independent ε 0.1: JS 0.946 vs VCS 0.995; CIFAR quad-3 false negatives ε 0.3: 0.740 vs 0.863).  Both readings agree here —
  the quadratic fit loses less of its clean score and ends higher.
- **AUROC (Q3) — structurally uninformative for cos-2:** a·cos + b is monotone in the cosine for every a > 0, so the two rules rank pairs identically and all 20
  cos-2 cells have dA_rel = 0 exactly.  Q3 is carried by the CIFAR quad-3 cells (dA_rel +0.27 to +0.38 under false negatives, +0.03 to +0.09 under independent
  pairs; 7 / 32 contaminated cells significant); on COCO quad-3 the AUROC differences are n.s. (AUROC ≥ 0.996 for both rules).  Q3's pooled significance should
  not be read as a general ranking advantage.
- **Calibration (ECE of (1 + T)/2):** VCS lower in **30 / 32** contaminated cells (Holm), most at ε 0.1 (COCO independent: JS 0.071–0.074 vs VCS 0.013–0.020;
  CIFAR false negatives: 0.08–0.09 vs 0.02–0.03); at ε 0.3 the gap narrows to 0.01–0.03 and vanishes for CIFAR cos-2 independent pairs.  Clean cells n.s. on
  COCO; CIFAR quad-3 clean +0.007 / +0.006.
- **Common-J selection (sensitivity):** outcome also FIT-ONLY (Q1 +0.028, Q2 +0.0014, Q3 +0.043, all Holm ≤ 2e-8); direction unchanged, size about half.
- **By factor (ε > 0, descriptive):** CIFAR +0.069 > COCO +0.044; quad-3 +0.074 > cos-2 +0.039; false negatives +0.066 > independent +0.046; ε 0.3 +0.063 >
  ε 0.1 +0.050 (the opposite ε ordering to P157's synthetic cells, where the gap narrowed with ε).

## 3. QC and limitations
- **lr-grid top edge (pre-registered limitation rule, > 50 % of selections):** quad-3 selects the top lr 1.0 in 65 / 80 (CIFAR, both rules), 60 / 80 (COCO JS) and
  51 / 80 (COCO VCS) selections → **limitation flagged for quad-3 on both sources**; no extension (as frozen).  cos-2 is inside the grid (10–27 / 80).  Not
  converged: 3–18 of 80 per source × class × arm.  The quad-3 effects are therefore read as "at the grid's edge for both rules"; the cos-2 effects are not
  affected.
- The CIFAR pair is a nuisance view of the same image through a frozen encoder (no two-augmentation cache exists); the view was changed after the first gate
  (blur_s0.5 saturated) and before any formal run — disclosed in the protocol.  COCO uses one caption per image per seed.
- 8 seeds re-draw FIT / SELECT / EVAL by image; EVAL is never contaminated; nothing on EVAL selects anything.  No truth exists on these features: "better" means
  a higher clean-set squared score, not a smaller distance to a known posterior.

## 4. Not claimed (protocol)
Nothing about SSL training, VL fine-tuning or downstream accuracy; nothing about expressive critics; no robustness statement; no claim beyond the two sources
and two contamination constructions; no novelty (Buja, Stuetzle & Shen 2005; the L² projection of the Brier score; the noisy-label literature of P156 §6).

## 5. Place in the evidence
P157 (synthetic, with truth) and P159 (frozen natural features, truth-free) now agree on direction: in parametric critic classes with contaminated fitting
data, the quadratic rule gives the better clean-set fit, degrades less and is better calibrated; at ε = 0 the two rules coincide on COCO and differ by 0.2 % on
CIFAR.  The scope stays "parametric critics, contaminated pairs" — the expressive-class and natural-data accuracy picture of the project map §2 (VCS ≈ JS or
JS ahead) is unchanged.
