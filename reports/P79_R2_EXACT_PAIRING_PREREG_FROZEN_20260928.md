# Pre-registration DRAFT — P79: R2 with exact pairing + cross-fitted R3, frozen CLIP + identity adapters (2026-09-28) — FROZEN 2026-09-28T15:50:28Z before GPU compute

Status: DRAFT (fork); the main session freezes it (rename to `*_FROZEN_*`) before the GPU run.  Results will be P80.
**Disclosure:** this unit repairs P72 §1 and was designed **after** the P72 results were seen.  P72 trained on topic pairing and evaluated
own-caption retrieval, so R@1 rose with the mismatch fraction for every method (raw CLIP 0.37 → 0.03–0.08 after topic training).  Here the
training relation and the evaluation relation coincide.  The R3 cross-fitted score follows the estimator package's §12 revision
("训练集错配 AUROC → 增加独立或交叉拟合评分"); the in-training native AUROC is kept alongside it.
Code: `scripts/robust_r2r3.py --pairing exact --crossfit` (the default `--pairing topic` path is unchanged; regression smoke below).

## Question
With the own caption as the clean positive, does VCS lose less clean own-caption retrieval than InfoNCE, pairwise logistic and JS (balanced
logistic on the same positive / K = 8 negative construction) as a known fraction of training pairs is mismatched — and does its native score
identify the mismatched training pairs better, in-training and cross-fitted?

## Data and splits (unchanged from P71; no official test set)
- animal: `outputs/P49_precheck_A/` (SRC-FIT 20 000, SRC-CAL 5 000, SRC-EVAL 5 000, TGT-EVAL 5 000).
- indoor → outdoor: `outputs/P57_precheck_A_wave2/features_indoor_outdoor/` (SRC-FIT 13 327, CAL / EVAL / TGT 5 000).
- Frozen CLIP ViT-B/32 features, identity-initialised 512 × 512 adapters (P50).
- **Exact pairing:** clean positive of image i = one of its own captions 0–3 (uniform per step); caption 4 held out for evaluation; SRC-CAL
  selection loss computed on exact pairs.

## Mismatch injection
m ∈ {0, 0.1, 0.2, 0.4, 0.6}; same seeds and generator as P71 (seed = [20260928, round(1000 m), sha(shift)]), so the injected index sets are
identical to P72's (sha256 recorded); an injected image's only positive is one random other image's captions.  CAL and evaluation splits clean.

## Methods, equal budget (as P71)
vcs (J, tanh(a·cos + b), K = 8 pool negatives), infonce (in-batch, learned τ), logistic (in-batch pairwise; +log K column rank-equivalent),
js (balanced logistic, Deep-InfoMax form, same positive / K = 8 construction).  Grid lr ∈ {1e-3, 3e-4, 1e-4} × epochs ∈ {5, 15, 40}, selected on
clean exact-pair SRC-CAL by each method's own loss; 3 seeds (0, 1, 2) of the selection; AdamW wd 1e-4, batch 256, cosine schedule.

## Metrics
- **R2:** image → text R@1 / R@5 of the own caption 4 on clean SRC-EVAL and TGT-EVAL; loss(m) = R@1(0) − R@1(m) per method (3-seed means);
  raw-CLIP row (no training) as reference.
- **R3** (m ∈ {0.2, 0.4}): native score (vcs (1+T)/2 ≡ T in rank; infonce cos/τ; logistic logit; js a·cos + b) of the training pairing
  (own image, or the injected partner; caption 0); AUROC clean vs injected.
  (a) *in-training*: the seed's model trained on all of SRC-FIT scores its own training pairs (as P71);
  (b) *cross-fitted* (primary for R3): SRC-FIT split once into two identity-disjoint halves (seed 20260929 + sha(shift)); the selected
  configuration is trained on one half with that half's injected pairs (K-pool negatives from that half's captions), scores the other half; swap;
  AUROC over the pooled out-of-fold scores; per seed.  Raw CLIP cosine AUROC as floor.

## Pre-committed reading (P71's loosened margins, fixed before compute)
- **R2 holds** if VCS's R@1 loss from m = 0 to 0.4 is smaller than each competitor's by ≥ 0.5 point on both targets (SRC-EVAL and TGT-EVAL) on
  both shifts; **refuted** if the four curves coincide within 0.5 point at m = 0.4 or VCS's loss is larger than any competitor's on either
  shift; **conditional** otherwise (one shift / one target; or beats infonce / logistic but not js → the effect belongs to the balanced-mixture
  family, written so).
- **R3 holds** if VCS's cross-fitted AUROC ≥ each competitor's + 0.01 at both m = 0.2 and 0.4 on both shifts; **refuted** if VCS is within 0.01
  of the best competitor or below it at any cell; conditional otherwise.  The in-training AUROC is read with the same rule and reported
  separately; if the two disagree, the cross-fitted verdict is the R3 verdict and the disagreement is reported.
- **R3 ceiling clause** (added after the smoke, before any full-scale compute): under exact pairing a random wrong caption is easy to spot —
  raw CLIP already reaches AUROC 0.995 / 0.992 in the smoke.  If the full-run raw-CLIP AUROC is ≥ 0.98 at a cell, that cell is **saturated**:
  R3 is reported but not read (neither holds nor refuted), and the R3 verdict is "not informative under exact pairing" if all cells are
  saturated.  An informative R3 needs structured (topic-matched) mismatches, which remain a separate follow-up unit.
- A method whose m = 0 R@1 is below raw CLIP on SRC-EVAL is flagged ("adapter training loses instance alignment") and its loss curve is not
  read as robustness.
- QC: injected counts = round(m · n_fit) and index sha256 equal to P72's; at least one selection strictly inside the grid (else noted).
- Not claimed: other towers, non-linear adapters, structured (topic-matched) mismatches, training from scratch.

## Smoke probe (CPU)
CPU job 1013074 (`slurm_logs/robust_r2r3_r2_exact_smoke_1013074.out`, exit 0, 24 s; 2 000 images per split, lr 1e-3 × 1 epoch, seed 0,
m ∈ {0, 0.4}, all four methods, both shifts, `--pairing exact --crossfit`) — code-path evidence only, not results:
- Exact pairing, injection, raw-CLIP R2 row and 2-fold cross-fitting run end to end on both shifts.
- Raw CLIP (2 000-image smoke splits): SRC R@1 0.492 / 0.372, TGT 0.454 / 0.299 (animal / indoor); R3 AUROC 0.995 / 0.992 → ceiling clause above.
- animal m = 0.4: R3 in-training / cross-fitted vcs 0.991 / 0.994, infonce 0.993 / 0.995, logistic 0.981 / 0.990, js 0.981 / 0.990.
- After one epoch VCS's R@1 at m = 0 is already well below raw CLIP (0.19 vs 0.49 animal SRC), the others near it (infonce 0.46) — noted,
  not read (1 epoch, 2 000 images).
- Regression: the default `--pairing topic` path reproduces the P71 smoke 1013017 line for line (CPU job 1013075, all 16 R@1 / AUROC values identical).

## Compute
P72's 240 trainings per shift took ≈ 17 min on one GPU for both shifts; cross-fitting adds 2 half-size trainings per (method, seed) at 2 m
values (≈ 24 extra full-size-equivalent trainings per shift).  Expected ≈ 25–40 min, one GPU job; wall limit 4 h.

Submit (after freezing; from `ssl_pilot/`, env vars through the environment, not `--export`):
```
SHIFT_DIRS="animal=/home/infres/yinwang/CS_QMI/outputs/P49_precheck_A,indoor_outdoor=/home/infres/yinwang/CS_QMI/outputs/P57_precheck_A_wave2/features_indoor_outdoor" \
REPORT=reports/P80_r2_exact EXTRA_ARGS="--pairing exact --crossfit" \
sbatch --job-name=r2_exact_pairing_P79 --time=04:00:00 slurm/robust_r2r3.sbatch
```
