# Pre-registration DRAFT — R2 / R3 (P71): noisy pairing in training and mismatch identification, frozen CLIP + identity adapters (2026-09-28) — FROZEN 2026-09-28T14:47:32Z before GPU compute (CPU smoke 1013017 disclosed inside; thresholds loosened per the owner; JS = balanced logistic on the same negatives is in the control set)

Status: DRAFT (implementation fork); the main session freezes it (rename to `*_FROZEN_*`) before the GPU run.  Source: `CS_QMI/VCS_QMI_Next_Round_Plan_v2.md`
R line (R2, R3) and `reports/NEXT_ROUND_EXECUTION_PLAN_DRAFT_20260928.md` §2, with the owner's instruction to loosen the threshold numbers.
Code: `scripts/robust_r2r3.py`, `slurm/robust_r2r3.sbatch`.  R1 (synthetic contamination) is a separate unit (P69 / E module).

## Question
Does the VCS objective, trained on pairs of which a known fraction is mismatched, lose less clean retrieval than InfoNCE, pairwise logistic and
JS (balanced logistic with the same negative construction) — and does its native score identify the mismatched training pairs better than the
competitors' native scores, with no clean calibration set?  (Brief §R: the claim is only VCS's if it survives JS.)

## Data and splits (unchanged from P49 / P57; identity = COCO image id; no official test set)
- animal shift: `/home/infres/yinwang/CS_QMI/outputs/P49_precheck_A/` (SRC-FIT 20 000, SRC-CAL 5 000, SRC-EVAL 5 000, TGT-EVAL 5 000 animal images).
- indoor → outdoor shift: `/home/infres/yinwang/CS_QMI/outputs/P57_precheck_A_wave2/features_indoor_outdoor/` (SRC-FIT 13 327, CAL 5 000, EVAL 5 000, TGT 5 000).
- Frozen CLIP ViT-B/32 features; identity-initialised 512 × 512 adapters (P50); topic pairing = the positive caption comes from a random image
  with the same supercategory set (P50 setting 2), captions 0–3 for training, caption 4 held out for evaluation.

## Mismatch injection (R2 knob)
For m ∈ {0, 0.1, 0.2, 0.4, 0.6}: a seeded fraction m of SRC-FIT images (seed = [20260928, round(1000 m), sha(shift)]) has its training positive
replaced by a single random *other* image's captions (uniform over the split, not topic-matched); all other images keep their topic candidates.
The injected indices, their wrong partners and a sha256 of the index list are written to the run JSON.  SRC-CAL (selection) and the
evaluation splits are clean.  Structured (topic-matched) mismatches are a follow-up, not this unit.

## Methods, equal budget
Grid lr ∈ {1e-3, 3e-4, 1e-4} × epochs ∈ {5, 15, 40} (the P49 grid), selected on clean SRC-CAL by each method's own loss (topic pairing on CAL);
3 seeds of the selected configuration; AdamW wd 1e-4, batch 256, cosine schedule.
- **vcs**: J with tanh(a·cos + b), K = 8 independent-pool caption negatives (P50).
- **infonce**: in-batch symmetric InfoNCE, learned temperature.
- **logistic**: pairwise logistic on the in-batch pairs (native; 1 positive : 255 negatives).  The "+log K intercept" variant is a score
  correction (logit + log 255): rank-equivalent, hence identical R@k and AUROC; it is reported as a column of the same model, not trained
  separately (it matters for calibration readings, which are not part of R2 / R3).
- **js**: balanced logistic in the Deep-InfoMax form  E_P[softplus(−f)] + E_Q[softplus(f)],  f = a·cos + b, on the *same* positive / K = 8
  pool-negative construction as vcs (equal weight of the two terms, like VCS's M = (P + Q)/2).  Same initialisation as logistic (a = 10, b = −10).
  Implemented in `robust_r2r3.py` (`train_r`), which replicates `precheck_a_adapters.train` byte-for-byte for the other three methods.

## Metrics
- **R2**: image → text R@1 and R@5 of the own caption (caption 4) on clean SRC-EVAL and TGT-EVAL, per m; degradation Δ(m) = R@1(m) − R@1(0) per
  method (3-seed means, sd printed).
- **R3** (m ∈ {0.2, 0.4}): each trained model scores the fixed training pairing of SRC-FIT — injected partner if injected, else the fixed topic
  partner (else the image itself), caption 0 — with its native score (vcs (1+T)/2; infonce cos/τ; logistic logit; js a·cos + b); AUROC of
  "clean" over "injected" (rank statistic).  Reference: raw CLIP cosine of the same pairs.  No clean calibration set is used anywhere in R3.

## Pre-committed reading (loosened per the owner)
- **R2 holds** if VCS's R@1 loss from m = 0 to m = 0.4 is smaller than each competitor's (infonce, logistic, js) by ≥ 0.5 point on *both* targets
  (SRC-EVAL and TGT-EVAL, 3-seed means); **refuted** if the four degradation curves coincide within 0.5 point at m = 0.4 or VCS's loss is larger than
  any competitor's; **conditional** otherwise (e.g. holds on one target, or only against infonce / logistic but not js — in which case the effect
  belongs to the balanced-mixture family, not to VCS, and is written so).
- **R3 holds** if VCS's AUROC ≥ each competitor's + 0.01 at both m = 0.2 and 0.4 on both shifts; **refuted** if VCS is within 0.01 of the best
  competitor or below it at either m; conditional otherwise.  The raw-CLIP AUROC is the floor every adapter must clear to count.
- QC sentinels: at m = 0 every method's R@1 is within 1 point of its P58 topic-pairing value on the same split; the injected count equals
  round(m · n_fit); the selection picks a grid point strictly inside the grid for at least one method (else the grid is too narrow and is noted).
- Not claimed: other towers, non-linear adapters, structured mismatches, anything about training from scratch.

## Smoke probe (CPU job 1013017, `slurm_logs/robust_r2r3_r2r3_smoke_1013017.out`, exit 0, 13 s; 2 000 images per split, lr 1e-3 × 1 epoch, seed 0,
m ∈ {0, 0.4}, all four methods, both shifts) — code-path evidence only, not results
- Injection: 800 / 2 000 images per shift, indices and wrong partners saved (sha256 cff80b00… animal, d4873af8… indoor).
- The js method trains and is selected like the others (CAL loss 6.05 / 5.49 at 1 epoch from the logistic initialisation).
- animal, m = 0.4: SRC R@1 vcs 0.437, infonce 0.473, logistic 0.381, js 0.382; R3 AUROC vcs 0.829, infonce 0.830, logistic 0.789 (+log K identical
  0.789), js 0.786, raw CLIP 0.811.  indoor → outdoor, m = 0.4: R3 AUROC 0.758 / 0.746 / 0.701 / 0.694, raw CLIP 0.728.  One epoch barely moves the
  identity-initialised adapters, so these numbers say only that the pipeline runs end to end.

## Compute
Per shift: 5 m × 4 trained methods × (9 grid + 3 seeds) = 240 adapter trainings (+ R3 scoring at two m) ≈ 20–30 min on one GPU (P50 setting 2 ran 36
trainings in 160 s); both shifts in one job ≈ 1 GPU-hour.
