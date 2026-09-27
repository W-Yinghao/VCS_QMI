# Pre-registration — pre-check A: calibration and threshold transfer (P49), frozen 2026-09-27 before launch (owner approved the CLIP download and GPU use)

**Property tested.** Pairwise VCS scores have absolute meaning: at the population optimum T* = tanh(PMI/2) = 2·P(joint | x, y) − 1 under a
balanced prior, so (1+T)/2 is a match probability and a T-threshold is a PMI threshold.  Question (brief §2, A): after finite training on
frozen cross-modal features, is (1+T)/2 calibrated, and does a T-threshold chosen on a source distribution transfer to a shifted target
better than a cosine threshold does?  Competitors on the same adapters and pairs: InfoNCE (learnable temperature) and pairwise logistic
with learnable scale and bias (SigLIP-style; Brier-vs-logistic is the only difference to VCS).

## Data (local, no download): COCO 2017 captions, identity = image id, one caption ↔ its own image only
Index built 2026-09-27 (`/home/infres/yinwang/CS_QMI/data/coco_index/`, metadata only): train2017 118 287 images (5 captions each),
`captions_train2017.json` sha256 4b620863…; annotations CC BY 4.0, images under their Flickr licences, research use; copy owned by the user.
- **Source distribution** = train2017 images whose instance annotations contain **no `animal` supercategory** (≈ 94 k); from these, three
  identity-disjoint subsets by a fixed seed: SRC-FIT (adapter training, 20 000 images), SRC-CAL (threshold / calibration selection, 5 000),
  SRC-EVAL (source-side evaluation, 5 000).
- **Target distribution** = train2017 images **with** an `animal` supercategory (≈ 24 k): TGT-EVAL 5 000 images; never used for fitting or
  selection.  This is a constructed covariate shift (scene and vocabulary change) with the same annotation protocol; a second external target
  (Flickr30k) is listed as optional and would need its own licence check and download.
- Evaluation pairs: every image with one held-out caption (joint) and, for the product samples, a caption of a *different* image drawn from an
  independent caption pool of the same split (never in-batch shuffles; Hoeffding-bound premises).  Balanced 50/50 joint/product for reliability
  diagrams; also the natural "1 : 9" mixture as a robustness row (calibration depends on the prior — reported, not claimed).

## Frozen towers and adapters
- Towers: open CLIP ViT-B/32 `laion2b_s34b_b79k` (open_clip_torch 3.3.0; weights 605 143 316 bytes, sha256 ac4f8c4b88af6d96…; source HF
  `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`; provenance file `models/open_clip/PROVENANCE_ViT-B-32_laion2b_s34b_b79k.json`).  Both towers frozen;
  features cached once on a GPU (`scripts/precheck_a_features.py`; image 512-d and the 5 captions per image, 512-d each).
- Adapters: identical for all methods — one linear layer per tower (512 → 256, bias) followed by L2 (VCS/InfoNCE) or raw (logistic uses the
  dot product with learnable scale/bias); trained on SRC-FIT only; AdamW, lr and epochs from a small grid *per method* with equal budget (6
  configurations each; selection on SRC-CAL by the method's own validation loss).  Batch 256 image–caption pairs; K = 8 independent-pool
  negatives for VCS; InfoNCE uses the in-batch 2B−2 negatives (its native form); logistic uses the same in-batch pairs as its negatives.
- VCS critic: T = tanh(a·⟨u, v⟩ + b) on the adapted L2 features (cosine critic, the SSL recipe form; a₀ = 5, b₀ = 0); the SSL detach is *not*
  used here (both towers frozen, adapters small; noted as a possible ablation).

## Measurements
1. **Reliability diagrams** on SRC-EVAL and TGT-EVAL (balanced pairs): predicted match probability p̂ = (1+T)/2 for VCS; for InfoNCE / cosine
   the uncalibrated sigmoid of the scaled cosine and, as the honest competitor, Platt scaling fitted on SRC-CAL; for logistic σ(a·cos + b).
   Metrics: ECE (15 equal-mass bins), Brier score, reliability curve max deviation.  Pre-committed claim: VCS ECE on TGT-EVAL ≤ the ECE of
   Platt-scaled cosine on TGT-EVAL (the transfer setting) and ≤ 0.05 absolute on SRC-EVAL.
2. **Threshold transfer**: on SRC-CAL choose the threshold giving a target false-negative rate of 5 % (and 10 %) for each score (T, cosine,
   logistic logit); apply unchanged on TGT-EVAL; report FNR and FPR drift |Δ|.  Claim: VCS drift ≤ ½ of the cosine drift for both rates.
3. **Mid-dependence check** (brief P1): held-out J on SRC-EVAL must be < 0.9; else the setting is degenerate and A is "not decidable" here.
4. Side rows: retrieval R@1/R@5 (secondary, ranking metric); learned a, b; saturation fractions; PMI recovery 2·artanh(T) numerical range.

## Reading (pre-committed)
Property holds: claims 1 and 2 hold on the target and J < 0.9.  Holds conditionally: only one of the two claims, or only at one FNR level, or
only on the source.  Does not hold: VCS ECE ≥ Platt-cosine ECE and no drift advantage; or J ≥ 0.9 (degenerate regime — report and stop).
Seeds: 3 adapter seeds (cheap on cached features).  Compute: CPU only.  Not claimed: retrieval quality; any task.

## Split rule (final)
Source pool = train2017 images with ≥ 1 instance annotation, no `animal` supercategory and 5 captions; target pool = images with `animal` and 5
captions; seed 20260927; SRC-FIT 20 000 / SRC-CAL 5 000 / SRC-EVAL 5 000 / TGT-EVAL 5 000, id lists and sha256 in `outputs/P49_precheck_A/splits.json`.

## Compute
Feature extraction and the adapter grids run on one GPU (short jobs); tests and calibration on CPU.  Adapter grid per method: lr ∈ {1e-3, 3e-4}
× epochs ∈ {5, 15, 40} on cached features (6 configurations, selection on SRC-CAL by the method's own loss); 3 seeds of the selected
configuration.  Step 2 code: `scripts/precheck_a_adapters.py` (to be committed before its run).

## Addendum 2026-09-27 (smoke-probe findings, frozen before the full run; probe output `P50_precheck_A_smoke.md` disclosed)
1. Randomly initialised 512→256 adapters destroy CLIP's joint space (R@1 ≈ 0.02 after the probe epoch).  Change: adapters are 512→512 linear
   layers initialised at the identity (zero bias) for every method, so all methods start from the same pre-trained alignment; a raw-CLIP
   (no adapter) reference row is added (cosine + Platt on CAL, R@1, FNR/FPR at the CAL-quantile threshold).
2. A quantile threshold set on CAL transfers identically for a monotone critic and its cosine (tanh(a·cos + b) is monotone in cos) — the probe
   rows were identical by construction.  The threshold-transfer claim is therefore evaluated on the **absolute** rule that only a calibrated
   score offers: VCS T ≥ 0 (PMI ≥ 0, p̂ ≥ ½) and logistic logit ≥ 0, applied unchanged on source and target with no calibration data; the cosine
   competitor receives the CAL threshold that matches the method's *source* FNR (same source operating point).  Reported: FNR, FPR, balanced
   error on SRC-EVAL / TGT-EVAL and their drift; the CAL-quantile row is kept as a sanity check.  Claim 2 now reads: drift of the balanced
   error under the absolute VCS rule ≤ ½ of the drift of the matched cosine rule.
3. Probe sanity: held-out J on SRC-EVAL 0.44 and on TGT-EVAL 0.10 after one epoch — the setting is in the mid-dependence regime (P1 satisfied).
4. Smoke 2 (identity adapters): raw CLIP R@1 0.49 / 0.45 (source / target); after one probe epoch InfoNCE 0.46, logistic 0.39, VCS 0.20 — the VCS
   objective moves the adapters fastest at lr 1e-3, so the grid gets a third learning rate, lr ∈ {1e-3, 3e-4, 1e-4} × epochs ∈ {5, 15, 40}
   (9 configurations per method, still equal budgets; selection unchanged).  Full run launched after this addendum.
