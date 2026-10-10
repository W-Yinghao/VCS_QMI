# R4-VL0 — candidate diagnostics on existing VL3 checkpoints (read-only) — definitions FROZEN 2026-10-11, before any score is computed

Package `r4/package_r4_20261010` (docs/04 §2–3).  Exploratory, descriptive: the DEV split has been used many times, the dataset / backbone choice
(RefCOCOg, RefCOCO+ × SigLIP 2 B/16) is post hoc, and nothing here is a confirmation.  No fitting, no new head; frozen forward passes of the 18 existing
CAL-Top-1 checkpoints (2 datasets × VCS / matched JS / candidate softmax × seeds 0–2).

## Scoring pass (`scripts/r4_vl0_scores.py`)
Cosine of every DEV expression against (a) the VL1-21 proposals as used by VL1-22 (top-100, score ≥ 0.05, per-class NMS, clipped, < 1 px dropped;
detector-score order) and (b) the image's annotated referred boxes.  Same crops / preprocessing / tokenisation as VL3 and VL1-22.
- **G1:** given-box DEV image-macro Top-1 from (b) reproduces the run's value within 0.3 point.
- **G2:** all-proposal Acc@IoU 0.5 image-macro from (a) reproduces the VL1-22 Table C value of the same checkpoint within 0.3 point (exact difference reported).
A checkpoint failing a gate is reported and excluded from the diagnostics.

## Diagnostics (`scripts/r4_vl0_analyse.py`; all methods share every subset, K and permutation)
Valid proposal = IoU ≥ 0.5 with the expression's referred box.  Covered query = at least one valid proposal.
1. **Main table (restored):** all proposals, all eligible queries, image-macro Acc@0.5 (uncovered queries count as errors); also query-weighted
   accuracy, covered-query conditional accuracy, coverage.
2. **Deployable count curve:** the first K proposals in detector-score order (query- and method-independent), K ∈ {4, 8, 16, 32, all}; accuracy and
   coverage at each K.
3. **Oracle-assisted distractor curve** (covered queries only; uses ground truth, not a deployable procedure): keep the single highest-IoU valid
   proposal, add proposals with IoU < 0.5 to the target in a fixed per-image random order (seed = image_id), n_distractors ∈ {1, 3, 7, 15, 31, all};
   other valid proposals are removed.  Accuracy = the kept valid proposal outranks every added distractor.
4. **Oracle-assisted 2×2** (covered queries): target box A = annotated box, B = highest-IoU proposal; distractors C = the image's other annotated
   referred boxes, D = the same number of proposals with IoU < 0.5 to the target, taken in detector-score order.  Cells A+C (= the given-box setting
   restricted to covered queries), B+C, A+D, B+D.
5. **Error types** (all-proposal top-1 errors; geometric strata, not semantic labels): (i) other annotated object — IoU ≥ 0.5 with another COCO
   annotation of the image (referred or not), split same category as the target / different category; (ii) part / duplicate of the target — not (i)
   and (IoU with the target ≥ 0.1 or ≥ 80 % of the proposal inside the target box); (iii) no matching annotation — everything else (not called background).
6. **Margins:** per covered query, best valid cosine − best invalid cosine (null when no valid or no invalid proposal).
7. **Contrasts:** per seed, VCS − softmax and VCS − JS on every quantity above; image-level bootstrap (1 000 resamples of images, seed-averaged
   per-image differences) and the seed spread.  No percentage causal decomposition across the factors.
