# Pre-registration DRAFT — P159 (SEP-3): the quadratic vs logistic rule in parametric critic classes under contamination, on frozen natural features (2026-10-15)

Status: DRAFT (P159 fork).  The main session freezes it (rename to `*_FROZEN_*`, freeze time in the title) before any formal cell.  Go-ahead field:
autonomous unit under the owner's idle-GPU authorisation of 2026-10-11 ("当你看到GPU有空位时，实验不需要我批准…你觉得有用的就自主提交"), following P157's
CONFIRMED outcome (`P157_SEP2_REPORT_20261015.md`) whose protocol limits every statement to synthetic Gaussians — this unit asks whether the same
comparison, on frozen natural features with no truth, shows the same direction on truth-free readouts.  No training of encoders, no image loading, no
official test split (COCO train2017 features and CIFAR-10 FIT-split features only); labels used only to describe the caches.  Code: `src/vcs_estim/p159.py`
(sources, draws, natural-data contamination, readouts, cells), `scripts/p159_sep3.py` (runner), `scripts/p159_aggregate.py` (tests + rule),
`tests/test_p159.py` (10 tests), `slurm/p159_sep3.sbatch` (gate | run).  `src/vcs_estim/sep.py` and `p157.py` are imported **unchanged** (asserted by a test):
critic classes, both scoring rules, trainer, lr grid, budget and the two selection criteria are P157's.

## Question
P156/P157: in parametric critic classes (cos-2 = the application critic a·cos + b; quad-3) under mixture contamination (independent pairs as positives;
false negatives among the negatives) the quadratic rule (VCS, γ = 1) fits the contaminated posterior more accurately than matched logistic (JS, γ = 0),
also at 4× the data; at ε = 0 the two are equal (P156 P3).  On natural features there is no posterior to compare against.  **Does the direction survive on
truth-free readouts of a clean, image-disjoint EVAL set — the common squared score J, AUROC for true-vs-product pairs and the calibration of (1 + T)/2 —
when the FIT/SELECT pairs are contaminated, and is it absent at ε = 0?**  Both scoring rules estimate the same posterior; any difference is a finite-sample /
contamination effect of the rule, not a property of the CS divergence (P157's framing is kept).

## Sources (existing caches only; provenance recorded in every cell JSON: file sha256, manifest sha256, draw indices sha256)
| source | cache | units | x | y (P pair) | d | N FIT |
|---|---|---|---|---|---|---|
| `coco_clip` | `outputs/P49_precheck_A/features/SRC-{FIT,CAL,EVAL}.pt` (manifest + splits.json; frozen open_clip ViT-B/32 laion2b; roles disjoint by image) | FIT 20 000 / CAL 5 000 / EVAL 5 000 images, 5 captions each | image feature (512, raw, norm ≈ 10) | one caption of the same image, drawn per seed | 512 | {4 096, 16 384} |
| `cifar_ap3` | `outputs/P150_features/P107_AP3_views4_800ep_seed1/features.pt` (manifest; frozen A-P3 encoder, P107 seed 1, 4v/800ep; 20 000 base images of the CIFAR-10 FIT split, versions clean / colour / blur; fp16 h) | 20 000 base images | h of the clean image (512, raw) | h of the **blur_s1** version (1 px Gaussian blur) of the same image | 512 | {4 096, 8 192} |
- Roles per draw (seed): FIT N, SELECT N / 4 (COCO from SRC-CAL; CIFAR from the pool), EVAL 4 096 (COCO from SRC-EVAL; CIFAR from the pool), **disjoint by image /
  base image** (asserted).  Q = product pairs by one non-zero cyclic shift of y inside the role (`cyclic1`; one shift per role per seed).
- **CIFAR pair view — disclosed change after the first gate (before any formal run).**  The draft first chose blur_s0.5 from the cache manifest's classifier
  accuracies (0.871 vs clean 0.877).  The first gate's timing cell (seed 300, clean, N 4 096, cos-2; job 1041362) gave J = 0.9995 for both arms with the lr at
  the top edge: the (clean, blur_s0.5) pair of the same image is almost identical through the frozen encoder (positive cos median 0.985) and perfectly
  separable from product pairs (cos median 0.17), so every readout saturates and the comparison is empty.  A descriptive cosine check on 4 000 base images
  (no fit, no EVAL readout; all seven versions × layers h / z / layer3) showed only blur_s1 gives overlapping pair laws on h (positive cos median 0.71,
  q10 0.46; product median 0.17, q90 0.31; colour_s0.2 0.988; layer3 is dominated by a common component, product cos 0.77).  The pair is therefore
  (h_clean, h_blur_s1) of the same base image: a nuisance view, not an SSL augmentation pair; there is no two-augmentation cache for A-P3 and the
  directive forbids new extraction.  The blur_s0.5 timing cell is not used.
- CIFAR N is {4 096, 8 192} (pool 20 000: 8 192 + 2 048 + 4 096), COCO {4 096, 16 384}; Q2's "larger N" is per source.
- Only the P159 fork read the caches; no selection used EVAL.

## Contamination (FIT and SELECT only — the practitioner's data; EVAL never contaminated; exactly round(ε n) units per role, indices SHA-256 recorded)
- `independent` (P_ε = (1 − ε) P + ε Q): the y of ε n P pairs replaced by the y of another unit of the same role (a product pair as a positive).
- `false_negative` (Q_ε = (1 − ε) Q + ε P): ε n Q pairs replaced by true pairs of another unit of the same role — COCO: a caption not used by that unit's P pair
  (a fresh true pair, as P156's fresh joint draws); CIFAR: the unit's only pair (a duplicate positive among the negatives, as in SSL batches) — disclosed.
- ε ∈ {0.1, 0.3}; ε = 0 ("clean") cells shared by both types.  Test: on a synthetic Gaussian draw the natural-data contaminator changes exactly the same
  number of units as P156's `contaminate_fn_roles`, leaves the complement and EVAL untouched, and the injected pairs are product-like / true-like.

## Critics, rules, trainer (P157, unchanged)
cos-2 and quad-3 on the raw 512-d features (quad-3: f = a x·y/d + b(|x|² + |y|²)/d + c, d = 512; both start at 0); arms VCS (γ 1, T = tanh f) and JS (γ 0,
T = tanh(f/2)); `sep.train_arm`: Adam, batch 256, 2 000 updates, lr grid {1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0}, selection every 100 updates on SELECT by
the arm's native risk (primary) and by the common J (sensitivity), finite-guarded re-selection (`reselect_finite`), lr-edge flags.

## Readouts on clean EVAL (per selected critic; T = arm_T(f); pair label 1 for P, 0 for Q)
(1) common squared score J = E_P[T − T²/2] + E_Q[−T − T²/2]; (2) the arm's native risk (reported, not compared across rules); (3) AUROC of f (rank based,
tie-averaged); (4) ECE of (1 + T)/2 against the label (15 equal-mass bins); plus gate mean 1 − T² and the |T| > 0.95 mass.
Effects per cell and seed (positive = VCS better): **dJ_rel = (J_VCS − J_JS) / mean-over-seeds J_JS** (P157's standardisation); **dA_rel = (AUROC_VCS −
AUROC_JS) / mean-over-seeds (1 − AUROC_JS)** (error-rate scale, since AUROC ≈ 1); dECE = ECE_JS − ECE_VCS.  Beside every absolute effect the aggregate
reports each rule's **degradation ratio J(ε) / J(0)** on the same seed's clean cell and the paired difference of ratios (the R4-E0 reading) — both, never one
alone (`reports/R4/R4_02_EVIDENCE_LEDGER_20261011.md` l.25).

## Cells and seeds
Per seed: 2 sources × 2 N × 2 classes × (1 clean + 2 types × 2 ε) = **40 cells**; seeds 300–307 (fresh; P156 used 100–107, P157 200–207) → 320 cells.
Each cell = 2 arms × 7 lrs × 2 000 updates (both selections read out).  Cell names `P159_S3_<source>_<cls>_<ctype>_e<ε>_N<N>_U2000_s<seed>.json`.

## Primary family (Holm over 3; paired t over the 8 seeds of the per-seed mean over the test's cells; `scripts/p159_aggregate.py`)
- **Q1** dJ_rel over every ε > 0 cell (both sources, classes, types, ε, N): is VCS's clean-EVAL J higher than JS's after contaminated fitting?
- **Q2** dJ_rel over the ε = 0 cells: expected ≈ 0 (P156 P3); a non-zero Q2 means the difference is not contamination-specific.
- **Q3** dA_rel over every ε > 0 cell.
**Outcome rule (fixed now):** **SUPPORTED on natural features** if Q1 > 0 (Holm) and Q2's 95 % interval contains 0; **FIT-ONLY** if Q1 > 0 and Q2 > 0
(Holm); **CONTAMINATION-SPECIFIC WITH CLEAN DEFICIT** if Q1 > 0 and Q2 < 0 (Holm); **NOT SUPPORTED** if Q1 is not > 0.  Q3 is reported with its reading in
every branch (ranking is a separate question from the score).  Permitted sentences: SUPPORTED — "On frozen natural features (CLIP image–caption pairs; A-P3
CIFAR view pairs), parametric critics fitted on contaminated pairs reach a higher clean-set squared score under the quadratic rule than under matched logistic,
with no difference on clean fitting"; FIT-ONLY — the same sentence without the last clause plus "the quadratic rule also leads on clean fitting in these classes
(a property of the fit, not of contamination)"; CLEAN DEFICIT — "… while matched logistic leads on clean fitting"; NOT SUPPORTED — "P157's synthetic result did not
show on truth-free readouts of frozen natural features".  Every branch is reported as a property of the scoring rule's fit (the Brier-score L² projection
P157 cites), never as robustness, never as a CS-divergence property, and never as an SSL / VL claim.

## Secondary and descriptive (pre-registered)
Per-cell dJ_rel, dA_rel, dECE (Holm within each list of 40); the degradation-ratio table (32 contaminated cells); Q1–Q3 under common-J selection; dJ_rel by
factor (source, class, type, ε, N); QC — lr-edge shares (top lr 1.0) and not-converged counts per source × class × arm; if a class selects the top lr in more
than half of its selections it is reported as a limitation (no extension).  Missing cells are listed; a test is read only if `complete` (8 seeds × all cells).

## Not claimed
Nothing about SSL training, VL fine-tuning or any downstream accuracy; nothing about expressive critics; no robustness statement; no claim beyond the two
feature sources and the two contamination constructions; no novelty (Buja, Stuetzle & Shen 2005; the L² projection of the Brier score; the noisy-label
literature of P156 §6).

## Gate (to be filled from the gate job before the freeze)
- CPU/GPU gate job: `sbatch --job-name=p159_gate slurm/p159_sep3.sbatch gate` → `reports/P159_GATE/`: tests (`tests/test_p159.py` + `tests/test_p157.py`),
  `--smoke` run of one seed's 40 cells (N 256, 40 updates, EVAL 512; outputs/P159_smoke, not used), smoke aggregate, and one timed full-size cell per
  source (outputs/P159_timing: seed 300, first cell = clean, N 4 096 cos-2 — its result is recomputed in the formal run; disclosed).
- Local run before the gate: 10 / 10 tests on the login node (CPU).

## Cost and submission
Measured in the gate (seconds per full cell per source).  Expected from P157 (23 s per cell at d 64 / N 16 384 on L40S): ≈ 20–40 s per cell → 40 cells ≈
15–30 min per seed; 8 seeds ≈ 2–4 GPU-h.  Submission as **2 jobs of 4 seeds each** (`run 300+301+302+303`, `run 304+305+306+307`; --time 03:00:00), not 8
short jobs (owner 2026-10-14: merge short GPU jobs); partitions RTX6000PRO / H100 / L40S excluding node51, node60; nice 400 so every V8 / REV10 job starts
first.  Aggregation: `scripts/p159_aggregate.py --dir outputs/P159_sep3 --out reports/P159_SEP3_results/p159_aggregate`; results-only commit, then the report.

## Delivery (v5 §8 fields)
```yaml
experiment_family: measurement (frozen natural features; scoring-rule comparison; no truth)
protocol_id: P159 (SEP-3)
sources: coco_clip (P49 roles), cifar_ap3 (P150 cache, clean / blur_s0.5 h)
estimator: cos2 / quad3 critics, VCS (gamma 1) vs matched JS (gamma 0); P157 trainer, grid, budget, selections
contamination: independent / false_negative at eps 0.1, 0.3 on FIT + SELECT; EVAL clean
readouts: clean-EVAL common J (primary), AUROC (Q3), ECE, native risk; degradation ratios beside absolute effects
negatives: cyclic1 (one non-zero cyclic shift per role per seed)
n_independent_units: 8 seeds x 40 cells; EVAL 4096 pairs per draw
status: draft
```
