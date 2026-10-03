# Pre-registration — P124: v6 V6-GRANULARITY — CIFAR-100 coarse / fine / conditional readouts of frozen CIFAR-100 SSL encoders, 2026-10-03 — FROZEN 2026-10-03T19:37:52Z before the full job (CPU gate 1020289, GPU smoke 1020284)

Status: FROZEN 2026-10-03T19:37:52Z (main session; owner 2026-10-03: organise and submit the v6 plan).  The main session freezes it (rename to `*_FROZEN_*`, freeze time in the title) before the full GPU job.  Source:
`VCS_Server_Tasks_and_Theory_v6_20261003.zip` → `VCS_Server_Tasks_v6_CN.md` §10 (V6-GRANULARITY), §11.  Code: `src/vcs_vtask/granularity.py` (new),
`scripts/p124_gran.py`, `scripts/p124_aggregate.py`, `slurm/p124_{gate,gpu_smoke,unit}.sbatch`, `tests/test_p124.py`.  No encoder training.

## Question (no direction assumed)
Does the scorer scale (fixed angular scale, G2 / A-P3) mainly improve the **coarse** (20 superclass) structure of the CIFAR-100 representation without
improving **within-coarse fine** readability — or the reverse, or neither?  CIFAR-100 from-scratch SSL showed A-P3 = recipe VCS on 100-way linear
(P107 layer 2: 60.00 vs 59.91) with kNN behind SimCLR; this unit asks *where* in the label hierarchy the methods differ.  The answer may go either way;
"C100 did not improve" is not taken as evidence of fine-information loss.

## Encoders (frozen, read-only, epoch 800; only COMPLETED runs)
recipe VCS `P91_c100_vcs_a5_views4_800ep_seed{0,1,2}`, G2 `P111_G2_c100_views4_800ep_seed{0,1,2}`, A-P3 `P107_AP3_c100_views4_800ep_seed{0,1,2}`, SimCLR
`P91_c100_simclr_views4_800ep_seed{0,1,2}` — 12 encoders, all trained on the P91 CIFAR-100 dev45k / val5k manifest (asserted per encoder).
CIFAR-10 → CIFAR-100 transfer (P119) is a different table and is **not mixed** with these from-scratch encoders.

## Data, tasks and readouts
- Images: the 45 000 FIT images train every readout; the 5 000 selection images are the evaluation set; the official CIFAR-100 test file is never opened.
- Labels: fine and coarse labels of the training images from the local python `train` file (`coarse_labels`); the fine labels are asserted equal to the
  torchvision order position by position; the fine→coarse map is asserted to be a function with 5 fine classes per coarse class (official table).
- Features: float32 h (512-d encoder output) with each run's own clean normalisation, cached under `outputs/P124_features/`.
- Tasks: (1) **coarse** 20-way; (2) **fine** 100-way; (3) **conditional** 5-way — the coarse label is given at test time; one dedicated probe per coarse
  group is trained on that group's FIT images and evaluated on that group's selection images; reported **macro** (mean over the 20 groups) and
  **overall** (pooled correct / total).  The conditional task is labelled as conditional and is never reported as unconditional 100-way accuracy.
- Readouts (identical map for every method; the code has no method argument):
  - **`recipe_raw` — PRIMARY** for every task: the original frozen-h linear readout (the training evaluation's recipe probe on raw h: SGD momentum 0.9,
    lr 0.1, wd 0, 100 epochs, cosine to 1e-3, batch 256, seed 20260925).  Anchor: the fine 100-way value must reproduce the stored training-evaluation
    linear accuracy within GPU noise (|Δ| ≤ 0.3); a failing anchor is flagged and the encoder's rows are not read.
  - `raw_unstd`, `raw_std`, `l2_std` (P119 maps; standardisation with training statistics only): (lr, wd) chosen on an inner 80 / 20 split of the FIT
    labels of that task only (grid 3 × 2), refit on all FIT labels; fixed inner-split seeds identical for every method (coarse 124020, fine 124100, group g
    124200 + g).
  - `knn`: recipe k 200, T 0.1, cosine on h; for the conditional task the bank is the FIT images of the same coarse group.
- Secondary conditional readout: `masked_fine_head` — the 100-way recipe head restricted to the 5 fine logits of the given coarse group.
- Derived (identity, tested): from the 100-way recipe head, implied-coarse accuracy (coarse class of the argmax fine class) and fine accuracy given
  implied coarse correct; fine accuracy = their product.

## Pre-stated reading (descriptive; seed-paired)
- Per family: mean ± sd over the 3 encoder seeds for every task × readout.
- Contrasts, paired by seed: **A-P3 − recipe VCS**, **G2 − recipe VCS** (the scorer-scale effect within VCS) and **A-P3 − SimCLR** (method); per-seed
  differences, mean, sd, 95 % t interval (df 2).  Labels per Δ (as P114 / P120): **close** |Δ| < 0.3; **clear +/−** |Δ| ≥ 0.3 and the interval excludes 0;
  **inconclusive** otherwise.
- **Pattern statement — PRIMARY readout only**, from the labels of Δcoarse and Δconditional-macro: "coarse-specific gain" if Δcoarse is clear + and
  Δconditional is close or clear −; "within-coarse gain" if the reverse; "both up" if both clear +; "neither" if both close; otherwise "no pattern
  (inconclusive at 3 seeds)".  The other readouts are reported as robustness columns; the primary readout is not switched after results.
- The decomposition table and the masked-head conditional readout are descriptive; no rule.
- Claim labels (v6 §11.2): the accuracies are **observed results**; the decomposition product is an **identity verified by definition**; any statement
  about *why* (e.g. "the fixed scale organises superclasses") is a **hypothesis to be tested**, not a finding of this unit.
- Not claimed: anything about the official test set, other datasets, CIFAR-10 → CIFAR-100 transfer, or information content (a readout difference is
  not an information measurement; higher measured J would never be read as better classification).

## Label use (disclosed)
Fine and coarse labels are used only by the downstream readouts (probes, kNN votes, conditional grouping) and the evaluation.  They never enter SSL
pretraining or pair generation.

## Optional dependence measurement — not built (open decision)
v6 §10 allows a small conditional dependence measurement of representation vs fine label given coarse (P109 / P118 machinery).  It is **not** part of
this unit: "readout first"; it can be added by an addendum after the readout results.

## Gate and smoke (disclosed; numbers are not results)
- CPU gate, job 1020289 (exit 0): `tests/test_p124.py` 7 / 7 (coarse / fine map vs the official superclass table, pickle vs torchvision order,
  conditional subsets restricted to the group, masked predictions never leave the given group, decomposition identity, readouts symmetric and scale
  invariant where they should be, aggregator labels and pattern rule); P119 / P110 regression 15 / 15; CPU runner smoke on A-P3 and SimCLR seed 0
  (subsets 1 500 / 500 images) and the aggregator ran.  The first gate run (job 1020283) failed only on a wrong hand-computed toy mean in the test
  (1.00, not 0.967); the test was corrected, no code changed.
- GPU smoke, job 1020284 (A100, exit 0): the full-size readout on A-P3 C100 seed 0 — **anchor reproduced exactly** (fine `recipe_raw` 60.20 = stored 60.20;
  fine kNN 55.92 = stored 55.92); coarse 72.06, conditional macro 76.84 (group range 44.0–90.0), implied coarse 73.34 × fine given coarse correct 82.08
  = 60.20; masked-head conditional 76.74.  750 s per encoder (selected readouts ≈ 75 s each per task; conditional = 20 group probes per readout).

## Cost
12 encoders × ≈ 750 s ≈ 2.5 GPU-h in ONE bundle job (`slurm/p124_lines.txt`; partitions A100 / L40S / RTX6000PRO / H100; normal QOS; never runfill;
node60 allowed); ≈ 1.2 GB float32 feature cache.

## Delivery (v6 §11.2 fields)
```yaml
logical_id: V6-GRANULARITY
p_number: P124
dataset: cifar100   pretrain-dataset: cifar100   eval-dataset: cifar100 (selection split)
encoders: recipe VCS / G2 / A-P3 / SimCLR, ResNet-18 CIFAR, seeds 0-2, epoch 800
readouts: recipe_raw (primary), raw_unstd, raw_std, l2_std, knn; masked_fine_head (secondary, conditional)
tasks: coarse 20-way, fine 100-way, conditional 5-way given coarse (macro / overall)
independent_units: encoder seeds (3 per family); evaluation images 5000
status: draft
```

## Decisions at the freeze (main session)
1. Launch the single bundled GPU job now.  Primary readout `recipe_raw` (fixed before results); labels close / clear / inconclusive and the pattern rule as drafted.
2. The optional conditional dependence measurement (representation vs fine label given coarse) is deferred to a possible addendum after the readouts (v6 §10:
   readout first).
3. The failed first gate (1020283; a wrong toy mean in a test, code unchanged) is kept as history.
