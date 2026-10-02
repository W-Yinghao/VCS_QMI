# Pre-registration — P113: light evaluation-only batch on A100 (A-P3 into P109 / P110; P111 mechanism diagnostic) — FROZEN 2026-10-02T17:06:26Z

Owner 2026-10-02: "现在排队的有什么轻量化一点的job吗，提交到A100上".  All units are evaluation-only on saved epoch-800 checkpoints (no training, official
test set closed), 3 GPU jobs pinned to the A100 partition.

1. **P110 addendum (V1 / V2 / V3 for the P107 winner).**  The P110 prereg says P107 winners are added by an addendum with new lines: A-P3 seeds 0–2
   (`P107_AP3_views4_800ep_seed{0,1,2}`), identical modules, seeds and readings as P110 (one bundle, `slurm/p110_unit.sbatch`, OUT reports/P110).
   Read next to P110's four families; same rules (V3: no claim unless rule B improves the reversal outcome).
2. **P109 addendum (X1 + I1 for A-P3).**  Same runners and frozen readings: X1 on A-P3 seeds 0–2; I1 features; I1 power + level on A-P3 seed 0
   (cells identical to the other families' seed-0 encoders; seeds 441–444); I1 prediction effects on seeds 1–2 (seeds 445–448).  One bundle
   (`slurm/p109_bundles/p113_ap3.txt`).  One seed-0 encoder per family for power: descriptive, no family ranking.
3. **P111 mechanism diagnostic.**  Hypothesis stated in P111 (untested): stronger augmentation lowers positive-pair cosines, so a fixed zero-score
   threshold κ = 0.5 sits in the wrong place.  `scripts/p104_diagnostics.py` (§9.1: positive / negative cosine mean and quantiles under each run's own
   training augmentation, a, b, threshold −b/a, clean gate, geometry) at epoch 800 of: G2 std s0–2, G2 strong s0–2, recipe VCS std s0–2, recipe VCS
   strong s0–2 (P43 s0, P89 s1–2), A-P3 s0.  **Pre-stated reading (descriptive):** report positive-pair cosine mean and the fraction of positive pairs
   with cosine below the run's own threshold (κ = 0.5 for G2), standard vs strong, per seed.  "Consistent with the hypothesis" if G2-strong's positive
   cosine mean is lower than G2-std's on every seed **and** the fraction of positives below κ is higher on every seed; otherwise "not supported".  The
   recipe (learned threshold) is reported alongside as the adaptive comparison.  No causal claim from this diagnostic alone (P112 is the intervention).
