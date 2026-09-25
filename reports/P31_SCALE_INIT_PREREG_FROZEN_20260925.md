# Pre-registration — cosine-critic initial scale (P31), frozen 2026-09-25 before launch

Trigger: wave G unit `P26_vcs_a5_learn_seed0` (cosine critic tanh(a⟨z1,z2⟩+b), K=8, negative partner detached, **a0 = 5** learnable, b0 = 0)
finished at **81.90 / kNN 77.00 / h-rank 45.0 / heldout-J 0.942**, vs the base (a0 = 1) 80.48 / 80.58 / 80.72 on seeds 0/1/2 (mean 80.59, SD 0.12):
+1.3 on a single seed.  The learned (a, b) end in the same place (10.08 / −8.22 vs 9.81 / −8.09; threshold cos* 0.82); in a5_learn `a` first *falls*
to 3.3 by epoch 20 and is back at 5.5 by epoch 50, so the two runs differ only in the first ≈50 epochs, yet kNN separates from epoch 100 on
(73.5 vs 71.8) and never closes.  Hypothesis: the initial sharpness of the score decides which encoder trajectory the pair-wise regression
selects (synthesis §4.3: the optimum is degenerate, dynamics pick the geometry).

## Units (stage `P31_vcs_scale_init`, 200 epochs, source `cifar10_hpG_a5_learn_vcs_seed0.yaml`)
| unit | a0 | seed | question |
|---|---|---|---|
| a5_learn seed1 / seed2 | 5 | 1, 2 | does +1.3 hold across seeds (new base candidate; base SD 0.12) |
| a2_learn | 2 | 0 | sweep: is the effect monotone in a0 |
| a10_learn | 10 | 0 | " |
| a20_learn | 20 | 0 | " (a20 starts ≈ where the base *ends*; tests whether the early-sharpness gain survives an extreme start) |

`a5_fixed` (a = 5 frozen; kNN 76.54 at epoch 150, running) is read with these: if a5_fixed ≈ a5_learn, late scale freedom is irrelevant and the
sweep's best a0 becomes the base; a fixed-scale sweep is only queued if a5_fixed > a5_learn.

## Reading
Primary: final linear-val vs 80.59 (the 3-seed base mean); a5 seeds give mean ± SD to compare with 81.90.  HELPS/HURTS at ±1.0 as before.
Secondary: kNN, h-rank, heldout-J, `cos_scale`/`cos_bias` trajectories (does every a0 converge to a ≈ 10, b ≈ −8?).  Sweep read as a curve
linear(a0) at a0 ∈ {1, 2, 5, 10, 20}, single seed except a0 ∈ {1, 5}.  No code change since the P28 gate (commit b12ef89): configs only.
Configs and sha256: `configs/HPARAM_I_SHA256.json`.
