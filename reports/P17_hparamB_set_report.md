# P17 — B-group "set" batch report (P16): augmentation, batch size, weight decay, first combinations, 800-epoch combination

Status: **COMPLETE** (14/14 COMPLETED, 2026-09-25 02:36 → 13:0x UTC).  Neutral table: `reports/P17_hparamB_set_results_table.md`.
Pre-registration: `reports/P16_HPARAM_B_SET_PREREG_FROZEN_20260925.md`.  Concat-MLP critic throughout (this batch predates the critic-form
result of P19).  Baselines: `P5_vcs_seed0` 74.84 % / kNN 64.64 / h-rank 12.9 (200 ep); `P8_vcs800_seed0` 79.24 % / 71.36 / 20.1 (800 ep).

## Results

| unit | change | linear | Δ | kNN | h-rank | heldout-J | signal |
|---|---|---|---|---|---|---|---|
| aug_weak | crop [0.5,1], jitter 0.2 | 67.78 | −7.06 | 55.20 | 12.4 | 0.975 | HURTS (large) |
| aug_crop008 | crop [0.08,1] | 74.04 | −0.80 | 63.12 | 13.8 | 0.827 | neutral |
| aug_cj08 | jitter 0.8/0.2 | 75.14 | +0.30 | 65.38 | 15.5 | 0.884 | neutral |
| aug_blur05 | + blur p 0.5 | 73.64 | −1.20 | 64.90 | 14.2 | 0.893 | HURTS |
| aug_strong | all three | 73.36 | −1.48 | 65.32 | 14.8 | 0.787 | HURTS |
| b128 / b512 / b1024 | batch size | 73.22 / 74.34 / 73.00 | −1.62 / −0.50 / −1.84 | 63.7 / 63.6 / 63.3 | 14.0 / 12.5 / 12.0 | 0.90 / 0.90 / 0.89 | HURTS / neutral / HURTS |
| wd1e-5 / wd5e-4 | matrix weight decay | 74.86 / 73.84 | +0.02 / −1.00 | 63.6 / 63.4 | 13.1 / 13.7 | 0.896 / 0.898 | neutral / HURTS (boundary) |
| k8_clr3 | K=8 + critic lr ×3 | 76.80 | +1.96 | 67.32 | 17.4 | 0.931 | HELPS (= K=8 alone) |
| **k8_clr10** | K=8 + critic lr ×10 | **77.44** | **+2.60** | 68.40 | 21.4 | 0.940 | HELPS (best 200-ep MLP-critic run) |
| k8_aug_strong | K=8 + strong aug | 76.70 | +1.86 | 68.96 | 18.4 | 0.821 | HELPS (= K=8 alone) |
| **k8_clr10_800ep** | K=8 + lr×10, 800 epochs | **79.56** | +0.32 vs P8 (800 ep) | **74.28** | 34.8 | 0.964 | neutral vs the 800-ep K=1 run in linear; +2.9 kNN |

800-epoch kNN trajectories (ep 20/50/100/200/400/600/800): k8_clr10 57.4/64.1/67.0/70.2/72.1/74.0/74.3 vs K=1 47.4/55.5/60.6/65.3/69.0/71.2/71.4.

## Reading
- **Augmentation is asymmetric**: weaker augmentation destroys learning (J → 0.975 while kNN falls to 55: the pair task becomes trivial), but
  harder positives (crop 0.08, blur, strong) lower J to 0.79–0.83 *without* improving features.  Saturation fraction alone is therefore not
  the mechanism; the default recipe (crop 0.2, jitter 0.4, no blur) is at the optimum of the tested range.
- **Batch size** 256 is optimal in {128, 512, 1024}; **matrix weight decay** 1e-4 is optimal in {1e-5, 5e-4}.
- **Combinations**: K=8 and critic lr ×10 are super-additive at 200 epochs (+1.3 and +0.2 alone → +2.6 together); ×3 adds nothing over K=8;
  strong augmentation adds nothing over K=8.
- **800 epochs**: the combination reaches 79.56 % — only +0.3 over the plain K=1 800-epoch run despite +2.9 kNN and rank 35 vs 20.  Under
  the concat-MLP critic the linear-probe ceiling is ≈ 79–80 % regardless of K / critic lr; the extra rank does not translate into linearly
  separable class information.  This is the same pattern as critic lr ×10 at 200 epochs and is superseded by the P19 finding that the
  critic *form* was the limiter (cosine critic: rank 58, 78.3 % at 200 epochs).

## Consequence
Freeze augmentation (default), B = 256, wd 1e-4.  Carry K = 8 forward.  Critic lr ×10 helped only with the MLP critic and is being
re-tested on the cosine base (P24).  The 800-epoch estimate for the cosine base is running (`P24_vcs_cos_800ep_seed0`).
