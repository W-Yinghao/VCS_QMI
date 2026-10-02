# Pre-registration — P112: fixed-scale sensitivity of G2 across augmentation and dataset (v4 §A3 item 1), ten seed-0 800-epoch units — FROZEN 2026-10-02T10:03:30Z

Owner 2026-10-02: "再排十个任务到队列中".  Basis: v4 §A3 item 1 ("固定刻度的小范围敏感性 … 只有在 G2 对 G2F 的结论明确后 … 少数单因素完整训练"); the G2 vs
G2F conclusion is clear (P104 confirmation: +2.33 linear on every seed).  Motivation from P111 (extra by its rule, reported descriptively): G2 loses
1.25 linear under the P89 strong augmentation (all seeds negative; SimCLR and the recipe gain) and is 0.54 below the recipe on CIFAR-100 (seed 0).
Question: **does the best fixed (a, κ) move with augmentation strength and dataset?**  (κ = −b/a, the zero-score cosine threshold; G2 = (2, 0.5).)

## Cells (stage P112_fixed_scale_sensitivity; `configs/make_p112_configs.py`; each = the G2 config of that setting with only (a, b) changed — verified by diff)
| setting | parent (G2, seed 0) | new (a, κ) | reference G2 (2, 0.5) seed 0 |
|---|---|---|---|
| CIFAR-10 standard aug | `cifar10_hpX_G2_views4_800ep_seed0` | (1.5, 0.5), (3, 0.5), (2, 0.25), (2, 0.75) | 88.66 / 86.96 |
| CIFAR-10 strong aug (P89 block) | `cifar10_hpZ_G2_augstrong_views4_800ep_seed0` | (2, 0.25), (2, 0.75), (3, 0.5) | 86.86 / 84.90 |
| CIFAR-100 (P91 8×) | `cifar100_hpZ_G2_c100_views4_800ep_seed0` | (2, 0.25), (2, 0.75), (3, 0.5) | 59.46 / 54.12 |

## Pre-stated reading (descriptive screen; one seed per cell)
- Per setting: linear-val and kNN of every (a, κ) next to the setting's G2 seed-0 reference; differences reported, no verdict on one seed.
- "The best fixed scale shifts" is stated only if the best cell of the strong-aug or CIFAR-100 setting is a different (a, κ) than in the standard
  setting **and** beats that setting's G2 reference by ≥ 0.5 linear (≈ 2.5 × G2's 5-seed sd 0.15–0.20); otherwise "no clear shift at one seed".
- Follow-up (separate addendum, submitted immediately to queue early — owner 2026-10-02): per setting, a cell beating its G2 reference by ≥ 0.5 linear
  gets seeds 1–2; G2 (2, 0.5) in that setting has seeds 0–2 already (P104 / P111) for the paired comparison.
- No early stopping on J / rank / kNN; only non-finite values or infrastructure failures stop a run.  Official test set closed.

## Compute
10 × 800 epochs ≈ 4.4 h each on RTX6000PRO (≈ 44 GPU-h); normal QOS, RTX6000PRO / H100 (owner: no A100 for these).
