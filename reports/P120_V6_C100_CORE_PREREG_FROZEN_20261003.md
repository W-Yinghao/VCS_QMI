# Pre-registration — P120: v6 V6-C100-CORE — matched JS and same-initialisation learned scale for A-P3 on CIFAR-100, seeds 0–2 — FROZEN 2026-10-03T19:15:48Z

Owner 2026-10-03: "这是新的一版分析和设计，你来理解，并组织实验提交" (package `VCS_Server_Tasks_and_Theory_v6_20261003.zip`, §6).  Deduplicated: no C100
JS / learned-scale A-P3 run exists.  Configs `configs/make_p120_configs.py`, `configs/P120_SHA256.json`; each differs from the completed CIFAR-100 A-P3
config of the same seed (`cifar100_hpY_AP3_c100_views4_800ep_seed{s}.yaml`, P107 addendum 2) only as listed (verified by diff):
- **V6-C100-JS** (3 units): objective → matched JS on the same logits (L = mean_P softplus(−2f) + mean_Q softplus(2f)), fixed f = 2s − 1, all-view tokens,
  full gradient — the P114 path (markers js_fixed_scorer, js_all_view_tokens); FixedCosineCritic, no trainable affine.
- **V6-C100-FREE** (3 units): critic affine_mode fixed → learned, a, b start at (2, −1) (as P107 A-P3F on CIFAR-10); pairing and routing unchanged.
ResNet-18 CIFAR, 4 views, B 256, 800 epochs, same manifest / augmentation / projector / schedule.

## Pre-stated reading (fixed-setting attribution, not a tuned-vs-tuned comparison — v6 §6)
- Primary endpoint: frozen-h linear (selection split); kNN reported as its own column; no switching of the primary readout after results.
- Paired by seed vs A-P3 C100 (P107: 60.20 / 60.08 / 59.72 linear; 55.92 / 55.96 / 55.52 kNN): per-seed differences, mean, sd, 95 % t interval.
- Labels (as P114): **close** |mean| < 0.3; **clear difference** |mean| ≥ 0.3 and the interval excludes 0; **inconclusive at 3 seeds** otherwise.
- FREE vs A-P3 decides whether the fixed scale is needed on CIFAR-100 too (on CIFAR-10 A-P3F was −6.11); JS vs A-P3 decides whether the quadratic
  objective contributes beyond the scorer structure on CIFAR-100.  A clear JS-vs-VCS difference requires an lr / budget sensitivity check before any
  general statement.  Early diagnostics only flag non-finite / implementation failures; no early stopping.
## Compute
6 × 800 epochs (≈ 4.4 h each on an unshared RTX6000PRO); normal QOS, RTX6000PRO / H100, node60 allowed.
