# P133 — MoCo-consistent momentum-encoder keys, no queue (VCS A-P3 and InfoNCE), CIFAR-10 seed 0 — report — 2026-10-06

Pre-registration `P133_MOCO_CONSISTENT_NOQUEUE_PREREG_FROZEN_20261005.md`; results-only commit `118fc27` (`P133_noqueue_results.json`).  ResNet-18,
4 views, batch 256, 800 epochs, development split (official test closed).  Keys from a momentum encoder (m = 0.99, eval mode), batch keys only.
Parents: VCS = A-P3 seed 0 (`P107_AP3_views4_800ep_seed0`), InfoNCE = P41 SimCLR 4-view seed 0.

## 1. Results (epoch 800)

| method | P133 linear / kNN | parent linear / kNN | Δ linear | Δ kNN | h eff. rank @800 (parent) | seeds-1–2 trigger |
|---|---|---|---|---|---|---|
| VCS (A-P3 scorer) | 89.04 / 87.74 | 89.06 / 87.30 | −0.02 | +0.44 | 117.8 (121.1) | no |
| InfoNCE (SimCLR) | 87.68 / 87.36 | 88.20 / 87.20 | −0.52 | +0.16 | **13.9 (158.3)** | no (sentinel + Δ < 0.30) |

VCS − InfoNCE at seed 0 (descriptive): linear +1.36 (parents: +0.86), kNN +0.38.  Train time ≈ 5.7 h each on RTX6000PRO (parents ≈ 4.4 h; the
key encoder's extra forward pass).

## 2. Collapse sentinel (frozen rule)
- **VCS: not triggered.**  No collapse flags; h effective rank 117.8 = 0.97 × parent; rank rises monotonically to 128 at epoch 400.  Held-out J
  (online–online, z_l2) 0.853 vs parent 0.860.  Unlike P100's stale-queue variant (VCS 12.58 %, rank 1.0), consistent momentum keys train normally.
- **InfoNCE: triggered by the rank criterion** (h rank 13.9 < 10 % of the parent's 158.3; ratio 0.088), **not** by the collapse flag (never set).
  Trajectory: h rank 158 at epoch 200 (= parent), then 31.5 (400), 16.8 (600), 13.9 (800); top eigenvalue fraction 0.06 → 0.58.  At the same time
  z rank keeps rising (85 → 110) and kNN stays at the parent level (87.36).  Per the frozen rule this is reported as a collapse of this design
  (dimensional collapse of h, after epoch 200) and not re-run with other settings; the accuracy shows it is not a performance collapse.

## 3. Pre-stated reading
No method triggers seeds 1–2 → **"no gain from momentum-encoder keys at seed 0"** for both VCS (Δ −0.02) and InfoNCE (Δ −0.52).  The unit closes.

## 4. Not claimed
Any multi-seed statement; queue variants (both failed the constant-online-map gate and were never launched); other momentum values; ImageNet.
