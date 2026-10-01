# Pre-registration — P111: G2 confirmation layer 2 (fresh seeds 3–4 paired with SimCLR, strong augmentation, CIFAR-100), ten 800-epoch units — FROZEN 2026-10-01T23:03:10Z

Owner 2026-10-02: "我看到只有一个在排队了，再提交十个任务".  Basis: package v3 §10.2 (layer 2: seeds 3/4 for a stably positive candidate with the control on
the same seeds; identical strong-augmentation block on both sides; CIFAR-100 after the frozen configuration) and v4 A0 / A4 (pre-fixed new seeds 3–4).
Configs: `configs/make_p111_configs.py` (G2 scorer from `make_p104_configs.apply(…, "G2")`), `configs/P111_SHA256.json`, `slurm/p111_lines.txt`.

## State at the freeze
G2 (fixed a, b = 2, −1) linear-val: seed 0 88.66, seed 1 88.92 (seed 2 running); same-seed SimCLR (P41) 88.20 / 88.10; same-init learned control G2F
86.70 / 86.08 / 86.32.  Layer 1 (G2 seeds 0–2 vs SimCLR seeds 0–2) is not complete.  **These ten runs are submitted before layer 1 completes, to use
idle quota (owner request).  If G2 is not stably positive in layer 1 (G2 seed-2 result included; rule below), the P111 runs are reported as extra runs
outside any candidate-vs-SimCLR claim.**  No hyper-parameter is changed after seeing any of these seeds.

## Cells (stage P111_G2_confirm2; each differs from its reference only as listed — verified by diff)
1. **Fresh seeds:** G2 seeds 3, 4 (vs P104 G2 seed 0: run.seed only) and tuned SimCLR seeds 3, 4 (vs P41 SimCLR seed 0: run.seed only).
2. **Strong augmentation:** G2 seeds 0, 1, 2 with the P89 strong block (random-resized-crop scale min 0.08; colour jitter 0.8 / 0.8 / 0.8 / 0.2) — vs P89
   SimCLR strong seeds 0–2 (89.49 ± 0.36) and P89 VCS-recipe strong (87.66 ± 0.12), same block.
3. **CIFAR-100:** G2 seeds 0, 1, 2 on the P91 CIFAR-100 8× recipe (only the G2 scorer + epoch-20 checkpoint differ) — vs P91 8× VCS 59.91 ± 0.20 and
   SimCLR 58.25 ± 0.35 linear (kNN 54.95 / 57.21).  Readout as P91/P98: linear and kNN side by side, the P98 robust-probe caveat applies.

## Pre-stated reading (descriptive; per-seed values, paired differences, mean ± sd, interval; no 1-point threshold, no seed-0 extreme called a win)
- **Layer 1 "stably positive"** (decides whether P111 is read as confirmation): G2 − SimCLR > 0 on all three paired seeds 0–2 *and* mean paired
  difference > 0 with a one-sided 90 % t interval above 0; otherwise P111 is "extra".
- **Seeds 3–4:** paired G2 − SimCLR per seed; pooled with seeds 0–2 (5 pairs): mean, sd, 95 % t interval.  "Holds at 5 seeds" if the interval excludes 0.
- **Strong augmentation:** G2-strong vs SimCLR-strong per seed (paired 0–2), and the augmentation interaction Δ_int = (G2strong − G2std) − (SimCLRstrong −
  SimCLRstd) as in P90; reported, not thresholded.
- **CIFAR-100:** G2 vs P91 VCS and SimCLR (3 seeds each), linear and kNN; under P98's finding (8× VCS > SimCLR depends on the probe), any ordering
  statement needs both readouts to agree.
- Not claimed: ImageNet, other backbones, the official test set (closed).

## Compute
10 × 800 epochs ≈ 4.4 h each on RTX6000PRO (CIFAR-100 same cost) ≈ 45 GPU-h; normal QOS, RTX6000PRO / H100.
