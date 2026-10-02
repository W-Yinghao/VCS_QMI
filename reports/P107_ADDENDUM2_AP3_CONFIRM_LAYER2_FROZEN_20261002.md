# P107 addendum 2 — A-P3 passes layer 1; confirmation layer 2 (seeds 3–4, strong augmentation, CIFAR-100) — FROZEN 2026-10-02T14:56:38Z

**Layer 1 by the frozen addendum-1 rule (linear-val, seeds 0–2, paired with P41 SimCLR):** A-P3 (fixed (2, −1), all-view tokens, full negative gradient)
89.06 / 88.96 / 89.06 = **89.03 ± 0.06**; Δ vs SimCLR +0.86 / +0.86 / +0.40, mean **+0.71**, one-sided 90 % lower bound **+0.42** → **stably positive**.
(kNN: 87.38 ± 0.09, Δ vs SimCLR +0.10 / −0.28 / −0.62, mean −0.27 — not stably positive; reported, the rule is on linear-val.)  Δ vs G2 +0.33 (lower
bound +0.05).  A-P2: 88.40 ± 0.51, Δ vs SimCLR +0.08 (lower bound −0.68) → not stably positive; no further A-P2 seeds.  Controls A-P2F / A-P3F still
running; they are read when complete (they decide the mechanism statement, not the layer-2 launch).

**Layer 2 submitted now (8 units; owner 2026-10-02: submit early):**
1. Pre-fixed new seeds 3, 4: A-P3 (SimCLR seeds 3, 4 exist from P111: 88.28 / 88.30).  Pooled with seeds 0–2 → 5 pairs; "holds at 5 seeds" if the
   paired 95 % t interval excludes 0 (as P111).
2. Strong augmentation (P89 block), seeds 0–2: vs P89 SimCLR strong (89.49 ± 0.36) and P111 G2 strong (87.45); Δ_int as in P90 / P111.
3. CIFAR-100 (P91 8×), seeds 0–2: vs P91 VCS (59.91) / SimCLR (58.25 linear, 57.21 kNN) and P111 G2 (59.93); P98 probe caveat — an ordering vs SimCLR
   is stated only if linear and kNN agree.
Configs: `configs/make_p107_confirm2_configs.py` (A-P3 from `make_p107_configs.apply`, strong block from `make_p111_configs.strong`); each differs from
its reference only as intended (verified by diff).  Stage `P107_confirm2`.  No hyper-parameter changes from here.
Note (P111): G2's gain did not survive strong augmentation or CIFAR-100; the same may hold for A-P3 — that outcome is reported as is.
