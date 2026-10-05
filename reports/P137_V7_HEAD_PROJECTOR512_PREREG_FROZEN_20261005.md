# Pre-registration — P137: v7 V7-HEAD — projector output dimension 128 → 512 for VCS (A-P3), matched JS and SimCLR, CIFAR-10 / CIFAR-100, seed 0 — FROZEN 2026-10-05T20:51:54Z

Owner 2026-10-05: "VCS_SSL_Server_Plan_v7 … 把这批次的分析也开始做" — plan v7 §4 (batch B, the first new performance candidate).
**Question:** does a wider projector output (512→512→512 instead of 512→512→128) let VCS improve fine-grained / neighbourhood readability while keeping
coarse structure (P124: A-P3's C100 gain over the recipe is coarse-specific; P122: h carries pair evidence that z under-expresses)?
**Cells (6):** {VCS A-P3, JS-AP3, SimCLR} × {CIFAR-10, CIFAR-100}, seed 0, 800 epochs; each config = that method's seed-0 parent with ONLY
model.projector.output_dim 128 → 512 plus the marker model.projector.v7_head_width = 512 (verified by diff).  Hidden layer, BN / ReLU, no output BN,
L2 normalisation, (a, κ) = (2, 0.5), lr 1e-3, SimCLR recipe all unchanged.  The 128-dim parents are the reused controls (A-P3 89.06 / 60.20,
JS-AP3 88.72 / 58.76, SimCLR 88.20 / 58.66 seed 0).  Same seed → same encoder init and RNG streams; the projector's last layer has a new shape
(same init rule).  Code: config policy accepts output_dim 512 for every method only with the marker (absent → unchanged; gate job 1023072: 63 / 63
tests incl. config-hash invariance).  `configs/make_p137_configs.py`, `configs/P137_SHA256.json`, `slurm/p137_lines.txt`.

## Pre-stated reading (plan §4.3)
- Primary: final frozen-h linear; kNN, C100 coarse / fine / conditional (P124 protocol, evaluation-only addendum), wall time, peak memory alongside.
- **Trigger:** VCS linear gain vs its 128-dim control ≥ 0.30 (CIFAR-10) / ≥ 0.50 (CIFAR-100) → freeze the candidate and add seeds 1–2 for **all three
  methods** on that dataset (addendum, submitted when the rule triggers).  A gain only in kNN / coarse is recorded as a trade-off, no replacement.
  No gain → the single dimension ablation ends (no 256 / 1024 / depth / new regularisers).
- Attribution uses the interaction Δ_HEAD(VCS) − Δ_HEAD(JS) (and vs SimCLR), never new-VCS vs old-JS.  If 512 becomes the main configuration, the
  SimCLR 512 control stays in the final table.  No early stopping; official test closed.
