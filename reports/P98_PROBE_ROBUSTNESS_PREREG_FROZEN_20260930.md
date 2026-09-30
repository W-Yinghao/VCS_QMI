# Pre-registration — P98: linear-probe robustness of the P91 CIFAR-100 readings (evaluation only), 2026-09-30 — FROZEN 2026-09-30T19:58:11Z before GPU compute (CPU gate 1015840: 6 tests pass, smokes end to end)

Status: FROZEN 2026-09-30T19:58:11Z (main session).  The main session freezes it (renamed `*_FROZEN_*`, freeze time in the title) before the GPU job.  Owner go 2026-09-30
("除了imagenet的先不提交，后续都可以提交").  Source: `P92_CIFAR100_REPORT_20260930.md` §3 — the frozen recipe probe ends at a training CE of 1.12 for
SimCLR 8× on CIFAR-100 (0.86–0.89 for SimCLR 2× and VCS; VICReg 1.18–1.21), so the P92 linear readings (a) ordering and (b) scaling may be probe-limited.

## Question
Do the P92 linear-probe conclusions on CIFAR-100 survive two pre-stated probe variants that remove the suspected underfitting?

## Units (no training; checkpoints and feature caches exist)
All 18 P91 checkpoints (`P91_c100_{vcs_a5,simclr,vicreg}_views4_{200ep,800ep}_seed{0,1,2}`, final checkpoints epoch_200 / epoch_800) and, as a reference,
the CIFAR-10 8× cells (P35 VCS seeds 0–2, P41 SimCLR and VICReg 4-view 800-epoch seeds 0–2) — 27 checkpoints.

## Probes (identical for every checkpoint; features from the run's own `vcs_ssl.evaluate` pilot cache, re-extracted with the same function on a miss)
- **frozen:** the recipe probe exactly (SGD 0.1, momentum 0.9, cosine, 100 epochs, batch 256, un-normalised h); must reproduce the stored pilot value
  (`frozen_reproduces_stored`, |Δ| < 0.05).
- **l2:** the same probe on row-L2-normalised h (fit and selection).
- **tuned:** the recipe probe with 300 epochs and lr ∈ {0.1, 0.3, 1.0} chosen by top-1 on a seeded 10 % hold-out of the FIT features (probe trained on the
  other 90 %; ties → smaller lr), then refitted on all FIT features and scored once on the selection split.
The selection split is only ever scored, never used for a choice; the official CIFAR test split is not opened.

## Readouts
Per checkpoint and probe: selection top-1, selection CE, final probe train CE (and the tuned lr).  P91 cells: mean ± sd over 3 seeds per probe.

## Reading (pre-stated)
- The P92 quantities re-read under each probe: SimCLR − VCS, VCS − VICReg, SimCLR − VICReg at 2× and 8×; gain(VCS) − gain(SimCLR) and gain(VCS) − gain(VICReg).
- **Robust** if every pairwise difference with |Δ| ≥ 1.0 under any probe has the same sign under all three probes (|Δ| < 1.0 counts as a tie).
  A quantity whose sign flips across probes is reported "probe-dependent"; P92's wording for it must then carry the kNN comparison as the primary number.
- The CIFAR-10 8× cells are reference only (does a probe change the CIFAR-10 ordering?), not part of the verdict.

## Code
`scripts/probe_robustness.py` (new; reuses `vcs_ssl.evaluate.feature_cache_key` / `cached_features`, `vcs_ssl.diagnostics.linear_probe`); tests in
`tests/test_p98_p101.py` (the FIT hold-out is FIT-internal and deterministic).  Gate: CPU job 1015840 — tests 6/6 pass; `--smoke --cpu` on `P91_c100_simclr_views4_800ep_seed0` and `P35_vcs_a5_views4_800ep_seed0` (feature cache hits, 2-epoch probes, sub-sampled rows) ran all three probes and wrote JSON + table (values meaningless at smoke budget; the frozen-probe reproduction flag is only evaluated in the full run).

## Cost
Features cached (hit) → per checkpoint ≈ 14 s (frozen) + 14 s (l2) + 3 × 38 s (lr scan on 90 %) + 42 s (tuned refit) ≈ 3 min on a GPU; 27 checkpoints ≈ 1.5 h.
One job, `slurm/p98_probe.sbatch` (normal QOS; RTX6000PRO,H100,A100,L40S; excl. node51,node60; 4 h wall).

## Delivery (Spec v2 §13.2)
```yaml
experiment_family: full_ssl (evaluation-only addendum)
protocol_id: P98_probe_robustness
estimator: frozen-h linear probe variants
evaluation_readout: selection top-1 / CE / probe train CE
n_independent_units: 27 checkpoints (18 CIFAR-100 + 9 CIFAR-10 reference)
split_manifest_hash: cifar100_dev45k_val5k (96dbd6d4...), cifar10_dev45k_val5k (f819026a...)
status: <status>
```
