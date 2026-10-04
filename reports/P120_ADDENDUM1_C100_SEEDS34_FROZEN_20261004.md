# P120 addendum 1 — CIFAR-100 A-P3 vs matched JS, fresh seeds 3 and 4 — FROZEN 2026-10-04T06:42:47Z

**Trigger (disclosed as a post-hoc decision to extend, made after the 3-seed result and before any seed-3/4 run):** P120 JS seeds 0–2 gave
A-P3 − JS = +0.80 [−0.59, +2.19] linear (+1.44 / +0.40 / +0.56, all seeds positive) and +0.51 [−0.80, +1.83] kNN — frozen label **inconclusive at 3
seeds** (P120 pre-stated no follow-up for this case).  On CIFAR-10 the same contrast was **close** (P114, +0.21).  The A-P3 vs SimCLR question was
settled the same way (P107 addendum 2: fresh seeds 3, 4).  Owner standing instruction: submit follow-ups early, with the rule written before the data.

## Units (4; `configs/make_p120_addendum1_configs.py`, `configs/P120_ADDENDUM1_SHA256.json`)
A-P3 CIFAR-100 seeds 3, 4 (`P120A1_AP3_c100_views4_800ep_seed{3,4}`) and JS-AP3 CIFAR-100 seeds 3, 4 (`P120_JS_AP3_c100_views4_800ep_seed{3,4}`) = the
seed-0 configs with only run.seed (and run.stage) changed — the P107 addendum 2 construction.  RTX6000PRO / H100 / L40S; normal QOS.

## Pre-stated reading (5 seeds, paired by seed)
- A-P3 − JS linear (primary) and kNN, seeds 0–4: mean, sd, 95 % t interval.
- Labels: **close** |mean| < 0.3; **clear** |mean| ≥ 0.3 and the 95 % interval excludes 0 ("holds at 5 seeds"); otherwise **inconclusive at 5 seeds**
  (reported as such, no direction claimed, no further seeds).
- If clear: the P120 / P114 rule applies — a pre-specified small lr / optimiser-budget sensitivity check for both losses (a separate unit, designed then)
  before any general statement that the quadratic objective contributes on CIFAR-100.  Either way, the CIFAR-10 result (P114 close) is reported next to
  it; no pooling across datasets.
- Seeds 3–4 are reported on their own as well (A-P3 C100 seed sd check against seeds 0–2).
