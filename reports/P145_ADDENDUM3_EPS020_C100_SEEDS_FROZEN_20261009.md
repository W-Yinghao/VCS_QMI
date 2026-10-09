# P145 addendum 3 — CIFAR-100 seeds 1–2 for VCS / JS / SimCLR at STRESS ε = 0.20 — FROZEN 2026-10-09

Trigger (addendum 2 frozen rule): at seed 0 on CIFAR-100, Δ_VCS = −0.34 (59.86 vs 60.20) and Δ_JS = +0.78 (59.54 vs 58.76) → |Δ_VCS − Δ_JS| =
1.12 ≥ 1.00 → seeds 1–2 for **all three methods** on CIFAR-100 (`P145_ADDENDUM2_EPS020_REPORT_20261009.md`).  As planned, the trigger was reported
first; owner 2026-10-09: "Run seeds 1–2".  CIFAR-10 did not trigger.
Units (6): `configs/cifar100_hpST20_{vcs,js,simclr}_c100_seed{1,2}.yaml` = the ε 0.20 seed-0 configs with only run.seed / run.stage changed, and
asserted equal to the ε 0.10 seed-s configs (addendum 1) apart from pairing.stress_epsilon and run.stage (`configs/make_p145a3_configs.py`,
`configs/P145A3_SHA256.json`, `slurm/p145a3_lines.txt`; nice 1000, behind the VL GPU jobs, ahead of P154).  Code unchanged.
Reading = addendum 1's frozen reading (`scripts/p145a3_aggregate.py` = `p145a1_aggregate.py` with the ε 0.20 run names): per method Δ_m paired by
seed 0–2 against the clean P107 / P120 / P91 runs; interaction Δ_VCS − Δ_m paired by seed with a 95 % t interval and the P114 labels (close
|mean| < 0.3; clear |mean| ≥ 0.3 and the interval excludes 0; else inconclusive).  All single-seed values reported; the known low JS seed-0 parent
is stated beside the reading.  No further seeds or ε levels.
