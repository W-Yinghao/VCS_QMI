# P145 addendum 2 — STRESS ε = 0.20, seed 0, VCS / JS / SimCLR × CIFAR-10 / CIFAR-100 — FROZEN 2026-10-08T11:04:46Z

Budget confirmation (P145 §4 "at most one later ε = 0.20 condition, with separate budget confirmation"): owner 2026-10-08 (selected "STRESS ε = 0.20"
when asked which jobs to submit).  Units (6): `configs/{cifar10,cifar100}_hpST20_{vcs,js,simclr}_{c10,c100}_seed0.yaml` = the ε = 0.10 seed-0 configs
with ONLY pairing.stress_epsilon 0.10 → 0.20 (and run.stage) — verified by diff; policy load passes (`configs/make_p145a2_configs.py`,
`configs/P145A2_SHA256.json`, `slurm/p145a2_lines.txt`).  Code unchanged (the P145 gate covers ε as a parameter).

Reading (P145 §4 unchanged): Δ_m = A_m(0.20) − A_m(0) at seed 0 against the same clean parents (A-P3 89.06 / 60.20, JS-AP3 88.72 / 58.76, SimCLR
88.20 / 58.66).  **Trigger:** |Δ_VCS − Δ_m| ≥ 0.50 (CIFAR-10) or ≥ 1.00 (CIFAR-100) for m ∈ {JS, SimCLR} → seeds 1–2 for all three methods on that
dataset.  Descriptive: the dose curve ε 0 → 0.10 → 0.20 per method.  Caveat carried from addendum 1: the ε = 0.10 seed-0 trigger did not replicate
over seeds 1–2, so a seed-0 gap is read only as a trigger, never as an effect.  No ε above 0.20.
