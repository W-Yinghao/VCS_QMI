# Pre-registration — P99: S2b continuation (P77 unit resumed and completed), 2026-09-30 — FROZEN 2026-09-30T19:40:45Z before GPU compute

Owner go 2026-09-30 ("除了imagenet的先不提交，后续都可以提交") after P78 reported that P77's probe gate failed on its J clause (J 0.519 < 0.70 at epoch 20;
a, b in range) and stopped the unit at epoch ≈ 276.  This is a new unit, not a re-reading of P77: the gate is not re-used and nothing about P77 is re-tuned.

**Units.**  The frozen P77 configuration (`s2b_vcs.yaml`, method vcs_s2b, files in `s2_solo_learn/s2b/`), seeds 0, 1, 2, 1000 epochs.  Seed 0 resumes from
its epoch-276 checkpoint (solo-learn auto_resume); seeds 1–2 start from scratch.  Normal QOS, H100 / RTX6000PRO first.

**Metric and test-set rule.**  As P75 / P77: online linear Acc@1 on the official CIFAR-10 test split at the end of epoch 1000, read once per run (the owner's S2
exemption; same protocol, same single read).

**Reading (pre-stated).**  gap_b = mean SimCLR (P75: 91.24) − mean VCS-S2b (3 seeds).  Holds if gap_b < 1.5; refuted if ≥ 1.5; "at the threshold" if
|gap_b − 1.5| < 2 × pooled SE.  Reported next to P76's gap (3.91).  Disclosure carried into the report: the P77 gate failed and seed 0's epochs 0–276 were
already observed (P78).

**Cost.**  ≈ 11 s / epoch on H100 → ≈ 3 h per seed (seed 0: ≈ 2.2 h remaining).
