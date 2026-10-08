# status_v7 — V7-CORE (P140): executive summary, task map, incidents

Generated 2026-10-08T07:37:15Z by `scripts/p140_v7_tables.py` from raw per-run files only (197 finished runs in table A; details in `results_v7.json`, `paired_comparisons_v7.json`, `runtime_v7.json`).  Development selection split; official test closed.  Seed-0 selection runs and confirmation seeds are kept apart; different scorer / lr runs are never pooled as seeds.

## 1. Executive summary

- **Shared configuration, VCS − matched JS** (only the loss differs): CIFAR-10 3 seeds 89.03 vs 88.81: Δ +0.21 [-1.03, +1.46] (close; n=3); kNN Δ -0.29 (close); CIFAR-100 5 seeds 60.15 vs 59.35: Δ +0.80 [+0.06, +1.54] (clear; n=5); kNN Δ +0.60 (clear).
- **Same contrast at other learning rates (seed 0):** c10 lr0.5x +0.44; c10 lr2x +0.08; c100 lr0.5x +1.08; c100 lr2x +1.16 (lr 1e-3 seed 0: c10 +0.34, c100 +1.44) — the sign of VCS − JS matches lr 1e-3 in every cell.
- **Selected-vs-selected cifar10** (VCS (2,0.5) vs JS (2,0.25); each loss with its own rule-selected scorer — not a single-factor contrast): 89.03 vs 88.97: Δ +0.05 [-0.65, +0.76] (close; n=3); kNN Δ -0.01 (close).
- **Selected-vs-selected cifar100** (VCS (2,0.5) vs JS (3,0.5); each loss with its own rule-selected scorer — not a single-factor contrast): 60.00 vs 60.49: Δ -0.49 [-0.77, -0.20] (clear; n=3); kNN Δ -0.51 (inconclusive).
- Grid VCS_cifar10_(a,kappa): default 89.06; highest tested (2,0.75) 89.30 (+0.24); rule-selected (2,0.5); full range 87.28–89.30.
- Grid VCS_cifar100_(a,kappa): default 60.20; highest tested (3,0.5) 60.64 (+0.44); rule-selected (2,0.5); full range 58.34–60.64.
- Grid JS_cifar10_(a,kappa): default 88.72; highest tested (2,0.25) 89.26 (+0.54); rule-selected (2,0.25); full range 88.28–89.26.
- Grid JS_cifar100_(a,kappa): default 58.76; highest tested (3,0.5) 60.72 (+1.96); rule-selected (3,0.5); full range 58.24–60.72.
- Grid VCS_cifar100_lr: default 60.20; highest tested 2e-3 60.52 (+0.32); rule-selected 1e-3; full range 60.20–60.52.
- Grid JS_cifar100_lr: default 58.76; highest tested 2e-3 59.36 (+0.60); rule-selected 2e-3; full range 58.76–59.36.

## 2. Task map (P129, P130, P133, P135, P136, P137)

**P129** — completed 18; running / resumable 0; in queue 0.
**P130** — completed 3; running / resumable 5; in queue 11.
  - running/resumable: P130_AP3_c100_r50_views4_800ep_seed1 (RUNNING, epoch 261)
  - running/resumable: P130_AP3_c100_r50_views4_800ep_seed2 (RUNNING, epoch 160)
  - running/resumable: P130_JS_AP3_c100_r50_views4_800ep_seed1 (RUNNING, epoch 151)
  - running/resumable: P130_JS_AP3_c100_r50_views4_800ep_seed2 (RUNNING, epoch 11)
  - running/resumable: P130_simclr_c100_r50_views4_800ep_seed1 (RUNNING, epoch 2)
  - queue: 1027285 p130a1_js_s2 RUNNING H100 node53 None
  - queue: 1027529 p130a1_simclr_s1 RUNNING H100 nodesumo01 None
  - queue: 1027193 p130a1_ap3_s2 RUNNING RTX6000PRO node59 None
  - queue: 1027253 p130a1_js_s1 RUNNING RTX6000PRO node61 None
  - queue: 1027191 p130a1_ap3_s1 RUNNING RTX6000PRO node58 None
  - queue: 1027724 p130a1_simclr_s2 PENDING RTX6000PRO,H100,L40S  QOSMaxGRESPerUser
  - queue: 1028242 p130a1_simclr_s1 PENDING RTX6000PRO,H100,L40S  Dependency
  - queue: 1028226 p130a1_js_s2 PENDING RTX6000PRO,H100,L40S  Dependency
  - queue: 1028174 p130a1_js_s1 PENDING RTX6000PRO,H100,L40S  Dependency
  - queue: 1028169 p130a1_ap3_s2 PENDING RTX6000PRO,H100,L40S  Dependency
  - queue: 1028140 p130a1_ap3_s1 PENDING RTX6000PRO,H100,L40S  Dependency
**P133** — completed 2; running / resumable 0; in queue 0.
**P135** — completed 10; running / resumable 0; in queue 0.
**P136** — completed 6; running / resumable 0; in queue 0.
**P137** — completed 6; running / resumable 0; in queue 0.

## 3. Infrastructure incidents (kept separate from algorithm results)

- **CUDA device not visible** (node hand-out without a usable GPU): 16 job logs on host(s) node52, ends 2026-10-05T12:04:58Z … 2026-10-05T12:06:34Z; all jobs exited before creating a run directory and were resubmitted with node52 excluded (default exclusion now node51, node52, node60).
- **Throttled GPUs** (nvidia-smi throttle reason 0x88, ~255 W vs ~450 W): node60 GPU 0 and 1, node61 GPU 1 — ≈ 44 s / epoch vs 20 s on healthy RTX6000PRO (ResNet-18, 4 views, B 256); node60 excluded for new starts.
- **GPU moves (requeue + epoch-boundary resume from last.pt):** 7 logged requeues: 1020331 P126_CV_AP3_c10_lamp025_views4_800ep_seed0 from node61:gpu1; 1020330 P126_CV_AP3_c10_lamm025_views4_800ep_seed0 from node60:gpu0; 1020405 P127_AP3_c10_a3_k0.5_seed0 from node61:gpu1; 1020332 P126_CV_AP3_c100_lamm025_views4_800ep_seed0 from node61:gpu1; 1020404 P127_AP3_c10_a2_k0.75_seed0 from node50; 1020580 P120A1_AP3_c100_views4_800ep_seed3 from node39; 1020749 P120_JS_AP3_c100_views4_800ep_seed3 from node50.  Training results of moved runs are kept (resume restores model, optimizer, RNG at an epoch boundary); their wall time spans several GPU types.
- Mover incidents (2026-10-04): an all-node exclusion made SLURM flag pending jobs `BadConstraints` (reverted, no age loss); an unpinned rotation let a waiting unit restart on the vacated slow GPU; the automatic mover is off — manual pinned moves only.  RTX node58 / 59 / 61 maintenance drain from 2026-10-05 13:16 local.
- **Algorithm / run failures (status FAILED):** none.  Gate failures by design: P133 queue variants (constant-map J > 0) — not trained.

## 4. Cross-checks of numbers quoted in earlier reports (recomputed from raw files)

- A-P3_C10_5seeds (P107 layer-2: 89.02 ± 0.04): n=5, linear mean 89.024, sample sd 0.0434, max−min 0.1; kNN mean 87.408
- SimCLR_C10_5seeds (P107 layer-2: 88.31 ± 0.21): n=5, linear mean 88.308, sample sd 0.2119, max−min 0.56; kNN mean 87.86
- A-P3_C100_5seeds (P120 A1: 60.15 ± 0.38): n=5, linear mean 60.152, sample sd 0.383, max−min 1.04; kNN mean 55.9
- JS_C100_5seeds (P120 A1: 59.35 ± 0.42): n=5, linear mean 59.348, sample sd 0.4228, max−min 1.06; kNN mean 55.296
- JS_C10_(2,0.25)_3seeds (P129 A1: 88.97, 'spread' 0.66): n=3, linear mean 88.9733, sample sd 0.3384, max−min 0.66; kNN mean 87.3933
- A-P3_C10_3seeds (P129 A1: 89.03, 'spread' 0.10): n=3, linear mean 89.0267, sample sd 0.0577, max−min 0.1; kNN mean 87.38
- SimCLR_C100_3seeds (P91: 58.25 / 57.21): n=3, linear mean 58.2533, sample sd 0.3535, max−min 0.64; kNN mean 57.2067
- FREE_C100_3seeds (P120: 56.74): n=3, linear mean 56.74, sample sd 0.2821, max−min 0.56; kNN mean 49.5933

## 5. Table B detail (same configuration, VCS − JS)

| contrast | linear | config keys differing besides the loss |
|---|---|---|
| P114_cifar10_shared_(2,0.5)_lr1e-3 | 89.03 vs 88.81: Δ +0.21 [-1.03, +1.46] (close; n=3); kNN Δ -0.29 (close) | none |
| P120+A1_cifar100_shared_(2,0.5)_lr1e-3 | 60.15 vs 59.35: Δ +0.80 [+0.06, +1.54] (clear; n=5); kNN Δ +0.60 (clear) | none |
| P135_c10_lr0.5x_(2,0.5) | 88.72 vs 88.28: Δ +0.44 (single seed (no interval); n=1); kNN Δ -0.12 (single seed (no interval)) | none |
| P135_c10_lr2x_(2,0.5) | 88.94 vs 88.86: Δ +0.08 (single seed (no interval); n=1); kNN Δ +0.28 (single seed (no interval)) | none |
| P127vsP129_c10_(1.5,0.5)_lr1e-3 | 88.98 vs 88.50: Δ +0.48 (single seed (no interval); n=1); kNN Δ +0.16 (single seed (no interval)) | none |
| P127vsP129_c10_(2,0.25)_lr1e-3 | 88.36 vs 89.26: Δ -0.90 (single seed (no interval); n=1); kNN Δ -1.30 (single seed (no interval)) | none |
| P127vsP129_c10_(2,0.75)_lr1e-3 | 89.30 vs 88.52: Δ +0.78 (single seed (no interval); n=1); kNN Δ +0.18 (single seed (no interval)) | none |
| P127vsP129_c10_(3,0.5)_lr1e-3 | 88.10 vs 88.60: Δ -0.50 (single seed (no interval); n=1); kNN Δ -0.48 (single seed (no interval)) | none |
| P127vsP129_c10_(3,0.25)_lr1e-3 | 87.28 vs 88.28: Δ -1.00 (single seed (no interval); n=1); kNN Δ -1.90 (single seed (no interval)) | none |
| P135_c100_lr0.5x_(2,0.5) | 60.38 vs 59.30: Δ +1.08 (single seed (no interval); n=1); kNN Δ +0.54 (single seed (no interval)) | none |
| P135_c100_lr2x_(2,0.5) | 60.52 vs 59.36: Δ +1.16 (single seed (no interval); n=1); kNN Δ +0.78 (single seed (no interval)) | none |
| P127vsP129_c100_(1.5,0.5)_lr1e-3 | 58.92 vs 58.86: Δ +0.06 (single seed (no interval); n=1); kNN Δ +0.44 (single seed (no interval)) | none |
| P127vsP129_c100_(2,0.25)_lr1e-3 | 60.30 vs 59.58: Δ +0.72 (single seed (no interval); n=1); kNN Δ -1.18 (single seed (no interval)) | none |
| P127vsP129_c100_(2,0.75)_lr1e-3 | 58.34 vs 58.24: Δ +0.10 (single seed (no interval); n=1); kNN Δ +0.90 (single seed (no interval)) | none |
| P127vsP129_c100_(3,0.5)_lr1e-3 | 60.64 vs 60.72: Δ -0.08 (single seed (no interval); n=1); kNN Δ -0.48 (single seed (no interval)) | none |
| P127vsP129_c100_(3,0.25)_lr1e-3 | 60.18 vs 60.24: Δ -0.06 (single seed (no interval); n=1); kNN Δ -1.70 (single seed (no interval)) | none |
