# Pre-registration — P146: v7 §7.1 direct-CS route-transfer controls on CIFAR-100 (S-Kernel and CS-K-native, P87 / P88 designs), seed 0 — FROZEN 2026-10-07T04:15:42Z

Owner 2026-10-05: "VCS_SSL_Server_Plan_v7 … 把这批次的分析也开始做" — plan v7 §7.1: CIFAR-100 has no same-protocol direct-CS training, so two separate
route-transfer controls are registered after the B batch (P137, reported): the RFF critic on the same J and the classical kernel CS objective, one
seed-0 800-epoch run each, strictly following their P87 / P88 pairing and gradient designs, compared with the original neural recipe; A-P3 listed
separately as the current complete method.  Queue below the owner's floor of 15 at the time of registration.

## 1. Configs (`configs/make_p146_configs.py`, `configs/P146_SHA256.json`, `slurm/p146_lines.txt`)
Each = the CIFAR-100 recipe-VCS base (P91 `cifar100_hpS_vcs_a5_views4_800ep_seed0.yaml`; differs from the CIFAR-10 recipe base only in the data
fields) + exactly the field changes that turn the CIFAR-10 recipe base into the P87 8× config of that route (computed, recorded in the sha file):
- **S-Kernel** (`P146_skernel_c100_views4_800ep_seed0`): `model.critic.input` cosine → rff_tanh, `rff_features` 4096, `rff_bandwidth_multiple` 0.5 —
  the same J, K = 8 cyclic shifts, detach of the shifted partner, 6 view pairs, AdamW 1e-3, warm-up 10, 800 epochs, B 256.
- **CS-K-native** (`P146_kcs_c100_views4_800ep_seed0`): method cs_kernel_native, objective negative_kernel_cs / classical_kernel_CS_QMI,
  `kernel_cs_bandwidth_multiple` 0.5, chunk 0, pairing k 1, no negative detach, critic off.
Bandwidth: the original scheme's FIT-side rule — median pairwise distance of the pair vectors (S-Kernel) / of z (CS-K-native) at step 0 ×
the multiple.  The multiples (0.5; m 4096) are the P87 CIFAR-10 dev-grid selections, **transferred, not re-screened on CIFAR-100** (disclosed); a
negative result is therefore not read as "all kernel estimators are worse" (plan §7.1).

## 2. Reading (descriptive, single seed)
Comparator for each route: the CIFAR-100 recipe VCS (P91 VCS-a5 seed 0) — same pairing / gradient design; A-P3 (60.20 / 55.92) listed separately.
Δ linear and Δ kNN vs recipe VCS; with P88's CIFAR-10 thresholds as description: |Δ linear| ≥ 1.0 → direction stated, else parity.  Also h
effective rank (P88: both kernel routes collapsed h dimensionally on CIFAR-10), D_CS / J trajectories, `kcs_underflow_frac`, failures.  A
numerically failing run (non-finite, collapse flag at two consecutive kNN epochs, underflow > 0.5 sustained) is recorded as the result, not re-run.

## 3. Not claimed
Tuned kernel bandwidths on CIFAR-100; other kernels or feature counts; statements about kernel CS estimators in general.

## Decisions at the freeze (main session)
- CPU smoke job 1025735: both routes load under the current policy and complete 3 optimizer steps on CIFAR-100, including the step-0 FIT-side
  bandwidth calibration (`bandwidth_calibration.json`); no code change was needed (both routes exist since P87).
- Single seed per route (plan: "各一个 seed-0、800-epoch"); no dev grid on CIFAR-100 (the transferred multiples are disclosed in §1).
- Launch: `slurm/p146_lines.txt` (normal QOS; RTX6000PRO / H100 / L40S; node51 / 52 / 60 excluded).
