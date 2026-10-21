# KD application: server delta note (2026-10-20)

Package `VCS_QMI_KD_AIGIQA_Server_Package_v1_20261019.zip` (sha256 `d77aee1d…6710765`) is committed unchanged at
`ae692a96` under `apps/VCS_QMI_KD_AIGIQA_Server_Package_v1_20261019/`. No file there is edited; `verify_package`
passes before and after every validation run. This note lists every server-side adaptation and every known
protocol risk. Losses, the λ grid `[.01, .1, 1, 10, 100]`, recipes, seeds, the CAL selection rule, the lock code and
the precision defaults are unchanged.

## Adaptations

1. **Launch directory.** Every package command runs with cwd = `KD_ROOT` = `/home/infres/yinwang/CS_QMI/outputs_apps/kd_20261020`
   and `PYTHONPATH=<pkg>:<pkg>/shared`, so the configs' relative paths (`data/cifar100`, `protocol/…`, `external/…`,
   `runs/teacher_*`) resolve in the outputs tree and nothing is written to `/projects` (`common.sh`).
2. **Data.** The server's CIFAR-100 python files were byte-copied read-only into `data/cifar100/cifar-100-python/`.
   Their MD5s equal the package's official ones. `--download` was never used. TEST was hashed but not parsed
   (`records/STAGING_RECORD.json`).
3. **Environment.** `/home/infres/yinwang/CS_QMI/env_kd` is an overlay venv: the same uv CPython 3.12.12,
   `--system-site-packages` (the base only has pip) and one `.pth` line exposing the V9/V10 env's site-packages
   (torch 2.9.0+cu128, torchvision 0.24.0+cu128, numpy 2.5.2, scipy 1.18.1, Pillow 12.3.0, PyYAML 6.0.3, which
   satisfies `requirements-core.txt`). Nothing is installed or duplicated, and the parent env is untouched (coordinator
   storage rule). `--system-site-packages` alone would not expose a venv parent. Every job compares the `pip freeze`
   sha256 with `env/ENV_LOCK.json` before training, so a later in-place change of the parent env stops KD jobs
   instead of silently changing them.
4. **Grid.** One `kd make-grid` call with all 15 arms produces 51 configs, instead of the README's three separate
   grids. All 51 configs are byte-identical to the three documented invocations (33 + 13 + 5). A single grid is
   needed so that one lock holds the documented matched group `dkd_bce,dkd_vcs_exact,dkd_vcs_sampled,dkd_infonce`
   (EXPERIMENT_LOCK §3).
5. **`PYTHONDONTWRITEBYTECODE=1`** everywhere, so the package tree and the upstream checkouts stay byte-identical.
6. **Locked-job wrapper** (`sbatch/kd_lock_train.sbatch`). An env/CUDA functional check and a preflight of the exact
   native path run before `vcs_common.experiment train` writes its receipt. The preflight covers the lock check, the
   receipt, a fresh run dir, the data, the teacher, the model, the configured loss incl. official CRD, and one real
   FIT batch with 4 workers. An infrastructure fault therefore leaves the job ID unused (exit 3/4: resubmit the same
   line). The training call itself is the package command.
7. **Storage** (coordinator, 2026-10-20). After a COMPLETED receipt is re-verified with the package's
   `_completed_record`, `last.pt` is deleted; its size and sha256 go to `ledger/<lock>/<job>__slurm<id>.post.json`.
   This deviates from kd/README_CN §6, which lists `last.pt` among the kept files. No lock, select, seal, freeze,
   evaluate, compare or profile step reads it, and the global entry cannot resume. The teacher's `last.pt` is deleted
   only after `teacher.pt` (the final-epoch weights) and its provenance are verified; both are then `chmod 444`. G0
   checkpoints and CSVs are deleted after hashing.
8. **Hardware.** GPU jobs use `--partition=RTX6000PRO,L40S,H100 --exclude=node51,node53,node60`. Only G0 is pinned,
   one job per type, because it is the cross-hardware gate and the timing stratum (INSTALL §2: never mix timings
   across environments). The CAL and final candidates are not pinned. The actual node, partition, GPU name/UUID,
   driver and event reasons go to `ledger/`. Timings are reported per stratum, and results are not claimed to be
   hardware-invariant.
9. **Slurm.** All jobs use qos normal and `--no-requeue`: a requeued locked job could only fail against its own
   RUNNING receipt. CAL jobs use nice 50; validation, teacher and G0 use nice 0. There is no `--dependency`; each line
   states its prerequisite. CAL `--time` is 3× the largest G0 240-epoch estimate over strata (provisional 12 h, or
   18 h for CRD/M1, until G0); the teacher gets 10 h and G0 3 h.
10. **G0** (`g0/G0_PLAN.json`, declared before any GPU job). It contains the package's GPU smoke and the 3-epoch
    DKD/BCE/VCS instrumentation (VALIDATION_AND_LIMITS). It adds per-family 2-epoch timing (INSTALL §5), a λ=100
    finiteness probe, a CUDA fp32 check against an independent fp64 formula, the package microbenchmark and a
    SIGKILL + `--resume` test. Seed 1729. Only `training.epochs` is changed, in G0-only config copies. G0 runs are
    never registered and never used to pick λ or to change the grid.
11. **Frozen execution source.** A lock binds `kd/` and `shared/vcs_common/` of this worktree by path and hash.
    Do not switch branches in `ssl_pilot_app_kd` or edit the package directory while a lock is active: `check`
    would fail and every queued job would refuse to start. The wrappers record the worktree HEAD and the
    `apps/kd_server` file hashes per job; on SIGTERM (time limit or scancel) they still classify the receipt and close
    the ledger entry.
12. **Round-1 scope.** Only the primary pair WRN-40-2 → WRN-16-2. The ResNet32x4 teacher line is optional (README
    priority 3, HANDOFF §八). The public-teacher track is not run.

## Pre-registration grid change (owner, 2026-10-20)

- **Change.** `dkd_vid_pooled_w100_seed27182` is removed, so the CAL grid has **50 candidates**
  (`protocol/kd_cal_grid50`, manifest sha256 `87e4376f…b1da6e19e`). The other 50 configs are byte-identical to the
  51-grid; the 51-grid stays in `protocol/kd_cal_grid` as a record and is never registered.
  - Built with two package `make-grid` calls: 14 arms × 5 λ, plus `dkd_vid_pooled` × λ {0.01, 0.1, 1, 10}.
  - Manifest = the 51-grid manifest minus that one row (checked equal to the union of the two make-grid manifests).
- **Owner, verbatim:** 「去掉这一个候选再登记 (Recommended)」. The option text read: 「登记前从网格中去掉 dkd_vid_pooled 的 λ=100，
  剩 50 个候选，并注明这是依据 G0 预检（只用 FIT 数据、2 个 epoch）在登记前做的改动。vid_pooled 不在配对比较组里，主比较不受影响。
  风险：240 个 epoch 的完整训练里，其他候选仍可能发散。」
- **Basis: pre-registration, FIT-only, 2-epoch finiteness** (G0 λ=100 probe, seed 1729, never registered).
  `dkd_vid_pooled` w100 hit a nonfinite loss at epoch 0 on both GPU types: inf on L40S
  (`g0/l40s_slurm1051709/G0_SUMMARY.json`) and NaN on RTX PRO 6000 (`g0/rtx6000pro_slurm1051708/G0_SUMMARY.json`).
  The other 8 λ families completed both epochs at w100.
- **Checks after the change:** design validator PASS (seed 27182, 50 runs; `dkd_vid_pooled` has 4 candidates and is
  outside the matched group). A scratch `register` and `check` pass with 50 jobs. Dry-run IDs equal the registered IDs
  (`records/PRECOMPUTED_CAL_JOB_IDS_50.json`; the 51-ID file is kept as a record).
- **Hardware at registration:** G0 PASS on RTX6000PRO (1051708) and L40S (1051709). CAL lines use
  `--partition=RTX6000PRO,L40S`; the coordinator adds H100 to pending jobs with scontrol after H100 G0 (1051710)
  passes. `--time` is 3× the larger of the RTX/L40S 240-epoch estimates.
- **R1 and R3, decided by the coordinator, following the package as written:** accept no-resume under the lock (with
  the mitigations below); `--match` stays the documented group, without M1.

## Known protocol risks (owner decisions; nothing was changed)

- **R1: the global lock has no resume.** `experiment train` refuses an attempted job. An interrupted (RUNNING) or
  FAILED candidate blocks `select-cal` and `seal`. The only remedy is `record-interruption` plus
  `register --amend-from` with a fresh grid, in which **all** candidates retrain (EXPERIMENT_LOCK §6). The native
  `kd train --resume` works (tested in G0) but only applies to the teacher, which is not a locked candidate.
  Mitigations: the normal QoS is not preemptible here (`PreemptMode=OFF`; preempt/qos only evicts runfill),
  `--no-requeue`, the pre-receipt preflight, 3× time limits and risk-first submission order.
- **R2: genuine method failures are FAILED candidates too.** Examples: a nonfinite loss at λ=100 (FloatingPointError),
  or ITRD's zero-variance guard (`std_floor` 0). Either blocks seal in the same way. The G0 λ=100 probe and the
  risk-first order (λ=100 first) surface this early. Whether to amend the grid or to adopt a pre-declared failure
  rule is a protocol decision for the owner or package author, to be made before registration.
- **R3: `--match` follows the documented group, without M1.** `dkd_vcs_m1` also passes the matched-recipe check (dry
  run), so it could be added for a stricter budget check. This must be decided before registration, because the lock
  is immutable.
- **R4: TEST access control is procedural**, as the package states. The TEST file sits read-only in the package's
  expected layout and is opened only by `evaluate-kd` after `seal`.
