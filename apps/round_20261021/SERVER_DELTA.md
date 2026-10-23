# Round 20261021 apps line (AIGIQA + KD probe): server delta note

Package `VCS_QMI_Server_Round_20261021_v1.zip` (sha256 29aa3285…), `tools/check_package.py` passed. Its `apps/` directory is
committed unchanged under `package/` (46 files, bytes equal to `PACKAGE_MANIFEST.json`). No package file is edited.
Owner decision 2026-10-22: 「三条主线 + KD 都做」. Evidence role of every result: PILOT/DEVELOPMENT; no TEST read.

## What was run as the package says
* `install_overlay.py --base-package <worktree>/apps/VCS_QMI_KD_AIGIQA_Server_Package_v1_20261019 --output-package
  round_20261021_application_adaptation/apps_stage/VCS_QMI_KD_AIGIQA_Server_Package_round_20261021`. All 10
  `required_base_sha256` and 5 `overlay_sha256` match; the base package is untouched.
* `tests/test_application_update.py --package-root <stage>`: CPU job 1055932, 10/10 PASS in both server envs
  (`records/cpu_tests/`), torch 2.9.0+cu128 (package ran 2.6.0+cpu); measurements equal the package's own.
* Configs: the 24 shipped configs are used as is. Their asset paths and hashes were checked before any job
  (ImageReward.pt, med_config, AGIQA-3k FIT/CAL hashes and image-bytes flag, KD teacher checkpoint = provenance sha,
  FIT-only teacher, CIFAR train sha = manifest, mdistiller roots). No `make_configs.py` re-materialisation was needed.
  Both run roots did not exist before the first job.
* Each run is one row of the package matrix through `run_one.py` (AIG index 0-14, KD index 0-8), no automatic 240-epoch
  extension, no λ/rank/batch grid.

## Server-side adaptations
| # | What | Why |
|---|---|---|
| 1 | Stage made read-only (`chmod -R a-w`) after install; `PYTHONDONTWRITEBYTECODE=1`, `PYTHONPATH` unset in every job | keeps the installed bytes fixed; `run_one.py` puts the stage first on `sys.path` itself |
| 2 | Environments of the previous round, unchanged: AIG `anaconda3/envs/vcs_aigiqa_ir` (checked by `aigiqa_server/freeze_env.py check`), KD `env_kd` (checked by `ssl_pilot_app_kd/apps/kd_server/tools/env_check.py --cuda`, read in place from the KD worktree) | the package asks for the hosts' existing envs |
| 3 | `server/run_one.sbatch <app> <index>`: env check and GPU check (AIG `gpu_health.py`; KD CUDA functional check) BEFORE `run_one.py` creates the run dir; start/end record with node, partition, `nvidia-smi` name/UUID/driver, elapsed seconds and exit code under `outputs_apps/round_20261021_apps_records/<app>/<row>/slurm_<id>/`; `--no-requeue`; HF offline cache as in the previous AIG round | the trainers have no resume and refuse an existing run dir; an infrastructure fault before training (exit 3/4) leaves the fixed run dir unused, so the same line can be resubmitted. Peak memory and FIT time come from the trainers' own result/run JSON |
| 4 | Hardware: `--partition=RTX6000PRO,L40S,H100 --exclude=node51,node53,node60`, no type pin. APPS_SERVER_CN §6 asks that the same-seed comparisons run on one GPU type "尽量"; this is not enforced. The actual GPU of every attempt is recorded and the summaries list it per unit | coordinator instruction for this round; cost/timing reported per GPU type, no hardware-invariance claim |
| 5 | Resources: AIG 4 CPU / 32 GB / 2 h; KD 8 CPU / 24 GB / 1 h (previous rounds: AIG 20 epochs about 21 min on L40S; KD 240 epochs about 20 min) | generous limits, since a time-out cannot be resumed |
| 6 | Order: two AIG runs first (index 9 alternating exact and index 3 joint exact, seed 4101) as the real-GPU integration check of the two new paths; the other 13 only after both pass the epoch where the auxiliary weight turns on. KD runs only after the checkpoint diagnostic; one alternating-exact KD run first, the rest after it completes | README §4: real assets/CUDA must be joined on the server in small steps; uses fixed runs instead of extra smoke jobs |
| 7 | `server/kd_grad_diag.py` (+ `.sbatch`, CPU): **added server-side, the package has no entry for the checkpoint gradient diagnostic** (APPS_SERVER_CN §5). See below | required before the KD probe |
| 8 | `server/summarize.py`: per-seed CAL table, the pre-registered pairs, prompt bootstrap per seed (package `paired_bootstrap`) and per-prompt regret wins/losses for AIG, gradient-probe/critic logs for KD, cost of all attempts | 03_RESULT_ANALYSIS §2-3, §6-8 |
| 9 | Submission through the existing `handoff/review_claude_20261019/deps_watch/app_lines_feed.sh` with TAGs `r21_aig` / `r21_kd`, normal < 28, home limit 1587 GiB; lines in `server/lines/` | shared normal-QOS cap; AIG best.pt files live in home |

## KD checkpoint gradient diagnostic (adaptation 7)
* Checkpoints: the previous round's sealed final release (`outputs_apps/kd_20261020/protocol/kd_final_release.native_freeze.json`,
  seal re-checked; run.json and best.pt sha re-checked), CAL-selected best.pt of `dkd_vcs_exact` (λ=10) and `dkd_bce` (λ=.01),
  3 seeds each. The `dkd` checkpoints have no auxiliary critic, so they are not in it.
* Loss: the stage's `kd.losses.DistillationLoss` built from each checkpoint's own config (no `application_update`, so it is
  the original joint loss) with the checkpoint's critic state. Anchor = `last_anchor` = CE + DKD at the checkpoint epoch
  (warm-up factor 1); aux = `last_representation_loss` = λ·loss. CE-only cosine is reported separately for contrast with the
  old `last_task` log. Gradients w.r.t. the pre-classifier feature (`representation_gradient_diagnostics`, the same target as
  the runner's in-run probe) and w.r.t. all student parameters. The ratio is also rescaled to the probe's λ=.01.
* Batches: 32 FIT batches × 64, fixed by seed 20261022 (subset and training crop/flip), identical for all checkpoints.
  Student BN in train mode, teacher eval + no_grad, no optimizer step, nothing saved, CPU float32 (training ran on GPU with
  the default TF32 flags, so values can differ slightly from in-training numbers).
* FIT only: TEST is never constructed; in the run dirs only run.json and best.pt are opened.

## Execution record
* CPU tests 1055932 (10/10 PASS, both envs). KD checkpoint diagnostic 1055947 (CPU, 521 s, result in `records/kd_grad_diag/`).
* AIG: 1055950/1055951 stopped with exit 4 before training (this wrapper passed a relative path to `gpu_health.py`;
  `os.makedirs('')`); run dirs never created; fixed in 3e06e0c4 and resubmitted as 1055959/1055960 (integration pair),
  then 1055967-1055979 through the feeder (TAG r21_aig). The failed attempts stay in the per-attempt records and count as cost.
* KD: 1055961 (alternating exact 5101, integration run: complete, 40 epochs, 288 s, teacher frozen verified), then
  1055985-1055990, 1055992, 1056003 through the feeder (TAG r21_kd).
* Submission TSVs: `handoff/review_claude_20261019/deps_watch/APPS_r21_{aig,kd}_SUBMITTED.tsv`.
* A line that stops with exit 3/4 (env/GPU check, run dir untouched) can be resubmitted as is (append the new ID to the TSV).
  A failure after the run dir exists needs a new run root through `make_configs.py`, recipe and seed unchanged.

## Summaries (after all runs end)
```bash
S=/home/infres/yinwang/CS_QMI/round_20261021_application_adaptation/apps_stage/VCS_QMI_KD_AIGIQA_Server_Package_round_20261021
A=/home/infres/yinwang/CS_QMI/ssl_pilot_app_round21/apps/round_20261021/package
R=/home/infres/yinwang/CS_QMI/outputs_apps/round_20261021_apps_records
SV=/home/infres/yinwang/CS_QMI/ssl_pilot_app_round21/apps/round_20261021/server
cd /home/infres/yinwang/CS_QMI/slurm_logs
sbatch --job-name=r21_aig_chk_summary --qos=normal --partition=cpu-high --cpus-per-task=2 --mem=8G --time=00:30:00 --nice=0 \
  --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out \
  --wrap "PYTHONDONTWRITEBYTECODE=1 /home/infres/yinwang/anaconda3/envs/vcs_aigiqa_ir/bin/python -I $SV/summarize.py --app aigiqa --stage $S --apps $A --records $R --out $R/summary"
sbatch --job-name=r21_kd_chk_summary --qos=normal --partition=cpu-high --cpus-per-task=2 --mem=8G --time=00:15:00 --nice=0 \
  --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out \
  --wrap "PYTHONDONTWRITEBYTECODE=1 /home/infres/yinwang/CS_QMI/env_kd/bin/python -I $SV/summarize.py --app kd --stage $S --apps $A --records $R --out $R/summary"
```
Outputs `$R/summary/{AIG,KD}_SUMMARY.json` and `{AIG,KD}_PER_SEED.csv`. The summarizer was smoke-tested on the previous
round's CAL artifacts (temporary matrix, scratchpad only).
