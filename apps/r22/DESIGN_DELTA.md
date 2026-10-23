# r22 apps line (R22_ALTERNATIVES_CN.md D and E): design delta

Owner 2026-10-22: only the theory of `Variational_Cauchy_Schwarz_divergence.pdf` is fixed (J, T* = tanh(PMI/2), T = tanh(f),
Q = product of marginals, eq. 40); the implementation is free. Evidence role of every result: PILOT/DEVELOPMENT, FIT/CAL only,
no TEST read (CIFAR-100 test file and AGIQA-3K sealed/test inputs are refused at runtime by `guard.py`).

## Engine (`engine/`)
Commit 9cda4320 holds the unchanged copies: `kd/*.py` from the previous KD package (`VCS_QMI_KD_AIGIQA_Server_Package_v1_20261019`,
source digest df7c218c = the digest of the crd_original/dkd finals), `aigiqa/*.py` and `shared/vcs_common/*.py` from the
round-21 installed stage (base + application_update overlay). Every later change is visible with `git diff 9cda4320 -- apps/r22/engine`.
Jobs run from a read-only snapshot `outputs_apps/r22/code_<commit>` (`server/snapshot.sh`).

## D. KD: VCS-CRD (`kd/crd_adapter.py`, +1 line in `kd/runner.py`)
* New CRD option `objective: "vcs"`. Kept from official CRD: embedding heads (linear 128 + L2 norm, student and teacher
  side, SGD lr .05 with the student, as in crd_original), the two momentum memory buffers (momentum .5, renormalized,
  updated at the batch's own indices exactly as `ContrastMemory`), 16,384 negative indices per anchor from the package
  sampler, CE (+ DKD) anchor, 240 epochs, WRN-40-2 -> WRN-16-2, strict-FIT teacher, optimizer and seeds.
* Replaced: the NCE criterion. T = tanh(cos(e_s, e_t)/tau), tau = .07 (CRD's temperature, fixed).
  P = the B same-image pairs (e_s(x_i), e_t(x_i)) of the current batch. Q, student anchor = (e_s(x_i), m_t(x_j)),
  teacher anchor = (e_t(x_i), m_s(x_j)), j drawn uniformly from the other FIT instances (B x 16,384 pairs each).
  Loss = -(J_s + J_t), CRD's two directions summed as in `CRDLoss` (equivalently -2 J(P, (Q_s+Q_t)/2)).
  Each J normalizes P and Q separately.
* Deviation 1 (theory): official CRD in crd_original draws negatives from other classes (`class_excluding_original`),
  which is not a product-of-marginals sample. VCS-CRD uses `uniform_other_instances` (the package's other mode,
  already used by dkd_crd_uniform); the adapter refuses `objective: vcs` with class-excluding negatives.
  To separate objective from negative sampling, a matched control **crd_uniform** (official NCE, uniform negatives,
  weight .8, plain CE) is added at the CAL seed and the three final seeds. dkd_crd_uniform (previous round) is the
  exact matched control of DKD+VCS-CRD.
* Deviation 2 (wording): the task text says "CE + KD + representation term", but the recorded crd_original recipe is
  CE + 0.8 CRD with anchor "none" (RepDistiller `-a 0 -b 0.8`). The recorded recipe is followed: vcscrd = CE + lambda(-J).
* P uses the current teacher embedding, not CRD's memory positive (task spec). Q uses memory rows, i.e. momentum
  averages of earlier embeddings of the other images (stale marginal samples; random unit-ish rows during epoch 1,
  as in official CRD).
* lambda in {0.8, 3, 10} on CAL seed 27182 (the previous round's CAL seed), selection = best-epoch CAL top-1 of that run
  (tie: lower CAL NLL); finals at 31415/16180/57721 = the seeds of the previous crd_original/dkd/dkd_crd_uniform finals.
  Every run's best epoch is selected on CAL as in the previous round.

## E. AIGIQA: MOS-weighted P and a text bank (`aigiqa/losses.py` class `BankedDependence`, 2 call lines in `engine.py`)
* Engine: the round-21 stage engine (record: its `alternating_target_sg` path and RNG isolation are needed for pairing with
  round-21 task_only, seeds 4101-4103).
* Critic: the package's rank-32 simplex-bilinear head with boxed W (T in [-1, 1]), constructed by the package regularizer,
  so its init and RNG draws equal the round-21 arms.
* Q = every (x_i, y_j), i != j, of the current batch (the package U support) plus every (x_i, y_k) for the 4,096 text
  embeddings y_k of previous FIT batches in a FIFO buffer (detached; the critic's right map is applied to them at use
  time). One denominator over all Q pairs (5*4 + 5*4096 when full). The in-batch pairs are kept so that Q is defined
  from the first step (empty bank); they are 0.1 % of Q once the bank is full (after about 2 epochs, i.e. before the host
  weight turns on at epoch 3).
* P = matched pairs, uniform or MOS-weighted: P_w puts w_i/B on pair i with w_i = MOS_i / batch-mean MOS (labels are
  MOS/5, so the ratio is the same); P_w is a probability measure. With P_w the optimal critic is tanh(1/2 log(w p/q)).
* Matched BCE: the package's bounded balanced BCE, 2(BCE - log 2), on the same score, the same P_w and Q.
* Update: alternating_target_sg (round 21): each batch one AdamW critic step on detached (x, y) with the unweighted
  objective, then the frozen critic gives the host -lambda J through x only (direct target edge stop-gradient), bank
  written after both. Chosen over joint_scheduled because round-21 CAL (AIG_SUMMARY.json) has alternating above joint
  for both objectives (+0.009 SRCC exact and BCE, 3/3 seeds) and joint below task_only, and lambda here is 10x larger.
  This choice uses round-21 CAL, i.e. it is CAL-informed. Warm-up 2 / ramp 4 epochs, critic lr .001 halved every 4
  epochs, host lr 1e-5, batch 5, 20 epochs, best CAL SRCC epoch per run, all as round 21. lambda = 0.1 fixed.
* Arms: exact_bank (P uniform), exact_bank_mosw, bce_bank_mosw (matched), and an added control **exact_inbatch**
  (= round-21 alternating exact, package code path, at lambda 0.1) that isolates the bank at the same lambda.
  Pairs: vs round-21 task_only and round-21 alternating exact/BCE (lambda .01) at the same seeds.

## Tests and smokes
* `tests/test_r22.py` (CPU, both envs): VCS-CRD loss = brute-force J over the memory bank (float64), no own instance in
  Q, memory update = official ContrastMemory, class-excluding Q refused, official NCE path unchanged, gradients reach
  student and both heads and not the teacher, synthetic runner for vcscrd/dkd_vcscrd (teacher frozen, only FIT/CAL roles);
  banked J and BCE = brute force (FIFO contents), bank 0 + uniform P = package objective, P_w normalization and bad-label
  refusal, alternating step (critic fit uses the bank, outer gradient to x only, nothing to the bank or critic), critic
  init / host RNG equal across arms, synthetic AIG engine run with bank + MOS (no TEST file opened), real FIT MOS >= 0;
  guard refuses unlisted files.
* GPU smokes `r22_kd_chk_*`, `r22_aig_chk_*`: `grad_check.py` on one real FIT batch, then 2 epochs of the real trainer.

## Execution
* CPU tests 1056105 (1 failure: float32 cast of the MOS weights, fixed in b6cf433b), 1056108: 13/13 PASS in env_kd and
  vcs_aigiqa_ir (`records/cpu_tests/`).
* GPU smokes (`records/smoke/`): KD 1056109 / 1056110 PASS (RTX PRO 6000, about 13 s per epoch). AIG 1056111 / 1056112 stopped
  at the grad check, whose direct-text test was wrong (x = fusion(visual, text), so text gets aux gradient through x by
  design); the check now detaches x and requires no graph to text (7635c220). Resubmitted as 1056114 / 1056115: PASS.
  Smoke best.pt files deleted.
* Snapshots: S1 and AIG lines run from `outputs_apps/r22/code_7635c220`; KD finals from `code_adc5de49` (engine, configs
  of S1/AIG and run_one identical; adds JOBS_S2_ALL.csv).
* Feeders (`handoff/review_claude_20261019/deps_watch/app_lines_feed.sh`, normal < 28, home < 1587 GiB): TAG r22_kd
  (`server/lines/KD_S1_LINES.txt`, 10 lines), TAG r22_aig (`server/lines/AIG_LINES.txt`, 12 lines). Submission TSVs
  `deps_watch/APPS_r22_{kd,aig}_SUBMITTED.tsv`.
* KD finals: `server/kd_after_cal.sh` polls every 5 min; when the 6 lambda candidates have finished it runs
  `select_kd.py` (rule above; record `outputs_apps/r22/kd/kd_lambda_selection.json`) and feeds the 6 selected lines
  (`server/lines/KD_S2_LINES.txt`) with TAG r22_kd. Exit 3 + a watch.log line = manual decision needed. By hand:
  `env_kd/bin/python -I apps/r22/select_kd.py <code_7635c220> <code_adc5de49> <out json> <out lines>`.
* A line that stops with exit 3/4 (env/GPU check; run dir untouched) can be resubmitted as is. A failure after the run dir
  exists needs a new run dir (recipe and seed unchanged).

## Summaries (after the runs)
```bash
C=/home/infres/yinwang/CS_QMI/outputs_apps/r22/code_adc5de49; S=/home/infres/yinwang/CS_QMI/ssl_pilot_app_r22/apps/r22
cd /home/infres/yinwang/CS_QMI/slurm_logs
sbatch --job-name=r22_kd_summary --qos=normal --partition=cpu-high --cpus-per-task=2 --mem=8G --time=00:20:00 \
  --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out \
  --wrap "PYTHONDONTWRITEBYTECODE=1 /home/infres/yinwang/CS_QMI/env_kd/bin/python -I $S/summarize.py --app kd --code $C --out $S/records/summary"
sbatch --job-name=r22_aig_summary --qos=normal --partition=cpu-high --cpus-per-task=2 --mem=8G --time=01:00:00 \
  --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out \
  --wrap "PYTHONDONTWRITEBYTECODE=1 /home/infres/yinwang/anaconda3/envs/vcs_aigiqa_ir/bin/python -I $S/summarize.py --app aigiqa --code $C --out $S/records/summary"
```
Outputs `records/summary/R22_{KD,AIG}_SUMMARY.json` and `R22_{KD,AIG}_PER_RUN.csv`: KD CAL-seed table (all lambda
candidates, crd_uniform, previous crd_original/dkd/dkd_crd_uniform CAL runs), the selection, final CAL top-1 per seed and
the paired differences (vcscrd vs crd_original / crd_uniform; dkd_vcscrd vs dkd / dkd_crd_uniform); AIG CAL SRCC/PLCC/
within-prompt pairwise/regret per seed, the pairs listed in `summarize.py` (primary exact_bank_mosw - bce_bank_mosw),
per-seed prompt bootstrap and per-prompt regret wins; cost of every attempt by GPU type. The previous KD finals are
read through best_cal_metrics.json only (their TEST files are never opened).
