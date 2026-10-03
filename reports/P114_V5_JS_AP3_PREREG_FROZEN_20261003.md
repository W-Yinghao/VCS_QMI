# Pre-registration — P114: matched-JS control of A-P3 (v5 NEXT-A-JS-AP3), three 800-epoch CIFAR-10 seeds, 2026-10-03 — FROZEN 2026-10-03T13:30:56Z before 800-epoch compute (CPU gate 1019936, GPU smoke 1019937)

Status: FROZEN 2026-10-03T13:30:56Z (main session; owner 2026-10-03: execute the v5 plan).  Source: owner 2026-10-03
("查看这个两个文档，现在的结束之后开始分析并执行这两个文档中的计划"), `VCS_Server_Next_Round_v5_CN.md` §2, `VCS_Results_Review_and_Next_Plan_v5_CN.md` §6 P1.

## Question
Does A-P3's gain still depend on the VCS quadratic objective when the scorer, pairing and gradient routing are identical?  P107's A-L1 (JS on the fixed
scorer) used K8 cross-view pairs with negative right-detach, so it is **not** the A-P3 control (v5 §2.3).  At K8 / detach, fixed-scale JS (88.54) and VCS
(88.66) were already close (seed 0); this unit asks the same question in A-P3's best setting (all view tokens, full gradients).

## Cells (stage `P114_v5_js_ap3`; `configs/make_p114_configs.py --write`; hashes in `configs/P114_SHA256.json`)
`configs/cifar10_hpJS_AP3_views4_800ep_seed{0,1,2}.yaml` = `configs/cifar10_hpY_AP3_views4_800ep_seed{s}.yaml` (P107 A-P3) with ONLY:
`objective.loss` negative_J → js_matched_logistic; `objective.js_fixed_scorer` → true; `objective.js_all_view_tokens` → true (new marker); `run.stage`.
Everything else is A-P3's: CIFAR-10 dev45k/val5k manifest, ResNet-18 from scratch, 4 views, B 256, 800 epochs, standard augmentation, optimizer /
schedule / projector, unit-norm z, s = zᵢᵀzⱼ, fixed f = 2s − 1 (buffers, no trainable critic parameter), all-view-token pairing (P = all ordered different-
view pairs of one image, Q = all view combinations of different images; P and Q averaged separately over their global counts; chunk 256), full (non-
detached) gradients on both sides, same encoder init / augmentation / data-order RNG roles (run.seed unchanged).
Loss: L_JS = mean_P softplus(−2f) + mean_Q softplus(2f) (the existing matched version; per-pair f-gradient equals −J's at f = 0; not an unbalanced BCE).

## Code (this unit)
`src/vcs_ssl/objectives.py`: `all_view_tokens_loss(..., objective="js")` — same tokens, masks, chunks and separate P / Q means on the logit matrix
f = a·C + b; VCS statistics from T = tanh f under no_grad (comparison only); dispatcher passes the loss choice.  `src/vcs_ssl/config.py`: optional
`objective.js_all_view_tokens` (not filled when absent → existing config hashes unchanged); JS on the all-view path is allowed ONLY with both markers and
affine_mode fixed; the P107 refusal (JS + all-view without the new marker) and every other refusal are kept.  Tests `tests/test_vcs_ssl_p114.py`.

## Reading (pre-stated, descriptive; v5 §2)
Report per seed: linear-val, kNN, wall time, peak memory; paired A-P3 − JS (same seed), mean, sd, 95 % t interval; references A-P3 seeds 0–2
(89.06 / 88.96 / 89.06; kNN 87.30 / 87.48 / 87.36) and tuned SimCLR seeds 0–2 (88.20 / 88.10 / 88.66).
- **"Close"** if |mean paired difference| < 0.3 linear (≈ the larger of the two methods' seed sds × 1.5) or the 95 % interval includes 0 — reported as
  "actual results close"; no statement that the quadratic objective has an own contribution in this setting.
- **"Clear difference"** if the interval excludes 0 and |mean| ≥ 0.3: reported with its sign, and **before any general statement** a pre-specified small
  learning-rate / optimiser-budget sensitivity check for both losses is required (v5 §2 Report) — designed as a separate unit only then.
- Early diagnostics only flag implementation / non-finite failures; no early stopping on low scores (v3 §10.3 rule kept).
- Not claimed: anything beyond CIFAR-10 standard augmentation; nothing about strong augmentation or CIFAR-100.

## Gate and GPU smoke (filled in from jobs)
CPU gate `slurm/p114_cpu_gate.sbatch` (suites ssl_core / integration / v2 / p95 / p100 / p104 / p107 / p114; validate_core; first-step gradients vs P35;
config-hash invariance HEAD vs new code; smokes + stop/resume of the 3 P114 and 4 P115 configs with the A-P3 parent as reference).  GPU smoke
`slurm/p114_gpu_smoke.sbatch` (100 real-CIFAR steps of P114 seed 0 next to A-P3 seed 0: s/step, peak memory, finite gradients, resume).
**CPU gate job 1019936: gate_rc 0.** Suites: ssl_core 29, integration 40, v2 14, p95 15, p100 8, p104 30, p107 17, **p114 15** passed; validate_core
25/25; recipe first-step gradients vs P35 max rel diff 8.5e-5; config-hash invariance 369 configs (368 loadable at HEAD — the one exception is the
plan file `configs/next_stage_plan.yaml`), **0 changed by the P114 code**; 6-step smokes + stop/resume COMPLETED for the 3 P114 and 4 P115 configs
(in `reports/P114_GATE_1019936/` the P114 cells are labelled `gate_AP3_seed{0,1,2}` — a label artefact of the gate script; configs
`cifar10_hpJS_AP3_*`); CPU step-time ratio vs A-P3 0.98–1.04.  **GPU smoke job 1019937 (L40S): smoke_rc 0.**  P114 seed 0 0.2139 s/step vs A-P3
0.2129 (ratio 1.005); peak 5137 MB allocated / 7412 MB reserved (A-P3 5137 / 7420); finite gradients; J at step 100 0.545 (A-P3 0.548); resumed-vs-
uninterrupted |ΔJ| ≤ 0.0034, rerun-vs-rerun |ΔJ| ≤ 0.0006, max state diff 1.72 (A-P3 in P107's smoke: 1.61 — same GPU non-determinism level);
encoder / projector init hashes equal to P35 seed 0.  Evidence: `reports/P114_GATE_1019936/`, `reports/P114_GATE_SMOKE_1019937/`.

## Cost
A-P3 measured 0.1139 s/step on RTX6000PRO (P107 GPU smoke) → ≈ 4.4 h per 800-epoch run on an unshared RTX6000PRO (observed 4.4–9.8 h depending on
node sharing / node60); JS adds no extra pair evaluations.  3 runs, normal QOS, RTX6000PRO / H100 (node60 allowed).  Launch: `slurm/p114_lines.txt`.

## Decisions at the freeze (main session)
1. **Reading tightened before any run** (the drafted "or the 95 % interval includes 0" would label every under-powered 3-seed comparison "close"):
   **close** = |mean paired difference| < 0.3 linear; **clear difference** = |mean| ≥ 0.3 **and** the 95 % interval excludes 0 (sign reported; the
   lr / optimiser-budget sensitivity check is then required before any general statement); **inconclusive at 3 seeds** = otherwise (|mean| ≥ 0.3 with an
   interval including 0) — reported as such, no direction claimed.  kNN reported alongside with the same labels.
2. Launch now (v5: the immediate batch), partitions RTX6000PRO / H100, normal QOS, node60 allowed.
