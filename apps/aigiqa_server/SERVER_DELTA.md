# AIGIQA server delta note (branch apps/aigiqa-20261020)

Package: `apps/VCS_QMI_KD_AIGIQA_Server_Package_v1_20261019/`, committed unchanged (zip sha256 d77aee1d…). No package file is
edited. Everything below is a server-side wrapper or an environment/storage decision. Scope: AGIQA-3K alignment, first round
(SERVER_HANDOFF_CN.md §七), the 78-run pilot of EXPERIMENT_LOCK_CN.md §4.

## Owner decisions, 2026-10-20 (verbatim; relayed by the coordinator)
* Storage: 「数据放 /projects，checkpoint 放 home」. Data and weights stay on `/projects/EEG-foundation-model/yinghao/{data/agiqa_3k,
  models/imagereward}`; the ~2 GB download is a one-time exception to the 1.2 TB floor. Every run output (each `best.pt` via the
  base-config `output_dir`, the G0 outputs including the best.pt-shaped probe, protocol/lock/receipts, per-run records,
  predictions) goes to `AIG_OUT=/home/infres/yinwang/CS_QMI/outputs_apps/aigiqa_20261020`. Applied in `paths.env` before any
  lock exists; `data_dir` and the model paths are unchanged. Record: `evidence/decisions/OWNER_DECISIONS_20261020.json`.
* Download: 「你来执行」. The main session runs Stage 1 with the project's tool: `DATA_FETCH_RUN_FLOOR_GB` +
  `DATA_FETCH_RUN_FLOOR_REASON` (this run only, explicit `--group` required; manifest policy stays 1200; each started item records
  `floor_exception`) together with `FLOOR_GB=<same value>` for `data_fetch_guard.sh`, so the guard still stops the download if
  free space falls below the run floor.

## Unchanged science
Losses, Corrected-CMIM rules, λ grid {0.01, 0.1, 1, 10, 100}, seeds 3101/3102/3103, 20 epochs, batch 5, rank 32,
temperature 1, q_mode u, learning rates, StepLR(4, 0.5), selection metric CAL SRCC (same λ per family by the 3-seed mean),
prompt-grouped 70/15/15 split from the package's `prepare` (default seed 20261019, near-prompt threshold 0.97), torch default
TF32 flags, no AMP. The grid comes from the package's `aigiqa.cli make_grid --stage pilot`; registration, training,
selection, sealing and release use the package's `python -m vcs_common.experiment`.

## Adaptations
| # | What | Why |
|---|---|---|
| 1 | Server base config = package `base_alignment.json` with only `model.checkpoint`, `model.checkpoint_sha256`, `model.med_config`, `model.med_config_sha256`, `data_dir`, `output_dir` replaced (`make_matrix.py base` refuses any other difference and writes `BASE_CONFIG_DELTA.json`) | README §3 asks the server to fill staged paths/hashes |
| 2 | Env `/home/infres/yinwang/anaconda3/envs/vcs_aigiqa_ir`: Python 3.11; torch 2.9.0+cu128 / torchvision 0.24.0+cu128; the package's pins (NumPy 1.26.4, SciPy 1.13.1, PyArrow 14.0.2, timm 0.6.13, transformers 4.30.2, fairscale 0.4.13, huggingface_hub 0.23.5, diffusers 0.19.3, accelerate 0.20.3, datasets 2.14.7); CLIP @d05afc4 and ImageReward @78f8051 with `--no-deps` | INSTALL §4. cu128 is the build already used on RTX PRO 6000 Blackwell (sm_120), L40S, H100 here; older CUDA wheels lack sm_120 |
| 3 | `setuptools==69.5.1` at runtime; `setuptools<81` only inside ImageReward's isolated build | accelerate 0.20.3 (imported by transformers/diffusers/ImageReward) and ImageReward's setup.py import `pkg_resources`, removed in setuptools≥81 |
| 4 | Offline tokenizer: `HF_HOME=/projects/.../models/imagereward/hf_home` holds `models--bert-base-uncased` linked to the existing local copy (google-bert/bert-base-uncased @86b5e09; vocab/tokenizer_config/config/tokenizer.json git blobs equal that revision); HF/transformers offline flags | ImageReward's BLIP calls `BertTokenizer.from_pretrained('bert-base-uncased')`; jobs must not download |
| 5 | `ENV_LOCK.json` (pip freeze sha256, torch/CUDA/cuDNN, pinned source commits); every job runs `freeze_env.py check` first | INSTALL §4 "冻结实际环境" |
| 6 | Per-run wrapper `train_job.sbatch`: env check, GPU health probe, job resolution (refuses a job that already has a receipt) all happen BEFORE `experiment train`; `--no-requeue`; per-run record of node/partition/GPU/exit under `records/runs/` | The native trainer has no resume and the lock refuses re-attempts; a node fault must not burn a registered candidate. The package records no GPU name per run |
| 7 | G0 (`g0_preflight.py`) = the package's `preflight()` unchanged + weight-load key audit (`ImageReward.load` uses `strict=False`), unfrozen-layer audit, one real AdamW update with frozen-bitwise / trainable-changed / W-changed-and-boxed checks, fusion gradients after a nonzero head, full timed FIT epoch, Gaussian warm-up timing, CAL-sized inference timing on FIT rows, checkpoint bytes, 3-step finiteness at λ=100; FIT only, CAL/TEST never opened | INSTALL §4, HANDOFF §七 G0, PROTOCOL §5 G0 |
| 8 | One G0 per GPU type (RTX6000PRO, L40S, H100) before the matrix | Owner rule: a code path needs a cross-hardware gate on each type before running there |
| 9 | Staging verification (`stage_data.py`): fetch records, pinned git trees, owner-MOS mapping check, decode of every image, per-image sha256 cross-check against the HF mirror's published LFS hashes, model hashes; then the package's `prepare`; then `sealed/` set to mode 000 and inputs/images read-only | HANDOFF §四 (hashes, owner MOS mapping); sealing is a workflow guard, not access control |
| 10 | Fetch items in `scripts/data_fetch.py` group `aigiqa` (+ optional pinned HF revision; + reasoned one-run floor override `DATA_FETCH_RUN_FLOOR_GB`/`_REASON`); group `aigiqa_fallback` = the package-listed HF mirror strawhat/agiqa-3k @bf9b7bb8, run only if the official Drive archive fails; the staging record names the image source used | project data-location rule; download only through the tool and its guard |
| 12 | G0's checkpoint-size probe is saved (fsync) in the G0 output dir under AIG_OUT (home), timed, deleted | measures the real best.pt write cost on the runs' filesystem (home, per the owner's storage decision) |
| 11 | No `__pycache__` in the package tree (`PYTHONDONTWRITEBYTECODE=1`) | the lock hashes `*.py` only, but the tree stays clean |

## Hardware rule
All three partitions (`RTX6000PRO,L40S,H100`, excluding node51/53/60). The package says timing tables must not mix
environments (INSTALL §2, BENCHMARK §5), so timing/cost is reported per GPU-type stratum from the per-run records; if a
matched-hardware timing comparison is wanted, it is a separate short benchmark on one type, not a pin of the matrix.
Seed pairs may land on different GPU types; this is recorded and reported, not prevented.

## Protocol consequences to keep in mind
* No resume. A run killed after `experiment train` started leaves a RUNNING receipt: confirm the process stopped,
  `record-interruption`, then an amendment with a fresh grid, and under the package's minimal rule all candidates of the new
  lock retrain. Hence generous time limits, `--no-requeue`, and the pre-lock checks above.
* Home quota: run outputs now live in home (measured 1,336 GB used on 2026-10-20 before the 4.9 GB env; per-user quota limit not
  recorded, it was hit on 2026-10-07). The 78 checkpoints add about 142 GB by the end of the pilot; a quota hit would fail the
  running AIG jobs (FAILED receipts, i.e. an amendment) and other jobs writing to home, so headroom should be confirmed first.
* Checkpoints: the trainer keeps one `best.pt` per run (CAL-selected epoch, rewritten on each CAL improvement; no per-epoch or
  resume files). `seal` and every `check-release` (hence every `predict-aigiqa`) re-hash ALL 78 registered checkpoints, so
  non-selected checkpoints can be deleted only after all TEST predictions under this release are written; after that,
  `check-release` can no longer pass for this lock.
* Cross-dataset targets (AIGCIQA2023, PKU-AIGIQA-4K T2I) are not in this first round. Their label-free inputs can only be
  evaluated under this lock if they are listed at `register` (or in a pre-training amendment, which is free while no job has
  started); adding them after any TEST look is forbidden (EXPERIMENT_LOCK §4).
* The first round is labelled DEV / pilot (`scope=pilot`, release `PILOT_DIAGNOSTIC_ONLY`).

## TEST stage, owner approval 2026-10-21 (verbatim, relayed by the coordinator)
「开始 (Recommended)」 for the option 「AIGIQA：18 个选中的模型和原生 ImageReward 先在 TEST 输入上预测，再解封 MOS 标签评估一次。两边结果都标为
pilot/诊断，看过 TEST 后不再做任何选择。」
* `test_predict.sbatch` (GPU): the package's §4 sequence only: `check-release`, `predict-aigiqa` for the 18 run keys of the
  native freeze, `native_imagereward` + `import_scores` (package provenance file with the staged checkpoint hash filled in).
  Label-free; writes `results/test/PREDICTIONS_DONE.json`. Each `predict-aigiqa` re-runs the package's full release check.
* `test_evaluate.py` / `.sbatch` (CPU): checks predictions against PREDICTIONS_DONE, then unseals only `sealed/test_labels.csv`
  (verified against prepare's manifest hash), runs the package's `evaluate` (19) and `compare` (VCS-exact vs BCE, each seed,
  prompt bootstrap), reseals in a `finally`, records `TEST_UNSEALED.json`; a second run is refused. No selection code exists.

## G0 diagnostic naming (correction, 2026-10-21)

In `g0_preflight.py`, the `mean_diagnostics` block mixes two kinds of value:
- **Loss diagnostics** are averaged over every step of the timed FIT epoch.
- **`fusion_task_grad_norm` and `fusion_dependence_grad_norm`** are measured on one pre-set minibatch per epoch (step 1). They are not epoch-average gradients.

A small dependence-gradient value in G0 therefore does not show that the dependence term is ineffective over training, and it is no reason to raise λ. (Found by the owner-side review; verified in the code.)
