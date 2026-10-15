# V9_03 — baseline fidelity: NCR / L2RM native evaluation protocol, released weights, local differences, ESC / ASL / ReCon executability (2026-10-15, fork A)

Status: code-level facts verified on the server; **no local number yet** (precomp features downloading; NCR checkpoints downloading).  Table of record: `EXTERNAL_FIDELITY.csv`.

## 1. Official evaluation protocols as read from the code
| host | code | split / gallery | rule | selection |
|---|---|---|---|---|
| SGR single (SGRAF python3.6 a19d1a9) | `evaluation.py: evalrank(model_path, data_path, split, fold5)` | Flickr30K `test` 1,000 images × 5,000 captions; COCO `testall` 5,000 images (`fold5=False`) or 5 × 1K folds averaged (`fold5=True`) | `i2t`: rank of the best of the 5 GT captions; `t2i`: rank of the GT image; R@1/5/10, medr, meanr; rSum = sum of the six recalls | best dev rSum (`train.py validate` every epoch and every 1,000 iterations; dev = 1,000 Flickr images, 5,000 COCO images) |
| NCR (60256ba) | `NCR/evaluation.py` (set `model_path`, `data_path`, `vocab_path`; `split='testall'`, `fold5` false → 5K / true → 5×1K) | same as SGR | same; **inference averages networks A and B** (paper §4.2) | best dev rSum of the averaged model |
| L2RM (a766c22) | `main_testing.py` (same evaluator lineage; SGR / SAF / SGRAF variants) | same | same | best dev rSum |

Both NCR and L2RM read the SCAN precomp layout `<data_path>/<data_name>/{train,dev,test,testall}_{ims.npy,caps.txt}` and the SGRAF vocab json; V9's frozen evaluator will be this `evalrank` (fixed gallery, stable ID-order ties) — the package's reference `retrieval_recall` is only the definition check.

## 2. Released weights
- **NCR:** seven checkpoints from the authors' CDN (`ncr_{f30k,coco}_{0,0.2,0.5}`, `ncr_cc152k`), ~145 MB each, registered as data item `ncr_models` (download in progress; sha256 recorded per file).  Native A-table reproduction = run `NCR/evaluation.py` on `f30k_0 / f30k_0.2 / f30k_0.5` (Flickr 1K) and `coco_*` (5K + 5×1K) once `ncr_precomp` is extracted; compare with paper Table 1 (quoted in `EXTERNAL_FIDELITY.csv`).  **Not done yet.**
- **L2RM:** Google-Drive folder only (`1NSEoPaZa3kkxKt3jI83aWDDOax-5IGvD`); not fetched (needs gdown / browser; owner decision).  Reproduction path = `train_f30k.sh` (35 epochs, seed 0) with the DECL/RCL 0.4 index; ≈ 1 SGR training per cell.
- **SGRAF/SGR:** checkpoints on Google Drive only; not fetched.

## 3. Paper numbers to reproduce (exact quotes; see the CSV for the full rows)
- NCR paper Table 1, Flickr30K 1K: NCR 0 % 77.3/94.0/97.5 | 59.6/84.4/89.9; 20 % 75.0/93.9/97.5 | 58.3/83.0/89.0; 50 % 72.9/93.0/96.3 | 54.3/79.8/86.5.  SGR 0 % 75.2/93.3/96.6 | 56.2/81.0/86.5; SGR* (pre-trained with vanilla triplet) 20 % 62.8/86.2/92.2 | 44.4/72.3/80.4; SGR-C (clean subset) 20 % 72.8/90.8/95.4 | 56.4/82.1/88.6.
- NCR paper Table 1, MS-COCO 1K (5-fold): NCR 0 % 78.7/95.8/98.5 | 63.3/90.4/95.8; 20 % 77.7/95.5/98.2 | 62.5/89.3/95.3; 50 % 74.6/94.6/97.8 | 59.1/87.8/94.5.  The paper reports 5K in its supplement (not extracted here).
- L2RM paper Table 1 (DECL/RCL indices; rSum): Flickr30K MRate 0.4 — NCR 467.1, L2RM-SGR 479.8 (73.1/92.4/96.3 | 52.3/79.4/86.3), L2RM-SGRAF 490.5; MS-COCO 1K 0.4 — NCR 509.7, L2RM-SGR 509.4 (75.2/94.8/98.1 | 59.4/87.8/94.1), L2RM-SGRAF 517.7.  At 0.2: NCR 491.1 / 515.0, L2RM-SGR 492.5 / 518.5.  At 0.6 NCR collapses (184.6 / 2.4).
- Noise-index difference: NCR's own index is a fresh random draw saved by its `data.py`; the L2RM table re-runs NCR on the DECL/RCL indices.  V9's A table uses the DECL/RCL `*_0.4.npy` for every native method so the rows are comparable to L2RM's table; this is written as the A-table condition.

## 4. Local differences known before any run
1. Environment: torch 2.9.0+cu128 / python 3.10 instead of torch 1.7 (NCR) / 1.12 (L2RM) / 0.4 (SGRAF).  Expected numerical differences at the 0.1–0.5 rSum level; recorded, not claimed as fidelity failure unless larger.
2. `tensorboard_logger` needs protobuf 3.20.3 (installed in `v9_sgr`).
3. L2RM's `pomegranate` (rayleizhu fork, `GeneralMixtureModel`, `TrueBetaDistribution`) — install into py3.10 TO_VERIFY (`slurm_logs/v9_env_pomegranate.log`); fallback = scikit-learn GMM with the same two-component fit, labelled as a deviation.
4. Seeds: SGRAF `train.py` has no seed argument; V9 wraps every host run with fixed torch/numpy/random seeds (901–903, tuning 917).
5. NCR evaluation averages two networks → NCR+A rows cost 2 × SGR; reported as an ensemble condition (package §05 §4).

## 5. ESC / ASL / ReCon executability (verified 2026-10-15)
- **ESC** (CVPR 2024): official code found at `github.com/yangyuchenn/ESC` (f1798b2, 2025-03-01; built on BiCro + NCR), cloned to `code_v9/ESC`; README has no run commands, hyper-parameters or checkpoints ("refer to NCR for the environment").  Executable in principle; a fidelity run needs the paper's settings (CVF page returned 403 to the fetcher; PDF to be obtained) → **prepare, not yet runnable with confidence**.
- **ASL** (AAAI 2025): publisher page lists no code; GitHub search for the title / "Asymmetric Similarity Learning noisy correspondence" → 0 repositories.  → **paper-only; no reproduction in the first round**.
- **ReCon** (CVPR 2025): `qxzha/ReCon` 75c60f7 (2025-09-29) — README "The code is coming soon!"; repo has `data.py / eval.py / opt.py / utils.py / vocab.py / models/SGRAF.py`, **no training script** → not reproducible.
- Found with public code on the same host (not in the package list): DECL (`QinYang79/DECL`, ACM MM 2022) and CREAM (`XLearning-SCU/2024-TIP-CREAM`, TIP 2024).  Owner decision whether to add them as external rows.

## 5b. Import checks in `envs/v9_sgr` (2026-10-15)
- SGRAF_py3 `opts, model, data, evaluation, vocab`: OK.
- NCR `model, data, evaluation, utils, vocab`: OK after adding `matplotlib` to the env.
- L2RM `models, data, evaluation, utils, vocab`: OK after adding `pot` (POT) and `prefetch_generator`; `main_L2RM.py` additionally imports `pomegranate` (`GeneralMixtureModel, TrueBetaDistribution`) — `pomegranate==0.14.8` fails to build in `v9_sgr` (python 3.10 / numpy 2: Cython error in `pomegranate/utils.pyx`); **resolved** in a second env `envs/v9_l2rm` (python 3.10, torch 2.9.0+cu128, numpy 1.26.4, Cython 0.29.37, setuptools, `git+https://github.com/rayleizhu/pomegranate` = 0.14.8 fork with `TrueBetaDistribution`, tensorboardX, POT, prefetch_generator): `import main_L2RM` OK (`slurm_logs/v9_env_l2rm.log`).  L2RM units run in `v9_l2rm`; every other host in `v9_sgr`.
- ESC `model.py` raises `NameError: name 'net' is not defined` at import (line 816 inside `class SGRAF`, reached from line 591) in the released commit f1798b2 — the repository is not importable as released; a fix would be a local patch and must be disclosed.

## 5c. Host finding from the timing job (1042033, node60, 2026-10-15)
Native SGRAF_py3 `train.py` from scratch on clean Flickr30K precomp (hardest in-batch negative from the first step) **collapsed**: loss 51.49 → 51.2004
(= 2 × 128 × margin 0.2: all pair scores equal), dev i2t/t2i R@1 0.1 / 0.1 after one epoch (`outputs_v9/timing/group_1042033/sgr_train.log`).  The NCR
paper reports SGR's sensitivity and pre-trains SGR with the vanilla triplet loss (SGR*); NCR's own warm-up (`co_train.warmup`, `ContrastiveLoss(margin)` =
sum-violation) is the locked warm-up for every SGR-based V9 row (HOST_LOCK `host_finding_20261015_timing_job_1042033`).  The native "SGRAF from scratch
with max_violation" recipe is recorded as not trainable here (one measured attempt under torch 2.9.0+cu128; no claim about other versions).  The
`torch.load` weights-only failure of the evaluators is handled by `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` (no code change).

## 6. What remains for this document
NCR checkpoint evaluation numbers (Flickr 1K ×3, COCO 5K/1K ×3) and their gaps to the paper; one SGR native 40-epoch reproduction on Flickr (counts against the budget; after the owner's go); L2RM native rerun at 0.4 (after pomegranate / fallback decision).
