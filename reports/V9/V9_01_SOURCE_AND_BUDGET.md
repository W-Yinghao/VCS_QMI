# V9_01 — sources, licences, independence and budget (2026-10-15, fork A)

## 1. Baseline code (official clones under `/projects/EEG-foundation-model/yinghao/code_v9/`)
| repo | commit (date) | licence | python / torch in the README | role in V9 |
|---|---|---|---|---|
| Paranioar/SGRAF, branch `python3.6` (`SGRAF_py3`) | a19d1a9 (2024-04-11) | Apache-2.0 | 3.6 / ≥ 0.4 | **host**: SGR single model (`model.py`, `train.py`, `evaluation.py`, `vocab/*_precomp_vocab.json`) |
| Paranioar/SGRAF, branch `main` (`SGRAF`) | 50d0c6f (2024-04-11) | Apache-2.0 | 2.7 / 0.4.1 | archive only; `opts.py` defaults identical to the py3 branch |
| XLearning-SCU/2021-NeurIPS-NCR | 60256ba (2023-11-06) | LICENSE = MIT, README = Apache-2.0 | 3.7 / ~1.7.1 | NCR / NCR+A (co-teaching of two SGR networks; GMM on per-sample loss; soft-margin triplet) |
| hhc1997/L2RM | a766c22 (2024-03-28) | Apache-2.0 | 3.8 / 1.12 + pomegranate (rayleizhu fork) | L2RM / L2RM+A (OT rematching; `noise_index/` = DECL/RCL indices 0.2–0.8) |
| yangyuchenn/ESC | f1798b2 (2025-03-01) | not stated in the repo listing (TO_VERIFY) | "refer to NCR" | external row candidate; no run commands / checkpoints |
| qxzha/ReCon | 75c60f7 (2025-09-29) | LICENSE present | — | not runnable ("code is coming soon"; no train script) |

Environment `/projects/EEG-foundation-model/yinghao/envs/v9_sgr` (uv, python 3.10; torch 2.9.0+cu128, torchvision 0.24.0; numpy < 2.3; scipy; scikit-learn; nltk + punkt / punkt_tab; tensorboard_logger with protobuf 3.20.3; pytest).  Import check: `torch, numpy, sklearn, nltk, scipy, tensorboard_logger` OK; `nltk.word_tokenize` OK.  SGRAF_py3 `train.py` / `evaluation.py` import their modules from this env (data-independent parts); a full `train.py` run is the timing job.  `pomegranate==0.14.8` build for L2RM: see `slurm_logs/v9_env_pomegranate.log` (TO_VERIFY at the time of writing).

## 2. Data sources and licences (details in `DATA_AUDIT.csv`)
- **NCR data release** `https://ncr-paper.cdn.bcebos.com/data/NCR-data.tar` — 100,154,890,240 B (HTTP HEAD, last-modified 2022-04-14).  SCAN precomputed BUTD region features for Flickr30K and MS-COCO + CC152K + vocab + NCR's noise files.  Public CDN, no login; licence basis: features redistributed by the SCAN/NCR authors (code Apache-2.0/MIT); underlying images: Flickr30K (research terms; original images held locally), COCO (CC BY 4.0); CC152K from Conceptual Captions (Google terms; conditional unit only).  Download: 8 byte-range streams (`scripts/v9/fetch_ncr_parallel.sh`; ~28 MB/s aggregate measured at start; single stream 5 MB/s) → `data/ncr_precomp/zips/NCR-data.tar` → `scripts/data_fetch.py run --ids ncr_precomp` (sha256, member-safety check, extraction; the V9 branch carries its own `data/DATA_MANIFEST.json`).  Disk: 1.83 TB free at start; floor 1.2 TB; this item ≈ 100 GB tar + ≈ 100 GB extracted (the tar can be deleted after extraction — owner's call; both kept for now).
- **NCR checkpoints** (7 × ~145 MB, same CDN) — item `ncr_models`; evaluation-only.
- **L2RM / SGRAF checkpoints** — Google Drive only; not fetched (owner decision; `gdown` would be needed).
- Existing: Flickr30K images + CSV, COCO 2014 images + Karpathy json, BF/SHG (test sealed), EPFL RGB-NIR (terms unresolved), P49 CLIP cache (exploration only).

## 3. Independence of training / validation / test
- Host splits are the SCAN/Karpathy splits (Flickr30K 29,000 / 1,000 / 1,000; COCO 113,287 / 5,000 / 5,000); image-level; captions follow their image.  V9 anchor roles are drawn inside **train** only (image-level, 10 %, 40/30/15/15), before any injection; U = the remaining 90 %; donors for injected pairs come only from U.
- The host prescribes checkpoint selection on **dev** (official val) by rSum; the official **test** is read once per trained unit by the frozen evaluator (this is the benchmark's own protocol, pre-registered in V9-N2 before any unit runs; test is never used for any selection, rho estimation or critic fitting).
- Reference-moment / critic-select roles are image-disjoint from U and from each other (`role_disjointness` of the reference core); multi-caption rows share an image → cluster = image for every interval (package §03 C).
- Pre-training exposure: every public checkpoint (NCR, L2RM, SGRAF) was trained on the same precomp train split; V9 trains from scratch (SGR from random init; teacher on the anchor role only), so no released weights enter B/C tables.

## 4. Budget (to be completed from the timing job)
| item | unit count | per-unit cost | total |
|---|---|---|---|
| SGR one epoch, Flickr30K precomp, batch 128, 1 GPU | — | **PENDING** (`slurm/v9/v9_timing_sgr.sbatch`: seconds / epoch, peak memory) | — |
| N2 formal Flickr30K: 8 methods × 2 noise × 3 seeds, 40 epochs each (host default; NCR 40 + 5 warm-up, L2RM 35 + 5) | 48 | 40 × epoch_s (NCR ×2 networks) | PENDING |
| N2 formal COCO: same, 20 epochs; train ≈ 3.9 × Flickr per epoch | 48 | 20 × 3.9 × epoch_s | PENDING |
| N2 tuning (3 lr multipliers × 8 methods × 2 datasets × 2 noise, seed 917) | ≤ 96 | same per-epoch cost | PENDING |
| teachers (SGR on the anchor teacher role, 5 epochs; shared by all methods) | 2 datasets × 2 noise (× 1 seed) | 5 × 0.04 × epoch_s | small |
| NCR / L2RM native A-table reproductions | 2 methods × 2 datasets (× 1 seed at 0.4) | native epochs | PENDING |
| critic / kernel fits for the weight methods (cos-2 on scalar s; RuLSIF, Nyström on the reference roles) | per unit, minutes on GPU | — | small |
| REG-G1 (fork C) | 48 − reused clean full units | from R6 timings (critic 3,000 steps + registrations) | fork C reports |
Rule from the package: measure one epoch first, fill this table, then submit blocks (Flickr block first; COCO second) after the owner's budget confirmation; short jobs grouped (owner 2026-10-13/14).

## 5. Anchor budget statement (labelled information every B-table method sees)
Flickr30K: 2,900 anchor images (14,500 captions) = 10 % of train, split 1,160 teacher / 870 critic-fit / 435 critic-select / 435 reference-moments; COCO: 11,329 = 4,532 / 3,399 / 1,699 / 1,699.  These are the only "trusted" pairs; all methods (incl. NCR+A, L2RM+A, Unweighted+A) receive them with weight 1; the A table (native) does not use them.  Any statement is "trust-assisted", never a native NCR leaderboard claim (package §05 §1).

## 6. Items needing the owner's decision
1. Full N2 GPU budget (after the timing job fills §4) and whether COCO runs in the first round or Flickr only.
2. Google-Drive checkpoints (L2RM, SGRAF) — fetch or not.
3. Keep or delete the 100 GB `NCR-data.tar` after extraction.
4. Add DECL / CREAM (public code, same host) as external rows, or keep the package's method list.
5. EPFL RGB-NIR usage basis (REG-G2) and BF/SHG sealed test (new confirmation).
6. CC152K (NR-N3b) — conditional; its features are inside the same tar (no extra download).
