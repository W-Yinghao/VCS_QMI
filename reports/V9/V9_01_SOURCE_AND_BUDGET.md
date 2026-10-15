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
- **NCR data release** `https://ncr-paper.cdn.bcebos.com/data/NCR-data.tar` — 100,154,890,240 B (HTTP HEAD, last-modified 2022-04-14).  SCAN precomputed BUTD region features for Flickr30K and MS-COCO + CC152K + vocab + NCR's noise files.  Public CDN, no login.  **Usage basis (owner review 2026-10-15):** official release used for non-commercial research reproduction (the NCR README is the provenance for using the SCAN-pipeline feature package); original data terms, feature provenance and code licence are registered separately — Flickr30K images: DenotationGraph release for researchers / educators, non-commercial, under the Flickr Terms of Use (the page says "We do not own the copyright of the images"); COCO: images are NOT owned by the COCO Consortium and remain under the respective Flickr terms, only the annotations / website are CC BY 4.0; CC152K: Conceptual Captions terms; feature provenance: SCAN bottom-up-attention extraction redistributed by the NCR authors; code licence: NCR LICENSE file = MIT while the README says Apache-2.0 (inconsistency recorded) — the code licence does NOT cover images, captions or derived features.  No redistribution of images, captions or full features from our repo / deliverables; third-party authorisation is not replaced by this project's approval.  Download: 8 byte-range streams (`scripts/v9/fetch_ncr_parallel.sh`; ~28 MB/s aggregate measured at start; single stream 5 MB/s) → `data/ncr_precomp/zips/NCR-data.tar` → `scripts/data_fetch.py run --ids ncr_precomp` (sha256, member-safety check, extraction; the V9 branch carries its own `data/DATA_MANIFEST.json`).  Disk: 1.83 TB free at start; floor 1.2 TB; this item ≈ 100 GB tar + ≈ 100 GB extracted (the tar can be deleted after extraction — owner's call; both kept for now).
- **NCR checkpoints** (7 × ~145 MB, same CDN) — item `ncr_models`; evaluation-only.
- **L2RM / SGRAF checkpoints** — Google Drive only; not fetched (owner decision; `gdown` would be needed).
- **Flickr30K usage status.**  Official page (shannon.cs.illinois.edu/DenotationGraph, fetched 2026-10-15): images "for researchers and educators who wish to use the dataset for non-commercial research and/or educational purposes", "Use of the images must abide by the Flickr Terms of Use", "We do not own the copyright of the images", access by request form; no licence statement for the captions / graph.  Local copy: Hugging Face mirror `nlphuji/flickr30k` (revision 2b239bef, manifest item `flickr30k`, 2026-10-11).  **Owner decision 2026-10-15: every dataset already on the server is cleared for use in these experiments** ("所有的数据，只要是现在有了的，我们就有权限…直接开始相关实验") — recorded as the usage clearance for Flickr30K, the NCR precomputed features (provenance: the official NCR release) and COCO (images under Flickr terms, annotations CC BY 4.0).  The factual register above stays as is; no redistribution of images, captions or full features from our deliverables.
- Existing: COCO 2014 images (Flickr terms; annotations CC BY 4.0) + Karpathy json, BF/SHG (CC BY 4.0; test sealed), EPFL RGB-NIR (terms unresolved), P49 CLIP cache (exploration only).

## 3. Independence of training / validation / test
- Host splits are the SCAN/Karpathy splits (Flickr30K 29,000 / 1,000 / 1,000; COCO 113,287 / 5,000 / 5,000); image-level; captions follow their image.  V9 anchor roles are drawn inside **train** only (image-level, 10 %, 40/30/15/15), before any injection; U = the remaining 90 %; donors for injected pairs come only from U.
- The host prescribes checkpoint selection on **dev** (official val) by rSum; the official **test** is read once per trained unit by the frozen evaluator (this is the benchmark's own protocol, pre-registered in V9-N2 before any unit runs; test is never used for any selection, rho estimation or critic fitting).
- Reference-moment / critic-select roles are image-disjoint from U and from each other (`role_disjointness` of the reference core); multi-caption rows share an image → cluster = image for every interval (package §03 C).
- Pre-training exposure: every public checkpoint (NCR, L2RM, SGRAF) was trained on the same precomp train split; V9 trains from scratch (SGR from random init; teacher on the anchor role only), so no released weights enter B/C tables.

## 4. Budget — first block = Flickr30K only (owner 2026-10-15); COCO is a separate later decision
Per-method costs are **measured, not multiplied**: one grouped GPU job (`slurm/v9/v9_timing_group.sbatch`, ≤ 1 GPU-h, nice 400) runs, on Flickr30K precomp at noise 0.4 with the DECL/RCL index, (a) native SGR for 1 epoch including the host's per-epoch dev validation and a final test `evalrank`; (b) NCR with `--warmup_epoch 1 --num_epochs 2` (one warm-up epoch + one co-training epoch of the two networks, GMM fit, dev validation of the averaged model, final eval); (c) L2RM with `--warmup_epoch 1 --num_epochs 2` (warm-up + one rematching epoch with the cost learner and Sinkhorn, eval); (d) the shared teacher = native SGR on the 1,160-image teacher role for 5 epochs (measured directly, it is small); (e) weight generation = fork B's critic / kernel fits on the reference roles (measured by fork B's gate; seconds–minutes).  Each unit's cost = its own measured per-epoch time × its native epoch count (+ warm-up) + its evaluation.

| item (Flickr30K block) | units | per-unit cost (from the grouped timing job) | total |
|---|---|---|---|
| teacher (SGR on the teacher role, 5 epochs; shared by all B-table methods) | 2 (noise 0 / 0.4) × 1 | t_teacher = measured (d) | PENDING |
| weight generation (VCS / BCE / RuLSIF / Nyström / RankMass on the reference roles) | per weighted unit | t_w = fork B's measured fit time | PENDING |
| Unweighted+A, VCS-W, BCE-W, RuLSIF-W, Nyström-W, RankMass: SGR 40 epochs + dev val + final eval | 6 methods × 2 noise × 3 seeds = 36 | 40 × t_SGR_epoch + 40 × t_val + t_eval | PENDING |
| NCR+A: warm-up 5 + 40 co-training epochs (2 networks) + eval | 2 × 3 = 6 | 5 × t_NCR_warm + 40 × t_NCR_cotrain + t_eval | PENDING |
| L2RM+A: warm-up 5 + 35 epochs (OT + cost learner) + eval | 2 × 3 = 6 | 5 × t_L2RM_warm + 35 × t_L2RM_epoch + t_eval | PENDING |
| **formal total (48 units)** | 48 | — | PENDING |
| tuning (3 lr multipliers × 8 methods × 2 noise, seed 917) | ≤ 48 | same per-method costs | PENDING |
| native A-table reproductions (NCR, L2RM at 0.4; SGR clean) | 3 | native epochs | PENDING |
| released NCR checkpoints evaluation | 1 job | minutes | — |
| **COCO block** (48 formal + ≤ 48 tuning; train 3.9 × Flickr per epoch, 20 native epochs) | — | measured separately before any COCO decision | **not in this block** |
Rule: the grouped timing job fills the per-unit column; the Flickr30K block is launched by the main session as soon as this table, HOST_LOCK and fork D's pre-check table are in (owner 2026-10-15); short per-unit jobs are grouped (owner 2026-10-13/14).

## 5. Anchor budget statement (labelled information every B-table method sees)
Flickr30K: 2,900 anchor images (14,500 captions) = 10 % of train, split 1,160 teacher / 870 critic-fit / 435 critic-select / 435 reference-moments; COCO: 11,329 = 4,532 / 3,399 / 1,699 / 1,699.  These are the only "trusted" pairs; all methods (incl. NCR+A, L2RM+A, Unweighted+A) receive them with weight 1; the A table (native) does not use them.  Any statement is "trust-assisted", never a native NCR leaderboard claim (package §05 §1).

## 6. Items needing the owner's decision
1. COCO block: a separate later decision (the Flickr30K block is launched by the main session once the measured budget, HOST_LOCK and fork D's pre-check table are in — owner 2026-10-15, no further budget confirmation needed).
2. Google-Drive checkpoints (L2RM, SGRAF) — fetch or not.
3. Keep or delete the 100 GB `NCR-data.tar` after extraction.
4. Add DECL / CREAM (public code, same host) as external rows, or keep the package's method list.
5. EPFL RGB-NIR (on the server → cleared by the owner's 2026-10-15 decision for experiments; REG-G2 scope still a separate decision) and BF/SHG sealed test (new confirmation).
6. CC152K (NR-N3b) — conditional; its features are inside the same tar (no extra download).
