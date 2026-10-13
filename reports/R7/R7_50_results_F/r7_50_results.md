# R7-50 official-test evaluation, phase F (LLaVA-1.5-7B, D1-FULL selectors)

Scored 2026-10-13T23:09:08Z; predictions manifest sha256 `fb86c5e825680861f0344e34a65d65477a06faa4d27659332c11943fe108b771` (committed before scoring); freeze {"protocol": "reports/R7/R7_50_PROTOCOL_FROZEN_20261012.yaml", "sha256": "3671da46d16e2c4aeb2aca6b65b2de26f638cd9fbb545bee4d5e7738745f01fd", "freeze_key": "r7_50_freeze_20261012"}.

**Latency.** latency measured on the pre-fixed profiling workload: the frozen D1-30 timing results of the mapped validation benchmark (gqa_dev for GQA testdev-balanced, chartqa_val for ChartQA test; NVIDIA H100 NVL on nodesumo01, 8 forced decode steps, B* = 4 primary).  Not re-measured on test and not the test sets' actual mean end-to-end latency (it depends on input shapes, generation lengths and batch composition).

**Exposure (graded).**
- Relative to this project: the test splits were never used for selector training, hyperparameters or any selection; every row, checkpoint, operating point and statistic was fixed before the read.
- Relative to the hosts' public training data: LLaVA-1.5's instruction mix (llava_v1_5_mix665k.json) includes GQA train-balanced, Visual Genome, COCO and TextCaps.  GQA's test splits extend Visual Genome with new images whose scene graphs are hidden (Hudson & Manning 2019), so test-dev images need not be Visual Genome images: the build counts the test-dev image ids by prefix ('n'-prefixed vs numeric Visual Genome ids) and the test-dev images and (image, question) pairs present in the mix (sources.<bench>.exposure_check; the measured sentence is exposure_statement_measured).  ChartQA is not part of LLaVA-1.5's published instruction mix.
- Undisclosed upstream pre-training exposure (CLIP ViT-L/14-336, Vicuna) is not guaranteed either way.
- measured: GQA test-dev: 398 of 398 images have 'n'-prefixed ids (not Visual Genome ids) and 0 numeric Visual Genome ids; 0 appear in LLaVA-1.5's instruction mix by image id (0 under gqa/, 0 under vg/); 0 of 12578 (image, normalised question) pairs appear among its GQA turns.
- measured: ChartQA test: 0 entries of LLaVA-1.5's instruction mix have chart image paths.

- primary: S1 - S2 and S1 - S3 at L25 / L50 / L75 (B* = 4) and S1 - SparseVLM budget-matched at 144 / 288 / 432 equal tokens (D30-1); Holm across these primary contrasts of both benchmarks at each latency point, within this phase
- no latency claim against SparseVLM at B* = 4 (official code batch-1 only); S1 - SparseVLM latency-matched at B = 1 is secondary
- ChartQA human / augmented: quality and contrasts per split are secondary (unadjusted); the combined benchmark is primary
- no equivalence / non-inferiority claim (no margin set in advance)
- phase F: full / uniform-grid / SparseVLM rows are phase L's predictions (sealed under manifest_L), not regenerated

## Sources

- gqa_testdev: official test (closed source, one read); 12578 items, 398 groups; sha256 14039069c0b3c797c7aa9bcd5f4c2aa4b5976e02c0b6773e1a584d942a03a318
  - overlap check: {"images": 398, "numeric_image_ids": 0, "n_prefixed_image_ids": 398, "other_image_ids": 0, "images_in_mix_gqa_paths": 0, "images_in_mix_vg_paths": 0, "images_in_mix_any": 0, "questions": 12578, "questions_seen_in_mix_gqa_turns": 0}
- chartqa_test: official test (closed source, one read); 2500 items, 1509 groups; sha256 84889bae30aaf6e24ff899e1604221606c0b65e658b6c70a7d02c79f50fc87df; splits {'human': 1250, 'augmented': 1250}
  - overlap check: {"mix_entries_with_chart_image_paths": 0, "note": "ChartQA is not part of LLaVA-1.5's published instruction mix"}

## Primary contrasts (points; B* = 4 latency points from the profiling workload; Holm per latency point)

| bench | contrast | Δ | 95 % CI | p | p_holm | seed Δ | basis |
|---|---|---|---|---|---|---|---|
| gqa_testdev | S1-S2@L25 | -0.78 | [-1.2, -0.4] | 0.0010 | 0.0060 | -0.2, +0.2, -2.4 | L25 = uniform-grid 144-token per-item latency at B = 4 on the profiling workload (64.38 ms) |
| gqa_testdev | S1-S3@L25 | +2.26 | [1.7, 2.8] | 0.0010 | 0.0060 | +0.6, +9.9, -3.7 | L25 = uniform-grid 144-token per-item latency at B = 4 on the profiling workload (64.38 ms) |
| gqa_testdev | S1-SparseVLM@L25[budget] | -3.06 | [-3.7, -2.4] | 0.0010 | 0.0060 | -3.2, -2.4, -3.6 | equal-token budget 144 (SparseVLM row sparsevlm:144:adapted) |
| gqa_testdev | S1-S2@L50 | -0.27 | [-0.6, 0.0] | 0.0780 | 0.3118 | +0.2, +0.2, -1.2 | L50 = uniform-grid 288-token per-item latency at B = 4 on the profiling workload (71.01 ms) |
| gqa_testdev | S1-S3@L50 | +0.05 | [-0.4, 0.5] | 0.8296 | 0.9625 | -0.4, +2.6, -2.1 | L50 = uniform-grid 288-token per-item latency at B = 4 on the profiling workload (71.01 ms) |
| gqa_testdev | S1-SparseVLM@L50[budget] | -1.47 | [-2.0, -1.0] | 0.0010 | 0.0060 | -1.7, -1.1, -1.7 | equal-token budget 288 (SparseVLM row sparsevlm:288:adapted) |
| gqa_testdev | S1-S2@L75 | +0.08 | [-0.1, 0.2] | 0.3548 | 1.0000 | +0.4, +0.1, -0.3 | L75 = uniform-grid 432-token per-item latency at B = 4 on the profiling workload (77.99 ms) |
| gqa_testdev | S1-S3@L75 | -0.02 | [-0.3, 0.2] | 0.8836 | 1.0000 | -0.0, +0.5, -0.6 | L75 = uniform-grid 432-token per-item latency at B = 4 on the profiling workload (77.99 ms) |
| gqa_testdev | S1-SparseVLM@L75[budget] | -0.01 | [-0.4, 0.3] | 0.9105 | 1.0000 | +0.2, -0.0, -0.2 | equal-token budget 432 (SparseVLM row sparsevlm:432:adapted) |
| chartqa_test | S1-S2@L25 | +0.75 | [0.3, 1.2] | 0.0020 | 0.0060 | +2.2, -0.1, +0.2 | L25 = uniform-grid 144-token per-item latency at B = 4 on the profiling workload (66.95 ms) |
| chartqa_test | S1-S3@L25 | +0.09 | [-0.4, 0.6] | 0.7356 | 0.7356 | +1.0, +0.7, -1.4 | L25 = uniform-grid 144-token per-item latency at B = 4 on the profiling workload (66.95 ms) |
| chartqa_test | S1-SparseVLM@L25[budget] | -1.99 | [-2.9, -1.2] | 0.0010 | 0.0060 | -1.2, -3.3, -1.4 | equal-token budget 144 (SparseVLM row sparsevlm:144:adapted) |
| chartqa_test | S1-S2@L50 | +0.21 | [-0.2, 0.6] | 0.3208 | 0.9625 | +2.0, -0.4, -1.0 | L50 = uniform-grid 288-token per-item latency at B = 4 on the profiling workload (73.99 ms) |
| chartqa_test | S1-S3@L50 | -0.20 | [-0.7, 0.3] | 0.4368 | 0.9625 | +0.6, -0.2, -1.1 | L50 = uniform-grid 288-token per-item latency at B = 4 on the profiling workload (73.99 ms) |
| chartqa_test | S1-SparseVLM@L50[budget] | -1.49 | [-2.2, -0.7] | 0.0010 | 0.0060 | -0.9, -2.3, -1.3 | equal-token budget 288 (SparseVLM row sparsevlm:288:adapted) |
| chartqa_test | S1-S2@L75 | -0.15 | [-0.4, 0.1] | 0.2499 | 1.0000 | +0.4, -0.4, -0.4 | L75 = uniform-grid 432-token per-item latency at B = 4 on the profiling workload (80.78 ms) |
| chartqa_test | S1-S3@L75 | -0.09 | [-0.5, 0.3] | 0.6317 | 1.0000 | +0.0, -0.2, -0.1 | L75 = uniform-grid 432-token per-item latency at B = 4 on the profiling workload (80.78 ms) |
| chartqa_test | S1-SparseVLM@L75[budget] | -0.13 | [-0.6, 0.4] | 0.5797 | 1.0000 | -0.1, -0.2, -0.0 | equal-token budget 432 (SparseVLM row sparsevlm:432:adapted) |

## gqa_testdev: fixed rows (n = 12578, 398 groups)

| row | visual tokens | quality [95 % CI] | retention (rel.) | latency B=4 / B=1, ms (profiling workload) |
|---|---|---|---|---|
| sel:vcs_lr0.0001_s0:72 | 72.0 | 52.72 [51.5, 54.0] | 0.851 | 62.5 / 155.5 |
| sel:vcs_lr0.0001_s0:144 | 144.0 | 55.61 [54.3, 56.9] | 0.898 | 64.9 / 156.0 |
| sel:vcs_lr0.0001_s0:288 | 288.0 | 59.05 [57.6, 60.5] | 0.953 | 72.0 / 160.3 |
| sel:vcs_lr0.0001_s0:432 | 432.0 | 61.72 [60.3, 63.1] | 0.996 | 79.1 / 166.9 |
| sel:vcs_lr0.0001_s1:72 | 72.0 | 53.58 [52.3, 54.9] | 0.865 | 62.5 / 155.5 |
| sel:vcs_lr0.0001_s1:144 | 144.0 | 56.34 [55.0, 57.7] | 0.910 | 64.9 / 156.0 |
| sel:vcs_lr0.0001_s1:288 | 288.0 | 59.60 [58.2, 61.1] | 0.962 | 72.0 / 160.3 |
| sel:vcs_lr0.0001_s1:432 | 432.0 | 61.48 [60.1, 63.0] | 0.992 | 79.1 / 166.9 |
| sel:vcs_lr0.0001_s2:72 | 72.0 | 52.23 [51.0, 53.4] | 0.843 | 62.5 / 155.5 |
| sel:vcs_lr0.0001_s2:144 | 144.0 | 55.23 [53.9, 56.5] | 0.892 | 64.9 / 156.0 |
| sel:vcs_lr0.0001_s2:288 | 288.0 | 59.05 [57.7, 60.4] | 0.953 | 72.0 / 160.3 |
| sel:vcs_lr0.0001_s2:432 | 432.0 | 61.35 [59.9, 62.8] | 0.990 | 79.1 / 166.9 |
| sel:logistic_lr0.0001_s0:72 | 72.0 | 53.14 [51.9, 54.4] | 0.858 | 62.5 / 155.5 |
| sel:logistic_lr0.0001_s0:144 | 144.0 | 55.76 [54.6, 57.0] | 0.900 | 64.9 / 156.0 |
| sel:logistic_lr0.0001_s0:288 | 288.0 | 58.82 [57.4, 60.3] | 0.950 | 72.0 / 160.3 |
| sel:logistic_lr0.0001_s0:432 | 432.0 | 61.27 [59.8, 62.8] | 0.989 | 79.1 / 166.9 |
| sel:logistic_lr0.0001_s1:72 | 72.0 | 52.81 [51.5, 54.1] | 0.853 | 62.5 / 155.5 |
| sel:logistic_lr0.0001_s1:144 | 144.0 | 56.30 [55.0, 57.6] | 0.909 | 64.9 / 156.0 |
| sel:logistic_lr0.0001_s1:288 | 288.0 | 59.39 [58.0, 60.8] | 0.959 | 72.0 / 160.3 |
| sel:logistic_lr0.0001_s1:432 | 432.0 | 61.41 [60.0, 62.9] | 0.991 | 79.1 / 166.9 |
| sel:logistic_lr0.0001_s2:72 | 72.0 | 54.32 [53.0, 55.7] | 0.877 | 62.5 / 155.5 |
| sel:logistic_lr0.0001_s2:144 | 144.0 | 57.66 [56.4, 59.0] | 0.931 | 64.9 / 156.0 |
| sel:logistic_lr0.0001_s2:288 | 288.0 | 60.01 [58.6, 61.4] | 0.969 | 72.0 / 160.3 |
| sel:logistic_lr0.0001_s2:432 | 432.0 | 61.49 [60.1, 62.9] | 0.993 | 79.1 / 166.9 |
| sel:direct_lr0.0001_s0:72 | 72.0 | 52.62 [51.3, 54.0] | 0.849 | 62.5 / 155.5 |
| sel:direct_lr0.0001_s0:144 | 144.0 | 54.83 [53.6, 56.2] | 0.885 | 64.9 / 156.0 |
| sel:direct_lr0.0001_s0:288 | 288.0 | 59.64 [58.3, 61.0] | 0.963 | 72.0 / 160.3 |
| sel:direct_lr0.0001_s0:432 | 432.0 | 61.62 [60.2, 63.1] | 0.995 | 79.1 / 166.9 |
| sel:direct_lr0.0001_s1:72 | 72.0 | 43.36 [42.2, 44.5] | 0.700 | 62.5 / 155.5 |
| sel:direct_lr0.0001_s1:144 | 144.0 | 46.57 [45.3, 47.9] | 0.752 | 64.9 / 156.0 |
| sel:direct_lr0.0001_s1:288 | 288.0 | 58.14 [56.7, 59.6] | 0.939 | 72.0 / 160.3 |
| sel:direct_lr0.0001_s1:432 | 432.0 | 61.11 [59.6, 62.6] | 0.987 | 79.1 / 166.9 |
| sel:direct_lr0.0001_s2:72 | 72.0 | 56.33 [55.1, 57.7] | 0.909 | 62.5 / 155.5 |
| sel:direct_lr0.0001_s2:144 | 144.0 | 58.84 [57.5, 60.2] | 0.950 | 64.9 / 156.0 |
| sel:direct_lr0.0001_s2:288 | 288.0 | 60.88 [59.5, 62.3] | 0.983 | 72.0 / 160.3 |
| sel:direct_lr0.0001_s2:432 | 432.0 | 61.68 [60.3, 63.1] | 0.996 | 79.1 / 166.9 |
| dyn:vcs_lr0.0001_s0:0.25 | 167.6 | 55.43 [54.2, 56.7] | 0.895 | 68.9 / 163.9 |
| dyn:vcs_lr0.0001_s0:0.5 | 347.2 | 58.91 [57.6, 60.2] | 0.951 | 76.4 / 167.7 |
| dyn:vcs_lr0.0001_s0:0.75 | 487.1 | 60.99 [59.5, 62.4] | 0.984 | 83.6 / 172.7 |
| dyn:vcs_lr0.0001_s1:0.25 | 130.2 | 55.27 [54.0, 56.6] | 0.892 | 68.4 / 161.9 |
| dyn:vcs_lr0.0001_s1:0.5 | 290.0 | 58.37 [57.0, 59.7] | 0.942 | 75.0 / 168.3 |
| dyn:vcs_lr0.0001_s1:0.75 | 465.9 | 60.96 [59.6, 62.4] | 0.984 | 82.8 / 173.1 |
| dyn:vcs_lr0.0001_s2:0.25 | 145.8 | 54.53 [53.3, 55.8] | 0.880 | 67.7 / 161.9 |
| dyn:vcs_lr0.0001_s2:0.5 | 314.0 | 58.28 [56.9, 59.6] | 0.941 | 74.5 / 167.4 |
| dyn:vcs_lr0.0001_s2:0.75 | 464.3 | 60.70 [59.3, 62.0] | 0.980 | 82.6 / 173.4 |
| full | 576.0 | 61.95 [60.5, 63.4] | 1.000 | 87.7 / 173.4 |
| static:grid:72 | 72.0 | 56.26 [55.0, 57.7] | 0.908 | 61.7 / 154.7 |
| static:grid:144 | 144.0 | 58.70 [57.4, 60.0] | 0.948 | 64.4 / 154.8 |
| static:grid:288 | 288.0 | 60.29 [58.9, 61.7] | 0.973 | 71.0 / 159.0 |
| static:grid:432 | 432.0 | 60.92 [59.6, 62.4] | 0.983 | 78.0 / 165.4 |
| sparsevlm:192:official | 192.0 | 59.37 [58.0, 60.8] | 0.958 | — / 205.6 |
| sparsevlm:128:official | 128.0 | 58.35 [57.0, 59.8] | 0.942 | — / 205.3 |
| sparsevlm:96:official | 96.0 | 56.98 [55.7, 58.4] | 0.920 | — / 203.7 |
| sparsevlm:64:official | 64.0 | 53.74 [52.5, 55.0] | 0.867 | — / 196.6 |
| sparsevlm:72:adapted | 72.0 | 54.83 [53.6, 56.2] | 0.885 | — / 203.7 |
| sparsevlm:144:adapted | 144.0 | 58.79 [57.5, 60.2] | 0.949 | — / 198.3 |
| sparsevlm:288:adapted | 288.0 | 60.70 [59.3, 62.1] | 0.980 | — / 203.2 |
| sparsevlm:432:adapted | 432.0 | 61.53 [60.1, 63.0] | 0.993 | — / 206.2 |

Quality at the latency points (seed mean; B = batch size of the profiling workload):

| B | family | L25 | L50 | L75 |
|---|---|---|---|---|
| 1 | S1 | not reached | 58.13 | 60.98 |
| 1 | S2 | not reached | 58.52 | 60.92 |
| 1 | S3 | not reached | 57.63 | 61.02 |
| 1 | S1_dyn | not reached | not reached | 56.85 |
| 1 | grid | 58.70 | 60.29 | 60.92 |
| 1 | SparseVLM | not reached | not reached | not reached |
| 4 | S1 | 55.06 | 58.74 | 61.15 |
| 4 | S2 | 55.85 | 59.01 | 61.07 |
| 4 | S3 | 52.80 | 58.69 | 61.17 |
| 4 | S1_dyn | not reached | 56.42 | 59.34 |
| 4 | grid | 58.70 | 60.29 | 60.92 |
| 16 | S1 | 55.74 | 58.96 | 61.39 |
| 16 | S2 | 56.58 | 59.18 | 61.28 |
| 16 | S3 | 53.42 | 59.06 | 61.37 |
| 16 | S1_dyn | not reached | 57.49 | 60.40 |
| 16 | grid | 58.70 | 60.29 | 60.92 |
| 32 | S1 | 55.13 | 59.25 | 61.22 |
| 32 | S2 | 55.92 | 59.42 | 61.13 |
| 32 | S3 | 52.86 | 59.57 | 61.22 |
| 32 | S1_dyn | not reached | 57.86 | 60.12 |
| 32 | grid | 58.70 | 60.29 | 60.92 |

S1 dynamic rows: kept tokens on test vs the profiling workload (B = 4):

| row | mean kept (test) | share 72/144/288/432/576 (test) | mean kept (profiling) |
|---|---|---|---|
| dyn:vcs_lr0.0001_s0:0.25 | 167.6 | 0.66 / 0.08 / 0.10 / 0.11 / 0.06 | 167.8 |
| dyn:vcs_lr0.0001_s0:0.5 | 347.2 | 0.23 / 0.09 / 0.13 / 0.25 / 0.30 | 315.4 |
| dyn:vcs_lr0.0001_s0:0.75 | 487.1 | 0.05 / 0.05 / 0.06 / 0.16 / 0.68 | 447.0 |
| dyn:vcs_lr0.0001_s1:0.25 | 130.2 | 0.77 / 0.07 / 0.07 / 0.07 / 0.03 | 151.9 |
| dyn:vcs_lr0.0001_s1:0.5 | 290.0 | 0.36 / 0.11 / 0.11 / 0.17 / 0.25 | 283.5 |
| dyn:vcs_lr0.0001_s1:0.75 | 465.9 | 0.08 / 0.07 / 0.07 / 0.11 / 0.66 | 429.3 |
| dyn:vcs_lr0.0001_s2:0.25 | 145.8 | 0.67 / 0.11 / 0.11 / 0.08 / 0.02 | 137.7 |
| dyn:vcs_lr0.0001_s2:0.5 | 314.0 | 0.25 / 0.13 / 0.17 / 0.22 / 0.24 | 276.9 |
| dyn:vcs_lr0.0001_s2:0.75 | 464.3 | 0.07 / 0.06 / 0.09 / 0.16 / 0.62 | 430.2 |

Secondary contrasts (unadjusted):

| contrast | Δ | 95 % CI | p |
|---|---|---|---|
| S1-grid@L25 | -3.63 | [-4.3, -3.0] | 0.0010 |
| S1-SparseVLM@L25[latency_b1] | not reached | | |
| S1-grid@L50 | -1.55 | [-2.0, -1.0] | 0.0010 |
| S1-SparseVLM@L50[latency_b1] | not reached | | |
| S1-grid@L75 | +0.24 | [-0.1, 0.6] | 0.1899 |
| S1-SparseVLM@L75[latency_b1] | not reached | | |
| A3:S1_dyn-fixed@f0.25 | -0.68 | [-1.0, -0.3] | 0.0010 |
| A3:S1_dyn-fixed@f0.5 | -1.22 | [-1.6, -0.9] | 0.0010 |
| A3:S1_dyn-fixed@f0.75 | -0.74 | [-1.0, -0.5] | 0.0010 |

Integrity: {"dropped_lines": 0, "blank_lines": 0, "duplicate_qids": 0}; files per row ≤ 6; SparseVLM counter records 24, outside tolerance / unmeasured 0.

## chartqa_test: fixed rows (n = 2500, 1509 groups)

| row | visual tokens | quality [95 % CI] | human | augmented | retention (rel.) | latency B=4 / B=1, ms (profiling workload) |
|---|---|---|---|---|---|---|
| sel:vcs_lr0.0001_s0:72 | 72.0 | 14.56 [13.1, 16.1] | 16.64 | 12.48 | 0.802 | 65.6 / 157.8 |
| sel:vcs_lr0.0001_s0:144 | 144.0 | 15.32 [13.9, 16.9] | 18.56 | 12.08 | 0.844 | 68.2 / 158.7 |
| sel:vcs_lr0.0001_s0:288 | 288.0 | 16.56 [15.1, 18.1] | 19.76 | 13.36 | 0.912 | 75.4 / 163.2 |
| sel:vcs_lr0.0001_s0:432 | 432.0 | 17.64 [16.1, 19.2] | 21.52 | 13.76 | 0.971 | 82.5 / 168.4 |
| sel:vcs_lr0.0001_s1:72 | 72.0 | 12.56 [11.3, 14.0] | 14.00 | 11.12 | 0.692 | 65.6 / 157.8 |
| sel:vcs_lr0.0001_s1:144 | 144.0 | 13.28 [12.0, 14.7] | 14.96 | 11.60 | 0.731 | 68.2 / 158.7 |
| sel:vcs_lr0.0001_s1:288 | 288.0 | 15.12 [13.7, 16.6] | 18.40 | 11.84 | 0.833 | 75.4 / 163.2 |
| sel:vcs_lr0.0001_s1:432 | 432.0 | 17.52 [16.1, 19.1] | 20.64 | 14.40 | 0.965 | 82.5 / 168.4 |
| sel:vcs_lr0.0001_s2:72 | 72.0 | 13.96 [12.7, 15.4] | 16.64 | 11.28 | 0.769 | 65.6 / 157.8 |
| sel:vcs_lr0.0001_s2:144 | 144.0 | 15.12 [13.7, 16.7] | 18.32 | 11.92 | 0.833 | 68.2 / 158.7 |
| sel:vcs_lr0.0001_s2:288 | 288.0 | 16.16 [14.7, 17.7] | 19.84 | 12.48 | 0.890 | 75.4 / 163.2 |
| sel:vcs_lr0.0001_s2:432 | 432.0 | 17.72 [16.2, 19.3] | 21.52 | 13.92 | 0.976 | 82.5 / 168.4 |
| sel:logistic_lr0.0001_s0:72 | 72.0 | 12.60 [11.4, 14.0] | 14.40 | 10.80 | 0.694 | 65.6 / 157.8 |
| sel:logistic_lr0.0001_s0:144 | 144.0 | 12.96 [11.6, 14.4] | 15.28 | 10.64 | 0.714 | 68.2 / 158.7 |
| sel:logistic_lr0.0001_s0:288 | 288.0 | 14.68 [13.3, 16.1] | 17.76 | 11.60 | 0.808 | 75.4 / 163.2 |
| sel:logistic_lr0.0001_s0:432 | 432.0 | 17.72 [16.2, 19.3] | 21.44 | 14.00 | 0.976 | 82.5 / 168.4 |
| sel:logistic_lr0.0001_s1:72 | 72.0 | 12.68 [11.4, 14.1] | 14.80 | 10.56 | 0.698 | 65.6 / 157.8 |
| sel:logistic_lr0.0001_s1:144 | 144.0 | 13.36 [12.0, 14.9] | 15.68 | 11.04 | 0.736 | 68.2 / 158.7 |
| sel:logistic_lr0.0001_s1:288 | 288.0 | 15.56 [14.2, 17.0] | 19.28 | 11.84 | 0.857 | 75.4 / 163.2 |
| sel:logistic_lr0.0001_s1:432 | 432.0 | 17.96 [16.4, 19.6] | 21.76 | 14.16 | 0.989 | 82.5 / 168.4 |
| sel:logistic_lr0.0001_s2:72 | 72.0 | 13.76 [12.5, 15.2] | 16.48 | 11.04 | 0.758 | 65.6 / 157.8 |
| sel:logistic_lr0.0001_s2:144 | 144.0 | 14.96 [13.6, 16.4] | 18.48 | 11.44 | 0.824 | 68.2 / 158.7 |
| sel:logistic_lr0.0001_s2:288 | 288.0 | 17.40 [15.9, 19.0] | 21.20 | 13.60 | 0.958 | 75.4 / 163.2 |
| sel:logistic_lr0.0001_s2:432 | 432.0 | 17.84 [16.4, 19.4] | 21.44 | 14.24 | 0.982 | 82.5 / 168.4 |
| sel:direct_lr0.0001_s0:72 | 72.0 | 13.72 [12.4, 15.2] | 16.48 | 10.96 | 0.756 | 65.6 / 157.8 |
| sel:direct_lr0.0001_s0:144 | 144.0 | 14.24 [12.9, 15.6] | 17.44 | 11.04 | 0.784 | 68.2 / 158.7 |
| sel:direct_lr0.0001_s0:288 | 288.0 | 16.04 [14.6, 17.7] | 19.28 | 12.80 | 0.883 | 75.4 / 163.2 |
| sel:direct_lr0.0001_s0:432 | 432.0 | 17.76 [16.3, 19.3] | 21.12 | 14.40 | 0.978 | 82.5 / 168.4 |
| sel:direct_lr0.0001_s1:72 | 72.0 | 12.60 [11.3, 14.0] | 14.24 | 10.96 | 0.694 | 65.6 / 157.8 |
| sel:direct_lr0.0001_s1:144 | 144.0 | 11.92 [10.7, 13.3] | 13.52 | 10.32 | 0.656 | 68.2 / 158.7 |
| sel:direct_lr0.0001_s1:288 | 288.0 | 15.64 [14.2, 17.2] | 18.32 | 12.96 | 0.861 | 75.4 / 163.2 |
| sel:direct_lr0.0001_s1:432 | 432.0 | 17.56 [16.1, 19.2] | 21.04 | 14.08 | 0.967 | 82.5 / 168.4 |
| sel:direct_lr0.0001_s2:72 | 72.0 | 15.44 [14.1, 16.9] | 18.64 | 12.24 | 0.850 | 65.6 / 157.8 |
| sel:direct_lr0.0001_s2:144 | 144.0 | 16.48 [15.0, 18.0] | 19.52 | 13.44 | 0.907 | 68.2 / 158.7 |
| sel:direct_lr0.0001_s2:288 | 288.0 | 17.16 [15.7, 18.8] | 20.80 | 13.52 | 0.945 | 75.4 / 163.2 |
| sel:direct_lr0.0001_s2:432 | 432.0 | 17.60 [16.1, 19.2] | 21.44 | 13.76 | 0.969 | 82.5 / 168.4 |
| dyn:vcs_lr0.0001_s0:0.25 | 120.9 | 15.12 [13.7, 16.7] | 17.36 | 12.88 | 0.833 | 69.8 / 166.4 |
| dyn:vcs_lr0.0001_s0:0.5 | 311.1 | 16.88 [15.4, 18.5] | 20.00 | 13.76 | 0.930 | 79.6 / 172.4 |
| dyn:vcs_lr0.0001_s0:0.75 | 477.2 | 17.92 [16.4, 19.5] | 21.52 | 14.32 | 0.987 | 88.7 / 180.3 |
| dyn:vcs_lr0.0001_s1:0.25 | 133.6 | 13.28 [12.0, 14.7] | 15.04 | 11.52 | 0.731 | 70.6 / 166.7 |
| dyn:vcs_lr0.0001_s1:0.5 | 312.3 | 15.64 [14.2, 17.2] | 18.24 | 13.04 | 0.861 | 79.2 / 173.8 |
| dyn:vcs_lr0.0001_s1:0.75 | 484.1 | 17.48 [16.0, 19.0] | 21.44 | 13.52 | 0.963 | 88.5 / 179.7 |
| dyn:vcs_lr0.0001_s2:0.25 | 118.0 | 14.52 [13.1, 16.0] | 17.28 | 11.76 | 0.800 | 69.8 / 166.7 |
| dyn:vcs_lr0.0001_s2:0.5 | 256.8 | 15.32 [13.9, 16.8] | 18.32 | 12.32 | 0.844 | 75.8 / 170.9 |
| dyn:vcs_lr0.0001_s2:0.75 | 433.3 | 16.88 [15.4, 18.4] | 20.96 | 12.80 | 0.930 | 85.3 / 178.2 |
| full | 576.0 | 18.16 [16.7, 19.7] | 22.08 | 14.24 | 1.000 | 90.9 / 177.3 |
| static:grid:72 | 72.0 | 13.52 [12.2, 15.0] | 15.76 | 11.28 | 0.744 | 64.6 / 157.4 |
| static:grid:144 | 144.0 | 14.80 [13.3, 16.4] | 17.76 | 11.84 | 0.815 | 67.0 / 157.1 |
| static:grid:288 | 288.0 | 16.92 [15.5, 18.5] | 20.48 | 13.36 | 0.932 | 74.0 / 162.0 |
| static:grid:432 | 432.0 | 17.44 [15.9, 19.0] | 21.12 | 13.76 | 0.960 | 80.8 / 168.2 |
| sparsevlm:192:official | 192.0 | 17.60 [16.1, 19.3] | 20.72 | 14.48 | 0.969 | — / 202.7 |
| sparsevlm:128:official | 128.0 | 15.96 [14.5, 17.4] | 18.40 | 13.52 | 0.879 | — / 200.5 |
| sparsevlm:96:official | 96.0 | 15.24 [13.8, 16.7] | 18.40 | 12.08 | 0.839 | — / 200.6 |
| sparsevlm:64:official | 64.0 | 14.08 [12.7, 15.5] | 16.48 | 11.68 | 0.775 | — / 199.6 |
| sparsevlm:72:adapted | 72.0 | 14.36 [12.9, 15.9] | 16.96 | 11.76 | 0.791 | — / 198.8 |
| sparsevlm:144:adapted | 144.0 | 16.56 [15.2, 18.1] | 19.44 | 13.68 | 0.912 | — / 201.6 |
| sparsevlm:288:adapted | 288.0 | 17.44 [15.9, 19.1] | 20.72 | 14.16 | 0.960 | — / 205.0 |
| sparsevlm:432:adapted | 432.0 | 17.76 [16.3, 19.3] | 21.92 | 13.60 | 0.978 | — / 208.8 |

Quality at the latency points (seed mean; B = batch size of the profiling workload):

| B | family | L25 | L50 | L75 |
|---|---|---|---|---|
| 1 | S1 | not reached | 15.56 | 17.57 |
| 1 | S2 | not reached | 15.29 | 17.77 |
| 1 | S3 | not reached | 15.70 | 17.59 |
| 1 | S1_dyn | not reached | not reached | 14.75 |
| 1 | grid | 14.80 | 16.92 | 17.44 |
| 1 | SparseVLM | not reached | not reached | not reached |
| 4 | S1 | 14.16 | 15.68 | 17.23 |
| 4 | S2 | 13.41 | 15.47 | 17.38 |
| 4 | S3 | 14.08 | 15.89 | 17.32 |
| 4 | S1_dyn | not reached | 15.05 | 16.37 |
| 4 | grid | 14.80 | 16.92 | 17.44 |
| 16 | S1 | 14.32 | 15.88 | 17.46 |
| 16 | S2 | 13.55 | 15.78 | 17.65 |
| 16 | S3 | 14.13 | 16.18 | 17.51 |
| 16 | S1_dyn | not reached | 15.50 | 16.99 |
| 16 | grid | 14.80 | 16.92 | 17.44 |
| 32 | S1 | 14.48 | 15.88 | 17.54 |
| 32 | S2 | 13.68 | 15.77 | 17.74 |
| 32 | S3 | 14.18 | 16.17 | 17.57 |
| 32 | S1_dyn | not reached | 15.60 | 17.03 |
| 32 | grid | 14.80 | 16.92 | 17.44 |

S1 dynamic rows: kept tokens on test vs the profiling workload (B = 4):

| row | mean kept (test) | share 72/144/288/432/576 (test) | mean kept (profiling) |
|---|---|---|---|
| dyn:vcs_lr0.0001_s0:0.25 | 120.9 | 0.76 / 0.09 / 0.08 / 0.06 / 0.01 | 117.3 |
| dyn:vcs_lr0.0001_s0:0.5 | 311.1 | 0.28 / 0.11 / 0.16 / 0.22 / 0.24 | 316.1 |
| dyn:vcs_lr0.0001_s0:0.75 | 477.2 | 0.07 / 0.05 / 0.08 / 0.11 / 0.69 | 479.7 |
| dyn:vcs_lr0.0001_s1:0.25 | 133.6 | 0.75 / 0.08 / 0.09 / 0.06 / 0.03 | 135.7 |
| dyn:vcs_lr0.0001_s1:0.5 | 312.3 | 0.27 / 0.13 / 0.14 / 0.23 / 0.23 | 309.2 |
| dyn:vcs_lr0.0001_s1:0.75 | 484.1 | 0.06 / 0.05 / 0.07 / 0.12 / 0.69 | 477.8 |
| dyn:vcs_lr0.0001_s2:0.25 | 118.0 | 0.78 / 0.08 / 0.08 / 0.05 / 0.01 | 115.2 |
| dyn:vcs_lr0.0001_s2:0.5 | 256.8 | 0.36 / 0.17 / 0.14 / 0.18 / 0.15 | 240.5 |
| dyn:vcs_lr0.0001_s2:0.75 | 433.3 | 0.10 / 0.09 / 0.10 / 0.16 / 0.54 | 416.2 |

Secondary contrasts (unadjusted):

| contrast | Δ | 95 % CI | p |
|---|---|---|---|
| S1-grid@L25 | -0.64 | [-1.4, 0.1] | 0.0940 |
| S1-SparseVLM@L25[latency_b1] | not reached | | |
| S1-grid@L50 | -1.24 | [-2.0, -0.5] | 0.0010 |
| S1-SparseVLM@L50[latency_b1] | not reached | | |
| S1-grid@L75 | -0.21 | [-0.9, 0.4] | 0.5107 |
| S1-SparseVLM@L75[latency_b1] | not reached | | |
| A3:S1_dyn-fixed@f0.25 | -0.01 | [-0.3, 0.3] | 0.9305 |
| A3:S1_dyn-fixed@f0.5 | -0.12 | [-0.6, 0.3] | 0.6287 |
| A3:S1_dyn-fixed@f0.75 | -0.33 | [-0.8, 0.1] | 0.1369 |

ChartQA human (n = 1250; secondary):

| contrast | Δ | 95 % CI | p |
|---|---|---|---|
| S1-S2@L25 | +0.68 | [0.1, 1.3] | 0.0310 |
| S1-S3@L25 | -0.08 | [-0.8, 0.7] | 0.8476 |
| S1-grid@L25 | -1.19 | [-2.3, -0.1] | 0.0370 |
| S1-SparseVLM@L25[budget] | -2.16 | [-3.3, -1.0] | 0.0010 |
| S1-SparseVLM@L25[latency_b1] | not reached | | |
| S1-S2@L50 | +0.09 | [-0.5, 0.7] | 0.8066 |
| S1-S3@L50 | -0.02 | [-0.8, 0.8] | 0.9745 |
| S1-grid@L50 | -1.54 | [-2.6, -0.4] | 0.0050 |
| S1-SparseVLM@L50[budget] | -1.39 | [-2.5, -0.3] | 0.0110 |
| S1-SparseVLM@L50[latency_b1] | not reached | | |
| S1-S2@L75 | -0.26 | [-0.6, 0.1] | 0.0930 |
| S1-S3@L75 | -0.01 | [-0.6, 0.6] | 0.9705 |
| S1-grid@L75 | -0.34 | [-1.3, 0.5] | 0.4918 |
| S1-SparseVLM@L75[budget] | -0.69 | [-1.4, -0.0] | 0.0410 |
| S1-SparseVLM@L75[latency_b1] | not reached | | |
| A3:S1_dyn-fixed@f0.25 | -0.21 | [-0.7, 0.3] | 0.4508 |
| A3:S1_dyn-fixed@f0.5 | -0.49 | [-1.2, 0.3] | 0.1929 |
| A3:S1_dyn-fixed@f0.75 | -0.12 | [-0.8, 0.5] | 0.7016 |

ChartQA augmented (n = 1250; secondary):

| contrast | Δ | 95 % CI | p |
|---|---|---|---|
| S1-S2@L25 | +0.83 | [0.3, 1.3] | 0.0020 |
| S1-S3@L25 | +0.25 | [-0.4, 0.9] | 0.4218 |
| S1-grid@L25 | -0.09 | [-1.1, 1.0] | 0.9155 |
| S1-SparseVLM@L25[budget] | -1.81 | [-3.1, -0.5] | 0.0040 |
| S1-SparseVLM@L25[latency_b1] | not reached | | |
| S1-S2@L50 | +0.33 | [-0.3, 0.9] | 0.2859 |
| S1-S3@L50 | -0.38 | [-1.1, 0.3] | 0.2679 |
| S1-grid@L50 | -0.93 | [-1.9, 0.1] | 0.0710 |
| S1-SparseVLM@L50[budget] | -1.60 | [-2.7, -0.6] | 0.0020 |
| S1-SparseVLM@L50[latency_b1] | not reached | | |
| S1-S2@L75 | -0.03 | [-0.4, 0.3] | 0.8886 |
| S1-S3@L75 | -0.17 | [-0.7, 0.3] | 0.5257 |
| S1-grid@L75 | -0.08 | [-0.9, 0.8] | 0.8716 |
| S1-SparseVLM@L75[budget] | +0.43 | [-0.2, 1.2] | 0.2349 |
| S1-SparseVLM@L75[latency_b1] | not reached | | |
| A3:S1_dyn-fixed@f0.25 | +0.26 | [-0.1, 0.7] | 0.1509 |
| A3:S1_dyn-fixed@f0.5 | +0.30 | [-0.2, 0.8] | 0.2779 |
| A3:S1_dyn-fixed@f0.75 | -0.53 | [-1.1, 0.0] | 0.0740 |

Integrity: {"dropped_lines": 0, "blank_lines": 0, "duplicate_qids": 0}; files per row ≤ 2; SparseVLM counter records 8, outside tolerance / unmeasured 0.
