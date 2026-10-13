# R7-50 official-test evaluation, phase L (LLaVA-1.5-7B, D1-20 selectors)

Scored 2026-10-13T06:44:21Z; predictions manifest sha256 `44116bd7fb7c15c50a049f9336fd5be74d5d4019c2fe7d4cfdf74748d8e6926c` (committed before scoring); freeze {"protocol": "reports/R7/R7_50_PROTOCOL_FROZEN_20261012.yaml", "sha256": "3671da46d16e2c4aeb2aca6b65b2de26f638cd9fbb545bee4d5e7738745f01fd", "freeze_key": "r7_50_freeze_20261012"}.

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

## Sources

- gqa_testdev: official test (closed source, one read); 12578 items, 398 groups; sha256 14039069c0b3c797c7aa9bcd5f4c2aa4b5976e02c0b6773e1a584d942a03a318
  - overlap check: {"images": 398, "numeric_image_ids": 0, "n_prefixed_image_ids": 398, "other_image_ids": 0, "images_in_mix_gqa_paths": 0, "images_in_mix_vg_paths": 0, "images_in_mix_any": 0, "questions": 12578, "questions_seen_in_mix_gqa_turns": 0}
- chartqa_test: official test (closed source, one read); 2500 items, 1509 groups; sha256 84889bae30aaf6e24ff899e1604221606c0b65e658b6c70a7d02c79f50fc87df; splits {'human': 1250, 'augmented': 1250}
  - overlap check: {"mix_entries_with_chart_image_paths": 0, "note": "ChartQA is not part of LLaVA-1.5's published instruction mix"}

## Primary contrasts (points; B* = 4 latency points from the profiling workload; Holm per latency point)

| bench | contrast | Δ | 95 % CI | p | p_holm | seed Δ | basis |
|---|---|---|---|---|---|---|---|
| gqa_testdev | S1-S2@L25 | -3.24 | [-3.8, -2.7] | 0.0010 | 0.0060 | -3.5, -3.5, -2.7 | L25 = uniform-grid 144-token per-item latency at B = 4 on the profiling workload (64.38 ms) |
| gqa_testdev | S1-S3@L25 | -0.99 | [-1.5, -0.5] | 0.0010 | 0.0060 | +0.6, -2.8, -0.7 | L25 = uniform-grid 144-token per-item latency at B = 4 on the profiling workload (64.38 ms) |
| gqa_testdev | S1-SparseVLM@L25[budget] | -4.03 | [-4.7, -3.3] | 0.0010 | 0.0060 | -4.0, -5.6, -2.5 | equal-token budget 144 (SparseVLM row sparsevlm:144:adapted) |
| gqa_testdev | S1-S2@L50 | -1.46 | [-1.8, -1.1] | 0.0010 | 0.0060 | -1.4, -2.0, -1.0 | L50 = uniform-grid 288-token per-item latency at B = 4 on the profiling workload (71.01 ms) |
| gqa_testdev | S1-S3@L50 | -0.15 | [-0.5, 0.2] | 0.4568 | 0.4568 | +0.8, -1.1, -0.2 | L50 = uniform-grid 288-token per-item latency at B = 4 on the profiling workload (71.01 ms) |
| gqa_testdev | S1-SparseVLM@L50[budget] | -1.59 | [-2.0, -1.1] | 0.0010 | 0.0060 | -1.5, -2.3, -1.0 | equal-token budget 288 (SparseVLM row sparsevlm:288:adapted) |
| gqa_testdev | S1-S2@L75 | -0.26 | [-0.4, -0.1] | 0.0020 | 0.0120 | -0.3, -0.2, -0.3 | L75 = uniform-grid 432-token per-item latency at B = 4 on the profiling workload (77.99 ms) |
| gqa_testdev | S1-S3@L75 | -0.03 | [-0.2, 0.2] | 0.7436 | 1.0000 | +0.2, -0.1, -0.1 | L75 = uniform-grid 432-token per-item latency at B = 4 on the profiling workload (77.99 ms) |
| gqa_testdev | S1-SparseVLM@L75[budget] | -0.05 | [-0.4, 0.3] | 0.7256 | 1.0000 | -0.0, +0.0, -0.1 | equal-token budget 432 (SparseVLM row sparsevlm:432:adapted) |
| chartqa_test | S1-S2@L25 | -0.87 | [-1.4, -0.4] | 0.0020 | 0.0060 | -2.6, +0.6, -0.6 | L25 = uniform-grid 144-token per-item latency at B = 4 on the profiling workload (66.95 ms) |
| chartqa_test | S1-S3@L25 | +1.07 | [0.6, 1.5] | 0.0010 | 0.0060 | +2.1, +0.4, +0.7 | L25 = uniform-grid 144-token per-item latency at B = 4 on the profiling workload (66.95 ms) |
| chartqa_test | S1-SparseVLM@L25[budget] | -1.65 | [-2.5, -0.9] | 0.0010 | 0.0060 | -1.5, -2.3, -1.2 | equal-token budget 144 (SparseVLM row sparsevlm:144:adapted) |
| chartqa_test | S1-S2@L50 | -0.32 | [-0.7, 0.1] | 0.1269 | 0.2539 | -1.3, +0.3, +0.1 | L50 = uniform-grid 288-token per-item latency at B = 4 on the profiling workload (73.99 ms) |
| chartqa_test | S1-S3@L50 | +0.76 | [0.3, 1.2] | 0.0030 | 0.0120 | +1.7, -0.2, +0.8 | L50 = uniform-grid 288-token per-item latency at B = 4 on the profiling workload (73.99 ms) |
| chartqa_test | S1-SparseVLM@L50[budget] | -0.88 | [-1.6, -0.2] | 0.0130 | 0.0390 | -0.8, -1.7, -0.1 | equal-token budget 288 (SparseVLM row sparsevlm:288:adapted) |
| chartqa_test | S1-S2@L75 | +0.12 | [-0.1, 0.3] | 0.2999 | 1.0000 | -0.1, +0.1, +0.3 | L75 = uniform-grid 432-token per-item latency at B = 4 on the profiling workload (80.78 ms) |
| chartqa_test | S1-S3@L75 | +0.25 | [-0.0, 0.5] | 0.0540 | 0.2699 | +0.4, +0.1, +0.2 | L75 = uniform-grid 432-token per-item latency at B = 4 on the profiling workload (80.78 ms) |
| chartqa_test | S1-SparseVLM@L75[budget] | +0.04 | [-0.4, 0.5] | 0.9035 | 1.0000 | -0.0, +0.1, +0.0 | equal-token budget 432 (SparseVLM row sparsevlm:432:adapted) |

## gqa_testdev: fixed rows (n = 12578, 398 groups)

| row | visual tokens | quality [95 % CI] | retention (rel.) | latency B=4 / B=1, ms (profiling workload) |
|---|---|---|---|---|
| full | 576.0 | 61.95 [60.5, 63.4] | 1.000 | 87.7 / 173.4 |
| static:grid:72 | 72.0 | 56.26 [55.0, 57.7] | 0.908 | 61.7 / 154.7 |
| static:grid:144 | 144.0 | 58.70 [57.4, 60.0] | 0.948 | 64.4 / 154.8 |
| static:grid:288 | 288.0 | 60.29 [58.9, 61.7] | 0.973 | 71.0 / 159.0 |
| static:grid:432 | 432.0 | 60.92 [59.6, 62.4] | 0.983 | 78.0 / 165.4 |
| sel:vcs_lr0.0003_s0:72 | 72.0 | 49.81 [48.6, 51.1] | 0.804 | 62.5 / 155.5 |
| sel:vcs_lr0.0003_s0:144 | 144.0 | 54.81 [53.5, 56.1] | 0.885 | 64.9 / 156.0 |
| sel:vcs_lr0.0003_s0:288 | 288.0 | 59.21 [57.8, 60.6] | 0.956 | 72.0 / 160.3 |
| sel:vcs_lr0.0003_s0:432 | 432.0 | 61.49 [60.1, 62.9] | 0.993 | 79.1 / 166.9 |
| sel:vcs_lr0.0003_s1:72 | 72.0 | 49.12 [47.9, 50.4] | 0.793 | 62.5 / 155.5 |
| sel:vcs_lr0.0003_s1:144 | 144.0 | 53.22 [51.9, 54.6] | 0.859 | 64.9 / 156.0 |
| sel:vcs_lr0.0003_s1:288 | 288.0 | 58.39 [57.0, 59.9] | 0.943 | 72.0 / 160.3 |
| sel:vcs_lr0.0003_s1:432 | 432.0 | 61.55 [60.1, 63.0] | 0.994 | 79.1 / 166.9 |
| sel:vcs_lr0.0003_s2:72 | 72.0 | 52.63 [51.4, 53.8] | 0.850 | 62.5 / 155.5 |
| sel:vcs_lr0.0003_s2:144 | 144.0 | 56.25 [55.0, 57.6] | 0.908 | 64.9 / 156.0 |
| sel:vcs_lr0.0003_s2:288 | 288.0 | 59.75 [58.4, 61.1] | 0.964 | 72.0 / 160.3 |
| sel:vcs_lr0.0003_s2:432 | 432.0 | 61.39 [60.0, 62.9] | 0.991 | 79.1 / 166.9 |
| sel:logistic_lr0.0001_s0:72 | 72.0 | 54.21 [52.9, 55.5] | 0.875 | 62.5 / 155.5 |
| sel:logistic_lr0.0001_s0:144 | 144.0 | 58.01 [56.6, 59.4] | 0.936 | 64.9 / 156.0 |
| sel:logistic_lr0.0001_s0:288 | 288.0 | 60.30 [58.9, 61.7] | 0.973 | 72.0 / 160.3 |
| sel:logistic_lr0.0001_s0:432 | 432.0 | 61.62 [60.2, 63.1] | 0.995 | 79.1 / 166.9 |
| sel:logistic_lr0.0001_s1:72 | 72.0 | 53.77 [52.4, 55.1] | 0.868 | 62.5 / 155.5 |
| sel:logistic_lr0.0001_s1:144 | 144.0 | 56.39 [55.1, 57.7] | 0.910 | 64.9 / 156.0 |
| sel:logistic_lr0.0001_s1:288 | 288.0 | 60.18 [58.8, 61.6] | 0.972 | 72.0 / 160.3 |
| sel:logistic_lr0.0001_s1:432 | 432.0 | 61.46 [60.1, 62.9] | 0.992 | 79.1 / 166.9 |
| sel:logistic_lr0.0001_s2:72 | 72.0 | 55.87 [54.7, 57.2] | 0.902 | 62.5 / 155.5 |
| sel:logistic_lr0.0001_s2:144 | 144.0 | 58.82 [57.5, 60.2] | 0.949 | 64.9 / 156.0 |
| sel:logistic_lr0.0001_s2:288 | 288.0 | 60.50 [59.1, 61.9] | 0.977 | 72.0 / 160.3 |
| sel:logistic_lr0.0001_s2:432 | 432.0 | 61.59 [60.2, 63.0] | 0.994 | 79.1 / 166.9 |
| sel:direct_lr0.0001_s0:72 | 72.0 | 50.72 [49.4, 52.1] | 0.819 | 62.5 / 155.5 |
| sel:direct_lr0.0001_s0:144 | 144.0 | 53.77 [52.3, 55.2] | 0.868 | 64.9 / 156.0 |
| sel:direct_lr0.0001_s0:288 | 288.0 | 58.44 [57.0, 59.9] | 0.943 | 72.0 / 160.3 |
| sel:direct_lr0.0001_s0:432 | 432.0 | 61.43 [60.0, 62.9] | 0.992 | 79.1 / 166.9 |
| sel:direct_lr0.0001_s1:72 | 72.0 | 52.62 [51.4, 53.9] | 0.849 | 62.5 / 155.5 |
| sel:direct_lr0.0001_s1:144 | 144.0 | 55.84 [54.5, 57.2] | 0.901 | 64.9 / 156.0 |
| sel:direct_lr0.0001_s1:288 | 288.0 | 59.22 [57.9, 60.6] | 0.956 | 72.0 / 160.3 |
| sel:direct_lr0.0001_s1:432 | 432.0 | 61.55 [60.1, 63.0] | 0.994 | 79.1 / 166.9 |
| sel:direct_lr0.0001_s2:72 | 72.0 | 54.01 [52.7, 55.2] | 0.872 | 62.5 / 155.5 |
| sel:direct_lr0.0001_s2:144 | 144.0 | 56.79 [55.5, 58.1] | 0.917 | 64.9 / 156.0 |
| sel:direct_lr0.0001_s2:288 | 288.0 | 59.85 [58.4, 61.3] | 0.966 | 72.0 / 160.3 |
| sel:direct_lr0.0001_s2:432 | 432.0 | 61.53 [60.1, 63.0] | 0.993 | 79.1 / 166.9 |
| dyn:vcs_lr0.0003_s0:0.25 | 158.1 | 52.93 [51.6, 54.2] | 0.854 | 68.5 / 162.9 |
| dyn:vcs_lr0.0003_s0:0.5 | 333.4 | 57.98 [56.6, 59.3] | 0.936 | 75.8 / 168.1 |
| dyn:vcs_lr0.0003_s0:0.75 | 489.1 | 60.81 [59.4, 62.2] | 0.982 | 82.8 / 175.0 |
| dyn:vcs_lr0.0003_s1:0.25 | 153.6 | 52.46 [51.1, 53.9] | 0.847 | 68.5 / 162.9 |
| dyn:vcs_lr0.0003_s1:0.5 | 330.7 | 57.66 [56.3, 59.1] | 0.931 | 76.1 / 169.7 |
| dyn:vcs_lr0.0003_s1:0.75 | 480.1 | 60.82 [59.4, 62.3] | 0.982 | 83.2 / 179.1 |
| dyn:vcs_lr0.0003_s2:0.25 | 162.4 | 55.38 [54.1, 56.7] | 0.894 | 67.9 / 173.3 |
| dyn:vcs_lr0.0003_s2:0.5 | 334.9 | 58.90 [57.6, 60.2] | 0.951 | 75.6 / 178.0 |
| dyn:vcs_lr0.0003_s2:0.75 | 474.0 | 60.94 [59.5, 62.4] | 0.984 | 83.3 / 181.3 |
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
| 1 | S1 | not reached | 57.75 | 60.92 |
| 1 | S2 | not reached | 59.52 | 61.27 |
| 1 | S3 | not reached | 58.01 | 60.96 |
| 1 | S1_dyn | not reached | not reached | not reached |
| 1 | grid | 58.70 | 60.29 | 60.92 |
| 1 | SparseVLM | not reached | not reached | not reached |
| 4 | S1 | 53.78 | 58.50 | 61.10 |
| 4 | S2 | 57.02 | 59.96 | 61.36 |
| 4 | S3 | 54.77 | 58.65 | 61.13 |
| 4 | S1_dyn | not reached | 55.20 | 58.96 |
| 4 | grid | 58.70 | 60.29 | 60.92 |
| 16 | S1 | 54.77 | 58.77 | 61.35 |
| 16 | S2 | 57.75 | 60.12 | 61.49 |
| 16 | S3 | 55.47 | 58.88 | 61.38 |
| 16 | S1_dyn | not reached | 56.43 | 60.19 |
| 16 | grid | 58.70 | 60.29 | 60.92 |
| 32 | S1 | 53.88 | 59.13 | 61.17 |
| 32 | S2 | 57.09 | 60.34 | 61.40 |
| 32 | S3 | 54.84 | 59.19 | 61.20 |
| 32 | S1_dyn | not reached | 57.12 | 59.90 |
| 32 | grid | 58.70 | 60.29 | 60.92 |

S1 dynamic rows: kept tokens on test vs the profiling workload (B = 4):

| row | mean kept (test) | share 72/144/288/432/576 (test) | mean kept (profiling) |
|---|---|---|---|
| dyn:vcs_lr0.0003_s0:0.25 | 158.1 | 0.64 / 0.11 / 0.12 / 0.11 / 0.03 | 145.7 |
| dyn:vcs_lr0.0003_s0:0.5 | 333.4 | 0.25 / 0.11 / 0.14 / 0.18 / 0.31 | 288.8 |
| dyn:vcs_lr0.0003_s0:0.75 | 489.1 | 0.06 / 0.05 / 0.07 / 0.11 / 0.71 | 425.4 |
| dyn:vcs_lr0.0003_s1:0.25 | 153.6 | 0.63 / 0.14 / 0.12 / 0.09 / 0.03 | 145.4 |
| dyn:vcs_lr0.0003_s1:0.5 | 330.7 | 0.22 / 0.14 / 0.17 / 0.19 / 0.29 | 293.2 |
| dyn:vcs_lr0.0003_s1:0.75 | 480.1 | 0.06 / 0.06 / 0.09 / 0.13 / 0.67 | 426.4 |
| dyn:vcs_lr0.0003_s2:0.25 | 162.4 | 0.58 / 0.15 / 0.14 / 0.10 / 0.03 | 134.6 |
| dyn:vcs_lr0.0003_s2:0.5 | 334.9 | 0.22 / 0.12 / 0.17 / 0.20 / 0.29 | 286.7 |
| dyn:vcs_lr0.0003_s2:0.75 | 474.0 | 0.06 / 0.06 / 0.09 / 0.14 / 0.65 | 430.3 |

Secondary contrasts (unadjusted):

| contrast | Δ | 95 % CI | p |
|---|---|---|---|
| S1-grid@L25 | -4.91 | [-5.6, -4.3] | 0.0010 |
| S1-SparseVLM@L25[latency_b1] | not reached | | |
| S1-grid@L50 | -1.79 | [-2.3, -1.3] | 0.0010 |
| S1-SparseVLM@L50[latency_b1] | not reached | | |
| S1-grid@L75 | +0.19 | [-0.1, 0.5] | 0.2759 |
| S1-SparseVLM@L75[latency_b1] | not reached | | |
| A3:S1_dyn-fixed@f0.25 | -1.57 | [-1.9, -1.2] | 0.0010 |
| A3:S1_dyn-fixed@f0.5 | -1.67 | [-2.0, -1.3] | 0.0010 |
| A3:S1_dyn-fixed@f0.75 | -0.78 | [-1.0, -0.5] | 0.0010 |

Integrity: {"dropped_lines": 0, "blank_lines": 0, "duplicate_qids": 0}; files per row ≤ 6; SparseVLM counter records 24, outside tolerance / unmeasured 0.

## chartqa_test: fixed rows (n = 2500, 1509 groups)

| row | visual tokens | quality [95 % CI] | human | augmented | retention (rel.) | latency B=4 / B=1, ms (profiling workload) |
|---|---|---|---|---|---|---|
| full | 576.0 | 18.16 [16.7, 19.7] | 22.08 | 14.24 | 1.000 | 90.9 / 177.3 |
| static:grid:72 | 72.0 | 13.52 [12.2, 15.0] | 15.76 | 11.28 | 0.744 | 64.6 / 157.4 |
| static:grid:144 | 144.0 | 14.80 [13.3, 16.4] | 17.76 | 11.84 | 0.815 | 67.0 / 157.1 |
| static:grid:288 | 288.0 | 16.92 [15.5, 18.5] | 20.48 | 13.36 | 0.932 | 74.0 / 162.0 |
| static:grid:432 | 432.0 | 17.44 [15.9, 19.0] | 21.12 | 13.76 | 0.960 | 80.8 / 168.2 |
| sel:vcs_lr0.0003_s0:72 | 72.0 | 13.84 [12.5, 15.3] | 15.68 | 12.00 | 0.762 | 65.6 / 157.8 |
| sel:vcs_lr0.0003_s0:144 | 144.0 | 15.08 [13.7, 16.6] | 17.36 | 12.80 | 0.830 | 68.2 / 158.7 |
| sel:vcs_lr0.0003_s0:288 | 288.0 | 16.60 [15.1, 18.1] | 20.24 | 12.96 | 0.914 | 75.4 / 163.2 |
| sel:vcs_lr0.0003_s0:432 | 432.0 | 17.72 [16.2, 19.3] | 21.60 | 13.84 | 0.976 | 82.5 / 168.4 |
| sel:vcs_lr0.0003_s1:72 | 72.0 | 13.68 [12.3, 15.1] | 15.60 | 11.76 | 0.753 | 65.6 / 157.8 |
| sel:vcs_lr0.0003_s1:144 | 144.0 | 14.28 [12.9, 15.7] | 16.88 | 11.68 | 0.786 | 68.2 / 158.7 |
| sel:vcs_lr0.0003_s1:288 | 288.0 | 15.76 [14.3, 17.3] | 18.56 | 12.96 | 0.868 | 75.4 / 163.2 |
| sel:vcs_lr0.0003_s1:432 | 432.0 | 17.88 [16.4, 19.5] | 21.68 | 14.08 | 0.985 | 82.5 / 168.4 |
| sel:vcs_lr0.0003_s2:72 | 72.0 | 14.00 [12.6, 15.5] | 16.24 | 11.76 | 0.771 | 65.6 / 157.8 |
| sel:vcs_lr0.0003_s2:144 | 144.0 | 15.36 [14.0, 16.9] | 18.48 | 12.24 | 0.846 | 68.2 / 158.7 |
| sel:vcs_lr0.0003_s2:288 | 288.0 | 17.32 [15.9, 18.9] | 21.28 | 13.36 | 0.954 | 75.4 / 163.2 |
| sel:vcs_lr0.0003_s2:432 | 432.0 | 17.80 [16.3, 19.4] | 21.52 | 14.08 | 0.980 | 82.5 / 168.4 |
| sel:logistic_lr0.0001_s0:72 | 72.0 | 17.00 [15.5, 18.6] | 19.68 | 14.32 | 0.936 | 65.6 / 157.8 |
| sel:logistic_lr0.0001_s0:144 | 144.0 | 17.20 [15.7, 18.8] | 20.56 | 13.84 | 0.947 | 68.2 / 158.7 |
| sel:logistic_lr0.0001_s0:288 | 288.0 | 17.72 [16.2, 19.3] | 21.44 | 14.00 | 0.976 | 75.4 / 163.2 |
| sel:logistic_lr0.0001_s0:432 | 432.0 | 17.48 [16.0, 19.1] | 21.12 | 13.84 | 0.963 | 82.5 / 168.4 |
| sel:logistic_lr0.0001_s1:72 | 72.0 | 13.20 [11.9, 14.6] | 15.44 | 10.96 | 0.727 | 65.6 / 157.8 |
| sel:logistic_lr0.0001_s1:144 | 144.0 | 13.56 [12.2, 15.0] | 16.32 | 10.80 | 0.747 | 68.2 / 158.7 |
| sel:logistic_lr0.0001_s1:288 | 288.0 | 15.60 [14.1, 17.2] | 19.12 | 12.08 | 0.859 | 75.4 / 163.2 |
| sel:logistic_lr0.0001_s1:432 | 432.0 | 17.76 [16.2, 19.3] | 21.68 | 13.84 | 0.978 | 82.5 / 168.4 |
| sel:logistic_lr0.0001_s2:72 | 72.0 | 14.64 [13.2, 16.0] | 17.52 | 11.76 | 0.806 | 65.6 / 157.8 |
| sel:logistic_lr0.0001_s2:144 | 144.0 | 15.96 [14.4, 17.5] | 18.88 | 13.04 | 0.879 | 68.2 / 158.7 |
| sel:logistic_lr0.0001_s2:288 | 288.0 | 17.08 [15.6, 18.6] | 20.80 | 13.36 | 0.941 | 75.4 / 163.2 |
| sel:logistic_lr0.0001_s2:432 | 432.0 | 17.48 [15.9, 19.1] | 21.12 | 13.84 | 0.963 | 82.5 / 168.4 |
| sel:direct_lr0.0001_s0:72 | 72.0 | 12.36 [11.1, 13.8] | 14.24 | 10.48 | 0.681 | 65.6 / 157.8 |
| sel:direct_lr0.0001_s0:144 | 144.0 | 12.40 [11.2, 13.8] | 14.56 | 10.24 | 0.683 | 68.2 / 158.7 |
| sel:direct_lr0.0001_s0:288 | 288.0 | 15.16 [13.8, 16.7] | 18.16 | 12.16 | 0.835 | 75.4 / 163.2 |
| sel:direct_lr0.0001_s0:432 | 432.0 | 17.60 [16.1, 19.2] | 20.96 | 14.24 | 0.969 | 82.5 / 168.4 |
| sel:direct_lr0.0001_s1:72 | 72.0 | 12.76 [11.4, 14.2] | 14.80 | 10.72 | 0.703 | 65.6 / 157.8 |
| sel:direct_lr0.0001_s1:144 | 144.0 | 14.36 [13.0, 15.8] | 16.40 | 12.32 | 0.791 | 68.2 / 158.7 |
| sel:direct_lr0.0001_s1:288 | 288.0 | 15.96 [14.5, 17.4] | 18.96 | 12.96 | 0.879 | 75.4 / 163.2 |
| sel:direct_lr0.0001_s1:432 | 432.0 | 17.68 [16.2, 19.3] | 21.60 | 13.76 | 0.974 | 82.5 / 168.4 |
| sel:direct_lr0.0001_s2:72 | 72.0 | 13.40 [12.1, 14.9] | 15.60 | 11.20 | 0.738 | 65.6 / 157.8 |
| sel:direct_lr0.0001_s2:144 | 144.0 | 14.56 [13.1, 16.1] | 16.88 | 12.24 | 0.802 | 68.2 / 158.7 |
| sel:direct_lr0.0001_s2:288 | 288.0 | 16.56 [15.1, 18.1] | 19.92 | 13.20 | 0.912 | 75.4 / 163.2 |
| sel:direct_lr0.0001_s2:432 | 432.0 | 17.76 [16.2, 19.3] | 21.60 | 13.92 | 0.978 | 82.5 / 168.4 |
| dyn:vcs_lr0.0003_s0:0.25 | 138.9 | 14.72 [13.3, 16.2] | 16.64 | 12.80 | 0.811 | 71.3 / 166.7 |
| dyn:vcs_lr0.0003_s0:0.5 | 301.6 | 16.52 [15.0, 18.1] | 19.92 | 13.12 | 0.910 | 79.4 / 171.0 |
| dyn:vcs_lr0.0003_s0:0.75 | 479.2 | 17.80 [16.3, 19.4] | 21.12 | 14.48 | 0.980 | 89.7 / 181.0 |
| dyn:vcs_lr0.0003_s1:0.25 | 148.7 | 14.08 [12.7, 15.6] | 16.16 | 12.00 | 0.775 | 71.5 / 166.0 |
| dyn:vcs_lr0.0003_s1:0.5 | 333.4 | 16.00 [14.5, 17.6] | 19.60 | 12.40 | 0.881 | 80.7 / 172.2 |
| dyn:vcs_lr0.0003_s1:0.75 | 483.1 | 17.08 [15.5, 18.6] | 21.20 | 12.96 | 0.941 | 89.2 / 179.8 |
| dyn:vcs_lr0.0003_s2:0.25 | 143.5 | 14.64 [13.2, 16.2] | 17.20 | 12.08 | 0.806 | 71.7 / 166.2 |
| dyn:vcs_lr0.0003_s2:0.5 | 296.0 | 16.32 [14.8, 18.0] | 19.60 | 13.04 | 0.899 | 78.7 / 173.1 |
| dyn:vcs_lr0.0003_s2:0.75 | 449.7 | 17.28 [15.8, 18.9] | 20.88 | 13.68 | 0.952 | 86.9 / 180.0 |
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
| 1 | S1 | not reached | 16.10 | 17.76 |
| 1 | S2 | not reached | 16.46 | 17.55 |
| 1 | S3 | not reached | 15.30 | 17.62 |
| 1 | S1_dyn | not reached | not reached | 15.08 |
| 1 | grid | 14.80 | 16.92 | 17.44 |
| 1 | SparseVLM | not reached | not reached | not reached |
| 4 | S1 | 14.41 | 16.24 | 17.51 |
| 4 | S2 | 15.28 | 16.57 | 17.39 |
| 4 | S3 | 13.34 | 15.49 | 17.26 |
| 4 | S1_dyn | not reached | 15.04 | 16.42 |
| 4 | grid | 14.80 | 16.92 | 17.44 |
| 16 | S1 | 14.60 | 16.48 | 17.68 |
| 16 | S2 | 15.40 | 16.74 | 17.50 |
| 16 | S3 | 13.51 | 15.79 | 17.51 |
| 16 | S1_dyn | not reached | 15.36 | 16.76 |
| 16 | grid | 14.80 | 16.92 | 17.44 |
| 32 | S1 | 14.79 | 16.47 | 17.74 |
| 32 | S2 | 15.50 | 16.74 | 17.53 |
| 32 | S3 | 13.67 | 15.78 | 17.59 |
| 32 | S1_dyn | not reached | 15.43 | 16.78 |
| 32 | grid | 14.80 | 16.92 | 17.44 |

S1 dynamic rows: kept tokens on test vs the profiling workload (B = 4):

| row | mean kept (test) | share 72/144/288/432/576 (test) | mean kept (profiling) |
|---|---|---|---|
| dyn:vcs_lr0.0003_s0:0.25 | 138.9 | 0.73 / 0.07 / 0.08 / 0.11 / 0.01 | 136.4 |
| dyn:vcs_lr0.0003_s0:0.5 | 301.6 | 0.30 / 0.12 / 0.13 / 0.22 / 0.22 | 309.1 |
| dyn:vcs_lr0.0003_s0:0.75 | 479.2 | 0.05 / 0.05 / 0.09 / 0.17 / 0.64 | 489.2 |
| dyn:vcs_lr0.0003_s1:0.25 | 148.7 | 0.62 / 0.14 / 0.14 / 0.09 / 0.01 | 150.0 |
| dyn:vcs_lr0.0003_s1:0.5 | 333.4 | 0.19 / 0.12 / 0.19 / 0.28 / 0.22 | 331.6 |
| dyn:vcs_lr0.0003_s1:0.75 | 483.1 | 0.04 / 0.06 / 0.07 / 0.18 / 0.65 | 481.6 |
| dyn:vcs_lr0.0003_s2:0.25 | 143.5 | 0.62 / 0.16 / 0.14 / 0.07 / 0.01 | 145.0 |
| dyn:vcs_lr0.0003_s2:0.5 | 296.0 | 0.23 / 0.16 / 0.22 / 0.22 / 0.17 | 294.8 |
| dyn:vcs_lr0.0003_s2:0.75 | 449.7 | 0.06 / 0.06 / 0.14 / 0.22 / 0.52 | 443.7 |

Secondary contrasts (unadjusted):

| contrast | Δ | 95 % CI | p |
|---|---|---|---|
| S1-grid@L25 | -0.39 | [-1.1, 0.3] | 0.2979 |
| S1-SparseVLM@L25[latency_b1] | not reached | | |
| S1-grid@L50 | -0.68 | [-1.4, 0.0] | 0.0680 |
| S1-SparseVLM@L50[latency_b1] | not reached | | |
| S1-grid@L75 | +0.07 | [-0.6, 0.7] | 0.8586 |
| S1-SparseVLM@L75[latency_b1] | not reached | | |
| A3:S1_dyn-fixed@f0.25 | -0.41 | [-0.9, 0.1] | 0.0780 |
| A3:S1_dyn-fixed@f0.5 | -0.55 | [-1.0, -0.1] | 0.0090 |
| A3:S1_dyn-fixed@f0.75 | -0.51 | [-0.9, -0.1] | 0.0150 |

ChartQA human (n = 1250; secondary):

| contrast | Δ | 95 % CI | p |
|---|---|---|---|
| S1-S2@L25 | -1.34 | [-2.0, -0.6] | 0.0010 |
| S1-S3@L25 | +1.31 | [0.6, 2.0] | 0.0010 |
| S1-grid@L25 | -1.00 | [-2.1, 0.1] | 0.0760 |
| S1-SparseVLM@L25[budget] | -1.87 | [-3.0, -0.7] | 0.0010 |
| S1-SparseVLM@L25[latency_b1] | not reached | | |
| S1-S2@L50 | -0.54 | [-1.2, 0.1] | 0.1069 |
| S1-S3@L50 | +1.13 | [0.4, 1.8] | 0.0030 |
| S1-grid@L50 | -0.92 | [-2.0, 0.2] | 0.0910 |
| S1-SparseVLM@L50[budget] | -0.69 | [-1.7, 0.3] | 0.1639 |
| S1-SparseVLM@L50[latency_b1] | not reached | | |
| S1-S2@L75 | +0.12 | [-0.2, 0.4] | 0.4128 |
| S1-S3@L75 | +0.40 | [0.0, 0.8] | 0.0310 |
| S1-grid@L75 | +0.11 | [-0.8, 1.0] | 0.7986 |
| S1-SparseVLM@L75[budget] | -0.32 | [-1.0, 0.3] | 0.3308 |
| S1-SparseVLM@L75[latency_b1] | not reached | | |
| A3:S1_dyn-fixed@f0.25 | -0.81 | [-1.4, -0.1] | 0.0170 |
| A3:S1_dyn-fixed@f0.5 | -0.70 | [-1.4, -0.0] | 0.0420 |
| A3:S1_dyn-fixed@f0.75 | -0.65 | [-1.3, 0.0] | 0.0570 |

ChartQA augmented (n = 1250; secondary):

| contrast | Δ | 95 % CI | p |
|---|---|---|---|
| S1-S2@L25 | -0.41 | [-1.0, 0.2] | 0.1889 |
| S1-S3@L25 | +0.83 | [0.1, 1.5] | 0.0200 |
| S1-grid@L25 | +0.21 | [-0.8, 1.2] | 0.6437 |
| S1-SparseVLM@L25[budget] | -1.44 | [-2.6, -0.3] | 0.0100 |
| S1-SparseVLM@L25[latency_b1] | not reached | | |
| S1-S2@L50 | -0.10 | [-0.7, 0.5] | 0.7046 |
| S1-S3@L50 | +0.38 | [-0.2, 1.0] | 0.2579 |
| S1-grid@L50 | -0.43 | [-1.4, 0.5] | 0.3938 |
| S1-SparseVLM@L50[budget] | -1.07 | [-2.0, -0.2] | 0.0190 |
| S1-SparseVLM@L50[latency_b1] | not reached | | |
| S1-S2@L75 | +0.11 | [-0.2, 0.4] | 0.4878 |
| S1-S3@L75 | +0.10 | [-0.3, 0.5] | 0.5887 |
| S1-grid@L75 | +0.03 | [-0.9, 0.9] | 0.9335 |
| S1-SparseVLM@L75[budget] | +0.40 | [-0.3, 1.2] | 0.2589 |
| S1-SparseVLM@L75[latency_b1] | not reached | | |
| A3:S1_dyn-fixed@f0.25 | +0.04 | [-0.6, 0.7] | 0.9125 |
| A3:S1_dyn-fixed@f0.5 | -0.40 | [-1.0, 0.2] | 0.1639 |
| A3:S1_dyn-fixed@f0.75 | -0.36 | [-0.9, 0.1] | 0.1539 |

Integrity: {"dropped_lines": 0, "blank_lines": 0, "duplicate_qids": 0}; files per row ≤ 2; SparseVLM counter records 8, outside tolerance / unmeasured 0.
