# R7-30 D1-30 smoke (R7-20 fixtures only)

Job output: /home/infres/yinwang/CS_QMI/outputs/R7_D1/d1_30_smoke.  GPU NVIDIA H100 NVL; pipeline checks on fixture subsets with the D1-20 *smoke* selectors (seeds 1-2 = seed-0 weights relabelled; dynamic tau calibrated on the fixture) — numbers are not results.

## Checks

- user_turn_equal: "200/200"
- selector_inputs: {"visual_max_abs_diff": 0.25, "visual_frac_exact": 0.4273015964798063, "question_ids_equal": "68/68"}
- repro_gqa_fixture: "200/200 identical to R7-20 answers"
- repro_textvqa_fixture: "200/200 identical to R7-20 answers"
- SparseVLM reproduction sparsevlm192_gqa_fixture: 198/200 identical to R7-20
- SparseVLM reproduction sparsevlm192_textvqa_fixture_ocr: 198/200 identical to R7-20
- SparseVLM reproduction sparsevlm64_gqa_fixture: 199/200 identical to R7-20
- SparseVLM reproduction sparsevlm64_textvqa_fixture_ocr: 200/200 identical to R7-20
- SparseVLM official counter == predicted for 34/34 rows

## SparseVLM rows (official 'Equal Tokens' counter)

| bench | row | layer list | predicted | official |
|---|---|---|---|---|
| chartqa_proxy | sparsevlm:192:official | [300, 200, 110] | 192.031 | 192.03125 |
| chartqa_proxy | sparsevlm:128:official | [303, 110, 36] | 128.625 | 128.625 |
| chartqa_proxy | sparsevlm:96:official | [238, 48, 26] | 96.938 | 96.9375 |
| chartqa_proxy | sparsevlm:64:official | [66, 30, 17] | 64.812 | 64.8125 |
| chartqa_proxy | sparsevlm:72:adapted | [106, 33, 19] | 71.875 | 71.875 |
| chartqa_proxy | sparsevlm:144:adapted | [302, 132, 54] | 143.969 | 143.96875 |
| chartqa_proxy | sparsevlm:288:adapted | [368, 294, 226] | 288.062 | 288.0625 |
| chartqa_proxy | sparsevlm:432:adapted | [472, 434, 401] | 432.219 | 432.21875 |
| gqa_fixture | sparsevlm:192:official | [300, 200, 110] | 192.031 | 192.03125 |
| gqa_fixture | sparsevlm:128:official | [303, 110, 36] | 128.625 | 128.625 |
| gqa_fixture | sparsevlm:96:official | [238, 48, 26] | 96.938 | 96.9375 |
| gqa_fixture | sparsevlm:64:official | [66, 30, 17] | 64.812 | 64.8125 |
| gqa_fixture | sparsevlm:72:adapted | [106, 33, 19] | 71.875 | 71.875 |
| gqa_fixture | sparsevlm:144:adapted | [302, 132, 54] | 143.969 | 143.96875 |
| gqa_fixture | sparsevlm:288:adapted | [368, 294, 226] | 288.062 | 288.0625 |
| gqa_fixture | sparsevlm:432:adapted | [472, 434, 401] | 432.219 | 432.21875 |
| textvqa_fixture | sparsevlm:192:official | [300, 200, 110] | 192.031 | 192.03125 |
| textvqa_fixture | sparsevlm:128:official | [303, 110, 36] | 128.625 | 128.625 |
| textvqa_fixture | sparsevlm:96:official | [238, 48, 26] | 96.938 | 96.9375 |
| textvqa_fixture | sparsevlm:64:official | [66, 30, 17] | 64.812 | 64.8125 |
| textvqa_fixture | sparsevlm:72:adapted | [106, 33, 19] | 71.875 | 71.875 |
| textvqa_fixture | sparsevlm:144:adapted | [302, 132, 54] | 143.969 | 143.96875 |
| textvqa_fixture | sparsevlm:288:adapted | [368, 294, 226] | 288.062 | 288.0625 |
| textvqa_fixture | sparsevlm:432:adapted | [472, 434, 401] | 432.219 | 432.21875 |
| textvqa_fixture | sparsevlm:192:official:ocr | [300, 200, 110] | 192.031 | 192.03125 |
| textvqa_fixture | sparsevlm:128:official:ocr | [303, 110, 36] | 128.625 | 128.625 |
| textvqa_fixture | sparsevlm:96:official:ocr | [238, 48, 26] | 96.938 | 96.9375 |
| textvqa_fixture | sparsevlm:64:official:ocr | [66, 30, 17] | 64.812 | 64.8125 |

## Scorer (smoke subsets)

- gqa_fixture: n = 64, full = 75.0; L25/L50/L75 at B = 4 (ms): {'L25': 65.5667424434796, 'L50': 72.50494707841426, 'L75': 79.31665854994208}
  - S1-S2@L25: Δ -1.3 [-6.5, +3.9], p_holm 1.0
  - S1-S3@L25: Δ +20.0 [+4.7, +33.2], p_holm 0.08955223880597014
  - S1-SparseVLM@L25[budget]: Δ -1.6 [-10.8, +4.9], p_holm 1.0
  - S1-S2@L50: Δ +1.0 [-0.8, +4.8], p_holm 1.0
  - S1-S3@L50: Δ +10.9 [+0.6, +20.8], p_holm 0.24875621890547264
  - S1-SparseVLM@L50[budget]: Δ +7.8 [+1.5, +15.9], p_holm 0.23880597014925373
  - S1-S2@L75: Δ +1.6 [+0.0, +4.5], p_holm 1.0
  - S1-S3@L75: Δ +3.8 [-1.1, +8.7], p_holm 1.0
  - S1-SparseVLM@L75[budget]: Δ +0.0 [-4.8, +4.7], p_holm 1.0
- textvqa_fixture: n = 64, full = 46.6; L25/L50/L75 at B = 4 (ms): {'L25': 75.39239735342562, 'L50': 82.68362923990935, 'L75': 89.20946205034852}
  - S1-S2@L25: Δ +5.8 [-0.8, +12.8], p_holm 0.7960199004975124
  - S1-S3@L25: Δ +12.9 [+4.9, +21.2], p_holm 0.08955223880597014
  - S1-SparseVLM@L25[budget]: Δ -22.3 [-31.2, -13.8], p_holm 0.08955223880597014
  - S1-S2@L50: Δ +2.2 [-5.3, +11.5], p_holm 1.0
  - S1-S3@L50: Δ +8.7 [+0.8, +18.4], p_holm 0.23880597014925373
  - S1-SparseVLM@L50[budget]: Δ -10.9 [-18.0, -4.5], p_holm 0.08955223880597014
  - S1-S2@L75: Δ +2.3 [-1.9, +7.4], p_holm 1.0
  - S1-S3@L75: Δ +6.9 [+0.9, +13.2], p_holm 0.1791044776119403
  - S1-SparseVLM@L75[budget]: Δ +0.0 [-4.5, +4.8], p_holm 1.0
- chartqa_proxy: n = 64, full = 75.0; L25/L50/L75 at B = 4 (ms): {'L25': 65.78714470379055, 'L50': 73.1690723914653, 'L75': 79.34047712478787}
  - S1-S2@L25: Δ +2.0 [-4.6, +8.5], p_holm 1.0
  - S1-S3@L25: Δ +20.4 [+4.4, +33.8], p_holm 0.11940298507462686
  - S1-SparseVLM@L25[budget]: Δ +1.6 [-6.4, +9.2], p_holm 1.0
  - S1-S2@L50: Δ +1.5 [-0.3, +5.7], p_holm 1.0
  - S1-S3@L50: Δ +10.2 [+0.8, +19.7], p_holm 0.24875621890547264
  - S1-SparseVLM@L50[budget]: Δ +7.8 [+1.5, +15.9], p_holm 0.23880597014925373
  - S1-S2@L75: Δ +1.6 [+0.0, +4.4], p_holm 1.0
  - S1-S3@L75: Δ +4.0 [-0.9, +8.4], p_holm 1.0
  - S1-SparseVLM@L75[budget]: Δ -1.6 [-6.5, +3.2], p_holm 1.0

## DEV-risk code path (probe records, fixture binary questions)

- vcs_lr0.0003_s0: {"n_pairs": 816, "spearman_r_h": 0.07815683241405236, "slope_h_on_r": -14.155352671144316}
- logistic_lr0.0003_s0: {"n_pairs": 816, "spearman_r_h": 0.1125089502351466, "slope_h_on_r": -4651.522737672033}
- direct_lr0.0003_s0: {"n_pairs": 816, "spearman_r_h": 0.7401502800247538, "slope_h_on_r": 0.9910251298223275}
- vcs_lr0.0003_s0_noq: {"n_pairs": 816, "spearman_r_h": 0.06023270832238252, "slope_h_on_r": -0.07941290144521744}
- direct_lr0.0003_s0_dtarget: {"n_pairs": 816, "spearman_r_h": 0.7401502800247538, "slope_h_on_r": 0.9910251298223275}

## Notes (worker, 2026-10-12)

- Job 1035097, nodesumo01 (NVIDIA H100 NVL), 58 min, exit 0.  Data read: the R7-20 fixtures only (GQA 200, TextVQA 200) plus the D1-20
  smoke artefacts (train_smoke checkpoints, cache_smoke, D1-10 probe records); no GQA DEV, TextVQA val or ChartQA data.
- Selector inputs: 42.7 % of the PCA'd fp16 values are bit-identical to the D1-20 smoke cache and the largest difference is 0.25 (one fp16
  ulp at magnitudes 256–512).  The cache was computed on an L40S (cache_smoke.json), this smoke on an H100 NVL: cross-hardware GEMM numerics,
  not a pipeline difference.  Question token ids 68 / 68 identical.  The torch 2.9 selector checkpoints load in the SparseVLM env (torch 2.1.2).
- Full-model answers through the D1-30 runner (batch 32, left padding, max_new_tokens 128): 200 / 200 GQA and 200 / 200 TextVQA (OCR prompt)
  identical to the R7-20 official-path answers.  SparseVLM 192 / 64 rows: 198–200 / 200 identical to R7-20 (SparseVLM's known run-to-run
  agreement, frozen protocol: 99.1 %).  The official counter equals the exact prediction for all 34 rows (4 anchors, 4 adapted, OCR anchors).
- Per-answer cost (H100 NVL, batch 32): GQA 23 ms, TextVQA 29 ms, ChartQA-prompt proxy 23 ms; SparseVLM batch 1: 0.10 / 0.14 / 0.10 s.
- Latency (gqa_fixture, 64-item pools): B = 4 full 88.3 ms / item, uniform grid 25 % 65.6 ms (−25.8 %), selector 25 % 66.3 ms (selection
  3.8 ms per batch); the frozen D1-10 timing smoke gave 84.4 / 61.4 ms (−27.2 %).  B = 1: SparseVLM rows 196–212 ms vs our full row 177 ms, so
  the batch-1 latency-matched SparseVLM curve did not reach L25 / L50 / L75 in the smoke ('not reached' is the scorer's documented outcome).
- Post-smoke timing edits (CPU tests updated, 18 / 18 pass): images are now read and decoded before the clock starts (as the frozen D1-10
  timing; this smoke's vision time included the disk read), and dynamic rows time full batches per budget bucket (items cycled to fill the
  last batch) instead of partial batches, which had inflated the smoke's B = 16 / 32 dynamic rows.  A GPU re-smoke of the timing path
  (gqa_fixture, 6 configurations, all batch sizes, plus SparseVLM batch 1) is job 1035183 (outputs/R7_D1/d1_30_smoke/timing_v2).

## Timing re-check after the two post-smoke edits (job 1035183, node50, NVIDIA L40S — not the reference GPU)

gqa_fixture, 64-item pools (16 at B = 1), smoke selector S1 seed 0; per-item latency median [bootstrap 95 % CI] and saving vs full:

| B | full | uniform grid 25 % | selector 25 % | dynamic f = 0.5 |
|---|---|---|---|---|
| 1 | 259.9 [258.6, 260.4] | 218.3 [218.0, 218.3] (−16.0 %) | 219.1 [219.1, 219.5] (−15.7 %) | 249.1 [247.1, 249.3] (−4.2 %) |
| 4 | 136.7 [136.2, 137.9] | 92.1 [91.6, 93.1] (−32.7 %) | 94.9 [93.7, 95.1] (−30.6 %) | 112.4 [111.8, 113.9] (−17.8 %) |
| 16 | 116.9 [116.1, 117.7] | 61.2 [60.9, 61.5] (−47.7 %) | 61.7 [61.3, 62.9] (−47.2 %) | 84.1 [83.1, 85.6] (−28.1 %) |
| 32 | 113.8 [112.6, 114.9] | 58.8 [58.4, 59.2] (−48.3 %) | 59.2 [58.4, 60.0] (−48.0 %) | 80.4 [80.0, 80.8] (−29.3 %) |

- Image read and decode outside the clock (by construction: the clock starts after the images are opened): vision per batch at B = 1 is
  26.3 ms here (L40S) against 32.2 ms in the first smoke (H100 NVL, read inside the clock); cross-GPU, so indicative only.
- Full batches per budget bucket: every stage-2 batch is full (items cycled where a bucket of the 64-item pool is smaller than B); the
  dynamic rows' median CIs at B = 16 / 32 are now 0.4–2.5 ms wide (first smoke: up to 62 ms with partial batches), and at every B each
  dynamic row lies between uniform 25 % and full (f = 0.5 at B = 4: 51.6 % of items at 72 tokens, 39.1 % at 576 on this fixture).
- SparseVLM batch 1 (L40S): 270–290 ms per item across the 8 rows (our full row: 260 ms).  The run exited 0; outputs in
  outputs/R7_D1/d1_30_smoke/timing_v2.
