# R7-30 D1-10 — CPU / GPU smokes (2026-10-12)

**Scope.** These smokes were run before any FIT / CAL / DEV unit, as the coordinator's R7-30 request asked.
- No full run was launched and nothing was committed.
- Data touched: only the R7-20 GQA val fixture (`/projects/EEG-foundation-model/yinghao/data/r7_fixtures/gqa.jsonl`).
- Protocol draft: `reports/R7/R7_D1_PROTOCOL_DRAFT_20261012.yaml`.

## CPU tests (jobs 1034887 and 1034888)

`tests/test_r7_d1.py` (+ `tests/test_r7_fixtures.py`) passes 23 / 23 in the SparseVLM env. The main env passes 23 / 23 with 1 skip, the m4c test, which needs llava.

**Masks:**
- Every ordering is a permutation, kept sets are nested, and counts are exact.
- Same seed gives the same ordering; a different question gives a different ordering.
- The grid structure and its set probabilities are checked against an independent enumeration of 4! × 2 choices.
- The uniform log-probability equals −log C(576, k).
- Plackett–Luce prefix probabilities sum to 1, checked by brute force.

**Data roles:**
- Role counts are exact, and FIT / CAL / DEV are image-disjoint (`assert_group_disjoint`).
- The output does not depend on input order.
- Class weights balance to 1/2.
- QA pairs seen by LLaVA mix665k are excluded.
- The shard plan puts CAL first. Feed lines contain no `--export`, use H100,L40S, and exclude node51 / 52 / 60.

**Teacher bookkeeping** (fake CPU host):
- Each variant sees exactly k visual tokens.
- Results do not depend on batching.
- After a crash, resume completes the shard with no duplicates.
- A reason-coded image skip is accounted for in coverage.
- The analyser's coverage guard refuses a missing row, and DEV stays sealed.

**Analysis:** hand-computed h, d and h − d match, and the identity error is below 1e-12.

## GPU smoke (job 1034889, nodesumo01, NVIDIA H100 NVL)

One host process ran probe then timing. Staging plus load took 380 s.

### Probe: 68 binary items on 57 images; all six sentinels pass

| sentinel | threshold | measured |
|---|---|---|
| S1 spliced full path vs official R7-20 path | max \|ΔP(yes)\| ≤ 0.02, no flips | 0.0038, 0 flips |
| S2 bit-identical replay | identical | identical |
| S3 masking acts: mean \|T_72 − T_576\| | ≥ 0.02 per mechanism | random 0.248, grid 0.257, qsim 0.272 |
| S4 yes / no mass (full) | ≥ 0.9 | min 0.997 |
| S5 batched left-padded generation = official answers (200 GQA fixture items) | ≥ 98 % | 100 % |
| S6 nested kept sets, exact counts | true | true |

Teacher throughput: 68 questions × 14 prefill passes (952 sequences) in 23.0 s, i.e. 0.34 s per question, including image encoding and embedding assembly.

### Timing: latency-regime calibration (no answer scored)

Uniform-grid masks, batch of fixture items. Per-item latency is the median of 3 after one warm-up, in ms. **total** = vision + prefill + 8 forced decode steps. Vision is the official fp32 CLIP path plus CPU preprocessing.

| batch | visual tokens | vision | prefill | prefill + 8 decode | total |
|---|---|---|---|---|---|
| 1 | 576 / 432 / 288 / 144 / 72 | 29.6 – 30.2 | 34.0 / 26.2 / 20.1 / 15.5 / 14.8 | 140.0 / 132.4 / 126.9 / 123.6 / 123.3 | 169.7 / 162.7 / 157.1 / 153.8 / 153.5 |
| 4 | same | 23.1 | 32.9 / 23.9 / 18.0 / 11.1 / 9.0 | 61.2 / 52.0 / 45.3 / 38.3 / 36.5 | 84.4 / 75.1 / 68.4 / 61.4 / 59.6 |
| 16 | same | 21.1 – 21.8 | 32.3 / 24.3 / 17.2 / 9.6 / 6.8 | 44.1 / 34.6 / 26.2 / 17.4 / 14.2 | 65.7 / 56.4 / 47.9 / 38.5 / 35.4 |
| 32 | same | 20.3 – 20.6 | 31.8 / 23.8 / 16.5 / 9.6 / 6.6 | 41.4 / 31.7 / 22.7 / 14.5 / 10.9 | 62.0 / 52.0 / 43.0 / 34.9 / 31.3 |

Per-item total latency saved by uniform-grid pruning, relative to 100 %:

| batch | 12.5 % | 25 % | 50 % |
|---|---|---|---|
| 1 | 9.5 % | 9.4 % | 7.4 % |
| 4 | 29.3 % | 27.2 % | 18.9 % |
| 16 | 46.2 % | 41.4 % | 27.1 % |
| 32 | 49.5 % | 43.6 % | 30.6 % |

**Regime rule.** The rule was written in the protocol draft before any timing row was read. The draft file was first written at about 00:03 UTC; the first timing row printed at 00:06:19 UTC; later edits to the file changed YAML quoting only. The rule picks the smallest batch size at which uniform-grid 25 % saves ≥ 25 % of per-item total latency. **That gives B\* = 4** (27.2 %). B = 1 fails (9.4 %).
- **Larger batches.** B = 16 saves 41 %. Moving the primary regime there is a coordinator decision (OPEN-2). It is not something to switch to after seeing quality.

**Readings.**
1. **Pruning at the LLM input saves prefill even at batch 1** (34 → 15.5 ms at 25 %). This differs from SparseVLM, which keeps all 576 tokens through layers 0–1 and showed no saving in R7-20.
2. **At batch 1, the 8 decode steps (about 106 ms) and vision (about 30 ms) dominate**, so the total barely moves.
3. **The official fp32 CLIP tower is about 20 ms per item even at batch 32.** That is a third of the 100 % total. Casting it to fp16 would raise the LLM-side share, but would leave the official host path. It is not proposed.

## D1-10 cost from the smoke

- **Teacher.** 0.34 s per question on H100 NVL, so about 56k questions ≈ 5.3 GPU-h on H100. L40S is expected about 1.5–2× slower.
- **Model loads.** 14 bundles × about 6 min ≈ 1.4 GPU-h.
- **Range: about 7–12 GPU-h total.** The probe's chunk was small (68 questions), so full shards should batch better.

## Launch gates (added after the smoke; tests 1034913 / 1034914: 23 / 23 in both envs)

A D1-10 bundle refuses to start in either of these cases:
- **Unfrozen manifest.** The manifest in `outputs/R7_D1/data` is not frozen (`scripts/r7_d1_data.py freeze --protocol R7-30-D1 --go <reference>`), or its sha256 differs from the frozen record.
- **No passing probe.** `outputs/R7_D1/smoke/probe/probe_report.json` does not show `pass: true`.

After the freeze, launch with:
```
CAP=27 nohup setsid bash slurm/feed_normal.sh slurm/r7_d1_10_feed_lines.txt >/dev/null 2>&1 &
```
The lines file holds 14 bundle lines b000–b013. Each line uses H100,L40S only, excludes node51 / 52 / 60, and passes no `--export`.

## Provisional manifest at full scale (job 1034911; NOT frozen, no mix665k exclusion)

| | value |
|---|---|
| GQA train-balanced questions | 943,000 |
| binary (yes / no) | 331,898 (35.2 %) |
| eligible images | 66,852 of 72,140 |
| FIT | 10,000 images / 31,105 q (yes 15,170) |
| CAL | 2,000 / 6,168 (yes 3,010) |
| DEV | 2,000 / 6,206 (yes 3,048) |
| pi_yes (FIT) | 0.488, so w_yes = 1.025 and w_no = 0.976 |
| shard plan | 56 shards / 14 bundles; 43,479 questions |
| images.zip check | all 14,000 images present (148,854 images in the zip) |

**Teacher cost at this size:** 43.5k questions × 0.34 s ≈ 4.1 GPU-h on H100, plus about 1.4 GPU-h of loads, so about 6–10 GPU-h with L40S.

## Data status

- GQA `images.zip` is complete (sha256 02ce5c49… per the steward) and extracted: 148,856 entries in `/projects/EEG-foundation-model/yinghao/data/gqa/images`.
- `questions1.2.zip` is being written by the steward and is not yet verified.
- The manifest has not been built yet. It waits for the questions file and for OPEN-3, the mix665k QA-pair exclusion.
