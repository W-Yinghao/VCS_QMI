# R7-20 — bounded feasibility fixtures for D1 / D2 / D3 (2026-10-12)

**Scope.** Directive R7-20: fixed small fixtures, not the full grid. D1 uses one host plus one official token-pruning baseline at 2 budgets. D2 runs the host on one POPE configuration and checks the M-HalDetect labels. D3 is the finite conditional-distribution check.

**Run conditions.**
- Everything ran through sbatch on normal QOS.
- GPU jobs ran on H100 or L40S, excluding nodes 51 / 52 / 60. The SparseVLM env (torch 2.1.2) has no Blackwell kernels. The Qwen run used the existing `vl_baselines` env unchanged.
- No GQA testdev / test, TextVQA test or other official test file was downloaded or read.
- Generated tables: `outputs/R7_fixtures/r7_fixture_tables.md`, written by `scripts/r7_fixtures_tables.py`. They are reproduced in the appendix.
- CSVs: `reports/R7/baseline_reproduction.csv` and `reports/R7/compute_budget.csv`. They contain measured numbers only.

## 0. Summary

**D1: the host and the official baseline reproduce offline.**
- **Hosts.** LLaVA-1.5-7B (`liuhaotian/llava-v1.5-7b`, official LLaVA code) and Qwen2.5-VL-7B-Instruct, the host the D1 plan names.
- **Baseline.** SparseVLM official code at 192 and 64 of 576 visual tokens.

| Benchmark | LLaVA-1.5-7B, ours | Paper |
|---|---|---|
| TextVQA val, 200 items | 58.7 [51.6, 65.5] | 58.2 |
| POPE adversarial, 300 items | F1 81.9 | 85.9 (F1 averaged over 3 splits) |

| Benchmark | SparseVLM-192, ours | SparseVLM v4 paper | SparseVLM-64, ours | SparseVLM v1 paper |
|---|---|---|---|---|
| TextVQA | 58.4 (Δ −0.4 vs full) | 57.8 (Δ −0.5) | 50.9 | 51.8 |
| POPE F1 | 80.0 | 85.3 (3 splits) | 75.5 | 75.1 (3 splits) |

- **Qwen2.5-VL TextVQA:** 83.0 [77.5, 88.0], against 84.9 in the report.
- **GQA does not reproduce, and that is expected.** Our GQA fixture is on val-balanced: LLaVA scores 76.0 there against 62.0 on testdev. GQA val images come from Visual Genome, which LLaVA-1.5 trained on, so this is not a reproduction gap.

**D1: pruning FLOPs do not become latency at batch 1.**
- **Same-GPU comparison (H100 NVL).** The full host needs 71–107 ms per item. SparseVLM needs 76–120 ms at 192 tokens and 74–115 ms at 64. So pruning is *slower* end to end, even though its own counters show 192.0 / 64.8 equal tokens and 1.8–2.1 → 0.9–1.3 TFLOPs.
- **Why.** Prefill, excluding the selector, shrinks by ≤ 1 ms at 192 tokens and by 2.6–4.3 ms at 64. The selector adds 2.9 ms, and each SparseVLM decode step costs 3.3–4.1 ms more.
- **Qwen2.5-VL is dominated by vision.** At native resolution the vision encoder takes 62–70 % of wall time (306–971 visual tokens). Post-encoder pruning cannot remove that cost.

**D1: matching the full model is not the same as keeping its risk.** Finite residual check on binary items:
- **Identity.** h − d = 2(C − b)(b − a) holds to 3e-16.
- **Sign flips across fixtures.** The mean residual is +0.027 [+0.010, +0.044] on GQA at 64 tokens, so d underestimates h. It is −0.026 [−0.046, −0.006] on POPE at 64 tokens, so d overestimates h.
- **Per-item behaviour.** h < 0 for 21–43 % of items. corr(h, d) is 0.25 on POPE at 192 tokens.

**D2: POPE has no text shortcut, but M-HalDetect does.**
- **POPE host.** POPE-adversarial accuracy is 82.3 [78.3, 86.3], P(yes) AUROC 0.909.
- **POPE text-only control.** Same prompt, no image: accuracy 49.3 [44.7, 54.0], AUROC 0.42. There is no text shortcut on POPE.
- **M-HalDetect has a strong pure-text shortcut.** Image-grouped folds of the official train split:
  - INACCURATE vs ACCURATE: AUROC 0.859 ± 0.007. The main cues are counts ("six", "several", "two", …).
  - ANALYSIS vs factual: AUROC 0.966.
- **M-HalDetect has image leakage.**
  - The official train and val splits share 27 images.
  - 2 of the 50 POPE-fixture images are also in M-HalDetect, because both use COCO val2014.

**D3: VCS and logistic have the same target, and n per condition decides usability.**
- **Same target.** VCS ≈ logistic in every one of the 13 distributions × 3 n.
- **n = 8 per condition.** Learned critics are worse than the null critic in 17 / 36 shifted cells (VCS) and 16 / 36 (logistic). The mean relative posterior MSE is 0.82.
- **n = 32.** Worse than null in 4 / 36 cells; relative error 0.40.
- **n = 128.** Worse than null in 0 / 36 cells; relative error 0.13.
- **Kernel critic.** The same-target kernel critic is better at n ≤ 32 (relative 0.53 / 0.33) but saturates at n = 128 (0.19).
- **The unconditional CMMD-like MMD² is blind.** It stays ≈ 0 for marginal-preserving conditional mismatch even at S = 0.667, while the grouped conditional MMD² reaches 0.29–0.33.

**Verdict.** D1 is feasible as the main route now, with the design consequences in §6. D3's finite results are ready to support it as theory and diagnostics. D2 is feasible only with a text-only branch control and image-disjoint splits.

## 1. Fixtures (pre-registered by a hash rule before any model output; grouped by image)

**Selection rule.**
1. Images are ordered by `sha256("r7-20/<fixture>-group/<image>")`.
2. Within an image, questions are ordered by `sha256("r7-20/<fixture>-item/<qid>")`.
3. GQA and TextVQA take up to 2 questions per image. POPE takes all 6 questions of 50 images.

The ids and per-file sha256 are in `/projects/EEG-foundation-model/yinghao/data/r7_fixtures/fixture_manifest.json`, which `outputs/R7_fixtures/fixture` links to.

| fixture | source (steward-verified sha256) | items | images | binary items | jsonl sha256 |
|---|---|---|---|---|---|
| GQA val-balanced | lmms-lab/GQA a6e72d6 (val only) | 200 | 102 | 68 | 913c9a0f…ca992f4 |
| TextVQA val | lmms-lab/textvqa 9c0699c (validation only) | 200 | 127 | 0 | 8242e5a3…e6f9521 |
| POPE adversarial | lmms-lab/POPE 4db1276 | 300 | 50 | 300 | 9d12927b…fe9d28d |

**Official-split exposure ledger.** Any later official evaluation must exclude or disclose these items:
- 200 GQA val-balanced questions. GQA testdev stays closed.
- 200 TextVQA val questions. The directive named this split.
- 300 POPE-adversarial questions. POPE is evaluation-only; the directive asked for one configuration.

**Checkpoint safety.** Both LLaVA `.bin` shards load with `torch.load(weights_only=True)`: 241 + 86 tensors, nothing else. Qwen ships as safetensors.

## 2. D1 — host + SparseVLM (numbers in Appendix A1–A5)

**Protocol.** The fixture follows LLaVA's official eval path:
- Data loader: `model_vqa_loader` with prompt `vicuna_v1` and the short-answer suffix. TextVQA uses the `v051_ocr` 3-line prompt, with OCR tokens from lmms-lab `ocr_tokens`.
- Decoding: greedy, fp16, batch 1, `max_new_tokens` 128.
- GQA: `convert_gqa_for_eval` normalisation, then exact match.
- TextVQA: the m4c evaluator from the repository.
- POPE: the `eval_pope.py` parsing.
- SparseVLM: USE_VERSION 1_0, the repository default; RETAIN_TOKN 192 and 64.
- Qwen2.5-VL: bf16 with SDPA, 256–1280 × 28² pixels, question + short-answer suffix with no OCR tokens (lmms-eval style), and the shipped generation config with sampling off.
- Confidence intervals: image-cluster bootstrap.

**Reproduction against the papers.**
- **TextVQA reproduces for both hosts.** LLaVA scores 58.7 against 58.2. Qwen scores 83.0 against 84.9. Both papers' values fall inside our 95 % intervals.
- **POPE is consistent.** We measure F1 81.9 on the adversarial split alone, which is the hardest of the three splits LLaVA averages to get 85.9.
- **SparseVLM's accuracy changes, paired on the same items:**

| Budget | TextVQA, ours | TextVQA, paper | POPE acc, ours | GQA, ours |
|---|---|---|---|---|
| 192 | −0.4 [−3.4, +2.7] | v4 −0.5 | −1.0 [−3.3, +1.0] | −5.5 [−9.4, −2.0] |
| 64 | −7.8 [−12.6, −3.0] | v1 −6.4 / v4 −4.9 | −4.0 [−7.7, −0.7] | −14.5 [−20.1, −9.0] |

- **Absolute SparseVLM-64 matches v1:** TextVQA 50.9 against v1 51.8, and POPE-adversarial F1 75.5 against v1 75.1.
- **GQA drops more on our val fixture than the paper's −2.4 to −9.2 on testdev.** On these images the full host is unusually strong, at 76.0.
- **v1 or v4?** The repository's default mode appears closer to the v1 table than to the ICML v4 table at 64 tokens. With n = 200 this cannot be decided.

**Repeatability.**
- The full host gives identical answers on L40S and H100 NVL for 700 / 700 items; max |ΔP(yes)| is 0.004.
- SparseVLM-192 is identical on 99.1 % of items; max |ΔP(yes)| is 0.138, because top-k selection is numerically fragile.
- The D1 and D2 jobs gave identical POPE answers for 300 / 300 items.

**Timing.** All numbers are CUDA-synchronised, per item; Appendix A4 has the full split.
- **Same GPU, H100 NVL.**
  - Full host: vision 22, prefill 31–34, decode 15–49 ms.
  - SparseVLM: vision 21, selector 2.9, prefill 28–33, decode 19–60 ms.
- **Where SparseVLM's own counter agrees.** It reports prefill-layer CUDA time of 32.6–34.7 ms at 192 tokens and 29.8–31.2 ms at 64.
- **Why there is no latency gain at batch 1.**
  - Prompts are short: 63–109 text tokens plus 576 visual.
  - Answers are 1–3 decode steps.
  - The pruned prefill is not compute-bound on an H100.
  - SparseVLM's custom decoder layers make each decode step 3.3–4.1 ms slower.
- **The CLIP tower runs in fp32.** This is the official builder's `device_map="auto"` path. It is why vision is slightly faster on L40S (18 ms) than on H100 NVL (21–22 ms).
- **Qwen2.5-VL** uses 306 (GQA), 971 (TextVQA) and 354 (POPE) visual tokens on average. Vision takes 73–203 ms, prefill 20–40 ms and decode 15–43 ms.

**Finite-model check (h vs d)** on binary items: 68 GQA and 300 POPE. C is the label, b = P_full(yes) and a = P_pruned(yes), with the yes/no mass about 1.000 throughout. Results are in Appendix A5.
- **The identity holds exactly.**
- **d is neither an upper nor a lower bound on the risk change h.**
  - On GQA, where the full model is right and confident, the residual is positive: significant at 64 tokens.
  - On POPE, where the full model makes errors, the residual is negative: significant at 64 tokens. Pruning moved some answers toward the truth, with h < 0 for 33–42 % of items.
- **Per item, d tracks h only on GQA.** corr(h, d) is 0.90–0.94 on GQA but 0.25–0.56 on POPE.

## 3. D2 — POPE host, text-only control, M-HalDetect (Appendix A6)

**POPE-adversarial, 300 items.**
- **With the image:** accuracy 82.3, precision 83.9, recall 80.0, F1 81.9, yes-ratio 47.7 %, P(yes) AUROC 0.909.
- **Text-only, same prompt with no image token:** accuracy 49.3 [44.7, 54.0], AUROC 0.424. That is at chance, and slightly anti-correlated, because adversarial negatives are plausible co-occurring objects.

**M-HalDetect** (`hendryx-scale/mhal-detect` 6ef75f5, annotations only, CC BY-NC 4.0). The loader `scripts/r7_fixtures_d2.py:load_mhal` turns every annotated segment into a row keyed by its source COCO-val2014 image. Grouped folds are checked with the reference `assert_group_disjoint`.

**Schema.**

| | train | val |
|---|---|---|
| responses | 10,979 | 3,164 |
| segments | 75,627 | 22,092 |
| images | 2,695 | 786 |
| distinct questions | 44 | 43 |
| responses per image | about 4 (max 8) | about 4 (max 8) |

- **Labels (train):** ACCURATE 40,209, ANALYSIS 17,289, INACCURATE 16,874, UNSURE 1,255.
- **Spans** reproduce the text exactly and cover 97 % of each response.

**Problems found.**
1. The official train and val splits share 27 images, so splits must be regrouped by image.
2. Of the 50 POPE-fixture images, 2 are in M-HalDetect. Any M-HalDetect-trained verifier must exclude POPE images.
3. **Pure-text shortcut**, out-of-fold over 5 image-grouped folds of the train split only:
   - INACCURATE vs ACCURATE: n-gram logistic AUROC 0.859 ± 0.007, position-only 0.739, length-only 0.542.
   - ANALYSIS vs ACCURATE ∪ INACCURATE: AUROC 0.966.
   - The INACCURATE cues are counts and quantifiers: "six", "several", "10", "12", "four", "two", "possibly".
   - Consequence: a verifier that reports, say, AUROC 0.88 has shown only about 0.02 of image dependence. The D2 design must report the text-only branch as the baseline to beat. As the atlas requires, ANALYSIS and UNSURE must not be counted as errors.

## 4. D3 — finite conditional distributions (Appendix A7)

**Construction** (`scripts/r7_fixtures_d3.py`).
- 16 conditions with uniform π. States are 16 blocks × 3 sub-modes × {hi, lo} quality = 96 states.
- P_ref lives on the block of its own condition.
- 13 Q laws: clean, plus 4 families at 3 strengths each:
  - Conditional mismatch keeps the z-marginal exactly.
  - Mode-missing, quality and contamination move the z-marginal. Shared cyclic modes would also have kept the marginal for mode-missing, so sub-modes are condition-specific.
- Critics see only fixed random embeddings. An image carries a noisy trace of its own class: e(z) = [0.7 g(b) + 0.7 u(b, k), onehot(q)].
- S, J(T), the posterior MSE S − J(T) = E_M(T − η)² and the per-condition versions are exact.
- The finite truth equals the package reference `conditional_s` to 1e-12; this is tested.

**Results**, 117 cells = 13 distributions × n ∈ {8, 32, 128} × 3 draws, in 323 s CPU.

**1. Same target.** VCS (J on T = tanh f) and logistic (BCE on 2f, read through T = tanh f) agree in every cell:
- within 0.01–0.02 at n = 8;
- within 0.0013 at n = 128.

**2. Small n per condition is unusable.**

| n per condition | worse than null (VCS / logistic / kernel) | mean relative posterior MSE (VCS / logistic / kernel) |
|---|---|---|
| 8 | 17 / 16 / 4 of 36 | 0.82 / 0.82 / 0.53 |
| 32 | 4 / 4 / 1 | 0.40 / 0.40 / 0.33 |
| 128 | 0 / 0 / 0 | 0.13 / 0.13 / 0.19 |

- Held-out Ĵ always underestimates S, by the posterior MSE plus small EVAL noise. It is conservative, and it stays negative on the clean law.
- Small effects are hardest. Contamination at λ = 0.1 (S = 0.049) still has about 53 % relative error at n = 128.

**3. The kernel critic trades bias for variance.** The same-target closed-form kernel critic (RFF D = 256, ridge and bandwidth chosen on SELECT) is best at n ≤ 32. At n = 128 it saturates: posterior MSE 0.03–0.11 on cond_mismatch and contamination, against 0.008–0.026 for the MLP critics.

**4. Detection is not S.** These are discrimination statistics and are never subtracted from S.
- The unconditional MMD² on e(z) is a CMMD-like control. It is −0.002…0.000 for conditional mismatch at every strength, up to S = 0.667: blind by construction.
- It is weak for mode-missing (≤ 0.006) and for contamination (≤ 0.006).
- It detects quality shifts (0.035–0.041 at λ = 0.8).
- The grouped conditional MMD² detects all families except the weakest contamination and quality settings.

## 5. Measured costs (`reports/R7/compute_budget.csv`)

**Inference per item, mixed fixture.**

| configuration | GPU | s / item |
|---|---|---|
| LLaVA full | L40S | 0.178 |
| LLaVA full | H100 NVL | 0.099 |
| SparseVLM-192 | H100 NVL | 0.108 |
| SparseVLM-64 | H100 NVL | 0.103 |
| Qwen2.5-VL-7B | H100 NVL | 0.185 |

**Extrapolation.** One configuration on GQA testdev-bal + TextVQA val (17,578 items) costs 0.48–0.90 GPU-h of inference. A POPE arm over 3 × 3,000 items costs about 0.35 GPU-h.

**Peak memory:** 14.7 GiB for LLaVA, 18.4 GiB for SparseVLM, 15.8 GiB for Qwen.

**The real cost was model loading over the shared NFS.**
- Direct mmap loading of the 13.5 GB checkpoint took 1,085 s on L40S, and 966 s for D2. Page faults ran at about 17 MB/s, against 63 MB/s measured for sequential direct reads.
- The host now stages the checkpoint to node-local ext4 first: 255 s for LLaVA, 333 s for Qwen, then a fast load.
- A full D1 grid should keep one process per host alive across arms and budgets, rather than reloading.

**CPU work.**
- D3: 323 s for 117 cells. Per fit: VCS 10.0 s, logistic 9.4 s, kernel 1.8 s.
- M-HalDetect loader and shortcut cross-validation: 73 s.

**GPU jobs run.**
- 1034742: d1all. Full host completed; 28 min, mostly two unstaged loads.
- 1034744: D2 POPE, 18 min.
- 1034798: SparseVLM 192 / 64 and Qwen, 18 min.
- 1034842: same-GPU full-host timing, 8 min.

## 6. Gaps, deviations, issues

**Deviations from the plan.**
- **Host.** LLaVA-1.5-7B, the directive's example, is the reproduction host, for parity with the SparseVLM and FastV tables. Qwen2.5-VL-7B, the plan's named host, ran host-only. Official SparseVLM code is LLaVA-based.
- **Baseline.** SparseVLM, because it runs offline under Apache-2.0. FastV was not run: its official code pins transformers 4.31 and has no licence file. Its published numbers appear in SparseVLM's tables, which are quoted above.
- **GQA split.** GQA uses val-balanced because testdev is closed, so GQA is not like-for-like. The main D1 comparison needs one pre-registered testdev read.

**Not run (outside R7-20).**
- PyramidDrop, FEATHER, IF-Prune.
- Selector training (VCS / logistic → J / direct risk). This needs FIT / CAL from GQA train_balanced and its images; the steward reports the GQA image download at 9.8 / 21.8 GB.
- ChartQA and DocVQA.
- VCD, HallusionBench, AMBER.
- POPE random and popular.
- M-HalDetect verifier training, which needs COCO val2014; the steward reports it is extracting.
- The natural-image D3: generators, CMMD / CFID / cFreD.

**Environment.**
- New env `/projects/EEG-foundation-model/yinghao/envs/sparsevlm`:
  - Python 3.10, torch 2.1.2+cu121, transformers 4.37.0 per the SparseVLM README (its pyproject says 4.37.2), flash-attn 2.3.3 wheel, numpy 1.26.4.
  - Frozen in `FREEZE.txt`.
- **User-site leak.** `~/.local/lib/python3.10` had masked several dependencies at install time. They were installed into the env, and every job runs with `PYTHONNOUSERSITE=1`.
- `vl_baselines` was used unchanged.

**Failures and fixes.**
- **Tensor counters broke JSON.** SparseVLM keeps 0-d tensor counters, and `run_sparsevlm192.json` failed to serialize (job 1034742, exit 3).
- **Fixes.**
  - All JSON goes through `V.jdump`, which handles torch, numpy and Path values.
  - Answers are appended and flushed one by one, and run files are rewritten per fixture.
  - Reruns skip finished (fixture, qid) pairs and drop a torn last line.
  - New CPU tests use a fake host that returns tensors and crashes mid-fixture, and cover the crash / resume / serialization path. 14 / 14 pass in both envs.
- **The untimed first SparseVLM-192 run** is kept as `outputs/R7_fixtures/d1/run1_untimed_pred_sparsevlm192.jsonl` and was used for the repeatability check.
- **Queue.** Normal QOS briefly reached 28–29 twice when other jobs arrived. Each time I cancelled or deferred my own pending job.

## 7. Which route looks feasible

**D1, question-conditioned visual token pruning: feasible as the main route now.** Host, official baseline, official evaluators, timing split and the h / d check all run offline on fixtures. Three design consequences follow from the measurements.

1. **Claim efficiency in measured latency, not FLOPs.** It should also be measured in a regime where prefill matters: higher-resolution or many-token inputs, or batched serving. At batch 1 on LLaVA-1.5 short answers, the official baseline gains nothing. On Qwen2.5-VL the vision encoder, not prefill, dominates.
2. **Train selectors on risk, and report task risk.** Fidelity to the full model (d) is not a surrogate for the risk change (h). The residual changes sign between GQA and POPE.
3. **Do not use GQA val for LLaVA-1.5.** Its images are exposed through Visual Genome. Pre-register one testdev read, or use benchmarks without that exposure.

**D3, conditional generation: the finite theory is clean and cheap.**
- It shows that S needs ≥ 32 (better ≥ 128) samples per condition.
- It shows that unconditional metrics miss marginal-preserving conditional errors.
- The natural-image version is not yet set up.

**D2, factual support: feasible but riskier.**
- POPE behaves well: the host reproduces and there is no text shortcut.
- The M-HalDetect training labels carry a strong text-only shortcut, cross-split image leakage and COCO overlap with POPE.
- The design needs a text-only branch baseline, image-regrouped splits and POPE-image exclusion, and it waits on COCO val2014.

## 8. Files
- **Scripts:**
  - `scripts/r7_fixtures_d1.py` (build / run / score)
  - `scripts/r7_fixtures_d2.py` (pope / text)
  - `scripts/r7_fixtures_d3.py`
  - `scripts/r7_vl_common.py`: fixtures, hosts, staging, timing, metrics, safe JSON
  - `scripts/r7_fixtures_tables.py`
- **Tests:** `tests/test_r7_fixtures.py` (14 tests).
- **Wrappers:**
  - `slurm/r7_fixtures_cpu.sbatch`: tests | tests_vl | d3 | d2text | d1build | tables
  - `slurm/r7_fixtures_gpu.sbatch`: d1all | d2pope | d1qwen | d1rest | d1fulltime
- **Outputs:**
  - `outputs/R7_fixtures/{d1,d2}/`: raw predictions, run records, summaries
  - `outputs/R7_D3/d3_finite.json`
- **Fixtures:** `/projects/EEG-foundation-model/yinghao/data/r7_fixtures/`

# Appendix — generated tables (verbatim from `outputs/R7_fixtures/r7_fixture_tables.md`)


## A1–A5. D1 — host and SparseVLM on the fixed fixtures

| config | GQA val-bal acc (n=200) | TextVQA val soft acc (n=200) | POPE-adv acc / F1 / yes-ratio (n=300) | equal visual tokens | GPU |
|---|---|---|---|---|---|
| full | 76.0 [70.1, 81.6] | 58.7 [51.6, 65.5] | 82.3 / 81.9 / 47.7 | 576 | NVIDIA L40S |
| full_h100nvl | 76.0 [70.1, 81.6] | 58.7 [51.6, 65.5] | 82.3 / 81.9 / 47.7 | 576 | NVIDIA H100 NVL |
| qwen25vl7b | 67.0 [60.5, 73.5] | 83.0 [77.5, 88.0] | 85.0 / 83.6 / 41.7 | 306 (GQA mean, dynamic) | NVIDIA H100 NVL |
| sparsevlm192 | 70.5 [64.0, 76.6] | 58.4 [51.7, 64.7] | 81.3 / 80.0 / 43.3 | 192.0 | NVIDIA H100 NVL |
| sparsevlm64 | 61.5 [54.5, 67.7] | 50.9 [44.1, 57.5] | 78.3 / 75.5 / 38.3 | 64.8 | NVIDIA H100 NVL |

Paper references (full official splits: GQA testdev-balanced 12,578 q, TextVQA val 5,000 q, POPE F1 over random / popular / adversarial):

| config | source | GQA | TextVQA | POPE F1 |
|---|---|---|---|---|
| full | LLaVA MODEL_ZOO (llava-v1.5-7b) | 62.0 | 58.2 | 85.9 |
| full | arXiv 2410.04417 (SparseVLM) v1 Tab.1 vanilla | 61.9 | 58.2 | 85.9 |
| full | arXiv 2410.04417 (SparseVLM) v4 Tab.1 vanilla | 61.9 | 58.3 | 85.9 |
| qwen25vl7b | arXiv 2502.13923 v1 (Qwen2.5-VL report) Tab.; GQA / POPE not reported | — | 84.9 | — |
| sparsevlm192 | arXiv 2410.04417 (SparseVLM) v1 Tab.1 | 57.6 | 56.1 | 83.6 |
| sparsevlm192 | arXiv 2410.04417 (SparseVLM) v4 (ICML 2025) Tab.1 | 59.5 | 57.8 | 85.3 |
| sparsevlm64 | arXiv 2410.04417 (SparseVLM) v1 Tab.1 | 52.7 | 51.8 | 75.1 |
| sparsevlm64 | arXiv 2410.04417 (SparseVLM) v4 (ICML 2025) Tab.1 | 53.8 | 53.4 | 77.5 |

Paired change vs the full host on the same items (points, image-cluster bootstrap 95 %):

| config | GQA | TextVQA | POPE-adv acc | answers changed (GQA / TextVQA / POPE) |
|---|---|---|---|---|
| sparsevlm192 | -5.5 [-9.4, -2.0] | -0.4 [-3.4, 2.7] | -1.0 [-3.3, 1.0] | 0.10 / 0.15 / 0.06 |
| sparsevlm64 | -14.5 [-20.1, -9.0] | -7.8 [-12.6, -3.0] | -4.0 [-7.7, -0.7] | 0.23 / 0.31 / 0.14 |

Timing split, ms per item (mean; CUDA-synchronised; batch 1, fp16, greedy):

| config | fixture | vision | selector | prefill (excl. selector) | decode (steps) | total wall | prompt tokens |
|---|---|---|---|---|---|---|---|
| full | gqa | 18.0 | 0.00 | 59.5 | 32.4 (1.3) | 112.2 | 64 |
| full | textvqa | 18.0 | 0.00 | 61.0 | 77.1 (3.2) | 159.1 | 109 |
| full | pope | 18.7 | 0.00 | 62.9 | 24.9 (1.0) | 108.7 | 63 |
| full_h100nvl | gqa | 22.1 | 0.00 | 31.7 | 20.2 (1.3) | 76.2 | 64 |
| full_h100nvl | textvqa | 21.8 | 0.00 | 33.6 | 48.5 (3.2) | 106.8 | 109 |
| full_h100nvl | pope | 22.4 | 0.00 | 31.4 | 15.3 (1.0) | 71.2 | 63 |
| qwen25vl7b | gqa | 73.0 | 0.00 | 21.6 | 19.2 (1.3) | 117.2 | 347 |
| qwen25vl7b | textvqa | 203.3 | 0.00 | 40.0 | 43.4 (2.9) | 291.5 | 1011 |
| qwen25vl7b | pope | 82.5 | 0.00 | 20.5 | 15.0 (1.0) | 121.4 | 393 |
| sparsevlm192 | gqa | 21.2 | 2.85 | 32.5 | 24.5 (1.3) | 83.4 | 64 |
| sparsevlm192 | textvqa | 21.3 | 2.89 | 33.0 | 60.1 (3.2) | 120.5 | 109 |
| sparsevlm192 | pope | 21.3 | 2.87 | 30.7 | 19.4 (1.0) | 76.5 | 63 |
| sparsevlm64 | gqa | 21.2 | 2.91 | 29.1 | 24.3 (1.3) | 79.8 | 64 |
| sparsevlm64 | textvqa | 21.3 | 2.91 | 29.3 | 58.7 (3.2) | 115.2 | 109 |
| sparsevlm64 | pope | 21.2 | 2.89 | 27.8 | 19.5 (1.0) | 73.6 | 63 |

Repeatability (full host, L40S vs H100 NVL): identical answers on 1.000 of 700 items; max |ΔP(yes)| = 0.004.

Repeatability (SparseVLM-192, L40S (untimed first run) vs H100 NVL): identical answers on 0.991 of 700 items; max |ΔP(yes)| = 0.138.

Finite-model check on binary items (C = label, b = P_full(yes), a = P_pruned(yes); h = (C−a)² − (C−b)², d = (b−a)², h − d = 2(C−b)(b−a)):

| config | fixture | n | mean h [95 %] | mean d [95 %] | mean residual [95 %] | P(h<0) | P(h>d) | corr(h,d) | Brier full → pruned | identity err |
|---|---|---|---|---|---|---|---|---|---|---|
| sparsevlm192 | gqa | 68 | 0.0296 [0.0031, 0.0641] | 0.0192 [0.0030, 0.0474] | 0.0103 [-0.0027, 0.0234] | 0.43 | 0.57 | 0.90 | 0.0783 → 0.1079 | 1.5e-16 |
| sparsevlm192 | pope | 300 | 0.0071 [-0.0056, 0.0195] | 0.0159 [0.0096, 0.0235] | -0.0089 [-0.0226, 0.0038] | 0.42 | 0.58 | 0.25 | 0.1288 → 0.1358 | 2.2e-16 |
| sparsevlm64 | gqa | 68 | 0.0881 [0.0472, 0.1366] | 0.0608 [0.0307, 0.0986] | 0.0273 [0.0104, 0.0437] | 0.21 | 0.79 | 0.94 | 0.0783 → 0.1664 | 1.4e-16 |
| sparsevlm64 | pope | 300 | 0.0291 [0.0078, 0.0504] | 0.0546 [0.0412, 0.0687] | -0.0255 [-0.0461, -0.0064] | 0.33 | 0.67 | 0.56 | 0.1288 → 0.1579 | 3.3e-16 |

## A6. D2 — POPE host fixture, text-only control, M-HalDetect

| condition | acc [95 %] | precision | recall | F1 | yes-ratio | AUROC of P(yes) | ms/item |
|---|---|---|---|---|---|---|---|
| image | 82.3 [78.3, 86.3] | 83.9 | 80.0 | 81.9 | 47.7 | 0.909 | 104.8 |
| text_only | 49.3 [44.7, 54.0] | 49.3 | 44.0 | 46.5 | 44.7 | 0.424 | 48.4 |

Repeatability: the D1 full-host job and the D2 job gave identical POPE answers on 1.000 of 300 items (max |ΔP(yes)| = 0.00e+00).

COCO-image overlap of the POPE fixture with M-HalDetect (train ∪ val): 2 of 50 images.

M-HalDetect schema (official splits): train: 10979 responses, 75627 segments, 2695 images, labels {'ACCURATE': 40209, 'ANALYSIS': 17289, 'INACCURATE': 16874, 'UNSURE': 1255}; val: 3164 responses, 22092 segments, 786 images, labels {'ACCURATE': 11770, 'INACCURATE': 4680, 'ANALYSIS': 5207, 'UNSURE': 435}

Official train/val image overlap: 27 shared images.

Pure-text shortcut check (official train only, 5 image-grouped folds; out-of-fold AUROC, fold sd):

| task | n | positive rate | n-gram logistic | position-only | length-only |
|---|---|---|---|---|---|
| inaccurate_vs_accurate | 57083 | 0.296 | 0.859 ± 0.007 | 0.739 | 0.542 |
| analysis_vs_factual | 74372 | 0.232 | 0.966 ± 0.002 | 0.759 | 0.573 |

Top n-gram cues for INACCURATE: six, several, 10, 12, four, two, ten, possibly, three, are two, nine, five

## A7. D3 — finite conditional distributions (exact S; 16 conditions; 3 draws)

Posterior MSE S − J(T) = E_M(T − η)² (population, exact), mean over draws; lower is better; the null critic scores S:

| distribution | S | z-marginal L1 shift | n=8 VCS / logistic / kernel | n=32 VCS / logistic / kernel | n=128 VCS / logistic / kernel |
|---|---|---|---|---|---|
| clean λ=0.00 | 0.0000 | 0.000 | 0.0090 / 0.0089 / 0.0015 | 0.0054 / 0.0054 / 0.0000 | 0.0019 / 0.0019 / 0.0001 |
| cond_mismatch λ=0.20 | 0.1111 | 0.000 | 0.1197 / 0.1195 / 0.1106 | 0.0701 / 0.0711 / 0.0575 | 0.0189 / 0.0186 / 0.0288 |
| cond_mismatch λ=0.50 | 0.3333 | 0.000 | 0.2314 / 0.2459 / 0.1960 | 0.0602 / 0.0586 / 0.0743 | 0.0173 / 0.0167 / 0.0439 |
| cond_mismatch λ=0.80 | 0.6667 | 0.000 | 0.1816 / 0.1917 / 0.1780 | 0.0380 / 0.0391 / 0.0866 | 0.0098 / 0.0085 / 0.0645 |
| mode_missing λ=0.33 | 0.0286 | 0.333 | 0.0374 / 0.0376 / 0.0250 | 0.0271 / 0.0270 / 0.0157 | 0.0089 / 0.0093 / 0.0080 |
| mode_missing λ=0.67 | 0.1250 | 0.667 | 0.1238 / 0.1123 / 0.0527 | 0.0328 / 0.0352 / 0.0365 | 0.0104 / 0.0098 / 0.0084 |
| mode_missing λ=1.00 | 0.3333 | 1.000 | 0.0728 / 0.0763 / 0.0617 | 0.0125 / 0.0163 / 0.0139 | 0.0043 / 0.0038 / 0.0078 |
| quality λ=0.20 | 0.0434 | 0.320 | 0.0790 / 0.0813 / 0.0164 | 0.0253 / 0.0255 / 0.0074 | 0.0043 / 0.0039 / 0.0029 |
| quality λ=0.50 | 0.1905 | 0.800 | 0.0668 / 0.0735 / 0.0211 | 0.0213 / 0.0228 / 0.0115 | 0.0045 / 0.0041 / 0.0045 |
| quality λ=0.80 | 0.4204 | 1.280 | 0.0771 / 0.0580 / 0.0288 | 0.0163 / 0.0155 / 0.0149 | 0.0043 / 0.0040 / 0.0068 |
| contamination λ=0.10 | 0.0492 | 0.080 | 0.0576 / 0.0576 / 0.0495 | 0.0563 / 0.0553 / 0.0497 | 0.0262 / 0.0257 / 0.0316 |
| contamination λ=0.30 | 0.1639 | 0.240 | 0.1708 / 0.1725 / 0.1387 | 0.0910 / 0.0917 / 0.0923 | 0.0254 / 0.0247 / 0.0590 |
| contamination λ=0.60 | 0.3931 | 0.480 | 0.2862 / 0.2878 / 0.2522 | 0.0962 / 0.0911 / 0.1459 | 0.0248 / 0.0239 / 0.1077 |

Held-out estimate error Ĵ_EVAL(T) − S (mean ± sd over draws; 512 EVAL samples per condition per side):

| distribution | n=8 VCS / logistic / kernel | n=32 VCS / logistic / kernel | n=128 VCS / logistic / kernel |
|---|---|---|---|
| clean λ=0.00 | -0.009±0.003 / -0.008±0.003 / -0.002±0.002 | -0.005±0.002 / -0.005±0.002 / -0.000±0.000 | -0.002±0.000 / -0.002±0.000 / -0.000±0.000 |
| cond_mismatch λ=0.20 | -0.119±0.003 / -0.119±0.003 / -0.110±0.009 | -0.071±0.010 / -0.072±0.012 / -0.057±0.006 | -0.017±0.001 / -0.016±0.001 / -0.027±0.002 |
| cond_mismatch λ=0.50 | -0.229±0.004 / -0.244±0.009 / -0.194±0.006 | -0.062±0.006 / -0.060±0.001 / -0.075±0.003 | -0.020±0.003 / -0.019±0.003 / -0.045±0.003 |
| cond_mismatch λ=0.80 | -0.185±0.027 / -0.195±0.023 / -0.180±0.022 | -0.030±0.013 / -0.031±0.015 / -0.079±0.013 | -0.012±0.001 / -0.011±0.000 / -0.067±0.005 |
| mode_missing λ=0.33 | -0.036±0.002 / -0.036±0.002 / -0.024±0.006 | -0.028±0.004 / -0.028±0.004 / -0.016±0.002 | -0.010±0.001 / -0.011±0.000 / -0.008±0.002 |
| mode_missing λ=0.67 | -0.125±0.044 / -0.116±0.053 / -0.054±0.011 | -0.034±0.011 / -0.036±0.011 / -0.037±0.008 | -0.016±0.001 / -0.016±0.001 / -0.014±0.001 |
| mode_missing λ=1.00 | -0.081±0.016 / -0.085±0.014 / -0.068±0.011 | -0.014±0.006 / -0.018±0.010 / -0.016±0.006 | -0.009±0.001 / -0.009±0.002 / -0.012±0.001 |
| quality λ=0.20 | -0.079±0.009 / -0.081±0.011 / -0.016±0.004 | -0.028±0.007 / -0.028±0.007 / -0.008±0.003 | -0.006±0.000 / -0.006±0.001 / -0.004±0.001 |
| quality λ=0.50 | -0.061±0.017 / -0.067±0.019 / -0.016±0.002 | -0.023±0.005 / -0.024±0.004 / -0.013±0.004 | -0.006±0.005 / -0.006±0.005 / -0.005±0.004 |
| quality λ=0.80 | -0.077±0.010 / -0.056±0.012 / -0.028±0.007 | -0.019±0.009 / -0.019±0.009 / -0.018±0.007 | -0.006±0.006 / -0.006±0.007 / -0.009±0.005 |
| contamination λ=0.10 | -0.059±0.002 / -0.059±0.002 / -0.050±0.001 | -0.057±0.003 / -0.055±0.002 / -0.050±0.001 | -0.027±0.002 / -0.026±0.002 / -0.032±0.002 |
| contamination λ=0.30 | -0.170±0.004 / -0.173±0.001 / -0.136±0.013 | -0.090±0.009 / -0.091±0.007 / -0.092±0.006 | -0.027±0.003 / -0.026±0.004 / -0.060±0.003 |
| contamination λ=0.60 | -0.283±0.043 / -0.284±0.034 / -0.250±0.018 | -0.098±0.009 / -0.093±0.009 / -0.147±0.014 | -0.023±0.003 / -0.023±0.003 / -0.106±0.016 |

Per-condition view (16 conditions): mean over conditions of the population posterior MSE S_c − J_c(T), mean over draws:

| distribution | n=8 VCS / logistic / kernel | n=32 VCS / logistic / kernel | n=128 VCS / logistic / kernel |
|---|---|---|---|
| clean λ=0.00 | 0.0090 / 0.0089 / 0.0015 | 0.0054 / 0.0054 / 0.0000 | 0.0019 / 0.0019 / 0.0001 |
| cond_mismatch λ=0.20 | 0.1197 / 0.1195 / 0.1106 | 0.0701 / 0.0711 / 0.0575 | 0.0189 / 0.0186 / 0.0288 |
| cond_mismatch λ=0.50 | 0.2314 / 0.2459 / 0.1960 | 0.0602 / 0.0586 / 0.0743 | 0.0173 / 0.0167 / 0.0439 |
| cond_mismatch λ=0.80 | 0.1816 / 0.1917 / 0.1780 | 0.0380 / 0.0391 / 0.0866 | 0.0098 / 0.0085 / 0.0645 |
| mode_missing λ=0.33 | 0.0374 / 0.0376 / 0.0250 | 0.0271 / 0.0270 / 0.0157 | 0.0089 / 0.0093 / 0.0080 |
| mode_missing λ=0.67 | 0.1238 / 0.1123 / 0.0527 | 0.0328 / 0.0352 / 0.0365 | 0.0104 / 0.0098 / 0.0084 |
| mode_missing λ=1.00 | 0.0728 / 0.0763 / 0.0617 | 0.0125 / 0.0163 / 0.0139 | 0.0043 / 0.0038 / 0.0078 |
| quality λ=0.20 | 0.0790 / 0.0813 / 0.0164 | 0.0253 / 0.0255 / 0.0074 | 0.0043 / 0.0039 / 0.0029 |
| quality λ=0.50 | 0.0668 / 0.0735 / 0.0211 | 0.0213 / 0.0228 / 0.0115 | 0.0045 / 0.0041 / 0.0045 |
| quality λ=0.80 | 0.0771 / 0.0580 / 0.0288 | 0.0163 / 0.0155 / 0.0149 | 0.0043 / 0.0040 / 0.0068 |
| contamination λ=0.10 | 0.0576 / 0.0576 / 0.0495 | 0.0563 / 0.0553 / 0.0497 | 0.0262 / 0.0257 / 0.0316 |
| contamination λ=0.30 | 0.1708 / 0.1725 / 0.1387 | 0.0910 / 0.0917 / 0.0923 | 0.0254 / 0.0247 / 0.0590 |
| contamination λ=0.60 | 0.2862 / 0.2878 / 0.2522 | 0.0962 / 0.0911 / 0.1459 | 0.0248 / 0.0239 / 0.1077 |

Detection statistics (not S estimates; mean over draws; clean row = null level):

| distribution | n=8 grouped cond. MMD² / uncond. MMD² | n=32 grouped cond. MMD² / uncond. MMD² | n=128 grouped cond. MMD² / uncond. MMD² |
|---|---|---|---|
| clean λ=0.00 | -0.0019 / -0.0016 | -0.0015 / -0.0006 | -0.0002 / -0.0002 |
| cond_mismatch λ=0.20 | 0.0206 / -0.0018 | 0.0169 / -0.0002 | 0.0204 / -0.0002 |
| cond_mismatch λ=0.50 | 0.1211 / -0.0011 | 0.1140 / -0.0003 | 0.1344 / -0.0003 |
| cond_mismatch λ=0.80 | 0.2920 / -0.0015 | 0.3115 / -0.0004 | 0.3284 / 0.0001 |
| mode_missing λ=0.33 | 0.0158 / -0.0010 | 0.0111 / 0.0001 | 0.0123 / 0.0003 |
| mode_missing λ=0.67 | 0.0347 / -0.0000 | 0.0387 / 0.0016 | 0.0495 / 0.0031 |
| mode_missing λ=1.00 | 0.1106 / 0.0039 | 0.1076 / 0.0058 | 0.1129 / 0.0062 |
| quality λ=0.20 | 0.0042 / 0.0015 | 0.0034 / 0.0021 | 0.0033 / 0.0017 |
| quality λ=0.50 | 0.0207 / 0.0163 | 0.0197 / 0.0133 | 0.0223 / 0.0154 |
| quality λ=0.80 | 0.0522 / 0.0346 | 0.0509 / 0.0368 | 0.0586 / 0.0411 |
| contamination λ=0.10 | -0.0014 / -0.0014 | 0.0022 / -0.0002 | 0.0025 / 0.0003 |
| contamination λ=0.30 | 0.0145 / -0.0011 | 0.0196 / 0.0016 | 0.0214 / 0.0017 |
| contamination λ=0.60 | 0.0828 / 0.0050 | 0.0808 / 0.0056 | 0.0881 / 0.0059 |

Cells where the learned critic is worse than the null critic T ≡ 0 (posterior MSE > S), out of 12 shifted distributions × draws; and mean relative posterior MSE (S − J)/S over the shifted distributions:

| n | VCS | logistic | kernel |
|---|---|---|---|
| 8 | 17/36; 0.82 | 16/36; 0.82 | 4/36; 0.53 |
| 32 | 4/36; 0.40 | 4/36; 0.40 | 1/36; 0.33 |
| 128 | 0/36; 0.13 | 0/36; 0.13 | 0/36; 0.19 |
