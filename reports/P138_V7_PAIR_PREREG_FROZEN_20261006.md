# Pre-registration — P138: v7 V7-PAIR — sampled image shifts × all view pairs (K = 16) for VCS (A-P3) and matched JS-AP3, CIFAR-10 / CIFAR-100 — FROZEN 2026-10-06T00:59:28Z 

Source: owner 2026-10-05 "查看这个压缩包和那个md文档，把这批次的分析也开始做" → `VCS_SSL_Server_Plan_v7_CN.md` §5 (V7-PAIR) and
`task_manifest_v7.yaml` (`V7-PAIR-*`, stage C, "conditional_on_implementation_checks").  Not the G2 K8 / one-sided-gradient line: all positive pairs,
all view roles and full two-sided gradients are kept; only the negatives between different images are subsampled.

## 1. Question
With the encoder forward, the images, the augmentation, the scorer and full backpropagation unchanged, can a random subset of the negative pairs keep
the same expected objective and a representation of the same quality — and what does it actually save?

## 2. Objective (code: `src/vcs_ssl/objectives.py` P138 block; `pairing.pair_scope: sampled_image_shifts_all_view_pairs`, `pairing.k` = K)
Tokens z[a, i], view a ∈ {0..V−1}, image i ∈ {0..B−1}.  P = all ordered different-view pairs of the same image, N_P = B·V(V−1) (computed exactly).
For each of K distinct nonzero cyclic shifts d drawn uniformly without replacement each step from the dedicated, checkpointed pair generator
(`pairing.rng_seed_offset`; unused by the all_view_tokens parents, so data order and augmentation streams are identical to the parents), the B·V²
pairs (z[a, i], z[b, (i + d) mod B]) for all (a, b) including a = b.  L_K = L_P + (1/K) Σ_d H_d, H_d = mean_{B·V²} ℓ_Q; VCS ℓ_Q = T + T²/2 with
T = tanh(2s − 1); JS ℓ_Q = softplus(2f), f = 2s − 1, positives softplus(−2f).  Conditional on the batch, E[L_K] = L_all and E[∇L_K] = ∇L_all;
Var(L_K | z) = (1/K)(1 − K/D) s_H², D = B − 1.  **The same holds for matched JS — no VCS-only property is claimed.**  K = B − 1 reproduces the
existing `all_view_tokens` loss exactly.  Only the selected pairs are scored (no full matrix + mask); the trainer logs `critic_pair_evals` = scores
actually computed.  B = 256, V = 4, K = 16: negatives 1,044,480 → 65,536 (15.94×); encoder forwards unchanged.

## 3. Implementation checks (gate) — filled in from the jobs
- Tests `tests/test_vcs_ssl_p138.py`: K = B−1 equals all_view_tokens (loss, J / risk statistics and gradients, VCS and JS, float64, 1e-12);
  exact-enumeration unbiasedness of loss and gradient (all K-subsets); the finite-population variance formula; distinct / deterministic /
  resumable shift sampler; only-selected-pairs count; config acceptance and refusals (negative detach, all_view_chunk, learned affine, k > B−1,
  V = 2, JS without the all-view markers); dispatcher pair counts; CPU trainer smoke.  Regression suites (P114, v2 core, P126, P133, ssl_core);
  config-hash invariance of every existing config against a HEAD worktree (`scripts/p138_config_hashes.py`).
- Stage 1 (read-only, `scripts/p138_pair_readonly.py`): see §5.

## 4. Cells (stage `P138_v7_pair_k16`; `configs/make_p138_configs.py`, `configs/P138_SHA256.json`, `slurm/p138_lines.txt`)
| run_id | dataset | loss | parent (K = all, reused control) |
|---|---|---|---|
| P138_PAIR_K16_vcs_c10_seed0 | CIFAR-10 | VCS A-P3 | P107_AP3_views4_800ep_seed0 (89.06 / 87.30) |
| P138_PAIR_K16_js_c10_seed0 | CIFAR-10 | matched JS | P114_JSAP3_views4_800ep_seed0 (88.72 / 87.50) |
| P138_PAIR_K16_vcs_c100_seed0 | CIFAR-100 | VCS A-P3 | P107_AP3_c100_views4_800ep_seed0 (60.20 / 55.92) |
| P138_PAIR_K16_js_c100_seed0 | CIFAR-100 | matched JS | P120_JS_AP3_c100_views4_800ep_seed0 (58.76 / 54.80) |
Each config differs from its parent only in pairing.pair_scope, pairing.k (8 → 16) and the removed, unused pairing.all_view_chunk (verified by
diff); (a, κ) = (2, 0.5), lr 1e-3, 800 epochs, ResNet-18, V = 4, B = 256, standard augmentation.

## 5. Pre-stated reading
**Stage 1 (read-only, descriptive):** per model × epoch (100, 800; missing checkpoints listed, never substituted): bias and variance of L_K vs L_all
(64 draws, K ∈ {1, 4, 16, 64, B−1}) next to the exact finite-population variance; dL/dz errors; encoder + projector gradient MSE / cosine /
relative error vs g_all at K = 4, 16 (16 draws; relative-error floor 1e-8 fixed in advance; ||g_all|| and absolute errors reported); the spread of
L_all / g_all across the 4 base batches (image + augmentation noise, which the reference does not remove); runtime.  g_all is the full-pair gradient
of a finite batch, not a population oracle gradient.  Stage 1 does not choose K and does not gate stage 2 beyond implementation correctness.

**Stage 2:** final frozen-h linear (primary), kNN, wall time per GPU type, peak memory, score elements per step, optimizer steps.
- **Development retention** (per loss × dataset): linear drop vs the parent ≤ 0.30 (CIFAR-10) / ≤ 0.50 (CIFAR-100) **and** a measured improvement in
  pair-scoring compute or peak memory → seeds 1–2 of that cell and of its K = all parent if missing (addendum, submitted as soon as triggered).
- **Final claim** ("accuracy retained") only from a paired non-inferiority interval over seeds (margin 0.30 / 0.50), never from "not significant".
- VCS and JS are read side by side; a VCS-specific statement needs a VCS × JS interaction (Δ_VCS(K16 − all) − Δ_JS(K16 − all)), not VCS alone.
- If only the pair-scoring cost improves and the end-to-end step time barely moves, that is reported as such (no "efficient" claim).
- InfoNCE token subsampling is not run here (its log-normaliser makes a K-subsampled estimate biased; not the same objective).
- No early stopping on J / rank / kNN; official test closed.

## 6. Cost
4 × 800 epochs.  Expected per-step cost close to the parent (the encoder forward / backward dominates; the pair-scoring saving is 15.94× on a
small part of the step) — measured in the GPU smoke (§7).

## 7. Gate and smoke results
- **CPU gate 1023084 (exit 0):** `tests/test_vcs_ssl_p138.py` 16 / 16 (K = B−1 = all_view_tokens to 1e-12 for loss, J / risk statistics and
  gradients, VCS and JS; exact-enumeration unbiasedness of loss and gradient; finite-population variance formula; sampler; refusals; dispatcher;
  CPU trainer smoke).  Regression (P114, v2 core, P126, P133, ssl_core) 107 / 107.  Config-hash invariance: all 452 configs that load under HEAD keep
  byte-identical resolved hashes in the working tree; only the 4 new P138 configs are new (the one non-loading file, `next_stage_plan.yaml`, is
  not a trainer config).  `reports/P138_GATE_1023084/`.
- **GPU job 1023092 (L40S node, exit 0), training smoke 60 steps, same GPU:** step time A-P3 parent 0.220 s, P138 VCS C10 0.218 s, JS parent
  0.218 s, P138 JS C10 0.220 s, P138 VCS C100 0.220 s; peak allocated memory 5137 MB in every config; scores per step 1,047,552 (parents) vs
  68,608 (P138: 3,072 positives + 65,536 negatives).  **The 15.3× cut in scored pairs does not change the end-to-end step time or peak memory at
  CIFAR scale** (ResNet-18 forward / backward dominates) — reported as such (§5).  `reports/P138_GPU_SMOKE_1023092/`.
- **Stage 1 (read-only; `reports/P138_readonly/`, table `P138_readonly_table.txt`):** P114 C10 JS seed 0 has no epoch-100 checkpoint (listed
  missing).  Over 7 model × epoch cells, 4 base batches each: L_K unbiased (|bias| / se 0.3–1.3) and its 64-draw variance matches the exact
  finite-population formula; K = B−1 reproduces L_all exactly.  Encoder + projector gradient vs g_all: K = 16 cos 0.980–0.993, relative error
  0.12–0.20 (VCS 0.12–0.16, JS 0.18–0.20); K = 4 cos 0.92–0.97, relative error 0.24–0.42.  For scale: the full-pair gradients of different base
  batches differ by relative deviation 0.66–1.87 from their mean (pairwise cos −0.04 to 0.65) — at K = 16 the pair-selection noise is far below
  the image / augmentation sampling noise that every SSL step already has.  Shared-graph protocol vs isolated train-mode forward: rel. diff ≤ 9e-7.

## 8. Open decisions for the main session
1. `pairing.k` is reused as "number of distinct image shifts" (its existing validator already reads "K distinct nonzero cyclic shifts"); the
   alternative is a new key.  2. `pairing.all_view_chunk` is removed in the P138 configs (refused by the policy because it would be unused).
3. JS keeps the `js_all_view_tokens` marker (now meaning "all-view token family", P114 or P138 scope) so its config differs from the parent only in
   the pairing fields.  4. P114 C10 JS seed 0 has no epoch-100 checkpoint (intermediates removed in the v4 cleanup) — stage 1 lists it as missing.

## Decisions at the freeze (main session)
Owner 2026-10-05 v7 plan, §5 ("实现检查通过即可进入完整训练").  Gate 1023084: P138 tests 16 / 16 (K = B−1 ≡ all_view_tokens to 1e-12 for VCS / JS, loss
and gradients; exact unbiasedness; variance formula; sampler; refusals), regression 107 / 107, 452 existing config hashes unchanged.  Stage-1 read-only
(1023092): loss bias consistent with 0, variance = formula; encoder-gradient cosine 0.980–0.993 / relative error 0.12–0.20 at K = 16; P114 C10 JS
epoch_100 missing (listed, not substituted).  GPU smoke: 15.3× fewer computed scores per step, **no wall-time or memory change at CIFAR scale**.
1. `pairing.k` is reused as "number of distinct image shifts" (its validator already reads "K distinct nonzero cyclic shifts"); `all_view_chunk`
   removed and refused for this scope; JS keeps `js_all_view_tokens` (now: the all-view token family, P114 or P138 scope).
2. Retention rule as in the plan: linear drop ≤ 0.30 (C10) / ≤ 0.50 (C100) **and** a measured reduction of actually computed pair scores (met by
   construction: 1,047,552 → 68,608 per step) → seeds 1–2; the report states measured wall time and memory per step next to it, and that at CIFAR
   scale they are unchanged — no efficiency claim beyond the score count.  Final claim only by a paired non-inferiority interval.
3. Submit the 4 seed-0 units now (normal QOS, RTX6000PRO / H100 / L40S, node51 + node52 + node60 excluded).
