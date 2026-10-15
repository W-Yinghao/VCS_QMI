# V81_05 — input / gradient checks and checkpoint-level inference (V8.1 fork 3, 2026-10-15)

Facts only (FULL_PLAN_CN §04 D05 checkpoint level, D06; §03 §3–5; §07 §3).  Every number below is read from `results/{checkpoint_inference,input_checks,gradient_checks}.json`, which `scripts/v81/v81_fork3_post.py` writes from the GPU summaries under `/projects/EEG-foundation-model/yinghao/outputs_v81/fork3`.  No V8 artefact was modified; no GenAI-Rank / GenAI-Bench / TIFA160 label or prediction was read; the only labels used are the FIT-AUDIT and CAL_SELECT development labels (package roles allowed for labelled diagnostics).  Seed policy: per seed, then the metric mean (no prediction ensemble).

## 0. Environment, standard-batch timing, inference budget

GPU jobs: 1042190 `v81_fork3_smoke` on node52 (NVIDIA L40S), 2026-10-15T19:16:34Z → 2026-10-15T19:20:06Z, exit 0, 0.06 GPU-h; 1042191 `v81_fork3_gpu` on node52 (NVIDIA L40S), 2026-10-15T19:20:07Z → 2026-10-15T19:38:20Z, exit 0, 0.30 GPU-h — total 0.36 GPU-h of the 6 GPU-h cap.

GPU `NVIDIA L40S` on node52, torch 2.9.0+cu128, CUDA 12.8, numpy 1.26.4, python 3.10.22; job start 2026-10-15T19:20:16Z.  Standard load: batch 32, full-frame resize 364 (BLIP eval processor), FIT-AUDIT prompts (tokens mean 15.1, max 77), 736 images loaded; warm-up 3 batches, then 20 timed batches (CUDA-synchronised).

| path | s / batch (mean ± sd) | ms / image | images / s | peak GPU mem (GB) |
|---|---|---|---|---|
| ViT + Q-former + head (v8_train.forward_logits) | 0.652 ± 0.001 | 20.4 | 49.1 | 6.4 |
| EVA ViT-g only | 0.648 ± 0.018 | 20.2 | 49.4 | 6.4 |
| ln_vision + Q-former + head on the cached ViT output | 0.019 ± 0.000 | 0.6 | 1698.6 | 6.3 |


CPU decode + preprocess, single process: 36.6 ms / image.  Derived full-model budget on this GPU: FIT-AUDIT 28 s, CAL_SELECT 34 s, a 15-candidate pool 0.31 s, a 7-candidate pool 0.14 s; one checkpoint on both roles 62 s through the full model vs 2 s through the cached-ViT path used below.

## 1. Checkpoint inventory (what exists per unit)

Only `best.pt` (selected epoch) and the final `resume.pt` (epoch 4, with optimiser state) exist per unit; no per-epoch weights (§03 §3: other epochs are `not_saved`).  `best == resume` means the selected epoch is the last one, in which case the two files hold the same trainable weights (max |Δparam| shown).

| unit | best epoch | resume epoch | log selected | resume.pt best field (epoch / MSE) | best == resume | max abs Δparam | CAL_SELECT MSE re-inferred − log (best / resume) | best sha256 | resume sha256 |
|---|---|---|---|---|---|---|---|---|---|
| V8-FIT-U-BCE-s0 | 2 | 4 | 2 | e2 / 0.3978 | no | — | +0.00002 / +0.00003 | 2dadb0e58a59 | 6a2454df5396 |
| V8-FIT-U-BCE-s1 | 1 | 4 | 1 | e1 / 0.4075 | no | — | -0.00002 / -0.00001 | 02da75350eb0 | 227d5257e1ac |
| V8-FIT-U-BCE-s2 | 2 | 4 | 2 | e2 / 0.4044 | no | — | +0.00003 / -0.00002 | 39e34d1c454f | 04d1fc1296aa |
| V8-FIT-U-CE-s0 | 2 | 4 | 2 | e2 / 0.3942 | no | — | -0.00001 / +0.00001 | 00a30ecac300 | 73f8675b9348 |
| V8-FIT-U-CE-s1 | 1 | 4 | 1 | e1 / 0.4010 | no | — | -0.00003 / -0.00003 | c895d406097d | 783903dbd349 |
| V8-FIT-U-CE-s2 | 0 | 4 | 0 | e0 / 0.4031 | no | — | +0.00007 / -0.00001 | 69cfe9a74626 | 6be790a74832 |
| V8-FIT-U-CS-s0 | 2 | 4 | 2 | e2 / 0.4141 | no | — | +0.00003 / +0.00001 | 12f5cde73e05 | bd46343d299d |
| V8-FIT-U-CS-s1 | 0 | 4 | 0 | e0 / 0.4059 | no | — | -0.00000 / +0.00002 | bbba82d1b7c4 | e21b7416bce7 |
| V8-FIT-U-CS-s2 | 2 | 4 | 2 | e2 / 0.3966 | no | — | +0.00005 / +0.00007 | c7dd5cde988a | 4c507973ad39 |
| V8-FIT-U-L1-s0 | 3 | 4 | 3 | e3 / 0.4004 | no | — | -0.00001 / +0.00002 | 55cda30fcf03 | c34380f0a0c7 |
| V8-FIT-U-L1-s1 | 0 | 4 | 0 | e0 / 0.4011 | no | — | +0.00005 / +0.00008 | 0e1eb76c6cbb | 446ca6fcc36f |
| V8-FIT-U-L1-s2 | 3 | 4 | 3 | e3 / 0.4012 | no | — | +0.00002 / +0.00003 | 763e1843ec1e | bdbf7ce4cdeb |
| V8-FIT-U-LIST-s0 | 2 | 4 | 2 | e2 / 0.4211 | no | — | +0.00004 / +0.00000 | 716b2fb2e3d0 | d71915859378 |
| V8-FIT-U-LIST-s1 | 1 | 4 | 1 | e1 / 0.4226 | no | — | -0.00002 / -0.00001 | 7721c729f889 | 50e8650742f6 |
| V8-FIT-U-LIST-s2 | 2 | 4 | 2 | e2 / 0.4311 | no | — | +0.00004 / +0.00008 | 329b4e66aebe | b3e58a1d8097 |
| V8-FIT-U-MSE-s0 | 2 | 4 | 2 | e2 / 0.4028 | no | — | +0.00000 / -0.00002 | 3f47e05d2efd | e0a00f26cfab |
| V8-FIT-U-MSE-s1 | 0 | 4 | 0 | e0 / 0.4021 | no | — | +0.00006 / -0.00001 | 14d3f3b7b1c5 | 6e5d3ddc549f |
| V8-FIT-U-MSE-s2 | 2 | 4 | 2 | e2 / 0.4164 | no | — | -0.00005 / -0.00011 | d5f0d87a420d | b4d28a14c186 |
| V8-FIT-U-RANK-s0 | 0 | 4 | 0 | e0 / 0.5178 | no | — | +0.00013 / -0.00004 | 366e433adf5d | 0f3249355318 |
| V8-FIT-U-RANK-s1 | 0 | 4 | 0 | e0 / 0.5787 | no | — | -0.00002 / -0.00004 | 80716aea2ed2 | 5ddd242b4450 |
| V8-FIT-U-RANK-s2 | 0 | 4 | 0 | e0 / 0.6038 | no | — | +0.00012 / -0.00014 | 28a0808f29f7 | ed134cdf1395 |
| V8-FIT-U-RPS-s0 | 2 | 4 | 2 | e2 / 0.3971 | no | — | -0.00004 / -0.00001 | 228226d2819e | 06d8ba00b3fc |
| V8-FIT-U-RPS-s1 | 0 | 4 | 0 | e0 / 0.3987 | no | — | -0.00005 / +0.00003 | 8e9a3c6be41e | 8f5c2875eb4b |
| V8-FIT-U-RPS-s2 | 4 | 4 | 4 | e4 / 0.4092 | yes | 0 | +0.00003 / +0.00003 | 7406db10f500 | 7dae3f180b4a |


Re-inferred CAL_SELECT metrics vs the training-time log of the same epoch over the 48 checkpoints: max abs Δ mean-rating MSE 0.00014, rps_micro 0.00001, ce_micro 0.00009, within_prompt_pairwise_acc_macro 0.00077.


## 2. Checkpoint-level inference (eval mode, FIT-AUDIT and CAL_SELECT)

Rows in the package schema per (unit, checkpoint, role) under `outputs_v81/fork3/infer/rows/` (paths + sha256 in `checkpoint_inference.csv`, every file re-validated with the package reference validator in labelled mode).  RECONSTRUCTED_INIT = V8's `build(seed)` with no training (BLIP-2 COCO weights, seeded head, FIT log-histogram bias; shared by the eight arms at a seed; not claimed bit-identical to the formal start).  The forward path caches the frozen EVA ViT-g output once per image; its agreement with the frozen CAL_SELECT export of each unit's `best.pt` is the `vs export` column (max |Δμ| over the 1,653 images).

Agreement with the frozen exports (best.pt, CAL_SELECT, 24 units, each exported on its training GPU; re-inference on NVIDIA L40S): max abs Δμ 0.0418 (V8-FIT-U-RANK-s0, exported on NVIDIA RTX PRO 6000 Blackwell Server Edition); per-unit mean abs Δμ 0.00040–0.00115.

### 2.1 FIT_AUDIT — seed means (per-seed values in `checkpoint_inference.csv`)

| method | ckpt | epochs (s0/s1/s2) | mean-rating MSE (prompt macro) | pairwise acc | Spearman pooled | Spearman prompt macro | RPS | CE | within-prompt μ sd | μ sd | vs export max abs Δμ (seed mean of unit max) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RECONSTRUCTED_INIT (all arms) | init | —/—/— | 1.0020 | 0.464 | -0.071 | -0.092 | 0.1622 | 1.5272 | 0.039 | 0.080 | — |
| U-CS | best | 2/0/2 | 0.2118 | 0.792 | 0.883 | 0.647 | 0.0958 | 1.0709 | 0.418 | 0.855 | — |
| U-CS | resume | 4/4/4 | 0.1433 | 0.829 | 0.922 | 0.711 | 0.0877 | 0.9966 | 0.498 | 0.943 | — |
| U-BCE | best | 2/1/2 | 0.1832 | 0.809 | 0.900 | 0.680 | 0.0924 | 1.0389 | 0.446 | 0.892 | — |
| U-BCE | resume | 4/4/4 | 0.1330 | 0.834 | 0.927 | 0.725 | 0.0868 | 0.9844 | 0.513 | 0.952 | — |
| U-RPS | best | 2/0/4 | 0.2071 | 0.783 | 0.883 | 0.631 | 0.0949 | 1.0753 | 0.440 | 0.862 | — |
| U-RPS | resume | 4/4/4 | 0.1580 | 0.812 | 0.912 | 0.684 | 0.0894 | 1.0291 | 0.493 | 0.936 | — |
| U-CE | best | 2/1/0 | 0.2224 | 0.779 | 0.875 | 0.624 | 0.0963 | 1.0790 | 0.437 | 0.877 | — |
| U-CE | resume | 4/4/4 | 0.1558 | 0.815 | 0.916 | 0.690 | 0.0884 | 1.0061 | 0.500 | 0.944 | — |
| U-MSE | best | 2/0/2 | 0.2141 | 0.779 | 0.879 | 0.625 | 0.1055 | 1.2563 | 0.440 | 0.867 | — |
| U-MSE | resume | 4/4/4 | 0.1591 | 0.810 | 0.912 | 0.681 | 0.1028 | 1.2553 | 0.505 | 0.950 | — |
| U-L1 | best | 3/0/3 | 0.2279 | 0.801 | 0.870 | 0.662 | 0.1029 | 1.1899 | 0.457 | 0.936 | — |
| U-L1 | resume | 4/4/4 | 0.1564 | 0.830 | 0.915 | 0.714 | 0.0946 | 1.1272 | 0.496 | 0.958 | — |
| U-RANK | best | 0/0/0 | 0.4036 | 0.783 | 0.796 | 0.636 | 0.1374 | 1.5804 | 0.684 | 0.971 | — |
| U-RANK | resume | 4/4/4 | 0.5323 | 0.882 | 0.848 | 0.812 | 0.1698 | 2.1672 | 1.046 | 1.288 | — |
| U-LIST | best | 2/1/2 | 0.2000 | 0.821 | 0.889 | 0.703 | 0.1130 | 1.3830 | 0.484 | 0.875 | — |
| U-LIST | resume | 4/4/4 | 0.1600 | 0.849 | 0.913 | 0.753 | 0.1153 | 1.4473 | 0.542 | 0.944 | — |


### 2.2 CAL_SELECT — seed means (per-seed values in `checkpoint_inference.csv`)

| method | ckpt | epochs (s0/s1/s2) | mean-rating MSE (prompt macro) | pairwise acc | Spearman pooled | Spearman prompt macro | RPS | CE | within-prompt μ sd | μ sd | vs export max abs Δμ (seed mean of unit max) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RECONSTRUCTED_INIT (all arms) | init | —/—/— | 0.9933 | 0.468 | -0.159 | -0.077 | 0.1647 | 1.5339 | 0.039 | 0.085 | — |
| U-CS | best | 2/0/2 | 0.4055 | 0.725 | 0.757 | 0.510 | 0.1163 | 1.2266 | 0.381 | 0.736 | 0.0066 |
| U-CS | resume | 4/4/4 | 0.4219 | 0.719 | 0.751 | 0.498 | 0.1183 | 1.2635 | 0.434 | 0.798 | — |
| U-BCE | best | 2/1/2 | 0.4033 | 0.724 | 0.759 | 0.508 | 0.1160 | 1.2249 | 0.398 | 0.756 | 0.0045 |
| U-BCE | resume | 4/4/4 | 0.4189 | 0.719 | 0.752 | 0.499 | 0.1184 | 1.2701 | 0.440 | 0.807 | — |
| U-RPS | best | 2/0/4 | 0.4016 | 0.717 | 0.764 | 0.498 | 0.1160 | 1.2276 | 0.395 | 0.753 | 0.0037 |
| U-RPS | resume | 4/4/4 | 0.4081 | 0.715 | 0.761 | 0.495 | 0.1170 | 1.2401 | 0.437 | 0.808 | — |
| U-CE | best | 2/1/0 | 0.3994 | 0.718 | 0.766 | 0.499 | 0.1151 | 1.2134 | 0.396 | 0.763 | 0.0041 |
| U-CE | resume | 4/4/4 | 0.4077 | 0.719 | 0.761 | 0.498 | 0.1169 | 1.2422 | 0.441 | 0.816 | — |
| U-MSE | best | 2/0/2 | 0.4071 | 0.718 | 0.764 | 0.502 | 0.1217 | 1.3280 | 0.400 | 0.772 | 0.0044 |
| U-MSE | resume | 4/4/4 | 0.4183 | 0.716 | 0.760 | 0.496 | 0.1238 | 1.3458 | 0.451 | 0.832 | — |
| U-L1 | best | 3/0/3 | 0.4009 | 0.723 | 0.764 | 0.508 | 0.1184 | 1.2826 | 0.412 | 0.793 | 0.0063 |
| U-L1 | resume | 4/4/4 | 0.4036 | 0.721 | 0.766 | 0.503 | 0.1185 | 1.2829 | 0.435 | 0.804 | — |
| U-RANK | best | 0/0/0 | 0.5669 | 0.720 | 0.718 | 0.502 | 0.1494 | 1.6108 | 0.623 | 0.936 | 0.024 |
| U-RANK | resume | 4/4/4 | 0.9148 | 0.713 | 0.654 | 0.488 | 0.1968 | 2.2076 | 0.871 | 1.214 | — |
| U-LIST | best | 2/1/2 | 0.4250 | 0.719 | 0.752 | 0.499 | 0.1303 | 1.4480 | 0.431 | 0.787 | 0.009 |
| U-LIST | resume | 4/4/4 | 0.4407 | 0.713 | 0.749 | 0.486 | 0.1363 | 1.5196 | 0.466 | 0.848 | — |


## 3. D06 input interventions on FIT-AUDIT (best.pt of the 24 units; RECONSTRUCTED_INIT per seed)

Conditions (same eval path, same preprocessing and tokenisation): **normal** pairs; **permuted** = every image moved to a different prompt by one fixed seeded derangement (`interventions/permutation.json`); **rotation** = the within-prompt candidate change derived from the normal predictions (each row vs the next lexical candidate of its own prompt; Δμ compared with the human-mean difference of the same two images; pairs with equal human means excluded); **grey** = one all-grey 364×364 image (RGB 128) paired with each of the 128 prompts.  Permuted predictions are never scored against the original image's human rating.  ρ(text) = Spearman of the permuted prediction with the normal prediction of the same row (same prompt, different image); ρ(image) = with the normal prediction of the image that was moved in (same image, different prompt).

### 3.1 Per unit

| method | seed | normal ρ pooled | normal pairwise | within-prompt μ sd | between-prompt μ sd | perm mean abs Δμ | perm p90 abs Δμ | perm frac abs Δμ < .05 | perm q L1 | ρ(text) | ρ(image) | rot ρ(Δμ, Δhuman) | rot sign agr. | grey μ | grey μ − prompt mean | grey ≥ prompt max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RECONSTRUCTED_INIT | 0 | -0.104 | 0.435 | 0.051 | 0.075 | 0.259 | 0.401 | 0.04 | 0.316 | 0.07 | 0.10 | -0.23 | 0.43 | 3.03 | 0.16 | 0.76 |
| RECONSTRUCTED_INIT | 1 | 0.023 | 0.478 | 0.034 | 0.065 | 0.144 | 0.256 | 0.17 | 0.322 | 0.08 | 0.14 | -0.08 | 0.48 | 2.83 | -0.02 | 0.15 |
| RECONSTRUCTED_INIT | 2 | -0.132 | 0.479 | 0.032 | 0.064 | 0.155 | 0.294 | 0.18 | 0.322 | 0.24 | 0.07 | -0.09 | 0.48 | 3.08 | 0.07 | 0.55 |
| U-CS | 0 | 0.905 | 0.808 | 0.426 | 0.705 | 1.170 | 2.384 | 0.02 | 1.072 | 0.20 | 0.18 | 0.77 | 0.79 | 1.51 | -1.35 | 0.00 |
| U-CS | 1 | 0.841 | 0.756 | 0.385 | 0.712 | 1.120 | 2.209 | 0.04 | 1.002 | 0.38 | 0.20 | 0.64 | 0.73 | 1.47 | -1.28 | 0.00 |
| U-CS | 2 | 0.905 | 0.811 | 0.444 | 0.746 | 1.479 | 2.876 | 0.02 | 1.318 | 0.11 | 0.06 | 0.76 | 0.78 | 1.52 | -1.35 | 0.02 |
| U-BCE | 0 | 0.908 | 0.815 | 0.444 | 0.710 | 1.232 | 2.461 | 0.02 | 1.118 | 0.19 | 0.19 | 0.78 | 0.80 | 1.36 | -1.49 | 0.00 |
| U-BCE | 1 | 0.884 | 0.795 | 0.440 | 0.777 | 1.279 | 2.644 | 0.03 | 1.141 | 0.24 | 0.24 | 0.73 | 0.77 | 1.34 | -1.46 | 0.00 |
| U-BCE | 2 | 0.908 | 0.817 | 0.452 | 0.744 | 1.505 | 2.899 | 0.02 | 1.343 | 0.07 | 0.03 | 0.77 | 0.80 | 1.42 | -1.45 | 0.01 |
| U-RPS | 0 | 0.901 | 0.793 | 0.446 | 0.718 | 1.136 | 2.328 | 0.03 | 1.079 | 0.33 | 0.12 | 0.75 | 0.76 | 1.44 | -1.42 | 0.00 |
| U-RPS | 1 | 0.836 | 0.749 | 0.384 | 0.662 | 1.211 | 2.288 | 0.02 | 1.106 | 0.29 | 0.16 | 0.63 | 0.72 | 1.48 | -1.33 | 0.00 |
| U-RPS | 2 | 0.911 | 0.806 | 0.491 | 0.761 | 1.582 | 3.034 | 0.02 | 1.437 | 0.01 | 0.04 | 0.76 | 0.78 | 1.42 | -1.45 | 0.01 |
| U-CE | 0 | 0.904 | 0.800 | 0.450 | 0.721 | 1.170 | 2.392 | 0.03 | 1.100 | 0.27 | 0.12 | 0.75 | 0.78 | 1.35 | -1.50 | 0.00 |
| U-CE | 1 | 0.885 | 0.785 | 0.440 | 0.756 | 1.250 | 2.551 | 0.02 | 1.134 | 0.30 | 0.15 | 0.70 | 0.76 | 1.38 | -1.43 | 0.00 |
| U-CE | 2 | 0.837 | 0.753 | 0.422 | 0.728 | 1.460 | 2.742 | 0.03 | 1.256 | 0.14 | 0.09 | 0.62 | 0.72 | 1.55 | -1.33 | 0.00 |
| U-MSE | 0 | 0.900 | 0.794 | 0.451 | 0.729 | 1.052 | 2.235 | 0.03 | 0.803 | 0.39 | 0.15 | 0.74 | 0.77 | 1.54 | -1.32 | 0.00 |
| U-MSE | 1 | 0.839 | 0.750 | 0.397 | 0.685 | 1.136 | 2.236 | 0.02 | 0.921 | 0.31 | 0.19 | 0.63 | 0.73 | 1.52 | -1.29 | 0.00 |
| U-MSE | 2 | 0.898 | 0.794 | 0.471 | 0.765 | 1.546 | 2.980 | 0.02 | 1.246 | 0.16 | 0.11 | 0.73 | 0.76 | 1.58 | -1.31 | 0.01 |
| U-L1 | 0 | 0.913 | 0.829 | 0.486 | 0.774 | 1.198 | 2.454 | 0.02 | 0.992 | 0.37 | 0.15 | 0.78 | 0.80 | 1.35 | -1.51 | 0.00 |
| U-L1 | 1 | 0.786 | 0.751 | 0.396 | 0.783 | 1.263 | 2.542 | 0.04 | 1.095 | 0.35 | 0.21 | 0.62 | 0.72 | 1.42 | -1.34 | 0.01 |
| U-L1 | 2 | 0.912 | 0.822 | 0.491 | 0.786 | 1.582 | 3.018 | 0.02 | 1.365 | 0.09 | 0.06 | 0.78 | 0.80 | 1.38 | -1.48 | 0.00 |
| U-RANK | 0 | 0.791 | 0.787 | 0.635 | 0.639 | 1.294 | 2.484 | 0.03 | 0.877 | 0.15 | 0.25 | 0.68 | 0.77 | 1.34 | -1.56 | 0.00 |
| U-RANK | 1 | 0.791 | 0.779 | 0.722 | 0.701 | 1.568 | 3.012 | 0.03 | 1.096 | 0.06 | 0.21 | 0.68 | 0.76 | 1.20 | -1.68 | 0.00 |
| U-RANK | 2 | 0.806 | 0.784 | 0.695 | 0.719 | 1.783 | 3.221 | 0.02 | 1.287 | -0.05 | 0.12 | 0.67 | 0.75 | 1.32 | -1.68 | 0.00 |
| U-LIST | 0 | 0.900 | 0.830 | 0.491 | 0.693 | 1.032 | 2.203 | 0.03 | 0.720 | 0.33 | 0.17 | 0.80 | 0.82 | 1.60 | -1.23 | 0.00 |
| U-LIST | 1 | 0.871 | 0.805 | 0.467 | 0.694 | 1.255 | 2.468 | 0.02 | 0.962 | 0.25 | 0.17 | 0.74 | 0.78 | 1.58 | -1.22 | 0.00 |
| U-LIST | 2 | 0.897 | 0.827 | 0.496 | 0.740 | 1.425 | 2.807 | 0.02 | 1.054 | 0.20 | 0.17 | 0.78 | 0.80 | 1.64 | -1.25 | 0.02 |


### 3.2 Seed means per arm

| method | n seeds | normal ρ pooled | normal pairwise | within-prompt μ sd | between-prompt μ sd | perm mean abs Δμ | perm p90 abs Δμ | perm frac abs Δμ < .05 | perm q L1 | ρ(text) | ρ(image) | rot ρ(Δμ, Δhuman) | rot sign agr. | grey μ | grey μ − prompt mean | grey ≥ prompt max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RECONSTRUCTED_INIT | 3 | -0.071 | 0.464 | 0.039 | 0.068 | 0.186 | 0.317 | 0.13 | 0.320 | 0.13 | 0.10 | -0.132 | 0.47 | 2.98 | 0.07 | 0.49 |
| U-CS | 3 | 0.883 | 0.792 | 0.418 | 0.721 | 1.256 | 2.490 | 0.03 | 1.131 | 0.23 | 0.14 | 0.721 | 0.77 | 1.50 | -1.32 | 0.01 |
| U-BCE | 3 | 0.900 | 0.809 | 0.446 | 0.744 | 1.339 | 2.668 | 0.02 | 1.200 | 0.17 | 0.15 | 0.758 | 0.79 | 1.37 | -1.47 | 0.00 |
| U-RPS | 3 | 0.883 | 0.783 | 0.440 | 0.714 | 1.310 | 2.550 | 0.02 | 1.207 | 0.21 | 0.11 | 0.710 | 0.75 | 1.45 | -1.40 | 0.00 |
| U-CE | 3 | 0.875 | 0.779 | 0.437 | 0.735 | 1.293 | 2.562 | 0.03 | 1.163 | 0.24 | 0.12 | 0.690 | 0.75 | 1.42 | -1.42 | 0.00 |
| U-MSE | 3 | 0.879 | 0.779 | 0.440 | 0.726 | 1.245 | 2.484 | 0.03 | 0.990 | 0.29 | 0.15 | 0.701 | 0.75 | 1.55 | -1.31 | 0.00 |
| U-L1 | 3 | 0.870 | 0.801 | 0.457 | 0.781 | 1.348 | 2.671 | 0.03 | 1.151 | 0.27 | 0.14 | 0.728 | 0.78 | 1.38 | -1.45 | 0.00 |
| U-RANK | 3 | 0.796 | 0.783 | 0.684 | 0.686 | 1.548 | 2.906 | 0.03 | 1.087 | 0.05 | 0.19 | 0.675 | 0.76 | 1.29 | -1.64 | 0.00 |
| U-LIST | 3 | 0.889 | 0.821 | 0.484 | 0.709 | 1.237 | 2.493 | 0.02 | 0.912 | 0.26 | 0.17 | 0.775 | 0.80 | 1.61 | -1.23 | 0.01 |


### 3.3 Gradient response (eval-mode autograd, first 64 FIT-AUDIT rows; ‖∂μ/∂pixels‖₂ through the frozen ViT, ‖∂μ/∂(ViT output)‖₂ at the Q-former input; normal vs permuted pairing)

| method | seed | px grad mean (normal) | px grad median | px grad mean (permuted) | ViT-out grad mean (normal) | ViT-out grad median | ViT-out grad mean (permuted) | CLS-token share | max-patch share | n |
|---|---|---|---|---|---|---|---|---|---|---|
| RECONSTRUCTED_INIT | 0 | 1.82 | 0.563 | 7.89 | 0.0152 | 0.0135 | 0.0234 | 0.26 | 0.36 | 64 |
| RECONSTRUCTED_INIT | 1 | 1.28 | 0.605 | 4.03 | 0.0139 | 0.0116 | 0.0165 | 0.25 | 0.38 | 64 |
| RECONSTRUCTED_INIT | 2 | 1.67 | 0.595 | 2.55 | 0.0122 | 0.0114 | 0.017 | 0.26 | 0.38 | 64 |
| U-CS | 0 | 3.29 | 1.25 | 5.09 | 0.0348 | 0.0264 | 0.0297 | 0.20 | 0.35 | 64 |
| U-CS | 1 | 2.2 | 1.12 | 1.52 | 0.0269 | 0.0241 | 0.0189 | 0.28 | 0.36 | 64 |
| U-CS | 2 | 2.25 | 1.33 | 1.12 | 0.0351 | 0.0338 | 0.0201 | 0.25 | 0.34 | 64 |
| U-BCE | 0 | 2.61 | 1.02 | 3.13 | 0.0355 | 0.0285 | 0.0326 | 0.21 | 0.35 | 64 |
| U-BCE | 1 | 2.48 | 0.991 | 2.25 | 0.0302 | 0.0253 | 0.0209 | 0.26 | 0.34 | 64 |
| U-BCE | 2 | 2.36 | 1.5 | 2.03 | 0.0343 | 0.0312 | 0.0207 | 0.24 | 0.33 | 64 |
| U-RPS | 0 | 4.72 | 1.37 | 4.18 | 0.0415 | 0.0318 | 0.0333 | 0.23 | 0.35 | 64 |
| U-RPS | 1 | 3.84 | 1.47 | 3.89 | 0.0344 | 0.0349 | 0.0222 | 0.25 | 0.33 | 64 |
| U-RPS | 2 | 4.21 | 1.89 | 2.59 | 0.0477 | 0.0369 | 0.0353 | 0.21 | 0.32 | 64 |
| U-CE | 0 | 5.56 | 1.56 | 7.22 | 0.0446 | 0.033 | 0.0388 | 0.22 | 0.35 | 64 |
| U-CE | 1 | 4.28 | 1.51 | 2.25 | 0.0383 | 0.0323 | 0.02 | 0.24 | 0.32 | 64 |
| U-CE | 2 | 3.84 | 1.64 | 1.7 | 0.0383 | 0.0335 | 0.0218 | 0.24 | 0.32 | 64 |
| U-MSE | 0 | 3.54 | 1.47 | 2.32 | 0.0412 | 0.0333 | 0.035 | 0.22 | 0.35 | 64 |
| U-MSE | 1 | 3.5 | 1.65 | 1.55 | 0.036 | 0.0351 | 0.0242 | 0.25 | 0.34 | 64 |
| U-MSE | 2 | 3.43 | 2.05 | 2.6 | 0.0411 | 0.0369 | 0.0258 | 0.22 | 0.32 | 64 |
| U-L1 | 0 | 2.67 | 1.04 | 2.76 | 0.0362 | 0.0299 | 0.0371 | 0.22 | 0.36 | 64 |
| U-L1 | 1 | 2.69 | 1.2 | 2.83 | 0.0324 | 0.0277 | 0.0193 | 0.29 | 0.35 | 64 |
| U-L1 | 2 | 3.18 | 1.59 | 1.98 | 0.0375 | 0.0309 | 0.0221 | 0.22 | 0.36 | 64 |
| U-RANK | 0 | 8.81 | 2.47 | 3.13 | 0.0641 | 0.0625 | 0.0409 | 0.25 | 0.35 | 64 |
| U-RANK | 1 | 11.3 | 3.54 | 3.35 | 0.0715 | 0.0652 | 0.0246 | 0.28 | 0.37 | 64 |
| U-RANK | 2 | 8 | 3.55 | 1.49 | 0.063 | 0.0612 | 0.0165 | 0.27 | 0.32 | 64 |
| U-LIST | 0 | 5.41 | 2.51 | 2.26 | 0.0573 | 0.0521 | 0.0405 | 0.22 | 0.34 | 64 |
| U-LIST | 1 | 5.97 | 2.31 | 1.57 | 0.0467 | 0.0412 | 0.0218 | 0.24 | 0.37 | 64 |
| U-LIST | 2 | 5.55 | 2.69 | 4.5 | 0.0548 | 0.0499 | 0.0249 | 0.23 | 0.34 | 64 |


## 4. Gradient checks on the frozen epoch-0 batches (U-RANK / U-LIST, U-CS as reference)

First 20 batches of `v8_core.batch_manifest(FIT, seed, epoch 0)` (hash checked against each unit's `log.json`), eval mode (dropout off), autocast fp16 forward as in training, fp32 losses from the unchanged `v8_core.batch_loss`, unscaled autograd (no GradScaler, no optimiser step).  Point term = the normalised mean-rating MSE (for U-RANK / U-LIST) or the CS loss; ranking term = the λ-weighted RankNet / ListNet part (total − point).  Norms are L2 over the group's parameters; cos = cosine between the point and ranking gradients; ‖g‖ and cos are means over the 20 batches; `rank/point` = the median over batches of the per-batch norm ratio.  μ saturation = fraction of μ within 0.05 of 1 or 5.

| arm | seed | ckpt | head ‖g_point‖ | head ‖g_rank‖ | head cos | head rank/point (median) | Q-former ‖g_point‖ | Q-former ‖g_rank‖ | Q-former cos | Q-former rank/point (median) | ln_vision ‖g_point‖ | query ‖g_point‖ | point loss | ranking loss | pairs / batch | subgroups / batch | μ saturated | μ sd in batch | non-finite batches | plan hash ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| U-RANK | 0 | RECONSTRUCTED_INIT | 0.523 | 0.739 | 0.26 | 1.40 | 2.67 | 3 | 0.39 | 1.08 | 0.062 | 0.168 | 0.0696 | 0.7015 | 23.3 | 2.0 | 0.000 | 0.079 | 0 | yes |
| U-RANK | 0 | best (e0) | 0.37 | 0.675 | 0.07 | 2.32 | 3.29 | 4.89 | 0.22 | 1.80 | 0.0934 | 0.184 | 0.0293 | 0.5722 | 23.3 | 2.0 | 0.000 | 0.723 | 0 | yes |
| U-RANK | 1 | RECONSTRUCTED_INIT | 0.378 | 0.609 | 0.31 | 1.91 | 2.82 | 4.3 | 0.38 | 1.66 | 0.0577 | 0.152 | 0.0639 | 0.6922 | 23.3 | 2.0 | 0.000 | 0.058 | 0 | yes |
| U-RANK | 1 | best (e0) | 0.32 | 0.678 | 0.18 | 2.78 | 2.28 | 4.63 | 0.29 | 2.45 | 0.0655 | 0.136 | 0.0336 | 0.5470 | 23.3 | 2.0 | 0.000 | 0.842 | 0 | yes |
| U-RANK | 2 | RECONSTRUCTED_INIT | 0.443 | 0.684 | 0.24 | 2.01 | 3.69 | 3.78 | 0.37 | 1.00 | 0.0766 | 0.185 | 0.0729 | 0.6938 | 26.3 | 2.0 | 0.000 | 0.057 | 0 | yes |
| U-RANK | 2 | best (e0) | 0.359 | 0.653 | 0.31 | 2.11 | 3.05 | 6.01 | 0.19 | 2.17 | 0.0651 | 0.15 | 0.0345 | 0.5886 | 26.3 | 2.0 | 0.000 | 0.869 | 0 | yes |
| U-LIST | 0 | RECONSTRUCTED_INIT | 0.523 | 0.683 | 0.26 | 1.09 | 2.67 | 3.03 | 0.29 | 1.23 | 0.062 | 0.168 | 0.0696 | 1.6924 | 23.3 | 2.0 | 0.000 | 0.079 | 0 | yes |
| U-LIST | 0 | best (e2) | 0.18 | 0.366 | 0.30 | 2.19 | 1.78 | 3.39 | 0.38 | 1.87 | 0.0468 | 0.128 | 0.0142 | 1.5722 | 23.3 | 2.0 | 0.000 | 0.680 | 0 | yes |
| U-LIST | 1 | RECONSTRUCTED_INIT | 0.378 | 0.558 | 0.27 | 2.14 | 2.82 | 4.22 | 0.43 | 1.38 | 0.0577 | 0.152 | 0.0639 | 1.6849 | 23.3 | 2.0 | 0.000 | 0.058 | 0 | yes |
| U-LIST | 1 | best (e1) | 0.189 | 0.415 | 0.44 | 2.71 | 2.12 | 3.59 | 0.40 | 1.96 | 0.0534 | 0.136 | 0.0183 | 1.5808 | 23.3 | 2.0 | 0.000 | 0.667 | 0 | yes |
| U-LIST | 2 | RECONSTRUCTED_INIT | 0.443 | 0.742 | 0.29 | 1.55 | 3.69 | 3.96 | 0.39 | 1.06 | 0.0766 | 0.185 | 0.0729 | 1.7407 | 26.3 | 2.0 | 0.000 | 0.057 | 0 | yes |
| U-LIST | 2 | best (e2) | 0.162 | 0.408 | 0.47 | 2.07 | 2.44 | 4.39 | 0.32 | 1.94 | 0.0516 | 0.151 | 0.0141 | 1.5814 | 26.3 | 2.0 | 0.000 | 0.794 | 0 | yes |
| U-CS | 0 | RECONSTRUCTED_INIT | 1.2 | — | — | — | 4.07 | — | — | — | 0.0972 | 0.244 | 0.2573 | — | — | — | 0.000 | 0.079 | 0 | yes |
| U-CS | 0 | best (e2) | 0.246 | — | — | — | 2.27 | — | — | — | 0.0633 | 0.183 | 0.1110 | — | — | — | 0.000 | 0.619 | 0 | yes |
| U-CS | 1 | RECONSTRUCTED_INIT | 1.05 | — | — | — | 3.87 | — | — | — | 0.0835 | 0.218 | 0.2458 | — | — | — | 0.000 | 0.058 | 0 | yes |
| U-CS | 1 | best (e0) | 0.389 | — | — | — | 2.77 | — | — | — | 0.0615 | 0.173 | 0.1455 | — | — | — | 0.000 | 0.674 | 0 | yes |
| U-CS | 2 | RECONSTRUCTED_INIT | 1.15 | — | — | — | 5.35 | — | — | — | 0.105 | 0.271 | 0.2691 | — | — | — | 0.000 | 0.057 | 0 | yes |
| U-CS | 2 | best (e2) | 0.213 | — | — | — | 3.34 | — | — | — | 0.0748 | 0.216 | 0.1195 | — | — | — | 0.000 | 0.732 | 0 | yes |

## 5. Parameter-level audit (checkpoints vs RECONSTRUCTED_INIT; CPU job, read-only)

`results/param_audit.{json,csv}` (`scripts/v81/v81_fork3_param_audit.py`).  For each seed V8's `build(seed)` is rebuilt on CPU; the trainable set and the AdamW grouping are rebuilt with V8's own rule and checked against the optimiser state's tensor counts and shapes.  rel = ‖θ − θ_init‖₂ / ‖θ_init‖₂ over the group.  AdamW steps = the per-tensor `step` counter in `resume.pt`; global iterations = the trainer's batch counter; their difference = optimiser steps skipped by the fp16 GradScaler.  visual_encoder.* is never saved in a V8 checkpoint; every V8 load rebuilds it from the public BLIP-2 COCO weights (requires_grad False).

Trainable set rebuilt: 338 tensors / 186,314,048 parameters (124 with weight decay, 214 without); ViT parameters 986,485,888, any requires_grad: False.  Trainable groups (tensor counts): head 2, ln_vision 2, qf.cls_head 5, qf.embeddings 4, query_tokens 1; per-layer Q-former groups 324 tensors.

Tied state-dict aliases (one storage, two names; counted once, under the parameter): `Qformer.cls.predictions.decoder.weight` = `Qformer.bert.embeddings.word_embeddings.weight`; `Qformer.cls.predictions.decoder.bias` = `Qformer.cls.predictions.bias` — alias equal to its parameter in 48/48 checkpoints.

| arm | seed | lr | best epoch | rel all trainable | rel head | rel ln_vision | rel query tokens | rel Q-former embeddings | rel self-attn | rel cross-attn | rel query FFN | rel text FFN | rel LM head | frozen tensors identical | never-stepped tensors (groups) | AdamW steps / iterations | skipped steps | final loss scale | resume vs best rel |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| U-CS | 0 | 1e-05 | 2 | 0.0107 | 0.0106 | 0.000641 | 0.0208 | 0.0056 | 0.0077 | 0.00793 | 0.0177 | 0.0166 | 0 | 18/18 | qf.cls_head 5 | 11893/11895 | 2 | 5.243e+05 | 0.00167 |
| U-CS | 1 | 1e-05 | 0 | 0.00748 | 0.00723 | 0.000446 | 0.0153 | 0.00348 | 0.00551 | 0.00524 | 0.0122 | 0.0119 | 0 | 18/18 | qf.cls_head 5 | 11892/11895 | 3 | 2.621e+05 | 0.00696 |
| U-CS | 2 | 1e-05 | 2 | 0.0106 | 0.0103 | 0.000625 | 0.0213 | 0.00552 | 0.00757 | 0.00783 | 0.0175 | 0.0164 | 0 | 18/18 | qf.cls_head 5 | 11893/11895 | 2 | 5.243e+05 | 0.00166 |
| U-BCE | 0 | 1e-05 | 2 | 0.0111 | 0.0117 | 0.000652 | 0.0217 | 0.00572 | 0.00793 | 0.00822 | 0.0185 | 0.0171 | 0 | 18/18 | qf.cls_head 5 | 11891/11895 | 4 | 1.311e+05 | 0.00169 |
| U-BCE | 1 | 1e-05 | 1 | 0.00989 | 0.01 | 0.000574 | 0.0194 | 0.00493 | 0.00711 | 0.00718 | 0.0164 | 0.0154 | 0 | 18/18 | qf.cls_head 5 | 11891/11895 | 4 | 1.311e+05 | 0.00406 |
| U-BCE | 2 | 1e-05 | 2 | 0.0109 | 0.0114 | 0.000641 | 0.0207 | 0.00566 | 0.00775 | 0.00793 | 0.0181 | 0.0168 | 0 | 18/18 | qf.cls_head 5 | 11892/11895 | 3 | 2.621e+05 | 0.00169 |
| U-RPS | 0 | 5e-06 | 2 | 0.00655 | 0.00789 | 0.000389 | 0.0127 | 0.00305 | 0.00472 | 0.00474 | 0.0111 | 0.01 | 0 | 18/18 | qf.cls_head 5 | 11894/11895 | 1 | 1.049e+06 | 0.000843 |
| U-RPS | 1 | 5e-06 | 0 | 0.00458 | 0.00559 | 0.000286 | 0.00874 | 0.00182 | 0.00335 | 0.00314 | 0.00778 | 0.00709 | 0 | 18/18 | qf.cls_head 5 | 11893/11895 | 2 | 5.243e+05 | 0.00384 |
| U-RPS | 2 | 5e-06 | 4 | 0.00667 | 0.00832 | 0.000399 | 0.0124 | 0.00318 | 0.00473 | 0.00483 | 0.0114 | 0.0101 | 0 | 18/18 | qf.cls_head 5 | 11893/11895 | 2 | 5.243e+05 | 0 |
| U-CE | 0 | 5e-06 | 2 | 0.0069 | 0.00888 | 0.000403 | 0.013 | 0.00312 | 0.00497 | 0.00498 | 0.0116 | 0.0105 | 0 | 18/18 | qf.cls_head 5 | 11891/11895 | 4 | 1.311e+05 | 0.000877 |
| U-CE | 1 | 5e-06 | 1 | 0.00607 | 0.00769 | 0.000365 | 0.0117 | 0.00266 | 0.00439 | 0.0043 | 0.0103 | 0.00932 | 0 | 18/18 | qf.cls_head 5 | 11890/11895 | 5 | 6.554e+04 | 0.00221 |
| U-CE | 2 | 5e-06 | 0 | 0.00468 | 0.00633 | 0.000303 | 0.00877 | 0.00185 | 0.00342 | 0.00319 | 0.00798 | 0.00723 | 0 | 18/18 | qf.cls_head 5 | 11890/11895 | 5 | 6.554e+04 | 0.00411 |
| U-MSE | 0 | 5e-06 | 2 | 0.0062 | 0.00555 | 0.000376 | 0.0121 | 0.00299 | 0.00444 | 0.00452 | 0.0104 | 0.00946 | 0 | 18/18 | qf.cls_head 5 | 11894/11895 | 1 | 1.049e+06 | 0.000823 |
| U-MSE | 1 | 5e-06 | 0 | 0.00437 | 0.00444 | 0.000279 | 0.00842 | 0.0018 | 0.00318 | 0.00302 | 0.0074 | 0.00679 | 0 | 18/18 | qf.cls_head 5 | 11894/11895 | 1 | 1.049e+06 | 0.00369 |
| U-MSE | 2 | 5e-06 | 2 | 0.00613 | 0.00574 | 0.000374 | 0.0117 | 0.00297 | 0.00436 | 0.00442 | 0.0104 | 0.00938 | 0 | 18/18 | qf.cls_head 5 | 11894/11895 | 1 | 1.049e+06 | 0.000807 |
| U-L1 | 0 | 1e-05 | 3 | 0.0111 | 0.00934 | 0.000667 | 0.0233 | 0.00582 | 0.00793 | 0.00825 | 0.0183 | 0.0172 | 0 | 18/18 | qf.cls_head 5 | 11893/11895 | 2 | 5.243e+05 | 0.000339 |
| U-L1 | 1 | 1e-05 | 0 | 0.00772 | 0.00698 | 0.00048 | 0.0161 | 0.00348 | 0.00566 | 0.00546 | 0.0127 | 0.0122 | 0 | 18/18 | qf.cls_head 5 | 11893/11895 | 2 | 5.243e+05 | 0.00695 |
| U-L1 | 2 | 1e-05 | 3 | 0.0111 | 0.00983 | 0.000656 | 0.0235 | 0.00577 | 0.00788 | 0.00817 | 0.0185 | 0.0171 | 0 | 18/18 | qf.cls_head 5 | 11893/11895 | 2 | 5.243e+05 | 0.000338 |
| U-RANK | 0 | 5e-06 | 0 | 0.00481 | 0.00582 | 0.000354 | 0.0095 | 0.00183 | 0.00345 | 0.00333 | 0.00837 | 0.00726 | 0 | 18/18 | qf.cls_head 5 | 11890/11895 | 5 | 6.554e+04 | 0.00466 |
| U-RANK | 1 | 5e-06 | 0 | 0.00476 | 0.0056 | 0.00034 | 0.00908 | 0.0018 | 0.00339 | 0.00328 | 0.0083 | 0.00723 | 0 | 18/18 | qf.cls_head 5 | 11891/11895 | 4 | 6.554e+04 | 0.00462 |
| U-RANK | 2 | 5e-06 | 0 | 0.00477 | 0.00597 | 0.00034 | 0.00959 | 0.00182 | 0.0034 | 0.00326 | 0.00839 | 0.0072 | 0 | 18/18 | qf.cls_head 5 | 11892/11895 | 3 | 2.621e+05 | 0.00465 |
| U-LIST | 0 | 5e-06 | 2 | 0.00679 | 0.00658 | 0.000452 | 0.0128 | 0.00304 | 0.00481 | 0.00488 | 0.0116 | 0.0103 | 0 | 18/18 | qf.cls_head 5 | 11891/11895 | 4 | 1.311e+05 | 0.000917 |
| U-LIST | 1 | 5e-06 | 1 | 0.00602 | 0.00614 | 0.000398 | 0.0114 | 0.00261 | 0.0043 | 0.00426 | 0.0102 | 0.00931 | 0 | 18/18 | qf.cls_head 5 | 11891/11895 | 4 | 6.554e+04 | 0.00238 |
| U-LIST | 2 | 5e-06 | 2 | 0.00671 | 0.0064 | 0.000437 | 0.0126 | 0.00303 | 0.00473 | 0.00479 | 0.0115 | 0.0102 | 0 | 18/18 | qf.cls_head 5 | 11890/11895 | 5 | 6.554e+04 | 0.000924 |


Cross-attention relative change per Q-former layer (best.pt vs RECONSTRUCTED_INIT; seed mean, per-seed values in the csv):

| arm | L00 cross | L02 cross | L04 cross | L06 cross | L08 cross | L10 cross | L00 self | L11 self |
|---|---|---|---|---|---|---|---|---|
| U-CS | 0.00734 | 0.00735 | 0.00878 | 0.00774 | 0.00717 | 0.00533 | 0.00782 | 0.00379 |
| U-BCE | 0.00793 | 0.00812 | 0.00967 | 0.00863 | 0.00804 | 0.00606 | 0.00844 | 0.00431 |
| U-RPS | 0.00426 | 0.00442 | 0.00533 | 0.00474 | 0.00429 | 0.00334 | 0.0046 | 0.00249 |
| U-CE | 0.00414 | 0.00431 | 0.00521 | 0.00462 | 0.00422 | 0.00333 | 0.00454 | 0.00251 |
| U-MSE | 0.00404 | 0.00418 | 0.00505 | 0.00447 | 0.00408 | 0.00306 | 0.00438 | 0.00215 |
| U-L1 | 0.0078 | 0.0077 | 0.00928 | 0.00807 | 0.00741 | 0.00538 | 0.00815 | 0.00374 |
| U-RANK | 0.00325 | 0.00339 | 0.00412 | 0.00359 | 0.00324 | 0.00277 | 0.00357 | 0.00198 |
| U-LIST | 0.00457 | 0.00479 | 0.00582 | 0.00527 | 0.00487 | 0.00361 | 0.00492 | 0.00242 |


Never-stepped trainable tensors (no AdamW state: no gradient path in `forward_logits`) — qf.cls_head: identical to the reconstruction in 24/24 units.


## 6. Not done / limits

- No per-epoch checkpoint exists for any unit (only `best.pt` and the final `resume.pt`), so the checkpoint-level selection analysis of D05 is limited to these two states per unit; no training was re-run to recover the trajectory (`not_saved`).
- RECONSTRUCTED_INIT is a reconstruction from the same recipe and seed, labelled as such; §5 reports where it can be checked against the checkpoints (frozen and never-stepped tensors); the trained tensors' starting values cannot be checked.
- The gradient checks run in eval mode without the GradScaler; the training step used train mode (Q-former dropout) and fp16 loss scaling.  They describe gradient magnitude and direction, not the realised update (§5 gives the realised parameter movement).
- The within-prompt "rotation" condition is derived from the normal predictions (the same prompt with another of its own candidates), not a separate forward pass.
- FIT-AUDIT is a training-set audit sample (every image was seen in every epoch); its metrics describe fit, not generalisation.
- No interpretation is written here; the phase report (V81_08) reads these facts together with D00–D05 and D07.
