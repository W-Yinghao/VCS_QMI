# P139 — v7 V7-IMAGENET-PIPELINE: GPU-side augmentation, distributed checks, measured cost (engineering only) — DRAFT 2026-10-05

Plan: `VCS_SSL_Server_Plan_v7_CN.md` §9 (owner 2026-10-05 "把这批次的分析也开始做").  Basis: P132 (data + distributed pairing passed; the CPU
PIL pipeline with Gaussian blur caps the loader at ~520 images/s with 32 workers).  No pretraining run in this unit: every training-like loop is
≤ 40 optimizer steps per cell, no checkpoints.  ImageNet-1K full training needs the owner's budget confirmation (§5 below).

## 1. What changed (code)
- `src/vcs_ssl/imagenet_gpu_aug.py` (new): CPU stage = JPEG decode once + per-view `RandomResizedCrop(224, (0.08, 1), bilinear, antialias)`
  (identical PIL code to P132) → uint8 tensor; GPU stage (vectorised over all crops of a step) = flip (p 0.5) → ColorJitter(0.4, 0.4, 0.4, 0.1)
  with p 0.8, factors and the random op order drawn per image → RandomGrayscale(p 0.2) → GaussianBlur(23, σ ~ U(0.1, 2.0)) with p 0.5 →
  ImageNet normalisation.  Same probabilities, ranges, order, pixel range, reflect padding and normalisation as P132; blur kept.
  Every parameter is drawn per image and per view (`sample_params`, dedicated generator per rank); the blur is a per-sample separable grouped
  convolution (torchvision v2 `GaussianBlur` draws one σ per call and is never batch-called).
- `scripts/p139_imagenet_pipeline.py` (new): `loader` (CPU-stage throughput), `check` (fixed- / random-parameter comparisons, GPU-stage cost,
  bf16 precision), `ddp` (torchrun; 1 / 2 / 4 GPU end-to-end throughput with SyncBN, global image-UID checks, communication micro-benchmarks).
- `tests/test_p139_imagenet_pipeline.py` (new, 12 tests, pass): per-sample blur == torchvision `gaussian_blur` per image (≤ 2e-6); the full
  vectorised stage == torchvision tensor functional ops applied one image at a time (≤ 3e-6) over 120 images with every op exercised; parameters
  independent per image (20 000 draws: 19 000+ distinct σ, all 24 op orders, probabilities within 1.5 pp, lag-1 σ correlation < 0.03); close to
  the P132 PIL ops for fixed parameters; `DistributedSampler(drop_last=True)` gives no duplicated image across ranks and equal step counts;
  gloo, 2 processes, real `DistributedDataParallel`: world-scaled loss + DDP averaging == single-process gradients for VCS, VCS with detached
  negatives, matched JS and SimCLR (1e-10), and the full gather carries the remote-token (key) gradient (differs from the detached version).
- `slurm/p139_{loader,check,ddp}.sbatch`.

## 2. Results so far
### 2.1 CPU stage throughput (job 1023093, nodecpu09, 32 CPUs; disjoint image subsets per cell, batch 64)

| workers | 2 views: images / s (P132 full PIL pipeline) | 4 views: images / s (P132) |
|---|---|---|
| 4 | 375 (53) | 283 (28) |
| 8 | 644 (108) | 511 (54) |
| 16 | 1127 (244) | 891 (137) |
| 32 | **1907 (520)** | **1779 (252)** |
CPU-stage speed-up 3.7× (2 views) / 7.1× (4 views) at 32 workers, ≈ 60 images / s per worker at 2 views.  Workers needed per GPU at the
P132 GPU-bound rates (2 views): H100 (1730 img/s) ≈ 29, RTX6000PRO (1204) ≈ 20, L40S (592) ≈ 10.  A 4×H100 node (64 CPUs) therefore stays
CPU-bound with CPU decoding → the optional GPU-decode path (§2.4) is measured as the scalable alternative.

### 2.2 Fixed parameters, 600 real crops (job 1023120, CPU device; the same code runs on the GPU)
- GPU-stage code vs torchvision tensor functional ops (one image at a time, same parameters): max 0.0004 uint8 levels (mean 0.00009) — exact
  up to float rounding.
- vs the P132 PIL ops with the same parameters: mean |Δ| 1.18 levels per pixel (p95 1.86, max 2.79); almost all of it from ColorJitter
  (1.36 with jitter on vs 0.11 off: PIL's integer contrast mean, integer L conversion and 8-bit HSV); grey / blur add ≤ 0.1.

### 2.3 Random parameters, 2000 images × 2 views: P132 PIL pipeline vs CPU-crop + augmentation stage (job 1023120)

| summary per view | P132 PIL mean (sd) | new stage mean (sd) | KS (crit. α 0.01: 0.036) |
|---|---|---|---|
| mean R / G / B | 0.463 / 0.442 / 0.405 | 0.468 / 0.447 / 0.410 | 0.018 / 0.018 / 0.025 |
| pixel std | 0.211 (0.076) | 0.210 (0.078) | 0.024 |
| saturation proxy | 0.116 (0.119) | 0.116 (0.122) | 0.028 |
| Laplacian variance (sharpness / blur) | 0.0098 (0.0185) | 0.0098 (0.0180) | 0.026 |
No summary distribution differs detectably (all KS below the α 0.01 critical value); the new stage is ≈ 0.005 brighter on average.

### 2.4 GPU check (job 1023094, node39, L40S)
- Same fixed- / random-parameter results on the GPU (vs torchvision tensor ops max 0.0005 levels; vs PIL mean 1.19; max KS 0.031 < 0.036).
- GPU-stage cost (flip / jitter / grey / blur / normalise, vectorised): **27.8 ms per 256 crops** (= one 2-view step of 128 images) on L40S,
  ≈ 13 % of an L40S training step at that batch (≈ 216 ms at 592 img/s).
- Optional GPU decode (nvjpeg) + GPU RandomResizedCrop: 200 images, 0 decode failures; vs PIL decode mean |Δ| 0.10 levels (max 1.27); torch vs
  PIL bilinear-antialias resized crop with identical boxes mean |Δ| 0.15 levels (max 0.21); **30.5 ms per 128 images × 2 views** on L40S.
- bf16: the init-time comparison is ill-conditioned (at random init all z are nearly parallel, pair-logit differences sit at bf16 rounding
  scale: gradient cosine to fp32 ≈ 0.1 for VCS / JS / SimCLR alike, in both bf16 modes) → rerun from a 200-step fp32-trained snapshot with fp32-
  repeat and input-noise baselines (job 1023140, §2.5).

### 2.5 Pending: bf16 snapshot check (job 1023140); 1- / 2-GPU DDP end-to-end throughput incl. the GPU-decode variant (job 1023121);
4-GPU (job 1023095, deferred by the main session with Nice 3000 — a whole-node 4-GPU request would hold freed GPUs idle).

## 3. Implementation notes (plan §9.2 checklist)
- PIL vs tensor: the GPU stage computes in float32; P132's PIL path rounds to uint8 after every op and uses PIL's integer grey (L), integer
  contrast mean and 8-bit HSV.  Outputs are therefore close but not bit-identical → the GPU stage is registered as a **new shared
  implementation** used identically for VCS / JS / SimCLR (not claimed bit-equal to P132).
- Distributed bookkeeping: image UID = dataset index; per step the UIDs of all ranks are all-gathered and checked unique; every view slot of an
  image is filled from the same dataset item on the same rank, so positives match across ranks by construction (the pair loss assigns global
  image id = rank × b + local index, consistent with the rank-ordered all-gather).  `DistributedSampler(drop_last=True)` + `DataLoader(drop_last=
  True)`: no padding duplicates (padding would put an image in the batch twice and create false negatives), every global batch full.
- VCS / JS global P / Q denominators: global counts N_P = V·B(V−1), N_Q = V·B·V(B−1) (B global); loss_r × world, DDP averaging → single-process
  gradient (tested); gather is differentiable (`torch.distributed.nn.functional.all_gather`), never detached unless `negative_detach`.
- Precision: encoder + projector under bf16 autocast; L2 normalisation, pair logits, tanh / softplus and the P / Q sums in float32 outside
  autocast (measured against fp32 and against bf16-everything in the check job).

## 4. Proposed ImageNet-1K protocol (ResNet-50, 224 px, 2 views, 100 epochs, global batch 512, seed 0; VCS / JS / SimCLR together)
Reference: SimCLR (Chen et al., ICML 2020, arXiv:2002.05709, §B / official `google-research/simclr`) for optimizer, schedule, BN and probe;
solo-learn ImageNet configs (local copy `solo_learn/scripts/pretrain/imagenet*/`) as the reproducible LARS / SyncBN implementation reference.

| item | proposal (all three methods identical unless noted) |
|---|---|
| backbone / projector | torchvision ResNet-50 (standard stem, default init); projector 2048 → 2048 (BN, ReLU) → 128, L2-normalised z (P132) |
| optimizer | LARS, momentum 0.9, trust coefficient 0.001, BN / bias excluded from weight decay and LARS adaptation |
| learning rate | base lr 0.3 × B / 256 = **0.6** (SimCLR linear scaling); dev candidate for VCS / JS (≤ 2 per method, one axis): 0.075 × √B = 1.70 (SimCLR square-root scaling) or the scorer |
| weight decay / warm-up / schedule | 1e-6 / 10 epochs linear / cosine to 0 over 100 epochs |
| BN | SyncBatchNorm over all GPUs (global BN, as SimCLR) |
| precision | bf16 autocast for encoder + projector; L2 normalisation, pair logits, tanh / softplus and the P / Q sums in float32 (§2.5) |
| augmentation | the P139 shared pipeline = P132 parameters (RRC 0.08–1, flip, ColorJitter 0.4 / 0.4 / 0.4 / 0.1 p 0.8, grey p 0.2, blur k 23 σ 0.1–2 p 0.5); **open decision:** SimCLR's paper uses jitter strength 1.0 (0.8 / 0.8 / 0.8 / 0.2) |
| pairing (VCS / JS) | all-view tokens over the GLOBAL batch, full gradient (no detach), fixed scorer (a, κ) = (2, 0.5): P = ordered cross-view same-image pairs (2 · 512), Q = all different-image token pairs incl. same view (2 · 512 · 2 · 511), separate P / Q means; matched JS on the same logits |
| SimCLR | NT-Xent over the 2 views, global negatives; **open decision:** τ = 0.1 (SimCLR ImageNet default) vs 0.2 (our CIFAR recipe / solo-learn ImageNet-100) |
| development selection | a train-side held-out subset (10 images / class = 10 000), never the ImageNet validation set |
| probe | frozen h (2048), linear head, 90 epochs SGD (Nesterov, momentum 0.9), batch 1024, lr 0.4 (0.1 × B / 256), cosine, wd 0, RRC + flip train / resize 256 + centre 224 eval; kNN (k 200, τ 0.1) alongside; ImageNet val reported once at the end |
| logging | config manifest per run (optimizer, lr, wd, warm-up, BN, augmentation, pairing, scorer, probe), per-segment GPU-hours, peak memory |

## 5. Cost (measured where available) and decisions for the owner
GPU-bound single-GPU rates (P132, 2 views, b 128 / GPU): H100 1730, RTX6000PRO 1204, L40S 592 images / s → GPU-hours per epoch 0.21 / 0.30 /
0.60 → **100 epochs ≈ 21 / 30 / 60 GPU-h per method** (three methods ≈ 62 / 89 / 180 GPU-h), if the input pipeline keeps up.
- Input pipeline: CPU stage ≈ 60 images / s per worker (2 views) → a GPU needs ≈ 29 (H100) / 20 (RTX) / 10 (L40S) workers; GPU stage adds
  ≈ 28 ms per 256 crops (L40S); the optional GPU-decode path adds ≈ 30 ms per 128 images × 2 views (L40S) and frees the CPU.
- **Pending:** multi-GPU scaling (2 GPUs: job 1023121; 4 GPUs: job 1023095 — both parked at Nice 3000 until the 800-epoch training queue
  drains, because whole-node multi-GPU requests would hold freed GPUs idle) and the 1-GPU end-to-end number with this pipeline (part of 1023121).
- Decisions for the owner: (1) ImageNet-1K full training budget (≈ 60–180 GPU-h for the three 100-epoch runs depending on GPU type, plus the
  ≤ 2 dev candidates per method for VCS / JS); (2) jitter strength (P132 0.4 vs SimCLR 1.0) and SimCLR τ (0.1 vs 0.2); (3) CPU decode (simpler,
  needs ≈ 20–30 CPU cores per fast GPU) vs GPU decode (frees the CPU, ≈ 30 ms GPU time per step on L40S); (4) when to run the 2- / 4-GPU scaling
  measurement (it needs a moment when the 8-GPU quota has free GPUs on one node).
