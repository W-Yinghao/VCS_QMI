# P132 — ImageNet-1k preflight (data, distributed pairing, throughput; no training) — report — 2026-10-05

Draft protocol `P132_V6_IMAGENET_PREFLIGHT_PREREG_DRAFT_20261004.md` (v6: "首批只做数据、分布式配对和吞吐量检查，不默认提交大规模训练"; owner "全部提交").
Code `src/vcs_ssl/imagenet.py`, `scripts/p132_imagenet_preflight.py` (commit cb00c8d); results `reports/P132/` (latest `3d6edd8`).  No model was trained.

## 1. Data (CPU job 1021158) — pass
ILSVRC-2012 at `/projects/common/imagenet/ILSVRC/Data/CLS-LOC`: 1,281,167 train / 50,000 val JPEGs, 1000 classes in both (train 732–1300 per class,
val 50 per class), manifest = filesystem, 0 missing or empty of 1,331,167 files (153.6 GB), 0 unreadable in a 100,000-image full decode
(4,688 decoded images / s with 32 processes).

## 2. Distributed pairing — pass
Cross-GPU all-view-token losses (A-P3 VCS, VCS with negative detach, matched JS, multi-view SimCLR): with one process they equal the repository's
single-GPU losses; with two gloo processes the summed loss and DDP-averaged gradients equal the single-process values to 1e-10 (tests 16 / 16).

## 3. GPU throughput, ResNet-50 at 224 px, real ImageNet batches, 1 GPU (300 steps; bf16 autocast, channels_last, TF32; projector 2048-2048-128)

| GPU (node) | A-P3, 2 views (B 128): images / s → GPU-h per epoch | A-P3, 4 views (B 64) | peak memory |
|---|---|---|---|
| H100 NVL (node53) | **1730 → 0.21** | 845 → 0.42 | 12.3 GB |
| RTX6000PRO, healthy (node58) | 1204 → 0.30 | 603 → 0.59 | 12.3 GB |
| L40S (node57) | 592 → 0.60 | 298 → 1.20 | 12.3 GB |
SimCLR within 3 % of A-P3 on every GPU.  Unlike the CIFAR ResNet-18 runs (where healthy RTX was fastest), H100 is the fastest GPU here.
Data wait inside these jobs was ≈ 3–8 ms per step (16 CPU cores per job).  Multi-GPU scaling was not measured (8-GPU quota fully used).

## 4. Data loader (CPU job 1021775, corrected: a fresh image subset per cell) — **the bottleneck**

| augmentation | workers | 2 views: images / s | 4 views: images / s |
|---|---|---|---|
| with Gaussian blur (p 0.5, PIL) | 4 / 8 / 16 / 32 | 53 / 108 / 244 / **520** | 28 / 54 / 137 / **252** |
| without blur | 32 | **1958** | 1040 |
With the standard SimCLR / BYOL blur on the CPU, 32 workers deliver 520 images / s — below one L40S and ≈ ⅓ of one H100.  Without blur the loader
keeps up with one H100.  Reading from `/projects/common` is not the limit once the files are decoded in parallel.

## 5. Consequences for an ImageNet protocol (planning only — no training authorised)
- A 2-view, 100-epoch ResNet-50 run costs ≈ 21 GPU-h (H100) / 30 (RTX) / 60 (L40S) of GPU time **if** the input pipeline keeps up; with CPU blur
  it would be capped at ≈ 520 images / s per job (≈ 68 h per 100 epochs on one GPU, whatever the GPU).
- Needed before any ImageNet training: GPU-side augmentation (at least the blur; e.g. torchvision v2 / kornia ops on CUDA tensors) or a
  pre-decoded / WebDataset path (`/projects/common/imagenet-1k-wds`), then a multi-GPU scaling check.  ImageNet-100 (CMC split) would cost ≈ 1/10.
- Recommendation for the owner's decision: if ImageNet is wanted, first a small engineering unit (GPU augmentation + 2- / 4-GPU scaling, ≈ 2 GPU-h),
  then ImageNet-100 ResNet-50 for VCS (A-P3) / JS / SimCLR as the first training step.
