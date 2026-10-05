# Pre-registration — P132: v6 §8.3 ImageNet-1k preflight (ResNet-50, 224 px): data, distributed pairing, throughput — DRAFT 2026-10-04

Owner 2026-10-04: "全部提交" (list item "ImageNet-1k preflight: data / distributed-pairing / throughput check only, no training").  v6 §8.3:
"ImageNet-1K 按 ResNet-50 另建完整协议；首批只做数据、分布式配对和吞吐量检查，不默认提交大规模训练".  **No ImageNet training run is part of
this unit**; a full protocol needs its own pre-registration and the owner's separate approval.  Nothing is downloaded (data at
/projects/common/imagenet via FMCA-AV/imagenet symlinks and manifests).

## Deliverables and pass criteria
1. **Data check** (`scripts/p132_imagenet_preflight.py datacheck`, CPU job 1021158): manifest counts (train 1 281 167, val 50 000 expected),
   1000 train classes = val classes, 50 val images per class; filesystem listing matches the manifest; `stat` of every manifest path (missing /
   empty files listed); full decode of a stratified train sample (50 per class) and all 50 k val images (unreadable files listed); decode-only rate.
   Pass: counts as expected, no missing class, unreadable / missing files listed and < 0.01 %.
2. **Distributed all-view-token pairing** (`src/vcs_ssl/imagenet.py`, new file; CIFAR path untouched): A-P3 objective (fixed f = 2s − 1, P / Q
   separate global means, optional negative detach), its matched JS, and multi-view SimCLR, with cross-GPU negatives through a differentiable
   all_gather; each rank sums its local anchor rows against all global tokens over the GLOBAL pair counts × world size.  Pass (tests/test_p132_imagenet.py):
   world size 1 equals the repo's single-process all_view_tokens_loss (VCS and JS, both Q routings) and the reference NT-Xent; with 2 gloo processes
   on CPU, Σ_r loss_r / W and the DDP-averaged parameter gradients equal the single-process loss and gradients on the same global batch (float64,
   tol 1e-10) for VCS, VCS-negdetach, JS, SimCLR.  **Result: 10 / 10 pass (login-node run and CPU job 1021158).**
3. **Data-loader throughput** (CPU job 1021158): images/s for 2 / 4 views, blur p 0.5 and 0, workers 4 / 8 / 16 / 32, batch 64 (20 k-image subset).
4. **Single-GPU training throughput** (`slurm/p132_gpu_throughput.sbatch`, one job per GPU type RTX6000PRO / H100 / L40S; 300 steps per job, no
   checkpoints): torchvision ResNet-50 + projector 2048-2048-128, bf16 autocast, channels_last; A-P3 and SimCLR at 2 views × 128 images and 4 views ×
   64 images on synthetic input (GPU ceiling), A-P3 2 / 4 views on the real loader (14 workers).  Reported: s/step, data wait, images/s, crops/s,
   peak memory, projected GPU-hours per epoch on 1 GPU.  **2-GPU runs: skipped** — the 8-GPU quota is fully used by the running P127 / P129 /
   P120-addendum units and the directive forbids displacing them; the 2-process gloo test covers the pairing logic, NCCL scaling is not measured.

## What the preflight decides (and does not)
- It produces the numbers needed to cost a full protocol: GPU-hours per epoch per method and view count, whether the loader keeps the GPU fed
  with the CPUs per GPU available on each node type (RTX nodes 48 / GPU, H100 node53 16 / GPU, L40S 16–32 / GPU), and the per-GPU batch that fits.
- It does not choose the ImageNet recipe (epochs, global batch, lr, augmentation incl. blur, a / κ — v6 allows re-selecting them on ImageNet) and
  does not compare methods.  Augmentation used here is the SimCLR ImageNet view (crop 0.08–1, flip, jitter 0.4/0.4/0.4/0.1 p 0.8, grayscale 0.2,
  blur p 0.5) purely for cost realism.

## Results so far
- **Data check (job 1021158, 204 s): pass.**  Manifest train 1 281 167 / val 50 000; 1000 classes in both; val 50 per class; train 732–1300 per
  class; filesystem listing equals the manifest (no missing class directory); `stat` of all 1 331 167 paths: 0 missing / empty, 153.6 GB; full
  decode of 100 000 images (50 per train class + all val): 0 unreadable; decode-only 4 115 images/s with 32 processes.
- **Distributed pairing: pass** (10 / 10, see item 2).
- **Loader (job 1021158): not usable as a rate yet.**  v1 reused one 20 k-image subset across cells, so later cells read from the page cache
  (16 → 32 workers jumps 232 → 1691 images/s at 2 views, blur 0.5); the early, cold cells (4 / 8 / 16 workers: 64 / 109 / 232 images/s at 2 views,
  33 / 78 / 103 at 4 views) suggest first reads from /projects/common are the bottleneck.  The script now uses disjoint subsets per cell and longer
  windows; rerun: `sbatch slurm/p132_cpu_check.sbatch` (or only the loader line).  An ImageNet run will likely need a local / node-cached copy or
  the WebDataset shards (/projects/common/imagenet-1k-wds) — to be measured, not assumed.
- **GPU throughput: pending** — jobs 1021697 (RTX6000PRO) and 1021698 (H100) queued on the 8-GPU quota; L40S job not yet submitted
  (`sbatch --partition=L40S slurm/p132_gpu_throughput.sbatch`).  Results → `reports/P132/gpu_<partition>_<job>.json`.
