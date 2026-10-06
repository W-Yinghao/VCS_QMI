# Decisions and budget after the MV6 intake (2026-10-06)

| field | resolved from the server | still open | affects |
|---|---|---|---|
| ImageNet authorisation / jobs | P132 preflight done (no training); P139 GPU-augmentation pipeline in progress (bf16 check pending; 2/4-GPU DDP parked) | full-training GPU budget (owner) | MV6-S1 |
| ImageNet dataset | ImageNet-1k at /projects/common/imagenet (manifests in FMCA-AV/imagenet); ImageNet-100 = CMC split exists | which one for the paper (owner) | MV6-S1 |
| new resource cap | measurement units ≈ 1.5 GPU-h + ≈ 35 CPU-node-h (estimates; smokes replace them) | go for P142–P144 (D1) | P142–P144 |
| Table 3 fixture | F-SOLVER-P116 features on disk; same draws reproducible (seeded) | add block B on the Table 4 SimCLR (D2) | MV6-I1 |
| same-target kernel / conditional HSIC entry points | RFF critic on J (P86 family) adapted to φ_K = [ψ(h)(2N−1),1]; `cond_test_t1.py` class-wise and deep-kernel conditional HSIC with shared permutations | — | MV6-I1 |
| additional encoders | VICReg 4v/800 ep seeds 0–2 (C10: P41 add. 2; C100: P91), selected logistic P129 cells seeds 0–2 | include CIFAR-100 stage (D3) | MV6-I2 |
| h→r→z / h→O map | projector + L2 norm + clean-h head in the checkpoints; tolerance gate in P144 | — | MV6-C1 |
