# Pre-registration — P154: ImageNet-100 pilot — VCS (A-P3) / matched JS / SimCLR, ResNet-18, 224 px, single seed — DRAFT 2026-10-08

Owner 2026-10-08: selected "ImageNet-100 pilot" (≈ 100+ GPU-h envelope) when asked how to use free GPU time.  Builds on P132 (ImageNet data and
the A-P3 / SimCLR losses with exact single-process reduction) and P139 (GPU-side augmentation, tested against torchvision / PIL).  First
pretraining beyond CIFAR; ImageNet-1K stays unbudgeted.

## 1. Data
ImageNet-1k train rows of the 100 CMC classes (solo-learn `imagenet100_classes.txt`; ≈ 126.7 k images).  Development split: 50 images per class
(seeded permutation, seed 20261008) held out = 5 000 development-validation images for the readout; the rest (≈ 121.7 k) train the encoder
(labels unused) and fit the probe.  **The ImageNet validation set is not used** (it plays the role of the closed official test).

## 2. Recipe (identical for the three methods — the CIFAR protocol, not a per-method tuned recipe)
torchvision ResNet-18 (standard ImageNet stem; h 512) + projector 512 → 512 (BN, ReLU) → 128, L2-normalised z; 4 views per image at 224 px;
P139 views = PIL RandomResizedCrop (0.08–1) per view + GPU flip / ColorJitter 0.4/0.4/0.4/0.1 p 0.8 / grey 0.2 / blur 23 σ 0.1–2 p 0.5 /
ImageNet normalisation; batch 256 images; AdamW lr 1e-3, betas (0.9, 0.999), weight decay 1e-4 on matrices (0 on biases / BN); 10 warm-up epochs,
cosine to 0.01 × lr per step; **200 epochs**; bf16 autocast for encoder + projector, loss in float32; seed 0.
VCS: all-view-token J with the fixed A-P3 scorer T = tanh(2 s − 1); matched JS: the balanced logistic on the same scores; SimCLR: multi-view
NT-Xent, τ 0.2 (the CIFAR control settings).  Disclosed: SimCLR's published ImageNet recipe (LARS, larger lr, τ 0.1) is not used — the
comparison is at a shared recipe, as on CIFAR; the SimCLR number is therefore not a reproduction of the literature value (solo-learn reports ≈ 77–78 %
linear for ResNet-18 / 400 epochs / LARS on ImageNet-100, shown only as context).

## 3. Readout (the CIFAR pilot readout)
Frozen h from centre crops (resize 256 → 224); linear probe = `diagnostics.linear_probe` with the frozen pilot settings (SGD 0.1, momentum 0.9, 100
epochs, batch 256, cosine, no weight decay, final probe epoch) fitted on the training split, scored on the 5 000 development images; kNN (k 200,
τ 0.1, cosine) alongside.  Checkpoints at epochs 50 / 100 / 200 (the 200-epoch one is primary).

## 4. Pre-stated reading (single seed — descriptive)
Δ = VCS − SimCLR and VCS − JS on linear (kNN alongside).  **Trigger for seeds 1–2 of all three methods:** |Δ| ≥ 1.0 for either contrast, in either
direction.  Otherwise the pilot is reported as single-seed "within ±1.0".  No claim about ImageNet-1k, other backbones or longer schedules.

## 5. Cost and execution
Measured in the gate (`slurm/p154_gate.sbatch`): throughput, data wait, evaluation time.  Expected ≈ 1 000–1 400 images / s on RTX6000PRO with 30
decode workers → ≈ 1.5–2 min per epoch → ≈ 5–7 h per 200-epoch run; three runs ≈ 15–21 GPU-h.  One GPU per run (32 CPUs), RTX6000PRO / H100,
nodes 51 / 52 / 60 excluded; resume from `last.pt` (singleton second link only if the measured rate exceeds 22 h).
