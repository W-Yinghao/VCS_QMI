# P97 — 16× compute controls report — 2026-10-01

Pre-registration `P97_CONTROLS_16X_PREREG_FROZEN_20260930.md`; results-only commit (`P97_controls_16x_table.md`).  Tuned SimCLR and VICReg (P41 4-view recipes)
extended to 1600 epochs exactly as the VCS 16× run; seed 0; selection split, frozen-h linear + kNN at epoch 1600.

| method | 8× seed 0 (linear / kNN) | 8× 3-seed mean | 16× seed 0 (linear / kNN) | 8× → 16× (seed 0) |
|---|---|---|---|---|
| VCS (P35 → P43) | 86.42 / 85.60 | 87.01 | **87.50 / 85.84** | +1.08 |
| SimCLR (P41 → P97) | 88.20 / 87.20 | 88.32 | **87.64 / 87.56** | −0.56 |
| VICReg (P41 → P97) | 87.50 / 84.38 | 87.11 | **86.06 / 83.40** | −1.44 |

**Pre-stated reading (one seed per cell; direction only).**
- SimCLR − VCS at 16× = +0.14 against +1.31 at 8× (3-seed): the difference is 1.17 smaller → **"the VCS–SimCLR gap narrows at 16×"** (linear).  On kNN SimCLR
  still leads by 1.72 (8×: 2.19).
- VICReg − VCS at 16× = −1.44 against +0.10 at 8×: 1.54 smaller → **"narrows"** (in fact reverses: VCS above VICReg at 16× on both metrics).

**Reading the shape, not only the rule.**  The narrowing comes mostly from the controls *declining* from 8× to 16× (SimCLR −0.56, VICReg −1.44 on seed 0) while VCS
keeps gaining (+1.08 on seed 0; +0.49 against its 3-seed 8× mean).  The tuned control recipes were selected at ≤ 8× (P41) and their linear accuracy saturates or
drops with longer training; VCS's does not yet.  One seed per 16× cell — the direction is reported, no claim of a VCS lead over SimCLR is made (+0.14 is far
inside seed noise, and kNN still favours SimCLR).  This is consistent with the CIFAR-100 scaling finding (P92/P98: VCS's 2× → 8× gain exceeds both controls').

## Delivery
```yaml
experiment_family: full_ssl (compute curve)
protocol_id: P97_controls_16x
source_commit: ad11130 (frozen); results this commit's parent
estimator: simclr (tuned) | vicreg (tuned); VCS reference P43 (one seed)
evaluation_readout: frozen-h linear + kNN, selection split, epoch 1600
n_independent_units: 45000 fit images; 1 seed per cell
status: complete
```
