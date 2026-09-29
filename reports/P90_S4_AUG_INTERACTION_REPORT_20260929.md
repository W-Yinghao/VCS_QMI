# P90 — S4 report: method × augmentation strength at 8× (VCS vs tuned SimCLR) — 2026-09-29

Pre-registration `P89_S4_AUG_INTERACTION_PREREG_FROZEN_20260928.md`; results-only commits e879c5c (`P89_s4_table.md`) and this commit (mechanism records
`reports/P89_mechanism/*.json`, `reports/P89_mechanism.md`, job 1014424).  Selection split, frozen-h linear / kNN at epoch 800, 3 seeds per cell, seeds paired
by index (same initialisation and loader / pair RNG across methods and augmentations).  Strong block = the hpO augmentation block verbatim (crop 0.08–1,
jitter 0.8/0.8/0.8/0.2); VCS strong seed 0 is P43 (pre-existing), all other strong cells are new; standard cells reused (P35, P41).

## 1. Cells and the pre-stated reading
| cell | seed 0 | seed 1 | seed 2 | mean ± sd (linear) | kNN mean |
|---|---|---|---|---|---|
| VCS, standard | 86.42 | 87.16 | 87.44 | 87.01 ± 0.53 | 85.46 |
| VCS, strong | 87.78 | 87.54 | 87.66 | 87.66 ± 0.12 | 85.28 |
| SimCLR, standard | 88.20 | 88.10 | 88.66 | 88.32 ± 0.30 | 87.65 |
| SimCLR, strong | 89.74 | 89.08 | 89.66 | 89.49 ± 0.36 | 88.71 |

Gain from strong augmentation: VCS +1.36 / +0.38 / +0.22 (mean +0.65 ± 0.62); SimCLR +1.54 / +0.98 / +1.00 (mean +1.17 ± 0.32).
Δ_int(s) = gain_VCS − gain_SimCLR = −0.18 / −0.60 / −0.78, **mean −0.52** (range −0.78 to −0.18); on kNN −1.68 / −0.86 / −1.20, mean −1.25.

**Verdict (rule): "VCS gains more from strong augmentation than SimCLR" does not hold** (mean Δ_int ≤ 0, every seed negative).  The common-effect clause
does not apply (only SimCLR gains ≥ 1.0).  SimCLR's strong cells do not lose accuracy.  Absolute, reported but not the S4 question: the SimCLR–VCS gap is
1.31 under the standard block and 1.83 under the strong block.

## 2. Mechanism records (seeds 0–1 of each cell, epochs 100 / 400 / 800, same 2 048 selection images, views and partners; "own" = the run's training block)
| run (epoch 800, own block) | actual gate E_M(1−T²) | 1 − J | sat pos / neg | cos_z median pos / neg | ‖∂L/∂x‖ | align h | unif h | erank h |
|---|---|---|---|---|---|---|---|---|
| VCS standard (P35 s0 / s1) | 0.117 / 0.118 | 0.026 / 0.027 | 0.49 / 0.85 | 0.978 / 0.773 | 2.0e-3 | 0.33 | −2.47 | 150 / 146 |
| VCS strong (P43 s0 / P89 s1) | 0.334 / 0.334 | 0.127 / 0.127 | 0.00 / 0.43 | 0.988 / 0.889 | 8.3e-3 | 0.47 | −2.49 | 132 / 132 |
| SimCLR standard (P41 s0 / s1) | — | — | — | 0.924 / −0.004 | 3.2e-2 | 0.37–0.41 | −3.22 / −3.48 | 159 / 154 |
| SimCLR strong (P89 s0 / s1) | — | — | — | 0.886 / −0.018 | 5.4e-2 | 0.49 | −3.07 to −3.10 | 143 / 141 |

- **The desaturation mechanism happens, the payoff does not.**  Strong augmentation raises VCS's actual learning gate 2.9× (0.117 → 0.334), removes positive
  saturation (0.49 → 0.00) and quadruples the input gradient — exactly the change the supplement predicted (its 4.7× figure came from 1 − S; the measured gate
  ratio is 2.9×, and 1 − J overstates the change at 4.9×).  Yet the linear gain is half of SimCLR's and kNN does not move.
- **Why the extra gradient does not buy separation:** under the strong block the VCS negatives move *closer* (median shifted-pair cosine 0.773 → 0.889 of z,
  a narrower cone), h uniformity stays at −2.47/−2.49 while SimCLR's is −3.1 to −3.5, and VCS's effective rank drops 150 → 132 (SimCLR's drops too, 159 → 142).
  The bounded critic can be satisfied by a threshold inside a narrower cone; SimCLR's softmax keeps spreading negatives (median −0.02) under both blocks.
- The "standard"-block evaluation of the strong-trained models (gate 0.27, cos_neg 0.88) shows the desaturation is mostly the augmentation's harder positives,
  not a change in the learned critic scale.  Records under both blocks are identical for runs trained with the standard block, as they must be.

## 3. What can be written
- Can: at 8× on CIFAR-10, both methods gain from strong augmentation; SimCLR gains more (+1.17 vs +0.65 linear, +1.07 vs −0.18 kNN), so the gap widens.
- Can: strong augmentation desaturates VCS's critic (measured gate 0.12 → 0.33); the extra gradient does not translate into more uniform representations.
- Cannot: that VCS "benefits more from harder views" or that de-saturation is VCS's missing ingredient at this budget.

## Delivery (Spec v2 §13.2)
```yaml
experiment_family: full_ssl
protocol_id: P89_S4_method_x_augmentation
source_commit: 2e25d2b (frozen); results e879c5c; mechanism this commit
source_basis: supplement_af7172c_plus_v2
estimator: vcs_neural | simclr (tuned P41)
estimand: S (VCS) | InfoNCE_B (SimCLR)
evaluation_readout: frozen-h linear + kNN, selection split; mechanism records (gate, saturation, gradients, alignment/uniformity, erank)
reference_measure: mixture_equal (VCS)
critic_class: cosine_tanh_2_scalars | none
gradient_routing: negative_right_detach (VCS) | native (SimCLR)
n_independent_units: 45000 fit images; 3 seeds per cell; mechanism 2048 selection images x seeds 0-1
split_manifest_hash: cifar10_dev45k_val5k (f819026a...)
status: complete
```
