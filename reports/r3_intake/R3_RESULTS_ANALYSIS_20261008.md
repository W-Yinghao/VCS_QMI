## A. Where the evidence stands (all numbers recomputed from per-run files on 2026-10-07; development split)

### A1. Learning side — the A-P3 recipe is a flat local optimum on every tested axis
| | CIFAR-10 linear / kNN | CIFAR-100 linear / kNN |
|---|---|---|
| VCS A-P3 (2, 0.5) | **89.02 ± 0.04** / 87.41 ± 0.08 (5 seeds) | **60.15 ± 0.38** / 55.90 ± 0.31 (5 seeds; 60.00 ± 0.25 on seeds 0–2) |
| matched JS, shared scorer | 88.81 ± 0.45 / 87.67 (3) | 59.35 ± 0.42 / 55.30 (5) — VCS +0.80 [+0.06, +1.54] clear |
| matched JS, selected scorer | (2, 0.25): 88.97 ± 0.34 / 87.39 (3) — VCS +0.05 close | (3, 0.5): 60.49 ± 0.21 / 56.31 (3) — VCS −0.49 [−0.77, −0.20] clear **at lr 1e-3 only** (P129 add. 4: +0.16 / −0.52 / +0.26 at lr × 0.5 / 1 / 2) |
| SimCLR | 88.31 ± 0.21 / **87.86** (5) | 58.25 ± 0.35 / **57.21** (3) |
| VICReg | 87.11 ± 0.51 / 84.46 (3) | 56.01 ± 0.25 / 48.38 (3) |
| recipe VCS (pre-A-P3) | — | 59.91 ± 0.20 / 54.95 (3) |
| ResNet-50, seed 0 | — | VCS **64.66** / 57.86; JS 62.56 / 57.16; SimCLR 58.54 / **59.44** (seeds 1–2 queued) |

Single-seed exploration deltas vs A-P3 seed 0 (89.06 / 60.20), linear: (a, κ) grid best +0.24 / +0.44 (P127, keeps (2, 0.5)); lr × 0.5 / × 2
−0.34 / −0.12 and +0.18 / +0.32 (P135); 1600 epochs −0.38 / −0.64, 8 views +0.14 / −0.58, batch 512 −0.48 / −0.84 (P136), batch 512 with
lr 2e-3 −0.42 (P147, C100 running); projector 512 −0.64 / −0.32 (P137); weight decay 5e-4 +0.22 (C10), 1e-5 +0.32 (C100) (P148, two cells
running); momentum keys −0.02 (P133); curvature ±0.25 within ±0.08 (P126); free scorer −3.50 (P120); K = 16 sampled pairs +0.16 / −0.20 (P138,
a cost result).  No cell reached the pre-registered +0.30 / +0.50 thresholds; three cells (P127 C100, P148 C10 / C100) sit at +0.2–0.4, inside
one to two seed-sd.  **Reading:** within ±0.5 points the recipe is insensitive to lr, decay, schedule, views, batch, head width, scorer shape
and pairing; the only large effects are the backbone (+4.5 at ResNet-50, seed 0) and the two negative controls (free scorer, kernel routes).
Further single-axis SSL exploration therefore has low expected value; what remains on the learning side is closing the three triggered
multi-seed confirmations (P130 / P138 / P145) and the ImageNet question (owner budget).

Other learning-side facts to carry into the paper: kNN favours SimCLR on both datasets while frozen-h linear favours VCS (P141: VCS concentrates the
extra information in h, SimCLR in z); CIFAR-10-C mCA A-P3 70.58 ≈ SimCLR 70.63, matched JS 71.94 better on noise (P131); contamination ε = 0.10
leaves CIFAR-10 within ±0.3 for all three methods, CIFAR-100 seed 0 diverges (JS +1.18, VCS +0.10, SimCLR −1.30; seeds 1–2 running).

### A2. Measurement side — what each unit does and does not support (the r3 question)
| unit | supports | does not support | r3 use |
|---|---|---|---|
| P86 (+ SMILE correction) | same-S: VCS posterior MSE .169 vs RFF .507 / KDE .527 at d 50; N ladder .364 → .029; in-batch InfoNCE and corrected SMILE order the ladder | "VCS most accurate / lowest-variance among neural methods" (MI baselines were never put on the same error axis) | E0: re-aggregate into precision / stability / resolution panels with every method |
| P108 (+ correction) | the second-order gap identity; **plug-in better than raw J in 70 / 72 (bias) and 69 / 72 (RMSE)** | "raw J is the estimator of S"; "J is a per-draw lower bound" | E0: keep as the main honest caveat on the J readout |
| P134 (R1) | normalised shift of the readout under representation perturbation is smaller for VCS (0.88 / 0.18 / 0.62 vs JS 1.40 / 0.13 / 1.64) | a general stability advantage (one perturbation family) | E0 stability panel, labelled |
| P122 | two-view scalar-s readouts under the standard law sit at 0.97–0.99 (saturated); the strong law recovers structure (≈ 0.88); independent measurement critics differ from the fixed training scorer | any "low-variance" reading at saturation | V0 baseline fixture; the t-axis of V1 is what creates resolution |
| P101 | adding noise to frozen VCS checkpoints did not raise resolution while keeping kNN | that a controlled common channel is uninformative as a stress axis | V1 is a stress check, not an improvement route |
| P116 / P142 | on a fixed encoder, logistic ≈ VCS > same-target RFF > conditional HSIC (calibrated nulls) | numerical accuracy of any of them (power is invariant to monotone rescaling) | A0: supporting detection table |
| P143 | colour 0.1 far more detectable in the balanced-posterior encoders than in SimCLR / VICReg, both datasets | "VCS is more interpretable"; amounts of retained information | A0: model-observation table |
| P144 | real maps h→r→z→O measured exactly; held-out J at h ≤ 0 at n 3 000 | any compression quantity on images | C0: one crossed diagnostic, correct bootstrap |
| P118 (truth) | nested fitting lowers oracle increment error (.0105 → .0070) | that small real-image residuals mean small oracle error | §3.3 evidence stays as is |

**The gap r3 points at is real:** the paper's estimator claims rest on P86 / P108, but no existing table shows precision, stability and resolution of
the *same* fits side by side, and the MI baselines appear only as ordering statements.  The real-image measurement that connects the estimator to the
SSL object (two independent views of the same image) exists only as the saturated P122 scalar readouts.  Attribute detection (P142 / P143) is strong
as an application but cannot substitute for either.
