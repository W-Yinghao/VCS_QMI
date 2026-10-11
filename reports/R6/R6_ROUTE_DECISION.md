# R6-04 — route decision: one natural-data pilot — 2026-10-11

**Decision: REG (cross-modal registration) on BF/SHG, translation-only first.**  GAIN keeps its controlled validation (R6-02G) and its card
(R6-01).  It is ready to start as soon as the owner accepts a GAIN dataset's terms.  NUISANCE / COMPRESS / AMORTIZE stay cards.
**Basis:** the owner's standing authorization for useful experiments on free GPUs (2026-10-11), read with the package's limit of one natural
route (docs/08 §1).

## What was compared
| | REG | GAIN |
|---|---|---|
| object (R6-01) | within-scene-reference CS discrepancy F_T(φ) = J(T; P_φ, Q_φ) of a fixed, independently trained critic, used as a registration energy (type ii, not global QMI) | Δ = R*_U − R*_UV (binary S / multi-class Brier increment), plus the finite-predictor caveat |
| what is new if it works | a learned local-relation energy that recovers geometry more reliably than MI / NMI or CoMIR features under one optimizer, and an account of when it fails (critic error ↔ energy shape) | a pre-acquisition policy from a gain target that beats uncertainty and direct risk regression at measured cost |
| controlled evidence (R6-02) | learned critics ≈ the exact oracle on known shifts and degrade less than MI under heavy noise.  Main negative: cross-scene appearance changes favour per-image MI / NMI (0.85–0.87 vs 1.00).  Fixed interior support removes the leakage artefacts. | exact gains reproduced.  The squared disagreement z is unsafe with an imperfect full model; a realised-gain policy reaches the oracle only where uncertainty is partly irreducible; on XOR nothing beats random.  The useful policy is close to standard risk regression. |
| natural data (R6-03) | **BF/SHG: usable now.**  CC BY 4.0, official split (40 train / 25 val1 / 7 val2 / 134 test), CoMIR is the standard benchmark.  EPFL RGB-NIR is local (owner-accepted source).  Zurich: 20 correlated tiles, no published registration baseline. | **Blocked:** AV-MNIST (ESC-50 non-commercial noise, no audio IDs) and CREMA-D (ODbL + registration request, 7.5 GB of LFS media, no ffmpeg / PyAV) both need the owner's acceptance |
| baselines runnable now | our MI / NMI (P60); CoMIR (MIT) representations; same-target kernel | MultiBench (AV-MNIST); AdaMML archived; EDDI TF1 / tabular |
| risk | P60: raw-pixel closed-form VCS was a worse registration energy than MI / NMI on EPFL RGB-NIR.  A learned critic must beat per-image MI / NMI and CoMIR on geometric error, under scene-level appearance variation. | a CS-specific gain over direct risk regression is not expected from the controlled results |

## Why REG, and why BF/SHG (not chosen on any VCS result)
- **Feasible now** (open licence; official split and held-out test), with a published strong baseline.  GAIN cannot start without the owner.
- **The question is sharp and falsifiable.**  The energy either recovers geometry better than MI / NMI / CoMIR features under one optimizer
  or it does not.  A negative result still informs the paper (the P60 finding extended to learned critics).
- **BF/SHG over Zurich.**  BF/SHG has more independent scenes (72 development images vs 20 correlated tiles), an official split, and the
  standard CoMIR benchmark.  Its weakness: the original pairs' alignment method is not documented on the record.  Known synthetic
  perturbations give exact ground truth relative to the provided alignment, which is reported as such.
- **The test split (134 pairs) is not downloaded.**  It needs a separate, explicitly approved one-shot evaluation.

## Pilot
Frozen protocol: `reports/R6/R6_REG_P1_PROTOCOL_FROZEN_20261011.yaml`, written before any natural-data result.
- Data: BF/SHG full size, train / val1 / val2 only, downloaded to `/projects/EEG-foundation-model/yinghao/data/bf_shg_4550300`, md5 verified.
- Resources: measured by a smoke run before the full pilot.
