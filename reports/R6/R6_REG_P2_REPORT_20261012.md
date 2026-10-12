# r6 REG-P2 — the REG-P1 energies under rigid transforms (±30° + translations) on BF/SHG — report — 2026-10-12

Protocol `reports/R6/R6_REG_P2_PROTOCOL_FROZEN_20261012.yaml` (frozen before any REG-P2 result; decision-timing disclosure inside).
Results-only commit `4120ca3` (primary rows).  The three CoMIR warp-then-encode sensitivity rows (DEV only) are still running.  They will be
added as an addendum.
- **Models:** the REG-P1 models unchanged — critics from all 12 checkpoints re-selected on P2's CAL by the pre-specified lexicographic rule;
  CoMIR encode-once (its published use); MI / NMI; the kernel critic.
- **Perturbations:** 10 rigid perturbations per CAL / DEV image.  θ* is uniform on ±30° (CoMIR's test range); t* is REG-P1's translation for
  the same image and index, so every P2 registration has a P1 twin.
- **Optimiser and metric:** one shared rigid optimiser (3° × 4 px grid, then Adam on (θ, t)); CoMIR's mean corner error on the 834-px patch.

## 1. Results (DEV image-macro success, %; mean ± sd over seeds)
| method | < 42 px | < 9 px | median corner error (px) | p90 (px) | median rotation error | failures (energy) | P1 → P2 change at < 42 px (pp) |
|---|---|---|---|---|---|---|---|
| **VCS critic** | **94.2 ± 2.0** | **68.2 ± 0.7** | 5.7 | 18.5 | 0.06° | 56 (39) | −5.8 [−10.9, −0.8] |
| logistic critic | 93.1 ± 1.4 | 64.9 ± 1.5 | 5.8 | 20.0 | 0.07° | 66 (39) | −5.7 [−6.9, −4.5] |
| InfoNCE critic | 94.9 ± 1.7 | 59.6 ± 2.5 | 6.9 | 18.5 | 0.16° | 49 (49) | −0.3 [−9.0, +8.3] |
| CoMIR (encode-once) | 90.0 ± 1.1 | 62.0 ± 1.9 | 5.8 | 93.6 | 0.10° | 96 (94) | −0.9 [−3.0, +1.1] |
| MI | 74.1 | 49.4 | 9.3 | 269.2 | 0.19° | 83 (69) | −7.8 |
| NMI | 75.9 | 50.6 | 8.9 | 261.6 | 0.19° | 77 (67) | −4.4 |
| RFF kernel critic | 42.2 | 24.4 | 117.8 | 384.0 | 3.23° | 185 (174) | −15.0 |

Contrasts VCS − X (percentage points; 95 % t over seeds; image bootstrap):
| X | < 42 px (primary metric) | < 9 px (secondary) |
|---|---|---|
| **MI (primary)** | **+20.1 [+15.1, +25.2]; image [+8.8, +32.9]** | +18.9 [+17.2, +20.5]; image [+6.8, +32.5] |
| **CoMIR (primary)** | **+4.2 [−3.6, +11.9]; image [+0.3, +9.3]** | +6.2 [+2.9, +9.6]; image [+2.8, +10.3] |
| NMI | +18.2 [+13.2, +23.3]; image [+7.7, +30.3] | +17.6 [+16.0, +19.2]; image [+6.9, +30.6] |
| kernel | +52.0 [+46.9, +57.0]; image [+37.7, +65.8] | +43.9 [+42.2, +45.5]; image [+25.1, +60.3] |
| logistic | +1.0 [−6.9, +9.0]; image [+0.0, +2.9] | +3.3 [−2.0, +8.6]; image [+1.0, +6.2] |
| InfoNCE | −0.7 [−9.8, +8.3]; image [−4.2, +3.0] | +8.6 [+2.6, +14.7]; image [+3.2, +15.5] |

Reading rule as in REG-P1: a difference counts as detected only when both intervals exclude 0.

## 2. Pre-fixed readings (frozen wording)
1. **Ceiling readout not triggered.**  CAL success < 42 px is MI / NMI 0.81 and kernel 0.25, so < 42 px stays the primary metric.
2. **VCS − MI (primary): "VCS registers 20.1 percentage points more DEV registrations (< 42 px)"** (seed [+15.1, +25.2], image [+8.8, +32.9]).
   The margin over MI is unchanged from REG-P1 (+18.1); under rotation, MI's failures become catastrophic (p90 269 px).
3. **VCS − CoMIR (primary): +4.2 points; the seed interval contains 0 → "no detectable difference"** at < 42 px.
   - At < 9 px (secondary) both intervals exclude 0: +6.2 points.
   - CoMIR's failures are almost all energy failures (94 / 96), and its error tail is heavier (p90 93.6 vs 18.5 px).
4. **"VCS ≈ logistic ≈ InfoNCE → a learned-critic class effect; no VCS-specific claim":** VCS − logistic and VCS − InfoNCE contain 0 at < 42 px.
   - At < 9 px (secondary), VCS's precision exceeds InfoNCE's (+8.6, both intervals exclude 0) but not logistic's (seed interval contains 0).
   - Precision advantages are secondary and not claimed as VCS-specific.
5. **"REG-P1 critics reused frozen; REG-P2 measures transfer of the same energies to a harder transform class, not a retuned method."**

## 3. Secondary observations (descriptive)
- **Transfer to rotation.**  The critics were trained on translation-aligned crops with rot90 / flip augmentation only, never on ±30°, and are
  applied warp-then-encode.
  - They lose 0–6 points at < 42 px.  CoMIR, trained for rotation equivariance, loses about 1.
  - MI / NMI lose 4–8 points and their tails grow; the kernel loses 15.
  - Rotation errors stay small (median 0.06–0.16°) for every learned energy.
- **Failure attribution.**
  - Most failures are energy failures: the energy at the estimate is ≥ the energy at the truth.  That is a property of the energy, not of the
    search.
  - The logistic critic is the exception, with 27 of its 66 failures being search failures.
- **Cost.**  Critic registrations take 1.0–1.5 s on GPU (warp-then-encode per candidate).  CoMIR takes 0.27 s; MI / NMI 0.83 s.

## 4. Caveats
- 32 DEV images; one optimiser; CoMIR is evaluated as "CoMIR features + our optimiser".
- The rigid family is ±30°, as in CoMIR's protocol; larger rotations are not tested.
- The sensitivity rows (CoMIR warp-then-encode) are pending.
- REG-P1 → P2 changes are paired by image and translation but use different rotations by design.

## 5. Next
- **Addendum:** CoMIR warp-then-encode rows and the CoMIR wte − CoMIR sensitivity contrast.
- **r6 synthesis (R6-06):** REG-P1 / P2 and GAIN-P1 (AV-MNIST, CREMA-D).
