# r6 GAIN-P1 — CREMA-D: on-demand video branch for emotion recognition — report — 2026-10-12

Protocol `reports/R6/R6_GAIN_P1_CREMAD_PROTOCOL_FROZEN_20261012.yaml` (frozen before any full-size result).  Results-only commit `69e283b`
(`R6_GAIN_P1_cremad_results.json`, `R6_GAIN_P1_cremad_tables.md`).
- **Run:** two per-family GPU bundles (1034825 / 1034826) and an aggregate (1034827).
- **Systems:** 2 families (multi-class Brier, cross-entropy) × 10 seeds × 5 actor folds = 100 systems; none incomplete.
- **Model:** a shared ImageNet ResNet-18 audio encoder with a cheap audio head a(U), and a video branch (8 frames, ResNet-18) fused into b(U, V).
- **Policies:** see the audio side only.
- **Data:** every clip is DEV once (91 actors; 7 442 clips; 6 emotions).
- **Primary metrics:**
  - Risk: mean DEV Brier over budgets {10, 25, 50, 75} % under exact-rate routing.
  - Cost: end-to-end ms per clip at batch size 1, decoding included.
- **Intervals:** two-way bootstrap over seeds × actors (primary), with seed t and actors-given-systems alongside.

## 1. Results (primary: exact-rate routing; mean DEV Brier over the four budgets)
| policy | Brier family | CE family |
|---|---|---|
| never (audio only) | 0.6296 (acc 0.568) | 0.6209 (acc 0.578) |
| always (audio + video) | 0.4863 (acc 0.677) | 0.4835 (acc 0.682) |
| random | 0.5722 | 0.5659 |
| entropy of a | 0.5635 | 0.5563 |
| 1 − max a | 0.5638 | 0.5566 |
| gain_z (regress z = ‖b − a‖²) | 0.5486 | 0.5396 |
| **gain_h (regress h = ‖e − a‖² − ‖e − b‖²)** | **0.5432** | **0.5376** |
| oracle_z / oracle_h (not deployable) | 0.5064 / 0.4140 | 0.5039 / 0.4042 |

| contrast (Brier, < 0: first better; two-way 95 %) | Brier family | CE family |
|---|---|---|
| always − never | −0.1433 [−0.1676, −0.1160] | −0.1374 [−0.1638, −0.1110] |
| **gain_h − entropy** | **−0.0204 [−0.0279, −0.0136]** | **−0.0186 [−0.0261, −0.0112]** |
| gain_h − random | −0.0290 [−0.0345, −0.0234] | −0.0282 [−0.0338, −0.0226] |
| **gain_z − gain_h** | **+0.0055 [+0.0008, +0.0110]** | **+0.0019 [−0.0026, +0.0068]** |
| gain_z − entropy | −0.0149 [−0.0206, −0.0095] | −0.0167 [−0.0233, −0.0101] |
| entropy − random | −0.0086 [−0.0129, −0.0045] | −0.0096 [−0.0146, −0.0043] |
| gain_h − oracle_h | +0.1291 [+0.1188, +0.1393] | +0.1335 [+0.1226, +0.1446] |

**Measured cost** (ms per clip, batch 1, warm median of 3; single-threaded CPU work; both families timed on the same node type, AMD EPYC
9275F + RTX PRO 6000):
- never (audio path + head) 1.58 / 1.55; always 33.34 / 32.96.
- The video path costs 31.8 ms: decode 25.2, crop / resize 4.9, video branch + fusion 1.6.
- Policy overhead: MLP 0.31 ms, entropy 0.08 ms.
- Decode by codec (first fold's DEV): VP6F 26.2 ms (1 552 clips), H.264 22.3 ms (6 clips).

| gain_h (Brier family) | 25 % | 50 % | 75 % | always |
|---|---|---|---|---|
| DEV Brier | 0.5639 | 0.5174 | 0.4922 | 0.4863 |
| ms per clip (share of always) | 9.69 (29 %) | 17.58 (53 %) | 25.59 (77 %) | 33.34 |
| entropy at the same budget: Brier / ms | 0.5901 / 9.42 | 0.5455 / 17.28 | 0.5044 / 25.21 | |

## 2. Pre-fixed readings (frozen wording)
1. **always − never is below 0 in both families.**  The video branch adds task value (+10–11 accuracy points), so acquisition readings apply.
2. **"gain_h beats entropy and random → the realised-improvement regression ranks better; this is standard risk regression, not a
   CS-specific method"** holds in both families.  The margin over entropy, 0.019–0.020 Brier, is about seven times AV-MNIST's.
3. **Brier family: "gain_z worse than gain_h → the squared disagreement is not a safe acquisition target for these finite models"**
   (+0.0055 [+0.0008, +0.0110]; the gap grows with the budget, +0.011 at 50 %).
   **CE family:** the interval contains 0 (+0.0019 [−0.0026, +0.0068]) — "an interval containing 0 only contains 0".
4. **Cost, in the frozen claim form:**
   - "At a 25 % budget, gain_h reaches DEV Brier 0.564 at 9.7 ms per clip (29 % of always)."
   - "At 75 %, it reaches 0.492 at 25.6 ms (77 % of always), against always's 0.486 at 33.3 ms."
   - Matched budgets are near-matched cost: gain_h selects slightly longer clips (+0.1–0.4 ms per clip versus entropy at the same budget).
   - "Cost savings are measured wall-clock at batch size 1 on the stated CPU / GPU."

## 3. Secondary observations (descriptive)
- **Why the learned target matters here and not on AV-MNIST:**
  - corr(score, h) is 0.16 for gain_h and 0.14–0.15 for gain_z, against 0.04–0.05 for entropy / 1 − max p.  On AV-MNIST all four were 0.08–0.11.
  - The audio side carries information about when the video helps (which emotions and actors) that the cheap head's own uncertainty does not
    reflect.
- **z overstates the realised gain about 2×** (mean z 0.26–0.28 vs mean h 0.14).  h is positive for 67–68 % of clips.
- **The oracle gap stays large:** 0.13 Brier for both families.  At matched budget, gain_h closes about a fifth of the gap between random
  selection and oracle_h: (0.5722 − 0.5432) / (0.5722 − 0.4140) = 18 % (Brier); (0.5659 − 0.5376) / (0.5659 − 0.4042) = 17.5 % (CE).
- **Identity check:** h − z = 2 (b − a)·(e − b) holds to 1.3e-15.
- **Codecs:** decode dominates the video path (≈ 80 %); the 31 H.264 clips decode slightly faster than VP6F.

## 4. GAIN-P1 across the two datasets
| | AV-MNIST (audio branch) | CREMA-D (video branch) |
|---|---|---|
| always − never | −0.050 / −0.054 | −0.143 / −0.137 |
| gain_h − entropy | −0.0028 [−0.0056, −0.0002] / contains 0 | −0.020 / −0.019 (both below 0) |
| gain_z − gain_h | contains 0 / contains 0 | **+0.0055 (above 0)** / contains 0 |
| cost claim | none (µs, policy ≈ branch) | 29 % of always's wall-clock at a 25 % budget |

- The realised-improvement regression is the safer target.  It is never worse than z, and is better where the extra branch is valuable and
  its gain predictable (CREMA-D, Brier family).
- This matches the controlled validation's warning about z, now on natural data.
- Neither result is CS-specific: both are standard risk regression.

## 5. Caveats
- One architecture per dataset; ImageNet initialisation; no augmentation; hyperparameters not tuned.  Labels are the intended emotions.
- Costs are batch 1 only (stated in the protocol).
- Actors are the independent unit.  Sentences (12) are shared across folds.
