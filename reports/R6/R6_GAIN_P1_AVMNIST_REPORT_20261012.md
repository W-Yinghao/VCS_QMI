# r6 GAIN-P1 — AV-MNIST: routing the audio branch on demand — report — 2026-10-12

Protocol `reports/R6/R6_GAIN_P1_PROTOCOL_FROZEN_20261012.yaml` (frozen before any full-size result; owner: "这几个包的任务全都需要完成").
Results-only commit `8abc5aa` (`R6_GAIN_P1_avmnist_results.json`, `R6_GAIN_P1_avmnist_tables.md`, both audits).
- **Run:** one GPU bundle (1034756, RTX PRO 6000) trained 2 families (multi-class Brier, cross-entropy) × 10 seeds = 20 systems.  Each system has
  a cheap image head a(U) and a fused head b(U, V) on a shared image encoder.
- **Policies:** see only the image side (features + a's probabilities) and decide whether to run the audio branch.
- **Evaluation:** DEV read once; official test arrays never extracted.
- **Roles and units:** grouped by audio clip (FIT_TASK / FIT_POLICY / CAL / DEV = 961 / 320 / 160 / 160 clips, 36 086 / 11 957 / 5 968 / 5 989
  samples).  Intervals are two-way bootstraps over seeds × DEV clip clusters (primary), with seed-paired t and cluster-only intervals alongside.

## 1. Data finding that changed the design (before any result)
- **Exact-duplicate audit of the MultiBench AV-MNIST train release:**
  - The 60 000 audio spectrograms are only **1 601 byte-exact unique arrays, each reused ~37 times** (max 43); 99.998 % of samples have an
    exact duplicate.
  - One spectrogram is paired with images of a median of 8 different digits.  The mean label purity of a clip's samples is **0.44**, so the
    release's audio–image pairing is noisy by construction.  This is in line with MultiBench's ~42 % audio-only accuracy.
  - Images are all unique.
- **Under the original image-grouped split,** every DEV spectrogram appeared verbatim ~22 times in training: an "audio gain" there would have been
  clip memorisation.  The frozen protocol therefore groups roles by audio clip.
- **Cross-role audit after regrouping:** no DEV / CAL / FIT_POLICY spectrogram has a FIT_TASK neighbour above cosine 0.99.  Nearest-neighbour label
  agreement is 0.18–0.22, not elevated at high cosine.
- **Not groupable (disclosed):** the 6 FSDD speakers are shared across roles.

## 2. Results (primary: exact-rate routing — every policy acquires exactly ⌈βn⌉ DEV samples; mean DEV Brier over β ∈ {10, 25, 50, 75} %)
| policy | Brier family | CE family |
|---|---|---|
| never (image only) | 0.4380 (acc 0.663) | 0.4370 (acc 0.665) |
| always (image + audio) | 0.3876 (acc 0.712) | 0.3835 (acc 0.716) |
| random | 0.4177 | 0.4156 |
| entropy of a | 0.4107 | 0.4083 |
| 1 − max a | 0.4086 | 0.4060 |
| **gain_z** (regress z = ‖b − a‖²) | **0.4076** | **0.4053** |
| **gain_h** (regress h = ‖e − a‖² − ‖e − b‖²) | **0.4079** | **0.4061** |
| oracle_z (true z, not deployable) | 0.3972 | 0.3946 |
| oracle_h (true h, not deployable) | 0.3324 | 0.3310 |

| contrast (Brier, < 0: first better; two-way 95 %) | Brier family | CE family |
|---|---|---|
| always − never | −0.0504 [−0.0609, −0.0409] | −0.0535 [−0.0629, −0.0439] |
| **gain_h − entropy** | **−0.0028 [−0.0056, −0.0002]** | **−0.0021 [−0.0046, +0.0002]** |
| gain_h − random | −0.0098 [−0.0127, −0.0069] | −0.0094 [−0.0122, −0.0068] |
| **gain_z − gain_h** | **−0.0003 [−0.0021, +0.0013]** | **−0.0009 [−0.0030, +0.0012]** |
| gain_z − entropy | −0.0031 [−0.0055, −0.0006] | −0.0030 [−0.0056, −0.0006] |
| entropy − random | −0.0071 [−0.0092, −0.0049] | −0.0073 [−0.0093, −0.0055] |
| gain_h − oracle_h | +0.0755 [+0.0699, +0.0817] | +0.0751 [+0.0697, +0.0810] |

## 3. Pre-fixed readings (frozen wording)
1. **always − never is below 0 in both families.**  The audio branch adds task value (+4.9 / +5.2 accuracy points), so acquisition readings
   apply.
2. **Brier family: "gain_h beats entropy and random → the realised-improvement regression ranks better; this is standard risk regression, not a
   CS-specific method."**  The margin over entropy is small: −0.0028 Brier.
3. **CE family:** gain_h − entropy is −0.0021 [−0.0046, +0.0002].  **"An interval containing 0 only contains 0"**: no advantage over cheap
   uncertainty is detected, and the interval is not an equivalence claim.  gain_h still beats random.
4. **gain_z − gain_h contains 0 in both families**, so the wording "gain_z worse than gain_h → the squared disagreement is not a safe
   acquisition target" does not apply here.
   - On AV-MNIST, regressing the squared disagreement ranks no worse than regressing the realised improvement, and also beats entropy.
   - This is consistent with the controlled validation (R6-02G), where z failed only when the full model was worse than the cheap one.  Here b
     is better on average (h > 0 for 65 % of DEV samples).

## 4. Secondary observations (descriptive)
- **Little of the available gain is captured.**
  - Every deployable policy sits about 0.075 Brier above oracle_h.
  - corr(score, h) is only 0.08–0.11 for all four scored policies: the realised improvement is barely predictable from the image side.
- **Cheap uncertainty is a strong baseline.**  1 − max a matches the learned gain policies (0.4086 vs 0.4079 / 0.4076; 0.4060 vs 0.4061 / 0.4053).
  This contrast was not pre-registered.
- **Even exact z ranks far worse than exact h** (oracle_z 0.397 vs oracle_h 0.332).
  - The disagreement says where b and a differ, not which of them is right.
  - Its mean overstates the realised gain (mean z 0.074 vs mean h 0.050, Brier family).
- **Deployable CAL thresholds** give realised DEV rates close to nominal (0.09–0.10 at 10 %, 0.24–0.25 at 25 %, 0.48–0.49 at 50 %,
  0.73–0.74 at 75 %), with the same ordering as exact-rate routing.
- **Identity check:** h − z = 2 (b − a)·(e − b) holds to 6.6e-16.
- **Costs** are microseconds per sample (batch 1: never 0.18–0.21 ms, always 0.44–0.49 ms).  At batch 1 the policy MLP costs about as much as the
  audio branch it can skip.  As pre-registered, **AV-MNIST supports no cost claim**.

## 5. Caveats
- AV-MNIST's audio is a noisy-by-construction pairing (clip purity 0.44, 6 shared speakers).  Audio gains here are evidence about that pairing, not
  about spoken-digit recognition in general (frozen wording).
- One small architecture.  Policy differences are small (0.002–0.003 Brier) next to the oracle gap.
- The two-way interval is the primary; the narrower seed-paired t intervals ignore DEV sampling.

## 6. Next
- **CREMA-D,** the cost-bearing dataset of GAIN-P1, is running: frozen protocol `R6_GAIN_P1_CREMAD_PROTOCOL_FROZEN_20261012.yaml`, bundles
  1034825 / 1034826, aggregate 1034827.  There the video branch costs about 21 times the audio branch per clip.
