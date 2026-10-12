# R5-QX1 — query-normalized critic and distribution softmax on RefCOCO — report — 2026-10-12

Protocol `reports/R5/R5_QX1_REFCOCO_PROTOCOL_FROZEN_20261012.yaml` (standing authorization; frozen before any of its runs; the disclosure of
the R5-X1 seeds already read is inside).  Results-only commit `f103305` (`results_qnorm_refcoco.json`, `qnorm_comparison_refcoco.md`,
`qnorm_run_manifest_refcoco.json`; aggregate job 1035208).
- **Runs:** 9 full fine-tuning runs on RefCOCO × SigLIP 2 B/16 × seeds 0–2: VQ (VCS loss on the query-normalized critic), JQ (matched JS,
  normalized) and D (distribution softmax).
- **Fixed:** recipe, roles, proposals and selection identical to R5-X1 / R5-QN / R5-30.
- **Gates:** 9 / 9 pass; none missing.
- **Controls:** R5-X1's VCS / JS / E on the same seeds.
- **Execution note:** seven of the nine runs were queued as low-priority filler and re-queued once through a gated feeder (no compute lost).

## 1. Results (DEV all-proposal image-macro Acc@0.5, %, mean ± sd over 3 seeds)
| arm | RefCOCO | Acc@0.75 | given-box Top-1 | J_recal × 100 |
|---|---|---|---|---|
| **VQ: VCS, query-normalized** | **74.17 ± 0.33** | 60.22 | 81.82 | 59.69 |
| JQ: matched JS, query-normalized | 73.26 ± 1.10 | 59.19 | 81.28 | 60.12 |
| VCS (R5-X1) | 66.60 ± 0.63 | 53.69 | 75.42 | 61.00 |
| JS (R5-X1) | 66.54 ± 0.45 | 53.51 | 74.51 | 61.73 |
| E: set-event softmax (R5-X1) | 75.45 ± 0.30 | 62.53 | 83.33 | 58.20 |
| D: distribution softmax | 75.90 ± 0.14 | 61.27 | 83.51 | 60.37 |

| contrast (paired by seed, 95 % t, df 2) | RefCOCO |
|---|---|
| **VQ − VCS (primary)** | **+7.56 [+6.82, +8.30]** |
| **VQ − E (primary)** | **−1.28 [−2.35, −0.21]** |
| JQ − JS | +6.72 [+4.58, +8.86] |
| JQ − E | −2.18 [−5.19, +0.83] |
| VQ − JQ | +0.90 [−1.07, +2.87] |
| VQ − D | −1.73 [−2.80, −0.66] |
| D − E | +0.45 [−0.59, +1.49] |

## 2. Pre-fixed readings (frozen wording)
1. **"Within-query normalization improves VCS by 7.56 points"** on RefCOCO.  Matched JS improves by 6.72.
2. **"Softmax still leads by 1.28."**
3. **"A gap closed by normalization supports the within-query competition explanation; it is not a new estimator (KLIEP normalization of the
   critic)"** — normalization closes 86 % of the RefCOCO gap (7.56 of 8.84).
4. **D − E: "the interval contains 0".**
5. **Three-dataset pooled values are descriptive only.**

## 3. The decomposition on three datasets (exact, descriptive; each piece with its own paired interval in the R5-QN / R5-QX1 results)
VCS − E = (VCS − VQ) + (VQ − D) + (D − E)

| piece | RefCOCOg | RefCOCO+ | RefCOCO |
|---|---|---|---|
| total gap VCS − E | −1.85 | −3.22 | **−8.84** |
| critic class (VCS − VQ) | −1.13 | −2.22 | **−7.56** |
| criterion (VQ − D, same normalized model class and target) | −0.99 | −0.78 | **−1.73** |
| positive target (D − E) | +0.27 | −0.22 | +0.45 |
| share closed by normalization | 61 % | 69 % | **86 %** |

**Reading:**
- **The critic class carries most of the deficit, and more of it the larger the deficit.**  The independent pairwise critic does not couple a
  query's candidates.  On RefCOCO, with many same-category objects per image, that costs 7.6 points; the within-query normalization (KLIEP)
  recovers it while keeping the estimand.
- **The criterion gap is consistently 0.8–1.7 points with the interval below 0 on all three datasets.**  It is the price of the VCS criterion
  relative to log-likelihood inside the same normalized model class.
- **The positive target contributes nothing detectable.**

## 4. Secondary observations (descriptive)
- **Annotated boxes:** the RefCOCO given-box deficit also closes: VQ 81.8 against VCS 75.4 and E 83.3.  The deficit R5-X1 found on annotated boxes
  was part of the same critic-class effect.
- **Box quality:** Acc@0.75 recovers 6.5 of the 8.8-point deficit (VQ 60.2, VCS 53.7, E 62.5).
- **J_recal:** VQ's encoders move towards the softmax family (59.7 against VCS 61.0 and E 58.2), as on the other datasets.
- **Selected epochs:** VQ 8, 8, 8, about as late as VCS (8, 10, 7); D 8–9; E 5–7.

## 5. Next
- DEV scoring of the RefCOCO checkpoints (jobs 1035210–1035212) is running; the R5-20 diagnostics will follow as an addendum, with the same
  definitions as R5-20.
- Paper (owner's decision): the three-dataset decomposition is the clearest single result of the r5 line.  The normalized critic is a candidate
  method component (KLIEP-normalized VCS critic).  The remaining criterion gap is the honest limit.

## Addendum — R5-20 diagnostics on RefCOCO (results-only commit effb2c7; reports/R5/diag_refcoco/)
- **Inputs:** DEV scoring of all RefCOCO checkpoints (R5-X1 VCS / JS / E; R5-QX1 VQ / JQ / D; zero-shot).  Every rescored accuracy matches its run
  (G1 / G2 gates pass).
- **Definitions:** the R5-20 diagnostics with unchanged definitions, written to a separate directory (`--out-suffix _refcoco`).
- **Not available:** the R4-VL2 ν models were never trained on RefCOCO, so they are listed as missing.

**Nested distractors** (same query, same target box; accuracy difference × 100; 95 % image bootstrap; 1 053 images):
| contrast | k = 1 | 3 | 7 | 15 | 31 | all |
|---|---|---|---|---|---|---|
| VCS − E | −0.69 [−0.88, −0.51] | −1.51 | −2.59 | −4.17 | −6.43 | −8.88 [−10.09, −7.66] |
| **VQ − E** | **+0.00 [−0.13, +0.15]** | −0.04 [−0.31, +0.21] | −0.19 [−0.55, +0.18] | −0.49 [−0.99, +0.03] | −0.85 [−1.51, −0.19] | −1.40 [−2.23, −0.53] |
| VQ − VCS | +0.69 | +1.46 | +2.40 | +3.68 | +5.58 | +7.49 [+6.35, +8.60] |
| D − E | +0.18 | +0.32 | +0.42 | +0.43 | +0.54 | +0.85 [+0.11, +1.69] |

**Gradient allocation** (share of the invalid-candidate |gradient| on the model's own strongest wrong box):
- VCS 0.490, JS 0.480 → VQ 0.582, JQ 0.578.
- Softmaxes: D 0.643, E 0.708.
- Saturated wrong boxes: ≤ 0.6 % for every model.

**Readings (descriptive):**
- **On RefCOCO the pairwise deficit exists already against a single distractor** (−0.69, larger than RefCOCOg's −0.28 and RefCOCO+'s −0.19).
  It grows to −8.88 against all proposals.
- **The normalized critic ties set softmax up to 15 nested distractors** (intervals contain 0).  It trails only against 31 or more:
  −0.85 at 31 and −1.40 against all.
- **The same pattern holds on all three datasets.**  Normalization removes the growth of the deficit with the number of competitors and moves
  gradient onto the strongest wrong candidate.  What remains concentrates at the largest candidate sets.
