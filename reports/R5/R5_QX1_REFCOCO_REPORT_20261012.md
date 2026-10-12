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
