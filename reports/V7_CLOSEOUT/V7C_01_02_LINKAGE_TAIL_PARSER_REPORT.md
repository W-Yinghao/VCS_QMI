# V7C-01 / V7C-02 — element-to-utility linkage, tail errors and parser diagnostics on EvalMuse CAL / DEV (2026-10-15)

**Scope.** CAL_SELECT / CAL_PROB / DEV only: the training-side roles, with DEV already exposed during V7. No GenAI label is read.

**Inputs:**
- The 9 formal M units (M0 L1 / M1 logistic / M2 VCS × seeds 0–2).
- Their per-image readouts (`scripts/v7x/v7c_readouts.py`).
- E_auto from the frozen prompt-only parser on the 606 CAL / DEV prompts (job 1040548).
- Diagnostics: `scripts/v7x/v7c_linkage.py` (job 1040745).
- Aggregate tables: `V7C_01_02_tables/`. The per-candidate linkage table and the per-prompt tail table hold per-image human labels, so they stay in `outputs/V7_CLOSEOUT/linkage/`.

**Readouts:**
- **U_ann:** the image's mean human element label. This is an annotation-assisted reference: it uses human element labels, so it is neither deployable nor a noise-free ceiling.
- **E_gt:** the model's mean prior-corrected probability over the human elements.
- **E_auto:** the same over the parser's elements.
- **O:** native overall.
- **H_gt:** α(O − 1)/4 + (1 − α)E_gt, with α per unit fixed on CAL_SELECT.
- **Ecal_gt:** CAL_PROB-Platt-calibrated E.

**Metrics:**
- Within-prompt pairwise accuracy against the human overall mean H̄. Human-tied pairs are excluded and model ties count 1/2.
- Kendall τ-b, the selected H̄ (first-index ties) and the observed regret, computed per prompt and then prompt-macro averaged.
- Standard-size best-of-N on prompts with ≥ 9 images: one seeded 9-subset, and its three disjoint triples for N = 3.

## 1. Linkage on DEV (307 prompts; mean over the 3 seeds of each arm)
| readout | M0 L1 | M1 logistic | M2 VCS | regret (M2) |
|---|---|---|---|---|
| U_ann (annotation-assisted reference) | 0.818 | 0.818 | 0.818 | 0.152 |
| E_gt | 0.720 | 0.727 | 0.724 | 0.426 |
| E_auto | 0.709 | 0.714 | 0.710 | 0.515 |
| O | 0.729 | 0.729 | 0.727 | 0.464 |
| H_gt | 0.734 | 0.734 | **0.736** | 0.403 |
| Ecal_gt | 0.722 | 0.728 | 0.725 | 0.425 |

The first three numeric columns are pairwise accuracy. τ-b follows the same order: 0.63 for U_ann, 0.39–0.44 for the model readouts. CAL_SELECT (145 prompts) reproduces the pattern, with U_ann at 0.820 and the model readouts between 0.711 and 0.728.

**Reading.**
1. **The element-to-utility link has a ceiling.** Even the human element labels order the images only 82 % of the time (τ-b 0.63). Mean element satisfaction is therefore a strong but imperfect proxy for overall human quality.
2. **The larger loss is estimation.** E_gt is about 0.09 below U_ann. The model's element probabilities read the image less well than the human element votes do.
3. **The parser costs little.** E_auto is about 0.01 below E_gt.
4. **The direct overall score O matches E, and the frozen mixture H is the best model readout.** The loss arms do not separate: M0, M1 and M2 differ by ≤ 0.008 on every readout, consistent with Table A.
5. **The standard-size best-of-9 sample is small** (DEV: 33 prompts with ≥ 9 images and a complete E_gt; 140 with O), so it is descriptive only. E_gt M9 is 3.50–3.58, H_gt 3.58–3.63, U_ann 3.86, and the random expectation 3.10 (`standard_size_selection.csv`).

## 2. Tail errors (DEV, E_gt-selected image)
- **Element Brier:** the selected image's element Brier is slightly above the pool mean (M2, weighted over all DEV prompts: 0.177 vs 0.174), so the top-scored image is not better estimated than its pool.
- **Margin:** the top-1 to top-2 margin is small (≈ 0.044).
- **Over-prediction:** the largest positive error, E_gt − U_ann over the candidates, is ≈ 0.19.
- **By prompt complexity:** prompts in the middle FIT element-count tertile carry the largest regret (0.47–0.53, against 0.40–0.41 in the top tertile). The lowest tertile has only 24 prompt × unit rows.
- **Source:** `tail_error.csv` (by arm, element-count and prompt-length tertiles fixed on FIT).

## 3. Parser coverage (frozen prompt-only parser)
| role | prompts | ok / decomposition failed | prompts with an unresolved span | images with E_auto | auto vs human elements per image |
|---|---|---|---|---|---|
| CAL_SELECT | 145 | 144 / 1 | 11 | 1,638 / 1,653 | 4.00 vs 3.88 |
| CAL_PROB | 154 | 152 / 2 | 20 | 1,600 / 1,630 | 3.89 vs 4.45 |
| DEV | 307 | 306 / 1 | 21 | 3,242 / 3,257 | 3.93 vs 3.69 |

## 4. Label noise (vote arrays split once into two halves, seed 2718; array resampling, not independent raters)
| role | 3-vote images (1 vs 2) | 6-vote images (3 vs 3) |
|---|---|---|
| CAL_SELECT | 0.773 (140 prompts) | 0.420 (71) |
| CAL_PROB | 0.785 (152) | 0.501 (67) |
| DEV | 0.790 (297) | 0.456 (154) |

The figures are within-prompt pair agreement between the two halves' means. The 6-vote images come from the re-annotation step, i.e. disputed images. Their two halves order candidates almost at random, so pairs involving them carry little reliable signal, for every method alike. No image is filtered on this basis.

## 5. Not claimed
- No GenAI result.
- U_ann is not an upper bound.
- The GT-vs-auto difference mixes the change of element list with the model's response to a new text interface; it is not a pure parser effect.
- Strata are descriptive.
- Nothing here changes a frozen readout, α, parser, calibrator or denominator.
