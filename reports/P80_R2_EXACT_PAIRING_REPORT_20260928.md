# P80 report — R2 with exact pairing + cross-fitted R3 (repairs P72 §1) (2026-09-28)

Pre-registration `P79_R2_EXACT_PAIRING_PREREG_FROZEN_20260928.md` (frozen 15:50:28Z, written after seeing P72 — disclosed there); GPU job 1013076
(exit 0, 59 min); tables `P80_r2_exact.{md,json}` (results-only commit before this report).  Positive = the image's own caption (0–3 train, 4 held
out); SRC-CAL selection on exact pairs; same injection seeds as P71; 3 seeds.

## Verdicts
| line | verdict | basis |
|---|---|---|
| **R2** noisy pairing | **refuted** (both shifts, both targets) | VCS loses the *most* clean R@1 from m = 0 to 0.4 |
| **R3** mismatch identification | **not informative under exact pairing** (all 4 cells saturated) | raw CLIP AUROC 0.990–0.997 ≥ 0.98 in every cell (pre-committed saturation clause) |

## R2 (R@1 loss m = 0 → 0.4, points; positive = loss)
| shift | target | vcs | infonce | logistic | js |
|---|---|---|---|---|---|
| animal | SRC-EVAL | **13.1** | 1.9 | 2.0 | −0.7 |
| animal | TGT-EVAL | **11.6** | 0.8 | −0.2 | −0.2 |
| indoor → outdoor | SRC-EVAL | **8.0** | 0.3 | 1.5 | −1.8 |
| indoor → outdoor | TGT-EVAL | **4.6** | −1.4 | 0.5 | −1.5 |

VCS's clean retrieval degrades steadily with the mismatch fraction (animal SRC 0.267 → 0.099 at m = 0.6); InfoNCE stays within 3 points of its
m = 0 value at every m.  Even clean (m = 0), VCS keeps less of CLIP's alignment than InfoNCE (0.267 vs 0.363 against raw CLIP 0.368, animal).
JS's flat curve is a floor, not robustness: its m = 0 R@1 is already 0.147 / 0.076 (animal / indoor) — the balanced logistic on K = 8 pool
negatives from the a = 10, b = −10 initialisation moves the adapters far from CLIP — so JS is not a meaningful R2 comparator in this setting.

## R3
Every trained adapter is *below* raw CLIP (0.99+) in every cell; cross-fitting raises VCS the most (0.949 → 0.985 animal m = 0.4) but VCS stays
the lowest in-training score.  With random wrong captions under exact pairing, CLIP alone identifies them almost perfectly; an informative R3 needs
structured (topic-matched) mismatches — a separate unit, not launched.

## Combined reading with P72
Under topic pairing (P72) R2 was uninterpretable and R3 was won by JS; under exact pairing (P80) R2 is refuted and R3 is saturated.  The R line
gives no robustness advantage to VCS on frozen-CLIP adapters.
