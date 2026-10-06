# P144 (MV6-C1) — dependence change under the real maps h→r→z and h→O — report — 2026-10-07

Pre-registration `P144_MV6_C1_TRUE_COMPRESSION_PREREG_FROZEN_20261006.md`; job 1024533 (CPU, exit 0); results-only commit `bf19668` (`reports/P144/visual_*.json`).
CIFAR-10, P143 feature store (paired clean / modified, FIT 10k / VAL 2k / EVAL 8k), VCS = A-P3 and SimCLR 4-view, seeds 1–2; n 3 000, 5 repeats,
300 steps; observations standardised on a 4 000-image PROBE set.

## 1. Map gate (all four encoders, all versions)
r cached vs CPU recompute ≤ 9.5e-6 (tolerance 1e-4); fp16 z vs recompute ≤ 1.2e-4 (2e-3); stored logits vs recompute ≤ 2.9e-5 (1e-4).  The
measured maps are the CPU recomputations, so A = g(B) holds exactly.

## 2. Results — primary readout (VCS objective, nested, exact Q); mean ± repeat sd over 5 repeats
| encoder / seed | cell | J_h | J_r | J_z | ΔJ h→r (D_T) | ΔJ r→z (D_T) | ΔJ h→O (D_T) |
|---|---|---|---|---|---|---|---|
| VCS / 1 | colour 0.1 | −0.0007 | −0.0007 | −0.0006 | −0.0001 ± .0002 (.0004) | −0.0001 ± .0003 (.0002) | −0.0020 ± .0043 (.0034) |
| VCS / 1 | blur 0.25 | −0.0004 | −0.0004 | −0.0005 | +0.0000 (.0001) | +0.0000 (.0000) | +0.0001 ± .0003 (.0003) |
| VCS / 1 | null | −0.0006 | −0.0006 | −0.0007 | +0.0000 (.0000) | +0.0001 ± .0001 (.0001) | +0.0001 ± .0001 (.0001) |
| VCS / 2 | colour 0.1 | −0.0007 | −0.0006 | −0.0006 | −0.0001 ± .0001 (.0002) | −0.0000 (.0000) | +0.0003 ± .0002 (.0005) |
| VCS / 2 | blur 0.25 | −0.0004 | −0.0004 | −0.0004 | +0.0000 (.0000) | +0.0000 (.0001) | +0.0002 ± .0003 (.0004) |
| VCS / 2 | null | −0.0005 | −0.0005 | −0.0005 | +0.0000 (.0000) | +0.0000 (.0000) | +0.0001 ± .0001 (.0002) |
| SimCLR / 1 | colour 0.1 | −0.0007 | −0.0007 | −0.0007 | +0.0000 (.0000) | +0.0000 (.0000) | +0.0002 ± .0001 (.0003) |
| SimCLR / 1 | blur 0.25 | −0.0007 | −0.0007 | −0.0007 | +0.0000 (.0000) | +0.0000 (.0001) | +0.0000 ± .0004 (.0004) |
| SimCLR / 1 | null | −0.0009 | −0.0009 | −0.0008 | +0.0000 ± .0001 (.0001) | −0.0001 ± .0003 (.0003) | +0.0001 ± .0001 (.0002) |
| SimCLR / 2 | colour 0.1 | −0.0009 | −0.0008 | −0.0008 | −0.0001 ± .0001 (.0002) | −0.0000 (.0000) | +0.0001 ± .0001 (.0002) |
| SimCLR / 2 | blur 0.25 | −0.0006 | −0.0006 | −0.0007 | +0.0000 (.0000) | +0.0000 (.0001) | +0.0002 ± .0003 (.0004) |
| SimCLR / 2 | null | −0.0006 | −0.0006 | −0.0006 | +0.0000 (.0000) | +0.0000 (.0000) | +0.0001 ± .0002 (.0002) |

Secondary readouts (in the JSON): JS nested gives the same picture; **independent** fits (VCS and JS) give ΔJ within ±0.003 with repeat sd up to
0.0044 and D_T 0.0011–0.0045 **in every cell including the nulls** — refit variation, not a dependence change.

## 3. Sensitivity (VCS / 1, colour 0.1)
300 vs 1 000 steps with the main seed: identical readouts (the VAL-selected step is early, 30–110, so the longer budget is never used).  Second
seed (20261005; it changes the data draws as well as the critic initialisation — disclosed): J_h −0.0005, h→r ΔJ +0.0000 (D_T .0000), h→O ΔJ
+0.0002 ± .0003 (D_T .0004) — the main seed's h→O value (−0.0020 ± .0043, D_T .0034) is not reproduced → **estimation-sensitive**.

## 4. Reading (frozen rules)
- **The dependence itself is not estimable above zero at this n and critic class**: J_h ≤ 0 in every planted cell and at the null level
  (−0.0004 to −0.0009), for both encoders, as in P118 visual on the same encoder (J(h) ≈ −0.0027 at colour 0.1).  The permutation test detects
  the colour shift at h (P143 / Table 4: power 0.58–0.60 at n 2 000) because it compares the statistic with its own null, not with zero.
- Consequently the signed decrements along h→r→z and h→O are **zero within the repeat spread** in every cell; the single non-zero primary value
  (VCS / 1, colour, h→O) is estimation-sensitive (§3).  No compression ordering between encoders or maps is supported.
- Nested D_T ≈ 0 is partly structural (the nested critic starts at the lifted coarse critic and VAL often keeps step 0); it is not evidence of
  accuracy.  ΔJ − D_T ≈ 0 is not used as a check.
- The real maps and the measurement pipeline work (map gate exact, nulls flat); what limits the result is the size of the attribute dependence
  relative to finite-sample J estimation at n 3 000.

## 5. Deviations
The prereg listed a paired bootstrap over base images; the runner records per-repeat summaries only, so the across-repeat spread (fresh base
images, fresh N draws and refits per repeat — the wider uncertainty) is reported instead.  The second sensitivity seed changes the draws as well
as the critic initialisation.

## 6. Not claimed
Any encoder retaining less nuisance; monotone loss along the head; probe accuracy as information; results at larger n or other critic classes.
