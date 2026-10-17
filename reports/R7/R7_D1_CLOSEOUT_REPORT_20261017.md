# R7 D1 — closeout report (2026-10-17)

**Status:** closes r7 D1 (question-conditioned visual-token selection; host LLaVA-1.5-7B, second host Qwen2.5-VL-7B).  D1 is frozen after REV10, as
recorded in `reports/OWNER_DECISIONS_20261015.md` line 20 (review item forwarded by the owner: "do not run; finish REV10, then freeze D1"; status
"no new D1 unit; REV10 evals running → score → closeout report").  REV10's results-only commit is 241a2c9.  Nothing was run for this report: no
training, evaluation or scoring job, and no label or test read.  §5 is also the report owed by REV10's protocol ("results-only commit to
reports/REV9/REV10_RESULTS/, then a report").

**Numbers.**  Every table, and every result quoted in the text, is printed by `scripts/r7_d1_closeout_tables.py` from the 14 committed results
files of §9, each added by a results-only commit and not edited since; design facts (budgets, step counts, replicate counts, rule texts) are quoted
from the freeze files.  The script formats, counts, takes seed means / sd / minima / maxima and re-applies the frozen decision rules (DF-8, E2-8,
the REV10 rule) to the stored deltas and Holm p-values.  It does no resampling and computes no new test or p-value.
`python3 scripts/r7_d1_closeout_tables.py --check reports/R7/R7_D1_CLOSEOUT_REPORT_20261017.md` confirms that each of its tables appears here
verbatim.  Deltas are accuracy points; [95 % CI] is the stored image- (chart-) grouped paired bootstrap (2,000 replicates); "p" is
the stored Holm-adjusted p of a primary contrast, or the stored unadjusted bootstrap p where a table says "secondary" or "unadjusted"; "seeds < 0"
counts the per-seed deltas below zero.

**Rules that govern this closeout** (quoted from the freeze files; line numbers checked):
- **D1's pre-registered refutation rule**, `reports/R7/R7_D1_PROTOCOL_FROZEN_20261012.yaml:257-260`, in the file since the protocol freeze
  2424556 (2026-10-12, before any quality result) and unchanged since: "S1 not better than S3 (direct risk) at equal information -> the gain is
  the selector, not the divergence; S1 not better than SparseVLM / uniform at matched latency -> no practical advantage; D1-10 shows h − d large
  and unpredictable -> the squared-gap read-out is not a risk estimate; the regime rule finds no saving -> no acceleration claim."
- **Claim rules of the confirmatory reads.**  LLaVA, DF-8 (`R7_D1_FULL_STAGED_FREEZE_20261012.yaml:21-24`): a claim only where the 10k-FIT
  (D1-30) and the full-FIT (D1-FULL) results agree in sign and both pass their Holm-corrected primary tests.  Qwen, E2-8
  (`R7_E2_STAGED_FREEZE_20261013.yaml:16-17`): the same between R7-40c (10k) and E2 (full).  Official test, R7-50: Holm per phase and latency
  point over its 6 primary contrasts (`R7_50_PROTOCOL_FROZEN_20261012.yaml:159`); phase L vs phase F is descriptive only (F4, line 305).
- **Mechanism units.**  REV9: "a positive effect is a mechanism finding for this scale, not a claim that the selector beats SparseVLM or the
  grid" (`REV9_PROTOCOL_FROZEN_20261014.yaml:27`).  REV10: pre-fixed decision rule, "otherwise MIXED (reported descriptively, no attribution)"
  (`REV10_LR_FACTORIAL_PROTOCOL_FROZEN_20261015.yaml:29-32`).
- No equivalence or non-inferiority claim (`R7_D1_30_STAGED_FREEZE_20261012.yaml:103`; `R7_50_PROTOCOL_FROZEN_20261012.yaml:170`).

Notation: S1 = VCS-trained selector; S2 = its logistic sibling (same architecture, same read-out r(K) = (T_B − T_A(K))²); S3 = direct-risk
regressor (same backbone, regresses the teacher's realised h); seed means over seeds 0–2.  LLaVA: L25 / L50 / L75 = quality at the uniform
grid's 144 / 288 / 432-token latency at B* = 4.  Qwen has no B*: fixed fractions f 0.25 / 0.5 / 0.75 of the merged visual tokens.

## 1. Closing statement

- **The pre-registered D1 hypothesis is refuted by its own rule (clause 2: no practical advantage).**  On LLaVA-1.5-7B, S1 is below
  budget-matched SparseVLM at L25 and L50 on GQA DEV, TextVQA val and ChartQA val at both FIT scales (6 of the 9 DF-8 claims) and on GQA testdev
  and ChartQA test in both official-test phases; every seed is negative in each of these 20 cells, and no S1 − SparseVLM contrast is detected in
  S1's favour anywhere (T1).  On Qwen2.5-VL-7B, where the uniform grid is the primary external row, 7 of the 9 S1 − grid contrasts are E2-8
  claims, all "S1 below", and S1 is below the grid in all 6 official-test cells (T3).
- **Clause 1 holds on GQA and ChartQA, not on TextVQA.**  Under the two-scale claim rules S1 is better than S3 only on TextVQA (DF-8 claims at
  L25 / L50 / L75; E2-8 claims at f 0.25 / 0.5), a transfer benchmark for selectors trained on GQA yes / no risk.  On in-distribution GQA, Qwen has
  E2-8 claims "S1 below" at all three fractions and LLaVA has no claim; on ChartQA neither host has a claim (LLaVA's 10k-FIT lead at L25 / L50
  changes sign at full FIT) (T4, T5).  With clause 2 fired there is no gain for clause 1 to attribute.
- **Clause 3 did not fire.**  The frozen D1-10 readout judged h − d "not large and unpredictable": the residual has a stable negative sign, and d
  over-states the realised risk change at small budgets (T6).  The host's squared-gap read-out is not refuted as a risk estimate.
- **Clause 4 did not fire.**  The regime rule found a saving (B* = 4).  No acceleration claim is made all the same: nothing has a quality
  advantage at matched latency, SparseVLM's official code is batch-1 only, and Qwen has no B*.
- **VCS against its matched logistic sibling: no ordering.**  No DF-8 claim at any of the 9 LLaVA points (6 scale-dependent, 3 failing Holm at
  one or both scales); on Qwen the only E2-8 claims are "S1 below" on ChartQA.  On the official tests the 10k-FIT S1 is below S2 wherever a
  difference is detected, and the full-FIT LLaVA S1 has two detected cells of opposite sign, each carried mainly by one seed (T7, T8).  This is
  not an equivalence statement.
- **Phase-B mechanics move delivered accuracy, and are not attributed.**  With the critic held identical (phase B never updates the estimator;
  every I0 guard exact), the eval-mode critic and straight-through hard top-k move DEV / val accuracy by −10.52 to +4.28 points in
  Holm-significant cells, in a direction that depends on objective × phase-A learning rate × FIT scale.  REV10's frozen rule returned MIXED, so the 10k pattern is attributed neither to the objective
  nor to the learning rate (§5).
- **Neither closing label of `reports/PROJECT_MAP_20261014.md` line 237 is used** (§4): its positive condition holds at one scale and reverses at
  the other, and its negative label ("后验读数不适合候选选择") claims more than the data show.

## 2. The refutation rule, clause by clause

| clause (frozen text) | read on | outcome |
|---|---|---|
| S1 not better than SparseVLM / uniform at matched latency -> no practical advantage | LLaVA: S1 − SparseVLM, primary on the budget-matched basis D30-1 (T1); S1 − uniform grid, secondary at matched latency (T2).  Qwen: S1 − uniform grid, primary (T3) | **fired** on both hosts, both FIT scales, DEV / val and official test |
| S1 not better than S3 (direct risk) at equal information -> the gain is the selector, not the divergence | S1 − S3, primary on both hosts (T4, T5) | **holds on GQA and ChartQA, not on TextVQA**; no gain to attribute (clause 2) |
| D1-10 shows h − d large and unpredictable -> the squared-gap read-out is not a risk estimate | frozen D1-10 CAL readout under the interpretation grid of protocol lines 138-143 (T6) | **did not fire** |
| the regime rule finds no saving -> no acceleration claim | regime rule, protocol lines 67-77, decision OPEN-2 (lines 297-299) | **did not fire** (B* = 4); no acceleration claim for the reasons in §2.4 |

### 2.1 Clause 2 — S1 against SparseVLM and the uniform grid

**T1 — S1 − SparseVLM on LLaVA** (budget-matched, D30-1: S1 at k tokens against the adapted SparseVLM row at k = 144 / 288 / 432; primary in every
LLaVA family; TextVQA has no official-test read).

| benchmark | point | 10k FIT, DEV / val (D1-30) | full FIT, DEV / val (D1-FULL) | DF-8 | 10k FIT, official test (R7-50 L) | full FIT, official test (R7-50 F) | seeds < 0: 10k / full / L / F |
|---|---|---|---|---|---|---|---|
| gqa_dev | L25 | -4.12 [-4.81, -3.42], p 0.009 | -3.59 [-4.26, -2.93], p 0.009 | claim: S1 lower | -4.03 [-4.73, -3.30], p 0.006 | -3.06 [-3.72, -2.37], p 0.006 | 3/3 / 3/3 / 3/3 / 3/3 |
| gqa_dev | L50 | -1.09 [-1.63, -0.56], p 0.009 | -1.19 [-1.71, -0.68], p 0.009 | claim: S1 lower | -1.59 [-2.04, -1.14], p 0.006 | -1.47 [-1.96, -0.99], p 0.006 | 3/3 / 3/3 / 3/3 / 3/3 |
| gqa_dev | L75 | -0.06 [-0.40, +0.30], p 1.000 | -0.18 [-0.54, +0.18], p 1.000 | no claim (Holm) | -0.05 [-0.40, +0.27], p 1.000 | -0.01 [-0.35, +0.32], p 1.000 | 2/3 / 3/3 / 2/3 / 2/3 |
| textvqa_val | L25 | -16.97 [-17.97, -15.99], p 0.009 | -19.59 [-20.60, -18.55], p 0.009 | claim: S1 lower | — | — | 3/3 / 3/3 / — / — |
| textvqa_val | L50 | -7.52 [-8.25, -6.81], p 0.009 | -11.11 [-11.97, -10.27], p 0.009 | claim: S1 lower | — | — | 3/3 / 3/3 / — / — |
| textvqa_val | L75 | -0.18 [-0.62, +0.21], p 1.000 | -1.09 [-1.53, -0.66], p 0.009 | no claim (Holm) | — | — | 2/3 / 3/3 / — / — |
| chartqa_val | L25 | -2.60 [-3.66, -1.45], p 0.009 | -3.59 [-4.67, -2.42], p 0.009 | claim: S1 lower | -1.65 [-2.53, -0.87], p 0.006 | -1.99 [-2.87, -1.17], p 0.006 | 3/3 / 3/3 / 3/3 / 3/3 |
| chartqa_val | L50 | -2.00 [-2.92, -1.12], p 0.009 | -2.62 [-3.59, -1.63], p 0.009 | claim: S1 lower | -0.88 [-1.56, -0.21], p 0.039 | -1.49 [-2.22, -0.73], p 0.006 | 3/3 / 3/3 / 3/3 / 3/3 |
| chartqa_val | L75 | +0.14 [-0.52, +0.79], p 1.000 | -0.10 [-0.73, +0.54], p 1.000 | scale-dependent sign | +0.04 [-0.43, +0.53], p 1.000 | -0.13 [-0.62, +0.36], p 1.000 | 1/3 / 2/3 / 1/3 / 3/3 |

- At L25 and L50, S1 is Holm-detected below SparseVLM in all 20 cells (3 benchmarks × 2 FIT scales on DEV / val, 2 benchmarks × 2 phases on the
  official test), with 3 / 3 seeds negative in each.  At L75 the only detected cell is full-FIT TextVQA (−1.09, S1 lower).  No cell is detected in
  S1's favour.
- The latency-matched S1 − SparseVLM contrast at B = 1 (secondary) was not reached in any of its 30 cells, and no latency claim is made against
  SparseVLM at B* = 4 (official code batch-1 only; D30-1, `R7_D1_30_STAGED_FREEZE_20261012.yaml:162`).

**T2 — S1 − uniform grid on LLaVA** (secondary at matched latency, B* = 4; unadjusted bootstrap p, no Holm).

| benchmark | point | 10k FIT, DEV / val (D1-30) | full FIT, DEV / val (D1-FULL) | 10k FIT, official test (R7-50 L) | full FIT, official test (R7-50 F) |
|---|---|---|---|---|---|
| gqa_dev | L25 | -4.04 [-4.67, -3.40], p 0.001 | -3.32 [-3.94, -2.68], p 0.001 | -4.91 [-5.56, -4.27], p 0.001 | -3.63 [-4.27, -3.02], p 0.001 |
| gqa_dev | L50 | -1.20 [-1.71, -0.69], p 0.001 | -1.21 [-1.75, -0.65], p 0.001 | -1.79 [-2.26, -1.32], p 0.001 | -1.55 [-2.01, -1.05], p 0.001 |
| gqa_dev | L75 | +0.13 [-0.25, +0.52], p 0.504 | +0.01 [-0.38, +0.42], p 0.894 | +0.19 [-0.14, +0.51], p 0.276 | +0.24 [-0.10, +0.58], p 0.190 |
| textvqa_val | L25 | -5.67 [-6.64, -4.69], p 0.001 | -8.31 [-9.27, -7.32], p 0.001 | — | — |
| textvqa_val | L50 | -1.97 [-2.84, -1.10], p 0.001 | -5.54 [-6.41, -4.69], p 0.001 | — | — |
| textvqa_val | L75 | +1.51 [+0.89, +2.17], p 0.001 | +0.41 [-0.22, +1.11], p 0.197 | — | — |
| chartqa_val | L25 | -1.29 [-2.24, -0.33], p 0.005 | -2.24 [-3.20, -1.27], p 0.001 | -0.39 [-1.14, +0.34], p 0.298 | -0.64 [-1.39, +0.08], p 0.094 |
| chartqa_val | L50 | +0.08 [-0.82, +1.00], p 0.828 | -0.61 [-1.49, +0.27], p 0.191 | -0.68 [-1.44, +0.04], p 0.068 | -1.24 [-2.01, -0.48], p 0.001 |
| chartqa_val | L75 | +0.20 [-0.55, +1.00], p 0.578 | -0.13 [-0.89, +0.66], p 0.795 | +0.07 [-0.61, +0.68], p 0.859 | -0.21 [-0.89, +0.43], p 0.511 |

- At L25 the point estimate is negative in all 10 cells (unadjusted p ≤ 0.005 in 8; ChartQA test p 0.298 in phase L and 0.094 in phase F).
- S1 is above the grid at one point only, 10k-FIT TextVQA val at L75 (+1.51 [+0.89, +2.17], unadjusted), where it is not above SparseVLM (T1:
  −0.18, p 1.000); at full FIT that cell is +0.41 (p 0.197).  So no matched-latency point has S1 better than both external rows.

**T3 — S1 − uniform grid on Qwen** (primary; E2-8 on DEV / val; R7-50 phase Q on the official test, 10k-FIT selectors only).

| benchmark | fraction | 10k FIT, DEV / val (R7-40c) | full FIT, DEV / val (E2) | E2-8 | 10k FIT, official test (R7-50 Q) | seeds < 0: 10k / full / Q |
|---|---|---|---|---|---|---|
| gqa_dev | f0.25 | -4.47 [-5.10, -3.81], p 0.009 | -4.72 [-5.42, -4.07], p 0.009 | claim: S1 below | -4.81 [-5.44, -4.15], p 0.006 | 3/3 / 3/3 / 3/3 |
| gqa_dev | f0.5 | -3.43 [-4.04, -2.83], p 0.009 | -3.07 [-3.67, -2.51], p 0.009 | claim: S1 below | -2.89 [-3.41, -2.33], p 0.006 | 3/3 / 3/3 / 3/3 |
| gqa_dev | f0.75 | -2.00 [-2.45, -1.55], p 0.009 | -1.68 [-2.10, -1.26], p 0.009 | claim: S1 below | -1.19 [-1.59, -0.78], p 0.006 | 3/3 / 3/3 / 3/3 |
| textvqa_val | f0.25 | -7.17 [-8.25, -6.15], p 0.009 | -1.83 [-2.88, -0.75], p 0.009 | claim: S1 below | — | 3/3 / 3/3 / — |
| textvqa_val | f0.5 | -2.42 [-3.20, -1.66], p 0.009 | +0.26 [-0.53, +1.04], p 0.511 | 10k only | — | 3/3 / 1/3 / — |
| textvqa_val | f0.75 | -1.10 [-1.63, -0.60], p 0.009 | -0.49 [-1.04, +0.04], p 0.158 | 10k only | — | 3/3 / 3/3 / — |
| chartqa_val | f0.25 | -17.26 [-19.35, -15.16], p 0.009 | -15.47 [-17.63, -13.26], p 0.009 | claim: S1 below | -16.25 [-18.09, -14.52], p 0.006 | 3/3 / 3/3 / 3/3 |
| chartqa_val | f0.5 | -13.80 [-15.46, -12.12], p 0.009 | -11.72 [-13.42, -10.00], p 0.009 | claim: S1 below | -13.73 [-15.19, -12.34], p 0.006 | 3/3 / 3/3 / 3/3 |
| chartqa_val | f0.75 | -4.62 [-5.85, -3.44], p 0.009 | -3.09 [-4.30, -1.88], p 0.009 | claim: S1 below | -5.48 [-6.57, -4.47], p 0.006 | 3/3 / 3/3 / 3/3 |

- 7 of the 9 contrasts are E2-8 claims, all "S1 below"; the other two (TextVQA f 0.5 / 0.75) are significant at the 10k scale only.  On the
  official test S1 is below the grid in all 6 cells, every seed negative.

### 2.2 Clause 1 — S1 against S3

**T4 — S1 − S3 on LLaVA** (primary).

| benchmark | point | 10k FIT, DEV / val (D1-30) | full FIT, DEV / val (D1-FULL) | DF-8 | 10k FIT, official test (R7-50 L) | full FIT, official test (R7-50 F) | seeds < 0: 10k / full / L / F |
|---|---|---|---|---|---|---|---|
| gqa_dev | L25 | -1.19 [-1.67, -0.71], p 0.009 | +0.38 [-0.17, +0.92], p 0.724 | scale-dependent sign | -0.99 [-1.49, -0.46], p 0.006 | +2.26 [+1.65, +2.84], p 0.006 | 3/3 / 2/3 / 2/3 / 1/3 |
| gqa_dev | L50 | -0.64 [-1.03, -0.27], p 0.009 | -0.49 [-0.91, -0.04], p 0.120 | no claim (Holm) | -0.15 [-0.52, +0.22], p 0.457 | +0.05 [-0.40, +0.48], p 0.963 | 3/3 / 2/3 / 2/3 / 2/3 |
| gqa_dev | L75 | -0.11 [-0.31, +0.09], p 1.000 | -0.26 [-0.51, -0.01], p 0.329 | no claim (Holm) | -0.03 [-0.21, +0.15], p 1.000 | -0.02 [-0.26, +0.23], p 1.000 | 2/3 / 2/3 / 2/3 / 2/3 |
| textvqa_val | L25 | +8.76 [+8.09, +9.50], p 0.009 | +1.18 [+0.51, +1.81], p 0.010 | claim: S1 higher | — | — | 0/3 / 1/3 / — / — |
| textvqa_val | L50 | +9.11 [+8.37, +9.84], p 0.009 | +1.24 [+0.50, +1.98], p 0.015 | claim: S1 higher | — | — | 0/3 / 1/3 / — / — |
| textvqa_val | L75 | +3.63 [+3.25, +4.03], p 0.009 | +3.46 [+2.97, +3.96], p 0.009 | claim: S1 higher | — | — | 0/3 / 1/3 / — / — |
| chartqa_val | L25 | +1.32 [+0.75, +1.90], p 0.009 | -0.19 [-0.82, +0.47], p 1.000 | scale-dependent sign | +1.07 [+0.62, +1.55], p 0.006 | +0.09 [-0.42, +0.57], p 0.736 | 0/3 / 1/3 / 0/3 / 1/3 |
| chartqa_val | L50 | +0.81 [+0.27, +1.36], p 0.014 | -0.64 [-1.30, +0.11], p 0.276 | scale-dependent sign | +0.76 [+0.28, +1.18], p 0.012 | -0.20 [-0.71, +0.32], p 0.963 | 0/3 / 2/3 / 1/3 / 2/3 |
| chartqa_val | L75 | +0.34 [+0.00, +0.67], p 0.306 | -0.11 [-0.55, +0.35], p 1.000 | scale-dependent sign | +0.25 [-0.00, +0.54], p 0.270 | -0.09 [-0.47, +0.29], p 1.000 | 0/3 / 1/3 / 0/3 / 2/3 |

**T5 — S1 − S3 on Qwen** (primary).

| benchmark | fraction | 10k FIT, DEV / val (R7-40c) | full FIT, DEV / val (E2) | E2-8 | 10k FIT, official test (R7-50 Q) | seeds < 0: 10k / full / Q |
|---|---|---|---|---|---|---|
| gqa_dev | f0.25 | -1.24 [-1.74, -0.76], p 0.009 | -3.89 [-4.51, -3.28], p 0.009 | claim: S1 below | -0.55 [-1.02, -0.08], p 0.052 | 3/3 / 3/3 / 2/3 |
| gqa_dev | f0.5 | -1.47 [-1.89, -1.05], p 0.009 | -2.84 [-3.35, -2.31], p 0.009 | claim: S1 below | -0.60 [-1.03, -0.17], p 0.018 | 3/3 / 3/3 / 2/3 |
| gqa_dev | f0.75 | -1.16 [-1.51, -0.81], p 0.009 | -1.47 [-1.85, -1.10], p 0.009 | claim: S1 below | -0.25 [-0.60, +0.07], p 0.130 | 3/3 / 3/3 / 2/3 |
| textvqa_val | f0.25 | +8.55 [+7.68, +9.45], p 0.009 | +29.80 [+28.60, +30.96], p 0.009 | claim: S1 above | — | 1/3 / 0/3 / — |
| textvqa_val | f0.5 | +2.87 [+2.24, +3.49], p 0.009 | +14.82 [+13.93, +15.75], p 0.009 | claim: S1 above | — | 1/3 / 0/3 / — |
| textvqa_val | f0.75 | +0.37 [-0.04, +0.81], p 0.312 | +3.07 [+2.52, +3.60], p 0.009 | full only | — | 1/3 / 0/3 / — |
| chartqa_val | f0.25 | +0.66 [-0.80, +2.20], p 0.400 | +1.79 [-0.04, +3.67], p 0.094 | neither scale | -0.91 [-2.21, +0.50], p 0.193 | 1/3 / 2/3 / 1/3 |
| chartqa_val | f0.5 | -0.99 [-2.46, +0.48], p 0.350 | +3.16 [+1.31, +4.94], p 0.009 | full only | -1.19 [-2.51, +0.08], p 0.076 | 1/3 / 0/3 / 1/3 |
| chartqa_val | f0.75 | -0.42 [-1.45, +0.56], p 0.786 | +1.51 [+0.38, +2.71], p 0.048 | full only | -1.04 [-1.96, -0.09], p 0.060 | 1/3 / 1/3 / 2/3 |

- Under the two-scale rules S1 is above S3 only on TextVQA: LLaVA at L25 / L50 / L75 and Qwen at f 0.25 / 0.5 (claims).  At full FIT one of the
  three LLaVA seeds is negative at each TextVQA point.
- On GQA, the in-distribution benchmark, S1 is below S3 on Qwen (claims at all three fractions) and has no claim on LLaVA (scale-dependent at
  L25, Holm-limited at L50 / L75).  On ChartQA neither host has a claim.
- Official tests: the detected cells point both ways — phase L ChartQA L25 / L50 S1 above (+1.07, +0.76) and GQA L25 S1 below (−0.99); phase F
  GQA L25 S1 above (+2.26, 1 of 3 seeds negative; the phase F report traces it to one weak S3 seed); phase Q GQA f 0.5 S1 below (−0.60).
- Under the clause: S1 is not shown better than S3 on GQA and ChartQA.  The clause's consequence ("the gain is the selector, not the
  divergence") presupposes a gain over the external rows, and clause 2 shows there is none.

### 2.3 Clause 3 — the teacher-level read-out (D1-10)

**T6 — D1-10 CAL readout** (the host's own squared gap d = (b − a)² against the realised risk change h; all 7,989 CAL questions; frozen readout
`reports/R7/R7_D1_10_CAL_READOUT.md`).

| mechanism | kept | mean h | mean d | mean h − d [95 % CI] | abs(h − d) / h | per-item Spearman(h, d) | calibrated-teacher band | rank rule |
|---|---|---|---|---|---|---|---|---|
| grid | 12.5 % | 0.1579 | 0.1672 | -0.0093 [-0.0220, +0.0032] | 0.06 | 0.472 | [0.463, 0.500] | does not fire |
| grid | 25 % | 0.0956 | 0.0993 | -0.0037 [-0.0141, +0.0067] | 0.04 | 0.402 | [0.380, 0.422] | does not fire |
| grid | 50 % | 0.0393 | 0.0430 | -0.0037 [-0.0112, +0.0034] | 0.10 | 0.249 | [0.231, 0.274] | does not fire |
| grid | 75 % | 0.0210 | 0.0205 | +0.0004 [-0.0048, +0.0055] | 0.02 | 0.194 | [0.173, 0.217] | does not fire |
| random | 12.5 % | 0.1616 | 0.1808 | -0.0192 [-0.0333, -0.0063] | 0.12 | 0.455 | [0.461, 0.502] | does not fire |
| random | 25 % | 0.0901 | 0.1017 | -0.0116 [-0.0224, -0.0007] | 0.13 | 0.384 | [0.378, 0.423] | does not fire |
| random | 50 % | 0.0436 | 0.0475 | -0.0039 [-0.0119, +0.0041] | 0.09 | 0.268 | [0.244, 0.296] | does not fire |
| random | 75 % | 0.0158 | 0.0189 | -0.0031 [-0.0082, +0.0019] | 0.20 | 0.172 | [0.154, 0.201] | does not fire |
| qsim | 12.5 % | 0.1410 | 0.1697 | -0.0287 [-0.0421, -0.0158] | 0.20 | 0.396 | [0.419, 0.461] | does not fire |
| qsim | 25 % | 0.0854 | 0.1069 | -0.0215 [-0.0335, -0.0104] | 0.25 | 0.317 | [0.336, 0.373] | does not fire |
| qsim | 50 % | 0.0440 | 0.0503 | -0.0064 [-0.0140, +0.0014] | 0.14 | 0.214 | [0.208, 0.256] | does not fire |
| qsim | 75 % | 0.0141 | 0.0193 | -0.0052 [-0.0103, -0.0000] | 0.37 | 0.107 | [0.104, 0.151] | does not fire |

- The frozen interpretation grid (protocol lines 138-143), as applied in the readout (lines 113-128): grid line 1 (|h − d| ≤ 0.1 · h with the CI
  covering 0) holds for grid masks only; grid line 2 (consistent sign, CI excluding 0) applies to random and qsim masks, i.e. d over-states h at
  small budgets and is biased; the per-item rank rule (calibrated-teacher band, coordinator note added before the readout) fires in 0 of 12 CAL
  cells; Package §F: the residual is "not 'large and unpredictable'".  Clause 3 did not fire.
- Recomputed from the results file: the residual is negative in 11 of 12 cells and at most 0.0287 in absolute value; |h − d| / h is ≤ 0.25 in 10
  cells, 0.252 at qsim 25 % and 0.367 at qsim 75 % (on a small h, 0.0141).  The readout's sentence "its magnitude is at most about 25 % of h"
  (line 128) therefore misses the qsim 75 % cell; the sign there is negative as elsewhere.
- The calibrated-teacher band is a reference generated under the teacher's own probabilities, not a bound for every method (readout
  clarification, lines 140-143).

### 2.4 Clause 4 — the regime rule

- The rule (protocol lines 67-71): B* is the smallest batch in {1, 4, 16, 32} at which the uniform grid at 25 % cuts per-item latency by at least
  25 %.  It was applied once, to the D1-10 timing smoke before any quality result: 9.4 % at B = 1, 27.2 % at B = 4, so B* = 4 (protocol lines
  73-77 and OPEN-2, lines 297-299; `R7_D1_10_SMOKE.md:67-72`).  The smoke's timing file is outside the repository
  (`outputs/R7_D1/smoke/timing/timing_report.json`); the script re-derives 9.4 % and 27.2 % from it when it is present.  At B* = 4 the D1-30
  timing runs (descriptive; the rule is not re-applied) give the uniform grid at 144 tokens a median per-item latency 26.6 % (GQA DEV), 24.0 %
  (TextVQA val) and 26.3 % (ChartQA val) below the full row.
- The clause did not fire, but no acceleration claim is made: clause 2 leaves no method of ours with a quality advantage at matched latency;
  SparseVLM has no B* = 4 latency (D30-1); Qwen has no B* (R7-40c basis, E2-13).

## 3. VCS against the matched logistic sibling (S1 − S2)

**T7 — S1 − S2 on LLaVA** (primary).

| benchmark | point | 10k FIT, DEV / val (D1-30) | full FIT, DEV / val (D1-FULL) | DF-8 | 10k FIT, official test (R7-50 L) | full FIT, official test (R7-50 F) | seeds < 0: 10k / full / L / F |
|---|---|---|---|---|---|---|---|
| gqa_dev | L25 | -2.86 [-3.36, -2.34], p 0.009 | -0.06 [-0.48, +0.37], p 1.000 | no claim (Holm) | -3.24 [-3.76, -2.66], p 0.006 | -0.78 [-1.16, -0.40], p 0.006 | 3/3 / 2/3 / 3/3 / 2/3 |
| gqa_dev | L50 | -1.23 [-1.59, -0.87], p 0.009 | +0.00 [-0.34, +0.36], p 1.000 | scale-dependent sign | -1.46 [-1.83, -1.09], p 0.006 | -0.27 [-0.56, +0.04], p 0.312 | 3/3 / 2/3 / 3/3 / 1/3 |
| gqa_dev | L75 | -0.19 [-0.35, -0.03], p 0.189 | -0.05 [-0.24, +0.15], p 1.000 | no claim (Holm) | -0.26 [-0.43, -0.10], p 0.012 | +0.08 [-0.08, +0.23], p 1.000 | 3/3 / 2/3 / 3/3 / 1/3 |
| textvqa_val | L25 | -4.37 [-5.05, -3.65], p 0.009 | +2.01 [+1.47, +2.56], p 0.009 | scale-dependent sign | — | — | 2/3 / 1/3 / — / — |
| textvqa_val | L50 | -2.10 [-2.67, -1.52], p 0.009 | +2.19 [+1.64, +2.77], p 0.009 | scale-dependent sign | — | — | 2/3 / 1/3 / — / — |
| textvqa_val | L75 | -0.37 [-0.60, -0.15], p 0.024 | +0.05 [-0.23, +0.36], p 1.000 | scale-dependent sign | — | — | 2/3 / 2/3 / — / — |
| chartqa_val | L25 | -1.13 [-1.67, -0.59], p 0.009 | -0.10 [-0.55, +0.35], p 1.000 | no claim (Holm) | -0.87 [-1.37, -0.41], p 0.006 | +0.75 [+0.34, +1.17], p 0.006 | 2/3 / 2/3 / 2/3 / 1/3 |
| chartqa_val | L50 | -0.62 [-1.13, -0.14], p 0.017 | +0.18 [-0.38, +0.70], p 1.000 | scale-dependent sign | -0.32 [-0.74, +0.11], p 0.254 | +0.21 [-0.21, +0.64], p 0.963 | 2/3 / 2/3 / 1/3 / 2/3 |
| chartqa_val | L75 | +0.00 [-0.29, +0.31], p 1.000 | -0.04 [-0.37, +0.26], p 1.000 | scale-dependent sign | +0.12 [-0.09, +0.34], p 1.000 | -0.15 [-0.40, +0.09], p 1.000 | 1/3 / 1/3 / 1/3 / 2/3 |

**T8 — S1 − S2 on Qwen** (primary).

| benchmark | fraction | 10k FIT, DEV / val (R7-40c) | full FIT, DEV / val (E2) | E2-8 | 10k FIT, official test (R7-50 Q) | seeds < 0: 10k / full / Q |
|---|---|---|---|---|---|---|
| gqa_dev | f0.25 | -0.61 [-1.11, -0.13], p 0.026 | -0.38 [-0.74, -0.01], p 0.094 | 10k only | -1.40 [-1.94, -0.83], p 0.006 | 3/3 / 2/3 / 3/3 |
| gqa_dev | f0.5 | -0.28 [-0.70, +0.11], p 0.350 | -0.54 [-0.87, -0.18], p 0.012 | full only | -1.05 [-1.48, -0.60], p 0.006 | 2/3 / 3/3 / 3/3 |
| gqa_dev | f0.75 | -0.27 [-0.62, +0.07], p 0.420 | -0.65 [-0.90, -0.40], p 0.009 | full only | -0.53 [-0.85, -0.20], p 0.006 | 2/3 / 3/3 / 3/3 |
| textvqa_val | f0.25 | +14.81 [+13.84, +15.74], p 0.009 | -2.40 [-2.95, -1.81], p 0.009 | opposite signs | — | 0/3 / 2/3 / — |
| textvqa_val | f0.5 | +3.98 [+3.32, +4.61], p 0.009 | -0.44 [-0.81, -0.10], p 0.030 | opposite signs | — | 1/3 / 2/3 / — |
| textvqa_val | f0.75 | +0.02 [-0.37, +0.41], p 0.886 | -0.19 [-0.47, +0.09], p 0.193 | neither scale | — | 1/3 / 2/3 / — |
| chartqa_val | f0.25 | -12.60 [-14.19, -11.05], p 0.009 | -3.58 [-4.82, -2.29], p 0.009 | claim: S1 below | -13.28 [-14.68, -11.93], p 0.006 | 3/3 / 3/3 / 3/3 |
| chartqa_val | f0.5 | -8.37 [-9.67, -7.01], p 0.009 | -3.14 [-4.20, -2.11], p 0.009 | claim: S1 below | -9.08 [-10.32, -7.85], p 0.006 | 3/3 / 3/3 / 3/3 |
| chartqa_val | f0.75 | -2.22 [-3.22, -1.25], p 0.009 | -0.89 [-1.62, -0.19], p 0.048 | claim: S1 below | -3.00 [-3.88, -2.21], p 0.006 | 3/3 / 3/3 / 3/3 |

- LLaVA: no DF-8 claim at any point.  The 10k deficit (Holm-detected at L25 / L50 on all three benchmarks) is not detected at full FIT on GQA and
  ChartQA and reverses on TextVQA (+2.01 / +2.19 at L25 / L50, Holm-detected): 6 points are scale-dependent and 3 have the same sign but fail Holm
  at one or both scales.  All seeds agree in sign only on GQA with the 10k-FIT selectors (DEV / val and test).
- Qwen: E2-8 claims "S1 below" on ChartQA at all three fractions (full −3.58 / −3.14 / −0.89); TextVQA has opposite signs at f 0.25 / 0.5 (10k
  +14.81, full −2.40 at f 0.25); GQA is significant at one scale only at every fraction.
- Official test.  10k-FIT selectors: S1 is below S2 on GQA testdev at L25 / L50 / L75 (3 / 3 seeds negative in each) and on ChartQA test at L25
  (−0.87; seeds −2.61 / +0.61 / −0.62) on LLaVA, and at every fraction on both benchmarks on Qwen (every seed negative).  Full-FIT LLaVA
  selectors: two detected cells of opposite sign at L25, GQA testdev −0.78 [−1.16, −0.40] (seeds −0.21 / +0.21 / −2.35) and ChartQA test +0.75
  [+0.34, +1.17] (seeds +2.17 / −0.10 / +0.18), both Holm p 0.006; the other four cells are not detected.  The intervals cover item sampling with
  the three trained seeds fixed (`R7_50_PHASE_F_REPORT_20261014.md:140`), so a cell carried by one seed says little about the objective.
- So there is no host-independent or scale-independent ordering of VCS and logistic, no claim either way on LLaVA, and no equivalence statement.

## 4. The closing rule of `PROJECT_MAP_20261014.md` §11 C, and why neither label is used

The rule (line 237; written by the coordinator in 6c91e9e, 2026-10-14 17:07 +0200, before the first REV9 result, 821ef73, 2026-10-15 08:15
+0200): "D1：等 REV9（LLaVA 10k + FULL）结果——若 I1/I2 显著缩小 S1−S2 / S1−grid 差距，则 D1 的报告以'训练缺陷修复后的匹配对照'收口；若不显著，D1 作为
'后验读数不适合候选选择'的负面结构发现收口。不再做新的 D1 变体。"  It is a suggestion in an interpretive document (§11 "下一步（建议…）"), not a frozen
protocol, and it defines no test of "显著缩小"; it is read here with the frozen REV9 Holm decisions.

**Positive condition: met at one scale only.**  The uniform-grid row is fixed per item, so the change of S1 − grid at a fixed budget is exactly
S1's Holm-tested I − frozen contrast.

**T12 — S1 − grid under I1 / I2 at fixed budgets** (frozen REV9 / REV9-FULL primary contrasts; grid = the frozen D1-30 rows on the same items).

| benchmark | budget | 10k VCS (3e-4): frozen − grid | I1 − frozen, Holm p | I2 − frozen, Holm p | full VCS (1e-4): frozen − grid | I1 − frozen, Holm p | I2 − frozen, Holm p |
|---|---|---|---|---|---|---|---|
| gqa_dev | 72 | -4.59 | +0.08, 1.000 | +2.49, 0.024 | -3.26 | +0.32, 0.610 | -4.94, 0.024 |
| gqa_dev | 144 | -3.30 | -0.01, 1.000 | +1.54, 0.024 | -2.76 | +0.51, 0.024 | -5.30, 0.024 |
| gqa_dev | 288 | -0.63 | +0.26, 0.180 | +0.57, 0.024 | -0.72 | +0.15, 0.960 | -0.58, 0.324 |
| gqa_dev | 432 | +0.45 | -0.01, 1.000 | +0.09, 1.000 | +0.33 | +0.10, 0.955 | -0.30, 0.610 |
| textvqa_val | 72 | -5.97 | +0.38, 0.455 | +4.06, 0.024 | -6.81 | -0.35, 0.610 | -7.65, 0.024 |
| textvqa_val | 144 | -5.99 | +0.24, 1.000 | +4.28, 0.024 | -8.61 | -0.19, 0.960 | -10.52, 0.024 |
| textvqa_val | 288 | -1.68 | -0.04, 1.000 | +2.22, 0.024 | -5.28 | +0.60, 0.045 | -3.70, 0.024 |
| textvqa_val | 432 | +2.04 | -0.10, 1.000 | -0.26, 0.406 | +1.14 | +0.63, 0.024 | -4.26, 0.024 |
| chartqa_val | 72 | -0.54 | +0.23, 1.000 | +1.01, 0.024 | -1.46 | +0.38, 0.605 | -1.13, 0.056 |
| chartqa_val | 144 | -0.94 | -0.03, 1.000 | +1.49, 0.024 | -1.93 | +0.33, 0.826 | -1.96, 0.024 |
| chartqa_val | 288 | +0.40 | +0.09, 1.000 | +0.85, 0.032 | -0.23 | +0.36, 0.826 | +0.31, 0.960 |
| chartqa_val | 432 | +0.61 | -0.17, 1.000 | -0.17, 1.000 | +0.36 | +0.10, 0.960 | -0.89, 0.065 |

- 10k (VCS at its CAL-chosen lr 3e-4): I2 narrows all 8 deficit cells (Holm), I1 none.  Full FIT (VCS at lr 1e-4): I2 widens 6 of the 9
  deficit cells and takes TextVQA 432 from above the grid (+1.14) to below it; I1 narrows 2 of 9 (GQA 144 +0.51, TextVQA 288 +0.60).  The
  narrowing does not hold at both scales the rule names; DF-8, D1's frozen two-scale claim rule (applied here by analogy), would call it
  scale-dependent.
- S1 − S2 was never tested: every REV9 / REV10 primary contrast is within one objective.  Descriptively, the post-intervention S1 − S2 depends on
  which arm is paired with which, from −8.66 to +3.11 at 10k (T13); choosing the arms from these DEV / val results would be selection on the
  evaluation data.  Pairings through the 10k S2 arms are also seed-fragile: frozen S2 TextVQA accuracy at 72 tokens is 34.10 / 15.69 / 25.73 by
  seed, and its I2 − frozen is −8.86 / +10.44 / −7.34.
- REV10, added after the rule, returned MIXED (§5): no defect with a stable fix is identified.

**T13 — S1 − S2 at fixed budgets by arm pairing** (seed means; descriptive, post hoc, no interval; not a test).

| scale | benchmark | budget | frozen S1 − frozen S2 | I1 S1 − I1 S2 | I2 S1 − I2 S2 | I2 S1 − I1 S2 (the arms that responded at 10k) |
|---|---|---|---|---|---|---|
| 10k | gqa_dev | 72 | -3.46 | -4.10 | +1.74 | -1.69 |
| 10k | gqa_dev | 144 | -2.68 | -3.26 | +1.93 | -1.71 |
| 10k | gqa_dev | 288 | -1.00 | -0.99 | +1.31 | -0.68 |
| 10k | gqa_dev | 432 | -0.04 | -0.12 | +0.83 | -0.02 |
| 10k | textvqa_val | 72 | -5.92 | -8.66 | +0.06 | -4.98 |
| 10k | textvqa_val | 144 | -4.44 | -7.76 | +3.02 | -3.72 |
| 10k | textvqa_val | 288 | -2.04 | -5.04 | +3.11 | -2.77 |
| 10k | textvqa_val | 432 | -0.24 | -0.48 | +2.33 | -0.65 |
| 10k | chartqa_val | 72 | -1.16 | -1.87 | +0.17 | -1.09 |
| 10k | chartqa_val | 144 | -1.09 | -2.10 | +0.95 | -0.57 |
| 10k | chartqa_val | 288 | -0.50 | -1.16 | +0.56 | -0.40 |
| 10k | chartqa_val | 432 | +0.16 | -0.28 | +0.68 | -0.28 |
| full | gqa_dev | 72 | -0.30 | -0.20 | +0.68 | — |
| full | gqa_dev | 144 | +0.01 | -0.06 | +0.60 | — |
| full | gqa_dev | 288 | +0.00 | +0.10 | +0.09 | — |
| full | gqa_dev | 432 | -0.05 | -0.03 | +0.02 | — |
| full | textvqa_val | 72 | +0.98 | -0.11 | -0.96 | — |
| full | textvqa_val | 144 | +2.01 | -0.04 | -1.04 | — |
| full | textvqa_val | 288 | +2.20 | +1.10 | +1.83 | — |
| full | textvqa_val | 432 | -0.11 | -0.02 | +1.32 | — |
| full | chartqa_val | 72 | +0.03 | -0.02 | +0.07 | — |
| full | chartqa_val | 144 | -0.23 | -0.21 | +0.21 | — |
| full | chartqa_val | 288 | +0.28 | +0.50 | +0.49 | — |
| full | chartqa_val | 432 | -0.14 | +0.14 | -0.17 | — |

**Negative label: claims more than the data show.**
- The host's own read-out is informative: clause 3 did not fire (T6), and inside a (question, budget) the teacher's d picks among the three D1-10
  masks with regret 0.087–0.093, against 0.147–0.173 for a random choice (T11; a reference that uses host outputs, not a deployable selector).
  What is at chance is the learned VCS and logistic critics: pairwise accuracy 0.498–0.504, regret 0.143–0.172 against 0.147–0.173 for a
  random choice.
- The shortfall is not specific to a posterior read-out: S3, which regresses the realised risk directly, is below the uniform grid at 25 % in all
  12 (host, FIT, benchmark) rows and below every external row (T18); S1 is too, and S2 is below the grid in 10 of 12 rows and below every external
  row.
- With the critic identical, phase-B gate training alone moves delivered accuracy by −10.52 to +4.28 points (§5).

**T11 — within-question mask choice of the frozen critics** (REV8, critic-only, CAL; 3 seeds per row; descriptive, no test).

| scale | objective (frozen lr) | pairwise accuracy within (q, k) | seeds with 95 % CI ∋ 0.5 | regret: critic's choice | regret: random choice | regret: teacher d choice (reference, uses host outputs) |
|---|---|---|---|---|---|---|
| LLaVA 10k | vcs (0.0003) | 0.499 ± 0.001 | 3 / 3 | 0.145 ± 0.003 | 0.147 ± 0.000 | 0.093 ± 0.000 |
| LLaVA 10k | logistic (0.0001) | 0.501 ± 0.003 | 2 / 3 | 0.145 ± 0.001 | 0.147 ± 0.000 | 0.093 ± 0.000 |
| LLaVA 10k | direct (0.0001) | 0.525 ± 0.004 | 0 / 3 | 0.137 ± 0.003 | 0.147 ± 0.000 | 0.093 ± 0.000 |
| LLaVA FULL | vcs (0.0001) | 0.500 ± 0.003 | 3 / 3 | 0.143 ± 0.001 | 0.147 ± 0.000 | 0.093 ± 0.000 |
| LLaVA FULL | logistic (0.0001) | 0.504 ± 0.005 | 2 / 3 | 0.143 ± 0.002 | 0.147 ± 0.000 | 0.093 ± 0.000 |
| LLaVA FULL | direct (0.0001) | 0.544 ± 0.005 | 0 / 3 | 0.129 ± 0.001 | 0.147 ± 0.000 | 0.093 ± 0.000 |
| Qwen 10k | vcs (0.0003) | 0.498 ± 0.002 | 3 / 3 | 0.172 ± 0.002 | 0.173 ± 0.000 | 0.087 ± 0.000 |
| Qwen 10k | logistic (0.0001) | 0.498 ± 0.003 | 2 / 3 | 0.170 ± 0.002 | 0.173 ± 0.000 | 0.087 ± 0.000 |
| Qwen 10k | direct (0.0003) | 0.527 ± 0.014 | 0 / 3 | 0.160 ± 0.002 | 0.173 ± 0.000 | 0.087 ± 0.000 |

**T18 — the learned selectors next to the training-free and external rows at 25 % of the visual tokens** (fixed budget, seed means, no interval;
descriptive.  Qwen external = E1 columns, own environments, not a same-condition comparison).

| host, FIT | benchmark | uniform grid | uniform random | external (LLaVA: SparseVLM adapted; Qwen: MMTok / HiPrune) | S1 VCS | S2 logistic | S3 direct risk | learned rows below the grid |
|---|---|---|---|---|---|---|---|---|
| LLaVA, 10k | gqa_dev | 81.12 | 80.37 | 81.94 | 77.82 | 80.50 | 78.82 | 3 / 3 |
| LLaVA, 10k | textvqa_val | 32.55 | 30.21 | 43.53 | 26.55 | 30.99 | 17.80 | 3 / 3 |
| LLaVA, 10k | chartqa_val | 16.98 | 16.61 | 18.65 | 16.04 | 17.14 | 14.51 | 2 / 3 |
| LLaVA, full | gqa_dev | 81.12 | 80.37 | 81.94 | 78.36 | 78.35 | 77.82 | 3 / 3 |
| LLaVA, full | textvqa_val | 32.55 | 30.21 | 43.53 | 23.94 | 21.93 | 22.76 | 3 / 3 |
| LLaVA, full | chartqa_val | 16.98 | 16.61 | 18.65 | 15.05 | 15.28 | 15.30 | 3 / 3 |
| Qwen, 10k | gqa_dev | 84.36 | — | 86.20 / 85.34 | 79.89 | 80.50 | 81.13 | 3 / 3 |
| Qwen, 10k | textvqa_val | 73.18 | — | 76.30 / 78.03 | 66.01 | 51.19 | 57.46 | 3 / 3 |
| Qwen, 10k | chartqa_val | 56.77 | — | 65.57 / 69.95 | 39.51 | 52.12 | 38.85 | 3 / 3 |
| Qwen, full | gqa_dev | 84.36 | — | 86.20 / 85.34 | 79.64 | 80.02 | 83.53 | 3 / 3 |
| Qwen, full | textvqa_val | 73.18 | — | 76.30 / 78.03 | 71.34 | 73.74 | 41.54 | 2 / 3 |
| Qwen, full | chartqa_val | 56.77 | — | 65.57 / 69.95 | 41.30 | 44.88 | 39.51 | 3 / 3 |

So the closeout rests on D1's pre-registered refutation rule (§2), with the phase-B readings kept descriptive (§5).  The rule's last clause
("不再做新的 D1 变体") stands.

## 5. Phase B × learning rate: REV8 → REV9 → REV9-FULL → REV10 (descriptive; REV10's report)

REV9 / REV10 restart phase B (gate training) of frozen selector units from their end-of-phase-A states; the estimator stays frozen (D1-20
freeze OPEN-5, `R7_D1_20_STAGED_FREEZE_20261012.yaml:59, 67`: gradients reach the gate MLP only).  I1 = critic in eval mode during phase B (REV8
R8-01: dropout perturbs the gate's inputs); I2 = I1 + straight-through hard top-k (R8-02: soft–hard relaxation gap)
(`REV9_PROTOCOL_FROZEN_20261014.yaml:16-17`).  Primary per cell: I1 − frozen and I2 − frozen on 3 DEV / val benchmarks × 4 fixed budgets, Holm
over the cell's 24 contrasts.  REV10 adds the two missing corners of objective × phase-A lr at 10k (seed 0 from the D1-20 lr sweep, seeds 1–2
newly trained; not CAL-chosen by construction).

**T14 — responses per cell** (Holm-positive, Holm-negative of 12 per intervention; patterns V / L as defined in the REV10 rule, which decides only
for the two REV10 cells).

| scale | objective | phase-A lr | CAL-chosen | source | guard I0 (estimator / gate max abs diff) | I1 − frozen: Holm +, − | I2 − frozen: Holm +, − | net I1 | net I2 | pattern (REV10 definitions) |
|---|---|---|---|---|---|---|---|---|---|---|
| 10k | VCS | 3e-4 | yes | REV9 | 0.0 / 0.0 | 0, 0 | 9, 0 | +0 | +9 | V |
| 10k | logistic | 1e-4 | yes | REV9 | 0.0 / 0.0 | 8, 0 | 0, 9 | +8 | -9 | L |
| 10k | direct | 1e-4 | yes | REV9 | 0.0 / 0.0 | 1, 0 | 0, 0 | +1 | +0 | neither |
| 10k | VCS | 1e-4 | no | REV10 | 0.0 / 0.0 | 1, 2 | 0, 10 | -1 | -10 | neither |
| 10k | logistic | 3e-4 | no | REV10 | 0.0 / 0.0 | 0, 3 | 0, 1 | -3 | -1 | neither |
| full | VCS | 1e-4 | yes | REV9-FULL | 0.0 / 0.0 | 3, 0 | 0, 7 | +3 | -7 | L |
| full | logistic | 1e-4 | yes | REV9-FULL | 0.0 / 0.0 | 5, 0 | 0, 8 | +5 | -8 | L |
| full | direct | 1e-4 | yes | REV9-FULL H100 secondary, seed 0 only | 0.0 / 0.0 | 0, 0 | 10, 0 | +0 | +10 | V |

- Every I0 guard is exact (estimator and gate max abs diff 0 on seed 0 of each cell) and phase B never updates the estimator, so frozen, I1 and
  I2 units share the identical critic and every effect is a gate-training effect.  Prediction files: 324 (REV9), 216 (REV9-FULL), 36 (H100 secondary), 216 (REV10), none with dropped, blank or duplicate
  lines.
- 67 of the 168 contrasts of the seven seeded cells are Holm-significant, from −10.52 to +4.28 points.
- **REV10's frozen rule:** vcs_lr0.0001 (net I1 −1, net I2 −10) shows neither pattern V nor pattern L, and neither does logistic_lr0.0003 (net I1
  −3, net I2 −1), so the outcome is **MIXED**: reported descriptively, no attribution to the objective or to the learning rate.  This is the
  computation recorded in 241a2c9.
- Descriptive regularities, not attributions: I2 is net-negative in all four VCS / logistic cells at phase-A lr 1e-4 (−10, −9, −8, −7 across both
  scales), net-positive only for 10k VCS at 3e-4 (+9) and near null for 10k logistic at 3e-4 (−1).  I1 helps logistic at 1e-4 (+8 at 10k, +5 at
  full) and VCS at full (+3), is null for 10k VCS at 3e-4, and is net-negative in both off-CAL 10k cells.  The single-seed H100 direct secondary
  (+10 for I2) is an unreplicated observation; seeding it was declined (`OWNER_DECISIONS_20261015.md` line 20).

**T15 — the gate's soft − hard selection gap at the end of phase B** (seed mean [min, max]; secondary).

| scale | objective | phase-A lr | frozen @72 | I1 @72 | I2 @72 | frozen @144 | I1 @144 | I2 @144 |
|---|---|---|---|---|---|---|---|---|
| 10k | VCS | 3e-4 | -0.0567 [-0.0639, -0.0528] | -0.0557 [-0.0644, -0.0473] | -0.0283 [-0.0421, -0.0192] | -0.0356 [-0.0371, -0.0330] | -0.0348 [-0.0387, -0.0321] | -0.0192 [-0.0255, -0.0147] |
| 10k | logistic | 1e-4 | -0.0037 [-0.0044, -0.0029] | -0.0020 [-0.0027, -0.0005] | +0.0173 [-0.0137, +0.0521] | -0.0036 [-0.0072, -0.0012] | -0.0025 [-0.0045, -0.0002] | +0.0073 [-0.0111, +0.0285] |
| 10k | direct | 1e-4 | -0.0000 [-0.0000, +0.0000] | -0.0000 [-0.0000, +0.0000] | -0.0000 [-0.0000, +0.0000] | -0.0000 [-0.0000, +0.0000] | -0.0000 [-0.0000, -0.0000] | +0.0000 [-0.0000, +0.0000] |
| 10k | VCS | 1e-4 | -0.0104 [-0.0223, -0.0010] | -0.0104 [-0.0180, -0.0001] | -0.0096 [-0.0177, +0.0013] | -0.0042 [-0.0093, -0.0005] | -0.0045 [-0.0085, +0.0002] | -0.0057 [-0.0112, +0.0013] |
| 10k | logistic | 3e-4 | -0.0477 [-0.0515, -0.0443] | -0.0515 [-0.0528, -0.0505] | -0.0275 [-0.0392, -0.0149] | -0.0298 [-0.0361, -0.0196] | -0.0327 [-0.0401, -0.0208] | -0.0239 [-0.0297, -0.0134] |
| full | VCS | 1e-4 | -0.0024 [-0.0032, -0.0012] | -0.0034 [-0.0060, -0.0011] | -0.0034 [-0.0070, -0.0001] | -0.0014 [-0.0017, -0.0011] | -0.0021 [-0.0037, -0.0010] | -0.0040 [-0.0066, -0.0001] |
| full | logistic | 1e-4 | -0.0013 [-0.0024, -0.0004] | -0.0024 [-0.0036, -0.0010] | -0.0023 [-0.0026, -0.0017] | -0.0008 [-0.0015, -0.0002] | -0.0012 [-0.0019, -0.0004] | -0.0020 [-0.0032, -0.0009] |
| full | direct | 1e-4 | +0.0000 (n 1) | +0.0001 (n 1) | +0.0006 (n 1) | +0.0000 (n 1) | -0.0000 (n 1) | +0.0004 (n 1) |

- The gap follows the phase-A learning rate, not the objective: |frozen gap| at 72 tokens is ≥ 0.044 in all six lr-3e-4 units and ≤ 0.022 in all
  twelve VCS / logistic units at lr 1e-4.  REV8's "at 1e-4 every |SD − HD| ≤ 0.004" (REV8 report line 104) describes its own 36 units only:
  REV10's newly trained VCS-1e-4 seeds 1 and 2 log −0.0223 and −0.0079.
- The gap does not predict the accuracy response: I2 roughly halves the gap in both lr-3e-4 cells (VCS −0.0567 → −0.0283, logistic −0.0477 →
  −0.0275), yet VCS gains (net +9) and logistic does not (net −1).

**T16 — REV10 secondary: frozen − uniform grid per cell at 10k** (seed means; the frozen D1-30 grid rows on the same items; no interval).

| benchmark | budget | uniform grid | VCS 3e-4 (CAL): frozen − grid | VCS 1e-4: frozen − grid | logistic 1e-4 (CAL): frozen − grid | logistic 3e-4: frozen − grid |
|---|---|---|---|---|---|---|
| gqa_dev | 72 | 79.18 | -4.59 | -2.71 | -1.13 | -1.52 |
| gqa_dev | 144 | 81.12 | -3.30 | -1.84 | -0.62 | -1.14 |
| gqa_dev | 288 | 82.56 | -0.63 | -0.29 | +0.37 | -0.03 |
| gqa_dev | 432 | 83.50 | +0.45 | +0.49 | +0.50 | +0.50 |
| textvqa_val | 72 | 25.22 | -5.97 | +1.04 | -0.05 | -2.17 |
| textvqa_val | 144 | 32.55 | -5.99 | +0.17 | -1.55 | -2.62 |
| textvqa_val | 288 | 39.34 | -1.68 | +1.33 | +0.35 | +0.87 |
| textvqa_val | 432 | 43.18 | +2.04 | +2.08 | +2.29 | +2.22 |
| chartqa_val | 72 | 15.83 | -0.54 | +0.59 | +0.62 | -0.52 |
| chartqa_val | 144 | 16.98 | -0.94 | +0.17 | +0.16 | -0.56 |
| chartqa_val | 288 | 17.29 | +0.40 | +0.71 | +0.90 | +0.99 |
| chartqa_val | 432 | 18.80 | +0.61 | +0.49 | +0.45 | +0.36 |

**T17 — REV10 secondary: I2 − I1** (unadjusted bootstrap p; outside every Holm family).

| benchmark | budget | VCS 1e-4: I2 − I1 [95 % CI], p (unadjusted) | logistic 3e-4: I2 − I1 [95 % CI], p (unadjusted) |
|---|---|---|---|
| gqa_dev | 72 | -1.83 [-2.42, -1.25], p 0.001 | -0.24 [-0.69, +0.20], p 0.273 |
| gqa_dev | 144 | -1.75 [-2.35, -1.17], p 0.001 | -0.30 [-0.70, +0.10], p 0.141 |
| gqa_dev | 288 | -1.30 [-1.74, -0.89], p 0.001 | -0.06 [-0.36, +0.24], p 0.684 |
| gqa_dev | 432 | -0.60 [-0.91, -0.30], p 0.001 | -0.07 [-0.23, +0.09], p 0.363 |
| textvqa_val | 72 | -6.72 [-7.45, -6.00], p 0.001 | +0.97 [+0.42, +1.54], p 0.003 |
| textvqa_val | 144 | -9.10 [-9.90, -8.23], p 0.001 | +0.60 [+0.03, +1.16], p 0.041 |
| textvqa_val | 288 | -7.38 [-8.19, -6.61], p 0.001 | -0.31 [-0.77, +0.14], p 0.172 |
| textvqa_val | 432 | -4.72 [-5.27, -4.17], p 0.001 | -0.27 [-0.50, -0.05], p 0.011 |
| chartqa_val | 72 | -1.68 [-2.35, -1.03], p 0.001 | +0.85 [+0.29, +1.43], p 0.002 |
| chartqa_val | 144 | -2.10 [-2.90, -1.36], p 0.001 | +0.43 [-0.10, +0.98], p 0.130 |
| chartqa_val | 288 | -0.28 [-0.99, +0.47], p 0.483 | +0.05 [-0.43, +0.55], p 0.833 |
| chartqa_val | 432 | -0.64 [-1.24, -0.02], p 0.046 | +0.28 [+0.00, +0.56], p 0.048 |

- T16, descriptive: the off-CAL VCS units (lr 1e-4) are above the CAL-chosen VCS units (lr 3e-4) at every budget up to 288 on all three
  benchmarks (by +0.31 to +7.01).  Against logistic at the same lr (1e-4) they are lower on GQA (−1.58 / −1.22 at 72 / 144), higher on TextVQA
  (+1.10 / +1.73) and level on ChartQA (−0.03 / +0.02).  CAL chose 3e-4 for VCS by a Spearman margin of 0.0046 (0.0920 against 0.0874, seed 0;
  rule `R7_D1_20_STAGED_FREEZE_20261012.yaml:51-56`), about one seed sd of the selected units (0.0039).  So D1's 10k S1 − S2 contrasts compare
  VCS at 3e-4 with logistic at 1e-4: they describe the frozen pipelines as trained (objective × CAL-chosen lr × phase-B recipe), not the
  objectives.  This reads off-CAL units the frozen rule could not have selected and creates or removes no claim; on Qwen, where every full-FIT
  unit was CAL-chosen at 1e-4 (E2), S1 is still below S2 on ChartQA (T8).
- T17: adding hard top-k to the eval-mode critic lowers VCS-1e-4 accuracy in all 12 cells (unadjusted p < 0.05 in 11; largest −9.10, TextVQA
  144) and moves logistic-3e-4 by −0.31 to +0.97.

## 6. Qwen2.5-VL-7B: R7-40c, E2, R7-50 phase Q, E1

**T9 — E2-8 verdicts over the 27 Qwen primary contrasts** (recomputed from the stored deltas and Holm p; identical to the stored verdicts).

| contrast | claim: S1 below | claim: S1 above | opposite signs | 10k only | full only | neither scale |
|---|---|---|---|---|---|---|
| S1-S2 | 3 | 0 | 2 | 1 | 2 | 1 |
| S1-S3 | 3 | 2 | 0 | 0 | 3 | 1 |
| S1-grid | 7 | 0 | 0 | 2 | 0 | 0 |
| all 27 | 13 | 2 | 2 | 3 | 5 | 2 |

**T10 — E1 external columns** (MMTok, HiPrune: official code in its own pinned environment at batch 1 on R7-40c's items; a separate host column
with no contrast or test computed, D-E1-7; descriptive only).

| benchmark | learned rows (S1, S2, S3) below the grid column, f 0.125–0.75 | MMTok / HiPrune above the grid column, f 0.125–0.5 | largest grid − S1 (f) |
|---|---|---|---|
| gqa_dev | 12 / 12 | 6 / 6 | 5.31 (f 0.125) |
| textvqa_val | 12 / 12 | 6 / 6 | 7.17 (f 0.25) |
| chartqa_val | 12 / 12 | 6 / 6 | 17.26 (f 0.25) |

- The learned selectors do not reach the training-free grid on Qwen: 13 of the 27 contrasts are "S1 below" claims (7 against the grid, 3 against
  S2 on ChartQA, 3 against S3 on GQA); the only "S1 above" claims are against S3 on TextVQA.  On the official test (phase Q, 10k-FIT selectors)
  S1 is below the grid and below S2 at every fraction on both benchmarks (T3, T8).
- E1, descriptive: both published training-free methods are above the grid column at f ≤ 0.5 on all three benchmarks (18 of 18), and the learned
  rows are below it in all 36 cells at f 0.125–0.75.  Not a same-condition comparison; it enters no claim.
- Mechanism evidence on Qwen is critic-level only: the Qwen VCS and logistic critics choose at chance within a (question, budget) (T11).  REV9 /
  REV10 did not run on Qwen.

## 7. Not claimed

- Any advantage of the VCS-trained selector over SparseVLM, the uniform grid, MMTok or HiPrune, and any acceleration claim.
- That VCS and logistic selectors are equivalent, or that either objective is better: the S1 − S2 contrasts describe the frozen pipelines
  (objective × CAL-chosen lr × phase-B recipe) on these hosts and scales.
- That the posterior read-out, or the squared-gap read-out, is unsuitable for candidate selection or explains the D1 shortfall.  REV8 is
  critic-only and descriptive; the teacher-d regret is a reference that needs host outputs.
- That phase-B training defects cause the D1 shortfall, that fixing them closes it, or that the 10k response pattern is driven by the objective
  or by the learning rate (REV10 MIXED).
- Any official-test, latency-matched or Qwen statement about the REV9 / REV10 interventions (DEV / val, fixed budgets, LLaVA only).
- That hard top-k improves the direct-risk selector at full FIT (one seed, one hardware class, descriptive secondary).
- "More training data does not help learned selectors" as a general statement: D1-FULL kept the frozen 20,000 + 5,000-step recipe (about 4.6
  passes per question at full scale against about 32 at 10k; DF-4, D1-FULL report §3), and its scale contrasts are descriptive.
- Holm-detected cells as properties of an objective: the intervals condition on the three trained seeds, and several detected cells rest on one
  seed (T7 and the per-seed fields of REV9 / REV10).

## 8. Corrections made with this report, and earlier readings not carried forward

**Corrections** (each a visible "Corrections (2026-10-17)" section appended to the file; the original sentence stays in place; no results file
changes):
1. `reports/PROJECT_MAP_20261014.md:175` wrote "grid 58.70 > SparseVLM 58.79 > S2 56.57 > S1 55.73 > S3 53.41" (R7-50 phase F, GQA testdev,
   25 %).  58.70 < 58.79: the order is SparseVLM (adapted) 58.79 > uniform grid 58.70 > S2 56.57 > S1 55.73 > S3 53.41 (full 61.95), as in the
   phase F table (`R7_50_PHASE_F_REPORT_20261014.md:52-58`) and its reading "SparseVLM is the best row at 25–50 %" (line 148).
2. `reports/R7/R7_50_PHASE_Q_REPORT_20261013.md:85` wrote "On both hosts and on both test sets, the VCS-trained selector does not beat its
   logistic sibling".  It was written before phase F and holds for the 10k-FIT selectors of phases L and Q only; read as "the VCS selector never
   beats its logistic sibling on the official test", it is wrong.  Phase F detected the full-FIT LLaVA S1 above S2 on ChartQA test at L25 (+0.75
   [+0.34, +1.17], Holm p 0.006; seeds +2.17 / −0.10 / +0.18), next to S1 below S2 on GQA testdev at L25 (−0.78 [−1.16, −0.40], Holm p 0.006;
   seeds −0.21 / +0.21 / −2.35).  The correct statement is §3's.
3. `reports/PROJECT_MAP_20261014.md` also gets a short "Update 2026-10-17: D1 closed" section pointing to this report (its §7.2 still describes
   REV9 as running).

**Earlier interpretive sentences not carried forward** (left in place; listed so they are not reused):
- PROJECT_MAP §7.3 (line 185): the pooled CAL Spearman means r(K) ranks "整体上哪个预算更安全", and S3's slightly better ranking "说明问题在'后验→风险'的
  读数".  REV8 found the pooled score not mostly between-budget variation (its flag is set in 1 of 36 units), and S3 is below S1 on TextVQA under
  both two-scale rules (T4, T5).
- PROJECT_MAP §9.1 (line 206): D1 "得到'无差异'".  D1 found Holm-detected S1 − S2 differences in both directions that depend on FIT scale, host
  and seed (§3); no equivalence margin was set.
- PROJECT_MAP §9.2 (line 207) and §11 C (line 237): the negative structural label (§4).
- REV9 report line 21, "the hard-selection fix works where REV8 located the relaxation gap", and line 33, "a substantial part of the D1 shortfall
  comes from phase-B training mechanics …, not from the objectives' targets": 10k-only readings.  At full FIT I2 widens the grid gap (T12), and at
  10k logistic at 3e-4 has the gap, I2 halves it, and accuracy does not improve (T14, T15).
- REV9-FULL report line 29, "consistent with the 10k pattern being driven by the learning rate": its designated test, REV10, returned MIXED.
  Line 34's "at most +1.86" is logistic's gain; S1's largest Holm-significant I1 change in a deficit cell is +0.60 (TextVQA 288).  Its provenance
  line 6 ("its H100 secondary rerun is pending") is superseded by its own §4 (9637b5b).
- D1-10 readout line 128, "its magnitude is at most about 25 % of h": not for qsim at 75 % (0.367; §2.3).
- REV8 report line 104, "at 1e-4 every |SD − HD| ≤ 0.004": true of REV8's 36 units, not a rule (§5).

## 9. Files

- This report; every table and number: `scripts/r7_d1_closeout_tables.py` (`--check` verifies the tables here).
- Results files read (T0):

| key | file | results-only commit | sha256 (first 16) |
|---|---|---|---|
| d1_10 | `reports/R7/R7_D1_10_results.json` | 4997cb6 | 3e79d140a1e7773b |
| d1_20 | `reports/R7/R7_D1_20_CAL_READOUT.json` | 3edb9cc | d3353a81a07e4814 |
| d1_30 | `reports/R7/R7_D1_30_RESULTS.json` | 750dc61 | 21831ddcab4e93db |
| d1_full | `reports/R7/R7_D1_FULL_results/d1full_results.json` | 7a89736 | 1c81c59afd0a1bd4 |
| r50_L | `reports/R7/R7_50_results_L/r7_50_results.json` | 81c89df | 4ccd69ff25e7386c |
| r50_F | `reports/R7/R7_50_results_F/r7_50_results.json` | ee172c7 | 60cfc0b72b875cef |
| r50_Q | `reports/R7/R7_50_results_Q/r7_50_results.json` | 1bbeb8e | e4e1f428837938a2 |
| e1 | `reports/R7/R7_E1_results/r7_e1_results.json` | 7d98a17 | 6775c798ce2c691e |
| e2 | `reports/R7/R7_E2_results/e2_claims.json` | 895ca7b | 9a5a84fa4d1a79d0 |
| rev8 | `reports/REV8/REV8_DIAG_RESULTS/rev8_diag_results.json` | 5dd8f37 | 781cfe24567b44ef |
| rev9 | `reports/REV9/REV9_RESULTS/rev9_results.json` | 821ef73 | 86665bc105a3412d |
| rev9f | `reports/REV9/REV9_FULL_RESULTS/rev9_full_results.json` | be9a829 | 7d35430c4ca9c124 |
| rev9h | `reports/REV9/REV9_FULL_H100_RESULTS/rev9_full_h100_direct_secondary.json` | 9637b5b | 58b0e29f3293d81e |
| rev10 | `reports/REV9/REV10_RESULTS/rev10_results.json` | 241a2c9 | 61cb280167aefe41 |

- Freeze files cited: `reports/R7/R7_D1_PROTOCOL_FROZEN_20261012.yaml`, `R7_D1_20_STAGED_FREEZE_20261012.yaml`,
  `R7_D1_30_STAGED_FREEZE_20261012.yaml`, `R7_D1_FULL_STAGED_FREEZE_20261012.yaml`, `R7_E2_STAGED_FREEZE_20261013.yaml`,
  `R7_50_PROTOCOL_FROZEN_20261012.yaml` (all in `reports/R7/`); `reports/REV9/REV9_PROTOCOL_FROZEN_20261014.yaml`;
  `reports/REV9/REV10_LR_FACTORIAL_PROTOCOL_FROZEN_20261015.yaml`.
- Reports these results come from: `reports/R7/` D1-10 and D1-20 CAL readouts, D1-30, D1-FULL, R7-40c, E1, E2, R7-50 phases L / Q / F;
  `reports/REV8/REV8_DIAG_REPORT_20261014.md`; `reports/REV9/REV9_REPORT_20261015.md`, `REV9_FULL_REPORT_20261015.md`.
