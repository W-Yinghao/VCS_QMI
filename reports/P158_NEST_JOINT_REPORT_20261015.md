# P158 report — nested increments with truth on the joint-vs-product chain (2026-10-15)

Protocol `P158_NEST_JOINT_PREREG_FROZEN_20261015.md` (frozen 2026-10-15T09:41:28Z; execution note 1: the first bundle 1041319 ran no condition, the
resubmitted bundle 1041372 ran all three, exit 0).  Results-only commit 4be627f: `reports/P158/p158_results.{md,json}`, `truth_{null,weak,moderate}.json`
(5 repeats each; every number below is copied from `p158_results.md`, which `scripts/p158_aggregate.py` writes from the JSON; the per-fit picks in §3 were
read from the `critics` records of the truth files).  Autonomous CPU unit (project map §11 B.2).

## 1. Outcome by the frozen rule (P118 §6.1 clauses, per objective × readout, vs the independent fit, both non-null conditions)
| objective | variant | common readout | JS-native readout | clause pattern weak / moderate (c1 c2 c3 c4) |
|---|---|---|---|---|
| VCS | nested | **no clear reduction** | no clear reduction | ✗ ✓ ✓ ✓ / ✗ ✓ ✓ ✓ |
| JS | nested | **no clear reduction** | no clear reduction | ✗ ✓ ✓ ✓ / ✗ ✓ ✓ ✓ |
| VCS, JS | zero (control) | shrinkage | shrinkage | ✗ ✓ ✗ ✗ (fails 1, 3, 4 by construction — not an improvement, as pre-registered) |

The pre-committed reading for this pattern is the third one: **nested fitting without an exact product term gives no clear reduction of the increment
error for either scoring rule**, so P118's asymmetry (VCS nested_exact "error reduced", JS failing one clause) **does not replicate** in the
joint-vs-product setting.  Clause (1) — M→A increment-error RMSE ≤ 0.75 × the independent fit — fails for both objectives in both conditions:
VCS 0.0027 vs 0.0034 (ratio 0.79) at weak and 0.0032 vs 0.0030 (1.07) at moderate; JS 0.0031 vs 0.0032 (0.97) and 0.0060 vs 0.0033 (1.82).  Clauses
(2)–(4) pass for both: B→M error 0, posterior MSE at B / A not worse, mean M→A increment within 0.0003–0.0058 of the oracle (0.0200 / 0.0530).

## 2. What the numbers show (common readout; mean over 5 repeats)
| cond | objective | variant | err B→M | err M→A (RMSE) | post MSE B | post MSE M | post MSE A | R_orth | ordering violations |
|---|---|---|---|---|---|---|---|---|---|
| weak (true M→A 0.0200) | VCS | indep | −0.0028 | −0.0000 (0.0034) | 0.0069 | 0.0038 | 0.0030 | +0.0050 | 0.8 |
| | VCS | nested | +0.0000 | +0.0003 (0.0027) | 0.0034 | 0.0034 | 0.0030 | +0.0000 | 0.0 |
| | JS | indep | −0.0011 | −0.0003 (0.0032) | 0.0068 | 0.0048 | 0.0040 | +0.0059 | 0.6 |
| | JS | nested | +0.0000 | +0.0008 (0.0031) | 0.0036 | 0.0036 | 0.0040 | +0.0000 | 0.0 |
| moderate (true M→A 0.0530) | VCS | indep | −0.0043 | +0.0007 (0.0030) | 0.0080 | 0.0050 | 0.0036 | +0.0040 | 1.0 |
| | VCS | nested | +0.0000 | +0.0018 (0.0032) | 0.0040 | 0.0040 | 0.0036 | +0.0000 | 0.0 |
| | JS | indep | −0.0024 | +0.0023 (0.0033) | 0.0078 | 0.0067 | 0.0074 | +0.0040 | 0.8 |
| | JS | nested | +0.0000 | +0.0053 (0.0060) | 0.0039 | 0.0039 | 0.0074 | +0.0000 | 0.0 |
- **Where nesting helps is the step the rule does not count.**  The independent fit of the full set B (20 signal + 20 irrelevant coordinates per side)
  scores *below* its own compression M in most repeats (B→M error −0.0011…−0.0043; ordering violations 0.6–1.0 per repeat; posterior MSE at B about
  twice M's): the extra irrelevant coordinates cost the finite critic more than they add.  The nested fit removes this by construction — see §3 — so its
  B→M increment, its R_orth and its B posterior error coincide with M's.  P118 saw the same B→M clean-up; the question was the informative M→A step.
- **The M→A increment is estimated to within ≈ 0.003 by both variants and both rules** (RMSE 0.0027–0.0034, i.e. 15 % of the weak and 6 % of the
  moderate increment); nesting changes it by less than the repeat-to-repeat spread, and for JS at moderate it is worse (0.0060).  The increment itself is
  not under-estimated here (mean errors −0.0003…+0.0023 for the independent fits), unlike the 20–30 % under-estimation P118 reported for its
  conditional-binary chain.
- **Self-checks.**  The paired-bootstrap CI of the M→A consistency residual r_BA contains 0 in 2–4 of 5 repeats for the independent and nested fits (the
  residual is small but not zero at these n); for the nested fits the B→M residual and R_orth are 0 in 5 / 5 by construction.  Null condition (ρ = 0): all
  mean increment errors ≤ 0.0023 (independent) and ≤ 0.0004 (nested); the nested M←A residual stayed at step 0 in 4 / 5 null repeats.
- The JS-native readout (G, Bregman increments) gives the same verdicts and the same pattern (`p158_results.md` §"JS native readout"); the Bregman
  truth of the M→A step (0.0209 / 0.0572) is within 3 % of the L² truth (0.0201 / 0.0530), so the two increment geometries do not separate at this
  dependence level.

## 3. Disclosure that shapes the reading: the nested B fit selected step 0 in every repeat
In all 15 repeats (3 conditions) and for both objectives the nested B ← M residual's VAL-selected step is **0** (g ≡ 0), i.e. the selection chose the coarse
model M as the fine model B — the correct choice for a compression that drops only irrelevant coordinates.  Consequently B→M = 0, R_orth = 0 and
posterior MSE(B) = MSE(M) are exact identities of the fitted object, not estimates; the prereg's gate record anticipated this and assigned the verdict to
the M→A clauses.  The M ← A residual was fitted (selected steps 90–300 in the non-null conditions).  Picked family: `quad` (the class containing the
oracle) in every non-null fit, both variants; in the null condition the independent B / M fits picked `mlp` in 2–3 of 5 repeats.

## 4. Not claimed
Nothing about natural data (P102's CIFAR chain stands as reported); nothing about the exact-Q variants (not definable here); no statement that nesting
is harmful (the M→A RMSEs differ by less than the repeat spread except JS at moderate); no robustness or SSL / VL statement.  P118's result stays as
reported for its conditional-binary chain with the exact product term.

## 5. Place in the evidence
Project map §4.4 listed "nested increments" as a medium-strength argument resting on P118 (truth chain, exact Q) and P102 (natural features, invisible).
P158 adds the sampled-Q truth chain: the L² increment of a compression is recovered to ≈ 0.003 by an ordinary independent fit, nesting adds nothing
measurable to that step, and the VCS/JS asymmetry of P118 is specific to its exact-product-term setting.  For the paper this supports describing the
nested fit as a consistency device for compression chains (exact B→M zero when the compression is information-preserving), not as an accuracy
improvement of the increment estimate.
