# P102 — T line: conditional dependence increments between nested information sets — report — 2026-10-01

Pre-registration: `P102_T_LINE_INCREMENTS_PREREG_FROZEN_20260930.md`. The job was 1016536, run on an H100 in 797 s with exit 0.
It was launched by the orchestrator after the P74 report. The results-only commit is `3a493a4`, holding `P102_t_line_results.{json,md}`.
The oracle self-check passed before any cell ran: the increment identity gap was 1.0e-17, R_orth at the oracle was 0, and J(T*) = S to within 1.4e-17.
The rule read-out script is reproduced at the end of this report.

## 1. Pre-stated reading
**Rule 1, whether Delta tracks the planted strength: no cell passes. As the rule states, Delta does not separate from the null at these n.**

The rule needs two things:
- the repeat-mean ΔJ is non-decreasing in s;
- at the two largest strengths, the lower end of the fixed-critic interval is above the upper end of the s = 0 interval.

**Monotonicity fails in all 8 (encoder × family × objective) groups.** It fails the same way each time.
- **The null floor is positive.** At s = 0, ΔJ is +0.0015 to +0.0038.
- **The smallest planted strength falls below that floor.** At colour 0.05 or blur 0.25, ΔJ is +0.0014 to +0.0025.

**The separation half of the rule does hold in three groups:**
- VCS encoder, colour, VCS objective;
- VCS encoder, colour, JS objective;
- VCS encoder, blur, VCS objective.

There, the two largest strengths sit clearly above the null. For example, VCS colour s = 0.2 has ΔJ = 0.047, with an interval of [0.043, 0.051], against the s = 0 interval [0.0019, 0.0036].

So the pre-registered criterion is not met. Descriptively, the increment rises well above the finite-critic noise floor only once the nuisance is strong enough to be detected at all. That means colour 0.1 and above and blur 0.5 and above on the VCS encoder, and blur 1.0 on SimCLR.

| group (VCS objective) | ΔJ at s = 0, s₁, s₂, s₃ |
|---|---|
| VCS colour (0 / 0.05 / 0.1 / 0.2) | +0.0028, +0.0017, +0.0063, **+0.0470** |
| VCS blur (0 / 0.25 / 0.5 / 1.0) | +0.0029, +0.0016, +0.0132, +0.0132 |
| SimCLR colour | +0.0021, +0.0016, +0.0014, +0.0008 |
| SimCLR blur | +0.0025, +0.0014, +0.0024, **+0.0672** |

**Rule 2, whether the nested increments are consistent: in all 32 cells (16 per objective) the R_orth interval excludes 0, always on the positive side.**
- Every cell is therefore reported as a **fit / nesting residual (positive)**, not as a finding about the representation.
- **R_orth grows with the planted strength.** It is about 0.008–0.011 at s = 0 and 0.08–0.15 at the strongest cells. That is as large as the increments themselves.
- **Related symptom: J falls on refinement where it should not.** Under true nesting, J of the finer set must be at least J of the coarser set. Yet the 512-d set scores below pca64 in three VCS-encoder cells:

  | VCS-encoder cell | J on h | J on pca64 |
  |---|---|---|
  | colour 0.1 | 0.0046 | 0.0095 |
  | blur 0.5 | 0.0122 | 0.0143 |
  | blur 1.0 | 0.121 | 0.164 |

  The h critic underfits relative to the pca64 one under the shared budget of 300 steps, with early stopping on the common score.
- **r_BA = ΔJ − D_T is negative in every cell.** Two imperfect critics inflate D_T, which is why the prereg says that a positive D_T is not evidence of dependence.

With finite critics at n = 3 000, then, the L² picture of orthogonal, additive increments along the chain is **not** visible. The residual terms are the same size as the signal.

**Rule 3, where the nuisance is lost.** Chain-step ΔJ, VCS objective, as mean ± refit SD over the 5 repeats:

| cell | J h | J pca64 | J pca16 | J logit16 | J logit_h | h→pca64 | pca64→pca16 | pca16→logit16 |
|---|---|---|---|---|---|---|---|---|
| VCS colour 0.2 | 0.045 | 0.046 | 0.003 | 0.000 | −0.002 | −0.000 ± 0.006 | **+0.043 ± 0.003** | +0.003 ± 0.002 |
| VCS blur 0.5 | 0.012 | 0.014 | 0.004 | 0.001 | −0.001 | −0.002 ± 0.007 | **+0.010 ± 0.006** | +0.003 ± 0.000 |
| VCS blur 1.0 | 0.121 | 0.164 | 0.145 | 0.132 | **0.108** | −0.043 ± 0.005 | +0.019 ± 0.007 | +0.014 ± 0.006 |
| SimCLR blur 1.0 | 0.096 | 0.085 | 0.045 | 0.044 | 0.029 | +0.011 ± 0.002 | **+0.040 ± 0.006** | +0.001 ± 0.002 |
| SimCLR colour (all s) | ≤ 0 | ≤ 0 | ≤ 0 | ≤ 0 | ≤ 0 | — | — | — |

- **Moderate nuisances are lost at the 64 → 16 PCA step.** This holds for VCS colour 0.1 and 0.2 and VCS blur 0.5. The nuisance survives pca64, and the class logits carry essentially none of it, with J(logit_h) ≤ 0.
- **Strong blur (σ 1.0) is different.** The class logits themselves keep most of the dependence: J(logit_h) is 0.108 against J(h) at 0.121 on the VCS encoder, and 0.029 against 0.096 on SimCLR.
  Strong blur changes the class evidence, so compressing to logits does not remove it. This is why the main-pair ΔJ at VCS blur 1.0 (0.013) is no larger than at blur 0.5.
- **SimCLR colour shows no conditional dependence at any strength.** J ≤ 0 for every set. This matches T1, where SimCLR colour is informative only at s = 0.2, n = 2000, and only for the permutation tests.
- **Disclosed gap.** The runner stores fixed-critic bootstrap intervals for the main pair, R_orth and each set's J, but **not** for the individual chain steps.
  Only the refit spread is reported for the steps. The critics were not saved, so a re-evaluation would need a re-fit.

**Rule 4, VCS against JS.** The two objectives agree closely. The difference in ΔJ is within ±0.0032 in 15 of 16 cells.
The exception is VCS blur 1.0, where VCS gives ΔJ = 0.0132 [0.0071, 0.0192] and JS gives 0.0016 [−0.0041, 0.0074]. R_orth there is 0.11 for VCS and 0.15 for JS.
Both are scored with the same squared score, as pre-registered. This is consistent with the prereg's statement that the squared increment is not exclusive to VCS. No ranking is made.

## 2. Reading next to P74 (not an input)
- **Where the two agree.** T1 (P74) found that the VCS conditional permutation test detects these planted nuisances at least as well as fair controls. The T-line magnitudes agree on where there is something to detect:
  - VCS-encoder colour at 0.1 and above, and blur at 0.5 and above;
  - SimCLR blur 1.0;
  - essentially nothing for SimCLR colour.
- **What the T line does not support.** Its increment readout has no resolving power below that threshold: a positive null floor and positive R_orth swamp the small cells.
  It also does not support reading finite-critic increments as additive information budgets.

## 3. Not claimed
- That the squared increment is exclusive to VCS.
- That a positive D_T or R_orth proves dependence.
- Anything about T1's level or power.
- Anything about continuous N, other encoders, or other datasets.

## Delivery
```yaml
experiment_family: diagnostic (T line, conditional increments)
protocol_id: P102_T_line_increments
source_commit: adf9f9b (job); results 3a493a4
estimator: vcs_neural | js_matched (common squared evaluation)
evaluation_readout: J per set, Delta_J, D_T, r_BA, R_orth; paired bootstrap by base image (main pair, R_orth, J); refit SD (5 repeats)
n_independent_units: EVAL 3000 base images per repeat; 5 repeats per cell
status: complete — rule 1 not met (all groups fail monotonicity at the smallest strength; separation holds for VCS-encoder colour (both objectives) and VCS-encoder blur (VCS objective)); rule 2: fit / nesting residual in every cell
```

## Rule read-out script
```python
import json,numpy as np
d=json.load(open("/home/infres/yinwang/CS_QMI/ssl_pilot/reports/P102_t_line_results.json"))["cells"]
S={"colour":[0.0,0.05,0.1,0.2],"blur":[0.0,0.25,0.5,1.0]}
def key(e,f,s): return f"{e}/{f}/s{('%g'%s)}"
print("== rule 1 (Delta tracks strength) ==")
for e in ("vcs4v800","simclr"):
  for f in ("colour","blur"):
    for o in ("vcs","js"):
      m=[d[key(e,f,s)]["summary"][o]["delta_J"]["mean"] for s in S[f]]
      ci=[d[key(e,f,s)]["summary"][o]["delta_J"]["ci_mean_fixed_critics"] for s in S[f]]
      mono=all(m[i+1]>=m[i] for i in range(3)); sep=[ci[j][0]>ci[0][1] for j in (2,3)]
      print(f"{e:9s} {f:6s} {o:3s} dJ="+" ".join(f"{x:+.4f}" for x in m)+f" nondecr={mono} sep(top2 vs s0)={sep} -> {'TRACKS' if mono and all(sep) else 'does not separate'}")
print("\n== rule 2 (R_orth CI contains 0) ==")
n0=0
for k,c in d.items():
  for o in ("vcs","js"):
    lo,hi=c["summary"][o]["R_orth"]["ci_mean_fixed_critics"]; n0+= lo<=0<=hi
print("cells with 0 in R_orth CI:",n0,"of",2*len(d))
print("\n== chain steps: mean ± refit sd over 5 repeats ==")
for k,c in d.items():
  for o in ("vcs",):
    st={}
    for r in c["repeats"]:
      for sn,sv in r["per_objective"][o]["chain"]["steps"].items(): st.setdefault(sn,[]).append(sv["delta_J"])
    Js=c["summary"][o]
    print(f"{k:22s} {o}: J h/pca64/pca16/logit16/logit_h = "+"/".join(f"{Js['J_'+n]['mean']:+.4f}" for n in ("h","pca64","pca16","logit16","logit_h"))+" | "+"; ".join(f"{sn} {np.mean(v):+.4f}±{np.std(v,ddof=1):.4f}" for sn,v in st.items()))
print("\n== VCS - JS on delta_J ==")
print(" ".join(f"{k}:{c['summary']['vcs']['delta_J']['mean']-c['summary']['js']['delta_J']['mean']:+.4f}" for k,c in d.items()))
```
