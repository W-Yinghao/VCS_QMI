# P106 — T1 critic-class ablation (P105) — report — 2026-10-01

Pre-registration: `P105_T1_CRITIC_CLASS_ABLATION_PREREG_FROZEN_20261001.md`. Jobs 1016890–1016896, all exit 0.
Results-only commit: `e1376b9`, files `P105_t1_ablation_*.json`. The rule script is reproduced at the end of this report.

## 1. Verdict by the frozen rule: **MIXED**
- **Objective test.** "VCS objective" needs d2 ≥ +0.05 in at least 6 of 9 cells **and** d3 ≥ +0.05 in at least 6.
  d2 = vcs2 − js2 reaches +0.05 in **0** cells. So the objective verdict fails.
- **Class test.** "Critic class / exact solver" needs |d2| ≤ 0.05 in at least 6 cells **and** |d3| ≤ 0.05 in at least 6.
  |d2| ≤ 0.05 holds in only 4 cells and |d3| ≤ 0.05 in only 2. So the class verdict fails as well.

| contrast | ≥ +0.05 | within ±0.05 | ≤ −0.05 | mean |
|---|---|---|---|---|
| d2 = vcs2 − js2 (each objective with its SGD linear / MLP critics) | 0 | 4 | **5** | **−0.068** |
| d3 = vcs3 − js3 (each with its exact linear solution available) | **7** | 2 | 0 | +0.081 |
| dc = vcs3 − vcs2 (what the closed form adds to VCS) | — | — | — | **+0.187** |
| replication: vcs3 − js2 (the T1 comparison) | 9 | 0 | 0 | +0.119 |
| forced: vcs_closed − js_exact (exact linear solution, both objectives) | 0 | 6 | **3** | **−0.040** |
| forced: vcs_lin − js_lin | | | | +0.032 |
| forced: vcs_mlp − js_mlp | | | | −0.070 |

**Level.** All four null cells (both colour nulls at n = 2000, both encoders, R = 200) put every test at 0.010–0.080. That is below the 0.09 gate, so no test is flagged and every power cell is read.

## 2. What carries the difference
- **Replication.** On new draws, the T1 comparison is reproduced in all 9 cells: vcs3 beats js2 by +0.05 to +0.31, mean +0.12.
- **With SGD-fitted critics, the VCS objective is not ahead of JS; it is behind or level.**
  - d2 is ≤ −0.05 in 5 cells and within ±0.05 in the other 4.
  - Forced MLP against MLP, VCS trails JS by 0.07 on average. Forced linear against linear, VCS leads by 0.03.
- **With the exactly solved linear critic, the two objectives are equal, with JS slightly ahead.**
  - js_exact ≥ vcs_closed in 8 of 9 cells, mean −0.04.
  - The largest gaps favour JS: −0.13 at VCS colour s 0.05, n 1000, and −0.10 at colour s 0.1, n 500.
  - The closed form is what lifts VCS: it adds +0.19 to VCS on average (dc).
- **Why d3 is still positive: the VAL selection treats the two objectives differently.**
  - VCS's VAL-J criterion picks its closed-form critic in 68–99 % of repeats.
  - JS's VAL-JS criterion picks the exact JS critic in only 14–43 % of repeats. It mostly prefers the MLP, which tests worse.
  - So vcs3 > js3 comes from how each objective's own validation score ranks the exact linear solution against the MLP, not from the power of the exact test.
  - The prereg's wording for this pattern, "d3 > 0 but d2 ≈ 0 → the advantage needs the exact solution and survives a matched exact JS", is only half right here. It survives the VAL-*picked* JS, but **not** the *forced* exact JS.

**Plain statement.** The T1 advantage of the VCS conditional test over the JS statistic (P74) comes from two things:
- the exactly solved linear critic, which only VCS had in T1;
- VCS's validation criterion reliably selecting that critic.

It does not come from greater power of the VCS objective at a fixed critic. Given the same exact linear solution, JS is at least as powerful.
This qualifies P74's conclusion as follows. "T1 holds" remains true as pre-registered: VCS's test, as specified, is at least as powerful as the fair controls specified then. But the advantage should not be attributed to the VCS objective. **Recommended wording:** "a closed-form, exactly solved critic, which the squared VCS objective admits in closed form and selects reliably".

A secondary observation: the squared objective admits a closed-form ridge solution, whereas JS needs an iterative convex solve. That is a practical advantage in computation, not in test power.

## 3. Scope
- These conclusions cover the 9 unsaturated informative T1 cells and the two encoders. They do not cover the 7 saturated cells or the guaranteed (Hoeffding) test.
- js_exact is new to this ablation. Its ridge grid matches the closed form's, and both are fitted by maximising their own VAL objective.

## Delivery
```yaml
experiment_family: dependence_testing (ablation)
protocol_id: P105 / P106
source_commit: d98452a (code); results e1376b9
estimator: VCS (lin / MLP / closed form) vs JS (lin / MLP / exact convex solve in the closed form's class); VAL-picked and forced
evaluation_readout: permutation-test rejection rate, delta 0.05, B 200, R 100 (power) / 200 (level)
status: complete — MIXED by the rule; the advantage is the exact linear critic plus VCS's selection of it, not the objective
```

## Rule script
```python
import json,glob,numpy as np
R="/home/infres/yinwang/CS_QMI/ssl_pilot/reports/"
cells=[];lev=[]
for f in sorted(glob.glob(R+"P105_t1_ablation_*.json")):
    if f.endswith("partial.json"): continue
    d=json.load(open(f)); g=f.split("ablation_")[1][:-5]
    for cn,c in d["cases"].items():
        for n,r in c["by_n"].items():
            p={t:v["power"] for t,v in r["summary"].items()}; picks={t:r["summary"][t].get("picked") for t in ("vcs3","vcs2","js2","js3")}
            (lev if c["mode"]!="planted" else cells).append((g,cn,int(n),r["repeats"],p,picks))
T=("vcs_lin","vcs_mlp","vcs_closed","js_lin","js_mlp","js_exact","vcs3","vcs2","js2","js3")
print("LEVEL (gate 0.09):")
for g,cn,n,R_,p,_ in lev: print(f"  {g:12s} {cn:28s} n={n} R={R_} "+" ".join(f"{t}={p[t]:.3f}" for t in T)+("  FLAG:"+str([t for t in T if p[t]>0.09]) if any(p[t]>0.09 for t in T) else ""))
print("POWER:")
d2=[];d3=[];dc=[];rep=[];fcl=[];fl=[];fm=[]
for g,cn,n,R_,p,pk in cells:
    d2.append(p["vcs2"]-p["js2"]); d3.append(p["vcs3"]-p["js3"]); dc.append(p["vcs3"]-p["vcs2"]); rep.append(p["vcs3"]-p["js2"])
    fcl.append(p["vcs_closed"]-p["js_exact"]); fl.append(p["vcs_lin"]-p["js_lin"]); fm.append(p["vcs_mlp"]-p["js_mlp"])
    print(f"  {g:14s} {cn:12s} n={n:>4} "+" ".join(f"{t}={p[t]:.2f}" for t in T)+f" | d2={d2[-1]:+.2f} d3={d3[-1]:+.2f} dc={dc[-1]:+.2f} rep(vcs3-js2)={rep[-1]:+.2f} closed-exact={fcl[-1]:+.2f}")
    print("     picks:",{k:v for k,v in pk.items()})
d2,d3,dc=np.array(d2),np.array(d3),np.array(dc); N=len(d2)
objective=(d2>=0.05).sum()>=6 and (d3>=0.05).sum()>=6
klass=(abs(d2)<=0.05).sum()>=6 and (abs(d3)<=0.05).sum()>=6 and dc.mean()>=0.05
print(f"\nN={N}; d2>=+.05: {(d2>=0.05).sum()}  |d2|<=.05: {(abs(d2)<=0.05).sum()}  d2<=-.05: {(d2<=-0.05).sum()}  mean d2 {d2.mean():+.3f}")
print(f"       d3>=+.05: {(d3>=0.05).sum()}  |d3|<=.05: {(abs(d3)<=0.05).sum()}  d3<=-.05: {(d3<=-0.05).sum()}  mean d3 {d3.mean():+.3f}")
print(f"       mean dc {dc.mean():+.3f}; replication vcs3-js2 mean {np.mean(rep):+.3f} (>=.05 in {(np.array(rep)>=0.05).sum()}); forced closed-exact mean {np.mean(fcl):+.3f}; lin-lin {np.mean(fl):+.3f}; mlp-mlp {np.mean(fm):+.3f}")
print("VERDICT:", "VCS objective" if objective else ("critic class / exact solver" if klass else "MIXED"))
```
