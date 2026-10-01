# P74 — T1: within-class conditional dependence test with fair controls — report — 2026-10-01

Pre-registration `P73_CONDITIONAL_TEST_T1_PREREG_FROZEN_20260928.md` and its runtime-split addendum `P73_ADDENDUM1_RUNTIME_SPLIT_FROZEN_20260928.md`.
Results-only commit `dcfa9ff`: the merged tables are `P74_cond_test_t1_{vcs4v800,simclr}_colour_merged.{json,md}` and `P74_cond_test_t1_simclr_blur_merged.{json,md}`.
Their power cells come from the monolithic jobs and their null cells from the 14 split jobs. VCS blur is `P74_cond_test_t1_vcs4v800_blur.{json,md}`, the monolithic job, which ran to completion as the addendum allows.
The read-out applies the frozen rules exactly. The rule script is reproduced in §6.

## 1. Verdict: **T1 holds**
- **Informative cells.** 16 (encoder, family, strength, n) planted cells have at least one test with power ≥ 0.5: 9 VCS-colour, 1 SimCLR-colour, 4 VCS-blur and 2 SimCLR-blur.
- **Holds rule.** In all 16 cells the VCS conditional permutation test has power ≥ max(controls) − 0.10.
  The one cell where a control is ahead is VCS-encoder colour s = 0.2, n = 200: single-bandwidth HSIC 1.00 against VCS 0.96, a deficit of 0.04, inside the tolerance.
- **Refuted rule (a): JS within 0.05 of VCS in every informative cell.** Not met. JS is within 0.05 in 9 of 16 cells, and 7 of those 9 are saturated, with every test at ≥ 0.97.
- **Refuted rule (b): some control ≥ 0.10 above VCS in at least half of the cells.** Not met. No control exceeds VCS by ≥ 0.10 in any cell.
- **Per group.** Each of the four encoder × family groups also satisfies the holds rule on its own.

## 2. Level (null cells)
The pre-committed level check uses R = 1000, on both colour nulls, for both encoders, at n ∈ {500, 2000}. That gives 8 cells × 7 tests.
- **Every rejection rate is ≤ 0.065**, so no test is invalid at any n and every power cell is read.
- **The maximum is 0.065, reached twice, exactly at the bound:** single-bandwidth HSIC on VCS colour `null_all_planted0.2` at n = 2000, and per-class HSIC on SimCLR colour `null_all_planted0.2` at n = 500.
- **VCS permutation test:** between 0.043 and 0.054 across the 8 cells.

The R = 100 null cells are reported, not gated, as pre-registered. These are the colour nulls at n ∈ {200, 1000} and all blur nulls.
- Two VCS cells exceed 0.09 at R = 100: SimCLR colour `null_label_only` at n = 1000 (0.10) and VCS blur `null_label_only` at n = 200 (0.11).
- At R = 100 the binomial 95 % half-width around 0.05 is ±0.043, so these are 1.2–1.4 half-widths above nominal.
- Both share their (encoder, family, n) with R = 1000 cells that sit at level, or lack such a cell entirely, so the frozen invalidity rule (R = 1000 only) does not apply.
- The second of them sits under the informative cell VCS blur s = 0.5, n = 200, where VCS scores 0.57 against single-bandwidth HSIC at 0.55. Dropping that cell leaves the verdict unchanged:
  the holds rule is a per-cell minimum, and the refutation counts are 0 either way.

## 3. Informative cells — VCS against the best control and against JS

| group | case | n | VCS perm | best control | Δ vs best | JS (same critic fitter) | Δ vs JS |
|---|---|---|---|---|---|---|---|
| VCS colour | s 0.05 | 1000 | 0.61 | c2st_logit 0.46 | +0.15 | 0.43 | +0.18 |
| VCS colour | s 0.05 | 2000 | 0.96 | c2st_logit 0.82 | +0.14 | 0.80 | +0.16 |
| VCS colour | s 0.1 | 500 | 0.86 | hsic_perm 0.84 | +0.02 | 0.82 | +0.04 |
| VCS colour | s 0.2 | 200 | 0.96 | hsic_perm 1.00 | −0.04 | 0.84 | +0.12 |
| SimCLR colour | s 0.2 | 2000 | 0.94 | js_perm 0.72 | +0.22 | 0.72 | +0.22 |
| VCS blur | σ 0.5 | 200 | 0.57 | hsic_perm 0.55 | +0.02 | 0.44 | +0.13 |
| VCS blur | σ 0.5 | 500 | 0.98 | hsic_perm 0.95 | +0.03 | 0.94 | +0.04 |
| SimCLR blur | σ 0.5 | 1000 | 0.79 | c2st_logit 0.56 | +0.23 | 0.54 | +0.25 |
| SimCLR blur | σ 0.5 | 2000 | 1.00 | js_perm 0.87 | +0.13 | 0.87 | +0.13 |
| (7 saturated cells: VCS colour s 0.1 n 1000 / 2000 and s 0.2 n 500 / 1000 / 2000; VCS blur σ 0.5 n 1000 / 2000) | | | 1.00 | ≥ 0.97 | ≤ +0.03 | ≥ 0.98 | ≤ +0.02 |

What the table shows:
- **No cell where VCS loses by more than 0.04.** In the 9 unsaturated informative cells VCS leads the best control by 0.13–0.23 in 5 cells and is level, within ±0.04, in the other 4.
- **The learned JS statistic is behind VCS by 0.12–0.25 in 7 of the 9 unsaturated cells.** It uses the same fitter, early stopping and FIT / VAL split.
- **The motivating P46 cell reproduces with fair controls.** That cell is SimCLR blur, n = 2000, where VCS scored 0.98 against HSIC at 0.18.
  Here VCS has 1.00, single-bandwidth HSIC 0.30, per-class-bandwidth HSIC 0.46, deep-kernel HSIC 0.64, C2ST 0.84 and JS 0.87.
  The bandwidth fixes and the learned kernel recover part of the gap but not all of it.
- **The encoder matters more than the test.** On the VCS encoder, single-bandwidth HSIC is nearly level with VCS. The VCS features expose the planted nuisance more linearly.
  On the SimCLR encoder every kernel control falls well behind. Beyond the informative cells, SimCLR colour is uninformative at s ≤ 0.1 for every test, with power ≤ 0.31 at n = 2000.

## 4. Guaranteed (Hoeffding) test — reported, not in the verdict
- **It never rejects,** in any cell, for any of the three critic classes (linear, MLP, closed-form).
- **The reason is scale.** At n = 2000 the observed J is about 1e-3 to 1e-2, while the bound τ_{n,n}(0.05) is about 0.13. The effects planted here are one to two orders of magnitude below what the distribution-free bound can certify at these n.

## 5. QC sentinels
- **Alignment.** uid / y alignment and N = 0 feature equality hold in every job, by load-time assertion.
- **`n1_frac_eval`.** The frozen sentinel is "within ± 0.05 of 0.5" per repeat.
  - **Out-of-band rates by n:**

    | n | repeats outside the band |
    |---|---|
    | 200 | 228 / 1800 (12.7 %) |
    | 500 | 128 / 5400 (2.4 %) |
    | 1000 | 3 / 1800 (0.2 %) |
    | 2000 | 0 / 5400 |

  - **Expected from noise alone:** for a Binomial(n, ½) fraction this rate is 15 %, 2.5 %, 0.2 % and 0.0 % respectively. The observed rates match that, so they show no bias in the construction.
  - **Observed range** is 0.395 to 0.60.
  - **Caveat on the rule itself.** As worded, the sentinel is a per-repeat bound that sampling noise alone violates at small n. This is disclosed, and nothing was re-drawn.
- **VAL-picked VCS critic,** counted over all repeats:

  | group | closed-form | MLP | linear |
  |---|---|---|---|
  | VCS colour | 4589 | 1002 | 9 |
  | SimCLR colour | 4114 | 1471 | 15 |
  | VCS blur | 1283 | 313 | 4 |
  | SimCLR blur | 1171 | 422 | 7 |
- **Deep-kernel HSIC.** Its best step was never 0. It always learned something, so its weakness is not a training failure.
- **Provenance.** The 14 split jobs carry seeds 101–114, matching the addendum. Each null file comes from one completed run, and every log shows exit 0.
  Some SimCLR split jobs were first queued on the preemptible runfill QOS. They were preempted before finishing and resubmitted on the normal QOS with identical arguments and seeds.
  The runner cannot resume, so preempted attempts wrote only `.partial.json` files, and none of those enter the merge.
  Code is commit `6e709ba` plus the uncommitted P95/P100-era working tree; the T1 runner itself is unchanged since the P73 freeze.

## 6. Scope and caveats
- **One caveat on attribution.** The VCS permutation test chooses on VAL among three critic classes. One of them, the closed-form linear critic, has no JS counterpart, and it is picked in 73–82 % of repeats.
  The JS control uses linear and MLP classes with the same fitter. So "VCS beats JS" means the VCS objective together with its closed-form critic beats JS with learned linear or MLP critics.
  This run cannot separate the objective from the closed-form class. A VCS test restricted to linear and MLP critics would be the clean ablation; it is not run and not claimed.
- **Not claimed (as pre-registered):** continuous or multi-class N; other encoders or datasets; nuisances other than the two planted families; the unconditional test.
- **Rule script.** `scratchpad t1_verdict.py`, reproduced below so the verdict can be re-run:
  level gate at R = 1000 with threshold > 0.09; a cell is informative if some test other than Hoeffding reaches ≥ 0.5;
  holds / refuted / conditional exactly as in P73 §"Pre-committed reading".

## Delivery
```yaml
experiment_family: dependence_testing (conditional, within-class)
protocol_id: P73 / P74 (T1) with P73 addendum 1
source_commit: 6e709ba (runner); results dcfa9ff
estimator: VCS permutation (lin / MLP / closed-form, VAL-picked) vs HSIC (median bw, per-class bw, deep kernel), C2ST-logit, JS (lin / MLP)
evaluation_readout: rejection rate at δ 0.05, B 200 shared within-class permutations; R 100 power, R 1000 level (colour, n 500 / 2000)
n_independent_units: repeats (fresh N | Y and subsample per repeat)
status: complete — T1 holds
```

```python
import json
R="/home/infres/yinwang/CS_QMI/ssl_pilot/reports/"
SRC={"vcs4v800_colour":"P74_cond_test_t1_vcs4v800_colour_merged.json","simclr_colour":"P74_cond_test_t1_simclr_colour_merged.json",
     "vcs4v800_blur":"P74_cond_test_t1_vcs4v800_blur.json","simclr_blur":"P74_cond_test_t1_simclr_blur_merged.json"}
TESTS=("vcs_perm","vcs_hoeff","hsic_perm","hsic_class","hsic_deep","c2st_logit","js_perm"); CTRL=("hsic_perm","hsic_class","hsic_deep","c2st_logit","js_perm")
inval={}; print("== LEVEL (null cells) ==")
for g,f in SRC.items():
    d=json.load(open(R+f))
    for cn,c in d["cases"].items():
        if c["mode"]=="planted": continue
        for n in sorted(c["by_n"],key=int):
            r=c["by_n"][n]; Rr=r["repeats"]; p={t:r["summary"][t]["power"] for t in TESTS}
            flag=[t for t in TESTS if p[t]>0.09]; over=[t for t in TESTS if 0.065<p[t]<=0.09]
            if Rr>=1000:
                for t in flag: inval.setdefault((g,int(n)),set()).add(t)
            print(f"{g:16s} {cn:28s} n={n:>4} R={Rr:>4} "+" ".join(f"{t.split('_')[0][:4]}{t.split('_')[1][:4]}={p[t]:.3f}" for t in TESTS)+(f"  >0.09:{flag}" if flag else "")+(f" (0.065-0.09:{over})" if over else ""))
print("\ninvalid (R=1000, >0.09):",{k:sorted(v) for k,v in inval.items()})
print("\n== POWER (planted) ==")
cells=[]
for g,f in SRC.items():
    d=json.load(open(R+f))
    for cn,c in d["cases"].items():
        if c["mode"]!="planted": continue
        for n in sorted(c["by_n"],key=int):
            sm=c["by_n"][n]["summary"]; p={t:sm[t]["power"] for t in TESTS}
            bad=inval.get((g,int(n)),set())
            # a level-invalid test at that n is not read
            readable={t:p[t] for t in TESTS if t not in bad and t!="vcs_hoeff"}
            info=max(readable.values())>=0.5
            cells.append((g,cn,int(n),p,bad,info,readable))
            print(f"{g:16s} {cn:14s} n={n:>4} "+" ".join(f"{t}={p[t]:.2f}" for t in TESTS)+("  INFO" if info else "")+(f" notread:{sorted(bad)}" if bad else ""))
inf=[c for c in cells if c[5]]
def verdict(cs,label):
    if not cs: print(label,"no informative cells"); return
    holds=all("vcs_perm" in c[6] and c[6]["vcs_perm"]>=max(c[6][t] for t in CTRL if t in c[6])-0.10 for c in cs)
    jspar=all(abs(c[6].get("js_perm",9)-c[6].get("vcs_perm",-9))<=0.05 for c in cs)
    worse={t:sum(1 for c in cs if t in c[6] and "vcs_perm" in c[6] and c[6][t]-c[6]["vcs_perm"]>=0.10) for t in CTRL}
    half=any(v>=len(cs)/2 for v in worse.values())
    fails=[(c[0],c[1],c[2],round(c[6].get('vcs_perm',float('nan')),2),max(((t,round(c[6][t],2)) for t in CTRL if t in c[6]),key=lambda x:x[1])) for c in cs if not("vcs_perm" in c[6] and c[6]["vcs_perm"]>=max(c[6][t] for t in CTRL if t in c[6])-0.10)]
    js_within=sum(1 for c in cs if abs(c[6].get("js_perm",9)-c[6].get("vcs_perm",-9))<=0.05)
    v="REFUTED" if (jspar or half) else ("HOLDS" if holds else "CONDITIONAL")
    print(f"\n{label}: informative={len(cs)} holds_rule={holds} js_within0.05={js_within}/{len(cs)} ctrl_exceed>=0.10 counts={worse} -> {v}")
    for x in fails: print("   deficit cell:",x)
verdict(inf,"POOLED")
for g in SRC: verdict([c for c in inf if c[0]==g],g)
```
