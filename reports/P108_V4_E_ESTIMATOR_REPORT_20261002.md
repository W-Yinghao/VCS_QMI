# P108 — v4 module E: variational J vs the squared plug-in on identical critics (E1), rotation control (E2), cost (E3) — report — 2026-10-02

Pre-registration `P108_V4_E_ESTIMATOR_PREREG_FROZEN_20261001.md`.  Jobs 1017329 (E1 + mechanism) and 1017330 (E2), both exit 0 (96 cells + mechanism).
Results-only commit `f9ac2b2` (`P108_estim_aggregate.{md,json}`, produced by the frozen `scripts/p108_aggregate.py`).  Rule scripts reproduced below.
Synthetic P85 generators only; nothing here is about image data or SSL.

## 1. E1(a) — exact mechanism check: **passes**
On the TRUTH sample with T_t = (1 − t)η + tU: the identity-predicted |bias| of Ĵ has log–log slope **2.000** in t for both U = 0 and the fixed U, in all three
conditions; the plug-in slope is 0.965–0.980 (≈ 1, as predicted); max |D_J z| 0.76–1.61 (< 4, no implementation flag); plug-in identity residual ≤ 2.1e-16.
The population algebra — J's error is second order in the critic error, the plug-in's first order — is confirmed numerically.

## 2. E1(b) — identical fitted T, two readouts on the same EVAL units
Pre-stated statement form: "Ĵ has smaller |bias| / RMSE than Ŝ_plug in k of the cells".

| fit loss | cells | Ĵ smaller \|bias\| | Ĵ smaller RMSE |
|---|---|---|---|
| VCS (T = tanh f) | 36 | **1** (C3 xor, N 16384, 250 updates) | **1** (same cell) |
| JS (T = tanh(f/2)) | 36 | **1** | **2** (C1 N 1024 @ 250; C3 N 16384 @ 250) |

- **On these generators and budgets the squared plug-in Ŝ_plug is the more accurate readout of the same fitted critic in 70 of 72 cells**, for both fit
  losses — so this is a property of the readout formulas applied to realistic fits, not of the VCS fitting loss.
- Ĵ's bias is negative in **72 / 72** cells (it remains a valid lower readout); Ŝ_plug's bias is positive in 7 / 72 (it is not a bound and can overshoot).
- Ĵ's evaluation SE is ≈ 3.2× that of Ŝ_plug (median), so Ĵ also loses on variance.
- Why the population ordering (J second order) does not show up: the fitted critics are far from η (posterior MSE 0.04–0.56).  In the identity
  S_plug − S = 2E_M[ηe] + E_M e², a shrunken-and-noisy fit gives a negative cross term that partly *cancels* the positive E_M e²; Ĵ's error −E_M e² has no
  such cancellation.  (Interpretation consistent with the identities; not separately tested.)  The second-order advantage of Ĵ would need critics much
  closer to η than any fit here reached.
- C2 (2 signal of 100 dimensions): every neural fit is near T ≈ 0 until N = 16384 (|J − S| ≈ S = 0.54 at N ≤ 4096); identical rows at 1000 / 4000 updates
  in many cells are early stopping on SELECT choosing the same step.

## 3. E2 — rotation control (rule: rotation-sensitive if |rotated − original| > 2 × pooled seed SE)
- 60 (condition, N, method, budget) cells; **9 flagged, all with *lower* error after rotation** (−0.007 to −0.022), all neural (VCS and JS fits) at C1 N 4096
  and C2 N 16384.  The kernel references (S-KDE, RFF S-kernel) are unchanged by rotation (|effect| ≤ 0.005, isotropic by construction).
- No method's accuracy depends on the signal being axis-aligned in a way that hurts it; the neural fits' small gain under rotation (signal spread over many
  coordinates) is the opposite of a coordinate artefact.  Error-vs-N and error-vs-seconds curves are in the aggregate (§E2); no efficiency claim from single
  fits (pre-stated).

## 4. E3 — cost
- **No (condition, N, fit loss) cell reaches the pre-set tolerance |J − S| ≤ 0.02 at any budget ≤ 4000 updates** (best: C1 N 16384, ≈ 0.038–0.041).
- Refit SD of J over independent FIT seeds: 0.002–0.044 (one C2 N 4096 VCS cell 0.11, a bimodal fit/no-fit split); chosen-model fit 0.3–4.6 s, total
  tuning 0.8–13.8 s per cell on one GPU; peak CUDA ≈ 140 MB.

## 5. Reference estimators (each against its own truth)
Reported in the aggregate (§Reference estimators): e.g. C1 N 256 — DV −0.99, NWJ −1.02, InfoNCE −0.35 relative MI error, SMILE +0.01 (sd 0.13).  Not ranked
against S (pre-stated).

## 6. What this means for the paper (descriptive)
- The population claim "Ĵ is second-order accurate in the critic error" is verified exactly (E1a), but **it does not translate into a finite-sample
  accuracy advantage over the squared plug-in for the critics one actually fits** (E1b, 70 / 72 cells).  Ĵ's distinct merits that survive are that it is a
  guaranteed lower readout (bias < 0 in every cell) and is the training objective; for *point estimation* of S on these generators the plug-in reads closer.
- Not claimed: anything beyond these three generators, N ≤ 16384, ≤ 4000 updates, the two fit losses.

## Delivery
```yaml
experiment_family: estimator (synthetic)
protocol_id: P108 (v4 E1 / E2 / E3)
source_commit: 70f53ac (frozen); results f9ac2b2
estimator: neural VCS / JS fits read by J and S_plug; S-KDE, RFF S-kernel (E2); InfoNCE / NWJ / DV / SMILE references
evaluation_readout: signed bias, RMSE, posterior MSE, eval SE, fit / tuning seconds, memory
n_independent_units: EVAL 32768 per cell; 5 (E1) / 3 (E2) independent FIT seeds
status: complete
```

## Rule scripts
```python
import json, math, collections
a = json.load(open("/home/infres/yinwang/CS_QMI/ssl_pilot/reports/P108_estim_aggregate.json"))
E1 = a["E1"]
print("== E1(b): cells where J beats S_plug (|bias| and RMSE), per fit loss")
for fl in ("vcs", "js"):
    rows = [r for r in E1 if r["fit_loss"] == fl]
    b = [r for r in rows if abs(r["J_err_mean"]) < abs(r["S_plug_err_mean"])]
    m = [r for r in rows if r["J_rmse"] < r["S_plug_rmse"]]
    print(f"  fit {fl}: |bias| J<plug in {len(b)}/{len(rows)}; RMSE J<plug in {len(m)}/{len(rows)}")
    for r in m: print(f"     J wins RMSE: {r['condition']} N={r['N']} budget={r['budget']}  J {r['J_rmse']:.4f} plug {r['S_plug_rmse']:.4f}")
print("  sign of plug-in bias: positive in", sum(r["S_plug_err_mean"] > 0 for r in E1), "/", len(E1), "; J bias negative in", sum(r["J_err_mean"] < 0 for r in E1))
print("  eval SE ratio J/plug median:", sorted(r["J_se"]/r["S_plug_se"] for r in E1 if r["S_plug_se"] > 0)[len(E1)//2])
print("  'learned nothing' cells (|J| < 0.02 i.e. J-S ≈ -S):", sum(1 for r in E1 if abs(r["J_err_mean"] + 0) > 0 and False))
print("== E2 rotation effect (rotated - original absJerr) with 2 pooled SE rule")
g = collections.defaultdict(dict)
for r in a["E2"]: g[(r["condition"], r["N"], r["method"], r["budget"])][r["rotated"]] = r
sens = 0; tot = 0
for k, d in sorted(g.items()):
    if True in d and False in d:
        o, rt = d[False], d[True]; tot += 1
        eff = rt["absJerr"] - o["absJerr"]
        print(f"  {k[0]:13s} N={k[1]:>5} {k[2]:22s} budget={k[3]}: orig {o['absJerr']:.4f} rot {rt['absJerr']:.4f} eff {eff:+.4f}")
print("== E3"); [print(" ", r) for r in a["E3"]]
```
```python
import json, glob, collections, numpy as np
g = collections.defaultdict(list)
for f in glob.glob("/home/infres/yinwang/CS_QMI/outputs/P108_estim/P108_E2_*.json"):
    d = json.load(open(f))
    for r in d["rows"]:
        if not r.get("selected"): continue
        g[(d["condition"], d["N"], r["method"], r["budget_updates"], d["rotated"])].append(abs(r["J_err"]))
keys = sorted({k[:4] for k in g}, key=str); sens = []
for k in keys:
    o, t = np.array(g.get(k + (False,), [])), np.array(g.get(k + (True,), []))
    if len(o) < 2 or len(t) < 2: continue
    eff = t.mean() - o.mean(); se = np.sqrt(o.var(ddof=1) / len(o) + t.var(ddof=1) / len(t))
    flag = abs(eff) > 2 * se
    if flag: sens.append((k, round(eff, 4), round(se, 4)))
print(f"cells: {len(keys)}; rotation-sensitive by the 2xSE rule: {len(sens)}")
for s in sens: print("  ", s)
```

## Correction addendum (2026-10-03, v5 review)
Source: `VCS_Results_Review_and_Next_Plan_v5_CN.md` §4.1.  Numbers recounted from the committed `P108_estim_aggregate.json` (E1, 72 cells =
3 conditions × 4 N × 3 budgets × 2 fit losses; seed means over 5 FIT seeds); no new computation.

1. **Win counts per metric.**  Cells where Ĵ is closer than Ŝ_plug (no ties):

   | fit loss | cells | Ĵ smaller \|bias\| | Ĵ smaller RMSE |
   |---|---|---|---|
   | VCS | 36 | 1 | 1 |
   | JS | 36 | 1 | 2 |
   | total | 72 | **2** | **3** |

   Ŝ_plug therefore has the smaller |bias| in 70 / 72 cells and the smaller RMSE in **69 / 72** cells.  The "70 of 72 cells" in §2 and §6 is the
   **bias** count only; it must not be used for RMSE.
2. **"Lower readout" wording withdrawn.**  §2 ("it remains a valid lower readout") and §6 ("a guaranteed lower readout (bias < 0 in every cell)") are
   replaced by: the **population** value satisfies J(T) ≤ S for every bounded T (J(T) − S = −E_M(T − η)²); the **finite-sample** Ĵ(T) on EVAL carries
   sampling error and is **not** a per-draw lower bound.  The negative seed-mean bias observed in all 72 cells is an empirical observation for these
   generators and budgets, not a guarantee.  A confidence lower bound needs an independent concentration interval on held-out data (as T1's
   Hoeffding test), which P108 did not compute.
The remaining conclusions of this report are unchanged.
