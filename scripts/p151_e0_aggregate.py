"""P151 (r3 R3-E0) — re-aggregate the existing estimator records into the three panels the original plan asks for, from raw per-cell files only.

Sources: outputs/P85_estim_benchmark/*.json (P86 benchmark: every method, 3 seeds; corrected SMILE rows from outputs/P85_estim_benchmark_smile_fix/,
which replace P85's void SMILE rows), outputs/P108_estim/*.json (J vs plug-in, budgets 250 / 1000 / 4000).  No fitting; nothing is recomputed from
rounded report numbers.  The registered selection (row flag `selected`) is used per method family and cell.

Panels (dependence ladder I in {2, 4, 6, 8, 10} x generator in {gaussian, cubic, xor}, d 20 / 10 pairs, N 4096, B 256, U 2000):
  E-Precision  same-target methods: posterior MSE and signed J - S (mean +- sd over seeds) with the oracle S; native-target methods: |value - own
               truth| and the value itself (own target) -- never on the S error axis.
  E-Stability  seed sd of the value; evaluation bootstrap SE (fixed fit, 200 draws); training-stability proxy = sd of the SELECT-risk curve over
               its last 5 points (every 100 updates); non-finite steps; fit seconds; critic saturation on EVAL where recorded.
  E-Resolution adjacent-level ordering probability P(value_{k+1} > value_k) from the 200 bootstrap draws of the same seed (coupled by seed,
               averaged over seeds), |Delta mean| / pooled sd, and the oracle Delta S.
Field matrix: presence of oracle / weights / per-sample T / bootstrap / curve / gradient / cost per family (what E1 must add).
P108 panel: J vs S_plug signed bias and RMSE per condition and budget.
    python scripts/p151_e0_aggregate.py --out reports/P151_E0
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

O = Path("/home/infres/yinwang/CS_QMI/outputs")
LADDER = [2, 4, 6, 8, 10]
SAME = ["neural:vcs:inbatch", "neural:vcs:product", "neural:vcs:cyclic8", "neural:js:inbatch", "neural:js:product", "neural:js:cyclic8",
        "s_kde:common_risk", "s_kernel:rff", "s_kernel:nystrom:c512", "rls:tanh"]
NATIVE = ["neural:infonce:inbatch", "neural:infonce:cyclic8", "neural:nwj:inbatch", "neural:nwj:product", "neural:dv:inbatch", "neural:dv:product",
          "neural:smile:inbatch", "neural:smile:product"]


def family(method: str) -> str:
    m = method.rsplit(":lr", 1)[0]
    if m.startswith("s_kernel:rff"):
        return "s_kernel:rff"
    if m.startswith("rls:tanh"):
        return "rls:tanh"
    if m.startswith("rls:raw"):
        return "rls:raw"
    return m


def load_cells():
    cells = {}
    for f in sorted(glob.glob(str(O / "P85_estim_benchmark" / "P85_*.json"))):
        d = json.load(open(f)); c = d["cell"]
        if c["methods"] != "all":
            continue
        cells[c["name"]] = d
    for f in sorted(glob.glob(str(O / "P85_estim_benchmark_smile_fix" / "*.json"))):
        d = json.load(open(f)); name = d["cell"]["name"]
        if name in cells:  # replace void SMILE rows by the corrected ones
            cells[name]["rows"] = [r for r in cells[name]["rows"] if family(r["method"]) not in ("neural:smile:inbatch", "neural:smile:product")] + d["rows"]
            cells[name]["smile_corrected"] = True
    return cells


def selected_rows(d):
    """Registered selection per family; families without any `selected` flag (closed-form kernels with grids) fall back to the best TUNE risk."""
    by = defaultdict(list)
    for r in d["rows"]:
        by[family(r["method"])].append(r)
    out = {}
    for fam, rows in by.items():
        sel = [r for r in rows if r.get("selected")]
        if not sel:
            with_risk = [r for r in rows if r.get("select", {}).get("risk") is not None]
            sel = [min(with_risk, key=lambda r: r["select"]["risk"])] if with_risk else rows[:1]
        out[fam] = sel[0]
    return out


def msd(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    return {"mean": float(x.mean()) if len(x) else None, "sd": float(x.std(ddof=1)) if len(x) > 1 else None, "n": int(len(x))}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="reports/P151_E0"); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cells = load_cells()
    # ---- field matrix (presence per family of record)
    def present(d, path):
        cur = d
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return False
        return cur is not None
    ex = next(iter(cells.values())); row = ex["rows"][0]
    fm = {"P85/P86": {"oracle_S": present(ex, ["truth", "S"]), "oracle_MI": present(ex, ["truth", "MI"]), "selected_weights": False, "per_sample_T_eval": False,
                      "bootstrap_values": present(row, ["native", "boot_values"]), "select_curve": present(row, ["select", "curve"]),
                      "critic_gradient_norms": False, "generator_channel_derivative": present(ex, ["channel_derivative"]), "nonfinite_steps": present(row, ["numerical", "nonfinite_steps"]),
                      "cost_fields": present(row, ["fit_seconds"]), "saturation_summary": present(row, ["posterior", "score_diagnostics"]), "seeds": 3},
          "P108": {"oracle_S": True, "selected_weights": False, "per_sample_T_eval": False, "bootstrap_values": False, "budgets": [250, 1000, 4000], "critic_gradient_norms": False, "cost_fields": True, "seeds": "5 (E1) / 3 (E2)"},
          "missing_for_O1_s12.2": ["critic-parameter gradient norms per update (gradient-norm variance)", "critic-output histograms (only quantiles / saturation fractions)",
                                   "per-sample T on EVAL (needed for paired bootstrap across methods)", "repeated independent EVAL batches for a fixed fit (only a bootstrap of one EVAL block)"]}
    # ---- ladder panels
    lad = defaultdict(lambda: defaultdict(dict))  # gen -> I -> seed -> (truth, selected rows)
    for name, d in cells.items():
        c = d["cell"]
        if c["N"] != 4096 or c["B"] != 256 or c["updates"] != 2000 or c["I"] not in LADDER:
            continue
        if not ((c["setting"] in ("gaussian", "cubic") and c["d_signal"] == 20 and c["d_total"] == 20) or c["setting"] == "xor_mixture"):
            continue
        lad[c["setting"]][int(c["I"])][c["seed"]] = (d["truth"], selected_rows(d))
    prec, stab, reso = {}, {}, {}
    for gen, byI in lad.items():
        prec[gen], stab[gen], reso[gen] = {}, {}, {}
        for I in LADDER:
            if I not in byI:
                continue
            seeds = byI[I]; truth = next(iter(seeds.values()))[0]
            P_ = {"S": truth["S"], "S_se": truth.get("S_se"), "MI": truth.get("MI"), "J_oracle": truth.get("J_oracle"), "same_target": {}, "native_target": {}}
            St = {}
            for fam in SAME + NATIVE:
                rows = [sr[fam] for _, sr in seeds.values() if fam in sr]
                if not rows:
                    continue
                vals = [r["native"]["value"] for r in rows]
                if fam in SAME:
                    P_["same_target"][fam] = {"posterior_mse": msd([r["posterior"].get("posterior_mse") for r in rows]),
                                              "J_minus_S": msd([r["native"]["signed_error"] for r in rows]), "J_eval": msd(vals)}
                else:
                    P_["native_target"][fam] = {"value": msd(vals), "own_truth": rows[0]["native"].get("own_truth"),
                                                "abs_error_to_own_truth": msd([r["native"].get("abs_error") for r in rows])}
                curve_tail = []
                for r in rows:
                    cv = r.get("select", {}).get("curve") or []
                    if len(cv) >= 5:
                        curve_tail.append(float(np.std([p[1] for p in cv[-5:]], ddof=1)))
                sat = [r["posterior"].get("score_diagnostics", {}).get("sat_pos_frac") for r in rows if isinstance(r.get("posterior"), dict)]
                St[fam] = {"seed_sd_value": msd(vals)["sd"], "eval_boot_se_mean": msd([r["native"].get("boot_se") for r in rows])["mean"],
                           "select_curve_tail_sd": msd(curve_tail)["mean"], "nonfinite_steps_total": int(sum(r.get("numerical", {}).get("nonfinite_steps", 0) or 0 for r in rows)),
                           "fit_seconds_mean": msd([r.get("fit_seconds") for r in rows])["mean"], "n_params": rows[0].get("n_params"),
                           "saturation_pos_frac": msd([s for s in sat if s is not None])["mean"] if any(s is not None for s in sat) else None}
            prec[gen][I] = P_; stab[gen][I] = St
        # resolution between adjacent levels
        for k in range(len(LADDER) - 1):
            I0, I1 = LADDER[k], LADDER[k + 1]
            if I0 not in byI or I1 not in byI:
                continue
            R_ = {"delta_S_oracle": float(next(iter(byI[I1].values()))[0]["S"] - next(iter(byI[I0].values()))[0]["S"]), "methods": {}}
            for fam in SAME + NATIVE:
                probs, d_means, sds = [], [], []
                for s in sorted(set(byI[I0]) & set(byI[I1])):
                    r0, r1 = byI[I0][s][1].get(fam), byI[I1][s][1].get(fam)
                    if not r0 or not r1:
                        continue
                    b0, b1 = r0["native"].get("boot_values"), r1["native"].get("boot_values")
                    if b0 and b1:
                        b0, b1 = np.asarray(b0, float), np.asarray(b1, float); probs.append(float((b1 > b0).mean())); sds.append(float(np.sqrt(0.5 * (b0.var(ddof=1) + b1.var(ddof=1)))))
                    d_means.append(r1["native"]["value"] - r0["native"]["value"])
                if d_means:
                    dm = np.array(d_means); pooled = float(np.sqrt(np.mean(np.square(sds)) + dm.var(ddof=1))) if len(dm) > 1 and sds else (float(np.mean(sds)) if sds else None)
                    R_["methods"][fam] = {"adjacent_order_probability": float(np.mean(probs)) if probs else None, "coupled_by": "fit seed (same seed, bootstrap draws paired by index)",
                                          "delta_mean": float(dm.mean()), "pooled_sd": pooled, "standardised": (float(abs(dm.mean()) / pooled) if pooled else None), "n_seeds": int(len(dm))}
            reso[gen][f"{I0}->{I1}"] = R_
    # ---- N axis and batch axes (gaussian d20 I4) for VCS / JS / MI methods
    axes = {"N_axis": defaultdict(dict), "batch_equal_updates": defaultdict(dict), "batch_equal_exposure": defaultdict(dict)}
    for name, d in cells.items():
        c = d["cell"]
        if not (c["setting"] == "gaussian" and c["d_signal"] == 20 and c["d_total"] == 20 and c["I"] == 4.0):
            continue
        sr = selected_rows(d)
        key = None
        if c["B"] == 256 and c["updates"] == 2000:
            key = ("N_axis", c["N"])
        elif c["N"] == 4096 and c["updates"] == 2000:
            key = ("batch_equal_updates", c["B"])
        elif c["N"] == 4096 and (c["B"], c["updates"]) in ((64, 8000), (1024, 500)):
            key = ("batch_equal_exposure", f"B{c['B']}_U{c['updates']}")
        if key is None:
            continue
        for fam in SAME + NATIVE:
            if fam in sr:
                axes[key[0]].setdefault(str(key[1]), {}).setdefault(fam, []).append({"posterior_mse": sr[fam]["posterior"].get("posterior_mse"), "value": sr[fam]["native"]["value"],
                                                                                     "abs_error": sr[fam]["native"].get("abs_error"), "nonfinite": sr[fam].get("numerical", {}).get("nonfinite_steps")})
    axes_s = {ax: {lvl: {fam: {"posterior_mse": msd([x["posterior_mse"] for x in xs]), "abs_error": msd([x["abs_error"] for x in xs]), "nonfinite_total": int(sum((x["nonfinite"] or 0) for x in xs))}
                         for fam, xs in fams.items()} for lvl, fams in sorted(levels.items(), key=lambda kv: float(re.sub(r'[^0-9.]', '', kv[0].split('_')[0]) or 0))}
              for ax, levels in axes.items()}
    # ---- P108: J vs S_plug per condition / budget
    p108 = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for f in sorted(glob.glob(str(O / "P108_estim" / "*.json"))):
        d = json.load(open(f))
        if "rows" not in d or "condition" not in d:
            continue
        for r in d["rows"]:
            if not r.get("selected", True):
                continue
            p108[d["condition"]][f"N{d['N']}"][f"{r['kind']}@{r['budget_updates']}"].append({"J_err": r.get("J_err"), "S_plug_err": r.get("S_plug_err"), "posterior_mse": r.get("posterior_mse")})
    p108_s = {c: {n: {k: {"J_bias": msd([x["J_err"] for x in v])["mean"], "J_rmse": float(np.sqrt(np.mean([x["J_err"] ** 2 for x in v if x["J_err"] is not None]))) if v else None,
                          "S_plug_bias": msd([x["S_plug_err"] for x in v])["mean"], "S_plug_rmse": float(np.sqrt(np.mean([x["S_plug_err"] ** 2 for x in v if x["S_plug_err"] is not None]))) if v else None, "n": len(v)}
                      for k, v in ks.items()} for n, ks in ns.items()} for c, ns in p108.items()}
    json.dump({"field_matrix": fm, "cells_loaded": len(cells), "ladder_cells": {g: {I: len(s) for I, s in b.items()} for g, b in lad.items()}}, open(out / "field_matrix.json", "w"), indent=1)
    json.dump(prec, open(out / "e_precision.json", "w"), indent=1); json.dump(stab, open(out / "e_stability.json", "w"), indent=1); json.dump(reso, open(out / "e_resolution.json", "w"), indent=1)
    json.dump(axes_s, open(out / "axes_N_batch.json", "w"), indent=1); json.dump(p108_s, open(out / "p108_j_vs_plugin.json", "w"), indent=1)
    # ---- markdown tables (gaussian ladder as the main table; others in JSON)
    L = ["# P151 E0 — precision / stability / resolution re-aggregated from the P85 / P86 / P108 records", "",
         f"Cells loaded: {len(cells)} (full-method cells; corrected SMILE rows substituted).  Ladder cells per generator: " + json.dumps({g: {I: len(s) for I, s in b.items()} for g, b in lad.items()}), ""]
    for gen in prec:
        L += [f"## {gen}: E-Precision (same target S; mean ± sd over seeds)", "", "| I | S (oracle) | " + " | ".join(SAME) + " |", "|---|---|" + "---|" * len(SAME)]
        for I in LADDER:
            if I in prec[gen]:
                P_ = prec[gen][I]
                L.append(f"| {I} | {P_['S']:.3f} | " + " | ".join((f"{v['posterior_mse']['mean']:.3f} ± {v['posterior_mse']['sd']:.3f}" if (v := P_['same_target'].get(f)) and v['posterior_mse']['mean'] is not None and v['posterior_mse']['sd'] is not None else "—") for f in SAME) + " |")
        L += ["", f"## {gen}: native-target methods (|value − own truth|; mean ± sd; MI methods on their own target)", "", "| I | MI | " + " | ".join(NATIVE) + " |", "|---|---|" + "---|" * len(NATIVE)]
        for I in LADDER:
            if I in prec[gen]:
                P_ = prec[gen][I]
                L.append(f"| {I} | {P_['MI']} | " + " | ".join((f"{v['value']['mean']:.3f} (err {v['abs_error_to_own_truth']['mean']:.3f})" if (v := P_['native_target'].get(f)) and v['value']['mean'] is not None and v['abs_error_to_own_truth']['mean'] is not None else "—") for f in NATIVE) + " |")
        L += ["", f"## {gen}: E-Stability at I = 6 (seed sd / eval boot SE / select-curve tail sd / non-finite steps / fit s)", "", "| method | seed sd | eval boot SE | curve tail sd | nonfinite | fit s |", "|---|---|---|---|---|---|"]
        if 6 in stab[gen]:
            for fam, s in stab[gen][6].items():
                L.append(f"| {fam} | {s['seed_sd_value'] if s['seed_sd_value'] is None else round(s['seed_sd_value'], 4)} | {s['eval_boot_se_mean'] if s['eval_boot_se_mean'] is None else round(s['eval_boot_se_mean'], 4)} | "
                         f"{s['select_curve_tail_sd'] if s['select_curve_tail_sd'] is None else round(s['select_curve_tail_sd'], 4)} | {s['nonfinite_steps_total']} | {s['fit_seconds_mean'] if s['fit_seconds_mean'] is None else round(s['fit_seconds_mean'], 1)} |")
        L += ["", f"## {gen}: E-Resolution (adjacent ladder levels; P(value_k+1 > value_k) from seed-coupled bootstrap draws; |Δ| / pooled sd)", "", "| step | ΔS | " + " | ".join(SAME[:2] + NATIVE[:1] + NATIVE[2:3] + NATIVE[4:5] + NATIVE[6:7]) + " |", "|---|---|" + "---|" * 6]
        for step, R_ in reso[gen].items():
            L.append(f"| {step} | {R_['delta_S_oracle']:.3f} | " + " | ".join((f"{m['adjacent_order_probability']:.2f} ({m['standardised']:.1f})" if (m := R_['methods'].get(f)) and m['adjacent_order_probability'] is not None and m['standardised'] is not None else "—") for f in SAME[:2] + NATIVE[:1] + NATIVE[2:3] + NATIVE[4:5] + NATIVE[6:7]) + " |")
        L.append("")
    L += ["## P108: J vs plug-in (signed bias / RMSE over seeds) per condition, N and budget", ""]
    for c, ns in p108_s.items():
        for n, ks in ns.items():
            L.append(f"- {c} {n}: " + "; ".join(f"{k}: J {v['J_bias']:+.4f}/{v['J_rmse']:.4f}, S_plug {v['S_plug_bias']:+.4f}/{v['S_plug_rmse']:.4f}" for k, v in ks.items() if v['J_bias'] is not None))
    L += ["", "## Field matrix", "", "```", json.dumps(fm, indent=1), "```"]
    (out / "tables.md").write_text("\n".join(L))
    print(f"{len(cells)} cells -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
