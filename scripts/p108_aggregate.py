"""P108 aggregate: E1 (readouts on identical critics), E2 (rotation, error vs samples / budget / seconds), E3 (cost, stopping and refit spread).

    python scripts/p108_aggregate.py --in-dir outputs/P108_estim --out reports/P108_estim_aggregate
Only lr-selected rows (lowest SELECT risk, as P85) enter the main tables; every lr row stays in the per-cell JSONs.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

TOL = 0.02  # pre-declared absolute tolerance on |J_eval − S| for the E3 "budget to reach" readout


def ms(v):
    v = np.asarray([x for x in v if x is not None and math.isfinite(x)], float)
    return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, len(v)) if len(v) else (float("nan"), float("nan"), 0)


def rmse(v):
    v = np.asarray(v, float); return float(np.sqrt((v ** 2).mean())) if len(v) else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--in-dir", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    cells = [json.load(open(f)) for f in sorted(glob.glob(f"{a.in_dir}/P108_E*.json")) if "_smoke" not in f and "mechanism" not in f]
    mech = [json.load(open(f)) for f in glob.glob(f"{a.in_dir}/P108_E1_mechanism.json")]
    L = ["# P108 aggregate (package v4 module E)", ""]; out = {"E1": [], "E1_refs": [], "E2": [], "E3": []}
    # ---------------------------------------------------------------- E1 mechanism
    if mech:
        L += ["## E1(a) mechanism (T_t = (1 − t) eta + t U on the TRUTH sample; slopes of the identity-predicted |bias| in t ≤ 0.1)", "",
              "| condition | U | slope J | slope S_plug | max |D_J z| | max plug-in identity residual |", "|---|---|---|---|---|---|"]
        for c, r in mech[0]["conditions"].items():
            for u, v in r["U"].items():
                L.append(f"| {c} | {u} | {v['loglog_slope_t_le_0.1']['J']:.3f} | {v['loglog_slope_t_le_0.1']['S_plug']:.3f} | {v['max_abs_D_J_z']:.2f} | {v['max_abs_S_plug_identity_residual']:.1e} |")
        L.append("")
    # ---------------------------------------------------------------- E1 fits
    g = defaultdict(list); gref = defaultdict(list)
    for c in cells:
        if c["unit"] != "E1":
            continue
        for r in c["rows"]:
            if r["selected"]:
                g[(c["condition"], c["N"], r["kind"], r["budget_updates"])].append((r, c["truth"]["S"]))
        for r in c.get("reference_rows", []):
            if r["selected"]:
                gref[(c["condition"], c["N"], r["kind"])].append(r)
    L += ["## E1(b) identical fitted T, two readouts on the same EVAL units (lr-selected; mean over independent FIT seeds)", "",
          "| condition | N | fit loss | budget | n seeds | J − S mean (sd) | J RMSE | S_plug − S mean (sd) | S_plug RMSE | posterior MSE | eval SE J / S_plug | fit s (chosen / tuning) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in sorted(g):
        rs = [x[0] for x in g[k]]; ej = [r["J_err"] for r in rs]; ep = [r["S_plug_err"] for r in rs]
        mj, sj, n = ms(ej); mp, sp, _ = ms(ep)
        row = {"condition": k[0], "N": k[1], "fit_loss": k[2], "budget": k[3], "n": n, "J_err_mean": mj, "J_err_sd": sj, "J_rmse": rmse(ej), "S_plug_err_mean": mp,
               "S_plug_err_sd": sp, "S_plug_rmse": rmse(ep), "posterior_mse": ms([r["posterior_mse"] for r in rs])[0], "J_se": ms([r["J_se"] for r in rs])[0],
               "S_plug_se": ms([r["S_plug_se"] for r in rs])[0], "fit_s": ms([r["fit_seconds"] for r in rs])[0], "tuning_s": ms([r["tuning_seconds_total"] for r in rs])[0],
               "selected_update_sd": ms([r["selected_update"] for r in rs])[1]}
        out["E1"].append(row)
        L.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {n} | {mj:+.4f} ({sj:.4f}) | {row['J_rmse']:.4f} | {mp:+.4f} ({sp:.4f}) | {row['S_plug_rmse']:.4f} | {row['posterior_mse']:.4f} | "
                 f"{row['J_se']:.4f} / {row['S_plug_se']:.4f} | {row['fit_s']:.1f} / {row['tuning_s']:.1f} |")
    L += ["", "### Reference estimators at the largest budget (each against its own truth; no common raw number)", "",
          "| condition | N | estimator | own target | relative error mean (sd) | n |", "|---|---|---|---|---|---|"]
    for k in sorted(gref):
        rs = gref[k]; m, s, n = ms([r["rel_error"] for r in rs]); out["E1_refs"].append({"condition": k[0], "N": k[1], "kind": k[2], "rel_err_mean": m, "rel_err_sd": s, "n": n})
        L.append(f"| {k[0]} | {k[1]} | {k[2]} | {rs[0]['own_target']} | {m:+.3f} ({s:.3f}) | {n} |")
    # ---------------------------------------------------------------- E2
    g2 = defaultdict(list)
    for c in cells:
        if c["unit"] != "E2":
            continue
        for r in c["rows"]:
            if r["selected"]:
                g2[(c["condition"], c["rotated"], c["N"], r["method"], r["budget_updates"])].append(r)
    L += ["", "## E2 original vs rotated coordinates (lr / bandwidth selected; mean over seeds)", "",
          "| condition | rotated | N | method | budget | n | |J − S| mean | posterior MSE | fit s chosen | tuning s total | peak CUDA MB |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in sorted(g2, key=lambda x: (x[0], x[3], x[2], x[4] or 0, x[1])):
        rs = g2[k]; aj = ms([abs(r["J_err"]) for r in rs]); pm = ms([r["posterior_mse"] for r in rs])
        mem = ms([(r.get("memory") or {}).get("cuda_peak_alloc_mb") for r in rs])[0]
        row = {"condition": k[0], "rotated": k[1], "N": k[2], "method": k[3], "budget": k[4], "n": aj[2], "absJerr": aj[0], "posterior_mse": pm[0],
               "fit_s": ms([r["fit_seconds"] for r in rs])[0], "tuning_s": ms([r["tuning_seconds_total"] for r in rs])[0], "cuda_mb": mem}
        out["E2"].append(row)
        L.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {k[4]} | {row['n']} | {row['absJerr']:.4f} | {row['posterior_mse']:.4f} | {row['fit_s']:.1f} | {row['tuning_s']:.1f} | {mem:.0f} |")
    # ---------------------------------------------------------------- E3
    L += ["", f"## E3 cost readouts (E1 fits): smallest budget whose seed-mean |J − S| ≤ {TOL}; refit spread = sd of J over independent FIT seeds", "",
          "| condition | N | fit loss | budget reaching tol | refit sd of J at 4000 | sd of selected update at 4000 |", "|---|---|---|---|---|---|"]
    byc = defaultdict(dict)
    for r in out["E1"]:
        byc[(r["condition"], r["N"], r["fit_loss"])][r["budget"]] = r
    for k in sorted(byc):
        bud = sorted(byc[k]); reach = next((b for b in bud if abs(byc[k][b]["J_err_mean"]) <= TOL), None); last = byc[k][bud[-1]]
        out["E3"].append({"condition": k[0], "N": k[1], "fit_loss": k[2], "budget_reaching_tol": reach, "refit_sd_J": last["J_err_sd"], "selected_update_sd": last["selected_update_sd"]})
        L.append(f"| {k[0]} | {k[1]} | {k[2]} | {reach if reach is not None else '—'} | {last['J_err_sd']:.4f} | {last['selected_update_sd']:.0f} |")
    Path(a.out + ".md").write_text("\n".join(L) + "\n"); Path(a.out + ".json").write_text(json.dumps(out, indent=1))
    print(f"-> {a.out}.md ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
