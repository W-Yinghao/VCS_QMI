"""P125 aggregate: selected rows (lr by SELECT risk) per (condition, N, orig/rot, candidate, loss); paired vs the JointMLP reference (same seed, same roles);
rotation effect (rot − orig) for joint and inter.

    python scripts/p125_aggregate.py --in-dir outputs/P125_struct --out reports/P125_struct_results
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def ms(v):
    v = np.asarray(v, float); return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--in-dir", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    cells = [json.load(open(f)) for f in sorted(glob.glob(f"{a.in_dir}/P125_*.json"))]
    sel = defaultdict(dict)   # (cond, N, rot, cand, kind) -> {seed: row}
    for c in cells:
        for r in c["rows"]:
            if r["selected"]:
                sel[(c["condition"], c["N"], c["rotated"], r["candidate"], r["kind"])][c["seed"]] = {**r, "S": c["truth"]["S"]}
    rows = []
    for k, by_seed in sorted(sel.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3], kv[0][4])):
        cond, N, rot, cand, kind = k; seeds = sorted(by_seed); R = [by_seed[s] for s in seeds]
        ref = sel.get((cond, N, rot, "joint", kind), {}); common = [s for s in seeds if s in ref]
        d_post = [by_seed[s]["posterior_mse"] - ref[s]["posterior_mse"] for s in common] if cand != "joint" else []
        orig = sel.get((cond, N, False, cand, kind), {}) if rot else {}
        d_rot = [by_seed[s]["posterior_mse"] - orig[s]["posterior_mse"] for s in seeds if s in orig]
        rows.append({"condition": cond, "N": N, "rotated": rot, "candidate": cand, "kind": kind, "seeds": seeds, "S": R[0]["S"],
                     "posterior_mse": ms([r["posterior_mse"] for r in R]), "J_err_mean": ms([r["J_err"] for r in R])[0],
                     "J_rmse": math.sqrt(float(np.mean([r["J_err"] ** 2 for r in R]))), "S_plug_err_mean": ms([r["S_plug_err"] for r in R])[0],
                     "S_plug_rmse": math.sqrt(float(np.mean([r["S_plug_err"] ** 2 for r in R]))), "eval_se_J": float(np.mean([r["J_se"] for r in R])),
                     "fit_seconds_chosen": float(np.mean([r["fit_seconds"] for r in R])), "tuning_seconds_total": float(np.mean([r["tuning_seconds_total"] for r in R])),
                     "n_params": R[0]["n_params"], "selected_update_mean": float(np.mean([r["selected_update"] for r in R])),
                     "lr_selected": [r["lr"] for r in R],
                     "delta_post_vs_joint": ms(d_post) if d_post else None, "delta_post_rot_minus_orig": ms(d_rot) if d_rot else None})
    out = {"protocol": "P125", "n_cells": len(cells), "rows": rows}
    Path(a.out + ".json").write_text(json.dumps(out, indent=1))
    L = ["# P125 — structured measurement critics (v6 V6-STRUCT): results", "",
         f"{len(cells)} cell files.  Selected = lr by SELECT native risk (the P108 rule).  Mean (sd) over fit seeds; Δ vs JointMLP paired by seed (same roles).", "",
         "| condition | N | rot | candidate | loss | seeds | params | posterior MSE | Δ post vs joint | J bias / RMSE | S_plug bias / RMSE | eval SE J | fit s (chosen / tuning) | rot − orig Δpost |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda t: "—" if t is None else (f"{t[0]:+.4f} ({t[1]:.4f})" if not math.isnan(t[1]) else f"{t[0]:+.4f}")
    for r in rows:
        L.append(f"| {r['condition']} | {r['N']} | {'rot' if r['rotated'] else 'orig'} | {r['candidate']} | {r['kind']} | {len(r['seeds'])} | {r['n_params']} | "
                 f"{r['posterior_mse'][0]:.4f} | {f(r['delta_post_vs_joint'])} | {r['J_err_mean']:+.4f} / {r['J_rmse']:.4f} | {r['S_plug_err_mean']:+.4f} / {r['S_plug_rmse']:.4f} | "
                 f"{r['eval_se_J']:.4f} | {r['fit_seconds_chosen']:.1f} / {r['tuning_seconds_total']:.1f} | {f(r['delta_post_rot_minus_orig'])} |")
    Path(a.out + ".md").write_text("\n".join(L) + "\n"); print(f"-> {a.out}.md ({len(cells)} cells, {len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
