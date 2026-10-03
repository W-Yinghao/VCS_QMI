"""P121 — v6 V6-THEORY runner (CPU only).  Writes reports/P121_theory_checks.json, reports/P121_gram_fit.csv, reports/P121_conditions.md.

    python scripts/p121_theory.py [--smoke] [--out-prefix reports/P121]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vcs_theory import p121  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--out-prefix", default="reports/P121"); a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    t0 = time.time(); steps = 200 if a.smoke else p121.STEPS; starts = 1 if a.smoke else p121.STARTS
    dims = (2,) if a.smoke else p121.DIMS; settings = p121.GRAM_SETTINGS[:1] if a.smoke else p121.GRAM_SETTINGS
    out = {"protocol": "P121", "logical_id": "V6-THEORY", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "smoke": a.smoke,
           "tolerance_rel": p121.TOL_REL, "steps": steps, "starts": starts, "dims": list(dims), "gram_settings": [list(s) for s in settings], "conditions": {}}
    fit_rows_all, diag_rows_all = [], []
    for name, kw in p121.CONDITIONS.items():
        d = p121.latent(**kw)
        oracle_J = p121.j_population(d["P"], d["Q"], d["eta"]); zero_J = p121.j_population(d["P"], d["Q"], np.zeros_like(d["P"]))
        cond = {"params": kw, "S": d["S"], "J_oracle_free_pair": oracle_J, "J_constant_zero": zero_J, "oracle_gap": d["S"] - oracle_J,
                "max_abs_half_log_ratio": float(np.abs(0.5 * np.log(d["P"] / d["Q"])).max()),
                "kernel_identity": p121.kernel_identity(d), "nested": p121.nested_check(d, n_draws=10 if a.smoke else 50)}
        dr, fr = p121.gram_block(name, d, settings, dims, starts, steps)
        cond["gram_target"] = dr; diag_rows_all += dr; fit_rows_all += fr
        out["conditions"][name] = cond
        print(f"[{name}] S={d['S']:.5f} kernel_rel={cond['kernel_identity']['max_rel_error']:.2e} nested_rel={cond['nested']['max_rel_residual']:.2e} "
              f"({time.time() - t0:.0f}s)", flush=True)
    out["gradient_checks"] = p121.gradient_checks()
    out["fits_summary"] = [{k: r[k] for k in ("condition", "geometry", "a", "kappa", "dim", "start", "J", "S", "J_over_S", "gap_S_minus_J", "gram_dist2_M", "B",
                                                "lipschitz_holds", "lower_over_gap")} for r in fit_rows_all]
    out["all_identities_pass"] = bool(all(c["kernel_identity"]["pass"] and c["nested"]["pass"] for c in out["conditions"].values())
                                      and out["gradient_checks"]["pass"])
    out["lipschitz_holds_all"] = bool(all(r["lipschitz_holds"] for r in fit_rows_all))
    out["wall_seconds"] = time.time() - t0
    pre = Path(a.out_prefix); pre.parent.mkdir(parents=True, exist_ok=True)
    Path(str(pre) + "_theory_checks.json").write_text(json.dumps(out, indent=1))
    keys = ["condition", "geometry", "a", "kappa", "dim", "start", "steps", "S", "J", "J_over_S", "gap_S_minus_J", "posterior_mse", "gram_dist2_M", "B",
            "lipschitz_lower", "lipschitz_upper", "lipschitz_holds", "lower_over_gap"]
    with open(str(pre) + "_gram_fit.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); [w.writerow({k: r[k] for k in keys}) for r in fit_rows_all]
    L = ["# P121 — condition table (finite latent model; exact float64)", "",
         "| condition | seed | gamma | S | max |f*| | kernel identity rel err | nested max rel residual | J(oracle free pair) | J(const 0) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, c in out["conditions"].items():
        L.append(f"| {name} | {c['params']['seed']} | {c['params']['gamma']} | {c['S']:.5f} | {c['max_abs_half_log_ratio']:.3f} | {c['kernel_identity']['max_rel_error']:.1e} | "
                 f"{c['nested']['max_rel_residual']:.1e} | {c['J_oracle_free_pair']:.5f} | {c['J_constant_zero']:.1f} |")
    L += ["", "## Gram target G* = kappa + log(P/Q)/(2a)", "", "| condition | a | kappa | sym err | diag err (max |G*_ii − 1|) | range excess | neg-eig mass | frac neg eigs | min / max |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in diag_rows_all:
        L.append(f"| {r['condition']} | {r['a']} | {r['kappa']} | {r['symmetry_error']:.1e} | {r['diagonal_error']:.3f} | {r['range_excess']:.3f} | {r['negative_eigen_mass']:.3f} | "
                 f"{r['frac_negative_eigs']:.2f} | {r['min']:.3f} / {r['max']:.3f} |")
    L += ["", "## Unit-vector fits (all starts; J / S; not an optimality certificate; J(const 0) = 0, J(free oracle) = S)", "",
          "shared = one unit vector per state on both sides (unit diagonal G_uu = 1, a finite-model constraint); two_tower = separate unit vectors per side.", "",
          "| condition | geometry | a | kappa | d | J/S per start | best J/S | Lipschitz bound holds (all starts) |",
          "|---|---|---|---|---|---|---|---|"]
    from itertools import groupby
    key = lambda r: (r["condition"], r["geometry"], r["a"], r["kappa"], r["dim"])
    for k, grp in groupby(sorted(fit_rows_all, key=key), key=key):
        g = list(grp)
        L.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {k[4]} | {', '.join(f'{r['J_over_S']:.3f}' for r in g)} | {max(r['J_over_S'] for r in g):.3f} | "
                 f"{all(r['lipschitz_holds'] for r in g)} |")
    Path(str(pre) + "_conditions.md").write_text("\n".join(L) + "\n")
    print(f"identities pass: {out['all_identities_pass']}; Lipschitz holds on all fits: {out['lipschitz_holds_all']}; {out['wall_seconds']:.0f}s")
    return 0 if out["all_identities_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
