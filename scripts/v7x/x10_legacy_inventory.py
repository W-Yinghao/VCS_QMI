"""V7X-10: recompute the legacy controlled-data facts the coverage table cites, straight from the raw P85 JSONs (read only).
Selection edges of the kernel rows, classical CS-K values against their own truth, non-finite step counts per estimator family.
    python -I scripts/v7x/x10_legacy_inventory.py --p85 DIR --out OUT.json"""
import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict


def cell_of(path):
    m = re.match(r"P85_(\w+?)_ds(\d+)_dt(\d+)_I([\d.]+)_N(\d+)_B(\d+)_U(\d+)_s(\d+)\.json", os.path.basename(path))
    return None if m is None else dict(zip(["gen", "ds", "dt", "I", "N", "B", "U", "seed"], m.groups()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--p85", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.p85, "P85_*.json"))); out = {"source_dir": a.p85, "n_files": len(files)}
    edge = Counter(); n_kde = n_rls = n_rff = 0; nonfinite_all = Counter(); nonfinite_sel = Counter(); csk = []; methods_per_file = Counter()
    unparsed = []
    for f in files:
        d = json.load(open(f)); c = cell_of(f) or {}; rows = d["rows"]; methods_per_file[len(rows)] += 1
        if not c:
            unparsed.append(os.path.basename(f))
        for r in rows:
            fam = ":".join(r["method"].split(":")[:3]) if r["method"].startswith("neural:") else r["method"].split(":")[0]
            nf = (r.get("numerical") or {}).get("nonfinite_steps") or 0
            nonfinite_all[fam] += nf
            if r.get("selected") is True:
                nonfinite_sel[fam] += nf
        kde = [r for r in rows if r["method"] == "s_kde:common_risk"]
        if kde:
            r = kde[0]; g = r["bandwidth_grid"]; n_kde += 1
            best = max(g, key=lambda e: e["J_kernel"]); mx = sorted({e["mult_x"] for e in g}); my = sorted({e["mult_y"] for e in g})
            edge["kde_x_at_min"] += best["mult_x"] == mx[0]; edge["kde_y_at_min"] += best["mult_y"] == my[0]
            edge["kde_both_at_min"] += best["mult_x"] == mx[0] and best["mult_y"] == my[0]
            edge["kde_any_edge"] += best["mult_x"] in (mx[0], mx[-1]) or best["mult_y"] in (my[0], my[-1])
            out.setdefault("kde_grid_mults", [mx, my])
        rls = [r for r in rows if r["method"].startswith("rls:raw")]
        if rls:
            r = rls[0]["rls"]; g = r["grid"]; n_rls += 1
            ms, mu, la = sorted({e["m"] for e in g}), sorted({e["mult"] for e in g}), sorted({e["lam"] for e in g})
            edge["rls_m_at_max"] += r["m"] == ms[-1]; edge["rls_lam_at_min"] += r["lambda"] == la[0]
            edge["rls_mult_at_edge"] += r["bandwidth_multiple"] in (mu[0], mu[-1]); edge["rls_any_edge"] += (r["m"] in (ms[0], ms[-1]) or r["lambda"] in (la[0], la[-1]) or r["bandwidth_multiple"] in (mu[0], mu[-1]))
            out.setdefault("rls_grid", {"m": ms, "mult": mu, "lam": la})
        rff = [r for r in rows if r["method"].startswith("s_kernel:rff")]
        sel = [r for r in rff if r.get("selected") is True]
        if sel:
            n_rff += 1; mm = re.match(r"s_kernel:rff:m(\d+):mult([\d.]+):lr", sel[0]["method"]); m_sel, mult_sel = int(mm.group(1)), float(mm.group(2))
            gm = sorted({int(re.match(r"s_kernel:rff:m(\d+)", r["method"]).group(1)) for r in rff})
            gu = sorted({float(re.match(r"s_kernel:rff:m\d+:mult([\d.]+)", r["method"]).group(1)) for r in rff})
            edge["rff_m_at_max"] += m_sel == gm[-1]; edge["rff_mult_at_min"] += mult_sel == gu[0]; out.setdefault("rff_grid", {"m": gm, "mult": gu})
        for r in rows:
            if r["method"].startswith("cs_k:"):
                nat = r["native"]; csk.append({**c, "method": r["method"], "value": nat["value"], "own_truth": nat["own_truth"],
                                               "truth_CS_lebesgue": d["truth"].get("CS_lebesgue"), "absent_reason": d["truth"].get("CS_lebesgue_absent_reason")})
    out.update({"n_cells_with_kde": n_kde, "n_cells_with_rls": n_rls, "n_cells_with_rff_selected": n_rff, "edge_counts": dict(edge),
                "nonfinite_steps_all_rows": dict(nonfinite_all), "nonfinite_steps_selected_rows": dict(nonfinite_sel), "rows_per_file": dict(methods_per_file),
                "unparsed_names": unparsed, "cs_k": csk,
                "cs_k_truth_available_cells": sorted({(r["gen"], r["I"], r["N"], r["dt"]) for r in csk if r["own_truth"] is not None})})
    json.dump(out, open(a.out, "w"), indent=1, default=list)
    print(json.dumps({k: out[k] for k in ["n_files", "n_cells_with_kde", "n_cells_with_rls", "n_cells_with_rff_selected", "edge_counts",
                                          "nonfinite_steps_all_rows", "nonfinite_steps_selected_rows", "kde_grid_mults", "rls_grid", "rff_grid"] if k in out}, indent=1))
    print("cs_k rows with own truth:", sum(r["own_truth"] is not None for r in csk), "of", len(csk))


if __name__ == "__main__":
    main()
