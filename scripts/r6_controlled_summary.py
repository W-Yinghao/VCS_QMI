"""R6-02 summary tables from outputs/R6_controlled/{reg_controlled,gain_controlled}.json -> reports/R6/R6_controlled_tables.md (descriptive).
REG: per condition × method, over seeds × test scenes (scenes are the units): success (final |error|∞ <= 0.5 px), median / p90 error, capture
rate by initial offset, median sharpness; finite-difference gradient check.  GAIN: exact Δ; per construction × family × class (mean ± sd over
seeds): DEV mean z, mean h, mean residual, cheap / full risk; risk at budgets .25 / .5 per policy; the worse-full negative check.
    python scripts/r6_controlled_summary.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

IN = Path("/home/infres/yinwang/CS_QMI/outputs/R6_controlled"); OUT = Path(__file__).resolve().parents[1] / "reports" / "R6"


def main() -> int:
    reg = json.load(open(IN / "reg_controlled.json")); gain = json.load(open(IN / "gain_controlled.json")); L = ["# R6-02 controlled validation — tables (descriptive)", ""]
    L += ["## REG (synthetic two-modality scenes, known shift; scenes are the units)", "",
          "| condition | method | scenes | success ≤ 0.5 px | median err | p90 err | capture r = 1 / 2 / 4 / 6 | median sharpness |", "|---|---|---|---|---|---|---|---|"]
    conds = sorted({r["condition"] for r in reg}, key=lambda c: [r["condition"] for r in reg].index(c)); fd = []
    for c in conds:
        rs = [r for r in reg if r["condition"] == c]; fd += [x["max_abs_diff"] for r in rs for x in r["fd_checks"]]
        for m in ("oracle", "vcs", "logistic", "mi", "nmi", "corr"):
            vs = [r["methods"][m] for r in rs if m in r["methods"]]
            if not vs:
                continue
            fe = np.concatenate([v["final_err_per_scene"] for v in vs]); cap = {k: np.mean([v["capture_rate"][k] for v in vs]) for k in ("1", "2", "4", "6")}
            L.append(f"| {c} | {m} | {len(fe)} | {np.mean(fe <= 0.5):.2f} | {np.median(fe):.2f} | {np.quantile(fe, 0.9):.2f} | " + " / ".join(f"{cap[k]:.2f}" for k in cap)
                     + f" | {np.median([v['median_sharpness'] for v in vs]):.2f} |")
    L += ["", f"Finite-difference vs analytic dF/dφ (VCS critic, non-integer shifts): max |diff| = {max(fd):.1e} over {len(fd)} checks.", ""]
    L += ["## GAIN (balanced binary constructions; label noise ε = 0.1 for the fitted cells)", "", "| construction | exact Δ (ε = 0) | exact Δ (ε = 0.1) |", "|---|---|---|"]
    for n in ("redundant", "irrelevant", "xor", "context", "context_noise"):
        L.append(f"| {n} | {gain['exact'][f'{n}|eps=0.0']['delta']:.3f} | {gain['exact'][f'{n}|eps=0.1']['delta']:.3f} |")
    L += ["", "| construction | family | class | DEV mean z | mean h | mean residual | risk cheap / full | risk at budget .5: random / uncertainty / gain-square / gain-utility / oracle d |",
          "|---|---|---|---|---|---|---|---|"]
    cells = gain["cells"]; key = lambda x: (x["construction"], x["family"], x["class"], x["bad_full"])
    for k in sorted({key(x) for x in cells}, key=lambda k: (["redundant", "irrelevant", "xor", "context", "context_noise"].index(k[0]), k[1], k[2], k[3])):
        xs = [x for x in cells if key(x) == k]; mz, mh, mr = (np.mean([x["dev"][f] for x in xs]) for f in ("mean_z", "mean_h", "mean_res"))
        sh = np.std([x["dev"]["mean_h"] for x in xs]); rc, rf = np.mean([x["dev"]["risk_cheap"] for x in xs]), np.mean([x["dev"]["risk_full"] for x in xs])
        r50 = {p: np.mean([x["curves"][p][2] for x in xs]) for p in ("random", "cheap_uncertainty", "gain_square", "gain_utility", "oracle_d")}
        L.append(f"| {k[0]}{' (worse full)' if k[3] else ''} | {k[1]} | {k[2]} | {mz:.3f} | {mh:+.3f} ± {sh:.3f} | {mr:+.3f} | {rc:.3f} / {rf:.3f} | " + " / ".join(f"{r50[p]:.3f}" for p in r50) + " |")
    L += ["", f"Reference counterexample (redundant; a = sign, b = 0): {gain['counterexamples']['reference_redundant_b_zero']}", ""]
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "R6_controlled_tables.md").write_text("\n".join(L)); print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
