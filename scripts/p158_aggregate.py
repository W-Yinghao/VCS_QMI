"""P158 aggregate: truth-compression tables (both readouts) + the frozen "error reduced" rule (P118 §6.1 clauses) per objective and readout,
the bootstrap consistency counts, and the oracle identity gaps.

    python scripts/p158_aggregate.py --in reports/P158/truth --out reports/P158/p158_results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import p158_common as C  # noqa: E402
from vcs_ssl.utils import atomic_write_json  # noqa: E402

VARIANTS = ("indep", "nested", "zero")
READOUTS = ("common", "js_native")
BASE = "indep"
CLAUSES = ("c1_rmse", "c2_null_step", "c3_posterior", "c4_no_shrinkage")


def true_ma(cell: dict, rd: str) -> float:
    return C.true_increments(cell["oracle"])["vcs" if rd == "common" else "js"]["M->A"]


def rule(cells: dict, obj: str, v: str, rd: str) -> dict:
    """Frozen rule (P118 §6.1), per objective and readout, in BOTH non-null conditions vs the indep baseline of the same objective / readout:
    (1) RMSE of the M→A increment error ≤ 0.75 × baseline; (2) |mean B→M error| ≤ baseline's; (3) posterior MSE at B and at A ≤ 1.10 × baseline;
    (4) |mean M→A increment| ≥ 0.5 × the oracle increment.  All four → 'error reduced'; (4) failing while |R_orth| falls → 'shrinkage';
    otherwise 'no clear reduction'.  The zero control is scored by the same rule (it fails (1) / (4) by construction)."""
    out = {}
    for cond in ("weak", "moderate"):
        sm, b = cells[cond]["summary"][f"{obj}/{v}/{rd}"], cells[cond]["summary"][f"{obj}/{BASE}/{rd}"]
        bp, sp = cells[cond]["summary"][f"{obj}/{BASE}/common"], cells[cond]["summary"][f"{obj}/{v}/common"]   # posterior MSE is readout-independent
        ti = true_ma(cells[cond], rd)
        out[cond] = {"c1_rmse": sm["inc_rmse_M->A"] <= 0.75 * b["inc_rmse_M->A"],
                     "c2_null_step": abs(sm["inc_err_B->M"]["mean"]) <= abs(b["inc_err_B->M"]["mean"]),
                     "c3_posterior": sp["post_B"]["mean"] <= 1.10 * bp["post_B"]["mean"] and sp["post_A"]["mean"] <= 1.10 * bp["post_A"]["mean"],
                     "c4_no_shrinkage": abs(sm["inc_M->A"]["mean"]) >= 0.5 * ti,
                     "R_orth_lower": abs(sm["R_orth"]["mean"]) < abs(b["R_orth"]["mean"])}
    allc = all(all(x[k] for k in CLAUSES) for x in out.values())
    shrink = any((not x["c4_no_shrinkage"]) and x["R_orth_lower"] for x in out.values())
    out["verdict"] = "error reduced" if allc else ("shrinkage" if shrink else "no clear reduction")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    cells = {}
    for cond in ("null", "weak", "moderate"):
        p = Path(f"{a.inp}_{cond}.json")
        if p.is_file():
            cells[cond] = json.load(open(p))
    if not cells:
        sys.exit(f"no condition files at {a.inp}_<cond>.json")
    J = {"conditions": {k: {"rho": c["rho"], "oracle": c["oracle"], "summary": c["summary"], "repeats_n": len(c["repeats"]), "seconds": c.get("seconds")} for k, c in cells.items()}, "rule": {}}
    L = ["# P158 — nested increments with truth on the joint-vs-product chain: results", "",
         "## Oracle (Monte-Carlo truth sample; identities reported as gaps, not assumed)", "",
         "| cond | ρ | S_B | S_M | S_A | G_B | G_M | G_A | L2 inc M→A (S diff − E_M(η_B−η_A)²) | Bregman inc M→A (G diff − 2E_M KL) | n_truth |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for cond, c in cells.items():
        o = c["oracle"]; s = o["sets"]; st = o["steps"]["M->A"]
        L.append(f"| {cond} | {c['rho']} | {s['B']['S']:.4f} | {s['M']['S']:.4f} | {s['A']['S']:.4f} | {s['B']['G']:.4f} | {s['M']['G']:.4f} | {s['A']['G']:.4f} | "
                 f"{st['L2_increment']:.4f} ({st['identity_gap_L2']:+.1e}) | {st['Bregman_increment']:.4f} ({st['identity_gap_Bregman']:+.1e}) | {o['n_truth']} |")
    for rd, lab in (("common", "common squared-score readout (J / D_T / R_orth on T; truth = S differences)"), ("js_native", "JS native readout (G / Bregman D_JS / R_orth_js on logits; truth = G differences)")):
        L += ["", f"## {lab} — mean over repeats", "",
              "| cond | true M→A | obj | variant | err B→M | err M→A (RMSE) | err B→A | D M→A | r M→A | post B | post M | post A | R_orth | viol | boot: 0∈CI r_M→A / r_B→M / R_orth (of n reps) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for cond, c in cells.items():
            ti = true_ma(c, rd)
            for obj in ("vcs", "js"):
                for v in VARIANTS:
                    s = c["summary"][f"{obj}/{v}/{rd}"]; sc = c["summary"][f"{obj}/{v}/common"]; nrep = len(c["repeats"])
                    bz = " / ".join(str(s.get(f"boot_zero_in_ci_{nm}", "—")) for nm in ("r_M->A", "r_B->M", "R_orth"))
                    L.append(f"| {cond} | {ti:.4f} | {obj} | {v} | {s['inc_err_B->M']['mean']:+.4f} | {s['inc_err_M->A']['mean']:+.4f} ({s['inc_rmse_M->A']:.4f}) | {s['inc_err_B->A']['mean']:+.4f} | "
                             f"{s['D_M->A']['mean']:.4f} | {s['r_M->A']['mean']:+.4f} | {sc['post_B']['mean']:.4f} | {sc['post_M']['mean']:.4f} | {sc['post_A']['mean']:.4f} | {s['R_orth']['mean']:+.4f} | "
                             f"{s['ordering_violations']['mean']:.1f} | {bz} (of {nrep}) |")
    if all(k in cells for k in ("weak", "moderate")):
        L += ["", "## Frozen rule vs indep (same objective and readout; both non-null conditions)", "", "| obj | variant | readout | verdict | weak (c1 c2 c3 c4) | moderate |", "|---|---|---|---|---|---|"]
        for obj in ("vcs", "js"):
            for v in VARIANTS:
                if v == BASE:
                    continue
                for rd in READOUTS:
                    r = rule(cells, obj, v, rd); J["rule"][f"{obj}/{v}/{rd}"] = r
                    f = lambda x: " ".join("✓" if x[k] else "✗" for k in CLAUSES)
                    L.append(f"| {obj} | {v} | {rd} | {r['verdict']} | {f(r['weak'])} | {f(r['moderate'])} |")
    if "null" in cells:
        L += ["", "Null floor (ρ = 0): |mean increment error| B→M / M→A per objective and variant (common readout): " + "; ".join(
            f"{obj}/{v} {abs(cells['null']['summary'][f'{obj}/{v}/common']['inc_err_B->M']['mean']):.4f} / {abs(cells['null']['summary'][f'{obj}/{v}/common']['inc_err_M->A']['mean']):.4f}"
            for obj in ("vcs", "js") for v in VARIANTS)]
    Path(a.out + ".md").write_text("\n".join(L) + "\n"); atomic_write_json(Path(a.out + ".json"), J); print(f"-> {a.out}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
