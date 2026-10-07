"""P142 (MV6-I1) aggregation: fixed-encoder estimator comparison (Table 3 axis) for block A (P116 fixture, same draws) and block B (Table 4's
SimCLR seed 1, new fixture).  Per cell x method: rejection rate with 95 % Wilson interval, discordant repeats vs A1, common J on EVAL (same-target
methods), fit + selection seconds and permutation seconds; null cells with the > 0.09 flag; the P116 reproduction gate per cell.
    python scripts/p142_aggregate.py --out reports/P142_results
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FILES = {"A": ["A_simclr_colour", "A_simclr_blur", "A_simclr_level", "A_simclr_level_n1000"],
         "B": ["B_simclrs1_blur", "B_simclrs1_colour", "B_simclrs1_level"]}
FIXTURE = {"A": "F-SOLVER-P116 (P5_simclr_seed0, 2 views, 200 epochs)", "B": "F-SOLVER-P142B (P41_simclr_views4_800ep_seed1 = Table 4 SimCLR / 1)"}
METHODS = ["A1", "A3@50", "A3@200", "A3@800", "K1", "H1", "H2"]
LABEL = {"A1": "VCS (ridge-tanh two-stage)", "A3@50": "matched logistic, 50 closures", "A3@200": "matched logistic, 200", "A3@800": "matched logistic, 800",
         "K1": "same-target kernel (RFF, A1 solver)", "H1": "conditional HSIC, class-wise median kernel", "H2": "conditional HSIC, deep kernel"}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    rows, gates = [], []
    for block, files in FILES.items():
        for fn in files:
            d = json.load(open(REPO / "reports" / "P142" / f"{fn}.json"))
            for case, cv in d["cases"].items():
                for n, cell in cv["by_n"].items():
                    if "reproduction_gate" in cell:
                        gates.append({"block": block, "case": case, "n": int(n), **cell["reproduction_gate"]})
                    for m in METHODS:
                        s = cell["summary"][m]
                        rows.append({"block": block, "fixture": FIXTURE[block], "case": case, "mode": cv["mode"], "strength": cv["strength"], "n": int(n),
                                     "repeats": cell["repeats"], "method": m, "method_label": LABEL[m], "rejection_rate": s["power_own"],
                                     "wilson95": s["power_own_wilson95"], "null_flag": (cv["mode"] != "planted" and s["power_own"] > 0.09),
                                     "vs_A1_method_only": s.get("vs_A1_discordant", {}).get("method_only"),
                                     "vs_A1_A1_only": s.get("vs_A1_discordant", {}).get("A1_only"),
                                     "Jcommon_eval": s.get("Jcommon_eval"), "fit_seconds": s["fit_seconds_mean"], "perm_seconds": s["perm_seconds_mean"],
                                     "source": f"reports/P142/{fn}.json"})
    Path(a.out + ".json").write_text(json.dumps({"gates": gates, "rows": rows}, indent=1))
    with open(a.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    L = ["# P142 results — fixed encoder, vary estimator (Table 3 axis)", "",
         "P116 reproduction gate (block A, A1 and A3 per repeat): " + "; ".join(f"{g['case']} n{g['n']}: pass={g['pass']} (max |Δ| {g['max_abs_stat_diff']:.1e}, mismatches {g['decision_mismatches']})" for g in gates), ""]
    for block in FILES:
        L += [f"## Block {block} — {FIXTURE[block]}", "", "| cell | n | R | " + " | ".join(METHODS) + " |", "|---|---|---|" + "---|" * len(METHODS)]
        cells = []
        for r in rows:
            if r["block"] == block and (r["case"], r["n"]) not in cells:
                cells.append((r["case"], r["n"]))
        for case, n in cells:
            rr = {r["method"]: r for r in rows if r["block"] == block and r["case"] == case and r["n"] == n}
            L.append(f"| {case} | {n} | {rr['A1']['repeats']} | " + " | ".join(f"{rr[m]['rejection_rate']:.2f}" + (" ⚑" if rr[m]["null_flag"] else "") for m in METHODS) + " |")
        L.append("")
        L += ["Cost per repeat (n 2 000 planted cells, mean seconds; fit includes selection; permutation = 200 permutations):", ""]
        rr = [r for r in rows if r["block"] == block and r["mode"] == "planted" and r["n"] == 2000]
        L.append("| method | fit s | perm s |"); L.append("|---|---|---|")
        for m in METHODS:
            x = [r for r in rr if r["method"] == m]
            L.append(f"| {LABEL[m]} | {sum(r['fit_seconds'] for r in x) / len(x):.2f} | {sum(r['perm_seconds'] for r in x) / len(x):.3f} |")
        L.append("")
    Path(a.out + ".md").write_text("\n".join(L))
    print(f"{len(rows)} rows, {len(gates)} gated cells -> {a.out}.{{json,csv,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
