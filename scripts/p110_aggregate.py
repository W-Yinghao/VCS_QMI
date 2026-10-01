"""P110 aggregate: reports/P110/{v1,v2,v3}_<run>.json → reports/P110_v_results.{md,json}.

    python scripts/p110_aggregate.py --in reports/P110 --out reports/P110_v_results
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def family(run: str) -> str:
    return re.sub(r"_seed\d+$", "", run)


def ms(v):
    v = np.asarray(v, dtype=float)
    return f"{v.mean():.2f} ± {v.std(ddof=1):.2f}" if len(v) > 1 else (f"{v.mean():.2f}" if len(v) else "—")


def cluster_ci(per_run: dict[str, list[float]], B: int = 2000, seed: int = 0) -> list[float]:
    """95 % bootstrap CI of the grand mean, resampling runs (each run's replicate values kept together)."""
    runs = list(per_run); rng = np.random.default_rng(seed); means = []
    for _ in range(B):
        pick = rng.choice(len(runs), len(runs)); means.append(np.mean(np.concatenate([per_run[runs[i]] for i in pick])))
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="inp", default="reports/P110"); ap.add_argument("--out", default="reports/P110_v_results")
    a = ap.parse_args(); L = [f"# P110 — package v4 V1 / V2 / V3 results (selection split; official test closed)", ""]; J = {}
    # ---- V1
    v1 = [json.load(open(f)) for f in sorted(glob.glob(f"{a.inp}/v1_*.json"))]
    if v1:
        tab = defaultdict(lambda: defaultdict(list)); tabr = defaultdict(lambda: defaultdict(list))
        for r in v1:
            for ds, d in r["datasets"].items():
                for row in d["rows"]:
                    tab[(ds, row["fraction"])][family(r["run"])].append(row["acc_selected_pct"]); tabr[(ds, row["fraction"])][family(r["run"])].append(row["acc_recipe_fixed_pct"])
        fams = sorted({family(r["run"]) for r in v1})
        L += ["## V1 — label efficiency (CIFAR-10) and frozen transfer (CIFAR-100): selection-split top-1, FIT-selected probe (recipe-fixed probe)",
              "mean ± sd over encoder seeds × subset draws", "", "| dataset | labels | " + " | ".join(fams) + " |", "|---|---|" + "---|" * len(fams)]
        for k in sorted(tab):
            L.append(f"| {k[0]} | {k[1] * 100:g} % | " + " | ".join(f"{ms(tab[k][f])} ({ms(tabr[k][f])})" for f in fams) + " |")
        J["v1"] = {f"{k[0]}@{k[1]}": {f: {"selected": tab[k][f], "recipe": tabr[k][f]} for f in fams} for k in tab}
        L.append("")
    # ---- V2
    v2 = [json.load(open(f)) for f in sorted(glob.glob(f"{a.inp}/v2_*.json"))]
    if v2:
        by = defaultdict(list)
        for r in v2:
            by[family(r["run"])].append(r)
        L += ["## V2 — development corruptions (6 families × 5 severities, selection images; recipe probe on clean FIT)", "",
              "| encoder family | clean | mCA | mean relative drop % | mean consistency % | " + " | ".join(sorted({c['family'] for c in v2[0]['cells']})) + " |",
              "|---|---|---|---|---|" + "---|" * len({c['family'] for c in v2[0]['cells']})]
        for f, rs in sorted(by.items()):
            fams_c = sorted({c["family"] for c in rs[0]["cells"]})
            per = [ms([np.mean([c["acc_pct"] for c in r["cells"] if c["family"] == fc]) for r in rs]) for fc in fams_c]
            L.append(f"| {f} | {ms([r['acc_clean_pct'] for r in rs])} | {ms([r['mCA_pct'] for r in rs])} | {ms([r['mean_rel_drop_pct'] for r in rs])} | "
                     f"{ms([r['mean_consistency_pct'] for r in rs])} | " + " | ".join(per) + " |")
        J["v2"] = {f: [{k: r[k] for k in ("run", "acc_clean_pct", "mCA_pct", "mean_rel_drop_pct", "mean_consistency_pct")} for r in rs] for f, rs in by.items()}
        L.append("")
    # ---- V3
    v3 = [json.load(open(f)) for f in sorted(glob.glob(f"{a.inp}/v3_*.json"))]
    if v3:
        rules = ["A", "B_vcs_closed_J", "B_js_exact", "B_hsic_class", "C_random_eligible"]
        L += ["## V3 — audit-assisted selection: reversal-test accuracy (worst-group) of the selected member", "",
              "| cue | ρ | encoder family | " + " | ".join(rules) + " | pool best |", "|---|---|---|" + "---|" * (len(rules) + 1)]
        delta = {k: defaultdict(list) for k in rules[1:]}; J["v3"] = {}
        cells = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for r in v3:
            for cue, byrho in r["families"].items():
                for rho, reps in byrho.items():
                    for rep in reps:
                        s = rep["selection"]
                        for k in rules:
                            cells[(cue, rho)][family(r["run"])][k].append((s[k]["test_acc"], s[k]["test_worst_group"]))
                        cells[(cue, rho)][family(r["run"])]["best"].append((s["pool_best_test"], np.nan))
                        for k in rules[1:]:
                            delta[k][r["run"]].append(s[k]["test_acc"] - s["A"]["test_acc"])
        for (cue, rho), byf in sorted(cells.items()):
            for f, d in sorted(byf.items()):
                L.append(f"| {cue} | {rho} | {f} | " + " | ".join(f"{np.mean([x[0] for x in d[k]]):.2f} ({np.mean([x[1] for x in d[k]]):.2f})" for k in rules)
                         + f" | {np.mean([x[0] for x in d['best']]):.2f} |")
        L += ["", "**Pre-stated reading (pooled over encoders, cues, ρ; 95 % bootstrap CI resampling encoder runs):** Δ_k = test acc(B_k) − test acc(A).", "",
              "| rule | mean Δ vs A (pts) | 95 % CI |", "|---|---|---|"]
        for k in rules[1:]:
            vals = np.concatenate(list(delta[k].values())); ci = cluster_ci({kk: np.asarray(v) for kk, v in delta[k].items()})
            L.append(f"| {k} | {vals.mean():+.2f} | [{ci[0]:+.2f}, {ci[1]:+.2f}] |"); J["v3"][k] = {"mean_delta_vs_A": float(vals.mean()), "ci": ci}
    Path(a.out + ".md").write_text("\n".join(L) + "\n"); Path(a.out + ".json").write_text(json.dumps(J, indent=1, default=float))
    print(f"-> {a.out}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
