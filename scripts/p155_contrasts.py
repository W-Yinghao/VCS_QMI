"""P155 — pre-stated paired contrasts on the official CIFAR-10 test (frozen in reports/P155_AP3_OFFICIAL_TEST_PROTOCOL_FROZEN_20261014.md).

    python scripts/p155_contrasts.py --test reports/P155_AP3_OFFICIAL_TEST.json --out reports/P155_AP3_OFFICIAL_TEST_contrasts

Pairs by seed (the seed index in the run id).  Paired 95 % t interval over seeds; reading: 'ahead' if the lower end > 0, 'behind' if the upper
end < 0, else 'no detectable difference'.  Primary: A-P3 - SimCLR linear (seeds 0-4).  Secondary: A-P3 - SimCLR kNN; A-P3 - JS-AP3 linear / kNN
(seeds 0-2).  Also prints test - selection per run (agreement of the development metric with the test).
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
from scipy import stats

CELLS = {"A-P3": "A-P3|8x|4v/800ep(final)", "SimCLR": "SimCLR|8x|4v/800ep", "JS-AP3": "JS-AP3|8x|4v/800ep(matched)"}
CONTRASTS = [("primary", "A-P3", "SimCLR", "test_linear"), ("secondary", "A-P3", "SimCLR", "test_knn"),
             ("secondary", "A-P3", "JS-AP3", "test_linear"), ("secondary", "A-P3", "JS-AP3", "test_knn")]


def by_seed(rows, cell, key):
    out = {}
    for r in rows:
        if r.get("cell") == cell and r.get("status") in ("evaluated", "read"):
            out[int(re.search(r"seed(\d+)$", r["run_id"]).group(1))] = float(r[key])
    return out


def paired(a: dict, b: dict) -> dict:
    seeds = sorted(set(a) & set(b)); d = np.array([a[s] - b[s] for s in seeds])
    n = len(d); m = float(d.mean()); sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    h = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else float("nan"); lo, hi = m - h, m + h
    reading = "ahead" if lo > 0 else ("behind" if hi < 0 else "no detectable difference")
    return {"seeds": seeds, "per_seed": [float(x) for x in d], "mean": m, "sd": sd, "ci95": [lo, hi], "n": n, "reading": reading,
            "sign_count": f"{int((d > 0).sum())} of {n} positive"}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--test", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    T = json.load(open(a.test)); rows = T["rows"]
    if T.get("standin"):
        raise SystemExit("refusing: stand-in file, not the official test")
    res = {"source": a.test, "contrasts": []}
    for fam, x, y, key in CONTRASTS:
        c = paired(by_seed(rows, CELLS[x], key), by_seed(rows, CELLS[y], key)); c.update({"family": fam, "contrast": f"{x} - {y}", "metric": key}); res["contrasts"].append(c)
    res["test_minus_selection"] = [{"run_id": r["run_id"], "linear": r["test_linear"] - r["sel_linear"], "knn": r["test_knn"] - r["sel_knn"]}
                                   for r in rows if r.get("sel_linear") is not None]
    json.dump(res, open(a.out + ".json", "w"), indent=1)
    L = ["# P155 — paired contrasts on the official CIFAR-10 test (pre-stated rule; script output)", "",
         "| family | contrast | metric | n | mean | 95 % t interval | per seed | reading |", "|---|---|---|---|---|---|---|---|"]
    for c in res["contrasts"]:
        L.append(f"| {c['family']} | {c['contrast']} | {c['metric']} | {c['n']} | {c['mean']:+.2f} | [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}] | "
                 f"{' / '.join(f'{v:+.2f}' for v in c['per_seed'])} | {c['reading']} |")
    L += ["", "## Test minus selection, per run", "", "| run | linear | kNN |", "|---|---|---|"]
    L += [f"| {r['run_id']} | {r['linear']:+.2f} | {r['knn']:+.2f} |" for r in res["test_minus_selection"]]
    open(a.out + ".md", "w").write("\n".join(L) + "\n"); print("\n".join(L)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
