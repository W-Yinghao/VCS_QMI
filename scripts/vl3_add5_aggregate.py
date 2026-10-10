"""VL3 addendum 5 readings (reports/VL3/VL3_ADDENDUM5_STRATA_FROZEN_20261010.md) over outputs/VL3_strata/.
Strata (fixed before any per-query result):
  S1 expression length in words (whitespace split), within-dataset tertiles of the DEV expressions: short <= q1/3, mid, long > q2/3;
  S2 location word present (fixed list below, word-boundary, lower case) — RefCOCO and RefCOCOg only (RefCOCO+ forbids location words);
  S3 number of referred objects in the image: 2, 3, >= 4.
Per cell (dataset x backbone), objective, seed and stratum: query-level DEV Top-1; per stratum the paired contrasts VCS − softmax and VCS − JS.
Primary: S1 long − short of (VCS − softmax), per cell the mean over seeds, pooled over the 12 cells (95 % t).  Secondary: S2 (with − without)
over the 8 RefCOCO / RefCOCOg cells; S3 (>= 4 − 2) over the 12 cells; the same for VCS − JS.  Writes reports/VL3/VL3_add5_results.json.
    python scripts/vl3_add5_aggregate.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vl1_10_aggregate import tint  # noqa: E402

D = Path("/home/infres/yinwang/CS_QMI/outputs/VL3_strata"); OUT = Path(__file__).resolve().parents[1] / "reports" / "VL3" / "VL3_add5_results.json"
DATASETS, BACKBONES, OBJS, SEEDS = ("refcocog", "refcoco", "refcocoplus"), ("clip_b16", "siglip2_b16", "clip_l14_336", "siglip2_l16"), ("vcs", "js", "softmax"), (0, 1, 2)
LOC = ("left", "right", "top", "bottom", "middle", "center", "centre", "front", "back", "behind", "near", "nearest", "far", "farthest", "closest",
       "first", "second", "third", "upper", "lower", "corner", "side", "leftmost", "rightmost")
LOC_RE = re.compile(r"\b(" + "|".join(LOC) + r")\b")


def strata(ds):
    q = json.load(open(D / f"{ds}_meta.json"))["queries"]; n = np.array([len(x["text"].split()) for x in q]); q1, q2 = np.quantile(n, [1 / 3, 2 / 3])
    s = {"S1_short": n <= q1, "S1_mid": (n > q1) & (n <= q2), "S1_long": n > q2}
    if ds != "refcocoplus":
        loc = np.array([bool(LOC_RE.search(x["text"].lower())) for x in q]); s["S2_with"] = loc; s["S2_without"] = ~loc
    r = np.array([x["n_referred"] for x in q]); s["S3_2"] = r == 2; s["S3_3"] = r == 3; s["S3_ge4"] = r >= 4
    return s, {"length_tertile_cuts": [float(q1), float(q2)], "sizes": {k: int(v.sum()) for k, v in s.items()}}


def main() -> int:
    res = {"strata": {}, "cells": {}, "missing": [], "gate_S": []}
    for ds in DATASETS:
        if not (D / f"{ds}_meta.json").exists():
            res["missing"].append(f"{ds}/meta"); continue
        S, info = strata(ds); res["strata"][ds] = info
        for bb in BACKBONES:
            hits = {}
            for o in OBJS:
                for s in SEEDS:
                    f = D / f"{ds}_{bb}_{o}_s{s}.json"
                    if not f.exists():
                        res["missing"].append(f"{ds}/{bb}/{o}/s{s}"); continue
                    r = json.load(open(f)); res["gate_S"].append({"run": r["tag"], **r["gate_S"]})
                    if r["hits"] is not None:
                        hits[(o, s)] = np.array(r["hits"], dtype=float)
            seeds = [s for s in SEEDS if all((o, s) in hits for o in OBJS)]
            if len(seeds) < 3:
                continue
            cell = {"acc": {}, "contrast": {}}
            for k, m in S.items():
                cell["acc"][k] = {o: tint([hits[(o, s)][m].mean() for s in seeds]) for o in OBJS}
                for other in ("js", "softmax"):
                    cell["contrast"][f"{k}/vcs_minus_{other}"] = [float(hits[("vcs", s)][m].mean() - hits[(other, s)][m].mean()) for s in seeds]
            res["cells"][f"{ds}/{bb}"] = cell
    tests = {"S1_long_minus_short": ("S1_long", "S1_short"), "S2_with_minus_without": ("S2_with", "S2_without"), "S3_ge4_minus_2": ("S3_ge4", "S3_2")}
    res["pooled"] = {}
    for name, (hi, lo) in tests.items():
        for other in ("softmax", "js"):
            per_cell = {k: float(np.mean(np.array(c["contrast"][f"{hi}/vcs_minus_{other}"]) - np.array(c["contrast"][f"{lo}/vcs_minus_{other}"])))
                        for k, c in res["cells"].items() if f"{hi}/vcs_minus_{other}" in c["contrast"]}
            if len(per_cell) >= 2:
                res["pooled"][f"{name}/vcs_minus_{other}"] = {**tint(list(per_cell.values())), "n_cells": len(per_cell), "per_cell": per_cell}
        for k in (hi, lo):
            for other in ("softmax", "js"):
                v = [float(np.mean(c["contrast"][f"{k}/vcs_minus_{other}"])) for c in res["cells"].values() if f"{k}/vcs_minus_{other}" in c["contrast"]]
                if len(v) >= 2:
                    res["pooled"][f"{k}/vcs_minus_{other}"] = {**tint(v), "n_cells": len(v)}
    json.dump(res, open(OUT, "w"), indent=1)
    q = lambda t: f"{100 * t['mean']:+.2f}" + (f" [{100 * t['ci95'][0]:+.2f},{100 * t['ci95'][1]:+.2f}]" if t.get("ci95") else "")
    for ds, info in res["strata"].items():
        print(f"{ds}: length cuts {info['length_tertile_cuts']}  sizes {info['sizes']}")
    for k, v in res["pooled"].items():
        print(f"{k:40s} {q(v)}  (n cells {v['n_cells']})")
    bad = [g for g in res["gate_S"] if not g["pass"]]
    print(f"gate S: {len(res['gate_S']) - len(bad)} / {len(res['gate_S'])} checkpoints pass" + (f"; FAIL {bad}" if bad else ""))
    if res["missing"]:
        print("missing:", res["missing"][:12], "..." if len(res["missing"]) > 12 else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
