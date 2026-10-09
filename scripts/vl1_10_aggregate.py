"""VL1-10 / VL1-11 aggregation (frozen readings, reports/VL1/VL1_FROZEN_PROTOCOL.md) over outputs/VL1_10/*.json.
Per route x N: DEV Top-1 (image-macro; query-weighted; all-objects) of the task-selected checkpoint (selected ON DEV — optimistic) and of the
estimator-selected checkpoint (selected on CAL by common J — unbiased for DEV), CAL common J, mean +- sd over seeds.  Paired-by-seed contrasts with
95 % t intervals: VCS − JS (Top-1 and CAL J), each learned route − raw CLIP (estimator-selected Top-1; task-selected for softmax with the bias noted).
VL1-11: phrase-uniform law, prior-corrected − raw-f Top-1 per route.  Writes reports/VL1/VL1_10_results.json and prints the tables.
    python scripts/vl1_10_aggregate.py
    python scripts/vl1_10_aggregate.py --dir outputs/VL1_12 --out reports/VL1/VL1_12_table.json --ns all --routes vcs js softmax rff   (VL1-12 cache)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

D = Path("/home/infres/yinwang/CS_QMI/outputs/VL1_10"); OUT = Path(__file__).resolve().parents[1] / "reports" / "VL1" / "VL1_10_results.json"
ROUTES, NS, SEEDS = ("vcs", "js", "softmax", "siglip", "rff"), ("1000", "4000", "all"), (0, 1, 2)


def tint(x):
    x = np.asarray(x, float); m = float(x.mean())
    if len(x) < 2:
        return {"mean": m, "ci95": None, "n": len(x)}
    h = float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))); return {"mean": m, "sd": float(x.std(ddof=1)), "ci95": [m - h, m + h], "n": len(x)}


def load(route, n, seed, suffix=""):
    f = D / f"{route}_N{n}_s{seed}{suffix}.json"
    return json.load(open(f)) if f.exists() else None


def readouts(r, route):
    if route == "rff":
        return {"task_dev": r["dev"], "est_dev": r["dev"], "cal_J": r["cal"]["J_common"]}
    out = {"task_dev": r["task_selected"]["dev"]}
    if r.get("estimator_selected"):
        out["est_dev"] = r["estimator_selected"]["dev"]; out["cal_J"] = r["estimator_selected"]["cal"]["J_common"]
    return out


def main() -> int:
    global D, OUT, ROUTES, NS
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default=str(D)); ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--ns", nargs="+", default=list(NS)); ap.add_argument("--routes", nargs="+", default=list(ROUTES)); a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    D = Path(a.dir) if Path(a.dir).is_absolute() else Path("/home/infres/yinwang/CS_QMI") / a.dir
    OUT = Path(a.out) if Path(a.out).is_absolute() else root / a.out; ROUTES, NS = tuple(a.routes), tuple(a.ns)
    raw = json.load(open(D / "raw.json")) if (D / "raw.json").exists() else None
    res = {"raw": raw, "table": {}, "contrasts": {}, "vl1_11": {}, "missing": []}
    for route in ROUTES:
        for n in NS:
            rows = []
            for s in SEEDS:
                r = load(route, n, s)
                if r is None:
                    res["missing"].append(f"{route}_N{n}_s{s}"); continue
                rows.append(readouts(r, route))
            if not rows:
                continue
            cell = {"n_seeds": len(rows)}
            for key in ("task_dev", "est_dev"):
                if all(key in x for x in rows):
                    for m in ("top1_image_macro", "top1_query", "top1_all_objects"):
                        cell[f"{key}_{m}"] = tint([x[key][m] for x in rows])
            if all("cal_J" in x for x in rows):
                cell["cal_J"] = tint([x["cal_J"] for x in rows])
            res["table"][f"{route}/N{n}"] = cell
    # contrasts (paired by seed)
    rawm = raw["dev"]["top1_image_macro"] if raw else None
    for n in NS:
        a = [load("vcs", n, s) for s in SEEDS]; b = [load("js", n, s) for s in SEEDS]
        if all(a) and all(b):
            ra, rb = [readouts(x, "vcs") for x in a], [readouts(x, "js") for x in b]
            res["contrasts"][f"vcs_minus_js/N{n}"] = {"est_dev_top1_macro": tint([x["est_dev"]["top1_image_macro"] - y["est_dev"]["top1_image_macro"] for x, y in zip(ra, rb)]),
                                                     "task_dev_top1_macro": tint([x["task_dev"]["top1_image_macro"] - y["task_dev"]["top1_image_macro"] for x, y in zip(ra, rb)]),
                                                     "cal_J": tint([x["cal_J"] - y["cal_J"] for x, y in zip(ra, rb)])}
        if rawm is not None:
            for route in ROUTES:
                rr = [load(route, n, s) for s in SEEDS]
                if all(rr):
                    key = "task_dev" if route == "softmax" else "est_dev"
                    res["contrasts"][f"{route}_minus_raw/N{n}"] = {"which": key + (" (selected on DEV: optimistic)" if key == "task_dev" else " (selected on CAL)"),
                                                                  "top1_macro": tint([readouts(x, route)[key]["top1_image_macro"] - rawm for x in rr])}
    # VL1-11 (phrase-uniform law, N all)
    for route in ("vcs", "js", "softmax"):
        rr = [load(route, "all", s, "_phraseuniform") for s in SEEDS]
        if all(rr):
            key = "task_selected" if route == "softmax" else "estimator_selected"
            d = [x[key]["dev"]["top1_image_macro_prior_corrected"] - x[key]["dev"]["top1_image_macro"] for x in rr]
            t = tint(d); t["reading"] = "holds" if (t["mean"] >= 0 and (t["ci95"] is None or t["ci95"][0] > -0.002)) else "not shown"
            res["vl1_11"][route] = {"prior_corrected_minus_raw_f_top1_macro": t, "raw_f": tint([x[key]["dev"]["top1_image_macro"] for x in rr]),
                                    "prior_corrected": tint([x[key]["dev"]["top1_image_macro_prior_corrected"] for x in rr])}
    for route in ("vcs", "js"):
        c = load(route, "all", 0, "_concat")
        if c:
            res["table"][f"{route}/Nall_concat_sensitivity"] = {"est_dev_top1_macro": c["estimator_selected"]["dev"]["top1_image_macro"], "cal_J": c["estimator_selected"]["cal"]["J_common"],
                                                                "task_dev_top1_macro": c["task_selected"]["dev"]["top1_image_macro"]}
    json.dump(res, open(OUT, "w"), indent=1)
    f = lambda t: "—" if t is None else (f"{100 * t['mean']:.2f}" + (f"±{100 * t['sd']:.2f}" if t.get("sd") is not None else ""))
    if raw:
        print(f"raw ({D.name} features) DEV top1 macro {100 * rawm:.2f} (all-obj {100 * raw['dev']['top1_all_objects']:.2f})")
    print("route/N        est-sel DEV top1     task-sel DEV top1 (DEV-selected)   CAL J (x100)")
    for k, c in res["table"].items():
        if "sensitivity" in k:
            print(f"{k:28s} est {100 * c['est_dev_top1_macro']:.2f}  task {100 * c['task_dev_top1_macro']:.2f}  J {100 * c['cal_J']:.2f}"); continue
        print(f"{k:14s} {f(c.get('est_dev_top1_image_macro')):>18s}   {f(c.get('task_dev_top1_image_macro')):>18s}   {f(c.get('cal_J')):>14s}")
    for k, v in res["contrasts"].items():
        print(k, json.dumps({kk: (round(100 * vv["mean"], 2), [round(100 * z, 2) for z in vv["ci95"]] if vv.get("ci95") else None) if isinstance(vv, dict) and "mean" in vv else vv for kk, vv in v.items()}))
    for k, v in res["vl1_11"].items():
        t = v["prior_corrected_minus_raw_f_top1_macro"]; print(f"VL1-11 {k}: corrected − raw f = {100 * t['mean']:+.2f} {[round(100 * z, 2) for z in t['ci95']] if t.get('ci95') else ''} -> {t['reading']}")
    if res["missing"]:
        print("missing:", res["missing"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
