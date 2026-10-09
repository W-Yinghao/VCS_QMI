"""VL2 frozen readings (reports/VL2/VL2_REFCOCO_PLUS_FROZEN_20261009.md) over outputs/VL2_<dataset>_<features>/: per dataset and feature set, the
Table A cells (raw; VCS / JS / RFF estimator-selected, softmax task-selected; mean ± sd over seeds; CAL J), VCS − JS and learned − raw paired by
seed (95 % t), softmax − VCS; the estimator-probe consistency (Spearman ρ between the VCS critic's CAL J and its learned Top-1 across the five
feature sets, descriptive); RefCOCO+ − RefCOCO per feature set and route (unpaired means, descriptive).  Writes reports/VL2/VL2_results.json.
Cells that are not complete are listed as missing and left out (partial runs are never averaged).
    python scripts/vl2_aggregate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vl1_10_aggregate import readouts, tint  # noqa: E402

O = Path("/home/infres/yinwang/CS_QMI/outputs"); OUT = Path(__file__).resolve().parents[1] / "reports" / "VL2" / "VL2_results.json"
DATASETS, FEATS, SEEDS = ("refcoco", "refcocoplus"), ("clip", "siglip2", "fgclip2", "fgclip1", "reclip"), (0, 1, 2)


def load(d, route, s):
    f = O / d / f"{route}_Nall_s{s}.json"
    return readouts(json.load(open(f)), route) if f.exists() else None


def main() -> int:
    res = {"cells": {}, "contrasts": {}, "probe": {}, "plus_minus_refcoco": {}, "missing": []}
    for ds in DATASETS:
        for ft in FEATS:
            d = f"VL2_{ds}_{ft}"; key = f"{ds}/{ft}"; cell = {}
            if (O / d / "raw.json").exists():
                cell["raw"] = json.load(open(O / d / "raw.json"))["dev"]["top1_image_macro"]
            rows = {}
            for route in ("vcs", "js", "softmax", "rff"):
                rr = [load(d, route, s) for s in SEEDS]
                if not all(rr):
                    res["missing"].append(f"{key}/{route}"); continue
                rows[route] = rr; k = "task_dev" if route == "softmax" else "est_dev"
                cell[route] = {"top1": tint([x[k]["top1_image_macro"] for x in rr])}
                if "cal_J" in rr[0]:
                    cell[route]["cal_J"] = tint([x["cal_J"] for x in rr])
            res["cells"][key] = cell; con = {}
            if "vcs" in rows and "js" in rows:
                con["vcs_minus_js_top1"] = tint([a["est_dev"]["top1_image_macro"] - b["est_dev"]["top1_image_macro"] for a, b in zip(rows["vcs"], rows["js"])])
                con["vcs_minus_js_J"] = tint([a["cal_J"] - b["cal_J"] for a, b in zip(rows["vcs"], rows["js"])])
            if "raw" in cell:
                for route, rr in rows.items():
                    k = "task_dev" if route == "softmax" else "est_dev"
                    con[f"{route}_minus_raw"] = tint([x[k]["top1_image_macro"] - cell["raw"] for x in rr])
            if "vcs" in rows and "softmax" in rows:
                con["softmax_minus_vcs"] = tint([a["task_dev"]["top1_image_macro"] - b["est_dev"]["top1_image_macro"] for a, b in zip(rows["softmax"], rows["vcs"])])
            res["contrasts"][key] = con
        js = [(ft, res["cells"][f"{ds}/{ft}"]["vcs"]["cal_J"]["mean"], res["cells"][f"{ds}/{ft}"]["vcs"]["top1"]["mean"], res["cells"][f"{ds}/{ft}"].get("raw"))
              for ft in FEATS if "vcs" in res["cells"].get(f"{ds}/{ft}", {})]
        if len(js) >= 3:
            rho = stats.spearmanr([x[1] for x in js], [x[2] for x in js]).statistic
            rho_raw = stats.spearmanr([x[1] for x in js], [x[3] for x in js]).statistic if all(x[3] is not None for x in js) else None
            res["probe"][ds] = {"n_feature_sets": len(js), "spearman_J_vs_learned_top1": float(rho), "spearman_J_vs_raw_top1": None if rho_raw is None else float(rho_raw),
                                "order_by_J": [x[0] for x in sorted(js, key=lambda z: -z[1])], "order_by_learned_top1": [x[0] for x in sorted(js, key=lambda z: -z[2])]}
    for ft in FEATS:
        a, b = res["cells"].get(f"refcocoplus/{ft}", {}), res["cells"].get(f"refcoco/{ft}", {}); out = {}
        if "raw" in a and "raw" in b:
            out["raw"] = a["raw"] - b["raw"]
        for route in ("vcs", "js", "softmax", "rff"):
            if route in a and route in b:
                out[route] = {"top1": a[route]["top1"]["mean"] - b[route]["top1"]["mean"]}
                if "cal_J" in a[route]:
                    out[route]["cal_J"] = a[route]["cal_J"]["mean"] - b[route]["cal_J"]["mean"]
        res["plus_minus_refcoco"][ft] = out
    json.dump(res, open(OUT, "w"), indent=1)
    p = lambda t: f"{100 * t['mean']:.2f}" + (f"±{100 * t['sd']:.2f}" if t.get("sd") is not None else "")
    q = lambda t: f"{100 * t['mean']:+.2f} [{100 * t['ci95'][0]:+.2f},{100 * t['ci95'][1]:+.2f}]" if t.get("ci95") else f"{100 * t['mean']:+.2f}"
    for key, c in res["cells"].items():
        line = f"{key:22s} raw {100 * c['raw']:.2f}" if "raw" in c else f"{key:22s} raw —"
        for route in ("vcs", "js", "softmax", "rff"):
            if route in c:
                line += f" | {route} {p(c[route]['top1'])}" + (f" J {p(c[route]['cal_J'])}" if "cal_J" in c[route] else "")
        print(line)
        cc = res["contrasts"].get(key, {})
        if cc:
            print("    " + "  ".join(f"{k} {q(v)}" for k, v in cc.items()))
    for ds, v in res["probe"].items():
        print(f"probe {ds}: rho(J, learned) {v['spearman_J_vs_learned_top1']:.2f}  rho(J, raw) {v['spearman_J_vs_raw_top1']}  J order {v['order_by_J']}  learned order {v['order_by_learned_top1']}")
    for ft, v in res["plus_minus_refcoco"].items():
        if v:
            print(f"RefCOCO+ − RefCOCO {ft}: " + json.dumps({k: (round(100 * x, 2) if not isinstance(x, dict) else {kk: round(100 * xx, 2) for kk, xx in x.items()}) for k, x in v.items()}))
    if res["missing"]:
        print("missing:", res["missing"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
