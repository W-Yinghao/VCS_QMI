"""VL1-22 Table C readings (reports/VL1/VL1_22_TABLE_C_FROZEN_20261010.md) over outputs/VL1_22/<dataset>_<backbone>_<objective>_s<seed>.json.
Per dataset x backbone cell and objective: DEV Acc@IoU 0.5 (image-macro; query) with detector proposals, mean ± sd over seeds; paired VCS − JS and
VCS − softmax (95 % t); pooled over the cells complete at three seeds (cells as units); the given-box -> detection drop per objective (paired by
seed); the proposal ceiling (any proposal IoU >= 0.5); Grounding DINO-T zero-shot on the same DEV expressions (outputs/VL1_20/gdino_tiny*_dev.jsonl);
gate T1 for every checkpoint.  Writes reports/VL1/VL1_22_results.json.
    python scripts/vl1_22_aggregate.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vl1_10_aggregate import tint  # noqa: E402

O = Path("/home/infres/yinwang/CS_QMI/outputs"); D = O / "VL1_22"; OUT = Path(__file__).resolve().parents[1] / "reports" / "VL1" / "VL1_22_results.json"
DATASETS, BACKBONES, OBJS, SEEDS = ("refcocog", "refcoco", "refcocoplus"), ("clip_b16", "siglip2_b16", "clip_l14_336", "siglip2_l16"), ("vcs", "js", "softmax"), (0, 1, 2)
GD = {"refcocog": "gdino_tiny_dev.jsonl", "refcoco": "gdino_tiny_refcoco_dev.jsonl", "refcocoplus": "gdino_tiny_refcocoplus_dev.jsonl"}


def gdino(ds):
    f = O / "VL1_20" / GD[ds]
    if not f.exists():
        return None
    per = defaultdict(list)
    for line in open(f):
        r = json.loads(line); per[r["image_id"]].append(r["hit_iou"])
    return {"acc_iou50_image_macro": float(np.mean([np.mean(v) for v in per.values()])), "acc_iou50_query": float(np.mean([x for v in per.values() for x in v])),
            "n_images": len(per), "n_queries": sum(len(v) for v in per.values())}


def main() -> int:
    res = {"cells": {}, "gate_T1": [], "missing": [], "gdino_dev": {ds: gdino(ds) for ds in DATASETS}}
    for ds in DATASETS:
        for bb in BACKBONES:
            cell, per = {}, {}
            for o in OBJS:
                rr = {s: json.load(open(D / f"{ds}_{bb}_{o}_s{s}.json")) for s in SEEDS if (D / f"{ds}_{bb}_{o}_s{s}.json").exists()}
                for s, r in rr.items():
                    res["gate_T1"].append({"run": r["tag"], **r["gate_T1"]})
                rr = {s: r for s, r in rr.items() if "detection_dev" in r}
                if not rr:
                    res["missing"].append(f"{ds}/{bb}/{o}"); continue
                per[o] = rr; det = [r["detection_dev"]["acc_iou50_image_macro"] for r in rr.values()]
                cell[o] = {"seeds": sorted(rr), "acc_iou50_image_macro": tint(det), "acc_iou50_query": tint([r["detection_dev"]["acc_iou50_query"] for r in rr.values()]),
                           "given_box_dev_top1": tint([r["given_box_dev"]["top1_image_macro"] for r in rr.values()]),
                           "detection_minus_given": tint([r["detection_dev"]["acc_iou50_image_macro"] - r["given_box_dev"]["top1_image_macro"] for r in rr.values()]),
                           "ceiling_iou50_image_macro": float(np.mean([r["detection_dev"]["ceiling_iou50_image_macro"] for r in rr.values()]))}
            con = {}
            for other in ("js", "softmax"):
                if "vcs" in per and other in per:
                    common = sorted(set(per["vcs"]) & set(per[other]))
                    d = [per["vcs"][s]["detection_dev"]["acc_iou50_image_macro"] - per[other][s]["detection_dev"]["acc_iou50_image_macro"] for s in common]
                    con[f"vcs_minus_{other}"] = {**tint(d), "seeds": common}
            cell["contrasts"] = con; res["cells"][f"{ds}/{bb}"] = cell
    full = {k: c for k, c in res["cells"].items() if all(o in c and len(c[o]["seeds"]) == 3 for o in OBJS)}
    pool = {"n_cells": len(full)}
    if len(full) >= 2:
        for other in ("js", "softmax"):
            k = f"vcs_minus_{other}"; pool[k] = {**tint([c["contrasts"][k]["mean"] for c in full.values()]),
                                                 "cells_ci_above_0": sum(c["contrasts"][k]["ci95"][0] > 0 for c in full.values()),
                                                 "cells_ci_below_0": sum(c["contrasts"][k]["ci95"][1] < 0 for c in full.values())}
        for o in OBJS:
            pool[f"{o}_detection_minus_given"] = tint([c[o]["detection_minus_given"]["mean"] for c in full.values()])
    res["pooled"] = pool
    json.dump(res, open(OUT, "w"), indent=1)
    p = lambda t: f"{100 * t['mean']:.2f}" + (f"±{100 * t['sd']:.2f}" if t.get("sd") is not None else "")
    q = lambda t: f"{100 * t['mean']:+.2f}" + (f" [{100 * t['ci95'][0]:+.2f},{100 * t['ci95'][1]:+.2f}]" if t.get("ci95") else " (1 seed)")
    for ds, g in res["gdino_dev"].items():
        if g:
            print(f"Grounding DINO-T zero-shot {ds} DEV: Acc@0.5 macro {100 * g['acc_iou50_image_macro']:.2f} query {100 * g['acc_iou50_query']:.2f} ({g['n_queries']} queries)")
    for k, c in res["cells"].items():
        line = f"{k:24s}" + "".join(f" | {o} {p(c[o]['acc_iou50_image_macro'])} (given {p(c[o]['given_box_dev_top1'])}; ceiling {100 * c[o]['ceiling_iou50_image_macro']:.2f})" for o in OBJS if o in c)
        print(line)
        if c["contrasts"]:
            print("    " + "  ".join(f"{kk} {q(v)}" for kk, v in c["contrasts"].items()))
    if pool.get("n_cells", 0) >= 2:
        print(f"pooled over {pool['n_cells']} cells: " + "  ".join(f"{k} {q(v)}" for k, v in pool.items() if isinstance(v, dict)))
    bad = [g for g in res["gate_T1"] if not g["pass"]]
    print(f"gate T1: {len(res['gate_T1']) - len(bad)} / {len(res['gate_T1'])} checkpoints pass" + (f"; FAIL {bad}" if bad else ""))
    if res["missing"]:
        print("missing:", res["missing"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
