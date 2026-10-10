"""VL3 frozen readings (reports/VL3/VL3_FULL_FINETUNE_FROZEN_20261010.md) over outputs/VL3/<dataset>_<backbone>_<objective>_s<seed>.json.
Per dataset x backbone cell: DEV image-macro / query Top-1 at the CAL-Top-1-selected epoch per objective (mean ± sd over the seeds present); paired
contrasts VCS − JS and VCS − softmax (95 % t over seeds when 3 exist; seed 0 only otherwise, labelled); fine-tuned − zero-shot (step 0); the
estimator view (DEV J_own at the CAL-J-selected epoch; DEV J_recal at the CAL-Top-1 epoch); the frozen-feature Table A value of the same objective
(VL1 / VL2 critic on frozen features, CAL-selected) for the fine-tuned − frozen contrast.  Gate 2 (step-0 DEV Top-1 within 0.3 of the frozen raw
value) is re-checked for every run and listed.  Writes reports/VL3/VL3_results.json.
    python scripts/vl3_aggregate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vl1_10_aggregate import readouts, tint  # noqa: E402

O = Path("/home/infres/yinwang/CS_QMI/outputs"); V = O / "VL3"; OUT = Path(__file__).resolve().parents[1] / "reports" / "VL3" / "VL3_results.json"
DATASETS, BACKBONES, OBJS, SEEDS = ("refcocog", "refcoco", "refcocoplus"), ("clip_b16", "siglip2_b16"), ("vcs", "js", "softmax"), (0, 1, 2)
RAW = {("refcocog", "clip_b16"): 0.6599, ("refcoco", "clip_b16"): 0.5814, ("refcocoplus", "clip_b16"): 0.6427,
       ("refcocog", "siglip2_b16"): 0.7548, ("refcoco", "siglip2_b16"): 0.6639, ("refcocoplus", "siglip2_b16"): 0.7385}
FROZEN_DIR = {("refcocog", "clip_b16"): "VL1_10", ("refcocog", "siglip2_b16"): "VL1_12_siglip2", ("refcoco", "clip_b16"): "VL2_refcoco_clip",
              ("refcoco", "siglip2_b16"): "VL2_refcoco_siglip2", ("refcocoplus", "clip_b16"): "VL2_refcocoplus_clip", ("refcocoplus", "siglip2_b16"): "VL2_refcocoplus_siglip2"}


def run(ds, bb, o, s):
    f = V / f"{ds}_{bb}_{o}_s{s}.json"
    return json.load(open(f)) if f.exists() else None


def frozen(ds, bb, o):
    vals = []
    for s in SEEDS:
        f = O / FROZEN_DIR[(ds, bb)] / f"{o}_Nall_s{s}.json"
        if not f.exists():
            return None
        r = readouts(json.load(open(f)), o); vals.append(r["task_dev" if o == "softmax" else "est_dev"]["top1_image_macro"])
    return float(np.mean(vals))


def main() -> int:
    res = {"cells": {}, "gate2": [], "missing": []}
    for ds in DATASETS:
        for bb in BACKBONES:
            cell = {}; per = {}
            for o in OBJS:
                rr = {s: run(ds, bb, o, s) for s in SEEDS}; rr = {s: r for s, r in rr.items() if r is not None}
                if not rr:
                    res["missing"].append(f"{ds}/{bb}/{o}"); continue
                per[o] = rr
                for s, r in rr.items():
                    st0 = r["history"][0]["own"]["dev"]["top1_image_macro"]; ok = abs(st0 - RAW[(ds, bb)]) <= 0.003
                    res["gate2"].append({"run": f"{ds}/{bb}/{o}/s{s}", "step0": st0, "frozen_raw": RAW[(ds, bb)], "pass": ok})
                top = [r["dev_at_cal_top1"]["own"]["dev"]["top1_image_macro"] for r in rr.values()]
                cell[o] = {"seeds": sorted(rr), "dev_top1_macro": tint(top), "dev_top1_query": tint([r["dev_at_cal_top1"]["own"]["dev"]["top1_query"] for r in rr.values()]),
                           "cal_top1_epoch": [r["selected"]["cal_top1_epoch"] for r in rr.values()],
                           "dev_J_own_at_calJ": tint([r["dev_at_cal_J"]["own"]["dev"]["J"] for r in rr.values()]),
                           "dev_J_recal_at_calTop1": tint([r["dev_at_cal_top1"]["recal"]["dev"]["J"] for r in rr.values()]),
                           "finetuned_minus_zeroshot": tint([t - r["history"][0]["own"]["dev"]["top1_image_macro"] for t, r in zip(top, rr.values())]),
                           "frozen_tableA": frozen(ds, bb, o)}
                if cell[o]["frozen_tableA"] is not None:
                    cell[o]["finetuned_minus_frozen"] = cell[o]["dev_top1_macro"]["mean"] - cell[o]["frozen_tableA"]
            con = {}
            for other in ("js", "softmax"):
                if "vcs" in per and other in per:
                    common = sorted(set(per["vcs"]) & set(per[other]))
                    d = [per["vcs"][s]["dev_at_cal_top1"]["own"]["dev"]["top1_image_macro"] - per[other][s]["dev_at_cal_top1"]["own"]["dev"]["top1_image_macro"] for s in common]
                    con[f"vcs_minus_{other}"] = {**tint(d), "seeds": common}
            cell["contrasts"] = con; res["cells"][f"{ds}/{bb}"] = cell
    json.dump(res, open(OUT, "w"), indent=1)
    p = lambda t: f"{100 * t['mean']:.2f}" + (f"±{100 * t['sd']:.2f}" if t.get("sd") is not None else "")
    q = lambda t: f"{100 * t['mean']:+.2f}" + (f" [{100 * t['ci95'][0]:+.2f},{100 * t['ci95'][1]:+.2f}]" if t.get("ci95") else " (1 seed)")
    for k, c in res["cells"].items():
        line = f"{k:22s}"
        for o in OBJS:
            if o in c:
                line += f" | {o} {p(c[o]['dev_top1_macro'])} (n {len(c[o]['seeds'])}; frozen {100 * c[o]['frozen_tableA']:.2f}; J_recal {100 * c[o]['dev_J_recal_at_calTop1']['mean']:.2f})" if c[o]["frozen_tableA"] is not None else f" | {o} {p(c[o]['dev_top1_macro'])}"
        print(line)
        if c["contrasts"]:
            print("    " + "  ".join(f"{kk} {q(v)}" for kk, v in c["contrasts"].items()))
    bad = [g for g in res["gate2"] if not g["pass"]]
    print(f"gate 2: {len(res['gate2']) - len(bad)} / {len(res['gate2'])} runs pass" + (f"; FAIL {bad}" if bad else ""))
    if res["missing"]:
        print("missing:", res["missing"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
