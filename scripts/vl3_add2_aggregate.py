"""VL3 addendum 2 readings (reports/VL3/VL3_ADDENDUM2_20EPOCH_FROZEN_20261010.md): seed 0, B/16, three datasets, 20 vs 10 epochs.
Per dataset x backbone x objective: DEV image-macro Top-1 at the CAL-Top-1-selected epoch at 20 epochs (`_e20`) and at 10 epochs (VL3 seed 0), and the
selected epochs; per cell VCS − JS and VCS − softmax at both schedules, flagged when the sign changes or the contrast moves by more than 0.5 point;
gate 2 re-checked (step-0 within 0.3 of the frozen raw value).  Descriptive (one seed).  Writes reports/VL3/VL3_add2_results.json.
VL3 add. 4 ("paired" block): for every cell and objective, the seeds present at both schedules; per seed e20 − e10, and the schedule effect on the
contrasts [(VCS − X)_e20 − (VCS − X)_e10] paired by seed; pooled over the cells complete at three seeds at both schedules (cells as units, 95 % t).
    python scripts/vl3_add2_aggregate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vl1_10_aggregate import tint  # noqa: E402
from vl3_aggregate import RAW, V  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "reports" / "VL3" / "VL3_add2_results.json"
DATASETS, BACKBONES, OBJS = ("refcocog", "refcoco", "refcocoplus"), ("clip_b16", "siglip2_b16"), ("vcs", "js", "softmax")


def load(ds, bb, o, suffix):
    f = V / f"{ds}_{bb}_{o}_s0{suffix}.json"
    return json.load(open(f)) if f.exists() else None


def load_s(ds, bb, o, s, suffix):
    f = V / f"{ds}_{bb}_{o}_s{s}{suffix}.json"
    return json.load(open(f)) if f.exists() else None


def top1(r):
    return r["dev_at_cal_top1"]["own"]["dev"]["top1_image_macro"]


def main() -> int:
    res = {"cells": {}, "gate2": [], "missing": []}
    for ds in DATASETS:
        for bb in BACKBONES:
            cell = {}
            for o in OBJS:
                a, b = load(ds, bb, o, "_e20"), load(ds, bb, o, "")
                if a is None or b is None:
                    res["missing"].append(f"{ds}/{bb}/{o}"); continue
                st0 = a["history"][0]["own"]["dev"]["top1_image_macro"]
                res["gate2"].append({"run": f"{ds}/{bb}/{o}/e20", "step0": st0, "frozen_raw": RAW[(ds, bb)], "pass": abs(st0 - RAW[(ds, bb)]) <= 0.003})
                cell[o] = {"e20": top1(a), "e10": top1(b), "e20_minus_e10": top1(a) - top1(b),
                           "epoch_e20": a["selected"]["cal_top1_epoch"], "epoch_e10": b["selected"]["cal_top1_epoch"],
                           "dev_J_recal_e20": a["dev_at_cal_top1"]["recal"]["dev"]["J"], "dev_J_recal_e10": b["dev_at_cal_top1"]["recal"]["dev"]["J"]}
            for other in ("js", "softmax"):
                if "vcs" in cell and other in cell:
                    c20, c10 = cell["vcs"]["e20"] - cell[other]["e20"], cell["vcs"]["e10"] - cell[other]["e10"]
                    cell[f"vcs_minus_{other}"] = {"e20": c20, "e10": c10, "flag": (c20 > 0) != (c10 > 0) or abs(c20 - c10) > 0.005}
            res["cells"][f"{ds}/{bb}"] = cell
    paired = {"cells": {}}                                                        # VL3 add. 4
    for ds in DATASETS:
        for bb in BACKBONES:
            rr = {(o, s, suf): load_s(ds, bb, o, s, suf) for o in OBJS for s in (0, 1, 2) for suf in ("_e20", "")}
            seeds = [s for s in (0, 1, 2) if all(rr[(o, s, suf)] is not None for o in OBJS for suf in ("_e20", ""))]
            if not seeds:
                continue
            c = {"seeds": seeds}
            for o in OBJS:
                c[f"{o}_e20_minus_e10"] = tint([top1(rr[(o, s, "_e20")]) - top1(rr[(o, s, "")]) for s in seeds])
                c[f"{o}_e20"] = tint([top1(rr[(o, s, "_e20")]) for s in seeds])
            for other in ("js", "softmax"):
                d20 = [top1(rr[("vcs", s, "_e20")]) - top1(rr[(other, s, "_e20")]) for s in seeds]
                d10 = [top1(rr[("vcs", s, "")]) - top1(rr[(other, s, "")]) for s in seeds]
                c[f"vcs_minus_{other}_e20"] = tint(d20); c[f"vcs_minus_{other}_e10"] = tint(d10)
                c[f"schedule_effect_on_vcs_minus_{other}"] = tint([x - y for x, y in zip(d20, d10)])
            paired["cells"][f"{ds}/{bb}"] = c
    full = {k: c for k, c in paired["cells"].items() if len(c["seeds"]) == 3}
    paired["n_cells_three_seeds"] = len(full)
    if len(full) >= 2:
        for key in ("vcs_minus_js_e20", "vcs_minus_softmax_e20", "vcs_minus_js_e10", "vcs_minus_softmax_e10", "schedule_effect_on_vcs_minus_js",
                    "schedule_effect_on_vcs_minus_softmax", "vcs_e20_minus_e10", "js_e20_minus_e10", "softmax_e20_minus_e10"):
            paired[f"pooled_{key}"] = tint([c[key]["mean"] for c in full.values()])
    res["paired"] = paired
    json.dump(res, open(OUT, "w"), indent=1)
    for k, c in res["cells"].items():
        line = f"{k:22s}"
        for o in OBJS:
            if o in c:
                x = c[o]; line += f" | {o} {100 * x['e20']:.2f} (ep {x['epoch_e20']}) vs {100 * x['e10']:.2f} (ep {x['epoch_e10']}) {100 * x['e20_minus_e10']:+.2f}"
        print(line)
        for other in ("js", "softmax"):
            if f"vcs_minus_{other}" in c:
                x = c[f"vcs_minus_{other}"]; print(f"    VCS − {other}: e20 {100 * x['e20']:+.2f}  e10 {100 * x['e10']:+.2f}" + ("  FLAG" if x["flag"] else ""))
    q = lambda t: f"{100 * t['mean']:+.2f}" + (f" [{100 * t['ci95'][0]:+.2f},{100 * t['ci95'][1]:+.2f}]" if t.get("ci95") else "")
    for k, c in paired["cells"].items():
        if len(c["seeds"]) > 1:
            print(f"paired {k} (seeds {c['seeds']}): " + "  ".join(f"{kk} {q(c[kk])}" for kk in ("vcs_minus_softmax_e20", "vcs_minus_softmax_e10", "schedule_effect_on_vcs_minus_softmax", "vcs_minus_js_e20")))
    for k, v in paired.items():
        if k.startswith("pooled_"):
            print(f"{k} over {paired['n_cells_three_seeds']} cells: {q(v)}")
    bad = [g for g in res["gate2"] if not g["pass"]]
    print(f"gate 2: {len(res['gate2']) - len(bad)} / {len(res['gate2'])} runs pass" + (f"; FAIL {bad}" if bad else ""))
    if res["missing"]:
        print("missing:", res["missing"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
