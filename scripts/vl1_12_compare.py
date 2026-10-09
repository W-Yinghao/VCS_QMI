"""VL1-12 frozen reading (reports/VL1/VL1_12_ADDENDUM_FROZEN_20261009.md): per route at N = all, FG-CLIP 2 Base − CLIP ViT-B/16 on CAL common J
and on DEV image-macro Top-1, paired by init seed (95 % t intervals); VCS − JS on the FG-CLIP 2 features.  Readout per route as in VL1-10:
estimator-selected checkpoint (CAL max common J) for VCS / JS / RFF; task-selected (DEV max Top-1, optimistic) for softmax; raw has one value.
Writes reports/VL1/VL1_12_results.json.
    python scripts/vl1_12_compare.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vl1_10_aggregate import readouts, tint  # noqa: E402

O = Path("/home/infres/yinwang/CS_QMI/outputs"); CLIP, FG = O / "VL1_10", O / "VL1_12"
OUT = Path(__file__).resolve().parents[1] / "reports" / "VL1" / "VL1_12_results.json"
SEEDS = (0, 1, 2)


def get(d, route, s):
    f = d / f"{route}_Nall_s{s}.json"
    return readouts(json.load(open(f)), route) if f.exists() else None


def main() -> int:
    res = {"features": {"clip": "open_clip ViT-B-16-quickgelu (OpenAI), tight crop", "fgclip2": "FG-CLIP 2 Base released checkpoint, official RoIAlign region API"},
           "raw": {}, "routes": {}, "vcs_minus_js_fgclip2": None, "missing": []}
    for name, d in (("clip", CLIP), ("fgclip2", FG)):
        r = json.load(open(d / "raw.json"))
        res["raw"][name] = {k: r["dev"][k] for k in ("top1_image_macro", "top1_query", "top1_all_objects")}
    res["raw"]["fgclip2_minus_clip_top1_macro"] = res["raw"]["fgclip2"]["top1_image_macro"] - res["raw"]["clip"]["top1_image_macro"]
    for route in ("vcs", "js", "softmax", "rff"):
        a = [get(CLIP, route, s) for s in SEEDS]; b = [get(FG, route, s) for s in SEEDS]
        if not (all(a) and all(b)):
            res["missing"].append(route); continue
        key = "task_dev" if route == "softmax" else "est_dev"
        cell = {"readout": key + (" (selected on DEV: optimistic)" if key == "task_dev" else " (selected on CAL)")}
        for nm, rows in (("clip", a), ("fgclip2", b)):
            cell[f"{nm}_top1_macro"] = tint([x[key]["top1_image_macro"] for x in rows])
            cell[f"{nm}_top1_all_objects"] = tint([x[key]["top1_all_objects"] for x in rows])
            if "cal_J" in rows[0]:
                cell[f"{nm}_cal_J"] = tint([x["cal_J"] for x in rows])
        cell["fgclip2_minus_clip_top1_macro"] = tint([y[key]["top1_image_macro"] - x[key]["top1_image_macro"] for x, y in zip(a, b)])
        if "cal_J" in a[0]:
            cell["fgclip2_minus_clip_cal_J"] = tint([y["cal_J"] - x["cal_J"] for x, y in zip(a, b)])
        cell["fgclip2_minus_raw_fgclip2_top1_macro"] = tint([y[key]["top1_image_macro"] - res["raw"]["fgclip2"]["top1_image_macro"] for y in b])
        res["routes"][route] = cell
    v = [get(FG, "vcs", s) for s in SEEDS]; j = [get(FG, "js", s) for s in SEEDS]
    if all(v) and all(j):
        res["vcs_minus_js_fgclip2"] = {"est_dev_top1_macro": tint([x["est_dev"]["top1_image_macro"] - y["est_dev"]["top1_image_macro"] for x, y in zip(v, j)]),
                                       "task_dev_top1_macro": tint([x["task_dev"]["top1_image_macro"] - y["task_dev"]["top1_image_macro"] for x, y in zip(v, j)]),
                                       "cal_J": tint([x["cal_J"] - y["cal_J"] for x, y in zip(v, j)])}
    json.dump(res, open(OUT, "w"), indent=1)
    p = lambda t: f"{100 * t['mean']:.2f}" + (f" [{100 * t['ci95'][0]:+.2f}, {100 * t['ci95'][1]:+.2f}]" if t.get("ci95") else "")
    print(f"raw DEV macro: CLIP {100 * res['raw']['clip']['top1_image_macro']:.2f}  FG-CLIP 2 {100 * res['raw']['fgclip2']['top1_image_macro']:.2f}")
    for route, c in res["routes"].items():
        line = f"{route:8s} CLIP {p(c['clip_top1_macro'])}  FG {p(c['fgclip2_top1_macro'])}  Δ {p(c['fgclip2_minus_clip_top1_macro'])}"
        if "fgclip2_minus_clip_cal_J" in c:
            line += f" | CAL J CLIP {p(c['clip_cal_J'])} FG {p(c['fgclip2_cal_J'])} Δ {p(c['fgclip2_minus_clip_cal_J'])}"
        print(line)
    if res["vcs_minus_js_fgclip2"]:
        print("VCS − JS on FG-CLIP 2:", {k: p(t) for k, t in res["vcs_minus_js_fgclip2"].items()})
    if res["missing"]:
        print("missing:", res["missing"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
