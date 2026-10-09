"""VL1-12 frozen reading (reports/VL1/VL1_12_ADDENDUM_FROZEN_20261009.md): per route at N = all, FG-CLIP 2 Base − CLIP ViT-B/16 on CAL common J
and on DEV image-macro Top-1, paired by init seed (95 % t intervals); VCS − JS on the FG-CLIP 2 features.  Readout per route as in VL1-10:
estimator-selected checkpoint (CAL max common J) for VCS / JS / RFF; task-selected (DEV max Top-1, optimistic) for softmax; raw has one value.
Writes reports/VL1/VL1_12_results.json.  Generalised (VL1-12 addendum 2) to any two feature sets of the same protocol:
    python scripts/vl1_12_compare.py
    python scripts/vl1_12_compare.py --a clip:VL1_10 --b siglip2:VL1_12_siglip2 --out reports/VL1/VL1_12s_siglip2_minus_clip.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vl1_10_aggregate import readouts, tint  # noqa: E402

O = Path("/home/infres/yinwang/CS_QMI/outputs")
OUT = Path(__file__).resolve().parents[1] / "reports" / "VL1" / "VL1_12_results.json"
SEEDS = (0, 1, 2)


def get(d, route, s):
    f = d / f"{route}_Nall_s{s}.json"
    return readouts(json.load(open(f)), route) if f.exists() else None


DESC = {"clip": "open_clip ViT-B-16-quickgelu (OpenAI), tight crop", "fgclip2": "FG-CLIP 2 Base released checkpoint, official RoIAlign region API",
        "siglip2": "SigLIP 2 Base (patch16-224), tight crop (adaptation)", "reclip": "ReCLIP isolation features (RN50x16 + ViT-B/32 x crop / blur)"}


def main() -> int:
    global OUT
    ap = argparse.ArgumentParser(); ap.add_argument("--a", default="clip:VL1_10"); ap.add_argument("--b", default="fgclip2:VL1_12"); ap.add_argument("--out", default=str(OUT)); a = ap.parse_args()
    (na, da), (nb, db) = [x.split(":") for x in (a.a, a.b)]; A, B = O / da, O / db
    OUT = Path(a.out) if Path(a.out).is_absolute() else Path(__file__).resolve().parents[1] / a.out
    res = {"features": {na: DESC.get(na, da), nb: DESC.get(nb, db)}, "pair": f"{nb} − {na}", "raw": {}, "routes": {}, f"vcs_minus_js_{nb}": None, "missing": []}
    for name, d in ((na, A), (nb, B)):
        r = json.load(open(d / "raw.json"))
        res["raw"][name] = {k: r["dev"][k] for k in ("top1_image_macro", "top1_query", "top1_all_objects")}
    res["raw"][f"{nb}_minus_{na}_top1_macro"] = res["raw"][nb]["top1_image_macro"] - res["raw"][na]["top1_image_macro"]
    for route in ("vcs", "js", "softmax", "rff"):
        xa = [get(A, route, s) for s in SEEDS]; xb = [get(B, route, s) for s in SEEDS]
        if not (all(xa) and all(xb)):
            res["missing"].append(route); continue
        key = "task_dev" if route == "softmax" else "est_dev"
        cell = {"readout": key + (" (selected on DEV: optimistic)" if key == "task_dev" else " (selected on CAL)")}
        for nm, rows in ((na, xa), (nb, xb)):
            cell[f"{nm}_top1_macro"] = tint([x[key]["top1_image_macro"] for x in rows])
            cell[f"{nm}_top1_all_objects"] = tint([x[key]["top1_all_objects"] for x in rows])
            if "cal_J" in rows[0]:
                cell[f"{nm}_cal_J"] = tint([x["cal_J"] for x in rows])
        cell[f"{nb}_minus_{na}_top1_macro"] = tint([y[key]["top1_image_macro"] - x[key]["top1_image_macro"] for x, y in zip(xa, xb)])
        if "cal_J" in xa[0]:
            cell[f"{nb}_minus_{na}_cal_J"] = tint([y["cal_J"] - x["cal_J"] for x, y in zip(xa, xb)])
        cell[f"{nb}_minus_raw_{nb}_top1_macro"] = tint([y[key]["top1_image_macro"] - res["raw"][nb]["top1_image_macro"] for y in xb])
        res["routes"][route] = cell
    v = [get(B, "vcs", s) for s in SEEDS]; j = [get(B, "js", s) for s in SEEDS]
    if all(v) and all(j):
        res[f"vcs_minus_js_{nb}"] = {"est_dev_top1_macro": tint([x["est_dev"]["top1_image_macro"] - y["est_dev"]["top1_image_macro"] for x, y in zip(v, j)]),
                                       "task_dev_top1_macro": tint([x["task_dev"]["top1_image_macro"] - y["task_dev"]["top1_image_macro"] for x, y in zip(v, j)]),
                                       "cal_J": tint([x["cal_J"] - y["cal_J"] for x, y in zip(v, j)])}
    json.dump(res, open(OUT, "w"), indent=1)
    p = lambda t: f"{100 * t['mean']:.2f}" + (f" [{100 * t['ci95'][0]:+.2f}, {100 * t['ci95'][1]:+.2f}]" if t.get("ci95") else "")
    print(f"raw DEV macro: {na} {100 * res['raw'][na]['top1_image_macro']:.2f}  {nb} {100 * res['raw'][nb]['top1_image_macro']:.2f}")
    for route, c in res["routes"].items():
        line = f"{route:8s} {na} {p(c[f'{na}_top1_macro'])}  {nb} {p(c[f'{nb}_top1_macro'])}  Δ {p(c[f'{nb}_minus_{na}_top1_macro'])}"
        if f"{nb}_minus_{na}_cal_J" in c:
            line += f" | CAL J {na} {p(c[f'{na}_cal_J'])} {nb} {p(c[f'{nb}_cal_J'])} Δ {p(c[f'{nb}_minus_{na}_cal_J'])}"
        print(line)
    if res[f"vcs_minus_js_{nb}"]:
        print(f"VCS − JS on {nb}:", {k: p(t) for k, t in res[f"vcs_minus_js_{nb}"].items()})
    if res["missing"]:
        print("missing:", res["missing"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
