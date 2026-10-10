"""VL3 addendum 6 readings (reports/VL3/VL3_ADDENDUM6_TRANSFER_FROZEN_20261010.md) over outputs/VL3_transfer/ (and outputs/VL3_strata/ for the in-domain
reference).  Cells = directed (source -> target) x backbone = 24.  Per cell and objective: transfer DEV image-macro Top-1 on the kept target images
(mean ± sd over seeds); zero-shot on the same images; the in-domain model (trained on the target, same backbone / objective / seed) on the same
images, from the add. 5 per-query hits.  Paired by seed: VCS − softmax and VCS − JS (transfer); the transfer gap (in-domain − transfer) per objective.
Primary: VCS − softmax transfer Top-1 pooled over the 24 cells (95 % t).  Secondary: the same for VCS − JS; gap contrasts; split UNC pair
(RefCOCO <-> RefCOCO+, same images) vs RefCOCOg <-> UNC.  Gate X listed.  Writes reports/VL3/VL3_add6_results.json.
    python scripts/vl3_add6_aggregate.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vl1_10_aggregate import tint  # noqa: E402

D = Path("/home/infres/yinwang/CS_QMI/outputs/VL3_transfer"); ST = Path("/home/infres/yinwang/CS_QMI/outputs/VL3_strata")
OUT = Path(__file__).resolve().parents[1] / "reports" / "VL3" / "VL3_add6_results.json"
DS, BACKBONES, OBJS, SEEDS = ("refcocog", "refcoco", "refcocoplus"), ("clip_b16", "siglip2_b16", "clip_l14_336", "siglip2_l16"), ("vcs", "js", "softmax"), (0, 1, 2)


def macro_on(hits, img_of_query, keep):
    per = defaultdict(list)
    for h, i in zip(hits, img_of_query):
        if i in keep:
            per[i].append(h)
    return float(np.mean([np.mean(v) for v in per.values()])) if per else None


def main() -> int:
    res = {"cells": {}, "gate_X": [], "missing": []}
    meta = {t: [q["image_id"] for q in json.load(open(ST / f"{t}_meta.json"))["queries"]] for t in DS if (ST / f"{t}_meta.json").exists()}
    for S in DS:
        for T in DS:
            if S == T:
                continue
            for bb in BACKBONES:
                z = D / f"{S}_to_{T}_{bb}_zeroshot.json"
                if not z.exists():
                    res["missing"].append(f"{S}->{T}/{bb}/zeroshot"); continue
                z = json.load(open(z)); res["gate_X"].append({"cell": f"{S}->{T}/{bb}", **z["gate_X"]}); keep = set(z["kept"]["image_ids"])
                cell = {"zeroshot_kept": z["kept"]["top1_image_macro"], "n_kept_images": z["kept"]["n_images"]}; tr, ind = {}, {}
                for o in OBJS:
                    for s in SEEDS:
                        f = D / f"{S}_to_{T}_{bb}_{o}_s{s}.json"
                        if f.exists():
                            tr[(o, s)] = json.load(open(f))["top1_image_macro"]
                        g = ST / f"{T}_{bb}_{o}_s{s}.json"
                        if g.exists() and T in meta:
                            h = json.load(open(g))["hits"]
                            if h is not None:
                                ind[(o, s)] = macro_on(h, meta[T], keep)
                seeds = [s for s in SEEDS if all((o, s) in tr for o in OBJS)]
                if len(seeds) < 3:
                    res["missing"].append(f"{S}->{T}/{bb}"); continue
                for o in OBJS:
                    cell[o] = {"transfer": tint([tr[(o, s)] for s in seeds])}
                    if all((o, s) in ind for s in seeds):
                        cell[o]["in_domain_same_images"] = tint([ind[(o, s)] for s in seeds]); cell[o]["gap"] = tint([ind[(o, s)] - tr[(o, s)] for s in seeds])
                for other in ("js", "softmax"):
                    cell[f"vcs_minus_{other}"] = tint([tr[("vcs", s)] - tr[(other, s)] for s in seeds])
                    if all(("vcs", s) in ind and (other, s) in ind for s in seeds):
                        cell[f"gap_vcs_minus_{other}"] = tint([(ind[("vcs", s)] - tr[("vcs", s)]) - (ind[(other, s)] - tr[(other, s)]) for s in seeds])
                res["cells"][f"{S}->{T}/{bb}"] = cell
    groups = {"all": lambda k: True, "unc_pair": lambda k: "refcocog" not in k, "refcocog_unc": lambda k: "refcocog" in k}
    res["pooled"] = {}
    for gname, sel in groups.items():
        for key in ("vcs_minus_softmax", "vcs_minus_js", "gap_vcs_minus_softmax", "gap_vcs_minus_js"):
            v = [c[key]["mean"] for k, c in res["cells"].items() if sel(k) and key in c]
            if len(v) >= 2:
                res["pooled"][f"{gname}/{key}"] = {**tint(v), "n_cells": len(v),
                                                   "cells_ci_above_0": sum(c[key]["ci95"][0] > 0 for k, c in res["cells"].items() if sel(k) and key in c and c[key].get("ci95")),
                                                   "cells_ci_below_0": sum(c[key]["ci95"][1] < 0 for k, c in res["cells"].items() if sel(k) and key in c and c[key].get("ci95"))}
    json.dump(res, open(OUT, "w"), indent=1)
    p = lambda t: f"{100 * t['mean']:.2f}" + (f"±{100 * t['sd']:.2f}" if t.get("sd") is not None else "")
    q = lambda t: f"{100 * t['mean']:+.2f}" + (f" [{100 * t['ci95'][0]:+.2f},{100 * t['ci95'][1]:+.2f}]" if t.get("ci95") else "")
    for k, c in res["cells"].items():
        print(f"{k:34s} zs {100 * c['zeroshot_kept']:.2f} ({c['n_kept_images']} img) | " + " | ".join(
            f"{o} {p(c[o]['transfer'])}" + (f" (in-dom {p(c[o]['in_domain_same_images'])})" if "in_domain_same_images" in c[o] else "") for o in OBJS)
              + f"  VCS−sm {q(c['vcs_minus_softmax'])}")
    for k, v in res["pooled"].items():
        print(f"pooled {k:32s} {q(v)} (n {v['n_cells']}; CI>0 {v['cells_ci_above_0']}, CI<0 {v['cells_ci_below_0']})")
    bad = [g for g in res["gate_X"] if not g["pass"]]
    print(f"gate X: {len(res['gate_X']) - len(bad)} / {len(res['gate_X'])} pass" + (f"; FAIL {bad}" if bad else ""))
    if res["missing"]:
        print("missing:", res["missing"][:10], "..." if len(res["missing"]) > 10 else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
