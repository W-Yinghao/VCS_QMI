"""VL1-14 critic screen — frozen reading (reports/VL1/VL1_14_CRITIC_SCREEN_FROZEN_20261009.md).  Per feature set (CLIP, SigLIP 2) and cell C1-C4
vs C0 (the frozen F2r fits of VL1-10 / VL1-12 add. 2), VCS, N all, paired by seed: CAL common J (selection criterion and primary) and DEV
image-macro Top-1 of the CAL-selected checkpoint (secondary), 95 % t intervals; selected lr / update per cell.  Decision: "critic search needed" if
on either feature set the best cell beats C0 on CAL J by >= 0.50 (x100) with the interval excluding 0; winner = largest CAL J, ties within 0.10 →
fewer parameters.  Writes reports/VL1/VL1_14_results.json; exits 1 if a cell is incomplete (no partial reading)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vl1_10_aggregate import tint  # noqa: E402

O = Path("/home/infres/yinwang/CS_QMI/outputs"); OUT = Path(__file__).resolve().parents[1] / "reports" / "VL1" / "VL1_14_results.json"
FEATS = {"clip": ("VL1_10", "VL1_14_clip"), "siglip2": ("VL1_12_siglip2", "VL1_14_siglip2")}
CELLS = {"C1": "F2r, lr past the edge, 10k updates", "C2": "residual MLP 512x3, 10k", "C3": "bilinear rank 64, 10k", "C4": "affine a*cos+b, 10k"}
NPARAM_ORDER = ["C4", "C3", "C1", "C2"]          # fewer parameters first (tie-break)
SEEDS = (0, 1, 2)


def load(d, tag):
    f = O / d / f"{tag}.json"
    return json.load(open(f)) if f.exists() else None


def main() -> int:
    res = {"cells": CELLS, "per_feature": {}, "decision": {}, "missing": []}
    for feat, (d0, d1) in FEATS.items():
        c0 = [load(d0, f"vcs_Nall_s{s}") for s in SEEDS]; out = {}
        for cell in CELLS:
            rows = [load(d1, f"vcs_Nall_s{s}_{cell}") for s in SEEDS]
            if not all(rows):
                res["missing"].append(f"{feat}/{cell}"); continue
            J = [r["estimator_selected"]["cal"]["J_common"] for r in rows]; J0 = [r["estimator_selected"]["cal"]["J_common"] for r in c0]
            T = [r["estimator_selected"]["dev"]["top1_image_macro"] for r in rows]; T0 = [r["estimator_selected"]["dev"]["top1_image_macro"] for r in c0]
            out[cell] = {"cal_J": tint(J), "dev_top1": tint(T), "dJ_vs_C0": tint([a - b for a, b in zip(J, J0)]), "dTop1_vs_C0": tint([a - b for a, b in zip(T, T0)]),
                         "selected": [(r["estimator_selected"]["lr"], r["estimator_selected"]["update"]) for r in rows]}
        out["C0"] = {"cal_J": tint([r["estimator_selected"]["cal"]["J_common"] for r in c0]), "dev_top1": tint([r["estimator_selected"]["dev"]["top1_image_macro"] for r in c0])}
        res["per_feature"][feat] = out
    if res["missing"]:
        print("missing:", res["missing"]); json.dump(res, open(OUT, "w"), indent=1); return 1
    for feat, out in res["per_feature"].items():
        passing = {c: v for c, v in out.items() if c != "C0" and v["dJ_vs_C0"]["mean"] >= 0.005 and v["dJ_vs_C0"]["ci95"][0] > 0}
        best = max((c for c in out if c != "C0"), key=lambda c: out[c]["cal_J"]["mean"])
        ties = [c for c in out if c != "C0" and out[best]["cal_J"]["mean"] - out[c]["cal_J"]["mean"] <= 0.001]
        winner = min(ties, key=NPARAM_ORDER.index)
        res["decision"][feat] = {"cells_passing_rule": sorted(passing), "best_by_cal_J": best, "winner_after_tiebreak": winner}
    res["decision"]["search_needed"] = any(v["cells_passing_rule"] for k, v in res["decision"].items() if isinstance(v, dict))
    json.dump(res, open(OUT, "w"), indent=1)
    p = lambda t: f"{100 * t['mean']:.2f}" + (f" [{100 * t['ci95'][0]:+.2f},{100 * t['ci95'][1]:+.2f}]" if t.get("ci95") else "")
    for feat, out in res["per_feature"].items():
        print(f"== {feat}: C0 J {p(out['C0']['cal_J'])} Top-1 {p(out['C0']['dev_top1'])}")
        for c in CELLS:
            v = out[c]; print(f"  {c} J {p(v['cal_J'])} ΔJ {p(v['dJ_vs_C0'])} | Top-1 {p(v['dev_top1'])} ΔTop-1 {p(v['dTop1_vs_C0'])} | selected {v['selected']}")
        print("  decision:", res["decision"][feat])
    print("critic search needed:", res["decision"]["search_needed"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
