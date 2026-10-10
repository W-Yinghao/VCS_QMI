"""VL1-15 (+ add. 1) frozen readings — model scale on RefCOCOg: per family, Large / So400m − Base for each route (paired by seed; CAL J and DEV
image-macro Top-1, estimator-selected for VCS / JS / RFF, task-selected for softmax), and the probe consistency over all feature sets with a passed
gate (Spearman ρ between the VCS critic's CAL J and its learned DEV Top-1, and with the raw zero-shot Top-1).  FG-CLIP 2 Large failed its gate and is
excluded.  Writes reports/VL1/VL1_15_results.json.
    python scripts/vl1_15_aggregate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vl1_10_aggregate import readouts, tint  # noqa: E402

O = Path("/home/infres/yinwang/CS_QMI/outputs"); OUT = Path(__file__).resolve().parents[1] / "reports" / "VL1" / "VL1_15_results.json"
SETS = {"clip_b16": "VL1_10", "clip_l14_336": "VL1_15_clip_large", "siglip2_b16": "VL1_12_siglip2", "siglip2_l16": "VL1_15_siglip2_large",
        "siglip2_so400m": "VL1_15_siglip2_so400m", "fgclip1_base": "VL1_12_fgclip1", "fgclip1_large": "VL1_15_fgclip1_large", "fgclip2_base": "VL1_12",
        "fgclip2_so400m": "VL1_15_fgclip2_so400m", "reclip_features": "VL1_13"}
LADDERS = {"clip": ["clip_b16", "clip_l14_336"], "siglip2": ["siglip2_b16", "siglip2_l16", "siglip2_so400m"], "fgclip1": ["fgclip1_base", "fgclip1_large"],
           "fgclip2": ["fgclip2_base", "fgclip2_so400m"]}
SEEDS = (0, 1, 2)


def rows(d, route):
    out = []
    for s in SEEDS:
        f = O / d / f"{route}_Nall_s{s}.json"
        if not f.exists():
            return None
        out.append(readouts(json.load(open(f)), route))
    return out


def main() -> int:
    res = {"sets": {}, "ladders": {}, "probe": {}, "missing": []}
    for name, d in SETS.items():
        cell = {}
        if (O / d / "raw.json").exists():
            cell["raw"] = json.load(open(O / d / "raw.json"))["dev"]["top1_image_macro"]
        for route in ("vcs", "js", "softmax", "rff"):
            r = rows(d, route)
            if r is None:
                res["missing"].append(f"{name}/{route}"); continue
            k = "task_dev" if route == "softmax" else "est_dev"
            cell[route] = {"top1": tint([x[k]["top1_image_macro"] for x in r])}
            if "cal_J" in r[0]:
                cell[route]["cal_J"] = tint([x["cal_J"] for x in r])
        res["sets"][name] = cell
    for fam, ladder in LADDERS.items():
        base = ladder[0]; res["ladders"][fam] = {}
        for big in ladder[1:]:
            out = {"raw": res["sets"][big].get("raw", float("nan")) - res["sets"][base].get("raw", float("nan"))}
            for route in ("vcs", "js", "softmax", "rff"):
                a, b = rows(SETS[big], route), rows(SETS[base], route)
                if a is None or b is None:
                    continue
                k = "task_dev" if route == "softmax" else "est_dev"
                out[route] = {"top1": tint([x[k]["top1_image_macro"] - y[k]["top1_image_macro"] for x, y in zip(a, b)])}
                if "cal_J" in a[0]:
                    out[route]["cal_J"] = tint([x["cal_J"] - y["cal_J"] for x, y in zip(a, b)])
            res["ladders"][fam][f"{big}_minus_{base}"] = out
    js = [(n, c["vcs"]["cal_J"]["mean"], c["vcs"]["top1"]["mean"], c.get("raw")) for n, c in res["sets"].items() if "vcs" in c and "cal_J" in c["vcs"]]
    if len(js) >= 3:
        res["probe"] = {"n_sets": len(js), "spearman_J_vs_learned": float(stats.spearmanr([x[1] for x in js], [x[2] for x in js]).statistic),
                        "spearman_J_vs_raw": float(stats.spearmanr([x[1] for x in js], [x[3] for x in js]).statistic),
                        "order_by_J": [x[0] for x in sorted(js, key=lambda z: -z[1])], "order_by_learned": [x[0] for x in sorted(js, key=lambda z: -z[2])],
                        "order_by_raw": [x[0] for x in sorted(js, key=lambda z: -z[3])]}
    json.dump(res, open(OUT, "w"), indent=1)
    p = lambda t: f"{100 * t['mean']:+.2f} [{100 * t['ci95'][0]:+.2f},{100 * t['ci95'][1]:+.2f}]" if t.get("ci95") else f"{100 * t['mean']:+.2f}"
    for n, c in res["sets"].items():
        print(f"{n:16s} raw {100 * c.get('raw', float('nan')):.2f} | VCS top1 {100 * c['vcs']['top1']['mean']:.2f} J {100 * c['vcs']['cal_J']['mean']:.2f}" if "vcs" in c else n)
    for fam, v in res["ladders"].items():
        for k, out in v.items():
            print(f"{k}: raw {100 * out['raw']:+.2f} | " + " | ".join(f"{r} top1 {p(out[r]['top1'])}" + (f" J {p(out[r]['cal_J'])}" if "cal_J" in out[r] else "") for r in ("vcs", "js", "softmax", "rff") if r in out))
    if res["probe"]:
        print(f"probe: rho(J, learned) {res['probe']['spearman_J_vs_learned']:.2f}  rho(J, raw) {res['probe']['spearman_J_vs_raw']:.2f}\n  J order {res['probe']['order_by_J']}\n  learned {res['probe']['order_by_learned']}")
    if res["missing"]:
        print("missing:", res["missing"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
