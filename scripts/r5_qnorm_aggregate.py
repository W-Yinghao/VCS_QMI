"""R5-QN readings (reports/R5/R5_QNORM_PROTOCOL_FROZEN_20261011.yaml), fixed before any result: per dataset, mean ± sd over seeds 0–2 of DEV
all-proposal image-macro Acc@0.5 at the CAL-selected epoch for VCS and JS with the query-normalized critic (VQ, JQ) and the reference arms
(set softmax E and VCS / JS from R4-VL1; distribution softmax D from R5-30); primary VQ − E and VQ − VCS; secondary JQ − E, JQ − JS, VQ − JQ,
VQ − D; each paired by seed (95 % t, df 2); secondary metrics; every per-seed value; gates; missing runs listed, never filled.  Wording: an interval
containing 0 is reported as containing 0 (no equivalence margin was pre-specified); datasets read separately (pooled mean descriptive).
Writes reports/R5/results_qnorm.json, reports/R5/qnorm_comparison.md, reports/R5/qnorm_run_manifest.json.
    python scripts/r5_qnorm_aggregate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vl1_10_aggregate import tint  # noqa: E402

V1 = Path("/home/infres/yinwang/CS_QMI/outputs/R4_VL1"); V5 = Path("/home/infres/yinwang/CS_QMI/outputs/R5_DIST"); VQ = Path("/home/infres/yinwang/CS_QMI/outputs/R5_QNORM")
CK = Path("/projects/EEG-foundation-model/yinghao/models/r5_qnorm")
OUT = Path(__file__).resolve().parents[1] / "reports" / "R5"; DS, BB, SEEDS = ("refcocog", "refcocoplus"), "siglip2_b16", (0, 1, 2)
ARMS = {"VQ": (VQ, "vcs_qnorm"), "JQ": (VQ, "js_qnorm"), "E": (V1, "softmax"), "VCS": (V1, "vcs"), "JS": (V1, "js"), "D": (V5, "distribution_softmax")}
CONTRASTS = {"VQ_minus_E (primary)": ("VQ", "E"), "VQ_minus_VCS (primary)": ("VQ", "VCS"), "JQ_minus_E": ("JQ", "E"), "JQ_minus_JS": ("JQ", "JS"),
             "VQ_minus_JQ": ("VQ", "JQ"), "VQ_minus_D": ("VQ", "D")}
NEW = ("VQ", "JQ")


def reading(t):
    if t.get("ci95") is None:
        return "fewer than 2 seeds"
    lo, hi = t["ci95"]
    return "interval contains 0" if lo <= 0 <= hi else ("above 0" if lo > 0 else "below 0")


def main() -> int:
    res = {"protocol": "reports/R5/R5_QNORM_PROTOCOL_FROZEN_20261011.yaml", "cells": {}, "missing": [], "gates": []}; manifest = []
    for ds in DS:
        runs = {}
        for arm, (d, o) in ARMS.items():
            for s in SEEDS:
                f = d / f"{ds}_{BB}_{o}_s{s}.json"
                if not f.exists():
                    res["missing"].append(f.name); continue
                r = json.load(open(f)); runs[(arm, s)] = r
                if arm in NEW:
                    res["gates"].append({"run": f.stem, **r["gate_step0_given_box"]})
                    manifest.append({"dataset": ds, "seed": s, "objective": o, "loss_impl": r.get("loss_impl"), "selected_epoch": r["selected"]["cal_acc50_epoch"],
                                     "critic_a_b": r["selected"]["critic_a_b"], "gpu": r.get("gpu"), "chunk_crops_effective": r.get("chunk_crops_effective"),
                                     "seconds": r.get("seconds"), "checkpoint": str(CK / f"{f.stem}_calAcc.pt"), "gate": r["gate_step0_given_box"]})
        cell = {}
        for arm in ARMS:
            ss = [s for s in SEEDS if (arm, s) in runs]
            if not ss:
                continue
            sel = {s: runs[(arm, s)]["dev_at_selected"] for s in ss}
            cell[arm] = {"seeds": ss, "per_seed_acc50": {s: sel[s]["acc50"]["image_macro"] for s in ss}, "acc50": tint([sel[s]["acc50"]["image_macro"] for s in ss]),
                         "acc75": tint([sel[s]["acc75"]["image_macro"] for s in ss]), "query_acc50": tint([sel[s]["acc50"]["query"] for s in ss]),
                         "given_box_top1": tint([sel[s]["given_box_top1_image_macro"] for s in ss]), "J_recal": tint([sel[s]["J_recal"] for s in ss]),
                         "selected_epoch": [runs[(arm, s)]["selected"]["cal_acc50_epoch"] for s in ss]}
        for name, (x, y) in CONTRASTS.items():
            ss = [s for s in SEEDS if (x, s) in runs and (y, s) in runs]
            if ss:
                t = tint([runs[(x, s)]["dev_at_selected"]["acc50"]["image_macro"] - runs[(y, s)]["dev_at_selected"]["acc50"]["image_macro"] for s in ss])
                cell[name] = {**t, "seeds": ss, "reading": reading(t)}
        res["cells"][ds] = cell
    OUT.mkdir(parents=True, exist_ok=True); json.dump(res, open(OUT / "results_qnorm.json", "w"), indent=1)
    json.dump(manifest, open(OUT / "qnorm_run_manifest.json", "w"), indent=1)
    p = lambda t: f"{100 * t['mean']:.2f}" + (f" ± {100 * t['sd']:.2f}" if t.get("sd") is not None else "")
    q = lambda t: f"{100 * t['mean']:+.2f}" + (f" [{100 * t['ci95'][0]:+.2f}, {100 * t['ci95'][1]:+.2f}]" if t.get("ci95") else "")
    L = ["# R5-QN — VCS / JS with the query-normalized critic vs the R4-VL1 / R5-30 objectives (DEV all-proposal image-macro Acc@0.5; exploratory)", "",
         f"Gates: {sum(1 for g in res['gates'] if g['pass'])} / {len(res['gates'])} pass.  Missing: {res['missing'] or 'none'}.", ""]
    for ds, cell in res["cells"].items():
        L += [f"## {ds}", "", "| arm | seed 0 | seed 1 | seed 2 | mean ± sd | Acc@0.75 | query | given-box | J_recal | epochs |", "|---|---|---|---|---|---|---|---|---|---|"]
        for arm in ARMS:
            if arm in cell:
                c = cell[arm]; ps = [f"{100 * c['per_seed_acc50'][s]:.2f}" if s in c["per_seed_acc50"] else "—" for s in SEEDS]
                L.append(f"| {arm} | " + " | ".join(ps) + f" | {p(c['acc50'])} | {p(c['acc75'])} | {p(c['query_acc50'])} | {p(c['given_box_top1'])} | {p(c['J_recal'])} | {c['selected_epoch']} |")
        L += ["", "| contrast (paired by seed) | value | reading |", "|---|---|---|"] + [f"| {k} | {q(cell[k])} | {cell[k]['reading']} |" for k in CONTRASTS if k in cell] + [""]
    (OUT / "qnorm_comparison.md").write_text("\n".join(L)); print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
