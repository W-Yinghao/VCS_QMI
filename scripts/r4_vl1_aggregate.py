"""R4-VL1 readings (reports/R4/R4_VL_TRAINING_PROTOCOL_FROZEN_20261011.yaml, statistics / claims) over outputs/R4_VL1/<ds>_siglip2_b16_<obj>_s<seed>.json.

Per dataset (RefCOCOg, RefCOCO+) and objective (VCS, matched JS, set-event softmax), mean ± sd over seeds 0–2:
  primary    DEV all-proposal image-macro Acc@IoU 0.5 at the CAL-selected epoch (uncovered queries count as errors)
  contrasts  VCS − softmax and VCS − JS paired by seed (95 % t over 3 seeds); pooled over the 2 datasets descriptively
  adaptation each objective's proposal-trained − given-box-trained (VL3 checkpoint of the same seed on the same proposals = Table C), paired by seed
  arms       zero-shot SigLIP 2 B/16 (step 0); Table C (VL3 checkpoints); detector-score top-K with K ∈ {4, 8, 16, 32} chosen on CAL, for the new
             runs (recorded by the trainer) and for the VL3 checkpoints (from the R4-VL0 DEV scores and the --role CAL scores)
  secondary  Acc@IoU 0.75; query-level accuracy; given-box Top-1 of the new encoders; DEV J_recal on the same p / q; margin; selected epoch
Gate list: step-0 given-box DEV Top-1 within 0.3 of the frozen raw value.  Wording (fixed): VCS − softmax interval containing 0 -> "the objectives
tie on proposals"; below 0 -> "softmax leads by x"; above 0 -> "VCS leads by x"; a gain shared by all objectives is a candidate-support adaptation
result, not a VCS-only advantage.  Writes reports/R4/R4_VL1_results.json and R4_VL1_tables.md.
    python scripts/r4_vl1_aggregate.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vl1_10_aggregate import tint  # noqa: E402
from vl1_22_tableC import iou_xyxy  # noqa: E402
from r4_vl0_scores import meta_path, score_path  # noqa: E402

V1 = Path("/home/infres/yinwang/CS_QMI/outputs/R4_VL1"); T22 = Path("/home/infres/yinwang/CS_QMI/outputs/VL1_22")
OUT = Path(__file__).resolve().parents[1] / "reports" / "R4"; DS, BB, OBJS, SEEDS, KS = ("refcocog", "refcocoplus"), "siglip2_b16", ("vcs", "js", "softmax"), (0, 1, 2), (4, 8, 16, 32)


def topk_acc(meta, scores, K):
    """Image-macro Acc@0.5 of the top-1 among the first K proposals in detector-score order (uncovered queries count as errors)."""
    per = []
    for m, s in zip(meta, scores):
        cp = s["cos_prop"].numpy(); n_p = cp.shape[0]; nq = len(m["target"])
        if n_p == 0:
            per.append(0.0); continue
        k = min(K, n_p); iou = np.stack([iou_xyxy(m["ref_boxes"][t], m["prop_boxes"][:k]) for t in m["target"]])
        pred = cp[:k].argmax(0); per.append(float((iou[np.arange(nq), pred] >= 0.5).mean()))
    return float(np.mean(per))


def vl3_topk(ds, o, s):
    """K chosen on CAL for the VL3 checkpoint, DEV accuracy at that K; None when CAL scores are not there yet."""
    fc, fd = score_path(ds, BB, o, s, "CAL"), score_path(ds, BB, o, s, "DEV")
    if not (fc.exists() and fd.exists() and meta_path(ds, "CAL").exists()):
        return None
    mc, md = torch.load(meta_path(ds, "CAL"), weights_only=False), torch.load(meta_path(ds, "DEV"), weights_only=False)
    sc, sd = torch.load(fc, weights_only=False)["scores"], torch.load(fd, weights_only=False)["scores"]
    cal = {K: topk_acc(mc, sc, K) for K in KS}; K = max(KS, key=lambda k: cal[k])
    return {"K_cal": K, "cal_acc": cal[K], "dev_acc": topk_acc(md, sd, K)}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--datasets", nargs="+", default=list(DS)); ap.add_argument("--suffix", default="", help="R5-X1: '_refcoco' (separate files)")
    a = ap.parse_args()
    res = {"protocol": "reports/R4/R4_VL_TRAINING_PROTOCOL_FROZEN_20261011.yaml" if not a.suffix else "reports/R5/R5_X1_REFCOCO_PROTOCOL_FROZEN_20261011.yaml",
           "cells": {}, "gates": [], "missing": [], "pooled": {}}
    for ds in a.datasets:
        runs, base = {}, {}
        for o in OBJS:
            for s in SEEDS:
                f = V1 / f"{ds}_{BB}_{o}_s{s}.json"
                if not f.exists():
                    res["missing"].append(f.name); continue
                r = json.load(open(f)); runs[(o, s)] = r; g = r["gate_step0_given_box"]; res["gates"].append({"run": f.stem, **g})
                t = T22 / f"{ds}_{BB}_{o}_s{s}.json"
                if t.exists():
                    base[(o, s)] = json.load(open(t))["detection_dev"]["acc_iou50_image_macro"]
        cell = {}
        for o in OBJS:
            ss = [s for s in SEEDS if (o, s) in runs]
            if not ss:
                continue
            sel = [runs[(o, s)]["dev_at_selected"] for s in ss]; z = [runs[(o, s)]["dev_step0_zero_shot"] for s in ss]
            c = {"seeds": ss, "acc50": tint([x["acc50"]["image_macro"] for x in sel]), "acc50_query": tint([x["acc50"]["query"] for x in sel]),
                 "acc75": tint([x["acc75"]["image_macro"] for x in sel]), "given_box_top1": tint([x["given_box_top1_image_macro"] for x in sel]),
                 "J_recal_dev": tint([x["J_recal"] for x in sel]), "margin": tint([x["margin_mean"] for x in sel if x["margin_mean"] is not None]),
                 "selected_epoch": [runs[(o, s)]["selected"]["cal_acc50_epoch"] for s in ss], "coverage": sel[0]["coverage_image_macro"],
                 "zero_shot_acc50": tint([x["acc50"]["image_macro"] for x in z]),
                 "topK_new": {"K_cal": [runs[(o, s)]["selected"]["topK_chosen_on_cal"] for s in ss], "dev_acc": tint([runs[(o, s)]["dev_topK_at_cal_chosen_K"] for s in ss])}}
            sb = [s for s in ss if (o, s) in base]
            if sb:
                c["table_c_acc50"] = tint([base[(o, s)] for s in sb]); c["adaptation_gain"] = tint([runs[(o, s)]["dev_at_selected"]["acc50"]["image_macro"] - base[(o, s)] for s in sb])
            vt = [vl3_topk(ds, o, s) for s in ss]
            if all(v is not None for v in vt):
                c["topK_vl3"] = {"K_cal": [v["K_cal"] for v in vt], "dev_acc": tint([v["dev_acc"] for v in vt])}
            cell[o] = c
        for other in ("softmax", "js"):
            ss = [s for s in SEEDS if ("vcs", s) in runs and (other, s) in runs]
            if ss:
                d = [runs[("vcs", s)]["dev_at_selected"]["acc50"]["image_macro"] - runs[(other, s)]["dev_at_selected"]["acc50"]["image_macro"] for s in ss]
                cell[f"vcs_minus_{other}"] = {**tint(d), "seeds": ss}
        res["cells"][ds] = cell
    for key in ("vcs_minus_softmax", "vcs_minus_js"):
        v = [res["cells"][ds][key]["mean"] for ds in a.datasets if key in res["cells"].get(ds, {}) and len(res["cells"][ds][key]["seeds"]) == 3]
        if len(v) > 1:                                                                      # pooling needs >= 2 datasets
            res["pooled"][key] = {"mean_over_datasets": float(np.mean(v)), "n_datasets": len(v), "note": f"descriptive ({len(v)} cells)"}
    OUT.mkdir(parents=True, exist_ok=True); json.dump(res, open(OUT / f"R4_VL1_results{a.suffix}.json", "w"), indent=1)
    p = lambda t: "—" if t is None else f"{100 * t['mean']:.2f}" + (f" ± {100 * t['sd']:.2f}" if t.get("sd") is not None else "")
    q = lambda t: f"{100 * t['mean']:+.2f}" + (f" [{100 * t['ci95'][0]:+.2f}, {100 * t['ci95'][1]:+.2f}]" if t.get("ci95") else " (1 seed)")
    L = ["# R4-VL1 — proposal-aware full fine-tuning (option A): results", "",
         f"Gates (step-0 given-box DEV within 0.3 of the frozen raw value): {sum(1 for g in res['gates'] if g['pass'])} / {len(res['gates'])} pass.  Missing runs: {res['missing'] or 'none'}.", ""]
    for ds, cell in res["cells"].items():
        L += [f"## {ds}", "", "| quantity | " + " | ".join(OBJS) + " |", "|---|---|---|---|"]
        rows = [("DEV Acc@0.5, proposal-trained (primary)", "acc50"), ("Table C (given-box-trained VL3, same proposals)", "table_c_acc50"), ("adaptation gain (paired)", "adaptation_gain"),
                ("zero-shot (step 0)", "zero_shot_acc50"), ("Acc@0.75", "acc75"), ("query-level Acc@0.5", "acc50_query"), ("given-box Top-1 of the new encoder", "given_box_top1"),
                ("DEV J_recal (same p / q)", "J_recal_dev"), ("margin (×100 cos)", "margin")]
        for name, k in rows:
            L.append(f"| {name} | " + " | ".join((q(cell[o][k]) if k == "adaptation_gain" else p(cell[o].get(k))) if o in cell else "—" for o in OBJS) + " |")
        L.append("| top-K, new encoder (K on CAL) | " + " | ".join(f"{p(cell[o]['topK_new']['dev_acc'])} (K {cell[o]['topK_new']['K_cal']})" if o in cell else "—" for o in OBJS) + " |")
        L.append("| top-K, VL3 checkpoint (K on CAL) | " + " | ".join(f"{p(cell[o]['topK_vl3']['dev_acc'])} (K {cell[o]['topK_vl3']['K_cal']})" if o in cell and "topK_vl3" in cell[o] else "—" for o in OBJS) + " |")
        L.append("| selected epoch | " + " | ".join(str(cell[o]["selected_epoch"]) if o in cell else "—" for o in OBJS) + " |")
        L += ["", " ".join(f"**{k}** {q(cell[k])}" for k in ("vcs_minus_softmax", "vcs_minus_js") if k in cell), ""]
    if res["pooled"]:
        L += [f"Pooled over the {len(a.datasets)} datasets (descriptive): " + "; ".join(f"{k} {100 * v['mean_over_datasets']:+.2f}" for k, v in res["pooled"].items()), ""]
    (OUT / f"R4_VL1_tables{a.suffix}.md").write_text("\n".join(L)); print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
