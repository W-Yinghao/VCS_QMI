"""R4-VL2 readings (reports/R4/R4_VL2_NU_PROTOCOL_FROZEN_20261011.yaml): detector-score reference ν for VCS / matched JS.

Inputs: outputs/R4_VL2/<ds>_siglip2_b16_<vcs|js>_nuscore_s<seed>.json (new runs); outputs/R4_VL1/<ds>_siglip2_b16_<obj>_s<seed>.json (uniform-ν
controls, no new compute); the R4-VL1 checkpoints' DEV proposal cosines (outputs/R4_VL0/scores_r4vl1/, r4_vl0_scores.py --source r4vl1) for the
no-training score-fused controls, ranked by log s_r + 2 α cos with each checkpoint's own CAL-selected critic slope α.
Per dataset, mean ± sd over seeds 0–2 of DEV all-proposal image-macro Acc@IoU 0.5 (uncovered queries count as errors):
  primary     VCS(ν) with the corrected rank log ν + 2 α cos (selected on CAL with the same rank)
  contrasts   paired by seed, 95 % t over 3 seeds: C1 VCS(ν) − softmax; C2 VCS(ν) − VCS(uniform); C3 VCS(ν) − softmax(fused) and
              VCS(ν) − VCS(uniform, fused); C4 VCS(ν) − JS(ν); fusion alone: softmax(fused) − softmax and VCS(uniform, fused) − VCS(uniform)
  secondary   plain-cosine rank of the ν-trained models; Acc@0.75; J_recal on the ν laws; margin; given-box Top-1; selected epoch; top-K (K on CAL)
Wording (fixed): C1 interval containing 0 -> "a detector-score reference closes the gap"; below 0 -> "softmax still leads by x"; above 0 ->
"VCS(ν) leads by x" (C3 vs softmax(fused) worded the same way, as the comparison with equal test-time information).  C2 above 0 while
VCS(ν) − VCS(uniform, fused) contains 0 or lies below 0 -> "the gain is test-time score fusion, not the estimator law"; VCS(ν) − VCS(uniform,
fused) above 0 -> "training with the score reference adds beyond test-time fusion".  Writes reports/R4/R4_VL2_results.json and R4_VL2_tables.md.
    python scripts/r4_vl2_aggregate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vl1_10_aggregate import tint  # noqa: E402
from vl1_22_tableC import iou_xyxy  # noqa: E402
from r4_vl0_scores import meta_path, score_path  # noqa: E402

V1 = Path("/home/infres/yinwang/CS_QMI/outputs/R4_VL1"); V2 = Path("/home/infres/yinwang/CS_QMI/outputs/R4_VL2")
OUT = Path(__file__).resolve().parents[1] / "reports" / "R4"; DS, BB, SEEDS = ("refcocog", "refcocoplus"), "siglip2_b16", (0, 1, 2)


def rank_acc(meta, scores, alpha, fuse):
    """Image-macro Acc@0.5 of argmax_r [2 α cos (+ log s_r when fused)] over all proposals (images without proposals count as errors)."""
    per = []
    for m, s in zip(meta, scores):
        cp = s["cos_prop"].numpy().astype(np.float64); n_p = cp.shape[0]; nq = len(m["target"])
        if n_p == 0:
            per.append(0.0); continue
        S = 2 * alpha * cp + (np.log(np.asarray(m["det_score"], np.float64))[:, None] if fuse else 0.0)
        iou = np.stack([iou_xyxy(m["ref_boxes"][t], m["prop_boxes"]) for t in m["target"]]); pred = S.argmax(0)
        per.append(float((iou[np.arange(nq), pred] >= 0.5).mean()))
    return float(np.mean(per))


def main() -> int:
    res = {"protocol": "reports/R4/R4_VL2_NU_PROTOCOL_FROZEN_20261011.yaml", "cells": {}, "gates": [], "missing": [], "pooled": {}}
    for ds in DS:
        acc, extra, cell = {}, {}, {}
        for o in ("vcs", "js"):                                                              # ν-trained runs (corrected rank)
            for s in SEEDS:
                f = V2 / f"{ds}_{BB}_{o}_nuscore_s{s}.json"
                if not f.exists():
                    res["missing"].append(f.name); continue
                r = json.load(open(f)); x = r["dev_at_selected"]; res["gates"].append({"run": f.stem, **r["gate_step0_given_box"]})
                acc[(f"{o}_nu", s)] = x["acc50"]["image_macro"]; acc[(f"{o}_nu_plain", s)] = x["acc50_plain_cos"]["image_macro"]
                extra[(f"{o}_nu", s)] = {"acc75": x["acc75"]["image_macro"], "J_recal": x["J_recal"], "margin": x["margin_mean"], "given_box": x["given_box_top1_image_macro"],
                                         "epoch": r["selected"]["cal_acc50_epoch"], "topK": r["dev_topK_at_cal_chosen_K"], "K": r["selected"]["topK_chosen_on_cal"]}
        md = torch.load(meta_path(ds, "DEV"), weights_only=False) if meta_path(ds, "DEV").exists() else None
        repro = []
        for o in ("vcs", "js", "softmax"):                                                   # uniform-ν controls: plain and score-fused
            for s in SEEDS:
                f = V1 / f"{ds}_{BB}_{o}_s{s}.json"
                if not f.exists():
                    res["missing"].append(f.name); continue
                r = json.load(open(f)); acc[(o, s)] = r["dev_at_selected"]["acc50"]["image_macro"]; fs = score_path(ds, BB, o, s, "DEV", "r4vl1")
                if md is None or not fs.exists():
                    res["missing"].append(fs.name + " (r4vl1 DEV scores)"); continue
                sc = torch.load(fs, weights_only=False)["scores"]; al = r["selected"]["critic_a_b"][0]
                acc[(f"{o}_fused", s)] = rank_acc(md, sc, al, True); plain = rank_acc(md, sc, al, False)
                repro.append({"run": f.stem, "plain_from_scores": plain, "recorded": acc[(o, s)], "diff": plain - acc[(o, s)], "alpha": al})
        cell["reproduction_plain_rank"] = repro
        for arm in ("vcs_nu", "js_nu", "vcs_nu_plain", "js_nu_plain", "vcs", "js", "softmax", "vcs_fused", "js_fused", "softmax_fused"):
            ss = [s for s in SEEDS if (arm, s) in acc]
            if ss:
                cell[arm] = {"seeds": ss, "acc50": tint([acc[(arm, s)] for s in ss])}
                if (arm, ss[0]) in extra:
                    for k in ("acc75", "J_recal", "margin", "given_box", "topK"):
                        cell[arm][k] = tint([extra[(arm, s)][k] for s in ss if extra[(arm, s)][k] is not None])
                    cell[arm]["selected_epoch"] = [extra[(arm, s)]["epoch"] for s in ss]; cell[arm]["K_cal"] = [extra[(arm, s)]["K"] for s in ss]
        contrasts = {"C1_vcs_nu_minus_softmax": ("vcs_nu", "softmax"), "C2_vcs_nu_minus_vcs_uniform": ("vcs_nu", "vcs"),
                     "C3a_vcs_nu_minus_softmax_fused": ("vcs_nu", "softmax_fused"), "C3b_vcs_nu_minus_vcs_uniform_fused": ("vcs_nu", "vcs_fused"),
                     "C4_vcs_nu_minus_js_nu": ("vcs_nu", "js_nu"), "js_nu_minus_js_uniform": ("js_nu", "js"),
                     "fusion_softmax": ("softmax_fused", "softmax"), "fusion_vcs_uniform": ("vcs_fused", "vcs"), "fusion_js_uniform": ("js_fused", "js")}
        cell["contrasts"] = {}
        for name, (x, y) in contrasts.items():
            ss = [s for s in SEEDS if (x, s) in acc and (y, s) in acc]
            if ss:
                cell["contrasts"][name] = {**tint([acc[(x, s)] - acc[(y, s)] for s in ss]), "seeds": ss}
        res["cells"][ds] = cell
    for name in ("C1_vcs_nu_minus_softmax", "C2_vcs_nu_minus_vcs_uniform", "C3a_vcs_nu_minus_softmax_fused", "C3b_vcs_nu_minus_vcs_uniform_fused"):
        v = [res["cells"][ds]["contrasts"][name]["mean"] for ds in DS if name in res["cells"][ds]["contrasts"] and len(res["cells"][ds]["contrasts"][name]["seeds"]) == 3]
        if len(v) == len(DS):
            res["pooled"][name] = {"mean_over_datasets": float(np.mean(v)), "note": "descriptive (2 cells)"}
    OUT.mkdir(parents=True, exist_ok=True); json.dump(res, open(OUT / "R4_VL2_results.json", "w"), indent=1)
    p = lambda t: "—" if t is None else f"{100 * t['mean']:.2f}" + (f" ± {100 * t['sd']:.2f}" if t.get("sd") is not None else "")
    q = lambda t: f"{100 * t['mean']:+.2f}" + (f" [{100 * t['ci95'][0]:+.2f}, {100 * t['ci95'][1]:+.2f}]" if t.get("ci95") else " (1 seed)")
    L = ["# R4-VL2 — detector-score reference ν for VCS / JS: results", "",
         f"Gates (step-0 given-box DEV within 0.3 of the frozen raw value): {sum(1 for g in res['gates'] if g['pass'])} / {len(res['gates'])} pass.  Missing: {res['missing'] or 'none'}.", ""]
    rows = [("VCS(ν), corrected rank (primary)", "vcs_nu"), ("JS(ν), corrected rank", "js_nu"), ("VCS(ν), plain cosine", "vcs_nu_plain"), ("JS(ν), plain cosine", "js_nu_plain"),
            ("VCS(uniform) — R4-VL1", "vcs"), ("JS(uniform) — R4-VL1", "js"), ("softmax — R4-VL1", "softmax"),
            ("VCS(uniform) + log s (no training)", "vcs_fused"), ("JS(uniform) + log s (no training)", "js_fused"), ("softmax + log s (no training)", "softmax_fused")]
    for ds, cell in res["cells"].items():
        L += [f"## {ds}", "", "| arm | DEV Acc@0.5 |", "|---|---|"] + [f"| {n} | {p(cell[k]['acc50']) if k in cell else '—'} |" for n, k in rows] + [""]
        L += ["| ν-trained secondary | VCS(ν) | JS(ν) |", "|---|---|---|"]
        for n, k in (("Acc@0.75", "acc75"), ("J_recal (ν laws)", "J_recal"), ("margin (×100 cos)", "margin"), ("given-box Top-1", "given_box"), ("top-K (K on CAL)", "topK")):
            L.append(f"| {n} | " + " | ".join(p(cell[a].get(k)) if a in cell else "—" for a in ("vcs_nu", "js_nu")) + " |")
        L.append("| selected epoch / K | " + " | ".join(f"{cell[a]['selected_epoch']} / {cell[a]['K_cal']}" if a in cell else "—" for a in ("vcs_nu", "js_nu")) + " |")
        L += ["", "| contrast (paired by seed) | value |", "|---|---|"] + [f"| {k} | {q(v)} |" for k, v in cell["contrasts"].items()]
        if cell["reproduction_plain_rank"]:
            L += ["", f"Plain-rank reproduction from the score files (|diff| max): {max(abs(r['diff']) for r in cell['reproduction_plain_rank']) * 100:.2f} points.", ""]
    if res["pooled"]:
        L += ["Pooled over the two datasets (descriptive): " + "; ".join(f"{k} {100 * v['mean_over_datasets']:+.2f}" for k, v in res["pooled"].items()), ""]
    (OUT / "R4_VL2_tables.md").write_text("\n".join(L)); print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
