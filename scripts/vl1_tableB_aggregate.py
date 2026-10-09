"""VL1 Table B on DEV (descriptive placement; no task training in the public rows): ReCLIP (official ensemble + relation parsing, and IPS-only)
read on the same eligible DEV images and the same candidate set as Table A's primary metric (the image's referred objects; 1 356 images, 6 442
queries), plus the raw zero-shot rows from VL1-10 (CLIP ViT-B/16) and VL1-12 (FG-CLIP 2 Base).  Metrics as Table A: Top-1 per query and
image-macro (per image mean over its queries, then mean over images).  ReCLIP is scored two ways: exact target (pred index ∈ gold_index, the
Table A definition) and IoU ≥ 0.5 with the target box (ReCLIP's own `correct`; overlapping candidate boxes can both count).
Secondary: ReCLIP on all COCO objects of the DEV images (its official candidate set incl. crowd boxes) restricted to the eligible images, beside
Table A's all-objects metric (non-crowd, non-degenerate distractors) — candidate sets differ slightly; reported, not compared.
Writes reports/VL1/VL1_tableB_dev.json.
    python scripts/vl1_tableB_aggregate.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vcs_vl import refcocog as RG  # noqa: E402

O = Path("/home/infres/yinwang/CS_QMI/outputs"); R = O / "VL1_reclip"; OUT = REPO / "reports" / "VL1" / "VL1_tableB_dev.json"


def iid(fn: str) -> int:
    return int(re.search(r"COCO_train2014_(\d+)", fn).group(1))


def score(rows, keep=None) -> dict:
    per = defaultdict(lambda: [[], []])
    for r in rows:
        i = iid(r["file_name"])
        if keep is not None and i not in keep:
            continue
        per[i][0].append(int(r["pred"] in r["gold_index"])); per[i][1].append(int(r["correct"]))
    ex = [x for v in per.values() for x in v[0]]; io = [x for v in per.values() for x in v[1]]
    return {"n_images": len(per), "n_queries": len(ex), "top1_query_exact": float(np.mean(ex)), "top1_query_iou": float(np.mean(io)),
            "top1_image_macro_exact": float(np.mean([np.mean(v[0]) for v in per.values()])),
            "top1_image_macro_iou": float(np.mean([np.mean(v[1]) for v in per.values()]))}


def main() -> int:
    scenes, _ = RG.load_scenes(resolve_paths=False); role = RG.dev_roles(scenes)
    elig = {s.image_id for s in scenes if role[s.image_id] == "DEV" and len(s.referred) >= 2}
    res = {"eligible_dev_images": len(elig), "reclip": {}, "zero_shot_features": {}}
    for variant, tag in (("ensemble + relations (official)", "parse"), ("IPS only", "baseline")):
        rows = [json.loads(l) for l in open(R / f"reclip_{tag}_dev_referred.json") if l.strip()]
        allr = [json.loads(l) for l in open(R / f"reclip_{tag}_dev.json") if l.strip()]
        res["reclip"][tag] = {"variant": variant, "referred_candidates": score(rows), "all_objects_eligible_images": score(allr, keep=elig)}
    for name, d in (("CLIP ViT-B/16 (crop, cosine)", "VL1_10"), ("FG-CLIP 2 Base (official region API, cosine)", "VL1_12")):
        r = json.load(open(O / d / "raw.json"))["dev"]
        res["zero_shot_features"][name] = {k: r[k] for k in ("top1_query", "top1_image_macro", "top1_all_objects", "n_images", "n_queries")}
    json.dump(res, open(OUT, "w"), indent=1)
    for tag, v in res["reclip"].items():
        a, b = v["referred_candidates"], v["all_objects_eligible_images"]
        print(f"ReCLIP {tag:8s} referred: n {a['n_queries']} query exact {100 * a['top1_query_exact']:.2f} IoU {100 * a['top1_query_iou']:.2f} | "
              f"macro exact {100 * a['top1_image_macro_exact']:.2f} IoU {100 * a['top1_image_macro_iou']:.2f} || all-objects (eligible) query exact "
              f"{100 * b['top1_query_exact']:.2f} IoU {100 * b['top1_query_iou']:.2f} n {b['n_queries']}")
    for k, v in res["zero_shot_features"].items():
        print(f"{k}: query {100 * v['top1_query']:.2f} macro {100 * v['top1_image_macro']:.2f} all-obj {100 * v['top1_all_objects']:.2f} n {v['n_queries']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
