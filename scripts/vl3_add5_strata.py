"""VL3 addendum 5 (reports/VL3/VL3_ADDENDUM5_STRATA_FROZEN_20261010.md): per-query DEV hits of every VL3 CAL-Top-1 checkpoint (given boxes), for the
pre-registered strata (expression length, location words, number of referred candidates).  Re-evaluates the reloaded fp16 checkpoint on DEV with
the VL3 evaluation code; gate S (per checkpoint): image-macro DEV Top-1 within 0.3 point of the run's value.  Writes
outputs/VL3_strata/<dataset>_<backbone>_<objective>_s<seed>.json (hits in records order) and outputs/VL3_strata/<dataset>_meta.json (per query:
image id, text, number of referred objects in the image).  Dataset from VL_DATASET.
    python scripts/vl3_add5_strata.py --backbone clip_b16 [--objectives vcs js softmax] [--seeds 0 1 2]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vl3_finetune import CKPT, CLIP_ARCHS, SIGLIP_REPOS, RG, Backbone, Scenes, collate, collect_cos, metrics, records  # noqa: E402

V3 = Path("/home/infres/yinwang/CS_QMI/outputs/VL3"); OUT = Path("/home/infres/yinwang/CS_QMI/outputs/VL3_strata")


def per_query_hits(d):
    """Hits per query, flattened image by image in records order (texts of an image in their records order)."""
    rmask = d["A"].sum(2) > 0; wmask = d["A"].sum(1) > 0
    pred = d["C"].masked_fill(~rmask[:, :, None], -torch.inf).argmax(1); hit = (pred == d["A"].argmax(1)) & wmask
    return [int(h) for i in range(hit.shape[0]) for h, w in zip(hit[i].tolist(), wmask[i].tolist()) if w]


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--backbone", required=True, choices=list(CLIP_ARCHS) + list(SIGLIP_REPOS))
    ap.add_argument("--objectives", nargs="+", default=["vcs", "js", "softmax"]); ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=8); a = ap.parse_args()
    dev = torch.device("cuda"); ds = RG.DATASET.replace("+", "plus"); OUT.mkdir(parents=True, exist_ok=True)
    scenes, _ = RG.load_scenes(); role = RG.dev_roles(scenes); dv = records(scenes, role, "DEV")
    meta = OUT / f"{ds}_meta.json"
    if not meta.exists():
        q = [{"image_id": r["image_id"], "text": t, "n_referred": len(r["boxes"])} for r in dv for t in r["texts"]]
        json.dump({"dataset": ds, "n_queries": len(q), "queries": q}, open(meta, "w"))
    bb = Backbone(a.backbone).to(dev)
    dl = torch.utils.data.DataLoader(Scenes(dv, bb), batch_size=64, shuffle=False, num_workers=a.workers, collate_fn=collate, pin_memory=True)
    for o in a.objectives:
        for s in a.seeds:
            tag = f"{ds}_{a.backbone}_{o}_s{s}"; out = OUT / f"{tag}.json"
            if out.exists():
                print("exists", tag, flush=True); continue
            run = json.load(open(V3 / f"{tag}.json")); ck = torch.load(CKPT / f"{tag}_calTop1.pt", map_location="cpu", weights_only=False)
            bb.load_state_dict({k: v.float() for k, v in ck["backbone"].items()}); d = collect_cos(bb, dl, dev)
            m = metrics(d, 1.0, 0.0); ref = run["dev_at_cal_top1"]["own"]["dev"]["top1_image_macro"]; ok = abs(m["top1_image_macro"] - ref) <= 0.003
            hits = per_query_hits(d); assert len(hits) == m["n_queries"]
            print(f"[{tag}] gate S DEV top1 {m['top1_image_macro']:.4f} vs run {ref:.4f} -> {'pass' if ok else 'FAIL'}; {len(hits)} queries", flush=True)
            json.dump({"tag": tag, "gate_S": {"dev_top1": m["top1_image_macro"], "run_dev_top1": ref, "pass": bool(ok)}, "hits": hits if ok else None}, open(out, "w"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
