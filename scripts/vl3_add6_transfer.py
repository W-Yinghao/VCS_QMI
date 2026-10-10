"""VL3 addendum 6 (reports/VL3/VL3_ADDENDUM6_TRANSFER_FROZEN_20261010.md): cross-dataset transfer of the VL3 checkpoints.  Every CAL-Top-1 checkpoint
fine-tuned on a source dataset S is evaluated (given boxes, VL3 evaluation code) on the DEV split of each other target dataset T, restricted to
target DEV images that are not in S's FIT or CAL images (no image the source model trained or was selected on).  Also the zero-shot backbone on the
same restricted set and on the full target DEV (gate X: full-DEV zero-shot within 0.3 point of the frozen raw value of T).  Writes
outputs/VL3_transfer/<S>_to_<T>_<backbone>_<objective>_s<seed>.json (image-macro, per query hits, image ids) and ..._zeroshot.json.
    python scripts/vl3_add6_transfer.py --source refcocog --backbone clip_b16
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vl3_aggregate import RAW  # noqa: E402
from vl3_finetune import CKPT, CLIP_ARCHS, SIGLIP_REPOS, Backbone, Scenes, collate, collect_cos, metrics, records  # noqa: E402
from vl3_add5_strata import per_query_hits  # noqa: E402

V3 = Path("/home/infres/yinwang/CS_QMI/outputs/VL3"); OUT = Path("/home/infres/yinwang/CS_QMI/outputs/VL3_transfer")
DATASETS = ("refcocog", "refcoco", "refcoco+")


def roles_of(dataset):
    os.environ["VL_DATASET"] = dataset
    from vcs_vl import refcocog as RG
    RG = importlib.reload(RG); scenes, _ = RG.load_scenes(); return scenes, RG.dev_roles(scenes)


def evaluate(bb, recs, dev, workers):
    dl = torch.utils.data.DataLoader(Scenes(recs, bb), batch_size=64, shuffle=False, num_workers=workers, collate_fn=collate, pin_memory=True)
    d = collect_cos(bb, dl, dev); m = metrics(d, 1.0, 0.0); h = per_query_hits(d); assert len(h) == m["n_queries"]
    return {"top1_image_macro": m["top1_image_macro"], "top1_query": m["top1_query"], "n_images": m["n_images"], "n_queries": m["n_queries"], "hits": h}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--source", required=True, choices=DATASETS); ap.add_argument("--backbone", required=True, choices=list(CLIP_ARCHS) + list(SIGLIP_REPOS))
    ap.add_argument("--objectives", nargs="+", default=["vcs", "js", "softmax"]); ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=8); a = ap.parse_args()
    dev = torch.device("cuda"); OUT.mkdir(parents=True, exist_ok=True); S = a.source.replace("+", "plus")
    src_scenes, src_role = roles_of(a.source); seen = {s.image_id for s in src_scenes if src_role[s.image_id] in ("FIT", "CAL")}
    targets = {}
    for t in DATASETS:
        if t == a.source:
            continue
        sc, ro = roles_of(t); full = records(sc, ro, "DEV"); kept = [r for r in full if r["image_id"] not in seen]
        targets[t.replace("+", "plus")] = (full, kept); print(f"{S} -> {t}: target DEV images {len(full)}, kept {len(kept)} (excluded {len(full) - len(kept)})", flush=True)
    bb = Backbone(a.backbone).to(dev); init = {k: v.detach().clone() for k, v in bb.state_dict().items()}
    for T, (full, kept) in targets.items():
        out = OUT / f"{S}_to_{T}_{a.backbone}_zeroshot.json"
        if not out.exists():
            bb.load_state_dict(init); zf = evaluate(bb, full, dev, a.workers); zk = evaluate(bb, kept, dev, a.workers)
            ok = abs(zf["top1_image_macro"] - RAW[(T, a.backbone)]) <= 0.003
            print(f"[{S}->{T} {a.backbone} zero-shot] full DEV {zf['top1_image_macro']:.4f} vs frozen raw {RAW[(T, a.backbone)]:.4f} -> gate X {'pass' if ok else 'FAIL'}; kept {zk['top1_image_macro']:.4f}", flush=True)
            zk["image_ids"] = [r["image_id"] for r in kept]; zf.pop("hits")
            json.dump({"gate_X": {"full_dev_top1": zf["top1_image_macro"], "frozen_raw": RAW[(T, a.backbone)], "pass": bool(ok)}, "full": zf, "kept": zk}, open(out, "w"))
    for o in a.objectives:
        for s in a.seeds:
            tag = f"{S}_{a.backbone}_{o}_s{s}"; outs = {T: OUT / f"{S}_to_{T}_{a.backbone}_{o}_s{s}.json" for T in targets}
            if all(p.exists() for p in outs.values()):
                print("exists", tag, flush=True); continue
            ck = torch.load(CKPT / f"{tag}_calTop1.pt", map_location="cpu", weights_only=False); bb.load_state_dict({k: v.float() for k, v in ck["backbone"].items()})
            for T, (full, kept) in targets.items():
                if outs[T].exists():
                    continue
                r = evaluate(bb, kept, dev, a.workers); r["image_ids"] = [x["image_id"] for x in kept]
                print(f"[{S}->{T} {a.backbone} {o} s{s}] kept DEV top1 {r['top1_image_macro']:.4f} ({r['n_images']} images)", flush=True)
                json.dump({"source": S, "target": T, "backbone": a.backbone, "objective": o, "seed": s, **r}, open(outs[T], "w"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
