"""VL1-22 Table C (reports/VL1/VL1_22_TABLE_C_FROZEN_20261010.md): the detection setting.  The clean VL1-21 detector's DEV proposals (top-100,
score >= 0.05, per-class NMS, sorted by score) are ranked by the VL3 fine-tuned encoders (CAL-Top-1 checkpoints); the top-1 proposal per referring
expression is correct when IoU >= 0.5 with the referred box.  DEV images and expressions are Table A's / VL3's (images with >= 2 referred objects).
The critic is a * cos + b with a = softplus(alpha) > 0, so ranking by cosine is ranking by the critic.
Gate T1 (per checkpoint): the reloaded fp16 weights reproduce the run's given-box DEV image-macro Top-1 within 0.3 point.
Writes outputs/VL1_22/<dataset>_<backbone>_<objective>_s<seed><suffix>.json (skips existing).  Dataset from VL_DATASET (as VL3).
    python scripts/vl1_22_tableC.py --backbone clip_b16 [--objectives vcs js softmax] [--seeds 0 1 2] [--suffix ""] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vl3_finetune import CKPT, CLIP_ARCHS, SIGLIP_REPOS, RG, Backbone, Scenes, collate, collect_cos, metrics, records  # noqa: E402

V3 = Path("/home/infres/yinwang/CS_QMI/outputs/VL3"); PROP = Path("/home/infres/yinwang/CS_QMI/outputs/VL1_21/proposals")
OUT = Path("/home/infres/yinwang/CS_QMI/outputs/VL1_22"); TOPK, CHUNK = 100, 256


def iou_xyxy(t, B):
    ix = np.clip(np.minimum(t[2], B[:, 2]) - np.maximum(t[0], B[:, 0]), 0, None); iy = np.clip(np.minimum(t[3], B[:, 3]) - np.maximum(t[1], B[:, 1]), 0, None)
    inter = ix * iy; return inter / ((t[2] - t[0]) * (t[3] - t[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - inter)


def prop_records(scenes, role, props):
    """Per eligible DEV image: proposal boxes (clipped xyxy; degenerate dropped), texts, and each text's referred box (xyxy, unclipped)."""
    out = []
    for s in scenes:
        if role[s.image_id] != "DEV" or len(s.referred) < 2 or not s.path:
            continue
        P = props[s.image_id][:TOPK].numpy(); boxes = []
        for x1, y1, x2, y2 in P[:, :4].tolist():
            b = RG.clip_box((x1, y1, x2 - x1, y2 - y1), s.width, s.height)
            if b is not None:
                boxes.append(b)
        texts, targets = [], []
        for o in s.referred:
            for e in o.expressions:
                texts.append(e["raw"]); targets.append([o.box_xywh[0], o.box_xywh[1], o.box_xywh[0] + o.box_xywh[2], o.box_xywh[1] + o.box_xywh[3]])
        out.append({"image_id": s.image_id, "path": s.path, "boxes": boxes, "texts": texts, "targets": np.array(targets, dtype=np.float64)})
    return out


class Props(torch.utils.data.Dataset):
    def __init__(self, recs, bb):
        self.recs, self.bb = recs, bb

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        with Image.open(r["path"]) as im:
            im = im.convert("RGB")
            crops = torch.stack([self.bb.preprocess(im.crop(tuple(b))) for b in r["boxes"]]) if r["boxes"] else None
        return i, crops, self.bb.tokenize(r["texts"])


@torch.no_grad()
def rank_proposals(bb, recs, dev, workers):
    """Per image: hits (top-1 proposal IoU >= 0.5) and the ceiling (any proposal IoU >= 0.5) for each text."""
    bb.eval(); dl = torch.utils.data.DataLoader(Props(recs, bb), batch_size=1, shuffle=False, num_workers=workers, collate_fn=lambda b: b[0])
    hit, ceil = {}, {}
    for i, crops, toks in dl:
        r = recs[i]; n_t = len(r["texts"])
        if crops is None:
            hit[i] = np.zeros(n_t); ceil[i] = np.zeros(n_t); continue
        with torch.autocast("cuda", dtype=torch.bfloat16):
            U = torch.cat([bb.enc_img(crops[k:k + CHUNK].to(dev, non_blocking=True)) for k in range(0, len(crops), CHUNK)])
            T = bb.enc_txt(toks.to(dev, non_blocking=True))
        C = F.normalize(U.float(), dim=-1) @ F.normalize(T.float(), dim=-1).T          # [n_prop, n_text]
        pred = C.argmax(0).cpu().numpy(); B = np.array(r["boxes"], dtype=np.float64)
        ious = np.stack([iou_xyxy(t, B) for t in r["targets"]])                         # [n_text, n_prop]
        hit[i] = (ious[np.arange(n_t), pred] >= 0.5).astype(float); ceil[i] = (ious.max(1) >= 0.5).astype(float)
    bb.train()
    h = [hit[i] for i in range(len(recs))]; c = [ceil[i] for i in range(len(recs))]
    return {"acc_iou50_image_macro": float(np.mean([x.mean() for x in h])), "acc_iou50_query": float(np.concatenate(h).mean()),
            "ceiling_iou50_image_macro": float(np.mean([x.mean() for x in c])), "ceiling_iou50_query": float(np.concatenate(c).mean()),
            "n_images": len(recs), "n_queries": int(sum(len(x) for x in h)), "mean_proposals": float(np.mean([len(r["boxes"]) for r in recs]))}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--backbone", required=True, choices=list(CLIP_ARCHS) + list(SIGLIP_REPOS))
    ap.add_argument("--objectives", nargs="+", default=["vcs", "js", "softmax"]); ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--suffix", default=""); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    dev = torch.device("cuda"); ds = RG.DATASET.replace("+", "plus"); OUT.mkdir(parents=True, exist_ok=True)
    scenes, _ = RG.load_scenes(); role = RG.dev_roles(scenes)
    props = torch.load(PROP / f"{ds}_DEV.pt", map_location="cpu", weights_only=False)
    dv, pr = records(scenes, role, "DEV"), prop_records(scenes, role, props)
    assert [r["image_id"] for r in dv] == [r["image_id"] for r in pr]
    if a.smoke:
        dv, pr = dv[:32], pr[:32]
    bb = Backbone(a.backbone).to(dev)
    dl_dev = torch.utils.data.DataLoader(Scenes(dv, bb), batch_size=64, shuffle=False, num_workers=a.workers, collate_fn=collate, pin_memory=True)
    for o in a.objectives:
        for s in a.seeds:
            tag = f"{ds}_{a.backbone}_{o}_s{s}{a.suffix}"; out = OUT / (tag + ("_smoke" if a.smoke else "") + ".json")
            if out.exists():
                print("exists", out.name, flush=True); continue
            t0 = time.time(); run = json.load(open(V3 / f"{tag}.json")); ck = torch.load(CKPT / f"{tag}_calTop1.pt", map_location="cpu", weights_only=False)
            bb.load_state_dict({k: v.float() for k, v in ck["backbone"].items()})
            given = metrics(collect_cos(bb, dl_dev, dev), 1.0, 0.0)
            ref = run["dev_at_cal_top1"]["own"]["dev"]["top1_image_macro"]
            gate = a.smoke or abs(given["top1_image_macro"] - ref) <= 0.003
            print(f"[{tag}] gate T1 given-box DEV top1 {given['top1_image_macro']:.4f} vs run {ref:.4f} -> {'pass' if gate else 'FAIL'}", flush=True)
            res = {"tag": tag, "epoch": ck["epoch"], "gate_T1": {"given_box_dev_top1": given["top1_image_macro"], "run_dev_top1": ref, "pass": bool(gate)},
                   "given_box_dev": given, "smoke": a.smoke}
            if gate:
                res["detection_dev"] = rank_proposals(bb, pr, dev, a.workers)
                d = res["detection_dev"]
                print(f"[{tag}] Acc@0.5 macro {d['acc_iou50_image_macro']:.4f} query {d['acc_iou50_query']:.4f}; ceiling {d['ceiling_iou50_image_macro']:.4f}; "
                      f"{d['mean_proposals']:.1f} proposals; {time.time() - t0:.0f}s", flush=True)
            json.dump(res, open(out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
