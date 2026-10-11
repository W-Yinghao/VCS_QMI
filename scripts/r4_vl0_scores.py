"""R4-VL0 scoring pass (reports/R4/R4_VL0_PROTOCOL_20261011.md): full cosine score matrices of existing VL3 checkpoints on the DEV split, over the
clean VL1-21 detector proposals AND the image's annotated referred boxes, for every DEV referring expression.  Read-only evaluation of saved
checkpoints; no fitting.  Ranking by cosine = the VL3 critic's ranking (a = softplus(alpha) > 0).

Per dataset (from VL_DATASET) writes once outputs/R4_VL0/meta_<ds>.pt (per eligible DEV image: proposal boxes clipped xyxy in detector-score order,
detector scores and labels, referred boxes / categories / ann ids, other COCO annotation boxes / categories, per expression: target object index and
sent_id) and per checkpoint outputs/R4_VL0/scores/<ds>_<backbone>_<objective>_s<seed>.pt (per image: cos_prop [n_prop, n_text], cos_ref [n_ref, n_text],
fp32).  Gates printed per checkpoint: G1 given-box DEV image-macro Top-1 from cos_ref = the run's recorded value within 0.3 point; G2 all-proposal
Acc@0.5 image-macro from cos_prop = the VL1-22 Table C value of the same checkpoint (same proposals and crops) within 0.3 point (bf16 batch composition differs; exact difference printed).
    python scripts/r4_vl0_scores.py --backbone siglip2_b16 [--objectives vcs js softmax] [--seeds 0 1 2]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vl1_22_tableC import TOPK, iou_xyxy  # noqa: E402
from vl3_finetune import CKPT, CLIP_ARCHS, SIGLIP_REPOS, RG, Backbone  # noqa: E402

V3 = Path("/home/infres/yinwang/CS_QMI/outputs/VL3"); PROP = Path("/home/infres/yinwang/CS_QMI/outputs/VL1_21/proposals")
R4OUT = Path("/home/infres/yinwang/CS_QMI/outputs/R4_VL1"); R4CK = Path("/projects/EEG-foundation-model/yinghao/models/r4_vl1")
T22 = Path("/home/infres/yinwang/CS_QMI/outputs/VL1_22"); OUT = Path("/home/infres/yinwang/CS_QMI/outputs/R4_VL0"); CHUNK = 256


def score_path(ds: str, backbone: str, objective: str, seed: int, role: str = "DEV", source: str = "vl3") -> Path:
    """One definition of the score-file layout, shared with the analysers: VL3 checkpoints in scores/ (DEV) and scores_<role>/; R4-VL1 checkpoints in
    scores_r4vl1/ (DEV) and scores_r4vl1_<role>/."""
    sub = ("scores" if role == "DEV" else f"scores_{role}") if source == "vl3" else f"scores_{source}" + ("" if role == "DEV" else f"_{role}")
    return OUT / sub / f"{ds}_{backbone}_{objective}_s{seed}.pt"


def meta_path(ds: str, role: str = "DEV") -> Path:
    return OUT / (f"meta_{ds}.pt" if role == "DEV" else f"meta_{ds}_{role}.pt")


def xyxy(b):
    return [b[0], b[1], b[0] + b[2], b[1] + b[3]]


def build_meta(scenes, role, props, want="DEV"):
    """Same eligibility as VL3 / VL1-22 (DEV images with >= 2 referred objects and a path); proposals as VL1-22 (top-100, clipped, < 1 px dropped)."""
    meta = []
    for s in scenes:
        if role[s.image_id] != want or len(s.referred) < 2 or not s.path:
            continue
        P = props[s.image_id][:TOPK].numpy(); boxes, det_score, det_label = [], [], []
        for x1, y1, x2, y2, sc, lb in P.tolist():
            b = RG.clip_box((x1, y1, x2 - x1, y2 - y1), s.width, s.height)
            if b is not None:
                boxes.append(b); det_score.append(sc); det_label.append(int(lb))
        texts, tgt, sent = [], [], []
        for i, o in enumerate(s.referred):
            for e in o.expressions:
                texts.append(e["raw"]); tgt.append(i); sent.append(e["sent_id"])
        meta.append({"image_id": s.image_id, "path": s.path, "width": s.width, "height": s.height,
                     "prop_boxes": np.array(boxes, dtype=np.float64).reshape(-1, 4), "det_score": np.array(det_score), "det_label": np.array(det_label, dtype=np.int64),
                     "ref_boxes_clip": [RG.clip_box(o.box_xywh, s.width, s.height) for o in s.referred], "ref_boxes": np.array([xyxy(o.box_xywh) for o in s.referred], dtype=np.float64),
                     "ref_cat": np.array([o.category_id for o in s.referred]), "ref_ann": np.array([o.ann_id for o in s.referred]),
                     "other_boxes": np.array([xyxy(o.box_xywh) for o in s.distractors], dtype=np.float64).reshape(-1, 4), "other_cat": np.array([o.category_id for o in s.distractors]),
                     "texts": texts, "target": np.array(tgt, dtype=np.int64), "sent_id": np.array(sent, dtype=np.int64)})
    return meta


class Imgs(torch.utils.data.Dataset):
    def __init__(self, meta, bb):
        self.meta, self.bb = meta, bb

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        m = self.meta[i]
        with Image.open(m["path"]) as im:
            im = im.convert("RGB")
            boxes = [tuple(b) for b in m["prop_boxes"].tolist()] + [tuple(b) for b in m["ref_boxes_clip"]]
            crops = torch.stack([self.bb.preprocess(im.crop(b)) for b in boxes])
        return i, crops, self.bb.tokenize(m["texts"])


@torch.no_grad()
def score_all(bb, meta, dev, workers):
    bb.eval(); out = [None] * len(meta)
    dl = torch.utils.data.DataLoader(Imgs(meta, bb), batch_size=1, shuffle=False, num_workers=workers, collate_fn=lambda b: b[0])
    for i, crops, toks in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            U = torch.cat([bb.enc_img(crops[k:k + CHUNK].to(dev, non_blocking=True)) for k in range(0, len(crops), CHUNK)])
            T = bb.enc_txt(toks.to(dev, non_blocking=True))
        C = (F.normalize(U.float(), dim=-1) @ F.normalize(T.float(), dim=-1).T).cpu()
        n_p = len(meta[i]["prop_boxes"]); out[i] = {"cos_prop": C[:n_p].clone(), "cos_ref": C[n_p:].clone()}
    bb.train(); return out


def gates(meta, sc):
    """G1 given-box image-macro Top-1 (argmax over referred boxes); G2 all-proposal Acc@0.5 image-macro (VL1-22 definition)."""
    g1, g2 = [], []
    for m, s in zip(meta, sc):
        pred_ref = s["cos_ref"].argmax(0).numpy(); g1.append(float((pred_ref == m["target"]).mean()))
        if len(m["prop_boxes"]):
            pred = s["cos_prop"].argmax(0).numpy(); iou = np.stack([iou_xyxy(m["ref_boxes"][t], m["prop_boxes"]) for t in m["target"]])
            g2.append(float((iou[np.arange(len(pred)), pred] >= 0.5).mean()))
        else:
            g2.append(0.0)
    return float(np.mean(g1)), float(np.mean(g2))


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--backbone", required=True, choices=list(CLIP_ARCHS) + list(SIGLIP_REPOS))
    ap.add_argument("--objectives", nargs="+", default=["vcs", "js", "softmax"]); ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--role", default="DEV", choices=["DEV", "CAL"], help="CAL: for the R4-VL1 top-K arm (K chosen on CAL); G2 has no CAL reference")
    ap.add_argument("--source", default="vl3", choices=["vl3", "r4vl1", "r4vl2", "zeroshot", "r5dist", "r5qnorm"],
                    help="r4vl1 / r4vl2 / r5dist: proposal-trained checkpoints (R4-VL1, R4-VL2 ν runs, r5 distribution softmax); zeroshot: the pretrained backbone")
    ap.add_argument("--workers", type=int, default=8); a = ap.parse_args()
    dev = torch.device("cuda"); ds = RG.DATASET.replace("+", "plus"); score_path(ds, a.backbone, "vcs", 0, a.role, a.source).parent.mkdir(parents=True, exist_ok=True)
    scenes, _ = RG.load_scenes(); role = RG.dev_roles(scenes)
    mp = meta_path(ds, a.role)
    if mp.exists():
        meta = torch.load(mp, weights_only=False)
    else:
        meta = build_meta(scenes, role, torch.load(PROP / f"{ds}_{a.role}.pt", map_location="cpu", weights_only=False), want=a.role); torch.save(meta, mp)
    print(f"{ds} {a.role}: {len(meta)} images, {sum(len(m['texts']) for m in meta)} expressions, {np.mean([len(m['prop_boxes']) for m in meta]):.1f} proposals / image", flush=True)
    bb = Backbone(a.backbone).to(dev); res = {}
    if a.source == "zeroshot":                                                     # one model: the pretrained weights, no checkpoint
        a.objectives, a.seeds = ["zeroshot"], [0]
    for o in a.objectives:
        for s in a.seeds:
            tag = f"{ds}_{a.backbone}_{o}" + ("_nuscore" if a.source == "r4vl2" else "") + f"_s{s}"; outp = score_path(ds, a.backbone, o, s, a.role, a.source)
            if outp.exists():
                res[tag] = torch.load(outp, weights_only=False)["gates"]; print("exists", tag, flush=True); continue
            if a.source == "zeroshot":
                ck = {"epoch": 0}
            else:
                ckp = {"vl3": CKPT / f"{tag}_calTop1.pt", "r4vl1": R4CK / f"{tag}_calAcc.pt", "r4vl2": R4CK.parent / "r4_vl2" / f"{tag}_calAcc.pt",
                       "r5dist": R4CK.parent / "r5_dist" / f"{tag}_calAcc.pt", "r5qnorm": R4CK.parent / "r5_qnorm" / f"{tag}_calAcc.pt"}[a.source]
                ck = torch.load(ckp, map_location="cpu", weights_only=False); bb.load_state_dict({k: v.float() for k, v in ck["backbone"].items()})
            sc = score_all(bb, meta, dev, a.workers); g1, g2 = gates(meta, sc)
            if a.source == "vl3":
                ref1 = json.load(open(V3 / f"{tag}.json"))["dev_at_cal_top1"]["own"]["dev" if a.role == "DEV" else "cal"]["top1_image_macro"]
                ref2 = json.load(open(T22 / f"{tag}.json"))["detection_dev"]["acc_iou50_image_macro"] if a.role == "DEV" else None      # Table C has no CAL readout
            elif a.source == "zeroshot":                                            # step-0 readouts recorded by the R4-VL1 seed-0 VCS run
                z = json.load(open(R4OUT / f"{ds}_{a.backbone}_vcs_s0.json"))
                ref1, ref2 = (z["dev_step0_zero_shot"]["given_box_top1_image_macro"], z["dev_step0_zero_shot"]["acc50"]["image_macro"]) if a.role == "DEV" else (None, None)
            else:                                                                   # the run's own readouts: DEV at the selected epoch, CAL from its history
                rj = json.load(open({"r4vl1": R4OUT, "r4vl2": R4OUT.parent / "R4_VL2", "r5dist": R4OUT.parent / "R5_DIST", "r5qnorm": R4OUT.parent / "R5_QNORM"}[a.source] / f"{tag}.json"))
                if a.role == "DEV":
                    rr = rj["dev_at_selected"]
                else:
                    rr = next(h for h in rj["history"] if h["epoch"] == rj["selected"]["cal_acc50_epoch"])["cal"]
                ref1 = rr["given_box_top1_image_macro"]; ref2 = (rr["acc50_plain_cos"] if "acc50_plain_cos" in rr else rr["acc50"])["image_macro"]   # the scorer ranks by cosine
            ok1 = True if ref1 is None else abs(g1 - ref1) <= 0.003; ok2 = True if ref2 is None else abs(g2 - ref2) <= 0.003
            print(f"[{tag}] {a.role} G1 given-box {g1:.4f} vs run {ref1 if ref1 is None else round(ref1, 4)} -> {'pass' if ok1 else 'FAIL'}; G2 all-proposal Acc@0.5 {g2:.4f}"
                  + (f" vs {'Table C' if a.source == 'vl3' else 'the run'} {ref2:.4f} (diff {g2 - ref2:+.4f}) -> {'pass' if ok2 else 'FAIL'}" if ref2 is not None else " (no reference on CAL) -> pass"), flush=True)
            torch.save({"tag": tag, "role": a.role, "epoch": ck["epoch"], "image_ids": [m["image_id"] for m in meta], "scores": sc, "gates": {"G1": [g1, ref1, ok1], "G2": [g2, ref2, ok2]}}, outp)
            res[tag] = {"G1": [g1, ref1, ok1], "G2": [g2, ref2, ok2]}
    json.dump(res, open(OUT / f"gates_{ds}_{a.backbone}{'' if a.role == 'DEV' else '_' + a.role}{'' if a.source == 'vl3' else '_' + a.source}.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
