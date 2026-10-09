"""VL1-20 — Table C full-grounding baselines (no given boxes): the model predicts one box per expression from the whole image; a hit is IoU >= 0.5
with the target box (MDETR's own refexp metric, GIoU >= 0.5, is reported alongside; GIoU <= IoU, so it is the stricter one).
  gdino : Grounding DINO Swin-T OGC (IDEA-Research/grounding-dino-tiny; O365 + GoldG + Cap4M, no COCO / RefCOCO) — zero-shot.  Text = the
          lower-cased expression + " ."; box = argmax over the 900 queries of the max sigmoid logit over the expression tokens ([CLS] / [SEP]
          excluded) — the protocol of the third-party RefCOCOg numbers (val 60.4 / test 59.7).  Splits: val (reproduction) and dev (Table A's
          eligible DEV images and expressions; DEV images come from the UMD train split, which Grounding DINO never saw).
  mdetr : MDETR R101 / EB3 fine-tuned on RefCOCOg (Zenodo 4721981; paper UMD val 81.64 / 83.35).  Trained on RefCOCOg TRAIN (and pretrained on
          RefCOCO/+/g train expressions), so it is evaluated on UMD val only (reproduction / supervised reference); a DEV number would be a
          training-set number and is not produced.  Inputs as the official eval: resize shorter side 800 (max 1333), ImageNet normalisation,
          caption from MDETR's own finetune_refcocog_val.json (REFER `sent` form) when present, else the REFER raw sentence (tag suffix _refercap); score = 1 − p(no-object), top box.
    python scripts/vl1_20_grounding.py gdino --split val|dev
    python scripts/vl1_20_grounding.py mdetr --model r101|eb3 --split val
Writes outputs/VL1_20/<tag>.jsonl (per expression) and reports/VL1/VL1_20_<tag>.json (summary).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vcs_vl import refcocog as RG  # noqa: E402

HF = Path("/projects/EEG-foundation-model/yinghao/models/hf"); MDETR_DIR = Path("/projects/EEG-foundation-model/yinghao/external/mdetr")
MDETR_W = Path("/projects/EEG-foundation-model/yinghao/models/mdetr"); MDETR_ANN = Path("/projects/EEG-foundation-model/yinghao/datasets/mdetr_annotations")
OUT = Path("/home/infres/yinwang/CS_QMI/outputs/VL1_20"); REP = REPO / "reports" / "VL1"


def snap(repo: str) -> str:
    return str(next((HF / f"models--{repo.replace('/', '--')}" / "snapshots").iterdir()))


def box_iou_giou(a, b):
    """a, b: xyxy lists → (IoU, GIoU)."""
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1); ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    iou = inter / ua if ua > 0 else 0.0
    cx1, cy1, cx2, cy2 = min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]); c = (cx2 - cx1) * (cy2 - cy1)
    return iou, (iou - (c - ua) / c) if c > 0 else iou


def items_refer(split: str):
    """(image_id, path, text, target xyxy, ref_id, sent_id) from REFER: split val = UMD val; dev = Table A's eligible DEV referred expressions."""
    if split == "val":
        refs = pickle.load(open(RG.ROOT / "refs(umd).p", "rb")); inst = json.load(open(RG.ROOT / "instances.json"))
        box = {a["id"]: a["bbox"] for a in inst["annotations"]}; out = []
        for r in refs:
            if r["split"] != "val":
                continue
            x, y, w, h = box[r["ann_id"]]; p = RG.coco_path(r["image_id"])
            out += [(r["image_id"], str(p), s["raw"], [x, y, x + w, y + h], r["ref_id"], s["sent_id"]) for s in r["sentences"]]
        return out
    scenes, _ = RG.load_scenes(); role = RG.dev_roles(scenes); out = []
    for s in scenes:
        if role[s.image_id] != "DEV" or len(s.referred) < 2 or not s.path:
            continue
        for o in s.referred:
            x, y, w, h = o.box_xywh
            out += [(s.image_id, s.path, e["raw"], [x, y, x + w, y + h], None, e["sent_id"]) for e in o.expressions]
    return out


def summarise(rows, tag, extra):
    per_img = defaultdict(list)
    for r in rows:
        per_img[r["image_id"]].append(r["hit_iou"])
    res = {"tag": tag, "n": len(rows), "acc_iou50": float(np.mean([r["hit_iou"] for r in rows])), "acc_giou50": float(np.mean([r["hit_giou"] for r in rows])),
           "acc_iou50_image_macro": float(np.mean([np.mean(v) for v in per_img.values()])), "n_images": len(per_img), **extra}
    REP.mkdir(parents=True, exist_ok=True); json.dump(res, open(REP / f"VL1_20_{tag}.json", "w"), indent=1)
    print(f"[{tag}] n {res['n']} Acc@IoU0.5 {100 * res['acc_iou50']:.2f} (GIoU {100 * res['acc_giou50']:.2f}; image-macro {100 * res['acc_iou50_image_macro']:.2f})", flush=True)


@torch.no_grad()
def gdino(a) -> int:
    from transformers import AutoProcessor, GroundingDinoForObjectDetection
    sp = snap("IDEA-Research/grounding-dino-tiny"); proc = AutoProcessor.from_pretrained(sp); model = GroundingDinoForObjectDetection.from_pretrained(sp).cuda().eval()
    if a.dataset == "refcocog":
        items = items_refer(a.split); tag = f"gdino_tiny_{a.split}"
    elif a.split == "val":
        items = mdetr_items(a.dataset, "val"); tag = f"gdino_tiny_{a.dataset.replace('+', 'plus')}_val"
    else:   # VL2 DEV: Table A's eligible DEV images / expressions of that dataset (needs VL_DATASET exported to the job)
        assert RG.DATASET == a.dataset, f"export VL_DATASET={a.dataset}"
        items = items_refer("dev"); tag = f"gdino_tiny_{a.dataset.replace('+', 'plus')}_dev"
    OUT.mkdir(parents=True, exist_ok=True); rows, t0 = [], time.time()
    with open(OUT / f"{tag}.jsonl", "w") as fh:
        for k, (iid, path, text, tgt, ref_id, sid) in enumerate(items):
            im = Image.open(path).convert("RGB"); W, H = im.size
            inp = proc(images=im, text=text.lower().strip().rstrip(".") + " .", return_tensors="pt").to("cuda")
            out = model(**inp); L = int(inp["attention_mask"][0].sum())
            prob = out.logits[0, :, 1:L - 1].sigmoid(); score = prob.max(-1).values; q = int(score.argmax())
            cx, cy, w, h = out.pred_boxes[0, q].tolist(); pb = [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]
            iou, giou = box_iou_giou(pb, tgt)
            r = {"image_id": iid, "sent_id": sid, "ref_id": ref_id, "text": text, "pred_xyxy": pb, "score": float(score[q]), "target_xyxy": tgt, "iou": iou, "giou": giou,
                 "hit_iou": int(iou >= 0.5), "hit_giou": int(giou >= 0.5)}
            rows.append(r); fh.write(json.dumps(r) + "\n")
            if k % 1000 == 0:
                print(f"  {k}/{len(items)} {time.time() - t0:.0f}s running Acc {100 * np.mean([x['hit_iou'] for x in rows]):.2f}", flush=True)
    summarise(rows, tag, {"model": "IDEA-Research/grounding-dino-tiny", "snapshot": sp, "split": a.split, "seconds": time.time() - t0,
                          "dataset": a.dataset, "reference": REF[("gdino", a.dataset)] if a.split == "val" else None})
    return 0


def mdetr_items(dataset: str = "refcocog", split: str = "val"):
    """MDETR's pre-processed annotations (finetune_<dataset>_<split>.json from mdetr_annotations.tar.gz): one entry per expression with the
    exact caption of the official evaluation (REFER's tokenised, lower-cased `sent` form), the COCO image id and one target box."""
    f = MDETR_ANN / f"finetune_{dataset}_{split}.json"
    if not f.exists():
        return None
    d = json.load(open(f)); anns = {x["image_id"]: x for x in d["annotations"]}; out = []
    for im in d["images"]:
        x, y, w, h = anns[im["id"]]["bbox"]; iid = int(im["original_id"])
        out.append((iid, str(RG.coco_path(iid)), im["caption"], [x, y, x + w, y + h], None, im["id"]))
    return out


def mdetr_items_val():
    return mdetr_items("refcocog", "val")


REF = {("mdetr", "r101", "refcocog"): "paper UMD val 81.64", ("mdetr", "eb3", "refcocog"): "paper UMD val 83.35",
       ("mdetr", "r101", "refcoco"): "paper UNC val 86.75", ("mdetr", "r101", "refcoco+"): "paper UNC val 79.52",
       ("gdino", "refcocog"): "third-party val 60.4 (paper 67.46 for a non-released model)",
       ("gdino", "refcoco"): "paper zero-shot val 50.41 [U: released checkpoint not verified]", ("gdino", "refcoco+"): "paper zero-shot val 51.40 [U]"}


@torch.no_grad()
def mdetr(a) -> int:
    import torchvision, timm
    import torchvision.transforms as T
    sys.path.insert(0, str(MDETR_DIR))
    for nm in ("resnet101",):            # the checkpoint replaces the ImageNet backbone weights: never download them
        orig = getattr(torchvision.models, nm)
        def ctor(*args, _orig=orig, **kw):
            kw.pop("pretrained", None); return _orig(*args, weights=None, **kw)
        setattr(torchvision.models, nm, ctor)
    import models.backbone as MB
    def _cm(*args, **kw):                 # TimmBackbone calls create_model(..., pretrained=True): force False (weights come from the checkpoint)
        kw["pretrained"] = False; return timm.create_model(*args, **kw)
    MB.create_model = _cm
    os.environ["HF_HUB_CACHE"] = str(HF)
    import hubconf
    from transformers import RobertaModel, RobertaTokenizerFast
    rb = snap("FacebookAI/roberta-base")
    _rm, _rt = RobertaModel.from_pretrained, RobertaTokenizerFast.from_pretrained
    RobertaModel.from_pretrained = classmethod(lambda cls, name, *x, **k: _rm.__func__(cls, rb if name == "roberta-base" else name, *x, **k))
    RobertaTokenizerFast.from_pretrained = classmethod(lambda cls, name, *x, **k: _rt.__func__(cls, rb if name == "roberta-base" else name, *x, **k))
    bb = {"r101": "resnet101", "eb3": "timm_tf_efficientnet_b3_ns"}[a.model]; wf = f"{a.dataset}_{'resnet101' if a.model == 'r101' else 'EB3'}_checkpoint.pth"
    model = hubconf._make_detr(bb); ck = torch.load(MDETR_W / wf, map_location="cpu", weights_only=False)
    sd = dict(ck["model"]); dropped = [k for k in sd if k.endswith("embeddings.position_ids")]   # arange buffer, non-persistent in current transformers
    for k in dropped:
        sd.pop(k)
    model.load_state_dict(sd, strict=True)
    model = model.cuda().eval()
    items = mdetr_items(a.dataset, "val"); cap_src = f"MDETR finetune_{a.dataset}_val.json"
    if items is None:
        assert a.dataset == "refcocog"; items, cap_src = items_refer("val"), "REFER raw sentence"
    tf = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    OUT.mkdir(parents=True, exist_ok=True); tag = f"mdetr_{a.model}" + ("" if a.dataset == "refcocog" else "_" + a.dataset.replace("+", "plus")) + "_val" + ("" if cap_src.startswith("MDETR") else "_refercap"); rows, t0 = [], time.time()
    with open(OUT / f"{tag}.jsonl", "w") as fh:
        for k, (iid, path, text, tgt, ref_id, sid) in enumerate(items):
            im = Image.open(path).convert("RGB"); W, H = im.size; s = 800 / min(W, H)
            if max(W, H) * s > 1333:
                s = 1333 / max(W, H)
            x = tf(im.resize((int(round(W * s)), int(round(H * s))), Image.BILINEAR)).unsqueeze(0).cuda()
            mem = model(x, [text], encode_and_save=True); out = model(x, [text], encode_and_save=False, memory_cache=mem)
            p = out["pred_logits"][0].softmax(-1); score = 1 - p[:, -1]; q = int(score.argmax())
            cx, cy, w, h = out["pred_boxes"][0, q].tolist(); pb = [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]
            iou, giou = box_iou_giou(pb, tgt)
            r = {"image_id": iid, "sent_id": sid, "text": text, "pred_xyxy": pb, "score": float(score[q]), "target_xyxy": tgt, "iou": iou, "giou": giou,
                 "hit_iou": int(iou >= 0.5), "hit_giou": int(giou >= 0.5)}
            rows.append(r); fh.write(json.dumps(r) + "\n")
            if k % 1000 == 0:
                print(f"  {k}/{len(items)} {time.time() - t0:.0f}s running Acc {100 * np.mean([x['hit_iou'] for x in rows]):.2f} (GIoU {100 * np.mean([x['hit_giou'] for x in rows]):.2f})", flush=True)
    summarise(rows, tag, {"model": f"MDETR {a.model} RefCOCOg fine-tuned", "weights": wf, "caption_source": cap_src, "dropped_keys": dropped, "seconds": time.time() - t0,
                          "dataset": a.dataset, "reference": REF.get(("mdetr", a.model, a.dataset)), "contamination": f"trained on {a.dataset} train; val only"})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gdino"); g.add_argument("--split", choices=["val", "dev"], required=True); g.add_argument("--dataset", choices=["refcocog", "refcoco", "refcoco+"], default="refcocog")
    m = sub.add_parser("mdetr"); m.add_argument("--model", choices=["r101", "eb3"], required=True); m.add_argument("--split", choices=["val"], default="val")
    m.add_argument("--dataset", choices=["refcocog", "refcoco", "refcoco+"], default="refcocog")
    a = ap.parse_args(); return gdino(a) if a.cmd == "gdino" else mdetr(a)


if __name__ == "__main__":
    sys.exit(main())
