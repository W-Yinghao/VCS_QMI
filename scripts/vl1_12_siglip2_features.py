"""VL1-12 addendum 2 (estimator as a probe): SigLIP 2 Base (google/siglip2-base-patch16-224) region / expression features.  SigLIP 2 has no
region interface, so the region input is the tight box crop (as the CLIP cache) through the model's own image processor (squash resize 224,
mean / std 0.5) — labelled an adaptation.  Text = lower-cased expression, padding "max_length" 64 (model card).  Runs in the vl_baselines venv.
  check : ImageNet-1k val zero-shot top-1 on a class-balanced subset (default 10 images per class), prompt "this is a photo of {name}." with the
          open_clip class names; the SigLIP 2 paper reports 78.2 for B/16 at 224 (its own class names / prompt set) — agreement within ~2 points
          validates loading, preprocessing and the text path.
  cache : RefCOCOg-UMD train-side (FIT / CAL / DEV) referred + distractor crops and all expressions, fp16, same key layout as the CLIP cache so
          `vl1_10_fit.py --features <this file>` runs Table A on SigLIP 2 features.
    python scripts/vl1_12_siglip2_features.py check [--per-class 10]
    python scripts/vl1_12_siglip2_features.py cache
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("HF_HUB_OFFLINE", "1")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vcs_vl import refcocog as RG  # noqa: E402

HF = Path("/projects/EEG-foundation-model/yinghao/models/hf")
SNAP = next((HF / "models--google--siglip2-base-patch16-224" / "snapshots").iterdir())
OUT_FEAT = Path("/projects/EEG-foundation-model/yinghao/datasets/refcocog_umd/features_siglip2_base")
OUT_REP = REPO / "reports" / "VL1"
IN_ROOT = Path("/projects/common/imagenet/ILSVRC/Data/CLS-LOC"); IN_VAL = Path("/projects/EEG-foundation-model/yinghao/FMCA-AV/imagenet/manifests/imagenet1k_val.tsv")
IN_NAMES = Path("/projects/EEG-foundation-model/yinghao/datasets/imagenet_classnames_openclip.json")


class Crops(torch.utils.data.Dataset):
    def __init__(self, items, proc):
        self.items, self.proc = items, proc          # items: (path, xyxy or None)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, box = self.items[i]
        with Image.open(path) as im:
            im = im.convert("RGB")
            if box is not None:
                im = im.crop(tuple(box))
            return self.proc(images=im, return_tensors="pt")["pixel_values"][0]


def load():
    from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained(str(SNAP)).cuda().eval()
    return model, AutoTokenizer.from_pretrained(str(SNAP)), AutoImageProcessor.from_pretrained(str(SNAP))


@torch.no_grad()
def image_feats(model, proc, items, workers=14) -> torch.Tensor:
    dl = torch.utils.data.DataLoader(Crops(items, proc), batch_size=256, num_workers=workers, pin_memory=True)
    return torch.cat([model.get_image_features(pixel_values=x.cuda(non_blocking=True)).float().cpu() for x in dl])


@torch.no_grad()
def text_feats(model, tok, texts: list[str]) -> torch.Tensor:
    out = []
    for s in range(0, len(texts), 1024):
        t = tok([x.lower() for x in texts[s:s + 1024]], padding="max_length", max_length=64, truncation=True, return_tensors="pt").to("cuda")
        out.append(model.get_text_features(input_ids=t["input_ids"]).float().cpu())
    return torch.cat(out)


def check(a) -> int:
    model, tok, proc = load(); names = json.load(open(IN_NAMES))["names"]
    wnids = sorted({l.split()[0] for l in open("/projects/common/imagenet/LOC_synset_mapping.txt")}); idx = {w: i for i, w in enumerate(wnids)}
    rows = [l.rstrip("\n").split("\t") for l in open(IN_VAL)][1:]; per, items, ys = {}, [], []
    for p, w in sorted(rows):
        if per.get(w, 0) < a.per_class:
            per[w] = per.get(w, 0) + 1; items.append((str(IN_ROOT / p), None)); ys.append(idx[w])
    t0 = time.time()
    T = torch.nn.functional.normalize(text_feats(model, tok, [f"this is a photo of {n}." for n in names]), dim=1)
    X = torch.nn.functional.normalize(image_feats(model, proc, items), dim=1)
    top1 = float(((X @ T.T).argmax(1).numpy() == np.array(ys)).mean())
    res = {"protocol": f"ImageNet-1k val, {a.per_class} images per class (first by file name), prompt 'this is a photo of {{name}}.' lower-cased, open_clip class names, max_length 64, official image processor",
           "n_images": len(items), "top1": top1, "paper_reference_top1": 78.2, "seconds": time.time() - t0, "snapshot": str(SNAP), "processor": proc.to_dict()}
    OUT_REP.mkdir(parents=True, exist_ok=True); json.dump(res, open(OUT_REP / "vl1_12_siglip2_check.json", "w"), indent=1)
    print(f"[check] ImageNet val zero-shot top-1 {100 * top1:.2f} on {len(items)} images (paper 78.2) {res['seconds']:.0f}s"); return 0


def cache(a) -> int:
    model, tok, proc = load(); scenes, _ = RG.load_scenes(); role = RG.dev_roles(scenes)
    sc = [s for s in scenes if role[s.image_id] in ("FIT", "CAL", "DEV") and len(s.referred) >= 2 and s.path]
    objs = [(s, o) for s in sc for o in s.referred + s.distractors]
    items = [(s.path, RG.clip_box(o.box_xywh, s.width, s.height)) for s, o in objs]
    texts = [(s, o, e) for s in sc for o in s.referred for e in o.expressions]
    t0 = time.time(); img = image_feats(model, proc, items); t_img = time.time() - t0
    t0 = time.time(); txt = text_feats(model, tok, [e["raw"] for _, _, e in texts]); t_txt = time.time() - t0
    OUT_FEAT.mkdir(parents=True, exist_ok=True); prov = json.load(open(HF / "PROVENANCE_siglip2-base-patch16-224.json"))
    torch.save({"image_feat_fp16": img.half(), "text_feat_fp16": txt.half(), "obj_keys": [(s.image_id, o.ann_id) for s, o in objs],
                "text_keys": [(s.image_id, o.ann_id, e["sent_id"]) for s, o, e in texts], "roles": ["CAL", "DEV", "FIT"],
                "model": "SigLIP 2 Base (google/siglip2-base-patch16-224)", "provenance": prov, "preprocess": "official image processor (squash 224, mean/std 0.5)",
                "region": "tight clipped box crop (adaptation; no official region interface)", "text": "lower-cased, max_length 64"}, OUT_FEAT / "train_side_features.pt")
    print(f"[cache] {len(items)} crops ({t_img:.0f}s), {len(texts)} expressions ({t_txt:.0f}s), {len(sc)} images -> {OUT_FEAT}"); return 0


def main() -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check"); c.add_argument("--per-class", type=int, default=10); sub.add_parser("cache")
    a = ap.parse_args(); return check(a) if a.cmd == "check" else cache(a)


if __name__ == "__main__":
    sys.exit(main())
