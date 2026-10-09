"""VL1-12 (estimator as a probe): FG-CLIP 2 Base region / expression features through the OFFICIAL region interface, plus the official
region-path check.  Runs in the vl_baselines venv (transformers 4.57, trust_remote_code).

Region features: `model.get_image_region_features(pixel_values, pixel_attention_mask, spatial_shapes, image_sizes=[(H, W)], region_infos=[[x1,y1,x2,y2]])`
(RoIAlign 1x1, aligned, on the dense-feature head output); image preprocessing = the model's image processor with the README's
`determine_max_value` rule for max_num_patches; text = lower-cased expression, tokenizer padding "max_length" 64, `get_text_features(walk_type="box")`.
  check   : COCO val2017 ground-truth boxes classified among the 80 category names (prompt "a photo of a {name}." lower-cased, walk_type box);
            the FG-CLIP 2 paper reports 74.9 top-1 for the Base model on this protocol (CLIP B/16 44.2, SigLIP 2 B/16 53.4) — our prompt is
            disclosed; agreement within a few points validates the region path.
  cache   : RefCOCOg-UMD train-side (FIT / CAL / DEV) referred + distractor boxes and all expressions, fp16, same key layout as the CLIP cache so
            `vl1_10_fit.py --features <this file>` runs Table A on FG-CLIP 2 features.
    python scripts/vl1_12_fgclip2_features.py check [--n-images 2000]
    python scripts/vl1_12_fgclip2_features.py cache
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
REPO_ID = os.environ.get("FGCLIP2_REPO", "qihoo360/fg-clip2-base")   # VL1-15 scale probe: qihoo360/fg-clip2-large
SIZE_TAG = "base" if "base" in REPO_ID else "large" if "large" in REPO_ID else "so400m"
SNAP = next((HF / f"models--{REPO_ID.replace('/', '--')}" / "snapshots").iterdir())
OUT_FEAT = RG.feature_dir(f"features_fgclip2_{SIZE_TAG}")   # dataset-aware (VL_DATASET)
OUT_REP = REPO / "reports" / "VL1"
COCO_VAL = Path("/projects/common/coco/val2017"); COCO_VAL_ANN = Path("/projects/common/coco/annotations/instances_val2017.json")


def determine_max_value(image):  # README rule, verbatim
    w, h = image.size; max_val = (w // 16) * (h // 16)
    return 1024 if max_val > 784 else 784 if max_val > 576 else 576 if max_val > 256 else 256 if max_val > 128 else 128


def load():
    from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(str(SNAP), trust_remote_code=True).cuda().eval()
    return model, AutoTokenizer.from_pretrained(str(SNAP)), AutoImageProcessor.from_pretrained(str(SNAP))


@torch.no_grad()
def region_feats(model, proc, path: str, boxes_xyxy: list[list[float]], readme_rule: bool = False) -> torch.Tensor:
    """Default: the official box-classification setting (fgclip2/eval/coco_box_ddp.py: image processor defaults = 256 patches, NaFlex)."""
    im = Image.open(path).convert("RGB"); W, H = im.size
    inp = proc(images=im, max_num_patches=determine_max_value(im), return_tensors="pt").to("cuda") if readme_rule else proc(images=im, return_tensors="pt").to("cuda")
    feats = model.get_image_region_features(pixel_values=inp["pixel_values"], pixel_attention_mask=inp["pixel_attention_mask"],
                                            spatial_shapes=inp["spatial_shapes"], image_sizes=[(H, W)], region_infos=[boxes_xyxy])
    return feats[0].float().cpu()


@torch.no_grad()
def text_feats(model, tok, texts: list[str], walk_type: str = "box") -> torch.Tensor:
    out = []
    for s in range(0, len(texts), 512):
        t = tok([x.lower() for x in texts[s:s + 512]], padding="max_length", max_length=64, truncation=True, return_tensors="pt").to("cuda")
        out.append(model.get_text_features(**t, walk_type=walk_type).float().cpu())
    return torch.cat(out)


def class_embeddings(model, tok, names):
    """As fgclip2/eval/coco_box_ddp.py zeroshot_classifier: per class, encode every ImageNet template, L2-normalise, average, renormalise."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("fgtemplates", "/projects/EEG-foundation-model/yinghao/external/FG-CLIP/fgclip2/eval/templates.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); templates = mod.imagenet_templates
    out = []
    for nm in names:
        e = torch.nn.functional.normalize(text_feats(model, tok, [t.format(nm).lower() for t in templates]), dim=1).mean(0)
        out.append(e / e.norm())
    return torch.stack(out)


def check(a) -> int:
    model, tok, proc = load(); ann = json.load(open(COCO_VAL_ANN)); cats = {c["id"]: c["name"] for c in ann["categories"]}
    cat_ids = sorted(cats); T = class_embeddings(model, tok, [cats[c] for c in cat_ids])   # official: ImageNet template ensemble, mean, renormalised
    by_img = {}
    for x in ann["annotations"]:
        if x["bbox"][2] > 0 and x["bbox"][3] > 0:   # official script scores every annotation (crowd included)
            by_img.setdefault(x["image_id"], []).append(x)
    imgs = {i["id"]: i for i in ann["images"]}; ids = sorted(by_img)[:a.n_images]
    hit = n = 0; t0 = time.time()
    for iid in ids:
        xs = by_img[iid]; im = imgs[iid]
        boxes = [[b["bbox"][0], b["bbox"][1], b["bbox"][0] + b["bbox"][2], b["bbox"][1] + b["bbox"][3]] for b in xs]
        R = torch.nn.functional.normalize(region_feats(model, proc, str(COCO_VAL / im["file_name"]), boxes), dim=1)
        pred = (R @ T.T).argmax(1).tolist(); hit += sum(cat_ids[p] == b["category_id"] for p, b in zip(pred, xs)); n += len(xs)
    res = {"protocol": "official coco_box_ddp.py setting: COCO val2017 GT boxes (all annotations), class embedding = mean over the ImageNet templates (lower-cased, walk_type box, max_length 64), image processor defaults (256 patches, NaFlex), RoIAlign region features via get_image_region_features",
           "n_images": len(ids), "n_boxes": n, "top1": hit / n, "paper_reference_top1_base": 74.9, "seconds": time.time() - t0, "snapshot": str(SNAP)}
    OUT_REP.mkdir(parents=True, exist_ok=True); json.dump(res, open(OUT_REP / "vl1_12_fgclip2_region_check" + ("" if SIZE_TAG == "base" else "_" + SIZE_TAG) + ".json", "w"), indent=1)
    print(f"[check] COCO val2017 GT-box top-1 {100 * hit / n:.2f} on {n} boxes / {len(ids)} images (paper Base 74.9) {res['seconds']:.0f}s"); return 0


def cache(a) -> int:
    model, tok, proc = load(); scenes, _ = RG.load_scenes(); role = RG.dev_roles(scenes)
    sc = [s for s in scenes if role[s.image_id] in ("FIT", "CAL", "DEV") and len(s.referred) >= 2 and s.path]
    keys_o, feats_o, keys_t, texts = [], [], [], []; t0 = time.time()
    for k, s in enumerate(sc):
        objs = s.referred + s.distractors; boxes = [list(RG.clip_box(o.box_xywh, s.width, s.height)) for o in objs]
        feats_o.append(region_feats(model, proc, s.path, boxes)); keys_o += [(s.image_id, o.ann_id) for o in objs]
        for o in s.referred:
            for e in o.expressions:
                keys_t.append((s.image_id, o.ann_id, e["sent_id"])); texts.append(e["raw"])
        if k % 500 == 0:
            print(f"  {k}/{len(sc)} images {time.time() - t0:.0f}s", flush=True)
    img = torch.cat(feats_o); txt = text_feats(model, tok, texts); OUT_FEAT.mkdir(parents=True, exist_ok=True)
    prov = json.load(open(HF / f"PROVENANCE_{REPO_ID.split('/')[1]}.json"))
    torch.save({"image_feat_fp16": img.half(), "text_feat_fp16": txt.half(), "obj_keys": keys_o, "text_keys": keys_t, "roles": ["CAL", "DEV", "FIT"],
                "model": f"FG-CLIP 2 ({REPO_ID})", "provenance": prov, "preprocess": "official image processor, max_num_patches by README rule",
                "region": "official get_image_region_features (RoIAlign 1x1 on dense-feature head), boxes xyxy absolute", "text": "lower-cased, max_length 64, walk_type box"},
               OUT_FEAT / "train_side_features.pt")
    print(f"[cache] {len(keys_o)} regions, {len(keys_t)} expressions, {len(sc)} images, {time.time() - t0:.0f}s -> {OUT_FEAT}"); return 0


def main() -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check"); c.add_argument("--n-images", type=int, default=2000); sub.add_parser("cache")
    a = ap.parse_args(); return check(a) if a.cmd == "check" else cache(a)


if __name__ == "__main__":
    sys.exit(main())
