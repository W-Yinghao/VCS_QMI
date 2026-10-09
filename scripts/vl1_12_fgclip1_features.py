"""VL1-12 addendum 3 / Table B row: FG-CLIP (v1) Base (qihoo360/fg-clip-base, OpenAI-CLIP ViT-B/16 initialisation + fine-grained training) region /
expression features through its region interface `get_image_box_roi_features(pixel_values, box_info)` (RoIAlign 1x1 on the projected
second-to-last-layer feature map), as in the repository's branch v1.0 `fgclip/eval/in_1K/coco_box_cls.py::test_clip_on_coco_boxes_base_roialign`:
image squashed to 224 x 224, CLIP image processor, box = [0, x1, y1, x2, y2] in 14 x 14 grid units.  (That script's evaluate() actually calls a
crop variant — the RoIAlign function is the documented region path and is used here; disclosed.)  Text: the expression as written (the v1 README
does not lower-case), max_length 77, padding "max_length", walk_short_pos True.  Runs in the vl_baselines venv (trust_remote_code).
  check : COCO val2017 GT boxes (all annotations) classified among the 80 names, class embedding = mean over the ImageNet templates (as the
          repository's zeroshot_classifier); reference: CLIP B/16 44.2 on this protocol (FG-CLIP 2 paper Table 2).  Gate: loads under
          transformers 4.57 and beats 44.2.
  cache : RefCOCOg-UMD train-side (FIT / CAL / DEV) referred + distractor boxes and all expressions, fp16, CLIP-cache key layout.
    python scripts/vl1_12_fgclip1_features.py check [--n-images 5000]
    python scripts/vl1_12_fgclip1_features.py cache
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

os.environ.setdefault("HF_HUB_OFFLINE", "1")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vcs_vl import refcocog as RG  # noqa: E402

HF = Path("/projects/EEG-foundation-model/yinghao/models/hf")
REPO_ID = os.environ.get("FGCLIP1_REPO", "qihoo360/fg-clip-base")   # VL1-15 scale probe: qihoo360/fg-clip-large (336 px, 24 x 24 grid)
SIZE_TAG = "large" if "large" in REPO_ID else "base"
SNAP = next((HF / f"models--{REPO_ID.replace('/', '--')}" / "snapshots").iterdir())
OUT_FEAT = RG.feature_dir(f"features_fgclip1_{SIZE_TAG}"); OUT_REP = REPO / "reports" / "VL1"   # dataset-aware (VL_DATASET)
COCO_VAL = Path("/projects/common/coco/val2017"); COCO_VAL_ANN = Path("/projects/common/coco/annotations/instances_val2017.json")
SIZE, GRID = (336, 24) if SIZE_TAG == "large" else (224, 14)   # as the v1.0 coco_box_cls script (feature_size 24 for 336)


def load():
    from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPImageProcessor
    model = AutoModelForCausalLM.from_pretrained(str(SNAP), trust_remote_code=True).cuda().eval()
    return model, AutoTokenizer.from_pretrained(str(SNAP)), CLIPImageProcessor.from_pretrained(str(SNAP))


@torch.no_grad()
def region_feats(model, proc, path: str, boxes_xywh) -> torch.Tensor:
    im = Image.open(path).convert("RGB"); W, H = im.size
    x = proc.preprocess(im.resize((SIZE, SIZE)), return_tensors="pt")["pixel_values"].cuda()
    bi = torch.tensor([[0, b[0] / W * GRID, b[1] / H * GRID, (b[0] + b[2]) / W * GRID, (b[1] + b[3]) / H * GRID] for b in boxes_xywh], dtype=torch.float32, device="cuda")
    return model.get_image_box_roi_features(x, box_info=bi).float().cpu()


@torch.no_grad()
def text_feats(model, tok, texts) -> torch.Tensor:
    out = []
    for s in range(0, len(texts), 512):
        ids = torch.tensor(tok(texts[s:s + 512], max_length=77, padding="max_length", truncation=True).input_ids, dtype=torch.long, device="cuda")
        out.append(model.get_text_features(ids, walk_short_pos=True).float().cpu())
    return torch.cat(out)


def check(a) -> int:
    model, tok, proc = load(); ann = json.load(open(COCO_VAL_ANN)); cats = {c["id"]: c["name"] for c in ann["categories"]}; cat_ids = sorted(cats)
    spec = importlib.util.spec_from_file_location("fgt", "/projects/EEG-foundation-model/yinghao/external/FG-CLIP/fgclip2/eval/templates.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    T = torch.stack([torch.nn.functional.normalize(text_feats(model, tok, [t.format(cats[c]) for t in mod.imagenet_templates]), dim=1).mean(0) for c in cat_ids])
    T = torch.nn.functional.normalize(T, dim=1)
    by_img = {}
    for x in ann["annotations"]:
        if x["bbox"][2] > 0 and x["bbox"][3] > 0:
            by_img.setdefault(x["image_id"], []).append(x)
    imgs = {i["id"]: i for i in ann["images"]}; ids = sorted(by_img)[:a.n_images]; hit = n = 0; t0 = time.time()
    for iid in ids:
        xs = by_img[iid]; R = torch.nn.functional.normalize(region_feats(model, proc, str(COCO_VAL / imgs[iid]["file_name"]), [b["bbox"] for b in xs]), dim=1)
        pred = (R @ T.T).argmax(1).tolist(); hit += sum(cat_ids[p] == b["category_id"] for p, b in zip(pred, xs)); n += len(xs)
    res = {"protocol": "COCO val2017 GT boxes (all annotations), ImageNet-template class embeddings, RoIAlign region path (v1.0 coco_box_cls roialign function), 224 squash",
           "n_images": len(ids), "n_boxes": n, "top1": hit / n, "reference_clip_b16": 44.2, "pass": hit / n > 0.442, "snapshot": str(SNAP), "seconds": time.time() - t0}
    OUT_REP.mkdir(parents=True, exist_ok=True); json.dump(res, open(OUT_REP / "vl1_12_fgclip1_region_check" + ("" if SIZE_TAG == "base" else "_" + SIZE_TAG) + ".json", "w"), indent=1)
    print(f"[check] COCO val2017 GT-box top-1 {100 * hit / n:.2f} on {n} boxes / {len(ids)} images (CLIP B/16 44.2) -> {'PASS' if res['pass'] else 'FAIL'}"); return 0


def cache(a) -> int:
    model, tok, proc = load(); scenes, _ = RG.load_scenes(); role = RG.dev_roles(scenes)
    sc = [s for s in scenes if role[s.image_id] in ("FIT", "CAL", "DEV") and len(s.referred) >= 2 and s.path]
    keys_o, feats_o, keys_t, texts = [], [], [], []; t0 = time.time()
    for k, s in enumerate(sc):
        objs = s.referred + s.distractors
        boxes = [list(RG.clip_box(o.box_xywh, s.width, s.height)) for o in objs]      # clipped xyxy
        feats_o.append(region_feats(model, proc, s.path, [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes])); keys_o += [(s.image_id, o.ann_id) for o in objs]
        for o in s.referred:
            for e in o.expressions:
                keys_t.append((s.image_id, o.ann_id, e["sent_id"])); texts.append(e["raw"])
        if k % 1000 == 0:
            print(f"  {k}/{len(sc)} images {time.time() - t0:.0f}s", flush=True)
    img = torch.cat(feats_o); txt = text_feats(model, tok, texts); OUT_FEAT.mkdir(parents=True, exist_ok=True)
    torch.save({"image_feat_fp16": img.half(), "text_feat_fp16": txt.half(), "obj_keys": keys_o, "text_keys": keys_t, "roles": ["CAL", "DEV", "FIT"],
                "model": f"FG-CLIP (v1) ({REPO_ID})", "provenance": json.load(open(HF / f"PROVENANCE_{REPO_ID.split('/')[1]}.json")),
                "preprocess": "224 squash + CLIP image processor", "region": "get_image_box_roi_features, 14x14 grid units (clipped boxes)", "text": "as written, max_length 77, walk_short_pos"},
               OUT_FEAT / "train_side_features.pt")
    print(f"[cache] {len(keys_o)} regions, {len(keys_t)} expressions, {len(sc)} images, {time.time() - t0:.0f}s -> {OUT_FEAT}"); return 0


def main() -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check"); c.add_argument("--n-images", type=int, default=5000); sub.add_parser("cache")
    a = ap.parse_args(); return check(a) if a.cmd == "check" else cache(a)


if __name__ == "__main__":
    sys.exit(main())
