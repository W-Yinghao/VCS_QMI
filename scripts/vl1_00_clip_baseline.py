"""VL1-00 raw-CLIP baseline on RefCOCOg (UMD), training side only (FIT / CAL / DEV roles of the UMD train split; official val / test untouched).
Frozen open_clip ViT-B/16 (OpenAI weights, local provenance file).  Region input = tight box crop -> the model's official preprocessing (resize
224 + centre crop + CLIP normalisation); text = the raw expression with the model tokenizer (context 77).  Features cached in float16 (pre-
normalisation) under /projects; an fp32 re-extraction of 512 crops / 512 texts checks the cache precision (cosine error, rank agreement).
Task readout (per expression, candidates = the image's referred objects [primary] or referred + distractor objects [secondary]): query-weighted
Top-1, image-macro Top-1, MRR; strata by candidate count, same-category scenes, long expressions; controls: random (exact expectation),
category-name text shortcut (ties scored at their expectation), largest-box prior.
    python scripts/vl1_00_clip_baseline.py [--roles FIT,CAL,DEV] [--workers 14]
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
from torch.nn import functional as F

os.environ.setdefault("HF_HUB_OFFLINE", "1")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vcs_vl import refcocog as RG  # noqa: E402

CLIP_CACHE = "/projects/EEG-foundation-model/yinghao/models/open_clip"
ARCH = os.environ.get("CLIP_ARCH", "ViT-B-16-quickgelu")   # VL1-15 scale probe: ViT-L-14-336-quickgelu
ATAG = {"ViT-B-16-quickgelu": "vitb16", "ViT-L-14-336-quickgelu": "vitl14_336"}[ARCH]
PROV = {"ViT-B-16-quickgelu": "PROVENANCE_ViT-B-16_openai.json", "ViT-L-14-336-quickgelu": "PROVENANCE_ViT-L-14-336_openai.json"}[ARCH]
FEAT_DIR = RG.feature_dir(f"features_clip_{ATAG}_openai")   # dataset-aware (VL_DATASET); refcocog ViT-B/16 path unchanged
OUT = REPO / "reports" / "VL1"


class Crops(torch.utils.data.Dataset):
    def __init__(self, items, preprocess):
        self.items, self.pre = items, preprocess          # items: (path, xyxy)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image
        path, box = self.items[i]
        with Image.open(path) as im:
            return self.pre(im.convert("RGB").crop(tuple(box)))


@torch.no_grad()
def encode_images(model, items, pre, dev, workers, dtype):
    dl = torch.utils.data.DataLoader(Crops(items, pre), batch_size=256, num_workers=workers, pin_memory=True)
    out = []
    for x in dl:
        with torch.autocast("cuda", dtype=torch.float16, enabled=dtype == torch.float16):
            out.append(model.encode_image(x.to(dev, non_blocking=True)).float().cpu())
    return torch.cat(out)


@torch.no_grad()
def encode_texts(model, tok, texts, dev, dtype):
    out = []
    for s in range(0, len(texts), 1024):
        t = tok(texts[s:s + 1024]).to(dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dtype == torch.float16):
            out.append(model.encode_text(t).float().cpu())
    return torch.cat(out)


def top1_expect(scores: np.ndarray, target: int) -> float:
    """Top-1 with ties scored at their expectation."""
    best = scores.max(); tied = np.flatnonzero(np.isclose(scores, best, rtol=0, atol=1e-7))
    return float(target in tied) / len(tied)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--roles", default="FIT,CAL,DEV"); ap.add_argument("--workers", type=int, default=14); a = ap.parse_args()
    import open_clip
    dev = torch.device("cuda", 0); t_all = time.time()
    # "ViT-B-16-quickgelu": the OpenAI weights were trained with QuickGELU; open_clip's plain "ViT-B-16" + "openai" only warns and runs GELU
    model, _, pre = open_clip.create_model_and_transforms(ARCH, pretrained="openai", cache_dir=CLIP_CACHE)
    acts = {type(m).__name__ for m in model.modules() if "GELU" in type(m).__name__}
    assert acts == {"QuickGELU"}, f"expected QuickGELU activations, found {acts}"
    model = model.to(dev).eval(); tok = open_clip.get_tokenizer(ARCH)
    prov = json.load(open(Path(CLIP_CACHE) / PROV))
    scenes, st = RG.load_scenes(); cats = st["categories"]; role = RG.dev_roles(scenes)
    want = set(a.roles.split(",")); sc = [s for s in scenes if role[s.image_id] in want and len(s.referred) >= 2 and s.path]
    objs = [(s, o) for s in sc for o in s.referred + s.distractors]
    items = [(s.path, RG.clip_box(o.box_xywh, s.width, s.height)) for s, o in objs]
    texts = [(s, o, e) for s in sc for o in s.referred for e in o.expressions]
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.time(); img = encode_images(model, items, pre, dev, a.workers, torch.float16); t_img = time.time() - t0
    t0 = time.time(); txt = encode_texts(model, tok, [e["raw"] for _, _, e in texts], dev, torch.float16); t_txt = time.time() - t0
    # precision check: fp32 re-extraction of 512 crops / texts
    g = np.random.default_rng(0); ci = g.choice(len(items), min(512, len(items)), replace=False); ti = g.choice(len(texts), min(512, len(texts)), replace=False)
    img32 = encode_images(model, [items[i] for i in ci], pre, dev, a.workers, torch.float32); txt32 = encode_texts(model, tok, [texts[i][2]["raw"] for i in ti], dev, torch.float32)
    c16 = F.normalize(img[ci], dim=1) @ F.normalize(txt[ti], dim=1).T; c32 = F.normalize(img32, dim=1) @ F.normalize(txt32, dim=1).T
    prec = {"max_abs_cos_err": float((c16 - c32).abs().max()), "mean_abs_cos_err": float((c16 - c32).abs().mean()),
            "argmax_agreement_over_512_queries": float((c16.argmax(0) == c32.argmax(0)).float().mean())}
    key_obj = [(s.image_id, o.ann_id) for s, o in objs]; key_txt = [(s.image_id, o.ann_id, e["sent_id"]) for s, o, e in texts]
    torch.save({"image_feat_fp16": img.half(), "text_feat_fp16": txt.half(), "obj_keys": key_obj, "text_keys": key_txt, "roles": sorted(want),
                "model": f"open_clip {ARCH} openai", "provenance": prov, "preprocess": str(pre), "crop": "tight clipped box, then official preprocess"},
               FEAT_DIR / "train_side_features.pt")
    cache_bytes = os.path.getsize(FEAT_DIR / "train_side_features.pt")
    # category-name text features for the shortcut control
    cat_ids = sorted(cats); cat_t = F.normalize(encode_texts(model, tok, [f"a photo of a {cats[c]}." for c in cat_ids], dev, torch.float32), dim=1)
    cat_row = {c: i for i, c in enumerate(cat_ids)}
    imgN = F.normalize(img, dim=1).numpy(); txtN = F.normalize(txt, dim=1).numpy(); cat_tN = cat_t.numpy()
    oi = {k: i for i, k in enumerate(key_obj)}; tj = 0
    res = {}
    acc = {r: {"primary": [], "secondary": [], "rand_p": [], "rand_s": [], "cat_p": [], "box_p": [], "mrr_p": [], "img": [], "m": [], "samecat": [], "long": []} for r in want}
    for s in sc:
        r = role[s.image_id]; A = acc[r]
        ref_rows = [oi[(s.image_id, o.ann_id)] for o in s.referred]; all_rows = ref_rows + [oi[(s.image_id, o.ann_id)] for o in s.distractors]
        same = len({o.category_id for o in s.referred}) < len(s.referred)
        img_hits = []
        for k, o in enumerate(s.referred):
            for e in o.expressions:
                v = txtN[tj]; tj += 1
                sp = imgN[ref_rows] @ v; sa = imgN[all_rows] @ v
                hit = top1_expect(sp, k); A["primary"].append(hit); A["secondary"].append(top1_expect(sa, k)); img_hits.append(hit)
                A["rand_p"].append(1 / len(ref_rows)); A["rand_s"].append(1 / len(all_rows))
                cs = np.array([cat_tN[cat_row[oo.category_id]] @ v for oo in s.referred]); A["cat_p"].append(top1_expect(cs, k))
                areas = np.array([oo.box_xywh[2] * oo.box_xywh[3] for oo in s.referred]); A["box_p"].append(top1_expect(areas, k))
                rank = 1 + int((sp > sp[k]).sum()); A["mrr_p"].append(1 / rank)
                A["m"].append(len(ref_rows)); A["samecat"].append(same); A["long"].append(e["n_tokens"] > 12)
        A["img"].append(float(np.mean(img_hits)))
    for r, A in acc.items():
        P = np.array(A["primary"]); m = np.array(A["m"]); scat = np.array(A["samecat"]); lg = np.array(A["long"])
        res[r] = {"n_images": len(A["img"]), "n_queries": int(len(P)), "top1_query_weighted": float(P.mean()), "top1_image_macro": float(np.mean(A["img"])),
                  "mrr": float(np.mean(A["mrr_p"])), "top1_all_objects_secondary": float(np.mean(A["secondary"])),
                  "random_expectation_primary": float(np.mean(A["rand_p"])), "random_expectation_all_objects": float(np.mean(A["rand_s"])),
                  "category_text_shortcut_top1": float(np.mean(A["cat_p"])), "largest_box_prior_top1": float(np.mean(A["box_p"])),
                  "by_candidate_count": {str(k): {"n": int((m == k).sum()), "top1": float(P[m == k].mean())} for k in sorted(set(m.tolist())) if (m == k).sum() >= 20},
                  "same_category_scenes": {"n": int(scat.sum()), "top1": float(P[scat].mean()) if scat.any() else None, "random": float(np.mean(1 / m[scat])) if scat.any() else None},
                  "long_expressions_gt12_tokens": {"n": int(lg.sum()), "top1": float(P[lg].mean()) if lg.any() else None}}
    out = {"model": f"open_clip {ARCH} openai", "provenance_sha256": {k: v["sha256"] for k, v in prov["files"].items()}, "roles": sorted(want),
           "n_crops": len(items), "n_texts": len(texts), "precision_check_fp16_vs_fp32": prec,
           "resources": {"image_encode_seconds": t_img, "crops_per_second": len(items) / t_img, "text_encode_seconds": t_txt, "cache_bytes": cache_bytes,
                         "peak_gpu_mem_gb": torch.cuda.max_memory_allocated(dev) / 1e9, "gpu": torch.cuda.get_device_name(dev), "total_seconds": time.time() - t_all},
           "results": res, "official_val_used": False, "official_test_used": False}
    json.dump(out, open(OUT / ("vl1_raw_clip_trainside" + RG.dataset_tag() + ("" if ATAG == "vitb16" else "_" + ATAG) + ".json"), "w"), indent=1)
    for r in sorted(res):
        v = res[r]; print(f"[{r}] imgs {v['n_images']} q {v['n_queries']} top1 {v['top1_query_weighted']:.4f} (macro {v['top1_image_macro']:.4f}, all-obj {v['top1_all_objects_secondary']:.4f}) "
                          f"rand {v['random_expectation_primary']:.3f} cat-shortcut {v['category_text_shortcut_top1']:.3f} largest-box {v['largest_box_prior_top1']:.3f} samecat {v['same_category_scenes']}")
    print("precision", prec, "resources", out["resources"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
