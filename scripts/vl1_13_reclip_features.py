"""VL1-13 (Table A on ReCLIP's isolation features): region / expression features built with ReCLIP's OWN preprocessing code (allenai/reclip,
patched copy; `ClipExecutor.tensorize_inputs`: crop and blur isolations, GaussianBlur 100 with the box pasted back, square resize, CLIP
normalisation; text "a photo of " + lower-cased expression) and its two CLIP models (RN50x16, ViT-B/32; OpenAI weights, fp16 as clip.load).
Per box u = ½ [crop_RN, blur_RN, crop_B32, blur_B32] (each L2-normalised), per expression v = ½ [t_RN, t_RN, t_B32, t_B32] → |u| = |v| = 1 and
cos(u, v) = mean of ReCLIP's four cosines.  ReCLIP IPS-only ranks by Σ 100·cos (both logit scales 100), so raw-cosine ranking on this cache = ReCLIP
IPS-only ranking.  Boxes = the raw COCO bbox, as ReCLIP's main.py (not the clipped box of the CLIP cache).  Runs in the vl_baselines venv.
  cache : RefCOCOg-UMD train-side (FIT / CAL / DEV, eligible images) referred + distractor boxes, all expressions; CLIP-cache key layout.
  agree : DEV check — per-query argmax over the referred candidates from this cache vs ReCLIP's own IPS-only output
          (outputs/VL1_reclip/reclip_baseline_dev_referred.json); pass = agreement ≥ 99 % (fp16 encodes on a different batch shape).
    python scripts/vl1_13_reclip_features.py cache [--workers 14] [--limit 20]
    python scripts/vl1_13_reclip_features.py agree
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

RECLIP = Path("/projects/EEG-foundation-model/yinghao/external/reclip_patched")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(RECLIP))
from vcs_vl import refcocog as RG  # noqa: E402

OUT_FEAT = Path("/projects/EEG-foundation-model/yinghao/datasets/refcocog_umd/features_reclip_ips")
OUT_REP = REPO / "reports" / "VL1"
MODELS = ("RN50x16", "ViT-B/32")


def prep_executor():
    """A CPU copy of ReCLIP's ClipExecutor holding only the preprocessing (no models), built exactly as ClipExecutor.__init__ with square_size."""
    import clip
    import torchvision.transforms as transforms
    from executor import ClipExecutor, Executor
    ex = ClipExecutor.__new__(ClipExecutor)
    Executor.__init__(ex, device="cpu", box_representation_method="crop,blur", method_aggregator="sum", square_size=True, blur_std_dev=100)
    ex.preprocesses = []
    for name in MODELS:
        model, pre = clip.load(name, device="cpu", jit=False)
        pre.transforms[0] = transforms.Resize((model.visual.input_resolution, model.visual.input_resolution), interpolation=transforms.InterpolationMode.BICUBIC)
        ex.preprocesses.append(pre); del model
    return ex


class Scenes(torch.utils.data.Dataset):
    def __init__(self, items, ex):
        self.items, self.ex = items, ex          # items: (path, [Box ...])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image
        path, boxes = self.items[i]
        with Image.open(path) as im:
            imgs, _ = self.ex.tensorize_inputs("x", im.convert("RGB"), boxes)   # per model: [crop_1..crop_n, blur_1..blur_n]
        return imgs


def cache(a) -> int:
    import clip
    from interpreter import Box
    scenes, _ = RG.load_scenes(); role = RG.dev_roles(scenes)
    sc = [s for s in scenes if role[s.image_id] in ("FIT", "CAL", "DEV") and len(s.referred) >= 2 and s.path][:a.limit or None]
    ex = prep_executor(); models = [clip.load(n, device="cuda", jit=False)[0].eval() for n in MODELS]
    items = [(s.path, [Box(x=o.box_xywh[0], y=o.box_xywh[1], w=o.box_xywh[2], h=o.box_xywh[3]) for o in s.referred + s.distractors]) for s in sc]
    dl = torch.utils.data.DataLoader(Scenes(items, ex), batch_size=None, num_workers=a.workers, prefetch_factor=4)
    feats, keys_o, t0 = [], [], time.time()
    with torch.no_grad():
        for k, (imgs, s) in enumerate(zip(dl, sc)):
            n = len(s.referred) + len(s.distractors); parts = []
            for m, x in zip(models, imgs):
                f = m.encode_image(x.cuda()).float(); f = f / f.norm(dim=-1, keepdim=True)
                parts += [f[:n], f[n:]]                                   # crop, blur
            feats.append((0.5 * torch.cat(parts, dim=1)).cpu()); keys_o += [(s.image_id, o.ann_id) for o in s.referred + s.distractors]
            if k % 1000 == 0:
                print(f"  {k}/{len(sc)} images {time.time() - t0:.0f}s", flush=True)
    t_img = time.time() - t0
    keys_t, texts = [], []
    for s in sc:
        for o in s.referred:
            for e in o.expressions:
                keys_t.append((s.image_id, o.ann_id, e["sent_id"])); texts.append("a photo of " + e["raw"].lower())
    ntrunc = sum(len(clip.tokenize([t], truncate=True)[0].nonzero()) >= 77 for t in texts)
    tv = []
    with torch.no_grad():
        for st in range(0, len(texts), 1024):
            tok = clip.tokenize(texts[st:st + 1024], truncate=True).cuda(); parts = []
            for m in models:
                f = m.encode_text(tok).float(); f = f / f.norm(dim=-1, keepdim=True); parts += [f, f]
            tv.append((0.5 * torch.cat(parts, dim=1)).cpu())
    img, txt = torch.cat(feats), torch.cat(tv); OUT_FEAT.mkdir(parents=True, exist_ok=True)
    torch.save({"image_feat_fp16": img.half(), "text_feat_fp16": txt.half(), "obj_keys": keys_o, "text_keys": keys_t, "roles": ["CAL", "DEV", "FIT"],
                "model": "ReCLIP isolation features: CLIP RN50x16 + ViT-B/32 (OpenAI) x crop / blur, concatenated, 1/2-scaled (cos = mean of 4 cosines)",
                "preprocess": "ReCLIP ClipExecutor.tensorize_inputs (square resize, blur std 100), raw COCO bbox", "text": "'a photo of ' + lower-cased expression, clip.tokenize(truncate=True)",
                "n_truncated_texts": int(ntrunc)}, OUT_FEAT / ("smoke_features.pt" if a.limit else "train_side_features.pt"))
    print(f"[cache] {len(keys_o)} boxes ({t_img:.0f}s), {len(keys_t)} expressions ({ntrunc} at the 77-token limit), {len(sc)} images -> {OUT_FEAT}"); return 0


def agree(a) -> int:
    D = torch.load(OUT_FEAT / "train_side_features.pt", weights_only=False)
    U = D["image_feat_fp16"].float(); V = D["text_feat_fp16"].float()
    oi = {tuple(k): i for i, k in enumerate(D["obj_keys"])}; ti = {tuple(k): i for i, k in enumerate(D["text_keys"])}
    scenes, _ = RG.load_scenes(resolve_paths=False); sc = {s.image_id: s for s in scenes}
    rows = [json.loads(l) for l in open("/home/infres/yinwang/CS_QMI/outputs/VL1_reclip/reclip_baseline_dev_referred.json") if l.strip()]
    same, hit_ours, hit_reclip, per_img, n_amb = 0, 0, 0, defaultdict(list), 0
    for r in rows:
        iid = int(re.search(r"COCO_train2014_(\d+)", r["file_name"]).group(1)); s = sc[iid]
        anns = sorted(o.ann_id for o in s.referred)                       # the ReCLIP input file lists the referred anns sorted by id
        tgt = anns[r["gold_index"][0]]
        sid = [e["sent_id"] for o in s.referred if o.ann_id == tgt for e in o.expressions if e["raw"] == r["text"]]
        if not sid:
            n_amb += 1; continue
        v = V[ti[(iid, tgt, sid[0])]]; scores = torch.stack([U[oi[(iid, x)]] @ v for x in anns]); p = int(scores.argmax())
        same += int(p == r["pred"]); hit_ours += int(p in r["gold_index"]); hit_reclip += int(r["pred"] in r["gold_index"]); per_img[iid].append(int(p in r["gold_index"]))
    n = len(rows) - n_amb
    res = {"n_queries": n, "unmatched_texts": n_amb, "argmax_agreement": same / n, "top1_query_exact_ours": hit_ours / n, "top1_query_exact_reclip": hit_reclip / n,
           "top1_image_macro_exact_ours": float(np.mean([np.mean(x) for x in per_img.values()])), "pass": same / n >= 0.99}
    json.dump(res, open(OUT_REP / "vl1_13_reclip_feature_agreement.json", "w"), indent=1)
    print(f"[agree] {n} DEV queries ({n_amb} unmatched): argmax agreement {100 * res['argmax_agreement']:.2f} %, Top-1 ours {100 * res['top1_query_exact_ours']:.2f} "
          f"vs ReCLIP {100 * res['top1_query_exact_reclip']:.2f}, macro ours {100 * res['top1_image_macro_exact_ours']:.2f} -> {'PASS' if res['pass'] else 'FAIL'}"); return 0


def main() -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache"); c.add_argument("--workers", type=int, default=14); c.add_argument("--limit", type=int, default=0, help="smoke: first N images, writes smoke_features.pt")
    sub.add_parser("agree")
    a = ap.parse_args(); return cache(a) if a.cmd == "cache" else agree(a)


if __name__ == "__main__":
    sys.exit(main())
