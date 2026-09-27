"""Second-application pre-check A/C — step 1: identity splits of COCO-2017 captions and frozen CLIP features.

    python scripts/precheck_a_features.py --index-dir /home/infres/yinwang/CS_QMI/data/coco_index --out <dir>

Splits (fixed seed 20260927; identity = image id; a caption is only ever paired with its own image):
  source distribution = train2017 images with >= 1 instance annotation and NO 'animal' supercategory; target distribution = images WITH 'animal'.
  SRC-FIT 20 000 / SRC-CAL 5 000 / SRC-EVAL 5 000 images (disjoint, from the source pool); TGT-EVAL 5 000 images (from the target pool).
Frozen towers: open_clip ViT-B-32 laion2b_s34b_b79k (local cache, provenance file next to it).  Image features (512-d, pre-normalisation) for
every split image; text features for all captions of those images (5 each).  Nothing is trained.  Output: features/*.pt + splits.json (with
sha256 of the id lists) + manifest.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

COCO = Path("/projects/EEG-foundation-model/yinghao/FMCA-AV/coco")
CLIP_CACHE = "/home/infres/yinwang/CS_QMI/models/open_clip"


class ImgDS(Dataset):
    def __init__(self, files, preprocess):
        self.files, self.pp = files, preprocess

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.pp(Image.open(self.files[i]).convert("RGB")), i


def sha_ids(ids):
    return hashlib.sha256(",".join(str(i) for i in ids).encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", default="/home/infres/yinwang/CS_QMI/data/coco_index"); ap.add_argument("--out", required=True)
    ap.add_argument("--n-fit", type=int, default=20000); ap.add_argument("--n-cal", type=int, default=5000); ap.add_argument("--n-eval", type=int, default=5000); ap.add_argument("--n-tgt", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260927); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()
    import open_clip
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    out = Path(a.out); (out / "features").mkdir(parents=True, exist_ok=True)
    idx = json.load(open(Path(a.index_dir) / "train2017_index.json")); images, caps = idx["images"], idx["captions"]
    src = sorted(int(i) for i, v in images.items() if v["supercats"] and "animal" not in v["supercats"] and v["n_captions"] >= 5)
    tgt = sorted(int(i) for i, v in images.items() if "animal" in v["supercats"] and v["n_captions"] >= 5)
    rng = np.random.default_rng(a.seed); rng.shuffle(src); rng.shuffle(tgt)
    splits = {"SRC-FIT": src[: a.n_fit], "SRC-CAL": src[a.n_fit: a.n_fit + a.n_cal], "SRC-EVAL": src[a.n_fit + a.n_cal: a.n_fit + a.n_cal + a.n_eval], "TGT-EVAL": tgt[: a.n_tgt]}
    assert len(set().union(*map(set, splits.values()))) == sum(len(v) for v in splits.values()), "splits overlap"
    json.dump({"seed": a.seed, "rule": "source = train2017 with instances and no 'animal' supercategory; target = with 'animal'; >= 5 captions", "pool_sizes": {"source": len(src), "target": len(tgt)},
               "splits": {k: v for k, v in splits.items()}, "sha256": {k: sha_ids(v) for k, v in splits.items()}}, open(out / "splits.json", "w"))
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", cache_dir=CLIP_CACHE); tok = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval()
    manifest = {"towers": "open_clip ViT-B-32 laion2b_s34b_b79k (see models/open_clip/PROVENANCE_*.json)", "device": str(device), "splits_sha256": {k: sha_ids(v) for k, v in splits.items()}, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": {}}
    with torch.no_grad():
        for name, ids in splits.items():
            files = [str(COCO / "train2017" / images[str(i)]["file"]) for i in ids]
            dl = DataLoader(ImgDS(files, preprocess), batch_size=a.batch, num_workers=a.workers, shuffle=False)
            feats = torch.zeros(len(ids), 512); t0 = time.time()
            for x, ii in dl:
                feats[ii] = model.encode_image(x.to(device)).float().cpu()
            # captions: 5 per image, kept in order; text features [n, 5, 512]
            texts = [caps[str(i)][:5] for i in ids]; tf = torch.zeros(len(ids), 5, 512)
            flat = [t for ts in texts for t in ts]
            for s in range(0, len(flat), 1024):
                tf.view(-1, 512)[s: s + 1024] = model.encode_text(tok(flat[s: s + 1024]).to(device)).float().cpu()
            torch.save({"image_ids": torch.tensor(ids), "img": feats, "txt": tf, "captions": texts}, out / "features" / f"{name}.pt")
            manifest["files"][name] = {"n_images": len(ids), "n_captions": len(flat), "seconds": time.time() - t0}
            print(f"{name}: {len(ids)} images, {len(flat)} captions, {time.time() - t0:.0f}s", flush=True)
    json.dump(manifest, open(out / "manifest.json", "w"), indent=2); print("done ->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
