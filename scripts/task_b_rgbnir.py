"""Second-application task-level pre-check B-T2 (wave 2): closed-form J* vs MI / NMI as a rigid-registration energy on REAL cross-spectral pairs.

    python scripts/task_b_rgbnir.py --data /home/infres/yinwang/CS_QMI/data/rgb_nir/nirscene1 --out <prefix> [--n-images 60] [--smoke]
    python scripts/task_b_rgbnir.py --synthetic-smoke <dir> --out <prefix>        (code probe without the dataset: 3 constructed pairs in the
                                                                                  dataset's file layout, built from COCO val2017 images)

Data: EPFL IVRL RGB–NIR Scene dataset (Brown & Süsstrunk, CVPR 2011): 477 RGB / NIR pairs in 9 scene categories, registered by the authors
(SIFT + RANSAC similarity transform, resampled to a common frame), TIFF 1024×768.  Files `<category>/<name>_rgb.tiff` + `<name>_nir.tiff`.
Pairs used: `--n-images` (60) chosen by the seed, stratified over the categories (round-robin over per-category shuffles); identity = file stem.
Modality A = grey(RGB), modality B = NIR, both centre-cropped to a square and resized to 256 px, [0, 1]; truth = identity (the authors'
registration).  Then exactly the P52 protocol (`precheck_b2_registration.run_pair`): J* (49-d Fourier class, ridge 1e-3), MI, NMI on identical pixel
samples; translation surfaces ±24 px / step 2, rotation profiles ±30° / step 1; Nelder–Mead ≤ 150 evaluations from `--inits-per-radius` random
offsets per radius R ∈ {5, 10, 20, 30}; success = < 1 px and < 1°; divergence = |t| > 64 px.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from precheck_b2_registration import COCO, S, aggregate, load_grey, make_modality_b, progress_line, run_pair, write_outputs  # noqa: E402, Variant


def load_any_grey(path):
    """TIFF (8- or 16-bit, 1 or 3 channels) -> square centre crop -> 256 px float tensor in [0, 1]."""
    arr = np.asarray(Image.open(path)); arr = arr.astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(-1)
    arr = arr / (65535.0 if arr.max() > 255 else 255.0)
    h, w = arr.shape; s = min(h, w); arr = arr[(h - s) // 2: (h - s) // 2 + s, (w - s) // 2: (w - s) // 2 + s]
    im = Image.fromarray(arr, mode="F").resize((S, S), Image.BILINEAR)
    return torch.tensor(np.asarray(im, dtype=np.float32)).clamp(0, 1)


def list_pairs(root: Path):
    pairs = []
    for rgb in sorted(root.rglob("*_rgb.tif*")):
        nir = rgb.with_name(rgb.name.replace("_rgb", "_nir"))
        if nir.exists():
            cat = rgb.parent.name if rgb.parent != root else rgb.stem.split("_")[0]
            pairs.append({"id": str(rgb.relative_to(root)).replace("_rgb" + rgb.suffix, ""), "category": cat, "rgb": str(rgb), "nir": str(nir)})
    return pairs


def stratified_select(pairs, n, seed):
    rng = np.random.default_rng(seed); by_cat = {}
    for p in pairs:
        by_cat.setdefault(p["category"], []).append(p)
    for c in by_cat:
        by_cat[c] = [by_cat[c][i] for i in rng.permutation(len(by_cat[c]))]
    out, cats = [], sorted(by_cat)
    while len(out) < n and any(by_cat[c] for c in cats):
        for c in cats:
            if by_cat[c] and len(out) < n:
                out.append(by_cat[c].pop())
    return out


def build_synthetic(dir_: Path, n=3, seed=20260927):
    """Dataset-layout stand-in for the code probe: grey COCO val2017 image as '_rgb', its tent-map modality as '_nir' (8-bit TIFF)."""
    idx = json.load(open("/home/infres/yinwang/CS_QMI/data/coco_index/val2017_index.json"))["images"]
    ids = sorted(int(i) for i in idx); rng = np.random.default_rng(seed); rng.shuffle(ids); gen = torch.Generator().manual_seed(seed)
    for k, iid in enumerate(ids[:n]):
        a = load_grey(COCO / "val2017" / idx[str(iid)]["file"]); b = make_modality_b(a, gen); d = dir_ / f"synth{k % 2}"; d.mkdir(parents=True, exist_ok=True)
        Image.fromarray((a.numpy() * 255).astype(np.uint8)).save(d / f"img{iid}_rgb.tiff"); Image.fromarray((b.numpy() * 255).astype(np.uint8)).save(d / f"img{iid}_nir.tiff")
    return dir_


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/infres/yinwang/CS_QMI/data/rgb_nir/nirscene1"); ap.add_argument("--out", required=True)
    ap.add_argument("--n-images", type=int, default=60); ap.add_argument("--n-pairs", type=int, default=20000); ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--inits-per-radius", type=int, default=10); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--synthetic-smoke", default=None, help="directory: build 3 constructed pairs there and run the smoke on them")
    ap.add_argument("--patch-features", action="store_true", help="B-S2 variant: add the 8x8 block-mean channel to the J* features"); ap.add_argument("--coarse-to-fine", action="store_true", help="B-S2 variant: Nelder-Mead 64 -> 128 -> 256 px")
    a = ap.parse_args()
    VAR = Variant(patch=a.patch_features, ctf=a.coarse_to_fine)
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    if a.synthetic_smoke:
        a.data = str(build_synthetic(Path(a.synthetic_smoke))); a.smoke = True
    if a.smoke:
        a.n_images, a.inits_per_radius = 3, 2
    pairs = list_pairs(Path(a.data))
    if not pairs:
        print(f"no *_rgb/*_nir pairs under {a.data}"); return 2
    chosen = stratified_select(pairs, a.n_images, a.seed); ids = [p["id"] for p in chosen]
    list_sha = hashlib.sha256(",".join(ids).encode()).hexdigest()
    print(f"{len(pairs)} pairs found in {len(set(p['category'] for p in pairs))} categories; using {len(chosen)} (sha256 of id list {list_sha[:16]})", flush=True)
    rng = np.random.default_rng(a.seed); per_image = []; t0 = time.time()
    for n_i, p in enumerate(chosen):
        A_, B_ = load_any_grey(p["rgb"]), load_any_grey(p["nir"])
        rec = run_pair(A_, B_, p["id"], a.seed, a.n_pairs, a.inits_per_radius, rng, device, var=VAR); rec["category"] = p["category"]; per_image.append(rec)
        print(progress_line(n_i, len(chosen), p["id"], rec, t0), flush=True)
    agg = aggregate(per_image)
    settings = dict(vars(a)); settings.update({"id_list_sha256": list_sha, "n_pairs_available": len(pairs), "categories": sorted(set(p["category"] for p in pairs)),
                                              "per_category_used": {c: sum(p["category"] == c for p in chosen) for c in sorted(set(p["category"] for p in chosen))}})
    write_outputs(a.out, f"Task pre-check B-T2 [{VAR.tag or 'P52'}] — closed-form VCS J* vs histogram MI/NMI as a rigid-registration energy on {len(chosen)} real RGB–NIR pairs (EPFL IVRL scene dataset)",
                  f"{a.n_pairs} pixel pairs per evaluation, identical overlap mask and samples for all measures; translation grid ±24 px step 2 (θ = 0); rotation ±30° step 1 (t = 0); "
                  f"Nelder–Mead ≤ 150 evaluations from {a.inits_per_radius} random initial offsets per radius; success = < 1 px and < 1°; truth = the authors' registration.", settings, device, ids, agg, per_image)
    return 0


if __name__ == "__main__":
    sys.exit(main())
