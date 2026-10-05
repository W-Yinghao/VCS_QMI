"""P132 ImageNet-1k preflight (v6 §8.3): data check, data-loader throughput, single-GPU training throughput.  No full training, no checkpoints.
    python scripts/p132_imagenet_preflight.py datacheck --out reports/P132/datacheck.json [--workers 32] [--sample-per-class 50]
    python scripts/p132_imagenet_preflight.py loader    --out reports/P132/loader.json    [--workers-list 4 8 16 32]
    python scripts/p132_imagenet_preflight.py gpu       --out reports/P132/gpu_<partition>.json [--workers 14]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import socket
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from vcs_ssl.imagenet import (IMAGENET_ROOT, ImageNetMultiView, class_index, dist_allview_tokens_loss,  # noqa: E402
                              dist_simclr_multiview_loss, read_manifest)


def _check_one(rel: str) -> tuple[str, str | None]:
    from PIL import Image
    try:
        with Image.open(os.path.join(IMAGENET_ROOT, rel)) as im:
            im.convert("RGB").load()
        return rel, None
    except Exception as e:  # noqa: BLE001
        return rel, f"{type(e).__name__}: {e}"


def datacheck(a):
    t0 = time.time(); out = {"host": socket.gethostname(), "root": IMAGENET_ROOT}
    tr, va = read_manifest("train"), read_manifest("val")
    cls = class_index(tr)
    out["manifest"] = {"train": len(tr), "val": len(va), "classes_train": len(cls), "classes_val": len({w for _, w in va}),
                       "val_wnids_subset_of_train": set(w for _, w in va) <= set(cls)}
    vc = Counter(w for _, w in va); tc = Counter(w for _, w in tr)
    out["per_class"] = {"val_min": min(vc.values()), "val_max": max(vc.values()), "train_min": min(tc.values()), "train_max": max(tc.values())}
    # filesystem vs manifest (directory listing of all 1000 train dirs + the flat val dir)
    fs_train = 0; missing_dirs = []
    for w in sorted(cls):
        d = os.path.join(IMAGENET_ROOT, "train", w)
        if not os.path.isdir(d): missing_dirs.append(w); continue
        fs_train += sum(1 for e in os.scandir(d) if e.name.endswith(".JPEG"))
    fs_val = sum(1 for e in os.scandir(os.path.join(IMAGENET_ROOT, "val")) if e.name.endswith(".JPEG"))
    out["filesystem"] = {"train_jpeg": fs_train, "val_jpeg": fs_val, "missing_train_dirs": missing_dirs}
    # stat of every manifest path (exists, size > 0)
    def st(rel):
        try: return rel, os.stat(os.path.join(IMAGENET_ROOT, rel)).st_size
        except OSError: return rel, -1
    with ThreadPoolExecutor(a.workers * 2) as ex:
        sizes = list(ex.map(st, [r for r, _ in tr + va], chunksize=2000))
    bad_stat = [r for r, s in sizes if s <= 0]
    out["stat_all"] = {"n": len(sizes), "missing_or_empty": len(bad_stat), "examples": bad_stat[:20],
                       "total_gb": round(sum(s for _, s in sizes if s > 0) / 1e9, 1)}
    # full decode of a stratified train sample + all val
    rng = random.Random(0); by = {}
    for r, w in tr: by.setdefault(w, []).append(r)
    sample = [r for w in sorted(by) for r in rng.sample(by[w], min(a.sample_per_class, len(by[w])))] + [r for r, _ in va]
    t1 = time.time()
    from multiprocessing import Pool
    with Pool(a.workers) as pool:
        res = pool.map(_check_one, sample, chunksize=200)
    dt = time.time() - t1
    bad = [(r, e) for r, e in res if e]
    out["decode_sample"] = {"n": len(sample), "train_per_class": a.sample_per_class, "val_all": len(va), "unreadable": len(bad), "examples": bad[:20],
                            "seconds": round(dt, 1), "images_per_s_decode_only": round(len(sample) / dt, 1), "workers": a.workers}
    out["seconds_total"] = round(time.time() - t0, 1)
    json.dump(out, open(a.out, "w"), indent=1); print(json.dumps(out, indent=1))


def loader(a):
    tr = read_manifest("train"); cls = class_index(tr)
    pool = random.Random(1).sample(tr, 200000); cur = 0   # every cell reads a DISJOINT subset (cold page cache; v1 reused one subset)
    res = {"host": socket.gethostname(), "cpus_visible": len(os.sched_getaffinity(0)), "batch_images": a.batch, "cells": [], "subsets": "disjoint per cell (cold reads)"}
    for blur in (0.5, 0.0):
        for views in (2, 4):
            for w in a.workers_list:
                if blur == 0.0 and w != max(a.workers_list): continue
                need = (a.warmup + a.batches + 4) * a.batch; rows = pool[cur:cur + need]; cur += need
                ds = ImageNetMultiView(views=views, blur_p=blur, rows=rows, classes=cls)
                dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=w, drop_last=True, persistent_workers=False,
                                                 prefetch_factor=4 if w > 0 else None)
                it = iter(dl)
                for _ in range(a.warmup): next(it)
                t0 = time.time()
                for _ in range(a.batches): next(it)
                dt = time.time() - t0; ips = a.batches * a.batch / dt
                cell = {"views": views, "blur_p": blur, "workers": w, "images_per_s": round(ips, 1), "crops_per_s": round(ips * views, 1)}
                res["cells"].append(cell); print(cell, flush=True)
                del it, dl
    json.dump(res, open(a.out, "w"), indent=1)


def _model(device):
    from torch import nn
    from torchvision.models import resnet50
    enc = resnet50(weights=None); enc.fc = nn.Identity()
    proj = nn.Sequential(nn.Linear(2048, 2048, bias=False), nn.BatchNorm1d(2048), nn.ReLU(inplace=True), nn.Linear(2048, 128))
    return enc.to(device).to(memory_format=torch.channels_last), proj.to(device)


def gpu(a):
    dev = torch.device("cuda", 0); torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    out = {"host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0), "cpus_visible": len(os.sched_getaffinity(0)),
           "precision": "bf16 autocast + channels_last + TF32", "projector": "2048-2048(BN,ReLU)-128", "cells": [], "steps_total": 0}
    cells = [("ap3", 2, 128, "synthetic"), ("ap3", 4, 64, "synthetic"), ("simclr", 2, 128, "synthetic"), ("simclr", 4, 64, "synthetic"),
             ("ap3", 2, 128, "real"), ("ap3", 4, 64, "real")]
    tr = read_manifest("train"); cls = class_index(tr); rows = random.Random(2).sample(tr, 30000)
    for method, V, b, src in cells:
        enc, proj = _model(dev)
        opt = torch.optim.AdamW(list(enc.parameters()) + list(proj.parameters()), lr=1e-3, weight_decay=1e-4)
        torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
        if src == "real":
            ds = ImageNetMultiView(views=V, blur_p=0.5, rows=rows, classes=cls)
            it = iter(torch.utils.data.DataLoader(ds, batch_size=b, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True,
                                                  prefetch_factor=4, persistent_workers=False))
            nxt = lambda: [v.to(dev, non_blocking=True).to(memory_format=torch.channels_last) for v in next(it)[0]]  # noqa: E731
        else:
            fixed = [torch.randn(b, 3, 224, 224, device=dev).to(memory_format=torch.channels_last) for _ in range(V)]
            nxt = lambda: fixed  # noqa: E731
        times, data_t = [], []
        try:
            for step in range(a.warmup + a.steps):
                t0 = time.time(); views = nxt(); torch.cuda.synchronize(); t1 = time.time()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    h = enc(torch.cat(views, 0)); z = torch.nn.functional.normalize(proj(h).float(), dim=-1)
                vz = list(z.split(b, 0))
                loss = (dist_allview_tokens_loss(vz, 2.0, -1.0) if method == "ap3" else dist_simclr_multiview_loss(vz, 0.2))["loss"]
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); torch.cuda.synchronize(); t2 = time.time()
                if step >= a.warmup: times.append(t2 - t0); data_t.append(t1 - t0)
                out["steps_total"] += 1
            st = sorted(times)[len(times) // 2]; ips = b / st
            cell = {"method": method, "views": V, "batch_images": b, "source": src, "median_step_s": round(st, 4), "median_data_wait_s": round(sorted(data_t)[len(data_t) // 2], 4),
                    "images_per_s": round(ips, 1), "crops_per_s": round(ips * V, 1), "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                    "gpu_hours_per_epoch_1gpu": round(1281167 / ips / 3600, 2), "finite_loss": bool(torch.isfinite(loss).item())}
        except torch.cuda.OutOfMemoryError:
            cell = {"method": method, "views": V, "batch_images": b, "source": src, "status": "OOM"}
        out["cells"].append(cell); print(cell, flush=True)
        del enc, proj, opt; torch.cuda.empty_cache()
    json.dump(out, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("datacheck"); p.add_argument("--out", required=True); p.add_argument("--workers", type=int, default=32); p.add_argument("--sample-per-class", type=int, default=50)
    p = sub.add_parser("loader"); p.add_argument("--out", required=True); p.add_argument("--workers-list", type=int, nargs="+", default=[4, 8, 16, 32])
    p.add_argument("--batch", type=int, default=64); p.add_argument("--warmup", type=int, default=6); p.add_argument("--batches", type=int, default=40)
    p = sub.add_parser("gpu"); p.add_argument("--out", required=True); p.add_argument("--workers", type=int, default=14)
    p.add_argument("--warmup", type=int, default=10); p.add_argument("--steps", type=int, default=40)
    a = ap.parse_args()
    {"datacheck": datacheck, "loader": loader, "gpu": gpu}[a.cmd](a)


if __name__ == "__main__":
    main()
