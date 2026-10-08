"""P139 (v7 V7-IMAGENET-PIPELINE): checks and throughput for the CPU-crop + GPU-augmentation ImageNet pipeline.  No pretraining runs, no
checkpoints; every training-like loop is <= 60 optimizer steps per cell.
    python scripts/p139_imagenet_pipeline.py loader --out reports/P139/loader.json [--workers-list 4 8 16 32]      (CPU job)
    python scripts/p139_imagenet_pipeline.py check  --out reports/P139/check_<gpu>.json                          (1 GPU)
    torchrun --nproc_per_node N scripts/p139_imagenet_pipeline.py ddp --out reports/P139/ddp_<gpu>_N.json        (N GPUs, one node)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import socket
import sys
import time

import torch
import torch.distributed as dist
from torch.nn import functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from vcs_ssl.imagenet import ImageNetMultiView, class_index, dist_allview_tokens_loss, dist_simclr_multiview_loss, read_manifest  # noqa: E402
from vcs_ssl.imagenet_gpu_aug import (GpuViewAug, ImageNetCropViews, ImageNetJpegBytes, apply_views, collate_bytes,  # noqa: E402
                                      gpu_decode_crop, params_to, pil_reference_view, rrc_params, sample_params, tensor_reference_view)

MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)


def _pool(n, seed):
    tr = read_manifest("train"); return tr, class_index(tr), random.Random(seed).sample(tr, n)


# ------------------------------------------------------------------------------------------------------------ loader (CPU)
def loader(a):
    tr, cls, pool = _pool(240000, 11); cur = 0
    res = {"host": socket.gethostname(), "cpus_visible": len(os.sched_getaffinity(0)), "batch_images": a.batch, "cells": [],
           "stage": "CPU: decode once + per-view RandomResizedCrop(224) + pil_to_tensor (uint8); colour/grey/blur/normalise on the GPU",
           "subsets": "disjoint per cell (cold reads)"}
    for views in (2, 4):
        for w in a.workers_list:
            need = (a.warmup + a.batches + 4) * a.batch; rows = pool[cur:cur + need]; cur += need
            dl = torch.utils.data.DataLoader(ImageNetCropViews(views=views, rows=rows, classes=cls), batch_size=a.batch, shuffle=True,
                                             num_workers=w, drop_last=True, prefetch_factor=4, persistent_workers=False)
            it = iter(dl)
            for _ in range(a.warmup): next(it)
            t0 = time.time()
            for _ in range(a.batches): next(it)
            ips = a.batches * a.batch / (time.time() - t0)
            cell = {"views": views, "workers": w, "images_per_s": round(ips, 1), "crops_per_s": round(ips * views, 1)}
            res["cells"].append(cell); print(cell, flush=True); del it, dl
    json.dump(res, open(a.out, "w"), indent=1)


# ------------------------------------------------------------------------------------------------------------ check (1 GPU)
def _summ(x01: torch.Tensor) -> torch.Tensor:
    """Per-image summaries of float [N, 3, H, W] in [0, 1]: channel means, global std, saturation proxy, Laplacian variance (sharpness)."""
    m = x01.mean(dim=(2, 3)); sd = x01.std(dim=(1, 2, 3))
    sat = (x01.max(1).values - x01.min(1).values).mean(dim=(1, 2))
    g = (0.2989 * x01[:, 0] + 0.587 * x01[:, 1] + 0.114 * x01[:, 2]).unsqueeze(1)
    lap = F.conv2d(g, torch.tensor([[0., 1, 0], [1, -4, 1], [0, 1, 0]], device=x01.device).view(1, 1, 3, 3)).var(dim=(1, 2, 3))
    return torch.cat((m, sd[:, None], sat[:, None], lap[:, None]), 1)


def _ks(a, b):
    a, _ = torch.sort(a); b, _ = torch.sort(b); allv = torch.cat((a, b)).sort().values
    ca = torch.searchsorted(a, allv, right=True).float() / len(a); cb = torch.searchsorted(b, allv, right=True).float() / len(b)
    return float((ca - cb).abs().max())


def check(a):
    dev = torch.device(a.device if a.device == "cpu" else "cuda", 0) if a.device != "cpu" else torch.device("cpu")
    out = {"host": socket.gethostname(), "device": str(dev), "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None}
    tr, cls, pool = _pool(6000, 21)
    # (a) fixed parameters on real crops: GPU vectorised vs torchvision tensor reference vs P132 PIL ops
    ds = ImageNetCropViews(views=2, rows=pool[:300], classes=cls)
    crops = torch.cat([ds[i][0] for i in range(len(ds))])                       # [600, 3, 224, 224] uint8
    p = sample_params(len(crops), torch.Generator().manual_seed(5))
    y = apply_views(crops.to(dev), params_to(p, dev), normalize=False).cpu()
    d_tens, d_pil = [], []
    for i in range(len(crops)):
        d_tens.append(float((y[i] - tensor_reference_view(crops[i], p, i)).abs().max()) * 255)
        d_pil.append(float((y[i] - pil_reference_view(crops[i], p, i)).abs().mean()) * 255)
    dt, dp = torch.tensor(d_tens), torch.tensor(d_pil)
    out["fixed_params"] = {"n_crops": len(crops),
                           "gpu_vs_torchvision_tensor_max_abs_levels": {"max": round(float(dt.max()), 4), "mean": round(float(dt.mean()), 5)},
                           "gpu_vs_P132_PIL_mean_abs_levels": {"mean": round(float(dp.mean()), 3), "p95": round(float(dp.quantile(0.95)), 3),
                                                               "max": round(float(dp.max()), 3)}}
    for key in ("jitter", "gray", "blur"):
        on = p[key]
        out["fixed_params"][f"PIL_diff_by_{key}"] = {"on": round(float(dp[on].mean()), 3), "off": round(float(dp[~on].mean()), 3)}
    print(out["fixed_params"], flush=True)
    # (b) random parameters: P132 full PIL pipeline vs CPU-crop + GPU stage, output-summary distributions (same images, 2 views each)
    rows = pool[300:2300]
    pil_ds = ImageNetMultiView(views=2, blur_p=0.5, rows=rows, classes=cls)
    pil = torch.cat([torch.stack(pil_ds[i][0]) for i in range(len(rows))])     # normalised float
    pil01 = pil * STD + MEAN
    crop_ds = ImageNetCropViews(views=2, rows=rows, classes=cls)
    aug = GpuViewAug(seed=7)
    gpu01 = []
    for lo in range(0, len(rows), 250):
        u8 = torch.cat([crop_ds[i][0] for i in range(lo, min(lo + 250, len(rows)))]).to(dev)
        gpu01.append((aug(u8) * STD.to(dev) + MEAN.to(dev)).cpu())
    gpu01 = torch.cat(gpu01)
    sp, sg = _summ(pil01.to(dev)).cpu(), _summ(gpu01.to(dev)).cpu()
    names = ["mean_R", "mean_G", "mean_B", "std", "saturation", "laplacian_var"]
    out["random_params"] = {"n_views_each": len(sp), "summaries": {n: {"P132_PIL_mean": round(float(sp[:, k].mean()), 5), "GPU_mean": round(float(sg[:, k].mean()), 5),
                                                                         "P132_PIL_sd": round(float(sp[:, k].std()), 5), "GPU_sd": round(float(sg[:, k].std()), 5),
                                                                         "KS": round(_ks(sp[:, k], sg[:, k]), 4)} for k, n in enumerate(names)},
                            "KS_crit_alpha_0.01": round(1.63 * math.sqrt(2 / len(sp)), 4)}
    print(out["random_params"], flush=True)
    if dev.type == "cpu":                                                   # (a)-(b) only; GPU-specific parts need the GPU job
        json.dump(out, open(a.out, "w"), indent=1); return
    # (c) GPU-stage cost (2 views x 128 images = 256 crops)
    u8 = torch.cat([crop_ds[i][0] for i in range(128)]).to(dev)
    for _ in range(3): aug(u8)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(20): aug(u8)
    torch.cuda.synchronize(); ms = (time.time() - t0) / 20 * 1000
    out["gpu_stage_ms_per_256_crops"] = round(ms, 2)
    # (d) bf16 precision: R50 + projector on one real batch (b 64, V 2): fp32 / bf16 encoder + fp32 loss / bf16 incl. loss
    out["bf16"] = _bf16_check(dev, crop_ds, aug)
    print(out["bf16"], flush=True)
    out["gpu_decode"] = _decode_check(dev, pool[2300:2500], cls)
    print(out["gpu_decode"], flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


def _decode_check(dev, rows, cls):
    """nvjpeg vs PIL decode (full image) and torch vs PIL bilinear-antialias crop+resize with identical crop boxes; GPU decode+crop speed."""
    from PIL import Image
    from torchvision.io import ImageReadMode, decode_jpeg, read_file
    from torchvision.transforms import functional as TF
    from vcs_ssl.imagenet import IMAGENET_ROOT
    dec, rsz, fails = [], [], 0
    g = torch.Generator().manual_seed(3)
    for rel, _ in rows:
        path = os.path.join(IMAGENET_ROOT, rel)
        with Image.open(path) as im:
            pil = im.convert("RGB"); ref = TF.pil_to_tensor(pil)
        try:
            gimg = decode_jpeg(read_file(path), mode=ImageReadMode.RGB, device=dev).cpu()
        except RuntimeError:
            fails += 1; continue
        if gimg.shape != ref.shape:
            fails += 1; continue
        dec.append(float((gimg.float() - ref.float()).abs().mean()))
        i, j, ch, cw = rrc_params(ref.shape[1], ref.shape[2], g)
        p_crop = TF.pil_to_tensor(TF.resized_crop(pil, i, j, ch, cw, [224, 224], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)).float()
        t_crop = F.interpolate(ref[None, :, i:i + ch, j:j + cw].float(), size=(224, 224), mode="bilinear", antialias=True, align_corners=False)[0].round().clamp(0, 255)
        rsz.append(float((t_crop - p_crop).abs().mean()))
    ds = ImageNetJpegBytes(rows=rows, classes=cls)
    data = [ds[i][0] for i in range(128)]
    for _ in range(2): gpu_decode_crop(data, 2, dev, g)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(5): gpu_decode_crop(data, 2, dev, g)
    torch.cuda.synchronize(); ms = (time.time() - t0) / 5 * 1000
    dt, rt = torch.tensor(dec), torch.tensor(rsz)
    return {"n": len(dec), "gpu_decode_failures": fails, "nvjpeg_vs_PIL_decode_mean_abs_levels": {"mean": round(float(dt.mean()), 3), "max": round(float(dt.max()), 3)},
            "torch_vs_PIL_resized_crop_mean_abs_levels": {"mean": round(float(rt.mean()), 3), "max": round(float(rt.max()), 3)},
            "gpu_decode_crop_ms_per_128_images_2_views": round(ms, 1)}


def _model(dev, syncbn=False):
    from torch import nn
    from torchvision.models import resnet50
    torch.manual_seed(0)
    enc = resnet50(weights=None); enc.fc = nn.Identity()
    proj = nn.Sequential(nn.Linear(2048, 2048, bias=False), nn.BatchNorm1d(2048), nn.ReLU(inplace=True), nn.Linear(2048, 128))
    m = nn.Sequential(enc, proj)
    if syncbn:
        m = nn.SyncBatchNorm.convert_sync_batchnorm(m)
    return m.to(dev).to(memory_format=torch.channels_last)


def _lossfn(method, vz):
    if method == "simclr":
        return dist_simclr_multiview_loss(vz, 0.2)["loss"]
    return dist_allview_tokens_loss(vz, 2.0, -1.0, objective=method)["loss"]


def _bf16_check(dev, crop_ds, aug):
    u8 = torch.cat([crop_ds[i][0] for i in range(1000, 1064)]).to(dev)          # 64 images x 2 views (view-major after reshape below)
    x = aug(u8).view(64, 2, 3, 224, 224).transpose(0, 1).reshape(128, 3, 224, 224).contiguous(memory_format=torch.channels_last)
    res = {}
    for method in ("vcs", "js", "simclr"):
        grads, losses = {}, {}
        for mode in ("fp32", "bf16_enc_fp32_loss", "bf16_all"):
            m = _model(dev); m.train()
            if mode == "fp32":
                torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
                z = F.normalize(m(x), dim=-1); loss = _lossfn(method, list(z.split(64)))
            elif mode == "bf16_enc_fp32_loss":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    r = m(x)
                z = F.normalize(r.float(), dim=-1); loss = _lossfn(method, list(z.split(64)))
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z = F.normalize(m(x), dim=-1); loss = _lossfn(method, list(z.split(64)))
            loss.backward()
            grads[mode] = torch.cat([q.grad.flatten().float() for q in m.parameters() if q.grad is not None]); losses[mode] = float(loss)
            torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
            del m
        g0 = grads["fp32"]
        res[method] = {mode: {"loss": round(losses[mode], 6), "loss_abs_diff_vs_fp32": round(abs(losses[mode] - losses["fp32"]), 7),
                              "grad_cos_vs_fp32": round(float(F.cosine_similarity(grads[mode], g0, dim=0)), 6),
                              "grad_rel_err_vs_fp32": round(float((grads[mode] - g0).norm() / g0.norm()), 5)} for mode in grads}
    return res


# ------------------------------------------------------------------------------------------------------------ ddp (torchrun)
def ddp(a):
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size(); local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local); dev = torch.device("cuda", local)
    torch.backends.cudnn.benchmark = True; torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    tr, cls, pool = _pool(a.images, 31)
    cpus = len(os.sched_getaffinity(0)); workers = max(2, a.workers if a.workers > 0 else cpus // world - 2)
    out = {"host": socket.gethostname(), "gpu": torch.cuda.get_device_name(dev), "world": world, "per_gpu_batch_images": a.batch, "views": a.views,
           "global_batch_images": a.batch * world, "workers_per_rank": workers, "cpus_visible_rank0": cpus, "syncbn": True,
           "precision": "bf16 autocast encoder+projector, fp32 normalise + pair loss, channels_last, TF32", "cells": []}
    cells = [(m, "cpu") for m in a.methods] + [(m, "gpu") for m in a.gpu_decode_methods]
    for method, decode in cells:
        if decode == "cpu":
            ds = ImageNetCropViews(views=a.views, rows=pool, classes=cls); coll, nw = None, workers
        else:
            ds = ImageNetJpegBytes(rows=pool, classes=cls); coll, nw = collate_bytes, max(2, min(6, workers))
        sampler = torch.utils.data.DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, seed=0, drop_last=True); sampler.set_epoch(0)
        dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, sampler=sampler, num_workers=nw, drop_last=True, pin_memory=decode == "cpu",
                                         prefetch_factor=4, persistent_workers=False, collate_fn=coll)
        gdec = torch.Generator().manual_seed(2000 + rank)
        model = torch.nn.parallel.DistributedDataParallel(_model(dev, syncbn=world > 1), device_ids=[local])
        opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-6)
        aug = GpuViewAug(seed=1000 + rank)
        torch.cuda.reset_peak_memory_stats()
        it = iter(dl); times, waits, aug_t, uid_ok, finite = [], [], [], True, True
        for step in range(a.warmup + a.steps):
            t0 = time.time()
            v, _, uid = next(it)
            uid = uid.to(dev)
            if decode == "cpu":
                v = v.to(dev, non_blocking=True)
            torch.cuda.synchronize(); t1 = time.time()
            if decode == "gpu":
                v = gpu_decode_crop(v, a.views, dev, gdec)
            b = v.shape[0]
            x = aug(v.view(b * a.views, 3, 224, 224))                                       # image-major (i, view)
            x = x.view(b, a.views, 3, 224, 224).transpose(0, 1).reshape(a.views * b, 3, 224, 224).contiguous(memory_format=torch.channels_last)
            torch.cuda.synchronize(); t2 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                r = model(x)
            z = F.normalize(r.float(), dim=-1); vz = list(z.split(b))
            loss = _lossfn(method, vz)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); torch.cuda.synchronize(); t3 = time.time()
            # global image ids: unique within the global batch; identical for every view slot (positives match across ranks)
            g = [torch.empty_like(uid) for _ in range(world)]; dist.all_gather(g, uid); gu = torch.cat(g)
            uid_ok &= bool(gu.unique().numel() == gu.numel())
            finite &= bool(torch.isfinite(loss).item())
            if step >= a.warmup:
                times.append(t3 - t0); waits.append(t1 - t0); aug_t.append(t2 - t1)
        med = lambda xs: sorted(xs)[len(xs) // 2]  # noqa: E731
        st = med(times); ips_rank = a.batch / st
        tot = torch.tensor([ips_rank], device=dev); dist.all_reduce(tot)
        mem = torch.tensor([torch.cuda.max_memory_allocated() / 1e9], device=dev); dist.all_reduce(mem, op=dist.ReduceOp.MAX)
        cell = {"method": method, "decode": decode, "workers": nw, "median_step_s": round(st, 4), "median_data_wait_s": round(med(waits), 4), "median_gpu_aug_s": round(med(aug_t), 4),
                "images_per_s_total": round(float(tot), 1), "images_per_s_per_gpu": round(float(tot) / world, 1),
                "peak_allocated_gb_max_rank": round(float(mem), 2), "uid_unique_all_steps": uid_ok, "finite_loss": finite,
                "gpu_hours_per_epoch": round(1281167 / float(tot) * world / 3600, 3), "wall_hours_per_epoch": round(1281167 / float(tot) / 3600, 3)}
        out["cells"].append(cell)
        if rank == 0: print(cell, flush=True)
        del it, dl, model, opt; torch.cuda.empty_cache()
    # communication micro-benchmarks: gradient all-reduce of the R50+projector size (fp32) and the token all-gather
    nparam = sum(q.numel() for q in _model(dev).parameters())
    buf = torch.randn(nparam, device=dev); tok = torch.randn(a.batch * a.views, 128, device=dev)
    for _ in range(3): dist.all_reduce(buf)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(10): dist.all_reduce(buf)
    torch.cuda.synchronize(); ar = (time.time() - t0) / 10
    gl = [torch.empty_like(tok) for _ in range(world)]
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(20): dist.all_gather(gl, tok)
    torch.cuda.synchronize(); ag = (time.time() - t0) / 20
    out["comm"] = {"params": nparam, "grad_allreduce_fp32_s": round(ar, 4), "token_allgather_s": round(ag, 5)}
    if rank == 0:
        print(out["comm"], flush=True); json.dump(out, open(a.out, "w"), indent=1)
    dist.destroy_process_group()


def bf16(a):
    """bf16 vs fp32 from a briefly fp32-trained snapshot (at random init all z are nearly parallel, so pair-logit differences sit at bf16
    rounding scale and the init-time comparison is ill-conditioned).  Baselines: fp32 repeat (cuDNN non-determinism) and fp32 with input
    noise at bf16 rounding scale.  Also reports the z spread (mean off-diagonal cosine) at init and at the snapshot."""
    import copy
    dev = torch.device("cuda", 0)
    tr, cls, pool = _pool(4000, 41)
    crop_ds = ImageNetCropViews(views=2, rows=pool, classes=cls); aug = GpuViewAug(seed=9)
    dl = torch.utils.data.DataLoader(crop_ds, batch_size=64, shuffle=True, num_workers=14, drop_last=True, prefetch_factor=4)
    def batch(v):
        b = v.shape[0]
        x = aug(v.to(dev).view(b * 2, 3, 224, 224)).view(b, 2, 3, 224, 224).transpose(0, 1).reshape(2 * b, 3, 224, 224)
        return x.contiguous(memory_format=torch.channels_last)
    def spread(m, x):
        with torch.no_grad():
            z = F.normalize(m(x).float(), dim=-1); c = z @ z.T
            return float((c.sum() - c.diagonal().sum()) / (c.numel() - c.shape[0]))
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    m = _model(dev); m.train(); opt = torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-6)
    it = iter(dl); xfix = batch(next(it)[0]); out = {"steps_fp32_pretrain": a.steps, "z_mean_offdiag_cos_init": round(spread(m, xfix), 4)}
    for step in range(a.steps):
        try: v = next(it)[0]
        except StopIteration: it = iter(dl); v = next(it)[0]
        z = F.normalize(m(batch(v)), dim=-1); loss = _lossfn("vcs", list(z.split(64)))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    out["z_mean_offdiag_cos_snapshot"] = round(spread(m, xfix), 4); snap = copy.deepcopy(m.state_dict())
    res = {}
    for method in ("vcs", "js", "simclr"):
        grads, losses = {}, {}
        for mode in ("fp32", "fp32_repeat", "fp32_input_noise", "bf16_enc_fp32_loss", "bf16_all"):
            mm = _model(dev); mm.load_state_dict(snap); mm.train()
            fp32 = mode.startswith("fp32")
            torch.backends.cuda.matmul.allow_tf32 = not fp32; torch.backends.cudnn.allow_tf32 = not fp32
            xin = xfix + (torch.randn_like(xfix) * xfix.abs() * 2 ** -9 if mode == "fp32_input_noise" else 0)
            if fp32:
                z = F.normalize(mm(xin), dim=-1); loss = _lossfn(method, list(z.split(64)))
            elif mode == "bf16_enc_fp32_loss":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    r = mm(xin)
                z = F.normalize(r.float(), dim=-1); loss = _lossfn(method, list(z.split(64)))
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z = F.normalize(mm(xin), dim=-1); loss = _lossfn(method, list(z.split(64)))
            loss.backward()
            grads[mode] = torch.cat([q.grad.flatten().float() for q in mm.parameters() if q.grad is not None]); losses[mode] = float(loss)
            del mm
        g0 = grads["fp32"]
        res[method] = {mode: {"loss": round(losses[mode], 6), "loss_abs_diff_vs_fp32": round(abs(losses[mode] - losses["fp32"]), 7),
                              "grad_cos_vs_fp32": round(float(F.cosine_similarity(grads[mode], g0, dim=0)), 5),
                              "grad_rel_err_vs_fp32": round(float((grads[mode] - g0).norm() / g0.norm()), 5)} for mode in grads}
        print(method, res[method], flush=True)
    out["methods"] = res
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    json.dump(out, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("loader"); p.add_argument("--out", required=True); p.add_argument("--workers-list", type=int, nargs="+", default=[4, 8, 16, 32])
    p.add_argument("--batch", type=int, default=64); p.add_argument("--warmup", type=int, default=6); p.add_argument("--batches", type=int, default=40)
    p = sub.add_parser("check"); p.add_argument("--out", required=True); p.add_argument("--device", default="cuda")
    p = sub.add_parser("bf16"); p.add_argument("--out", required=True); p.add_argument("--steps", type=int, default=200)
    p = sub.add_parser("ddp"); p.add_argument("--out", required=True); p.add_argument("--batch", type=int, default=128); p.add_argument("--views", type=int, default=2)
    p.add_argument("--methods", nargs="+", default=["vcs", "simclr"]); p.add_argument("--gpu-decode-methods", nargs="*", default=["vcs"]); p.add_argument("--warmup", type=int, default=10); p.add_argument("--steps", type=int, default=30)
    p.add_argument("--workers", type=int, default=0); p.add_argument("--images", type=int, default=120000)
    a = ap.parse_args()
    {"loader": loader, "check": check, "ddp": ddp, "bf16": bf16}[a.cmd](a)


if __name__ == "__main__":
    main()
