"""R4-VL1 resource measurement (package docs/04 §6: "先实测显存、吞吐、有效图片 batch、候选数与 crop 曝光").  Not training: a few forward/backward
steps of the proposal-aware objective on SigLIP 2 B/16 with ALL detector proposals of each image as candidates, to measure peak GPU memory and
seconds per step as a function of images per step, with and without activation checkpointing, and in the per-image-accumulation form.
Candidates come from the saved R4-VL0 meta (DEV proposals; the FIT proposal count is reported separately once extracted).  Writes
reports/R4/R4_VL1_measurement.json.
    python scripts/r4_vl1_measure.py --backbone siglip2_b16 --images 1 2 4 8 [--grad-ckpt]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vl3_finetune import CLIP_ARCHS, SIGLIP_REPOS, WD, Backbone, Critic  # noqa: E402
from vcs_vl.pairlaw import padded_batch_loss  # noqa: E402

META = Path("/home/infres/yinwang/CS_QMI/outputs/R4_VL0"); OUT = Path(__file__).resolve().parents[1] / "reports" / "R4" / "R4_VL1_measurement.json"


def proposal_laws(meta_img, iou_thr=0.5):
    """p_G(r, w) = pi(w) A(r, w) / sum_s A(s, w); q_G(r, w) = pi(w) nu(r) with uniform nu; queries without a valid proposal get zero laws (excluded)."""
    from vl1_22_tableC import iou_xyxy
    P, T = meta_img["prop_boxes"], meta_img["ref_boxes"]; n_p, n_q = len(P), len(meta_img["target"])
    A = np.stack([iou_xyxy(T[t], P) >= iou_thr for t in meta_img["target"]]).T.astype(np.float32) if n_p else np.zeros((0, n_q), np.float32)   # [n_p, n_q]
    ok = A.sum(0) > 0; pi = ok / max(1, ok.sum())
    p = A / np.maximum(A.sum(0, keepdims=True), 1) * pi[None, :]; q = np.full((n_p, n_q), 1.0 / max(1, n_p), np.float32) * pi[None, :]
    return p.astype(np.float32), q.astype(np.float32), int(ok.sum()), int(n_q - ok.sum())


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--backbone", default="siglip2_b16", choices=list(CLIP_ARCHS) + list(SIGLIP_REPOS)); ap.add_argument("--dataset", default="refcocog")
    ap.add_argument("--images", nargs="+", type=int, default=[1, 2, 4, 8]); ap.add_argument("--grad-ckpt", action="store_true"); ap.add_argument("--steps", type=int, default=3); a = ap.parse_args()
    dev = torch.device("cuda"); meta = torch.load(META / f"meta_{a.dataset}.pt", weights_only=False)
    meta = sorted(meta, key=lambda m: -len(m["prop_boxes"]))[:max(a.images)]                        # worst case: the images with the most proposals
    bb = Backbone(a.backbone, a.grad_ckpt).to(dev); critic = Critic().to(dev)
    opt = torch.optim.AdamW([{"params": [p for p in bb.parameters() if p.ndim >= 2], "lr": 1e-5, "weight_decay": WD}, {"params": [p for p in bb.parameters() if p.ndim < 2], "lr": 1e-5, "weight_decay": 0.0},
                             {"params": list(critic.parameters()), "lr": 1e-3, "weight_decay": 0.0}])
    prepared = []
    for m in meta:
        with Image.open(m["path"]) as im:
            im = im.convert("RGB"); crops = torch.stack([bb.preprocess(im.crop(tuple(b))) for b in m["prop_boxes"].tolist()])
        p, q, n_ok, n_excl = proposal_laws(m); prepared.append((crops, bb.tokenize(m["texts"]), torch.from_numpy(p), torch.from_numpy(q), n_ok, n_excl))
    res = {"backbone": a.backbone, "dataset_for_candidates": a.dataset, "grad_ckpt": a.grad_ckpt, "gpu": torch.cuda.get_device_name(0), "configs": []}
    for n_img in a.images:
        batch = prepared[:n_img]
        for mode in ("joint", "per_image_accumulation"):
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); times = []
            try:
                for step in range(a.steps):
                    t0 = time.time(); opt.zero_grad(set_to_none=True)
                    if mode == "joint":
                        crops = torch.cat([b[0] for b in batch]).to(dev, non_blocking=True); toks = torch.cat([b[1] for b in batch]).to(dev, non_blocking=True)
                        ms, ks = [b[0].shape[0] for b in batch], [b[1].shape[0] for b in batch]; R, W = max(ms), max(ks)
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            U = bb.enc_img(crops); V = bb.enc_txt(toks)
                        U = F.normalize(U.float(), dim=-1); V = F.normalize(V.float(), dim=-1)
                        C = U.new_zeros(n_img, R, W); P = U.new_zeros(n_img, R, W); Q = U.new_zeros(n_img, R, W); oi = ot = 0
                        for i, b in enumerate(batch):
                            m_, k_ = ms[i], ks[i]; C[i, :m_, :k_] = U[oi:oi + m_] @ V[ot:ot + k_].T; P[i, :m_, :k_] = b[2].to(dev); Q[i, :m_, :k_] = b[3].to(dev); oi += m_; ot += k_
                        loss = padded_batch_loss(critic(C), P, Q, "vcs"); loss.backward()
                    else:
                        for b in batch:
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                U = bb.enc_img(b[0].to(dev, non_blocking=True)); V = bb.enc_txt(b[1].to(dev, non_blocking=True))
                            C = (F.normalize(U.float(), dim=-1) @ F.normalize(V.float(), dim=-1).T)[None]
                            loss = padded_batch_loss(critic(C), b[2][None].to(dev), b[3][None].to(dev), "vcs") / n_img; loss.backward()
                    opt.step(); torch.cuda.synchronize(); times.append(time.time() - t0)
                peak = torch.cuda.max_memory_allocated() / 2**30; status = "ok"
            except torch.cuda.OutOfMemoryError:
                peak, status = None, "OOM"; opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
            cfg = {"images_per_step": n_img, "mode": mode, "crops_per_step": int(sum(b[0].shape[0] for b in batch)), "queries_per_step": int(sum(b[1].shape[0] for b in batch)),
                   "queries_without_valid_proposal": int(sum(b[5] for b in batch)), "peak_mem_GB": peak, "sec_per_step_mean_after_first": float(np.mean(times[1:])) if len(times) > 1 else (float(times[0]) if times else None), "status": status}
            res["configs"].append(cfg); print(json.dumps(cfg), flush=True)
            if status == "OOM" and mode == "joint":
                continue
    OUT.parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(OUT, "w"), indent=1); return 0


if __name__ == "__main__":
    raise SystemExit(main())
