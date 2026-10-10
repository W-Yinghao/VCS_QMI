"""VL3 — FULL end-to-end fine-tuning of a vision-language model's region / expression encoders with VCS, matched JS (balanced logistic) or candidate
softmax on the per-image pair law of Table A (region-uniform; candidates = the image's referred objects; eligible images = >= 2 referred objects).
Owner 2026-10-10: full experiments, not probes; all datasets.  Dataset via VL_DATASET (refcocog / refcoco / refcoco+), roles FIT / CAL / DEV as VL1 / VL2.
  backbones : clip_b16 / clip_l14_336 (open_clip ViT-B-16 / ViT-L-14-336 quickgelu, OpenAI; tight crop -> official preprocess; QuickGELU asserted)
              siglip2_b16 / siglip2_l16 (google/siglip2-{base-patch16-224, large-patch16-256}; tight crop -> its processor, channels_last; lower-cased
              text, max_length 64); --grad-ckpt for the Large backbones (VL3 add. 1)
  critic    : f(u, v) = a * cos(u, v) + b, a = softplus(alpha), (a, b) initialised at (10, -2.5) for every objective (step-0 ranking = raw cosine)
  objective : vcs -> 1 - mean_G J_G with T = tanh f; js -> mean_P softplus(-2f) + mean_Q softplus(2f); softmax -> multi-positive CE over the image's
              candidates (src/vcs_vl/pairlaw.py: padded_batch_loss)
  training  : all encoder parameters, AdamW lr --lr (1e-5) with weight decay 0.05 on matrices, critic scalars AdamW 1e-3; 200-step linear warm-up then cosine
              to 0; batch 32 images; --epochs (10); bf16 autocast; grad-norm clip 1.0; no image augmentation (left / right words forbid flips)
  readout   : CAL and DEV at step 0 and after every epoch: Top-1 over the referred candidates (query and image-macro), common J with the model's own
              critic (J_own) and with an affine critic a' cos + b' refitted on CAL by maximising J (J_recal, comparable across objectives).
              Selected checkpoints: max CAL Top-1 (primary, unbiased for DEV) and max CAL J_own (estimator view, VCS / JS).
    python scripts/vl3_finetune.py --backbone clip_b16 --objective vcs --seed 0 [--smoke]
Writes outputs/VL3/<dataset>_<backbone>_<objective>_s<seed>.json (+ the CAL-Top-1-selected weights under /projects/.../models/vl3/).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

os.environ.setdefault("HF_HUB_OFFLINE", "1")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vcs_vl import refcocog as RG  # noqa: E402
from vcs_vl.pairlaw import pair_laws, padded_batch_loss, padded_j  # noqa: E402

OUT = Path("/home/infres/yinwang/CS_QMI/outputs/VL3"); CKPT = Path("/projects/EEG-foundation-model/yinghao/models/vl3")
CLIP_CACHE = "/projects/EEG-foundation-model/yinghao/models/open_clip"; HF = Path("/projects/EEG-foundation-model/yinghao/models/hf")
OBJ = {"vcs": "vcs", "js": "balanced_logistic", "softmax": "conditional_softmax"}
BATCH, WARMUP, CRITIC_LR, WD = 32, 200, 1e-3, 0.05


CLIP_ARCHS = {"clip_b16": ("ViT-B-16-quickgelu", None), "clip_l14_336": ("ViT-L-14-336-quickgelu", None)}
SIGLIP_REPOS = {"siglip2_b16": "google/siglip2-base-patch16-224", "siglip2_l16": "google/siglip2-large-patch16-256"}


class Backbone(torch.nn.Module):
    def __init__(self, name: str, grad_ckpt: bool = False):
        super().__init__(); self.name = name; self.kind = "clip" if name in CLIP_ARCHS else "siglip"
        if self.kind == "clip":
            import open_clip
            arch = CLIP_ARCHS[name][0]
            self.m, _, self.pre = open_clip.create_model_and_transforms(arch, pretrained="openai", cache_dir=CLIP_CACHE)
            acts = {type(x).__name__ for x in self.m.modules() if "GELU" in type(x).__name__}
            assert acts == {"QuickGELU"}, acts
            self._tok = open_clip.get_tokenizer(arch)
            if grad_ckpt:
                self.m.set_grad_checkpointing(True)
        elif name in SIGLIP_REPOS:
            from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
            snap = str(next((HF / f"models--{SIGLIP_REPOS[name].replace('/', '--')}" / "snapshots").iterdir()))
            self.m = AutoModel.from_pretrained(snap); self.proc = AutoImageProcessor.from_pretrained(snap); self._tok = AutoTokenizer.from_pretrained(snap)
            if grad_ckpt:
                self.m.gradient_checkpointing_enable()
        else:
            raise ValueError(name)

    def preprocess(self, im):
        if self.kind == "clip":
            return self.pre(im)
        return self.proc(images=im, return_tensors="pt", input_data_format="channels_last")["pixel_values"][0]

    def tokenize(self, texts):
        if self.kind == "clip":
            return self._tok(texts)
        return self._tok([t.lower() for t in texts], padding="max_length", max_length=64, truncation=True, return_tensors="pt")["input_ids"]

    def enc_img(self, x):
        return self.m.encode_image(x) if self.kind == "clip" else self.m.get_image_features(pixel_values=x)

    def enc_txt(self, t):
        return self.m.encode_text(t) if self.kind == "clip" else self.m.get_text_features(input_ids=t)


class Critic(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.alpha = torch.nn.Parameter(torch.tensor(math.log(math.expm1(10.0)))); self.b = torch.nn.Parameter(torch.tensor(-2.5))

    def forward(self, cos):
        return F.softplus(self.alpha) * cos + self.b


def records(scenes, role, want):
    out = []
    for s in scenes:
        if role[s.image_id] != want or len(s.referred) < 2 or not s.path:
            continue
        A, _, _ = s.compatibility(); p, q = pair_laws(A, None)
        texts = [e["raw"] for o in s.referred for e in o.expressions]
        out.append({"image_id": s.image_id, "path": s.path, "boxes": [RG.clip_box(o.box_xywh, s.width, s.height) for o in s.referred], "texts": texts,
                    "A": A.astype(np.float32), "P": p.astype(np.float32), "Q": q.astype(np.float32)})
    return out


class Scenes(torch.utils.data.Dataset):
    def __init__(self, recs, bb):
        self.recs, self.bb = recs, bb

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        with Image.open(r["path"]) as im:
            im = im.convert("RGB"); crops = torch.stack([self.bb.preprocess(im.crop(tuple(b))) for b in r["boxes"]])
        return crops, self.bb.tokenize(r["texts"]), torch.from_numpy(r["A"]), torch.from_numpy(r["P"]), torch.from_numpy(r["Q"])


def collate(batch):
    crops = torch.cat([b[0] for b in batch]); toks = torch.cat([b[1] for b in batch]); ms = [b[0].shape[0] for b in batch]; ks = [b[1].shape[0] for b in batch]
    n, R, W = len(batch), max(ms), max(ks); A = torch.zeros(n, R, W); P = torch.zeros(n, R, W); Q = torch.zeros(n, R, W)
    for i, b in enumerate(batch):
        A[i, :ms[i], :ks[i]] = b[2]; P[i, :ms[i], :ks[i]] = b[3]; Q[i, :ms[i], :ks[i]] = b[4]
    return crops, toks, ms, ks, A, P, Q


def scores(U, V, ms, ks, R, W):
    """Per-image cosine matrices, padded to [n, R, W] (padding 0; masked by the laws / masks)."""
    out = U.new_zeros(len(ms), R, W); oi = ot = 0
    for i, (m, k) in enumerate(zip(ms, ks)):
        out[i, :m, :k] = U[oi:oi + m] @ V[ot:ot + k].T; oi += m; ot += k
    return out


@torch.no_grad()
def collect_cos(bb, dl, dev):
    """Cosine matrices of every image of an evaluation split, padded into one [n, R, W] tensor, with A / P / Q."""
    bb.eval(); rows = []
    for crops, toks, ms, ks, A, P, Q in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            U = bb.enc_img(crops.to(dev, non_blocking=True)); V = bb.enc_txt(toks.to(dev, non_blocking=True))
        U = F.normalize(U.float(), dim=-1); V = F.normalize(V.float(), dim=-1); C = scores(U, V, ms, ks, A.shape[1], A.shape[2]).cpu()
        rows += [(C[i, :m, :k], A[i, :m, :k], P[i, :m, :k], Q[i, :m, :k]) for i, (m, k) in enumerate(zip(ms, ks))]
    bb.train(); n = len(rows); R = max(r[0].shape[0] for r in rows); W = max(r[0].shape[1] for r in rows)
    out = {k: torch.zeros(n, R, W) for k in ("C", "A", "P", "Q")}
    for i, (C, A, P, Q) in enumerate(rows):
        m, k = C.shape; out["C"][i, :m, :k] = C; out["A"][i, :m, :k] = A; out["P"][i, :m, :k] = P; out["Q"][i, :m, :k] = Q
    return out


def metrics(d, a, b):
    f = a * d["C"] + b; rmask = d["A"].sum(2) > 0; wmask = d["A"].sum(1) > 0
    pred = f.masked_fill(~rmask[:, :, None], -torch.inf).argmax(1); hit = (pred == d["A"].argmax(1)) & wmask
    return {"top1_query": float(hit.sum() / wmask.sum()), "top1_image_macro": float((hit.sum(1).float() / wmask.sum(1).float()).mean()),
            "J": float(padded_j(f, d["P"], d["Q"]).mean()), "n_images": int(f.shape[0]), "n_queries": int(wmask.sum())}


def recal(d):
    """Affine critic a' cos + b' maximising mean J on CAL (Adam, 400 steps, vectorised); returns (a', b')."""
    al = torch.tensor(math.log(math.expm1(10.0)), requires_grad=True); be = torch.tensor(-2.5, requires_grad=True); opt = torch.optim.Adam([al, be], lr=0.05)
    for _ in range(400):
        J = padded_j(F.softplus(al) * d["C"] + be, d["P"], d["Q"]).mean(); opt.zero_grad(); (-J).backward(); opt.step()
    return float(F.softplus(al)), float(be)


def evaluate(bb, critic, dl_cal, dl_dev, dev):
    rc, rd = collect_cos(bb, dl_cal, dev), collect_cos(bb, dl_dev, dev); a, b = float(F.softplus(critic.alpha)), float(critic.b)
    ar, br = recal(rc)
    return {"own": {"cal": metrics(rc, a, b), "dev": metrics(rd, a, b), "a": a, "b": b},
            "recal": {"cal": metrics(rc, ar, br), "dev": metrics(rd, ar, br), "a": ar, "b": br}}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--backbone", required=True, choices=list(CLIP_ARCHS) + list(SIGLIP_REPOS)); ap.add_argument("--objective", required=True, choices=list(OBJ))
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--grad-ckpt", action="store_true", help="activation checkpointing (Large backbones; numerically the same objective)"); a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); dev = torch.device("cuda")
    tag = f"{RG.DATASET.replace('+', 'plus')}_{a.backbone}_{a.objective}_s{a.seed}" + ("_smoke" if a.smoke else "")
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / f"{tag}.json").exists() and not a.smoke:
        print("exists", tag); return 0
    bb = Backbone(a.backbone, a.grad_ckpt).to(dev); critic = Critic().to(dev)
    scenes, _ = RG.load_scenes(); role = RG.dev_roles(scenes)
    fit, cal, dv = records(scenes, role, "FIT"), records(scenes, role, "CAL"), records(scenes, role, "DEV")
    if a.smoke:
        fit, cal, dv = fit[:64], cal[:48], dv[:48]
    g = torch.Generator().manual_seed(a.seed)
    dl_fit = torch.utils.data.DataLoader(Scenes(fit, bb), batch_size=BATCH, shuffle=True, generator=g, num_workers=a.workers, collate_fn=collate, drop_last=True, pin_memory=True, persistent_workers=True)
    mk = lambda r: torch.utils.data.DataLoader(Scenes(r, bb), batch_size=64, shuffle=False, num_workers=a.workers, collate_fn=collate, pin_memory=True)
    dl_cal, dl_dev = mk(cal), mk(dv)
    decay = [p for n, p in bb.named_parameters() if p.ndim >= 2]; no_decay = [p for n, p in bb.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "lr": a.lr, "weight_decay": WD}, {"params": no_decay, "lr": a.lr, "weight_decay": 0.0},
                             {"params": list(critic.parameters()), "lr": CRITIC_LR, "weight_decay": 0.0}])
    base = [gp["lr"] for gp in opt.param_groups]; epochs = 1 if a.smoke else a.epochs; total = epochs * len(dl_fit); step = 0
    sched = lambda s: (s + 1) / WARMUP if s < WARMUP else 0.5 * (1 + math.cos(math.pi * (s - WARMUP) / max(1, total - WARMUP)))
    t0 = time.time(); hist = [{"epoch": 0, "step": 0, **evaluate(bb, critic, dl_cal, dl_dev, dev)}]
    print(f"[{tag}] step 0: DEV top1 macro {hist[0]['own']['dev']['top1_image_macro']:.4f} J_own {hist[0]['own']['cal']['J']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    best = {"cal_top1": (hist[0]["own"]["cal"]["top1_image_macro"], 0, None), "cal_J": (hist[0]["own"]["cal"]["J"], 0)}
    for ep in range(1, epochs + 1):
        run = []
        for i, (crops, toks, ms, ks, A, P, Q) in enumerate(dl_fit):
            for gp, b0 in zip(opt.param_groups, base):
                gp["lr"] = b0 * sched(step)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                U = bb.enc_img(crops.to(dev, non_blocking=True)); V = bb.enc_txt(toks.to(dev, non_blocking=True))
            U = F.normalize(U.float(), dim=-1); V = F.normalize(V.float(), dim=-1)
            f = critic(scores(U, V, ms, ks, A.shape[1], A.shape[2])); loss = padded_batch_loss(f, P.to(dev), Q.to(dev), OBJ[a.objective])
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {ep} iter {i}")
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(list(bb.parameters()) + list(critic.parameters()), 1.0); opt.step()
            step += 1; run.append(float(loss))
            if a.smoke and i >= 20:
                break
        e = evaluate(bb, critic, dl_cal, dl_dev, dev); hist.append({"epoch": ep, "step": step, "train_loss": float(np.mean(run)), **e})
        print(f"[{tag}] epoch {ep}: loss {np.mean(run):.4f} CAL top1 {e['own']['cal']['top1_image_macro']:.4f} DEV top1 {e['own']['dev']['top1_image_macro']:.4f} "
              f"J_own CAL {e['own']['cal']['J']:.4f} J_recal CAL {e['recal']['cal']['J']:.4f} a {e['own']['a']:.2f} ({time.time() - t0:.0f}s)", flush=True)
        if e["own"]["cal"]["top1_image_macro"] > best["cal_top1"][0]:
            best["cal_top1"] = (e["own"]["cal"]["top1_image_macro"], ep, {k: v.detach().to("cpu", torch.float16).clone() for k, v in bb.state_dict().items()})
        if e["own"]["cal"]["J"] > best["cal_J"][0]:
            best["cal_J"] = (e["own"]["cal"]["J"], ep)
    sel = {"cal_top1_epoch": best["cal_top1"][1], "cal_J_epoch": best["cal_J"][1]}
    res = {"dataset": RG.DATASET, "backbone": a.backbone, "objective": a.objective, "seed": a.seed, "lr": a.lr, "epochs": epochs, "batch_images": BATCH,
           "n_fit_images": len(fit), "steps": step, "seconds": time.time() - t0, "history": hist, "selected": sel,
           "dev_at_cal_top1": hist[sel["cal_top1_epoch"]], "dev_at_cal_J": hist[sel["cal_J_epoch"]], "smoke": a.smoke}
    json.dump(res, open(OUT / f"{tag}.json", "w"), indent=1)
    if best["cal_top1"][2] is not None and not a.smoke:
        CKPT.mkdir(parents=True, exist_ok=True); torch.save({"backbone": best["cal_top1"][2], "epoch": best["cal_top1"][1]}, CKPT / f"{tag}_calTop1.pt")
    print(f"[{tag}] selected (CAL top1) epoch {sel['cal_top1_epoch']}: DEV top1 macro {res['dev_at_cal_top1']['own']['dev']['top1_image_macro']:.4f}; "
          f"(CAL J) epoch {sel['cal_J_epoch']}: DEV J_own {res['dev_at_cal_J']['own']['dev']['J']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
