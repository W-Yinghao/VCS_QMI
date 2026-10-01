"""Shared pieces for P110: frozen-encoder h features (GPU, in-memory uint8 images), a feature cache, class-stratified label subsets, and the
recipe-style linear probe with hyper-parameters chosen inside the training labels only."""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from vcs_measure.common import OUTPUT_ROOT, load_run, run_status

CACHE_ROOT = OUTPUT_ROOT / "P110_features"
DEFAULT_RUNS = [f"P35_vcs_a5_views4_800ep_seed{s}" for s in range(3)] + [f"P41_simclr_views4_800ep_seed{s}" for s in range(3)] + \
               [f"P104_G2_views4_800ep_seed{s}" for s in range(3)] + [f"P104_U2_views4_800ep_seed{s}" for s in range(3)]
CIFAR100_MANIFEST = "/home/infres/yinwang/CS_QMI/manifests/cifar100_dev45k_val5k.json"
CIFAR100_ROOT = "/home/infres/yinwang/CS_QMI/data/cifar100"

# recipe probe (P35 config evaluation.linear): SGD momentum 0.9, batch 256, cosine to 1e-3, 100 epochs, lr 0.1, wd 0, seed 20260925
RECIPE_PROBE = {"lr": 0.1, "weight_decay": 0.0, "epochs": 100, "batch_size": 256, "momentum": 0.9, "min_lr_ratio": 1e-3, "seed": 20260925}
PROBE_GRID = [{"lr": lr, "weight_decay": wd} for lr in (0.03, 0.1, 0.3) for wd in (0.0, 5e-4)]


def device() -> torch.device:
    return torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")


def completed(run: str) -> bool:
    st, _ = run_status(run)
    return st == "COMPLETED" and (OUTPUT_ROOT / run / "checkpoints" / "epoch_800.pt").is_file()


def atomic_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    torch.save(obj, tmp); os.replace(tmp, path)


def atomic_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=float)); os.replace(tmp, path)


class Encoder:
    """Frozen encoder of one COMPLETED run; `h(images_uint8)` returns the 512-d encoder output with the run's own clean normalisation."""

    def __init__(self, run: str, dev: torch.device):
        self.R = load_run(run, device=dev); self.dev = dev; self.run = run
        v = self.R["cfg"]["views"]
        self.mean = torch.tensor(v["normalize_mean"], device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor(v["normalize_std"], device=dev).view(1, 3, 1, 1)
        self.manifest = self.R["manifest"]

    @torch.no_grad()
    def h(self, images: np.ndarray, bs: int = 1024) -> torch.Tensor:
        out = []
        for s in range(0, len(images), bs):
            x = torch.from_numpy(np.ascontiguousarray(images[s:s + bs])).to(self.dev).permute(0, 3, 1, 2).float().div_(255.0)
            x = (x - self.mean) / self.std
            out.append(self.R["encoder"](x).float().cpu())
        return torch.cat(out) if out else torch.zeros(0, 512)


def cached_h(enc: Encoder, key: str, images_fn, dtype=torch.float16) -> torch.Tensor:
    """h for a named image set, cached under outputs/P110_features/<run>/<key>.pt (float16; probes cast back to float32)."""
    p = CACHE_ROOT / enc.run / f"{key}.pt"
    if p.is_file():
        return torch.load(p, map_location="cpu").float()
    t0 = time.time(); h = enc.h(images_fn())
    atomic_torch_save(h.to(dtype), p)
    print(f"[{enc.run}] features {key} {tuple(h.shape)} in {time.time() - t0:.0f}s", flush=True)
    return h


def stratified_subset(y: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Class-stratified subset of positions 0..len(y)-1 (same count per class when classes are balanced); fixed seed; sorted."""
    if frac >= 1.0:
        return np.arange(len(y))
    rng = np.random.default_rng(seed); idx = []
    for c in np.unique(y):
        pos = np.where(y == c)[0]; k = max(1, int(round(frac * len(pos))))
        idx.append(rng.choice(pos, size=k, replace=False))
    return np.sort(np.concatenate(idx))


def inner_split(y: np.ndarray, seed: int, val_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Class-stratified 80 / 20 split of the labelled subset (positions into y) for probe hyper-parameter selection."""
    rng = np.random.default_rng(seed); tr, va = [], []
    for c in np.unique(y):
        pos = rng.permutation(np.where(y == c)[0]); k = max(1, int(round(val_frac * len(pos))))
        va.append(pos[:k]); tr.append(pos[k:])
    return np.sort(np.concatenate(tr)), np.sort(np.concatenate(va))


def train_linear(xf: torch.Tensor, yf: torch.Tensor, n_classes: int, hp: dict, dev: torch.device) -> nn.Linear:
    """The recipe probe (vcs_ssl.diagnostics.linear_probe: SGD momentum, cosine lr to min_lr_ratio, final epoch, PyTorch default init) with
    lr / weight_decay from `hp`; returns the head."""
    p = {**RECIPE_PROBE, **hp}
    gen = torch.Generator().manual_seed(p["seed"])
    with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
        torch.manual_seed(p["seed"]); head = nn.Linear(xf.shape[1], n_classes, bias=True).to(dev)
    opt = torch.optim.SGD(head.parameters(), lr=p["lr"], momentum=p["momentum"], weight_decay=p["weight_decay"])
    xf, yf = xf.to(dev).float(), yf.to(dev)
    n, bs = len(xf), p["batch_size"]; total = p["epochs"] * math.ceil(n / bs); step = 0
    for _ in range(p["epochs"]):
        perm = torch.randperm(n, generator=gen).to(dev)
        for s in range(0, n, bs):
            lr = p["lr"] * (p["min_lr_ratio"] + (1 - p["min_lr_ratio"]) * (1 + math.cos(math.pi * step / max(total - 1, 1))) / 2)
            for g in opt.param_groups:
                g["lr"] = lr
            idx = perm[s:s + bs]; loss = F.cross_entropy(head(xf[idx]), yf[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); step += 1
    head.eval()
    return head


@torch.no_grad()
def accuracy(head: nn.Module, x: torch.Tensor, y: torch.Tensor, dev: torch.device) -> float:
    return float((head(x.to(dev).float()).argmax(1).cpu() == y).float().mean()) * 100


def fit_probe_selected(xf: torch.Tensor, yf: torch.Tensor, n_classes: int, dev: torch.device, sel_seed: int) -> dict:
    """Choose (lr, wd) on an inner 80 / 20 split of the labelled training subset only, then refit on the whole subset with the choice."""
    tr, va = inner_split(yf.numpy(), sel_seed)
    scores = []
    for hp in PROBE_GRID:
        h = train_linear(xf[tr], yf[tr], n_classes, hp, dev); scores.append(accuracy(h, xf[va], yf[va], dev))
    best = int(np.argmax(scores))  # ties → first in grid order (smaller lr, then smaller wd)
    head = train_linear(xf, yf, n_classes, PROBE_GRID[best], dev)
    return {"head": head, "chosen": PROBE_GRID[best], "inner_scores": scores, "n_train": int(len(xf)), "n_inner_val": int(len(va))}


def sha_idx(a: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(a, dtype=np.int64).tobytes()).hexdigest()[:16]
