"""V2 — corruption / compression tolerance, DEVELOPMENT phase (package v4 §V2).

Head: the recipe linear probe (fixed recipe hyper-parameters) trained on clean frozen h of all 45k FIT images.  Evaluation: the 5k selection images,
clean and under the 6 × 5 fixed development corruptions of `corruptions.py`.  Per cell: accuracy, relative drop 1 − acc / acc_clean, prediction
consistency (same prediction as on the clean image); mean corruption accuracy (mCA) over the 30 cells.  CIs: bootstrap over base images (each
image's 31 versions resampled together = clustered by image).

The standard CIFAR-10-C (`run_cifar10c`) is built from OFFICIAL TEST images: it runs only with authorised=True (separate flag and lines file,
owner authorisation), is never used for any choice, and reports absolute accuracy per corruption × severity with the same frozen head."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from vcs_ssl.data.cifar import load_cifar10_train

from .common import Encoder, cached_h, train_linear
from .corruptions import cells, corrupt

CIFAR10C_DIR = Path("/projects/EEG-foundation-model/yinghao/FMCA-AV/robustness/cifar10-c")


def _boot(correct: np.ndarray, clean: np.ndarray, B: int, seed: int) -> dict:
    """correct [n_img, n_cell] bool, clean [n_img] bool → bootstrap (by image) percentile CIs of per-cell acc, mCA and mean relative drop."""
    rng = np.random.default_rng(seed); n = len(clean); mca, drop = [], []; cell = []
    for _ in range(B):
        i = rng.integers(0, n, n); a_c = clean[i].mean(); a = correct[i].mean(0)
        cell.append(a); mca.append(a.mean()); drop.append((1 - a / max(a_c, 1e-9)).mean())
    cell = np.array(cell)
    q = lambda v: [float(np.percentile(v, 2.5)) * 100, float(np.percentile(v, 97.5)) * 100]
    return {"cell_ci_pct": [q(cell[:, j]) for j in range(cell.shape[1])], "mCA_ci_pct": q(np.array(mca)), "mean_rel_drop_ci_pct": q(np.array(drop))}


@torch.no_grad()
def run_v2(enc: Encoder, dev: torch.device, smoke: bool = False, boot: int = 1000) -> dict:
    data = load_cifar10_train(enc.R["cfg"]["data"]["root"])
    fit, sel = np.asarray(enc.manifest["fit_uids"]), np.asarray(enc.manifest["selection_uids"])
    if smoke:
        fit, sel = fit[::30], sel[::10]
    tag = "smoke_" if smoke else ""
    hf = cached_h(enc, f"{tag}cifar10_fit_h", lambda: data.data[fit]); hs = cached_h(enc, f"{tag}cifar10_sel_h", lambda: data.data[sel])
    yf, ys = torch.as_tensor(data.targets[fit]), torch.as_tensor(data.targets[sel])
    with torch.enable_grad():
        head = train_linear(hf, yf, 10, {}, dev)
    pred_clean = head(hs.to(dev).float()).argmax(1).cpu(); correct_clean = (pred_clean == ys).numpy()
    cl = cells(); correct = np.zeros((len(sel), len(cl)), dtype=bool); consist = np.zeros_like(correct); t0 = time.time()
    for j, (fam, sev) in enumerate(cl):
        imgs = corrupt(data.data[sel], sel, fam, sev)
        p = head(enc.h(imgs).to(dev)).argmax(1).cpu()
        correct[:, j] = (p == ys).numpy(); consist[:, j] = (p == pred_clean).numpy()
    acc_clean = float(correct_clean.mean()) * 100
    per_cell = [{"family": f, "severity": s, "acc_pct": float(correct[:, j].mean()) * 100, "rel_drop_pct": (1 - correct[:, j].mean() / max(correct_clean.mean(), 1e-9)) * 100,
                 "consistency_pct": float(consist[:, j].mean()) * 100} for j, (f, s) in enumerate(cl)]
    ci = _boot(correct, correct_clean, 100 if smoke else boot, seed=4242)
    for j, r in enumerate(per_cell):
        r["acc_ci_pct"] = ci["cell_ci_pct"][j]
    out = {"run": enc.run, "head": "recipe probe, fixed hyper-parameters, clean FIT", "n_fit": int(len(fit)), "n_selection": int(len(sel)),
           "acc_clean_pct": acc_clean, "mCA_pct": float(correct.mean()) * 100, "mCA_ci_pct": ci["mCA_ci_pct"],
           "mean_rel_drop_pct": float(np.mean([r["rel_drop_pct"] for r in per_cell])), "mean_rel_drop_ci_pct": ci["mean_rel_drop_ci_pct"],
           "mean_consistency_pct": float(consist.mean()) * 100, "cells": per_cell, "seconds_corrupted": time.time() - t0}
    print(f"[{enc.run}] V2 clean {acc_clean:.2f} mCA {out['mCA_pct']:.2f} mean drop {out['mean_rel_drop_pct']:.1f}% consistency "
          f"{out['mean_consistency_pct']:.1f}% ({out['seconds_corrupted']:.0f}s)", flush=True)
    return out


@torch.no_grad()
def run_cifar10c(enc: Encoder, dev: torch.device, *, authorised: bool, root: Path = CIFAR10C_DIR) -> dict:
    """Standard CIFAR-10-C (official TEST images): only with explicit owner authorisation; reporting only, never a selection input."""
    if not authorised:
        raise PermissionError("CIFAR-10-C is built from the official CIFAR-10 test images: run only with the owner's authorisation "
                              "(--authorised-official-test-c10c and the separate lines file)")
    data = load_cifar10_train(enc.R["cfg"]["data"]["root"]); fit = np.asarray(enc.manifest["fit_uids"])
    hf = cached_h(enc, "cifar10_fit_h", lambda: data.data[fit]); yf = torch.as_tensor(data.targets[fit])
    with torch.enable_grad():
        head = train_linear(hf, yf, 10, {}, dev)
    labels = torch.as_tensor(np.load(root / "labels.npy").astype(np.int64))
    rows = []
    for f in sorted(p.stem for p in root.glob("*.npy") if p.stem != "labels"):
        arr = np.load(root / f"{f}.npy", mmap_mode="r")
        for s in range(5):
            x = np.asarray(arr[s * 10000:(s + 1) * 10000]); y = labels[s * 10000:(s + 1) * 10000]
            p = head(enc.h(x).to(dev)).argmax(1).cpu()
            rows.append({"corruption": f, "severity": s + 1, "acc_pct": float((p == y).float().mean()) * 100})
    return {"run": enc.run, "benchmark": "CIFAR-10-C (Hendrycks & Dietterich 2019), local copy", "authorised": True, "rows": rows,
            "mean_acc_pct": float(np.mean([r["acc_pct"] for r in rows]))}
