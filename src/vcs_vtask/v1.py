"""V1 — label efficiency (CIFAR-10, frozen h) and frozen transfer to CIFAR-100.

Label subsets: 1 %, 10 %, 100 % of the 45k FIT images, class-stratified, draws 0–2 for 1 % / 10 % (one for 100 %); the same index manifests for
every encoder (seeded by fraction and draw only).  Readout: the recipe linear probe, (lr, wd) chosen on an inner 80 / 20 split of the labelled
subset (grid 3 × 2), refit on the subset; the recipe's fixed hyper-parameters (lr 0.1, wd 0) reported alongside.  Evaluation: the 5k selection
split (all labels).  Transfer: the same CIFAR-10 encoder and its own normalisation on CIFAR-100 (local train partition, P91 dev45k / val5k
manifest), 100 % and 10 % labels; reported as transfer, never mixed with P91's from-scratch CIFAR-100 SSL.  Labels are used only for the probes."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from vcs_ssl.data.cifar import load_cifar100_train, load_cifar10_train

from .common import (CIFAR100_MANIFEST, CIFAR100_ROOT, RECIPE_PROBE, Encoder, accuracy, cached_h, fit_probe_selected, sha_idx,
                     stratified_subset, train_linear)

FRACTIONS = {"cifar10": [(0.01, 3), (0.1, 3), (1.0, 1)], "cifar100": [(0.1, 3), (1.0, 1)]}


def subset_seed(dataset: str, frac: float, draw: int) -> int:
    return {"cifar10": 11, "cifar100": 13}[dataset] * 100_000 + int(round(frac * 1000)) * 10 + draw


def run_v1(enc: Encoder, dev: torch.device, smoke: bool = False) -> dict:
    out = {"run": enc.run, "probe_recipe": RECIPE_PROBE, "datasets": {}}
    for ds in ("cifar10", "cifar100"):
        if ds == "cifar10":
            data = load_cifar10_train(enc.R["cfg"]["data"]["root"]); fit, sel = np.asarray(enc.manifest["fit_uids"]), np.asarray(enc.manifest["selection_uids"])
        else:
            data = load_cifar100_train(CIFAR100_ROOT); m = json.load(open(CIFAR100_MANIFEST)); fit, sel = np.asarray(m["fit_uids"]), np.asarray(m["selection_uids"])
        if smoke:
            fit, sel = fit[::30], sel[::10]
        tag = "smoke_" if smoke else ""
        hf = cached_h(enc, f"{tag}{ds}_fit_h", lambda: data.data[fit]); hs = cached_h(enc, f"{tag}{ds}_sel_h", lambda: data.data[sel])
        yf, ys = torch.as_tensor(data.targets[fit]), torch.as_tensor(data.targets[sel]); nc = int(data.targets.max()) + 1
        rows = []
        for frac, ndraw in FRACTIONS[ds]:
            for d in range(ndraw):
                t0 = time.time(); seed = subset_seed(ds, frac, d)
                idx = stratified_subset(yf.numpy(), frac, seed)
                sel_fit = fit_probe_selected(hf[idx], yf[idx], nc, dev, sel_seed=seed + 1)
                rec_head = train_linear(hf[idx], yf[idx], nc, {}, dev)
                rows.append({"fraction": frac, "draw": d, "subset_seed": seed, "n_labels": int(len(idx)), "subset_sha": sha_idx(fit[idx]),
                             "chosen": sel_fit["chosen"], "inner_scores": sel_fit["inner_scores"],
                             "acc_selected_pct": accuracy(sel_fit["head"], hs, ys, dev), "acc_recipe_fixed_pct": accuracy(rec_head, hs, ys, dev),
                             "seconds": time.time() - t0})
                print(f"[{enc.run}] V1 {ds} frac {frac} draw {d}: selected {rows[-1]['acc_selected_pct']:.2f} ({sel_fit['chosen']}) "
                      f"recipe {rows[-1]['acc_recipe_fixed_pct']:.2f} ({rows[-1]['seconds']:.0f}s)", flush=True)
        out["datasets"][ds] = {"n_fit": int(len(fit)), "n_selection": int(len(sel)), "rows": rows,
                               "role": "label efficiency" if ds == "cifar10" else "frozen transfer (CIFAR-10-pretrained encoder)"}
    return out
