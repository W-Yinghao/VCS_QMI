"""P119 (v5 NEXT-V-PROBE) — readout checks on frozen encoders, no encoder training.

(1) Label efficiency (CIFAR-10): extra independent subset draws 3–5 at 1 % / 10 % with exactly the P110 readout, so that readout variability within an
    encoder seed can be estimated; the main differences are paired by encoder seed in `paired_by_seed` (draws are not pretraining seeds).
(2) Frozen transfer (CIFAR-10 encoder → CIFAR-100, P91 dev45k / val5k manifest, 100 % and 10 % labels, P110 subset seeds): the same readouts for every
    method — `raw_unstd` (the P110 readout, anchor), and per-dimension standardised `raw_std`, `l2_std` (h/‖h‖), `lognorm_std` (log‖h‖ alone, 1-d),
    `l2_lognorm_std` ([h/‖h‖, log‖h‖]); standardisation statistics come from the labelled training subset only.  Every readout uses the P110 probe:
    (lr, wd) chosen on an inner 80 / 20 split of the labelled subset (grid 3 × 2), recipe stopping (100 epochs, cosine), refit on the subset.  kNN with the
    recipe hyper-parameters (k 200, T 0.1, cosine on h) on the same labelled bank.  Norm readouts are diagnostics; they never replace the standard table.
(3) float16 cache check: the P110 cache stores h in float16; `fp_check` re-extracts a small subset in float32 on the same device and compares h and
    the readouts.  Labels are used only by the probes and kNN.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from vcs_ssl.data.cifar import load_cifar100_train, load_cifar10_train
from vcs_ssl.diagnostics import knn_eval

from .common import (CACHE_ROOT, CIFAR100_MANIFEST, CIFAR100_ROOT, OUTPUT_ROOT, Encoder, accuracy, atomic_torch_save, cached_h,
                     fit_probe_selected, sha_idx, stratified_subset, train_linear)
from .v1 import subset_seed

READOUTS = ("raw_unstd", "raw_std", "l2_std", "lognorm_std", "l2_lognorm_std")
KNN = {"k": 200, "temperature": 0.1}  # recipe evaluation.knn (P35 config)
P119_CACHE = OUTPUT_ROOT / "P119_features"
EXTRA_DRAWS = (3, 4, 5)


def readout_features(h: torch.Tensor, kind: str) -> torch.Tensor:
    """Readout feature map applied identically for every method (before standardisation)."""
    h = h.float(); n = h.norm(dim=1, keepdim=True).clamp_min(1e-12)
    if kind in ("raw_unstd", "raw_std"):
        return h
    if kind == "l2_std":
        return h / n
    if kind == "lognorm_std":
        return n.log()
    if kind == "l2_lognorm_std":
        return torch.cat((h / n, n.log()), 1)
    raise ValueError(kind)


def standardise(train: torch.Tensor, *others: torch.Tensor, eps: float = 1e-6):
    """Per-dimension z-score with statistics of the labelled training subset only."""
    mu, sd = train.mean(0, keepdim=True), train.std(0, keepdim=True).clamp_min(eps)
    return [(train - mu) / sd] + [(o - mu) / sd for o in others]


def prepare(kind: str, h_train: torch.Tensor, h_eval: torch.Tensor):
    xt, xe = readout_features(h_train, kind), readout_features(h_eval, kind)
    if kind.endswith("_std"):
        xt, xe = standardise(xt, xe)
    return xt, xe


def cifar100_sets():
    data = load_cifar100_train(CIFAR100_ROOT); m = json.load(open(CIFAR100_MANIFEST))
    return data, np.asarray(m["fit_uids"]), np.asarray(m["selection_uids"])


def transfer_readouts(enc: Encoder, dev: torch.device, hf: torch.Tensor, hs: torch.Tensor, yf: torch.Tensor, ys: torch.Tensor,
                      fractions=((0.1, 3), (1.0, 1)), readouts=READOUTS) -> list[dict]:
    rows = []
    for frac, ndraw in fractions:
        for d in range(ndraw):
            seed = subset_seed("cifar100", frac, d); idx = stratified_subset(yf.numpy(), frac, seed)
            kn = knn_eval(hf[idx], yf[idx], hs, ys, k=min(KNN["k"], len(idx)), temperature=KNN["temperature"], n_classes=100, device=dev)
            row = {"fraction": frac, "draw": d, "subset_seed": seed, "n_labels": int(len(idx)), "knn_pct": kn["knn_val_top1_pct"], "readouts": {}}
            for kind in readouts:
                t0 = time.time(); xt, xe = prepare(kind, hf[idx], hs)
                sel = fit_probe_selected(xt, yf[idx], 100, dev, sel_seed=seed + 1)
                row["readouts"][kind] = {"acc_pct": accuracy(sel["head"], xe, ys, dev), "chosen": sel["chosen"], "inner_scores": sel["inner_scores"],
                                         "dim": int(xt.shape[1]), "seconds": time.time() - t0}
            rows.append(row)
            print(f"[{enc.run}] transfer frac {frac} draw {d}: kNN {row['knn_pct']:.2f} | " +
                  " ".join(f"{k} {v['acc_pct']:.2f}" for k, v in row["readouts"].items()), flush=True)
    return rows


def label_efficiency_extra(enc: Encoder, dev: torch.device, hf: torch.Tensor, hs: torch.Tensor, yf: torch.Tensor, ys: torch.Tensor,
                           draws=EXTRA_DRAWS, fractions=(0.01, 0.1)) -> list[dict]:
    """Exactly the P110 V1 readout (FIT-selected probe and recipe-fixed probe on raw h) on new independent subset draws."""
    rows = []
    for frac in fractions:
        for d in draws:
            seed = subset_seed("cifar10", frac, d); idx = stratified_subset(yf.numpy(), frac, seed)
            sel = fit_probe_selected(hf[idx], yf[idx], 10, dev, sel_seed=seed + 1); rec = train_linear(hf[idx], yf[idx], 10, {}, dev)
            rows.append({"fraction": frac, "draw": d, "subset_seed": seed, "n_labels": int(len(idx)), "chosen": sel["chosen"],
                         "acc_selected_pct": accuracy(sel["head"], hs, ys, dev), "acc_recipe_fixed_pct": accuracy(rec, hs, ys, dev)})
            print(f"[{enc.run}] C10 frac {frac} draw {d}: selected {rows[-1]['acc_selected_pct']:.2f} recipe {rows[-1]['acc_recipe_fixed_pct']:.2f}", flush=True)
    return rows


def features(enc: Encoder, ds: str, fp32: bool, smoke: bool = False):
    """h for FIT / selection of a dataset: the P110 float16 cache, or (fp32=True) a float32 cache under outputs/P119_features."""
    if ds == "cifar10":
        data = load_cifar10_train(enc.R["cfg"]["data"]["root"]); fit, sel = np.asarray(enc.manifest["fit_uids"]), np.asarray(enc.manifest["selection_uids"])
    else:
        data, fit, sel = cifar100_sets()
    if smoke:
        fit, sel = fit[::30], sel[::10]
    tag = "smoke_" if smoke else ""
    if fp32 or smoke:  # smoke features never go into the P110 cache
        out = []
        for part, uids in (("fit", fit), ("sel", sel)):
            p = P119_CACHE / enc.run / f"{tag}{ds}_{part}_h_fp32.pt"
            if p.is_file():
                out.append(torch.load(p, map_location="cpu"))
            else:
                h = enc.h(data.data[uids]).float(); atomic_torch_save(h, p); out.append(h)
        hf, hs = out
    else:
        hf = cached_h(enc, f"{tag}{ds}_fit_h", lambda: data.data[fit]); hs = cached_h(enc, f"{tag}{ds}_sel_h", lambda: data.data[sel])
    return hf, hs, torch.as_tensor(data.targets[fit]), torch.as_tensor(data.targets[sel])


def fp_check(enc: Encoder, dev: torch.device) -> dict:
    """float16 cache vs float32 re-extraction on a small subset (same device): CIFAR-100 10 % draw-0 FIT subset + the 5k selection images, and
    CIFAR-10 1 % draw-0 FIT subset + selection.  Compares h and the P110 readout / raw_std readout / kNN on the identical subset."""
    out = {"run": enc.run, "datasets": {}}
    for ds, frac, nc in (("cifar100", 0.1, 100), ("cifar10", 0.01, 10)):
        if ds == "cifar10":
            data = load_cifar10_train(enc.R["cfg"]["data"]["root"]); fit, sel = np.asarray(enc.manifest["fit_uids"]), np.asarray(enc.manifest["selection_uids"])
        else:
            data, fit, sel = cifar100_sets()
        h16f = torch.load(CACHE_ROOT / enc.run / f"{ds}_fit_h.pt", map_location="cpu"); h16s = torch.load(CACHE_ROOT / enc.run / f"{ds}_sel_h.pt", map_location="cpu")
        assert h16f.dtype == torch.float16, h16f.dtype
        yf, ys = torch.as_tensor(data.targets[fit]), torch.as_tensor(data.targets[sel])
        seed = subset_seed(ds, frac, 0); idx = stratified_subset(yf.numpy(), frac, seed)
        h32f = enc.h(data.data[fit[idx]]).float(); h32s = enc.h(data.data[sel]).float()
        a16f, a16s = h16f[idx].float(), h16s.float()
        rel = lambda a, b: float((a - b).norm() / b.norm().clamp_min(1e-12))
        rec = {"fraction": frac, "subset_seed": seed, "subset_sha": sha_idx(fit[idx]), "n_fit": int(len(idx)), "n_sel": int(len(sel)),
               "h_max_abs_diff": float(max((a16f - h32f).abs().max(), (a16s - h32s).abs().max())),
               "h_rel_frob_diff": max(rel(a16f, h32f), rel(a16s, h32s)), "readouts": {}}
        for prec, (xf, xs) in (("fp16", (a16f, a16s)), ("fp32", (h32f, h32s))):
            r = {"knn_pct": knn_eval(xf, yf[idx], xs, ys, k=min(KNN["k"], len(idx)), temperature=KNN["temperature"], n_classes=nc, device=dev)["knn_val_top1_pct"]}
            for kind in ("raw_unstd", "raw_std"):
                xt, xe = prepare(kind, xf, xs); s = fit_probe_selected(xt, yf[idx], nc, dev, sel_seed=seed + 1)
                r[kind] = accuracy(s["head"], xe, ys, dev)
            rec["readouts"][prec] = r
        rec["max_abs_acc_diff"] = max(abs(rec["readouts"]["fp16"][k] - rec["readouts"]["fp32"][k]) for k in ("knn_pct", "raw_unstd", "raw_std"))
        out["datasets"][ds] = rec
        print(f"[{enc.run}] fp-check {ds}: h rel diff {rec['h_rel_frob_diff']:.2e}, max |Δacc| {rec['max_abs_acc_diff']:.2f} "
              f"(fp16 {rec['readouts']['fp16']} / fp32 {rec['readouts']['fp32']})", flush=True)
    return out


def paired_by_seed(table: dict, ref: str, fams: list[str]) -> dict:
    """table[family][seed][fraction] = list of per-draw accuracies.  Per encoder seed: mean over draws; then paired differences family − ref by
    seed (mean, sd over seeds, 95 % t interval with df = n_seeds − 1) and the within-seed draw sd (readout variability).  Draws never add seeds."""
    from scipy import stats
    out = {}
    for f in fams:
        if f == ref:
            continue
        for frac in sorted({fr for s in table[f].values() for fr in s}):
            seeds = sorted(set(table[f]) & set(table[ref]))
            seeds = [s for s in seeds if frac in table[f][s] and frac in table[ref][s]]
            d = np.array([np.mean(table[f][s][frac]) - np.mean(table[ref][s][frac]) for s in seeds])
            within = [np.std(table[x][s][frac], ddof=1) for x in (f, ref) for s in seeds if len(table[x][s][frac]) > 1]
            if len(d) == 0:
                continue
            m = float(d.mean()); sd = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
            h = float(stats.t.ppf(0.975, len(d) - 1) * sd / np.sqrt(len(d))) if len(d) > 1 else float("nan")
            out[f"{f} - {ref} @ {frac}"] = {"per_seed": [float(x) for x in d], "seeds": seeds, "mean": m, "sd_over_seeds": sd,
                                            "ci95": [m - h, m + h], "mean_within_seed_draw_sd": float(np.mean(within)) if within else float("nan"),
                                            "n_draws": sorted({len(table[x][s][frac]) for x in (f, ref) for s in seeds})}
    return out
