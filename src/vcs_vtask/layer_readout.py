"""P141 (v7 V7-C100-LAYER-READOUT) — coarse / fine / conditional CIFAR-100 readouts of frozen SSL encoders at FOUR feature sites; no encoder training.

Sites (one forward pass per image, the run's own clean normalisation, eval mode):
  layer3  torchvision layer3 output with FIXED global average pooling (256-d ResNet-18, 1024-d ResNet-50)
  h       encoder output after avgpool (512-d / 2048-d) — identical to the h used by every earlier readout (checked against the P124 cache)
  r       the projector's UNNORMALISED output p = g(h) (128-d)
  z       z = r / max(||r||, eps) with the run's eps (the training's L2 map, objectives.forward_features)
r -> z is a deterministic compression (z is a function of r).  L2-normalised h is NOT a parent of r or z in the nested chain (r = g(h) reads the
unnormalised h); it is not a site here.
Every site gets its own probes trained with the IDENTICAL P124 rules (vcs_vtask.granularity: same FIT / selection images, same label handling, same
probe recipes, same inner-split seeds, same kNN): no rule depends on the site or the method.  The backbone metric stays the h readout — a better
score at another site never replaces it.  Labels are used only by these downstream readouts; the official test file is never opened.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import granularity as G
from .common import OUTPUT_ROOT, Encoder, atomic_torch_save, sha_idx

SITES = ("layer3", "h", "r", "z")
P141_CACHE = OUTPUT_ROOT / "P141_features"
DEFAULT_READOUTS = ("recipe_raw", "raw_std", "knn")  # P124 primary + one scale-robust selected map + kNN (all P124 rules); --readouts all = P124's five


@torch.no_grad()
def site_features(enc_mod, proj_mod, x: torch.Tensor, eps: float) -> dict[str, torch.Tensor]:
    """The four sites for a normalised image batch x [B, 3, H, W] (torchvision ResNet path; maxpool is Identity for the CIFAR stem)."""
    a = enc_mod.relu(enc_mod.bn1(enc_mod.conv1(x))); a = enc_mod.maxpool(a)
    a = enc_mod.layer1(a); a = enc_mod.layer2(a); l3 = enc_mod.layer3(a); l4 = enc_mod.layer4(l3)
    h = torch.flatten(enc_mod.avgpool(l4), 1)
    r = proj_mod(h)
    z = F.normalize(r, dim=1, eps=eps)
    return {"layer3": l3.mean(dim=(2, 3)), "h": h, "r": r, "z": z}


@torch.no_grad()
def encode_sites(enc: Encoder, images: np.ndarray, bs: int = 1024) -> dict[str, torch.Tensor]:
    R = enc.R; out = {s: [] for s in SITES}
    for s0 in range(0, len(images), bs):
        x = torch.from_numpy(np.ascontiguousarray(images[s0:s0 + bs])).to(enc.dev).permute(0, 3, 1, 2).float().div_(255.0)
        x = (x - enc.mean) / enc.std
        f = site_features(R["encoder"], R["projector"], x, R["eps"])
        for s in SITES:
            out[s].append(f[s].float().cpu())
    return {s: torch.cat(v) for s, v in out.items()}


def features(enc: Encoder, smoke: bool = False) -> tuple[dict, dict, object, np.ndarray, np.ndarray, dict]:
    """float32 site features of the FIT and selection images, cached under outputs/P141_features/<run>/ (one pass computes all four sites);
    QC: max |h − cached P124 h| when that cache exists (same images, same encoder → expected ~0)."""
    data, fit, sel = G.cifar100_split(enc)
    if smoke:
        fit, sel = fit[::30], sel[::10]
    tag = "smoke_" if smoke else ""; parts = {}; qc = {}
    for part, uids in (("fit", fit), ("sel", sel)):
        paths = {s: P141_CACHE / enc.run / f"{tag}c100_{part}_{s}_fp32.pt" for s in SITES}
        if all(p.is_file() for p in paths.values()):
            parts[part] = {s: torch.load(p, map_location="cpu") for s, p in paths.items()}
        else:
            t0 = time.time(); f = encode_sites(enc, data.data[uids])
            for s in SITES:
                atomic_torch_save(f[s], paths[s])
            parts[part] = f
            print(f"[{enc.run}] sites {part}: " + " ".join(f"{s}{tuple(f[s].shape)}" for s in SITES) + f" in {time.time() - t0:.0f}s", flush=True)
        p124 = G.P124_CACHE / enc.run / f"{tag}c100_{part}_h_fp32.pt"
        if p124.is_file():
            ref = torch.load(p124, map_location="cpu")
            qc[f"{part}_h_vs_p124_cache_maxabs"] = float((parts[part]["h"] - ref).abs().max()) if ref.shape == parts[part]["h"].shape else None
        qc[f"{part}_r_norm_mean"] = float(parts[part]["r"].norm(dim=1).mean()); qc[f"{part}_z_norm_mean"] = float(parts[part]["z"].norm(dim=1).mean())
    return parts["fit"], parts["sel"], data, fit, sel, qc


def tasks_from_features(hf: torch.Tensor, hs: torch.Tensor, L: dict, fit: np.ndarray, sel: np.ndarray, dev, readouts, tag: str) -> dict:
    """coarse / fine / conditional for one feature site — the P124 task loop (granularity.run_granularity) verbatim, on the given features."""
    cf, cs = torch.as_tensor(L["coarse"][fit]), torch.as_tensor(L["coarse"][sel])
    ff, fs = torch.as_tensor(L["fine"][fit]), torch.as_tensor(L["fine"][sel])
    out = {"dim": int(hf.shape[1]), "tasks": {}}
    for task, yf, ys, nc in (("coarse", cf, cs, 20), ("fine", ff, fs, 100)):
        r = G._readouts(hf, hs, yf, ys, nc, G.sel_seed(task), dev, readouts)
        if task == "fine" and "recipe_raw" in r:
            head = r["recipe_raw"]["head"]
            with torch.no_grad():
                logits = head(hs.to(dev).float()).cpu()
            out["fine_head_decomposition"] = G.fine_head_decomposition(logits.argmax(1), fs, L["f2c"])
            pm = G.masked_group_predictions(logits, cs, L["groups"])
            ok = (pm == fs).numpy(); per_g = {g: float(ok[(cs == g).numpy()].mean() * 100) for g in range(20) if (cs == g).any()}
            out["conditional_masked_fine_head"] = {"macro_pct": float(np.mean(list(per_g.values()))), "overall_pct": float(ok.mean() * 100), "per_group_pct": per_g}
        out["tasks"][task] = {k: {kk: vv for kk, vv in v.items() if kk != "head"} for k, v in r.items()}
        print(f"[{tag}] {task}: " + " ".join(f"{k} {v['acc_pct']:.2f}" for k, v in r.items()), flush=True)
    per = {k: {} for k in readouts}; corr = {k: 0.0 for k in readouts}; tot = 0
    for g in range(20):
        pf, lf = G.conditional_subset(L["fine"][fit], L["coarse"][fit], L["groups"], g)
        ps, ls = G.conditional_subset(L["fine"][sel], L["coarse"][sel], L["groups"], g)
        if len(ps) == 0 or len(np.unique(lf)) < 5:
            continue
        r = G._readouts(hf[pf], hs[ps], torch.as_tensor(lf), torch.as_tensor(ls), 5, G.sel_seed("conditional", g), dev, readouts)
        for k in readouts:
            per[k][g] = r[k]["acc_pct"]; corr[k] += r[k]["acc_pct"] * len(ps) / 100.0
        tot += len(ps)
    out["tasks"]["conditional"] = {k: {"macro_pct": float(np.mean(list(per[k].values()))), "overall_pct": 100.0 * corr[k] / max(tot, 1),
                                       "per_group_pct": per[k]} for k in readouts}
    print(f"[{tag}] conditional (macro): " + " ".join(f"{k} {v['macro_pct']:.2f}" for k, v in out["tasks"]["conditional"].items()), flush=True)
    return out


def run_layer_readout(enc: Encoder, dev, smoke: bool = False, readouts=DEFAULT_READOUTS, sites=SITES) -> dict:
    ff, fs, data, fit, sel, qc = features(enc, smoke=smoke)
    L = G.load_labels(data.targets)
    cfg = enc.R["cfg"]
    out = {"run": enc.run, "backbone": cfg["model"]["backbone"], "h_dim": int(cfg["model"]["h_dim"]),
           "projector": {k: cfg["model"]["projector"].get(k) for k in ("hidden_dim", "output_dim", "depth", "kind")},
           "train_normalization": cfg["model"]["normalization"]["vcs_and_simclr"], "eps": enc.R["eps"],
           "n_fit": int(len(fit)), "n_sel": int(len(sel)), "fit_sha": sha_idx(fit), "sel_sha": sha_idx(sel), "readouts": list(readouts),
           "knn": G.KNN, "qc": qc, "sites": {}}
    for s in sites:
        out["sites"][s] = tasks_from_features(ff[s], fs[s], L, fit, sel, dev, readouts, f"{enc.run}:{s}")
    return out
