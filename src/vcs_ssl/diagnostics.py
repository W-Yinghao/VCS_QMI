"""Frozen-feature evaluation and diagnostics (spec §10, §11.2).

All functions here take *frozen* modules (eval mode, ``requires_grad`` off, ``torch.no_grad``) and verify by hashing that
encoder/projector/critic state (parameters *and* BN buffers) is bit-identical before and after.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from reference.ssl_core import cyclic_negative_indices, vcs_from_scores

from .data.datasets import LabeledCleanDataset, TwoViewNoLabelEvalDataset, make_eval_loader
from .utils import Timer, state_dict_sha256


# ----------------------------------------------------------------------------------------------------------------------
# freezing helpers
# ----------------------------------------------------------------------------------------------------------------------
def freeze(*modules: nn.Module | None) -> dict[str, str]:
    hashes = {}
    for i, m in enumerate(modules):
        if m is None:
            continue
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        hashes[str(i)] = state_dict_sha256(m)
    return hashes


def assert_unchanged(before: dict[str, str], *modules: nn.Module | None) -> None:
    for i, m in enumerate(modules):
        if m is None:
            continue
        after = state_dict_sha256(m)
        if after != before[str(i)]:
            raise RuntimeError(f"frozen module {i} changed during evaluation (parameters or BN buffers)")


# ----------------------------------------------------------------------------------------------------------------------
# feature extraction
# ----------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def extract_features(encoder: nn.Module, projector: nn.Module | None, images: np.ndarray, targets: np.ndarray, uids: np.ndarray,
                     transform, *, device: torch.device, batch_size: int = 512, num_workers: int = 4, seed: int = 0,
                     l2_eps: float = 1e-8) -> dict[str, Any]:
    """Clean-transform features in UID order: h [N,512], p_raw [N,128], z_l2 [N,128] (float32, CPU)."""
    before = freeze(encoder, projector)
    gen = torch.Generator().manual_seed(seed)
    ds = LabeledCleanDataset(images, targets, uids, transform)
    loader = make_eval_loader(ds, batch_size=batch_size, num_workers=num_workers, generator=gen, pin_memory=device.type == "cuda")
    hs, ps, ys, us = [], [], [], []
    with Timer(device) as t:
        for x, y, u in loader:
            x = x.to(device, non_blocking=True)
            h = encoder(x)
            hs.append(h.float().cpu())
            if projector is not None:
                p = projector(h)
                ps.append(p.float().cpu())
            ys.append(y)
            us.append(u)
    assert_unchanged(before, encoder, projector)
    h = torch.cat(hs)
    out: dict[str, Any] = {"h": h, "labels": torch.cat(ys), "uids": torch.cat(us), "seconds": t.elapsed}
    if projector is not None:
        p = torch.cat(ps)
        out["p_raw"] = p
        out["z_l2"] = F.normalize(p, dim=1, eps=l2_eps)
    if not torch.equal(out["uids"], torch.as_tensor(uids, dtype=torch.int64)):
        raise RuntimeError("feature order does not match requested UID order")
    return out


# ----------------------------------------------------------------------------------------------------------------------
# kNN monitor (spec §10.3)
# ----------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def knn_eval(h_bank: Tensor, y_bank: Tensor, h_query: Tensor, y_query: Tensor, *, k: int = 200, temperature: float = 0.1,
             chunk: int = 256, n_classes: int = 10, device: torch.device | None = None) -> dict[str, Any]:
    device = device or torch.device("cpu")
    bank = F.normalize(h_bank.to(device).float(), dim=1)
    yb = y_bank.to(device)
    correct = 0
    total = 0
    for s in range(0, len(h_query), chunk):
        q = F.normalize(h_query[s:s + chunk].to(device).float(), dim=1)
        sims = q @ bank.T  # [c, Nbank]
        top_s, top_i = sims.topk(k, dim=1)
        w = torch.exp((top_s - top_s.max(dim=1, keepdim=True).values) / temperature)
        votes = torch.zeros(q.shape[0], n_classes, device=device)
        votes.scatter_add_(1, yb[top_i], w)
        pred = votes.argmax(dim=1)
        correct += int((pred.cpu() == y_query[s:s + chunk]).sum())
        total += q.shape[0]
    return {"knn_val_top1_pct": 100.0 * correct / total, "n_query": total, "n_bank": int(len(h_bank)), "k": k,
            "temperature": temperature, "weighting": "exp((s_j - max_k s_k)/T) over cosine top-k"}


# ----------------------------------------------------------------------------------------------------------------------
# spectrum / collapse diagnostics (spec §11.2)
# ----------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def spectrum_stats(x: Tensor, *, center: bool = True, ddof: int = 1, eig_tol: float = 1e-6) -> dict[str, Any]:
    x = x.double()
    n, d = x.shape
    if n < 2:
        raise ValueError("need at least two samples")
    norms = x.norm(dim=1)
    xc = x - x.mean(0, keepdim=True) if center else x
    cov = xc.T @ xc / (n - ddof)
    trace = float(cov.diagonal().sum())
    evals = torch.linalg.eigvalsh(cov)  # ascending
    neg_min = float(evals.min())
    tol = eig_tol * max(abs(float(evals.max())), 1e-300)
    numerical_problem = neg_min < -tol
    ev = evals.clamp_min(0.0)
    out: dict[str, Any] = {
        "n": int(n), "dim": int(d), "mean_per_dim_variance": trace / d, "covariance_trace": trace,
        "norm_mean": float(norms.mean()), "norm_std": float(norms.std(unbiased=True)),
        "eigenvalues_desc": ev.flip(0).tolist(), "min_raw_eigenvalue": neg_min, "negative_eig_within_tolerance": not numerical_problem,
    }
    if trace <= 1e-8 or float(ev.sum()) <= 0.0:
        out.update({"effective_rank": 0.0, "top_eig_fraction": None, "zero_covariance": True})
    else:
        p = ev / ev.sum()
        p = p[p > 0]
        out.update({"effective_rank": float(torch.exp(-(p * p.log()).sum())), "top_eig_fraction": float(ev.max() / ev.sum()),
                    "zero_covariance": False})
    out["collapse_flag"] = bool(trace <= 1e-8 or out["effective_rank"] <= 2.0)
    return out


def spectrum_report(feats: dict[str, Any], names: tuple[str, ...] = ("h", "p_raw", "z_l2"), **kw) -> dict[str, Any]:
    return {n: spectrum_stats(feats[n], **kw) for n in names if n in feats}


# ----------------------------------------------------------------------------------------------------------------------
# critic held-out diagnostic (spec §10.4) — VCS only
# ----------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def critic_holdout(encoder: nn.Module, projector: nn.Module, critic: nn.Module, images: np.ndarray, sel_uids: np.ndarray,
                   two_view_transform, *, device: torch.device, batch_size: int, repeats: int, rng_seed: int, k: int,
                   num_workers: int = 4, l2_eps: float = 1e-8, normalize_input: bool = True, symmetric: bool = False,
                   feature_source: str = "z") -> dict[str, Any]:
    """Train-distribution two-view pairs on selection images; whole model eval; global (count-weighted) means."""
    before = freeze(encoder, projector, critic)
    n = len(sel_uids)
    per_repeat = []
    hist_bins = torch.linspace(-1, 1, 41)
    with Timer(device) as t:
        for r in range(repeats):
            seed = rng_seed + r
            gen = torch.Generator().manual_seed(seed)  # loader/augmentation stream
            pair_gen = torch.Generator().manual_seed(seed + 1)  # shift stream
            ds = TwoViewNoLabelEvalDataset(images, sel_uids, two_view_transform)
            loader = make_eval_loader(ds, batch_size=batch_size, num_workers=num_workers, generator=gen, pin_memory=device.type == "cuda")
            pos_sum = pos_sq = neg_sum = neg_sq = 0.0
            n_pos = n_neg = 0
            sat_pos = sat_neg = 0
            pos_hist = torch.zeros(40)
            neg_hist = torch.zeros(40)
            batches = list(loader)  # materialize to apply the "merge a size-1 tail into the previous batch" rule
            if len(batches) >= 2 and batches[-1][0].shape[0] == 1:
                a, b = batches[-2], batches[-1]
                batches[-2] = tuple(torch.cat((u, v), dim=0) for u, v in zip(a, b))
                batches.pop()
            shifts = []
            k_eff_min = k
            for x1, x2, _ in batches:
                bsz = x1.shape[0]
                if bsz < 2:
                    raise RuntimeError("a batch of size 1 remained after merging; cannot form negatives")
                h_all = encoder(torch.cat((x1, x2)).to(device))
                if feature_source == "h_l2":
                    z = F.normalize(h_all, dim=1, eps=l2_eps)
                else:
                    p_all = projector(h_all)
                    z = F.normalize(p_all, dim=1, eps=l2_eps) if normalize_input else p_all
                z1, z2 = z.chunk(2, dim=0)
                k_eff = min(k, bsz - 1)  # a short tail batch cannot host K distinct nonzero shifts; capped and recorded
                k_eff_min = min(k_eff_min, k_eff)
                idx, sh = cyclic_negative_indices(bsz, k_eff, generator=pair_gen, device=z.device)
                shifts.append(int(sh[0]))
                tp = critic(z1, z2)
                tn = critic(z1.unsqueeze(0).expand(k_eff, -1, -1).reshape(-1, z1.shape[1]), z2[idx].reshape(-1, z2.shape[1]))
                if symmetric:  # named variant: also score the reversed order with the same shifts
                    tp = torch.cat((tp, critic(z2, z1)))
                    tn = torch.cat((tn, critic(z2.unsqueeze(0).expand(k_eff, -1, -1).reshape(-1, z2.shape[1]), z1[idx].reshape(-1, z1.shape[1]))))
                pos_sum += float(tp.sum()); pos_sq += float(tp.square().sum()); n_pos += tp.numel()
                neg_sum += float(tn.sum()); neg_sq += float(tn.square().sum()); n_neg += tn.numel()
                sat_pos += int((tp.abs() > 0.95).sum()); sat_neg += int((tn.abs() > 0.95).sum())
                pos_hist += torch.histc(tp.float().cpu(), bins=40, min=-1, max=1)
                neg_hist += torch.histc(tn.float().cpu(), bins=40, min=-1, max=1)
            mp, mq, sp, sq = pos_sum / n_pos, neg_sum / n_neg, pos_sq / n_pos, neg_sq / n_neg
            j = mp - mq - 0.5 * sp - 0.5 * sq
            per_repeat.append({"repeat": r, "seed": seed, "n_pos": n_pos, "n_neg": n_neg, "k_effective_min": k_eff_min, "heldout_J": j, "heldout_R_binary": 1.0 - j,
                               "t_pos_mean": mp, "t_neg_mean": mq, "t_pos_second": sp, "t_neg_second": sq,
                               "sat_pos_frac": sat_pos / n_pos, "sat_neg_frac": sat_neg / n_neg,
                               "shifts": shifts, "pos_hist": pos_hist.tolist(), "neg_hist": neg_hist.tolist()})
    assert_unchanged(before, encoder, projector, critic)
    js = torch.tensor([r["heldout_J"] for r in per_repeat], dtype=torch.float64)
    return {"n_selection_images": int(n), "repeats": repeats, "batch_size": batch_size, "k": k,
            "heldout_J_mean": float(js.mean()), "heldout_J_sd": float(js.std(unbiased=True)) if repeats > 1 else None,
            "heldout_R_binary_mean": float(1.0 - js.mean()), "per_repeat": per_repeat, "hist_bin_edges": hist_bins.tolist(),
            "seconds": t.elapsed,
            "critic_input": ("h_l2" if feature_source == "h_l2" else ("z_l2" if normalize_input else "p_raw")), "symmetric": symmetric,
            "note": "diagnostic value of the current critic on images not used for SSL fit; not a refit supremum, not S, not Shannon MI"}


# ----------------------------------------------------------------------------------------------------------------------
# linear probe (spec §10.2)
# ----------------------------------------------------------------------------------------------------------------------
def linear_probe(h_fit: Tensor, y_fit: Tensor, h_sel: Tensor, y_sel: Tensor, lcfg: dict[str, Any], *, device: torch.device,
                 n_classes: int = 10) -> dict[str, Any]:
    if lcfg["normalize_h"] or lcfg["head"] != "linear_with_bias" or lcfg["optimizer"] != "sgd" or lcfg["schedule"] != "cosine" \
            or lcfg["checkpoint_rule"] != "final_probe_epoch":
        raise ValueError("probe config deviates from the frozen pilot protocol")
    gen = torch.Generator().manual_seed(lcfg["seed"])
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(lcfg["seed"])
        head = nn.Linear(h_fit.shape[1], n_classes, bias=True).to(device)  # PyTorch default init
    opt = torch.optim.SGD(head.parameters(), lr=lcfg["lr"], momentum=lcfg["momentum"], weight_decay=lcfg["weight_decay"])
    xf, yf = h_fit.to(device).float(), y_fit.to(device)
    xs, ys = h_sel.to(device).float(), y_sel.to(device)
    n, bs, epochs = len(xf), lcfg["batch_size"], lcfg["epochs"]
    steps_per_epoch = math.ceil(n / bs)
    total = epochs * steps_per_epoch
    step = 0
    curve = []
    with Timer(device) as t:
        for ep in range(epochs):
            perm = torch.randperm(n, generator=gen).to(device)
            head.train()
            run_loss = 0.0
            for s in range(0, n, bs):
                progress = step / max(total - 1, 1)
                lr = lcfg["lr"] * (lcfg["min_lr_ratio"] + (1 - lcfg["min_lr_ratio"]) * (1 + math.cos(math.pi * progress)) / 2)
                for g in opt.param_groups:
                    g["lr"] = lr
                idx = perm[s:s + bs]
                loss = F.cross_entropy(head(xf[idx]), yf[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                run_loss += float(loss.detach()) * len(idx)
                step += 1
            if (ep + 1) % 10 == 0 or ep == epochs - 1:
                head.eval()
                with torch.no_grad():
                    logits = head(xs)
                    acc = float((logits.argmax(1) == ys).float().mean()) * 100
                    ce = float(F.cross_entropy(logits, ys))
                curve.append({"probe_epoch": ep + 1, "train_ce": run_loss / n, "val_top1_pct": acc, "val_ce": ce, "lr_end": lr})
    final = curve[-1]
    return {"linear_val_top1_pct": final["val_top1_pct"], "linear_val_ce": final["val_ce"], "final_train_ce": final["train_ce"],
            "probe_epochs": epochs, "probe_steps": step, "curve": curve, "seconds": t.elapsed, "selection_rule": "final probe epoch",
            "n_fit": int(n), "n_selection": int(len(xs))}
