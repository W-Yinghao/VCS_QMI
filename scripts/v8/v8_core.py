"""V8 core: the shared 5-logit rating head's outputs, the eight recipes' losses, and the grouped batch manifest (package docs/03, docs/04).

Outputs of logits [B, 5] with FIT priors pi [4] (thresholds H >= 2..5):  q = softmax, u_k = P(H >= k | Z) (stable log-sum-exp tails),
ell_k = logit(u_k) - logit(pi_k), T_k = tanh(ell_k / 2), mu = 1 + sum_k u_k.
Losses per image (target: the image's vote histogram; y_k = its tail):
  cs    mean_k [ y_k (1 - T_k)^2 / (8 pi_k) + (1 - y_k) (1 + T_k)^2 / (8 (1 - pi_k)) ]        (= 1/4 sum_k a_k (1 - J_k), a_k = 1/4)
  bce   mean_k [ y_k softplus(-ell_k) / (2 pi_k) + (1 - y_k) softplus(ell_k) / (2 (1 - pi_k)) ]  (balanced BCE on the same ell)
  rps   mean_k [ y_k (1 - u_k)^2 + (1 - y_k) u_k^2 ]   (random-rater form; differs from mean_k (u_k - y_k)^2 by a model-free constant)
  ce    - sum_h hist_h log q_h
  mse   ((mu - Hbar) / 4)^2          l1   |mu - Hbar| / 4
  rank  mse + RankNet BCE on (mu_i - mu_j) / tau over the subgroup's unordered pairs (target 1 / 0 / 0.5 for Hbar ties), lambda = 1
  list  mse + ListNet top-one CE, softmax(Hbar / tau) vs softmax(mu / tau), lambda = 1
Batch loss: point terms weighted by the natural prompt -> image weight w (sum w l / sum w over the batch); ranking terms averaged within each
subgroup of >= 2 images, then each subgroup weighted 1 / (number of the prompt's subgroups) and normalised over the batch's prompts.
Checked against the package reference (outputs/V8/package_ref_v1/.../reference/ordinal_torch.py) in tests/v8.
"""
from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F

LOSSES = ("cs", "bce", "rps", "ce", "mse", "l1", "rank", "list")
ARMS = {"U-CS": "cs", "U-BCE": "bce", "U-RPS": "rps", "U-CE": "ce", "U-MSE": "mse", "U-L1": "l1", "U-RANK": "rank", "U-LIST": "list"}
GROUP_MAX, BATCH_MAX, TAU, LAMBDA_RANK = 7, 14, 1.0, 1.0
LEVELS = torch.arange(1, 6)


def outputs(logits: torch.Tensor, prior: torch.Tensor) -> dict:
    if logits.ndim != 2 or logits.shape[-1] != 5 or prior.shape != (4,):
        raise ValueError("expected logits [B, 5] and prior [4]")
    if not ((prior > 0) & (prior < 1)).all():
        raise ValueError("degenerate threshold prior: no two-class P / Q")
    logits = logits.float(); prior = prior.to(logits)
    logq = torch.log_softmax(logits, -1)
    logu = torch.stack([torch.logsumexp(logq[:, k:], -1) for k in range(1, 5)], -1)
    log1mu = torch.stack([torch.logsumexp(logq[:, :k], -1) for k in range(1, 5)], -1)
    ell = logu - log1mu - torch.logit(prior); u = logu.exp()
    return {"q": logq.exp(), "logq": logq, "u": u, "ell": ell, "T": torch.tanh(ell / 2), "mu": 1 + u.sum(-1)}


def point_loss(o: dict, hist: torch.Tensor, prior: torch.Tensor, kind: str) -> torch.Tensor:
    """Per-image loss [B] of a point objective (rank / list use their mse part here)."""
    hist = hist.to(o["q"]); prior = prior.to(o["q"]); y = torch.flip(torch.cumsum(torch.flip(hist, [-1]), -1), [-1])[:, 1:]
    if kind == "cs":
        t = o["T"]; return (y * (1 - t).square() / (8 * prior) + (1 - y) * (1 + t).square() / (8 * (1 - prior))).mean(-1)
    if kind == "bce":
        e = o["ell"]; return (y * F.softplus(-e) / (2 * prior) + (1 - y) * F.softplus(e) / (2 * (1 - prior))).mean(-1)
    if kind == "rps":
        u = o["u"]; return (y * (1 - u).square() + (1 - y) * u.square()).mean(-1)
    if kind == "ce":
        return -(hist * o["logq"]).sum(-1)
    hbar = (hist * LEVELS.to(hist)).sum(-1)
    if kind in ("mse", "rank", "list"):
        return ((o["mu"] - hbar) / 4).square()
    if kind == "l1":
        return (o["mu"] - hbar).abs() / 4
    raise ValueError(kind)


def ranking_loss(mu: torch.Tensor, hbar: torch.Tensor, kind: str, tau: float = TAU) -> torch.Tensor:
    """One subgroup (one prompt, >= 2 images)."""
    if len(mu) < 2:
        return mu.sum() * 0
    if kind == "rank":
        i, j = torch.triu_indices(len(mu), len(mu), offset=1, device=mu.device)
        y = (hbar[i] > hbar[j]).to(mu.dtype) + 0.5 * (hbar[i] == hbar[j]).to(mu.dtype)
        return F.binary_cross_entropy_with_logits((mu[i] - mu[j]) / tau, y)
    if kind == "list":
        return -(torch.softmax(hbar / tau, -1) * torch.log_softmax(mu / tau, -1)).sum()
    raise ValueError(kind)


def batch_loss(logits: torch.Tensor, hist: torch.Tensor, w: torch.Tensor, prior: torch.Tensor, kind: str, subgroup: list, n_sub_of_prompt: list) -> tuple:
    """logits [B, 5]; hist [B, 5]; w [B] natural weights; subgroup[b] = subgroup id of image b; n_sub_of_prompt[b] = the prompt's subgroup count."""
    o = outputs(logits, prior); w = w.to(o["q"]); lp = point_loss(o, hist, prior, kind); point = (w * lp).sum() / w.sum(); parts = {"point": float(point.detach())}
    if kind not in ("rank", "list"):
        return point, parts, o
    hbar = (hist.to(o["q"]) * LEVELS.to(o["q"])).sum(-1); groups = {}
    for b, g in enumerate(subgroup):
        groups.setdefault(g, []).append(b)
    num, den, n_pairs = o["mu"].sum() * 0, 0.0, 0
    for g, idx in groups.items():
        if len(idx) < 2:
            continue
        ix = torch.tensor(idx, device=o["mu"].device); wt = 1.0 / n_sub_of_prompt[idx[0]]
        num = num + wt * ranking_loss(o["mu"][ix], hbar[ix], kind); den += wt; n_pairs += len(idx) * (len(idx) - 1) // 2
    rk = num / den if den > 0 else num
    parts.update({"ranking": float(rk.detach()), "ranking_subgroups": sum(1 for v in groups.values() if len(v) >= 2), "ranking_pairs": n_pairs})
    return point + LAMBDA_RANK * rk, parts, o


def split_sizes(n: int, cap: int = GROUP_MAX) -> list[int]:
    """ceil(n / cap) near-equal subgroups (15 -> 5, 5, 5; 13 -> 7, 6), so no singleton subgroup is created when it can be avoided."""
    k = math.ceil(n / cap); base, extra = divmod(n, k); return [base + (1 if i < extra else 0) for i in range(k)]


def batch_manifest(prompt_images: dict, seed: int, epoch: int) -> list[list[tuple]]:
    """The grouped batch plan shared by every arm at (seed, epoch): prompts shuffled, each prompt's images shuffled and split into near-equal
    subgroups of <= 7, subgroups shuffled and packed first-fit into batches of <= 14 images; every image appears exactly once (last batch kept).
    Returns [[(image_index, subgroup_id, n_subgroups_of_its_prompt), ...], ...]."""
    rng = random.Random(f"v8-batches-{seed}-{epoch}"); prompts = sorted(prompt_images); rng.shuffle(prompts); subs = []; gid = 0
    for p in prompts:
        imgs = list(prompt_images[p]); rng.shuffle(imgs); sizes = split_sizes(len(imgs)); s = 0
        for sz in sizes:
            subs.append((gid, imgs[s:s + sz], len(sizes))); s += sz; gid += 1
    rng.shuffle(subs); batches, cur = [], []
    for g, imgs, ns in subs:
        if len(cur) + len(imgs) > BATCH_MAX:
            batches.append(cur); cur = []
        cur += [(i, g, ns) for i in imgs]
    if cur:
        batches.append(cur)
    return batches


def warmup_cosine(step: int, total: int, lr: float, warmup: int = 100, start: float = 1e-8) -> float:
    """Linear warm-up from 1e-8 over 100 optimiser steps, then cosine to 0 over the remaining actual steps (the same for every recipe)."""
    if step < warmup:
        return start + (lr - start) * step / warmup
    return 0.5 * lr * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))
