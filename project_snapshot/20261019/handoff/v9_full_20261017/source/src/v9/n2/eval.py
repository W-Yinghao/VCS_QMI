"""V9-N2 retrieval evaluation (package §05.8, §08.2) with the official SGRAF / NCR semantics: captions 5i..5i+4 belong to image i (Flickr30K
and COCO precomp layout); I2T R@K = any of the image's captions in the top K; T2I R@K = the caption's image in the top K; rSum = the six recalls,
mR = rSum / 6; COCO: 5K full gallery (main) and the mean over 5 folds of 1K (separate column), Flickr: the 1K test.  Ties: the official code uses
numpy argsort (unstable quicksort); here the primary order is the fixed caption / image id order (stable argsort on -score) and a random-tie
sensitivity re-draws the order with a seeded jitter.  Checkpoint selection uses the official dev split's rSum only (never test).
"""
from __future__ import annotations

import numpy as np
import torch

from .sgr_host import SGRHost


@torch.no_grad()
def similarity_matrix(host: SGRHost, images: np.ndarray, captions: list[list[int]], device: str = "cpu", img_batch: int = 100, cap_batch: int = 100) -> np.ndarray:
    """images (n_img, R, D) — one row per image (no 5x redundancy); captions = list of token lists in official order.  Returns (n_img, n_cap)."""
    host.eval()
    img_embs = []
    for s in range(0, len(images), img_batch):
        img_embs.append(host.img_enc(torch.from_numpy(images[s:s + img_batch]).float().to(device)))
    img_embs = torch.cat(img_embs, 0)
    cap_embs, cap_lens = [], []
    for s in range(0, len(captions), cap_batch):
        chunk = captions[s:s + cap_batch]; lens = [len(c) for c in chunk]; t = torch.zeros(len(chunk), max(lens), dtype=torch.long)
        for i, c in enumerate(chunk):
            t[i, :len(c)] = torch.as_tensor(c)
        emb = host.txt_enc(t.to(device), lens)
        for i in range(len(chunk)):
            cap_embs.append(emb[i, :lens[i]].clone()); cap_lens.append(lens[i])
    sims = np.zeros((len(images), len(captions)), dtype=np.float64)
    for cs in range(0, len(captions), cap_batch):
        ce = min(cs + cap_batch, len(captions)); L = max(cap_lens[cs:ce]); pad = torch.zeros(ce - cs, L, img_embs.size(-1), device=device)
        for i in range(cs, ce):
            pad[i - cs, :cap_lens[i]] = cap_embs[i]
        for is_ in range(0, len(images), img_batch):
            ie = min(is_ + img_batch, len(images))
            sims[is_:ie, cs:ce] = host.sim_enc(img_embs[is_:ie], pad, cap_lens[cs:ce]).cpu().numpy()
    return sims


def _order(scores: np.ndarray, tie: str, rng=None) -> np.ndarray:
    if tie == "random":
        scores = scores + rng.uniform(0, 1e-9, size=scores.shape)
    return np.argsort(-scores, kind="stable")


def recall_metrics(sims: np.ndarray, per_captions: int = 5, tie: str = "stable", seed: int = 0, ks=(1, 5, 10)) -> dict:
    n_img, n_cap = sims.shape
    if n_cap != per_captions * n_img:
        raise ValueError("captions must be per_captions * images in official order")
    rng = np.random.default_rng(seed) if tie == "random" else None
    ranks_i2t = np.zeros(n_img, dtype=int)
    for i in range(n_img):
        inds = _order(sims[i], tie, rng); pos = np.empty(n_cap, dtype=int); pos[inds] = np.arange(n_cap)
        ranks_i2t[i] = pos[per_captions * i: per_captions * i + per_captions].min()
    ranks_t2i = np.zeros(n_cap, dtype=int); simsT = sims.T
    for j in range(n_cap):
        inds = _order(simsT[j], tie, rng); ranks_t2i[j] = int(np.where(inds == j // per_captions)[0][0])
    out = {}
    for k in ks:
        out[f"i2t_R{k}"] = 100.0 * float(np.mean(ranks_i2t < k))
    for k in ks:
        out[f"t2i_R{k}"] = 100.0 * float(np.mean(ranks_t2i < k))
    out["rsum"] = float(sum(out[f"i2t_R{k}"] for k in ks) + sum(out[f"t2i_R{k}"] for k in ks)); out["mR"] = out["rsum"] / (2 * len(ks))
    out["i2t_medr"] = float(np.floor(np.median(ranks_i2t)) + 1); out["t2i_medr"] = float(np.floor(np.median(ranks_t2i)) + 1); out["tie"] = tie
    return out


def evaluate_split(sims: np.ndarray, dataset: str, per_captions: int = 5, random_tie_seeds=(1, 2, 3)) -> dict:
    """dataset 'f30k': 1K full; 'coco': 5K full (main) + 5 x 1K folds mean (separate).  Both tie treatments."""
    out = {"primary": recall_metrics(sims, per_captions, "stable")}
    out["random_tie"] = [recall_metrics(sims, per_captions, "random", seed=s) for s in random_tie_seeds]
    if dataset == "coco" and sims.shape[0] == 5000:
        folds = []
        for f in range(5):
            folds.append(recall_metrics(sims[f * 1000:(f + 1) * 1000, f * 5000:(f + 1) * 5000], per_captions, "stable"))
        keys = [k for k in folds[0] if isinstance(folds[0][k], float)]
        out["coco_5x1k_mean"] = {k: float(np.mean([fo[k] for fo in folds])) for k in keys}; out["coco_5x1k_folds"] = folds
    elif dataset == "coco":
        out["coco_5x1k_mean"] = "unavailable (sims is not the 5K test)"
    return out


def reference_cross_check(sims: np.ndarray, per_captions: int = 5) -> dict:
    """The package's `retrieval_recall` on the same matrix (stable ties, same GT); must agree with recall_metrics(..., 'stable')."""
    from .weights import REF
    n_img = sims.shape[0]; image_ids = list(range(n_img)); cap_img = [j // per_captions for j in range(sims.shape[1])]
    return REF.retrieval_recall(sims, image_ids, cap_img)
