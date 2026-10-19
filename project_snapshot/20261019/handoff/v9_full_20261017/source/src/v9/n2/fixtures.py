"""Synthetic fixtures for the V9-N2 code path (CPU gate only; never evidence).  A fake 'precomp' dataset in the SGRAF layout: images with
`n_region` random region features (l2-normalised, as the real BUTD features are), five token-id captions per image from a tiny vocabulary, a
planted dependence so that true pairs score higher than product pairs once the host is trained for a few steps (caption tokens are drawn from the
image's hidden 'topic'), and the real NCR file layout (train_ims.npy, train_caps.txt-like token lists) is NOT imitated here — the loader for the
real data lives in data.py and reads the official files.
"""
from __future__ import annotations

import numpy as np
import torch


def make_fixture(n_images: int = 60, n_caps: int = 5, n_region: int = 6, img_dim: int = 32, vocab: int = 40, seed: int = 0, min_len: int = 4, max_len: int = 9):
    rng = np.random.default_rng(seed)
    topics = rng.integers(0, 4, size=n_images)
    centers = rng.normal(size=(4, n_region, img_dim))
    images = centers[topics] + 0.3 * rng.normal(size=(n_images, n_region, img_dim))
    images = images / (np.linalg.norm(images, axis=-1, keepdims=True) + 1e-8)
    caption_image, captions = {}, {}
    for i in range(n_images):
        for c in range(n_caps):
            cid = i * n_caps + c; caption_image[cid] = i
            L = int(rng.integers(min_len, max_len + 1)); lo = 3 + 9 * int(topics[i])
            body = rng.integers(lo, lo + 9, size=L - 2).tolist()
            captions[cid] = [1] + body + [2]              # <start>=1, <end>=2, <pad>=0
    return {"images": images.astype(np.float32), "captions": captions, "caption_image": caption_image, "vocab": vocab, "n_region": n_region, "img_dim": img_dim}


def collate_pairs(images: np.ndarray, captions: dict, rows: list[dict]):
    """rows: observed pair rows (caption_id, observed_image_id).  Returns (img tensor (B, R, D), cap tensor (B, L) long, lengths list)."""
    ims = torch.from_numpy(np.stack([images[int(r["observed_image_id"])] for r in rows])).float()
    caps = [captions[int(r["caption_id"])] for r in rows]; lengths = [len(c) for c in caps]
    out = torch.zeros(len(caps), max(lengths), dtype=torch.long)
    for i, c in enumerate(caps):
        out[i, :len(c)] = torch.as_tensor(c)
    return ims, out, lengths
