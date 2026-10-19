"""V9-N2 real-data loader for the NCR / SGRAF 'precomp' layout (f30k_precomp, coco_precomp: <split>_ims.npy (5N or N, 36, 2048) and
<split>_caps.txt, one caption per line, five per image in order; vocab <name>_vocab.json from SCAN).  Tokenisation follows the official code
(nltk word_tokenize, lower-case, <start> ... <end>, unknown -> <unk>).  Nothing here injects noise or reads the official noise index: the V9
mapping comes from noise.py, and the official index is read only by the A-table native reproduction (fork A).  Train captions are tokenised once
and cached next to the data with the vocab's sha256 so every unit reads identical token ids.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def load_vocab(path: str | Path) -> dict:
    v = json.loads(Path(path).read_text())
    return v["word2idx"] if "word2idx" in v else v


def tokenize(caption: str, word2idx: dict) -> list[int]:
    import nltk  # official dependency; the v9_sgr env must provide it with the punkt data
    tokens = nltk.tokenize.word_tokenize(str(caption).lower())
    unk = word2idx.get("<unk>")
    return [word2idx["<start>"]] + [word2idx.get(t, unk) for t in tokens] + [word2idx["<end>"]]


def load_precomp(data_dir: str | Path, split: str, word2idx: dict, cache: bool = True, cache_dir: str | Path | None = None) -> dict:
    """Returns {'images': (N, 36, 2048) float32 unique images, 'captions': {cid: tokens}, 'caption_image': {cid: image index}, 'hash': ...}."""
    d = Path(data_dir); ims = np.load(d / f"{split}_ims.npy"); caps_txt = (d / f"{split}_caps.txt").read_text().splitlines()
    im_div = 5 if ims.shape[0] != len(caps_txt) else 1
    if im_div == 1 and len(caps_txt) % 5 == 0 and ims.shape[0] == len(caps_txt):   # rkiros layout duplicates images 5x; keep unique rows
        ims = ims[::5]
    if ims.shape[0] * 5 != len(caps_txt):
        raise ValueError(f"{split}: {ims.shape[0]} images vs {len(caps_txt)} captions (expected 5 per image)")
    vh = hashlib.sha256(json.dumps(sorted(word2idx.items())).encode()).hexdigest()[:16]
    cdir = Path(cache_dir) if cache_dir else d; cdir.mkdir(parents=True, exist_ok=True) if cache_dir else None
    cpath = cdir / f"{d.name}_{split}_caps_tokens_{vh}.json"                   # the data directory is read-only for V9: caches go to cache_dir
    if cache and cpath.exists():
        tokens = json.loads(cpath.read_text())
    else:
        tokens = [tokenize(c, word2idx) for c in caps_txt]
        if cache:
            cpath.write_text(json.dumps(tokens))
    captions = {i: t for i, t in enumerate(tokens)}; caption_image = {i: i // 5 for i in range(len(tokens))}
    h = hashlib.sha256(np.ascontiguousarray(ims[:64]).tobytes() + json.dumps(tokens[:256]).encode()).hexdigest()
    return {"images": ims.astype(np.float32), "captions": captions, "caption_image": caption_image, "hash": h, "n_images": int(ims.shape[0]), "split": split}


def eval_arrays(ds: dict) -> tuple[np.ndarray, list[list[int]]]:
    """Official order: image i's captions are 5i..5i+4 (the loader keeps that order)."""
    return ds["images"], [ds["captions"][i] for i in range(len(ds["captions"]))]
