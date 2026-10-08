"""VL1 dataset adapter: RefCOCOg, UMD split (Nagaraja et al.), COCO-2014-train images read from the local COCO-2017 copy (same image ids).

Records are per base image.  Region set of the pair law = the image's REFERRED objects (each has >= 1 expression; RefCOCOg refers to one box
per ref, so no multi-box targets); all of a referred object's expressions are aliases of the same target (separate phrase columns, each with
one nonzero entry).  The image's other COCO objects are kept as `distractors` for a secondary, harder task-candidate set (task panel only:
they have no positive phrase, so they cannot be rows of P).  Canonical image keys: COCO image id and the Flickr photo id parsed from
flickr_url (for de-duplication against Flickr-sourced datasets such as Flickr30k).
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path("/projects/EEG-foundation-model/yinghao/datasets/refcocog_umd/refcocog")
COCO_DIRS = (Path("/projects/common/coco/train2017"), Path("/projects/common/coco/val2017"))
ARCHIVE = {"url": "https://web.archive.org/web/20220413012904/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip",
           "sha256": "3d1f7e5b2ff2205940bf59de55f861f5f2cc1403fb980669933a7f9af1aa8211",
           "route": "lichengunc/refer issues #14 / #22 / #26 (author's MAttNet commit c66ae95 points to these Internet Archive snapshots)"}
SPLIT_SEED, CAL_FRAC, DEV_FRAC, FIT_NS = 20261008, 0.1, 0.1, (1000, 4000, 16000)


@dataclass
class Obj:
    ann_id: int
    box_xywh: tuple[float, float, float, float]   # COCO convention: x, y (top-left, 0-based float), width, height
    category_id: int
    area: float
    iscrowd: int
    expressions: list[dict] = field(default_factory=list)   # [{"sent_id", "raw", "n_tokens"}] — empty for distractors


@dataclass
class Scene:
    image_id: int
    umd_split: str
    width: int
    height: int
    flickr_id: str | None
    referred: list[Obj]
    distractors: list[Obj]
    path: str | None = None

    def compatibility(self) -> tuple[np.ndarray, list[int], list[int]]:
        """A [m regions x k expressions] with A[r, w] = 1 iff expression w refers to region r; also (ann ids, sent ids)."""
        cols = [(i, e["sent_id"]) for i, o in enumerate(self.referred) for e in o.expressions]
        A = np.zeros((len(self.referred), len(cols)))
        for j, (i, _) in enumerate(cols):
            A[i, j] = 1.0
        return A, [o.ann_id for o in self.referred], [s for _, s in cols]


def flickr_id_of(url: str | None) -> str | None:
    m = re.search(r"/(\d+)_[0-9a-f]+(?:_[a-z])?\.jpg", url or "")
    return m.group(1) if m else None


def clip_box(b, w: int, h: int) -> tuple[float, float, float, float] | None:
    """xywh -> clipped xyxy in continuous pixel coordinates; None if degenerate (< 1 px in either side after clipping)."""
    x0, y0 = max(0.0, b[0]), max(0.0, b[1]); x1, y1 = min(float(w), b[0] + b[2]), min(float(h), b[1] + b[3])
    return (x0, y0, x1, y1) if (x1 - x0 >= 1.0 and y1 - y0 >= 1.0) else None


def coco_path(image_id: int) -> str | None:
    for d in COCO_DIRS:
        p = d / f"{image_id:012d}.jpg"
        if p.exists():
            return str(p)
    return None


def load_scenes(root: Path = ROOT, resolve_paths: bool = True) -> tuple[list[Scene], dict]:
    refs = pickle.load(open(root / "refs(umd).p", "rb"))
    inst = json.load(open(root / "instances.json"))
    imgs = {im["id"]: im for im in inst["images"]}
    anns_by_img: dict[int, list] = {}
    for a in inst["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    expr: dict[int, list] = {}; split_of: dict[int, set] = {}
    for r in refs:
        expr.setdefault(r["ann_id"], []).extend({"sent_id": s["sent_id"], "raw": s["raw"], "n_tokens": len(s["tokens"]), "ref_id": r["ref_id"]}
                                                for s in r["sentences"])
        split_of.setdefault(r["image_id"], set()).add(r["split"])
    stats = {"n_refs": len(refs), "n_sentences": sum(len(r["sentences"]) for r in refs), "images_with_mixed_split": 0,
             "degenerate_referred_boxes": 0, "degenerate_distractor_boxes": 0, "crowd_distractors": 0}
    scenes = []
    for iid in sorted(split_of):
        sp = split_of[iid]
        if len(sp) != 1:
            stats["images_with_mixed_split"] += 1; continue
        im = imgs[iid]; W, H = im["width"], im["height"]; ref, dis = [], []
        for a in anns_by_img.get(iid, []):
            o = Obj(a["id"], tuple(a["bbox"]), a["category_id"], a["area"], a["iscrowd"], list(expr.get(a["id"], [])))
            ok = clip_box(o.box_xywh, W, H) is not None
            if o.expressions:
                if ok:
                    ref.append(o)
                else:
                    stats["degenerate_referred_boxes"] += 1
            elif ok and not a["iscrowd"]:
                dis.append(o)
            else:
                stats["crowd_distractors" if a["iscrowd"] else "degenerate_distractor_boxes"] += 1
        ref.sort(key=lambda o: o.ann_id); dis.sort(key=lambda o: o.ann_id)
        scenes.append(Scene(iid, next(iter(sp)), W, H, flickr_id_of(im.get("flickr_url")), ref, dis, coco_path(iid) if resolve_paths else None))
    stats["categories"] = {c["id"]: c["name"] for c in inst["categories"]}
    return scenes, stats


def dev_roles(scenes: list[Scene], seed: int = SPLIT_SEED) -> dict[int, str]:
    """UMD train images -> FIT / CAL / DEV by image (10 % / 10 %, seeded permutation of sorted ids); UMD val -> VAL_OFFICIAL; test -> TEST."""
    train = sorted(s.image_id for s in scenes if s.umd_split == "train")
    perm = np.random.default_rng(seed).permutation(len(train)); n_cal, n_dev = int(round(CAL_FRAC * len(train))), int(round(DEV_FRAC * len(train)))
    role = {train[i]: ("CAL" if k < n_cal else "DEV" if k < n_cal + n_dev else "FIT") for k, i in enumerate(perm)}
    for s in scenes:
        if s.umd_split != "train":
            role[s.image_id] = "VAL_OFFICIAL" if s.umd_split == "val" else "TEST_CLOSED"
    return role


def fit_prefixes(scenes: list[Scene], role: dict[int, str], seed: int = SPLIT_SEED, ns=FIT_NS) -> dict[int, list[int]]:
    """Nested FIT prefixes of one fixed permutation (eligible = >= 2 referred objects); N above the available count -> all FIT images."""
    fit = sorted(s.image_id for s in scenes if role[s.image_id] == "FIT" and len(s.referred) >= 2)
    perm = [fit[i] for i in np.random.default_rng(seed + 1).permutation(len(fit))]
    return {n: perm[:min(n, len(perm))] for n in ns}


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()
