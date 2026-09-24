"""Fixed stratified split of the official CIFAR-10 training indices into ``fit`` (45k) and ``selection`` (5k).

Algorithm (spec §4.1, frozen): the official train index is the UID.  With ``numpy.random.default_rng(split_seed)``,
for class id 0..9 in order, permute all indices of that class; the first ``val_per_class`` (500) go to ``selection``
and the rest to ``fit``.  Each split is finally sorted by UID.  All runs share the same manifest file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..utils import atomic_write_json, canonical_json, read_json, sha256_bytes, utc_now

SPLIT_NAME = "dev45k_val5k"
SPLIT_ALGORITHM = ("numpy.random.default_rng(split_seed); for class c in 0..9 (in order): perm = rng.permutation(indices_of_c); "
                   "selection += perm[:val_per_class]; fit += perm[val_per_class:]; sort each split by UID")


def build_manifest(targets: np.ndarray, *, split_seed: int, val_per_class: int, source: dict[str, Any],
                   file_hashes: dict[str, Any]) -> dict[str, Any]:
    targets = np.asarray(targets, dtype=np.int64)
    n_classes = int(targets.max()) + 1
    rng = np.random.default_rng(split_seed)
    sel: list[np.ndarray] = []
    fit: list[np.ndarray] = []
    for c in range(n_classes):
        idx = np.flatnonzero(targets == c)
        perm = rng.permutation(idx)
        sel.append(perm[:val_per_class])
        fit.append(perm[val_per_class:])
    sel_arr = np.sort(np.concatenate(sel))
    fit_arr = np.sort(np.concatenate(fit))
    manifest: dict[str, Any] = {
        "split": SPLIT_NAME,
        "created_utc": utc_now(),
        "source": source,
        "raw_file_hashes": file_hashes,
        "split_algorithm": SPLIT_ALGORITHM,
        "split_seed": int(split_seed),
        "val_per_class": int(val_per_class),
        "uid_definition": "position in the official CIFAR-10 training partition (0..49999)",
        "n_total": int(len(targets)),
        "n_fit": int(len(fit_arr)),
        "n_selection": int(len(sel_arr)),
        "fit_class_counts": np.bincount(targets[fit_arr], minlength=n_classes).tolist(),
        "selection_class_counts": np.bincount(targets[sel_arr], minlength=n_classes).tolist(),
        "fit_uids": fit_arr.tolist(),
        "selection_uids": sel_arr.tolist(),
        "official_test_included": False,
    }
    _check_manifest(manifest)
    manifest["manifest_sha256"] = manifest_hash(manifest)
    return manifest


def manifest_hash(manifest: dict[str, Any]) -> str:
    body = {k: v for k, v in manifest.items() if k not in ("manifest_sha256", "created_utc")}
    return sha256_bytes(canonical_json(body).encode("utf-8"))


def _check_manifest(m: dict[str, Any]) -> None:
    fit = np.asarray(m["fit_uids"], dtype=np.int64)
    sel = np.asarray(m["selection_uids"], dtype=np.int64)
    if len(np.intersect1d(fit, sel)) != 0:
        raise ValueError("fit/selection UID sets intersect")
    if len(np.unique(fit)) != len(fit) or len(np.unique(sel)) != len(sel):
        raise ValueError("duplicate UIDs inside a split")
    if not (np.all(np.diff(fit) > 0) and np.all(np.diff(sel) > 0)):
        raise ValueError("splits must be sorted by UID")
    if len(fit) + len(sel) != m["n_total"]:
        raise ValueError("fit + selection must cover the training partition exactly")
    if fit.min() < 0 or sel.min() < 0 or max(fit.max(), sel.max()) >= m["n_total"]:
        raise ValueError("UID out of range")
    if m["official_test_included"]:
        raise ValueError("official test must not be part of the development manifest")


def write_manifest(manifest: dict[str, Any], path: str | Path) -> str:
    path = Path(path)
    atomic_write_json(path, manifest, indent=None)
    (path.with_suffix(".sha256")).write_text(f"{manifest['manifest_sha256']}  {path.name}\n", encoding="utf-8")
    return manifest["manifest_sha256"]


def load_manifest(path: str | Path, *, expected_seed: int | None = None, expected_val_per_class: int | None = None) -> dict[str, Any]:
    m = read_json(path)
    _check_manifest(m)
    h = manifest_hash(m)
    if h != m.get("manifest_sha256"):
        raise ValueError(f"manifest hash mismatch for {path}: stored {m.get('manifest_sha256')}, recomputed {h}")
    if expected_seed is not None and m["split_seed"] != expected_seed:
        raise ValueError("manifest split_seed differs from config")
    if expected_val_per_class is not None and m["val_per_class"] != expected_val_per_class:
        raise ValueError("manifest val_per_class differs from config")
    if m["split"] != SPLIT_NAME:
        raise ValueError(f"unexpected split name {m['split']!r}")
    return m
