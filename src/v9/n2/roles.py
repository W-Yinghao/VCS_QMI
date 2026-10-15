"""V9-N2 roles (package §05.2): the 10 % trusted anchor set A is drawn BEFORE any noise, by image id, with a fixed seed; inside A the
teacher-warmup / critic-fit / critic-select / reference-moments split (40 / 30 / 15 / 15) is again by image; U = every other image.  Captions follow
their image (caption groups are never split across roles).  Integer counts are allocated by largest remainder and written to the manifest; a role
below its minimum size is an error ("insufficient data"), never filled from val / test.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ANCHOR_ROLES = ("teacher_warmup", "critic_fit", "critic_select", "reference_moments")
DEFAULT_SPLIT = (0.4, 0.3, 0.15, 0.15)


def _largest_remainder(n: int, fractions) -> list[int]:
    raw = [n * f for f in fractions]; base = [int(np.floor(x)) for x in raw]
    rem = n - sum(base); order = np.argsort([-(x - b) for x, b in zip(raw, base)], kind="stable")
    for k in order[:rem]:
        base[int(k)] += 1
    return base


def sha256_ids(ids) -> str:
    return hashlib.sha256(("\n".join(str(i) for i in ids)).encode()).hexdigest()


def assign_roles(image_ids, seed: int, anchor_fraction: float = 0.1, split=DEFAULT_SPLIT, min_sizes=None) -> dict:
    """Return {'A': {role: sorted image ids}, 'U': sorted image ids, 'counts': ..., 'hashes': ..., 'seed': seed}.

    image_ids: the unique training image ids (strings or ints).  min_sizes: {role: minimum number of images}; default 50 for every anchor role.
    """
    ids = sorted(set(str(i) for i in image_ids))
    if len(ids) != len(list(image_ids)):
        raise ValueError("image ids must be unique (caption groups are handled by the pair table, not here)")
    if not 0 < anchor_fraction < 1 or abs(sum(split) - 1) > 1e-12 or len(split) != 4:
        raise ValueError("invalid anchor fraction or split")
    rng = np.random.default_rng([int(seed), 9_002])
    perm = rng.permutation(len(ids))
    n_anchor = int(round(anchor_fraction * len(ids)))
    anchor = [ids[i] for i in perm[:n_anchor]]; rest = sorted(ids[i] for i in perm[n_anchor:])
    counts = _largest_remainder(n_anchor, split)
    roles, start = {}, 0
    for role, c in zip(ANCHOR_ROLES, counts):
        roles[role] = sorted(anchor[start:start + c]); start += c
    min_sizes = dict(min_sizes or {});
    for role in ANCHOR_ROLES:
        need = int(min_sizes.get(role, 50))
        if len(roles[role]) < need:
            raise ValueError(f"insufficient data: role {role} has {len(roles[role])} images < minimum {need}; val / test are never used to fill it")
    out = {"seed": int(seed), "anchor_fraction": anchor_fraction, "split": list(split), "n_images": len(ids), "n_anchor": n_anchor,
           "counts": {role: len(roles[role]) for role in ANCHOR_ROLES}, "n_U": len(rest),
           "A": roles, "U": rest, "hashes": {role: sha256_ids(roles[role]) for role in ANCHOR_ROLES} | {"U": sha256_ids(rest)}}
    check_disjoint(out)
    return out


def check_disjoint(roles: dict) -> None:
    seen = {}
    for role in ANCHOR_ROLES:
        for i in roles["A"][role]:
            if i in seen:
                raise ValueError(f"image {i} in two roles ({seen[i]}, {role})")
            seen[i] = role
    for i in roles["U"]:
        if i in seen:
            raise ValueError(f"image {i} in U and in {seen[i]}")
        seen[i] = "U"
    if len(seen) != roles["n_images"]:
        raise ValueError("roles do not cover every image exactly once")


def role_of_image(roles: dict) -> dict:
    m = {i: role for role in ANCHOR_ROLES for i in roles["A"][role]}
    m.update({i: "U" for i in roles["U"]})
    return m


def write_roles(roles: dict, path: str | Path) -> str:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(roles, indent=1))
    return hashlib.sha256(path.read_bytes()).hexdigest()
