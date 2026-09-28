"""Estimator package v1 (§4.1, §4.3, §4.4): identity-level role split of the 45k fit images and role-local pair construction.

Roles FIT / TUNE / SELECT / EVAL = 27k / 6k / 6k / 6k images, drawn from `fit_uids` of the frozen dev45k_val5k manifest by a seeded
permutation (stratification is not required by the spec; class counts are recorded).  Every derived pair stays inside one role.
EVAL blocks: (anchor, partner) with distinct identities and no identity reused across blocks.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

ROLE_SIZES = (("FIT", 27000), ("TUNE", 6000), ("SELECT", 6000), ("EVAL", 6000))
ROLE_SEED = 20260928


def role_split(fit_uids, seed: int = ROLE_SEED) -> dict:
    uids = np.asarray(sorted(int(u) for u in fit_uids)); assert len(uids) == sum(n for _, n in ROLE_SIZES)
    perm = np.random.default_rng(seed).permutation(len(uids)); out, s = {}, 0
    for role, n in ROLE_SIZES:
        out[role] = sorted(uids[perm[s: s + n]].tolist()); s += n
    return out


def split_hash(roles: dict) -> str:
    return hashlib.sha256(json.dumps({k: roles[k] for k in sorted(roles)}).encode()).hexdigest()


def eval_blocks(uids, seed: int) -> np.ndarray:
    """Disjoint (anchor, partner) identity blocks inside one role: [n // 2, 2]."""
    u = np.asarray(uids); p = np.random.default_rng(seed).permutation(len(u)); k = len(u) // 2
    return np.stack([u[p[:k]], u[p[k: 2 * k]]], 1)


def check_disjoint(roles: dict) -> None:
    seen = set()
    for r, ids in roles.items():
        s = set(ids); assert len(s) == len(ids), f"duplicate identity inside {r}"
        assert not (s & seen), f"identity shared across roles ({r})"; seen |= s
