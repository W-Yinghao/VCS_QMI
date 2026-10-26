"""Runtime held-out guard: a sys audit hook that refuses to open any non-FIT/CAL data file.

Each guarded root has an allow-list; opening any other file under that root raises PermissionError in the
opening process (DataLoader workers are forked after install, so they inherit the hook).
KD:  root = <data>/cifar-100-python, allowed = {train} (the official test file is never opened).
AIG: root = prepared data dir, allowed = {manifest.json, fit.csv, cal.csv} (sealed/, test_inputs.csv,
     all_inputs.csv refused); root = image dir, allowed = the images listed in fit.csv and cal.csv.
"""
from __future__ import annotations

import csv
import os
import sys

_STATE = {"roots": [], "opened_allowed": 0}


def _real(path):
    return os.path.realpath(os.fspath(path))


def _hook(event, args):
    if event != "open" or not args:
        return
    path = args[0]
    if isinstance(path, int) or path is None:
        return
    try:
        path = os.fsdecode(path)
    except TypeError:
        return
    real = os.path.realpath(path)
    for root, allowed in _STATE["roots"]:
        if real == root or real.startswith(root + os.sep):
            if real in allowed:
                _STATE["opened_allowed"] += 1
                return
            raise PermissionError(f"r22 guard: refused to open held-out/unlisted file {real}")


def install(roots):
    """roots: iterable of (root_dir, iterable of allowed file paths)."""
    for root, allowed in roots:
        _STATE["roots"].append((_real(root), {_real(p) for p in allowed}))
    if not _STATE.get("installed"):
        sys.addaudithook(_hook)
        _STATE["installed"] = True


def kd_roots(cfg):
    data = os.path.join(cfg["data"]["root"], "cifar-100-python")
    return [(data, [os.path.join(data, "train")])]


def aig_roots(cfg):
    data = cfg["data_dir"]
    import json
    with open(os.path.join(data, "manifest.json")) as f:
        image_root = json.load(f)["image_root"]
    images = []
    for name in ("fit.csv", "cal.csv"):
        with open(os.path.join(data, name), newline="") as f:
            images += [os.path.join(image_root, row["image"]) for row in csv.DictReader(f)]
    return [(data, [os.path.join(data, n) for n in ("manifest.json", "fit.csv", "cal.csv")]), (image_root, images)]


def status():
    return {"guarded_roots": [r for r, _ in _STATE["roots"]], "allowed_opens_in_this_process": _STATE["opened_allowed"]}
