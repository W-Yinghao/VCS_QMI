"""P124 (v6 V6-GRANULARITY) — CIFAR-100 coarse / fine / conditional readouts of frozen CIFAR-100 SSL encoders; no encoder training.

Same images (P91 dev45k / val5k manifest: FIT = 45k training images for the probes, evaluation = the 5k selection images; the official test file is
never opened), same frozen h per encoder, three tasks:
  coarse       20-way, coarse labels from the local CIFAR-100 python `train` file ('coarse_labels'; fine labels checked against torchvision's order)
  fine         100-way
  conditional  given the coarse label at test time, 5-way among that coarse class's fine classes; one dedicated probe per coarse group trained on the
               FIT images of that group; reported macro (mean over the 20 groups) and overall (pooled correct / total).  CONDITIONAL — never an
               unconditional 100-way accuracy.
Readouts, identical for every method (no method argument anywhere): `recipe_raw` (PRIMARY — the original frozen-h linear readout: the training
evaluation's recipe probe on raw h, fixed hyper-parameters), `raw_unstd` / `raw_std` / `l2_std` (P119 maps; (lr, wd) chosen on an inner 80 / 20 split of
the FIT labels only), `knn` (recipe k 200, T 0.1, cosine on h; for the conditional task the bank is the FIT images of the same coarse group).
Secondary conditional readout: `masked_fine_head` — the 100-way recipe head restricted to the 5 fine logits of the given coarse group.
Derived from the 100-way recipe head: implied-coarse accuracy (coarse class of the argmax fine class) and fine accuracy given implied coarse correct;
fine accuracy = product of the two (an identity, tested).  Labels are used only by these downstream readouts.
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from vcs_ssl.data.cifar import load_cifar100_train
from vcs_ssl.diagnostics import knn_eval

from .common import CIFAR100_MANIFEST, CIFAR100_ROOT, OUTPUT_ROOT, Encoder, accuracy, atomic_torch_save, fit_probe_selected, sha_idx, train_linear
from .probe5 import KNN, prepare

P124_CACHE = OUTPUT_ROOT / "P124_features"
TRAIN_PICKLE = Path(CIFAR100_ROOT) / "cifar-100-python" / "train"
META_PICKLE = Path(CIFAR100_ROOT) / "cifar-100-python" / "meta"
SELECTED = ("raw_unstd", "raw_std", "l2_std")
READOUTS = ("recipe_raw",) + SELECTED + ("knn",)
SEL_SEED = {"coarse": 124020, "fine": 124100}  # inner-split seeds, identical for every method; group g uses 124200 + g


def sel_seed(task: str, group: int | None = None) -> int:
    return SEL_SEED[task] if group is None else 124200 + int(group)


def load_labels(fine_targets: np.ndarray | None = None) -> dict:
    """Coarse and fine labels of the 50k official TRAINING images from the local python files (the `test` file is never opened); the fine→coarse map
    (a function: every fine class in exactly one coarse class, 5 fine per coarse); label names from `meta`.  If `fine_targets` (torchvision order) is
    given, the pickle's fine labels must equal it position by position."""
    with open(TRAIN_PICKLE, "rb") as f:
        t = pickle.load(f, encoding="latin1")
    with open(META_PICKLE, "rb") as f:
        meta = pickle.load(f, encoding="latin1")
    fine = np.asarray(t["fine_labels"], dtype=np.int64); coarse = np.asarray(t["coarse_labels"], dtype=np.int64)
    assert fine.shape == coarse.shape == (50000,), (fine.shape, coarse.shape)
    if fine_targets is not None:
        assert np.array_equal(fine, np.asarray(fine_targets)), "pickle fine labels differ from the torchvision order"
    f2c = np.full(100, -1, dtype=np.int64)
    for fc, cc in zip(fine, coarse):
        if f2c[fc] == -1:
            f2c[fc] = cc
        elif f2c[fc] != cc:
            raise AssertionError(f"fine class {fc} maps to two coarse classes")
    assert (f2c >= 0).all() and len(meta["fine_label_names"]) == 100 and len(meta["coarse_label_names"]) == 20
    groups = {g: sorted(np.where(f2c == g)[0].tolist()) for g in range(20)}
    assert all(len(v) == 5 for v in groups.values()), {g: len(v) for g, v in groups.items()}
    return {"fine": fine, "coarse": coarse, "f2c": f2c, "groups": groups, "fine_names": list(meta["fine_label_names"]),
            "coarse_names": list(meta["coarse_label_names"])}


def conditional_subset(fine: np.ndarray, coarse: np.ndarray, groups: dict, g: int) -> tuple[np.ndarray, np.ndarray]:
    """Positions with coarse label g and their fine labels remapped to 0..4 (order of groups[g])."""
    pos = np.where(coarse == g)[0]
    remap = {c: i for i, c in enumerate(groups[g])}
    lab = np.array([remap[int(c)] for c in fine[pos]], dtype=np.int64)
    return pos, lab


@torch.no_grad()
def masked_group_predictions(logits: torch.Tensor, coarse_given: torch.Tensor, groups: dict) -> torch.Tensor:
    """Fine prediction restricted to the 5 fine classes of the GIVEN coarse class (logits of all other classes set to −inf)."""
    mask = torch.full_like(logits, float("-inf"))
    for g, fs in groups.items():
        rows = (coarse_given == g).nonzero(as_tuple=True)[0]
        if len(rows):
            mask[rows[:, None], torch.as_tensor(fs)[None, :]] = 0.0
    return (logits + mask).argmax(1)


def fine_head_decomposition(pred_fine: torch.Tensor, y_fine: torch.Tensor, f2c: np.ndarray) -> dict:
    """From a 100-way head: implied-coarse accuracy (coarse of the predicted fine class == true coarse) and fine accuracy given implied coarse
    correct; fine accuracy = implied_coarse_acc × fine_given_coarse_correct (exact identity)."""
    f2c_t = torch.as_tensor(f2c)
    cc = f2c_t[pred_fine] == f2c_t[y_fine]; fine_ok = pred_fine == y_fine
    n = len(y_fine); n_cc = int(cc.sum())
    return {"implied_coarse_acc_pct": 100.0 * n_cc / n, "fine_given_coarse_correct_pct": 100.0 * int((fine_ok & cc).sum()) / max(n_cc, 1),
            "fine_acc_pct": 100.0 * int(fine_ok.sum()) / n}


def cifar100_split(enc: Encoder | None = None):
    data = load_cifar100_train(CIFAR100_ROOT); m = json.load(open(CIFAR100_MANIFEST))
    fit, sel = np.asarray(m["fit_uids"]), np.asarray(m["selection_uids"])
    if enc is not None:  # the encoder was trained on this manifest's FIT images; its selection split must be the evaluation split here
        assert np.array_equal(np.asarray(enc.manifest["fit_uids"]), fit) and np.array_equal(np.asarray(enc.manifest["selection_uids"]), sel), \
            f"{enc.run}: manifest differs from {CIFAR100_MANIFEST}"
    return data, fit, sel


def features(enc: Encoder, smoke: bool = False):
    """float32 h of the FIT and selection images, cached under outputs/P124_features/<run>/ (smoke caches are separate files)."""
    data, fit, sel = cifar100_split(enc)
    if smoke:
        fit, sel = fit[::30], sel[::10]
    tag = "smoke_" if smoke else ""; out = []
    for part, uids in (("fit", fit), ("sel", sel)):
        p = P124_CACHE / enc.run / f"{tag}c100_{part}_h_fp32.pt"
        if p.is_file():
            out.append(torch.load(p, map_location="cpu"))
        else:
            t0 = time.time(); h = enc.h(data.data[uids]).float(); atomic_torch_save(h, p); out.append(h)
            print(f"[{enc.run}] features {part} {tuple(h.shape)} fp32 in {time.time() - t0:.0f}s", flush=True)
    return out[0], out[1], data, fit, sel


def _readouts(hf: torch.Tensor, hs: torch.Tensor, yf: torch.Tensor, ys: torch.Tensor, nc: int, seed: int, dev, readouts=READOUTS) -> dict:
    res = {}
    for kind in readouts:
        t0 = time.time()
        if kind == "recipe_raw":
            head = train_linear(hf, yf, nc, {}, dev); res[kind] = {"acc_pct": accuracy(head, hs, ys, dev), "chosen": "recipe_fixed", "head": head}
        elif kind == "knn":
            res[kind] = {"acc_pct": knn_eval(hf, yf, hs, ys, k=min(KNN["k"], len(hf)), temperature=KNN["temperature"], n_classes=nc, device=dev)["knn_val_top1_pct"]}
        else:
            xt, xe = prepare(kind, hf, hs); s = fit_probe_selected(xt, yf, nc, dev, sel_seed=seed)
            res[kind] = {"acc_pct": accuracy(s["head"], xe, ys, dev), "chosen": s["chosen"], "inner_scores": s["inner_scores"]}
        res[kind]["seconds"] = time.time() - t0
    return res


def run_granularity(enc: Encoder, dev, smoke: bool = False, readouts=READOUTS) -> dict:
    hf, hs, data, fit, sel = features(enc, smoke=smoke)
    L = load_labels(data.targets)
    cf, cs = torch.as_tensor(L["coarse"][fit]), torch.as_tensor(L["coarse"][sel])
    ff, fs = torch.as_tensor(L["fine"][fit]), torch.as_tensor(L["fine"][sel])
    out = {"run": enc.run, "n_fit": int(len(fit)), "n_sel": int(len(sel)), "fit_sha": sha_idx(fit), "sel_sha": sha_idx(sel), "readouts": list(readouts),
           "knn": KNN, "tasks": {}}
    # coarse and fine
    for task, yf, ys, nc in (("coarse", cf, cs, 20), ("fine", ff, fs, 100)):
        r = _readouts(hf, hs, yf, ys, nc, sel_seed(task), dev, readouts)
        if task == "fine" and "recipe_raw" in r:
            head = r["recipe_raw"]["head"]
            with torch.no_grad():
                logits = head(hs.to(dev).float()).cpu()
            out["fine_head_decomposition"] = fine_head_decomposition(logits.argmax(1), fs, L["f2c"])
            pm = masked_group_predictions(logits, cs, L["groups"])
            ok = (pm == fs).numpy(); per_g = {g: float(ok[(cs == g).numpy()].mean() * 100) for g in range(20) if (cs == g).any()}
            out["conditional_masked_fine_head"] = {"macro_pct": float(np.mean(list(per_g.values()))), "overall_pct": float(ok.mean() * 100), "per_group_pct": per_g}
        out["tasks"][task] = {k: {kk: vv for kk, vv in v.items() if kk != "head"} for k, v in r.items()}
        print(f"[{enc.run}] {task}: " + " ".join(f"{k} {v['acc_pct']:.2f}" for k, v in r.items()), flush=True)
    # conditional 5-way, one dedicated probe per coarse group (coarse label given at test time)
    per = {k: {} for k in readouts}; corr = {k: 0.0 for k in readouts}; tot = 0
    for g in range(20):
        pf, lf = conditional_subset(L["fine"][fit], L["coarse"][fit], L["groups"], g)
        ps, ls = conditional_subset(L["fine"][sel], L["coarse"][sel], L["groups"], g)
        if len(ps) == 0 or len(np.unique(lf)) < 5:
            continue
        r = _readouts(hf[pf], hs[ps], torch.as_tensor(lf), torch.as_tensor(ls), 5, sel_seed("conditional", g), dev, readouts)
        for k in readouts:
            per[k][g] = r[k]["acc_pct"]; corr[k] += r[k]["acc_pct"] * len(ps) / 100.0
        tot += len(ps)
    out["tasks"]["conditional"] = {k: {"macro_pct": float(np.mean(list(per[k].values()))), "overall_pct": 100.0 * corr[k] / max(tot, 1),
                                       "per_group_pct": per[k]} for k in readouts}
    print(f"[{enc.run}] conditional (macro): " + " ".join(f"{k} {v['macro_pct']:.2f}" for k, v in out["tasks"]["conditional"].items()), flush=True)
    return out
