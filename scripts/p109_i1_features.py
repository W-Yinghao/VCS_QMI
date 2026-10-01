"""P109 I1 step 1 — paired clean / planted frozen features per base image, at fixed layers, plus a clean-trained linear classifier (package v4 §I1).

    python scripts/p109_i1_features.py --runs P35_vcs_a5_views4_800ep_seed0,... --out outputs/P109_I1_features [--smoke]

Base images: 20 000 of the 45 000 FIT images of the frozen split, split by base image ID into FIT 10 000 / VAL 2 000 / EVAL 8 000 (fixed seed,
identical for every run); every version of a base image stays in its split.  Versions per base image: clean; colour shift s ∈ {0.05, 0.1, 0.2}
(R × (1 + s), B × (1 − s) on uint8 — the P45 constructor); Gaussian blur radius s ∈ {0.25, 0.5, 1.0} px (PIL, the P45 constructor).
Layers: layer3 (global-average-pooled, 256), h (512), z (L2 projector output, 128), logits (10) of a multinomial logistic classifier on standardised
clean h, fitted on the FIT base images only (labels used here: downstream / analysis label use, disclosed; never in SSL pretraining).
Stored fp16 (features) / fp32 (logits).  Read-only on the run directory.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
from precheck_d_features import PlantedDataset  # noqa: E402
from vcs_measure.common import layer_features, load_run, split_base_ids  # noqa: E402
from vcs_ssl.data.cifar import load_cifar10_train  # noqa: E402
from vcs_ssl.data.transforms import build_clean_transform  # noqa: E402
from vcs_ssl.utils import atomic_write_json, utc_now  # noqa: E402

SPLIT_SEED = 20261002
VERSIONS = [("clean", "colour", 0.0), ("colour_s0.05", "colour", 0.05), ("colour_s0.1", "colour", 0.1), ("colour_s0.2", "colour", 0.2),
            ("blur_s0.25", "blur", 0.25), ("blur_s0.5", "blur", 0.5), ("blur_s1", "blur", 1.0)]


@torch.no_grad()
def extract(R, ds, device, workers) -> dict[str, torch.Tensor]:
    out = {k: [] for k in ("layer3", "h", "z")}
    for x, _, _, _ in DataLoader(ds, batch_size=512, shuffle=False, num_workers=workers):
        f = layer_features(R["encoder"], R["projector"], x.to(device), R["eps"])
        for k in out:
            out[k].append(f[k].float().cpu())
    return {k: torch.cat(v) for k, v in out.items()}


def fit_classifier(h: torch.Tensor, y: torch.Tensor, wd: float = 1e-4, iters: int = 300) -> dict:
    """Multinomial logistic regression on standardised h (full-batch L-BFGS, L2 wd); deterministic."""
    mu, sd = h.mean(0), h.std(0) + 1e-6; X = ((h - mu) / sd).double(); Y = y.long()
    W = torch.zeros(X.shape[1], 10, dtype=torch.float64, requires_grad=True); b = torch.zeros(10, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(); loss = F.cross_entropy(X @ W + b, Y) + 0.5 * wd * (W ** 2).sum(); loss.backward(); return loss
    opt.step(closure)
    return {"mu": mu, "sd": sd, "W": W.detach().float(), "b": b.detach().float()}


def logits(clf, h):
    return ((h - clf["mu"]) / clf["sd"]) @ clf["W"] + clf["b"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--fit", type=int, default=10000); ap.add_argument("--val", type=int, default=2000); ap.add_argument("--eval", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.fit, a.val, a.eval = 600, 300, 600
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    data = None
    for run in a.runs.split(","):
        t0 = time.time(); R = load_run(run, device=device)
        if data is None:
            data = load_cifar10_train(R["cfg"]["data"]["root"])
        fit_pool = np.asarray(R["manifest"]["fit_uids"], dtype=np.int64)
        sp = split_base_ids(fit_pool, {"fit": a.fit, "val": a.val, "eval": a.eval}, SPLIT_SEED)
        ids = np.concatenate([sp["fit"], sp["val"], sp["eval"]]); part = np.repeat([0, 1, 2], [len(sp["fit"]), len(sp["val"]), len(sp["eval"])])
        y = torch.as_tensor(np.asarray(data.targets)[ids]); tf = build_clean_transform(R["cfg"]["views"])
        feats = {}
        for name, kind, s in VERSIONS:
            ds = PlantedDataset(data.data, data.targets, ids, np.ones(len(ids), dtype=np.int64), s, tf, kind=kind)
            feats[name] = extract(R, ds, device, a.workers)
        clf = fit_classifier(feats["clean"]["h"][part == 0], y[part == 0])
        acc = {}
        for name in feats:
            lg = logits(clf, feats[name]["h"]); feats[name]["logits"] = lg
            acc[name] = {k: float((lg[part == i].argmax(1) == y[part == i]).float().mean()) for i, k in enumerate(("fit", "val", "eval"))}
        store = {"run": run, "ids": torch.as_tensor(ids), "part": torch.as_tensor(part), "y": y, "versions": [v[0] for v in VERSIONS],
                 "features": {n: {k: (v.half() if k != "logits" else v.float()) for k, v in f.items()} for n, f in feats.items()},
                 "classifier": {k: v for k, v in clf.items()}, "split_seed": SPLIT_SEED}
        od = Path(a.out) / run; od.mkdir(parents=True, exist_ok=True); torch.save(store, od / "features.pt")
        atomic_write_json(od / "manifest.json", {"run": run, "utc": utc_now(), "epoch": R["epoch"], "sizes": {k: int(len(v)) for k, v in sp.items()},
                                                  "split_seed": SPLIT_SEED, "versions": [list(v) for v in VERSIONS], "classifier_acc": acc,
                                                  "label_use": "classifier fitted on FIT base images (clean h) with labels; analysis only",
                                                  "seconds": time.time() - t0})
        print(f"[{run}] features done in {time.time() - t0:.0f}s; classifier acc clean val {acc['clean']['val']:.3f} eval {acc['clean']['eval']:.3f}; "
              + " ".join(f"{n}:{acc[n]['eval']:.3f}" for n in feats), flush=True)
        del R
    return 0


if __name__ == "__main__":
    sys.exit(main())
