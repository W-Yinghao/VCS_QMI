"""P109 X1 — training score vs an independent measurement-critic family vs downstream quality, on frozen encoders (package v4 §X1).

    python scripts/p109_x1.py --runs P35_vcs_a5_views4_800ep_seed0,P104_G2_views4_800ep_seed0 --out outputs/P109_X1 [--smoke]

Data: the 5 000 SELECTION images of the frozen split (never used by SSL fitting), split by base image into FIT / TUNE / EVAL (3 000 / 1 000 / 1 000,
fixed seed, identical for every run).  Pairs: two train-distribution augmentations of the same base image (P) and cyclic-shift product pairs
inside the same split and augmentation draw (Q, K = 8).  FIT uses R_fit augmentation draws, TUNE / EVAL one draw each (fixed seeds).
Representations: z (L2 projector output = the VCS critic input) and L2(h) (the evaluated representation).
Per run and representation: the measurement family of `vcs_measure.xmeasure` (zero / calibrated cosine / pair MLP / product ridge / RFF ridge),
selection on TUNE, EVAL J unclipped.  Next to it: the run's own training critic scored on the same EVAL pairs (VCS runs; SimCLR has none), the
training evaluation's held-out J, final frozen linear / kNN.  Labels are not used anywhere in X1.  Nothing is trained except the measurement critics.
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vcs_measure.common import layer_features, load_run, run_eval_summary, split_base_ids  # noqa: E402
from vcs_measure.xmeasure import PairSet, j_value, measure  # noqa: E402
from vcs_ssl.data.cifar import load_cifar10_train  # noqa: E402
from vcs_ssl.data.datasets import TwoViewNoLabelEvalDataset, make_eval_loader  # noqa: E402
from vcs_ssl.data.transforms import build_two_view_transform  # noqa: E402
from vcs_ssl.utils import atomic_write_json, utc_now  # noqa: E402

SPLIT_SEED = 20261001


@torch.no_grad()
def two_view_features(R: dict, data, uids: np.ndarray, seed: int, device, workers: int) -> dict[str, torch.Tensor]:
    tf = build_two_view_transform(R["cfg"]["views"]); g = torch.Generator().manual_seed(seed)
    loader = make_eval_loader(TwoViewNoLabelEvalDataset(data.data, uids, tf), batch_size=512, num_workers=workers, generator=g, pin_memory=device.type == "cuda")
    out = {k: [] for k in ("z1", "z2", "h1", "h2")}
    for x1, x2, _ in loader:
        f1 = layer_features(R["encoder"], R["projector"], x1.to(device), R["eps"]); f2 = layer_features(R["encoder"], R["projector"], x2.to(device), R["eps"])
        out["z1"].append(f1["z"].cpu()); out["z2"].append(f2["z"].cpu())
        out["h1"].append(F.normalize(f1["h"], dim=1).cpu()); out["h2"].append(F.normalize(f2["h"], dim=1).cpu())
    return {k: torch.cat(v) for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--fit", type=int, default=3000); ap.add_argument("--tune", type=int, default=1000); ap.add_argument("--eval", type=int, default=1000)
    ap.add_argument("--r-fit", type=int, default=2); ap.add_argument("--k", type=int, default=8); ap.add_argument("--mlp-steps", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.fit, a.tune, a.eval, a.r_fit, a.mlp_steps = 300, 200, 200, 1, 100
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    data = None
    for run in a.runs.split(","):
        t0 = time.time()
        R = load_run(run, device=device)
        if data is None:
            data = load_cifar10_train(R["cfg"]["data"]["root"])        # official TRAIN file only
        sel = np.asarray(R["manifest"]["selection_uids"], dtype=np.int64)
        sp = split_base_ids(sel, {"fit": a.fit, "tune": a.tune, "eval": a.eval}, SPLIT_SEED)
        feats = {"fit": [two_view_features(R, data, sp["fit"], 1000 + r, device, a.workers) for r in range(a.r_fit)],
                 "tune": [two_view_features(R, data, sp["tune"], 2000, device, a.workers)], "eval": [two_view_features(R, data, sp["eval"], 3000, device, a.workers)]}
        rec = {"run": run, "utc": utc_now(), "device": str(device), "sizes": {k: int(len(v)) for k, v in sp.items()}, "r_fit": a.r_fit, "k": a.k,
               "split_seed": SPLIT_SEED, "summary": run_eval_summary(run), "representations": {}}
        for rep in ("z", "h"):
            ps = {}
            for part, seed in (("fit", 11), ("tune", 12), ("eval", 13)):
                u1 = torch.cat([f[f"{rep}1"] for f in feats[part]]); u2 = torch.cat([f[f"{rep}2"] for f in feats[part]])
                ps[part] = PairSet(u1, u2, a.k, seed, block=len(sp[part]))
            m = measure(ps["fit"], ps["tune"], ps["eval"], device=device, mlp_steps=a.mlp_steps)
            if rep == "z" and R["critic"] is not None:
                crit = R["critic"].cpu()
                with torch.no_grad():
                    m["train_critic_eval_J"] = j_value(crit(*ps["eval"].p()), crit(*ps["eval"].q()))
                    m["train_critic_tune_J"] = j_value(crit(*ps["tune"].p()), crit(*ps["tune"].q()))
                m["train_critic"] = {"class": type(crit).__name__, "a": float(getattr(crit, "scale", float("nan"))), "b": float(getattr(crit, "bias", float("nan")))}
            rec["representations"][rep] = m
            print(f"[{run}] {rep}: picked {m['picked']} EVAL J {m['picked_eval_J']:.4f} | " + " ".join(f"{k}={v['eval_J']:.4f}" for k, v in m["candidates"].items())
                  + (f" | train critic EVAL J {m['train_critic_eval_J']:.4f}" if "train_critic_eval_J" in m else ""), flush=True)
        rec["seconds"] = time.time() - t0
        atomic_write_json(out_dir / f"{run}.json", rec)
        del R
    return 0


if __name__ == "__main__":
    sys.exit(main())
