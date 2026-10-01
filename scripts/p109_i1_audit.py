"""P109 I1 step 2 — conditional nuisance audit on the cached paired features (see `vcs_measure.audit`).

    python scripts/p109_i1_audit.py --features outputs/P109_I1_features --run <run> --family colour --cells 'planted:0.1:2000;null_label_only:0:2000'
                                    --repeats 100 --perms 200 --seed 301 --out reports/P109_i1_<run>_<family>_<tag> [--effects] [--smoke]

Cells: mode:strength:n (strength names the planted version; 'null_label_only' ignores it; 'null_all_planted' uses the family's largest strength).
--effects additionally writes the paired prediction effects of every planted version of the family (EVAL base images, no test).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vcs_measure.audit import AUDIT_LAYERS, AuditData, prediction_effects, run_repeat  # noqa: E402
from vcs_ssl.utils import atomic_write_json, utc_now  # noqa: E402

STRENGTHS = {"colour": [0.05, 0.1, 0.2], "blur": [0.25, 0.5, 1.0]}


def vname(family: str, s: float) -> str:
    return f"{family}_s{s:g}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True); ap.add_argument("--run", required=True); ap.add_argument("--family", required=True, choices=("colour", "blur"))
    ap.add_argument("--cells", default=""); ap.add_argument("--repeats", type=int, default=100); ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--out", required=True); ap.add_argument("--effects", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    D = AuditData(Path(a.features) / a.run / "features.pt")
    R, perms = (3, 20) if a.smoke else (a.repeats, a.perms)
    rng = np.random.default_rng(a.seed)
    res = {"run": a.run, "family": a.family, "utc": utc_now(), "device": str(device), "layers": list(AUDIT_LAYERS), "settings": vars(a), "cells": {}}
    t0 = time.time()
    for tok in [t for t in a.cells.split(";") if t]:
        mode, s, n = tok.split(":"); s, n = float(s), int(n)
        if a.smoke:
            n = min(n, 200)
        version = vname(a.family, max(STRENGTHS[a.family]) if mode == "null_all_planted" else (s if mode == "planted" else STRENGTHS[a.family][0]))
        inst = [run_repeat(D, n, rng, mode=mode, version=version, perms=perms, seed=a.seed * 1000 + r, device=device) for r in range(R)]
        summ = {}
        for l in AUDIT_LAYERS:
            for st in ("vcs_closed", "js_exact"):
                summ[f"{l}/{st}"] = {"power": float(np.mean([i["layers"][l][st]["reject"] for i in inst])),
                                     "power_maxT": float(np.mean([i["layers"][l][st]["reject_maxT"] for i in inst])),
                                     "stat_mean": float(np.mean([i["layers"][l][st]["stat"] for i in inst]))}
        for st in ("vcs_closed", "js_exact"):
            summ[f"any_layer_maxT/{st}"] = {"power": float(np.mean([i[f"{st}_any_layer_reject_maxT"] for i in inst]))}
        key = f"{mode}:{s:g}:{n}"
        res["cells"][key] = {"mode": mode, "version": version, "n": n, "repeats": R, "perms": perms, "summary": summ, "instances": inst}
        print(f"[{a.run} {a.family} {key} R={R}] " + " ".join(f"{k}={v['power']:.2f}" for k, v in summ.items() if k.startswith(("h/", "logits/", "any_"))) + f" ({time.time() - t0:.0f}s)", flush=True)
        atomic_write_json(Path(a.out + ".partial.json"), res)
    if a.effects:
        res["prediction_effects"] = {vname(a.family, s): prediction_effects(D, vname(a.family, s), boot=200 if a.smoke else 1000, seed=a.seed) for s in STRENGTHS[a.family]}
        for v, e in res["prediction_effects"].items():
            o = e["overall"]
            print(f"[{a.run}] effect {v}: dP(true) {o['d_prob_true']:+.4f} {o['d_prob_true_ci']} dMargin {o['d_margin']:+.3f} acc {o['acc_clean']:.3f}->{o['acc_planted']:.3f} flips {o['prediction_flip_rate']:.3f}", flush=True)
    res["seconds"] = time.time() - t0
    atomic_write_json(Path(a.out + ".json"), res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
