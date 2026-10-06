"""P141 runner (v7 V7-C100-LAYER-READOUT): four-site (layer3 / h / r / z) coarse / fine / conditional readouts for a list of frozen CIFAR-100
encoders, sequentially in one process (bundle); an existing result file is skipped (resumable).  Only COMPLETED runs with epoch_800.pt are read
(others are listed as skipped and picked up by a later resubmission); no encoder training; official test set never opened.

    RUNS=<comma list> python scripts/p141_layers.py --out reports/P141 [--readouts recipe_raw,raw_std,knn | all] [--smoke] [--cpu]
RUNS may also be given as --runs; default = DEFAULT_RUNS (all candidate encoders).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from vcs_ssl.utils import utc_now  # noqa: E402
from vcs_vtask.common import Encoder, atomic_json, completed, device  # noqa: E402

FAMILIES = {  # method -> run names (seeds as available); P129 / P130 runs are read once COMPLETED
    "A-P3": [f"P107_AP3_c100_views4_800ep_seed{s}" for s in range(3)] + [f"P120A1_AP3_c100_views4_800ep_seed{s}" for s in (3, 4)],
    "JS-AP3": [f"P120_JS_AP3_c100_views4_800ep_seed{s}" for s in range(5)],
    "SimCLR": [f"P91_c100_simclr_views4_800ep_seed{s}" for s in range(3)],
    "recipe VCS": [f"P91_c100_vcs_a5_views4_800ep_seed{s}" for s in range(3)],
    "G2": [f"P111_G2_c100_views4_800ep_seed{s}" for s in range(3)],
    "JS (3,0.5) tuned": [f"P129_JS_c100_a3_k0.5_seed{s}" for s in range(3)],
    "R50 A-P3": ["P130_AP3_c100_r50_views4_800ep_seed0"],
    "R50 JS-AP3": ["P130_JS_AP3_c100_r50_views4_800ep_seed0"],
    "R50 SimCLR": ["P130_simclr_c100_r50_views4_800ep_seed0"],
}
DEFAULT_RUNS = [r for rs in FAMILIES.values() for r in rs]
ALL_READOUTS = ("recipe_raw", "raw_unstd", "raw_std", "l2_std", "knn")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.environ.get("RUNS", ",".join(DEFAULT_RUNS))); ap.add_argument("--out", required=True)
    ap.add_argument("--readouts", default=os.environ.get("READOUTS", "recipe_raw,raw_std,knn"))
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    import torch  # noqa: PLC0415
    from vcs_vtask import layer_readout as LR  # noqa: PLC0415
    readouts = ALL_READOUTS if a.readouts == "all" else tuple(x for x in a.readouts.split(",") if x)
    assert readouts and set(readouts) <= set(ALL_READOUTS), readouts
    dev = torch.device("cpu") if a.cpu else device(); out = Path(a.out); rc = 0; skipped = []
    for run in [r for r in a.runs.split(",") if r]:
        if not completed(run):
            print(f"[skip] {run}: not COMPLETED / no epoch_800.pt (picked up by a later resubmission)", flush=True); skipped.append(run); continue
        dest = out / f"layers_{run}.json"
        if dest.is_file():
            print(f"[skip] {run}: done", flush=True); continue
        t0 = time.time()
        try:
            enc = Encoder(run, dev); res = LR.run_layer_readout(enc, dev, smoke=a.smoke, readouts=readouts)
        except Exception as e:  # keep the bundle going
            print(f"[FAIL] {run}: {type(e).__name__}: {e}", flush=True); rc = 1; continue
        res.update(utc=utc_now(), device=str(dev), seconds=time.time() - t0, smoke=a.smoke,
                   gpu=(torch.cuda.get_device_name(0) if dev.type == "cuda" else None))
        atomic_json(res, dest); print(f"[done] {run} in {res['seconds']:.0f}s", flush=True)
        del enc
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    print(f"[summary] skipped (not yet completed): {len(skipped)} {skipped}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
