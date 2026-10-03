"""P124 runner (v6 V6-GRANULARITY): coarse / fine / conditional readouts for a list of frozen CIFAR-100 encoders, sequentially in one process (bundle);
an existing result file is skipped (resumable).  Only COMPLETED runs with epoch_800.pt are read; no encoder training; official test set never opened.

    python scripts/p124_gran.py [--runs <comma list>] --out reports/P124 [--smoke] [--cpu]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from vcs_ssl.utils import utc_now  # noqa: E402
from vcs_vtask.common import Encoder, atomic_json, completed, device  # noqa: E402

FAMILIES = ("P91_c100_vcs_a5_views4_800ep", "P111_G2_c100_views4_800ep", "P107_AP3_c100_views4_800ep", "P91_c100_simclr_views4_800ep")
RUNS12 = [f"{f}_seed{s}" for f in FAMILIES for s in range(3)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=",".join(RUNS12)); ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    import torch  # noqa: PLC0415
    from vcs_vtask import granularity  # noqa: PLC0415
    dev = torch.device("cpu") if a.cpu else device(); out = Path(a.out); rc = 0
    for run in [r for r in a.runs.split(",") if r]:
        if not completed(run):
            print(f"[skip] {run}: not COMPLETED / no epoch_800.pt", flush=True); rc = 1; continue
        dest = out / f"gran_{run}.json"
        if dest.is_file():
            print(f"[skip] {run}: done", flush=True); continue
        t0 = time.time()
        try:
            enc = Encoder(run, dev); res = granularity.run_granularity(enc, dev, smoke=a.smoke)
        except Exception as e:  # keep the bundle going
            print(f"[FAIL] {run}: {type(e).__name__}: {e}", flush=True); rc = 1; continue
        res.update(utc=utc_now(), device=str(dev), seconds=time.time() - t0, smoke=a.smoke,
                   gpu=(torch.cuda.get_device_name(0) if dev.type == "cuda" else None))
        atomic_json(res, dest); print(f"[done] {run} in {res['seconds']:.0f}s", flush=True)
        del enc
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return rc


if __name__ == "__main__":
    sys.exit(main())
