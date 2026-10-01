"""P110 runner (package v4 V1 / V2 / V3) — one process runs the requested modules for a list of encoders sequentially (bundle pattern);
an existing result file is skipped, so a re-submitted job resumes.  Only COMPLETED runs with epoch_800.pt are read.

    python scripts/p110_v.py --runs P35_vcs_a5_views4_800ep_seed0,P41_simclr_views4_800ep_seed0 --modules v1,v2,v3 --out reports/P110
    python scripts/p110_v.py --runs ... --modules c10c --authorised-official-test-c10c --out reports/P110_c10c   # owner authorisation only
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
from vcs_vtask.common import DEFAULT_RUNS, Encoder, atomic_json, completed, device  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=",".join(DEFAULT_RUNS)); ap.add_argument("--modules", default="v1,v2,v3")
    ap.add_argument("--out", required=True); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--v3-replicates", type=int, default=5); ap.add_argument("--authorised-official-test-c10c", action="store_true")
    a = ap.parse_args()
    from vcs_vtask import v1, v2, v3  # noqa: PLC0415
    dev = device(); out = Path(a.out); mods = [m for m in a.modules.split(",") if m]
    if "c10c" in mods and not a.authorised_official_test_c10c:
        sys.exit("c10c needs --authorised-official-test-c10c (official CIFAR-10 test images; owner authorisation)")
    rc = 0
    for run in [r for r in a.runs.split(",") if r]:
        if not completed(run):
            print(f"[skip] {run}: not COMPLETED / no epoch_800.pt", flush=True); continue
        todo = [m for m in mods if not (out / f"{m}_{run}.json").is_file()]
        if not todo:
            print(f"[skip] {run}: all of {mods} done", flush=True); continue
        enc = Encoder(run, dev)
        for m in todo:
            t0 = time.time()
            try:
                if m == "v1":
                    res = v1.run_v1(enc, dev, smoke=a.smoke)
                elif m == "v2":
                    res = v2.run_v2(enc, dev, smoke=a.smoke)
                elif m == "v3":
                    res = v3.run_v3(enc, dev, replicates=a.v3_replicates, smoke=a.smoke)
                elif m == "c10c":
                    res = v2.run_cifar10c(enc, dev, authorised=a.authorised_official_test_c10c)
                else:
                    raise ValueError(m)
            except Exception as e:  # keep the bundle going; report the failure
                print(f"[FAIL] {run} {m}: {type(e).__name__}: {e}", flush=True); rc = 1; continue
            res.update(module=m, utc=utc_now(), device=str(dev), seconds=time.time() - t0, smoke=a.smoke)
            atomic_json(res, out / f"{m}_{run}.json"); print(f"[done] {run} {m} in {res['seconds']:.0f}s -> {out / f'{m}_{run}.json'}", flush=True)
        del enc
    return rc


if __name__ == "__main__":
    sys.exit(main())
