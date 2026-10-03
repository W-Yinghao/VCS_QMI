"""P119 runner (v5 NEXT-V-PROBE): modules fpcheck | transfer | c10extra for a list of frozen encoders, sequentially in one process (bundle);
an existing result file is skipped (resumable).  Only COMPLETED runs with epoch_800.pt are read.  No encoder training; official test set untouched.

    python scripts/p119_v.py --runs <comma list> --modules fpcheck,transfer,c10extra --out reports/P119 [--fp32] [--smoke]
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

RUNS5 = [f"{f}_seed{s}" for f in ("P35_vcs_a5_views4_800ep", "P41_simclr_views4_800ep", "P104_G2_views4_800ep", "P104_U2_views4_800ep",
                                  "P107_AP3_views4_800ep") for s in range(3)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=",".join(RUNS5)); ap.add_argument("--modules", default="fpcheck,transfer,c10extra")
    ap.add_argument("--out", required=True); ap.add_argument("--fp32", action="store_true", help="float32 features (outputs/P119_features) for transfer / c10extra")
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    import torch  # noqa: PLC0415
    from vcs_vtask import probe5  # noqa: PLC0415
    dev = torch.device("cpu") if a.cpu else device(); out = Path(a.out); mods = [m for m in a.modules.split(",") if m]; rc = 0
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
                if m == "fpcheck":
                    res = probe5.fp_check(enc, dev)
                elif m == "transfer":
                    hf, hs, yf, ys = probe5.features(enc, "cifar100", fp32=a.fp32, smoke=a.smoke)
                    fr = ((0.1, 1), (1.0, 1)) if a.smoke else ((0.1, 3), (1.0, 1))
                    res = {"run": run, "rows": probe5.transfer_readouts(enc, dev, hf, hs, yf, ys, fractions=fr), "features_fp32": a.fp32,
                           "n_fit": int(len(hf)), "n_sel": int(len(hs)), "knn": probe5.KNN}
                elif m == "c10extra":
                    hf, hs, yf, ys = probe5.features(enc, "cifar10", fp32=a.fp32, smoke=a.smoke)
                    res = {"run": run, "rows": probe5.label_efficiency_extra(enc, dev, hf, hs, yf, ys, draws=(3,) if a.smoke else probe5.EXTRA_DRAWS),
                           "features_fp32": a.fp32}
                else:
                    raise ValueError(m)
            except Exception as e:  # keep the bundle going
                print(f"[FAIL] {run} {m}: {type(e).__name__}: {e}", flush=True); rc = 1; continue
            res.update(module=m, utc=utc_now(), device=str(dev), seconds=time.time() - t0, smoke=a.smoke)
            atomic_json(res, out / f"{m}_{run}.json"); print(f"[done] {run} {m} in {res['seconds']:.0f}s", flush=True)
        del enc
    return rc


if __name__ == "__main__":
    sys.exit(main())
