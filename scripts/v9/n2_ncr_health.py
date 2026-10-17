"""V9-N2 NCR_A full-data health check (main session 2026-10-17; pre-registered in V9_N2_PROTOCOL_FROZEN_20261015.yaml,
diagnostic_preregistration_20261017.ncr_health): the PATCHED official NCR with the anchor term ON, on the full Flickr30K training pool with the V9
mapping of seed 917 / rho 0.4 (x1), native schedule: 5 warm-up epochs, then --co-epochs co-training epochs; RTX6000PRO only.  Reads the dev
validations the trainer prints (post-warm-up reference, then one per co-training epoch) and applies the pre-registered rule:
    FAIL if the run does not finish, or any co-training validation has rSum < 0.5 x the post-warm-up rSum or t2i R@1 < 2.0 (4x chance on the
    200 dev images of NCR's own validation); PASS otherwise.
NCR_A is enabled only on PASS.  A gate, never evidence; the trainer's own end-of-training test read is ignored.

    python scripts/v9/n2_ncr_health.py --out <dir> [--co-epochs 3]"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "v9"))
from n2_official_smoke import dev_trajectory  # noqa: E402

OUT_ROOT = Path("/projects/EEG-foundation-model/yinghao/outputs_v9/n2")
DATA = Path("/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data")


def verdict(traj: list, finished: bool) -> dict:
    if not finished or len(traj) < 2:
        return {"pass": False, "reason": "run did not finish or printed no co-training validation", "trajectory": traj}
    ref, co = traj[0], traj[1:]
    bad = [i + 1 for i, x in enumerate(co) if x["rsum"] < 0.5 * ref["rsum"] or x["t2i_R1"] < 2.0]
    return {"pass": not bad, "reason": "" if not bad else f"co-training epoch(s) {bad} below the pre-registered floor", "post_warmup": ref, "co_training": co, "trajectory": traj}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--co-epochs", type=int, default=3)
    ap.add_argument("--rho", type=float, default=0.4); ap.add_argument("--seed", type=int, default=917); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); py = sys.executable; drv = str(ROOT / "scripts/v9/n2_official.py"); t0 = time.time()
    shared = OUT_ROOT / "f30k" / f"rho{a.rho:g}" / f"s{a.seed}"
    rc_e = subprocess.run([py, drv, "export", "--shared", str(shared), "--data-root", str(DATA / "f30k_precomp"), "--out", str(out)], cwd=ROOT).returncode
    rc_l = subprocess.run([py, drv, "launch", "--baseline", "ncr", "--unit", str(out), "--dataset", "f30k", "--seed", str(a.seed), "--rho", str(a.rho),
                           "--data-root", str(DATA), "--num-epochs", str(a.co_epochs)], cwd=ROOT).returncode if rc_e == 0 else -1
    tl = (out / "official_train.log").read_text(errors="replace") if (out / "official_train.log").exists() else ""
    traj = dev_trajectory(tl.split("-------- Testing --------")[0])
    res = {"export_rc": rc_e, "launch_rc": rc_l, "anchor_term_lines": sum(1 for ln in tl.splitlines() if "[V9] anchor-supervised term" in ln),
           "observed_index_loaded": "load V9 observed index" in tl, "seconds": round(time.time() - t0, 1), "co_epochs": a.co_epochs, "seed": a.seed, "rho": a.rho,
           **verdict(traj, rc_l == 0)}
    res["pass"] = bool(res["pass"] and res["observed_index_loaded"] and res["anchor_term_lines"] >= a.co_epochs)
    (out / "ncr_health.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
