"""P83 addendum 1 driver: run `vcs_estim.frozen_supplement` for every P84 checkpoint (skip-if-done), then aggregate.

    python scripts/diag_p84_supplement.py --out-dir <dir> --report <md> [--ckpts run:ckpt,...] [--smoke] [--cpu]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
P84_DIR = Path("/home/infres/yinwang/CS_QMI/outputs/P83_estim_frozen_diag")
OUT_ROOT = Path("/home/infres/yinwang/CS_QMI/outputs")
DEFAULT = ",".join([f"P35_vcs_a5_views4_800ep_seed0:{c}" for c in ("initial", "epoch_100", "epoch_400", "epoch_800")]
                   + [f"P41_simclr_views4_800ep_seed0:{c}" for c in ("initial", "epoch_100", "epoch_400", "epoch_800")] + ["P5_vcs_seed0:initial", "P5_vcs_seed0:epoch_200"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True); ap.add_argument("--report", required=True); ap.add_argument("--ckpts", default=DEFAULT)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--cpu", action="store_true"); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(); out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True); outs = []; rc = 0
    for item in a.ckpts.split(","):
        run, ck = item.split(":"); out = out_dir / f"{run}__{ck}.json"; p84 = P84_DIR / f"{run}__{ck}.json"
        if out.is_file() and out.stat().st_size > 0:
            print(f"skip (exists): {out}"); outs.append(str(out)); continue
        cmd = [PY, "-m", "vcs_estim.frozen_supplement", "--run-dir", str(OUT_ROOT / run), "--ckpt", ck, "--p84", str(p84), "--out", str(out), "--workers", str(a.workers)]
        cmd += (["--smoke"] if a.smoke else []) + (["--cpu"] if a.cpu else [])
        r = subprocess.run(cmd, cwd=str(REPO), env={**__import__("os").environ, "PYTHONPATH": f"{REPO / 'src'}:{REPO}"})
        if r.returncode != 0:
            rc = 1; print(f"FAILED {item}"); continue
        outs.append(str(out))
    if outs:
        subprocess.run([PY, "-m", "vcs_estim.frozen_supplement", "--stage", "aggregate", "--inputs", *outs, "--out", a.report], cwd=str(REPO), env={**__import__("os").environ, "PYTHONPATH": f"{REPO / 'src'}:{REPO}"})
    return rc


if __name__ == "__main__":
    sys.exit(main())
