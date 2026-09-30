"""P101 launcher: the reference checkpoint first (its calibrated tau becomes the common tau), then the other five P84 checkpoints, then the
aggregate table.  Skips checkpoints whose JSON exists (resumable).

    python scripts/o_line_frozen.py --out-dir outputs/P101_o_line --report reports/P101_o_line [--smoke] [--cpu]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

O = "/home/infres/yinwang/CS_QMI/outputs"
REF = ("P35_vcs_a5_views4_800ep_seed0", "epoch_800")
OTHERS = [("P35_vcs_a5_views4_800ep_seed0", "epoch_100"), ("P35_vcs_a5_views4_800ep_seed0", "epoch_400"),
          ("P41_simclr_views4_800ep_seed0", "epoch_100"), ("P41_simclr_views4_800ep_seed0", "epoch_400"), ("P41_simclr_views4_800ep_seed0", "epoch_800")]


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out-dir", required=True); ap.add_argument("--report", required=True)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--cpu", action="store_true"); ap.add_argument("--only-ref", action="store_true")
    a = ap.parse_args(); od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True); extra = (["--smoke"] if a.smoke else []) + (["--cpu"] if a.cpu else [])
    ref_json = od / f"{REF[0]}__{REF[1]}.json"; outs = []
    for i, (run, ck) in enumerate([REF] + ([] if a.only_ref else OTHERS)):
        out = od / f"{run}__{ck}.json"; outs.append(str(out))
        if out.exists() and out.stat().st_size > 0:
            print("skip (exists):", out, flush=True); continue
        cmd = [sys.executable, "-m", "vcs_estim.observation_frozen", "--run-dir", f"{O}/{run}", "--ckpt", ck, "--out", str(out)] + extra
        if i > 0:
            cmd += ["--ref-json", str(ref_json)]
        print(" ".join(cmd), flush=True); rc = subprocess.call(cmd)
        if rc != 0:
            print(f"FAILED {run}:{ck} rc={rc}", flush=True); return rc
    return subprocess.call([sys.executable, "-m", "vcs_estim.observation_frozen", "--stage", "aggregate", "--inputs", *outs, "--out", a.report + ".md"])


if __name__ == "__main__":
    raise SystemExit(main())
