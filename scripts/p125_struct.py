"""P125 — v6 V6-STRUCT runner.  One JSON per (condition, N, seed, orig|rot) cell; finished cells are skipped (resumable).

    python scripts/p125_struct.py --cells "C1_gauss_mid:4096:0:orig;C2_gauss_pad:4096:0:rot" --out-dir outputs/P125_struct [--cpu] [--smoke]
    python scripts/p125_struct.py --stage pilot|confirm --out-dir ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vcs_estim import p125  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--cells", default=None); ap.add_argument("--stage", default=None, choices=(None, "pilot", "confirm"))
    ap.add_argument("--out-dir", required=True); ap.add_argument("--cpu", action="store_true"); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    device = torch.device("cpu") if a.cpu or not torch.cuda.is_available() else torch.device("cuda", 0)
    if a.cells:
        cells = []
        for tok in a.cells.split(";"):
            c, N, s, r = tok.split(":"); cells.append((c, int(N), int(s), r == "rot"))
    else:
        seeds = p125.PILOT_SEEDS if a.stage == "pilot" else p125.CONFIRM_SEEDS
        cells = p125.struct_cells(seeds=seeds)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    for c, N, s, rot in cells:
        f = out / f"P125_{c}_{N}_{s}_{'rot' if rot else 'orig'}.json"
        if f.is_file():
            print(f"[skip] {f.name}", flush=True); continue
        kw = dict(budget=100, lrs=(5e-4,)) if a.smoke else {}
        res = p125.struct_cell(c, N, s, rot, device, smoke=a.smoke, **kw)
        tmp = f.with_suffix(".tmp"); tmp.write_text(json.dumps(res)); tmp.replace(f)
        sel = [r for r in res["rows"] if r["selected"]]
        print(f"[{c}:{N}:{s}:{'rot' if rot else 'orig'}] S={res['truth']['S']:.4f} " + " | ".join(
            f"{r['candidate']}/{r['kind']} post {r['posterior_mse']:.4f} Jerr {r['J_err']:+.4f}" for r in sel) + f" ({res['wall_seconds']:.0f}s; total {time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
