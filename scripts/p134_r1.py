"""P134 (R1 contamination) — run the cells of a units file (one cell per line: 'type eps seed'; eps = 0 lines run the clean P85 cell once).

    python scripts/p134_r1.py --units slurm/p134_units/seed0.txt --out /home/infres/yinwang/CS_QMI/outputs/P134_r1
    python scripts/p134_r1.py --units ... --out ... --smoke --cpu --N 256 --updates 40      # CPU smoke

Writes <out>/<cell name>.json atomically and skips cells whose JSON exists (resumable).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vcs_estim import r1  # noqa: E402


def parse(line: str):
    f = line.split()
    if not f or f[0].startswith("#"):
        return None
    return f[0], float(f[1]), int(f[2])


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--units", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--N", type=int, default=None); ap.add_argument("--updates", type=int, default=None)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda", 0) if torch.cuda.is_available() and not a.cpu else torch.device("cpu")
    cells = [c for c in (parse(l) for l in Path(a.units).read_text().splitlines()) if c]
    print(f"[p134] {len(cells)} cells from {a.units} on {device}", flush=True); rc = 0
    for ctype, eps, seed in cells:
        f = out / (r1.cell_name(ctype, eps, seed) + ".json")
        if f.exists():
            print("skip (exists)", f.name, flush=True); continue
        t0 = time.time()
        try:
            R = r1.run_r1_cell(ctype, eps, seed, device, smoke=a.smoke, N=a.N, updates=a.updates)
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc(); print(f"FAILED {f.name}: {type(e).__name__}: {e}", flush=True); rc = 1; continue
        tmp = f.with_suffix(".tmp"); tmp.write_text(json.dumps(R)); tmp.replace(f)
        tr = R["truth"]
        print(f"{f.name} [{time.time() - t0:.0f} s] truth S {tr['S']:.4f} JS2 {tr['JS2']:.4f} MI {tr['MI']:.3f} | "
              + " | ".join(f"{s['kind']}/{s['method'].split(':')[2]} {s['native_value']:+.4f}" for s in R["summary"]), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
