"""P85 — run one or more benchmark cells from a units file (one cell per line: 'setting d_signal d_total I N B updates seed [methods]').

    python scripts/estim_p85.py --units slurm/p85_units/pilot.txt --out /home/infres/yinwang/CS_QMI/outputs/P85_estim_benchmark
    python scripts/estim_p85.py --units ... --out ... --smoke --cpu          # CPU gate: tiny roles, one lr, one feature size

Writes <out>/<cell name>.json atomically and skips cells whose JSON exists (resumable); --methods overrides the units' method set.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vcs_estim.benchmark import run_cell  # noqa: E402
from vcs_estim.p85_grid import cell_name, parse_unit_line  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--units", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--cpu", action="store_true"); ap.add_argument("--methods", default=None)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda", 0) if torch.cuda.is_available() and not a.cpu else torch.device("cpu")
    cells = [c for c in (parse_unit_line(l) for l in Path(a.units).read_text().splitlines()) if c]
    print(f"[p85] {len(cells)} cells from {a.units} on {device}", flush=True); rc = 0
    for c in cells:
        f = out / (cell_name(c) + ".json")
        if f.exists():
            print("skip (exists)", f.name, flush=True); continue
        t0 = time.time()
        try:
            R = run_cell(c, device, smoke=a.smoke, methods=a.methods or c["methods"])
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc(); print(f"FAILED {f.name}: {type(e).__name__}: {e}", flush=True); rc = 1; continue
        tmp = f.with_suffix(".tmp"); tmp.write_text(json.dumps(R)); tmp.replace(f)
        sel = [s for s in R["summary"]]
        print(f"{f.name} [{time.time() - t0:.0f} s]: " + " | ".join(f"{s['kind']}{'/' + s['method'].split(':')[2] if s['family'] == 'neural' else ''} err {s['native_abs_error'] if s['native_abs_error'] is None else round(s['native_abs_error'], 4)}" for s in sel), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
