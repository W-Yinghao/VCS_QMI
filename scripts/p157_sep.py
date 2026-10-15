"""P157 (SEP-2) runner.  run --seeds 200+201 --out DIR | pop --out DIR | units.  Resumable (finished cells are skipped)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch  # noqa: E402

from vcs_estim import p157, sep  # noqa: E402


def atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(obj)); tmp.replace(path)


def run_list(cells, out: Path, smoke: bool) -> int:
    out.mkdir(parents=True, exist_ok=True); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); rc = 0; t0 = time.time()
    for i, c in enumerate(cells):
        f = out / (sep.cell_name(c) + ".json")
        if f.exists():
            print(f"[{i + 1}/{len(cells)}] skip (exists) {f.name}", flush=True); continue
        try:
            R = p157.reselect_finite(sep.run_cell(c, dev, smoke=smoke)); atomic_write_json(f, R)
            sel = {r["arm"]: (r["lr"], r["select"]["selected_update"]) for r in R["rows"] if r["selected"]}
            print(f"[{i + 1}/{len(cells)}] {f.name}: {R['wall_seconds']:.0f}s, selected (lr, update) {sel} (elapsed {time.time() - t0:.0f}s)", flush=True)
        except Exception as e:  # keep going; the cell stays missing and the aggregate reports it
            rc = 1; print(f"[{i + 1}/{len(cells)}] FAILED {f.name}: {type(e).__name__}: {e}", flush=True)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--seeds", required=True); r.add_argument("--out", required=True); r.add_argument("--smoke", action="store_true")
    r.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("pop"); p.add_argument("--out", required=True); p.add_argument("--smoke", action="store_true"); p.add_argument("--limit", type=int, default=0)
    sub.add_parser("units")
    a = ap.parse_args()
    if a.cmd == "units":
        print(json.dumps({"per_seed": len(p157.cells_for_seed(200)), "seeds": len(p157.SEEDS), "population": len(p157.population_cells())})); return 0
    cells = [c for s in a.seeds.replace(",", "+").split("+") for c in p157.cells_for_seed(int(s))] if a.cmd == "run" else p157.population_cells()
    if a.limit:
        cells = cells[:a.limit]
    return run_list(cells, Path(a.out), a.smoke)


if __name__ == "__main__":
    raise SystemExit(main())
