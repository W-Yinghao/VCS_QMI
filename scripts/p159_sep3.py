"""P159 (SEP-3) runner.  run --seeds 300+301 --out DIR [--sources coco_clip,cifar_ap3] [--smoke] [--limit k] | units.  Resumable (finished cells are
skipped); a failed cell is reported and skipped (the aggregate lists missing cells); a partial record (cell + rows so far) is not written because
every cell is short — the unit of resumption is the cell."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch  # noqa: E402

from vcs_estim import p159  # noqa: E402


def atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(obj)); tmp.replace(path)


def run_list(cells, out: Path, smoke: bool) -> int:
    out.mkdir(parents=True, exist_ok=True); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); rc = 0; t0 = time.time()
    for i, c in enumerate(cells):
        f = out / (p159.cell_name(c) + ".json")
        if f.exists():
            print(f"[{i + 1}/{len(cells)}] skip (exists) {f.name}", flush=True); continue
        try:
            R = p159.run_cell(c, dev, smoke=smoke); atomic_write_json(f, R)
            sel = {r["arm"]: (r["lr"], r["select"]["selected_update"], round(r["readout"]["J_eval"], 4)) for r in R["rows"] if r["selected"]}
            print(f"[{i + 1}/{len(cells)}] {f.name}: {R['wall_seconds']:.0f}s, selected (lr, update, J_eval) {sel} (elapsed {time.time() - t0:.0f}s)", flush=True)
        except Exception as e:  # keep going; the cell stays missing and the aggregate reports it
            rc = 1; print(f"[{i + 1}/{len(cells)}] FAILED {f.name}: {type(e).__name__}: {e}", flush=True)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--seeds", required=True); r.add_argument("--out", required=True); r.add_argument("--smoke", action="store_true")
    r.add_argument("--sources", default=",".join(p159.SOURCES)); r.add_argument("--limit", type=int, default=0)
    sub.add_parser("units")
    a = ap.parse_args()
    if a.cmd == "units":
        print(json.dumps({"per_seed": len(p159.cells_for_seed(300)), "seeds": len(p159.SEEDS), "sources": list(p159.SOURCES), "NS": {s: p159.SRC[s]["NS"] for s in p159.SOURCES}})); return 0
    srcs = tuple(a.sources.split(","))
    cells = [c for s in a.seeds.replace(",", "+").split("+") for c in p159.cells_for_seed(int(s), srcs)]
    if a.limit:
        cells = cells[:a.limit]
    return run_list(cells, Path(a.out), a.smoke)


if __name__ == "__main__":
    raise SystemExit(main())
