"""P108 runner (package v4 module E).  One process runs a list of cells and writes one JSON per cell (skips cells already written).

    python scripts/p108_estim.py --mech [--smoke] --out-dir outputs/P108_estim
    python scripts/p108_estim.py --cells 'E1:C1_gauss_mid:4096:0;E2:C2_gauss_pad:rot:1024:1' --out-dir outputs/P108_estim [--smoke] [--cpu]
    python scripts/p108_estim.py --list e1|e2          (print the full cell list, one token per line)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vcs_estim import p108  # noqa: E402
from vcs_estim.run import git_commit  # noqa: E402
from vcs_ssl.utils import atomic_write_json, utc_now  # noqa: E402


def token(c: tuple) -> str:
    if c[0] == "E1":
        return f"E1:{c[1]}:{c[2]}:{c[3]}"
    return f"E2:{c[1]}:{'rot' if c[2] else 'orig'}:{c[3]}:{c[4]}"


def name_of(tok: str) -> str:
    return "P108_" + tok.replace(":", "_")


def run_token(tok: str, device, smoke: bool) -> dict:
    f = tok.split(":")
    if f[0] == "E1":
        return p108.e1_cell(f[1], int(f[2]), int(f[3]), device, smoke)
    if f[0] == "E2":
        return p108.e2_cell(f[1], f[2] == "rot", int(f[3]), int(f[4]), device, smoke)
    raise ValueError(tok)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=None); ap.add_argument("--mech", action="store_true"); ap.add_argument("--list", default=None, choices=("e1", "e2"))
    ap.add_argument("--out-dir", default="/home/infres/yinwang/CS_QMI/outputs/P108_estim"); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    if a.list:
        for c in (p108.e1_cells() if a.list == "e1" else p108.e2_cells()):
            print(token(c))
        return 0
    device = torch.device("cpu") if a.cpu or not torch.cuda.is_available() else torch.device("cuda", 0)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True); meta = {"commit": git_commit(), "utc": utc_now(), "device": str(device), "smoke": a.smoke}
    if a.mech:
        t0 = time.time(); R = {"meta": meta, "conditions": {c: p108.mechanism_mc(c, smoke=a.smoke) for c in p108.CONDITIONS}}
        R["seconds"] = time.time() - t0; atomic_write_json(out / ("P108_E1_mechanism" + ("_smoke" if a.smoke else "") + ".json"), R)
        for c, r in R["conditions"].items():
            for u, v in r["U"].items():
                print(f"[mech {c} U={u}] slopes(all t) {v['loglog_slope_all_t']}  slopes(t<=0.1) {v['loglog_slope_t_le_0.1']}", flush=True)
        print(f"mechanism done ({R['seconds']:.0f}s)")
    for tok in [t for t in (a.cells or "").split(";") if t]:
        dst = out / (name_of(tok) + ("_smoke" if a.smoke else "") + ".json")
        if dst.is_file():
            print(f"skip {tok} (exists)", flush=True); continue
        r = run_token(tok, device, a.smoke); r["meta"] = meta; atomic_write_json(dst, r)
        sel = [x for x in r["rows"] if x.get("selected")]
        print(f"[{tok}] {r['wall_seconds']:.0f}s S={r['truth']['S']:.4f} " + " | ".join(f"{x['method']}@{x['budget_updates']}: J {x['J_err']:+.4f} plug {x['S_plug_err']:+.4f}" for x in sel[:8]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
