"""E line / R1 — run one or more staircase cells.

    python scripts/estim_staircase.py --setting gaussian --kind vcs --variant cyclic8 --N 64 --lr 5e-4 --seed 0 --out <dir>
    python scripts/estim_staircase.py --units <file> --out <dir>          # one 'setting kind variant N lr seed [contam_kind eps]' per line

Writes <out>/<setting>_<kind>_<variant>_N<N>_lr<lr>_s<seed>[_<contam><eps>].json (skips cells whose JSON exists).  --smoke: 300 steps per level,
2 levels, log every 20.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vcs_estim.staircase import run_cell  # noqa: E402


def cell_name(c):
    base = f"{c['setting']}_{c['kind']}_{c['variant']}_N{c['N']}_lr{c['lr']:g}_s{c['seed']}"
    return base + (f"_{c['contamination']['kind']}{c['contamination']['eps']:g}" if c.get("contamination") else "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--setting", default="gaussian"); ap.add_argument("--kind", default="vcs"); ap.add_argument("--variant", default="cyclic8")
    ap.add_argument("--N", type=int, default=64); ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--contam-kind", default=None); ap.add_argument("--eps", type=float, default=0.0)
    ap.add_argument("--steps-per-level", type=int, default=4000); ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--levels", default=None, help="comma list of level indices (default all five)"); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cells = []
    if a.units:
        for line in Path(a.units).read_text().splitlines():
            f = line.split()
            if not f or f[0].startswith("#"): continue
            c = {"setting": f[0], "kind": f[1], "variant": f[2], "N": int(f[3]), "lr": float(f[4]), "seed": int(f[5])}
            if len(f) >= 8: c["contamination"] = {"kind": f[6], "eps": float(f[7])}
            cells.append(c)
    else:
        c = {"setting": a.setting, "kind": a.kind, "variant": a.variant, "N": a.N, "lr": a.lr, "seed": a.seed}
        if a.contam_kind: c["contamination"] = {"kind": a.contam_kind, "eps": a.eps}
        cells.append(c)
    spl, le, levels = a.steps_per_level, a.log_every, ([int(x) for x in a.levels.split(",")] if a.levels else None)
    if a.smoke: spl, le, levels = 300, 20, levels or [0, 2]
    for c in cells:
        f = out / (cell_name(c) + ".json")
        if f.exists(): print("skip", f.name); continue
        t0 = time.time()
        res = run_cell(setting=c["setting"], kind=c["kind"], variant=c["variant"], N=c["N"], lr=c["lr"], seed=c["seed"], steps_per_level=spl, log_every=le, levels=levels,
                       contamination=c.get("contamination"), truth_mc=50_000 if a.smoke else 200_000)
        tmp = f.with_suffix(".tmp"); tmp.write_text(json.dumps(res)); tmp.replace(f)
        lv = res["levels"]
        print(f"{f.name}: " + " | ".join(f"L{l['level']} {l['mean_last']:.3f}±{l['sd_last']:.3f} (truth {l['truth'][res['target']]:.3f}, oracle {l['oracle_mean_last']:.3f}, div {l['divergences']})" for l in lv) + f"  [{time.time()-t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
