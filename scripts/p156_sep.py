"""P156 (SEP-1) runner.

    python scripts/p156_sep.py run --seed 100 --out /home/infres/yinwang/CS_QMI/outputs/P156_sep            # every cell of one seed (resumable)
    python scripts/p156_sep.py run --seed 100 --out ... --blocks A,E --limit 3 --smoke --cpu               # CPU smoke
    python scripts/p156_sep.py repro --out ...                                                              # QC: the P134 clean cell, seed 0
    python scripts/p156_sep.py match --out reports/P156_S_MATCHING                                          # block-B matching table (CPU)
    python scripts/p156_sep.py timing --out /home/infres/yinwang/CS_QMI/outputs/P156_TIMING                 # two full cells, not reused
    python scripts/p156_sep.py units --seed 100                                                             # list the cells

Writes <out>/<cell name>.json atomically and skips cells whose JSON exists.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vcs_estim import sep  # noqa: E402
from vcs_ssl.utils import atomic_write_json  # noqa: E402


def _device(cpu: bool):
    return torch.device("cuda", 0) if torch.cuda.is_available() and not cpu else torch.device("cpu")


def run_list(cells, out: Path, device, smoke: bool) -> int:
    out.mkdir(parents=True, exist_ok=True); rc = 0; t0 = time.time(); matching = None
    for i, c in enumerate(cells):
        f = out / (sep.cell_name(c) + ".json")
        if f.exists():
            print(f"[{i + 1}/{len(cells)}] skip (exists) {f.name}", flush=True); continue
        if c["ctype"] == "matched" and matching is None:
            matching = sep.load_matching()
        try:
            R = sep.run_cell(c, device, smoke=smoke, matching=matching); atomic_write_json(f, R)
            sel = {r["arm"]: (r["lr"], r["select"]["selected_update"]) for r in R["rows"] if r["selected"]}
            print(f"[{i + 1}/{len(cells)}] {f.name}: {R['wall_seconds']:.0f}s, selected (lr, update) {sel}, S {R['truth']['S']:.4f} "
                  f"(elapsed {time.time() - t0:.0f}s)", flush=True)
        except Exception as e:  # keep going; the cell stays missing and the aggregate reports it
            rc = 1; print(f"[{i + 1}/{len(cells)}] FAILED {f.name}: {type(e).__name__}: {e}", flush=True)
            import traceback; traceback.print_exc()
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--seed", type=int, required=True); r.add_argument("--out", required=True)
    r.add_argument("--blocks", default="A,E,C,D,Dp,B,B2"); r.add_argument("--limit", type=int, default=None)
    r.add_argument("--smoke", action="store_true"); r.add_argument("--cpu", action="store_true")
    q = sub.add_parser("repro"); q.add_argument("--out", required=True); q.add_argument("--cpu", action="store_true")
    m = sub.add_parser("match"); m.add_argument("--out", required=True); m.add_argument("--n", type=int, default=sep.MATCH_N)
    t = sub.add_parser("timing"); t.add_argument("--out", required=True); t.add_argument("--cpu", action="store_true")
    x = sub.add_parser("addendum1"); x.add_argument("--out", required=True); x.add_argument("--seeds", default="100+101+102+103+104+105+106+107")
    g = sub.add_parser("merge1"); g.add_argument("--src", required=True); g.add_argument("--ext", required=True); g.add_argument("--dst", required=True)
    u = sub.add_parser("units"); u.add_argument("--seed", type=int, default=100)
    a = ap.parse_args()
    if a.cmd == "units":
        cs = sep.cells_for_seed(a.seed)
        for c in cs:
            print(sep.cell_name(c))
        from collections import Counter
        print("per block:", dict(Counter(c["block"] for c in cs)), "total", len(cs), file=sys.stderr); return 0
    if a.cmd == "match":
        t0 = time.time(); rows = sep.matching_table(a.n)
        obj = {"protocol_id": sep.PROTOCOL, "n_mc": a.n, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": time.time() - t0,
               "method": "S_eps: seed-independent Monte Carlo of the contaminated block-A problem; rho': bisection on common random numbers for S_clean(rho'; d) = S_eps",
               "rows": rows}
        atomic_write_json(a.out + ".json", obj)
        L = ["# P156 block-B S-matching table (written before block B runs)", "", f"Monte Carlo n = {a.n} per side.", "",
             "S-matched (block B): rho' with S_clean(rho') = S_eps.  Saturation-matched (block B2, eps 0.3 rows used): rho'' with sat95_clean(rho'') = sat95_eps.", "",
             "| type | eps | d | I | S_eps (se) | rho' | I' | S_clean(rho') | sat95 eps (P / Q side) | sat95 S-matched (P / Q side) | rho'' | S at rho'' | sat95 at rho'' |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for x in rows:
            L.append(f"| {x['ctype']} | {x['eps']:g} | {x['d']} | {x['I']:g} | {x['S_eps']:.4f} ({x['S_eps_se']:.4f}) | {x['rho_prime']:.5f} | {x['I_prime']:.3f} | "
                     f"{x['S_clean_at_rho_prime']:.4f} | {x['sat95_eps']:.3f} ({x['sat95_P_side_eps']:.3f} / {x['sat95_Q_side_eps']:.3f}) | "
                     f"{x['sat95_matched']:.3f} ({x['sat95_P_side_matched']:.3f} / {x['sat95_Q_side_matched']:.3f}) | {x['rho_sat']:.5f} | {x['S_clean_at_rho_sat']:.4f} | "
                     f"{x['sat95_at_rho_sat']:.3f} |")
        Path(a.out + ".md").write_text("\n".join(L) + "\n"); print("\n".join(L)); return 0
    if a.cmd == "addendum1":
        cs = [c for sd in a.seeds.replace(",", "+").split("+") for c in sep.addendum1_cells(int(sd))]
        print(f"[p156] addendum 1: {len(cs)} extension cells", flush=True); return run_list(cs, Path(a.out), _device(False), False)
    if a.cmd == "merge1":
        import shutil
        src, extd, dst = Path(a.src), Path(a.ext), Path(a.dst); n_m = n_c = 0
        want = {sep.cell_name(c) for sd in range(100, 108) for c in sep.addendum1_cells(sd)}
        have = {f.stem for f in extd.glob("P156_*_lrext.json")}
        if want - have:
            print(f"[p156] merge1: {len(want - have)} extension cells missing, e.g. {sorted(want - have)[:3]}"); return 1
        dst.mkdir(parents=True, exist_ok=False)
        for f in sorted(src.glob("P156_*.json")):
            ext = extd / (f.stem + "_lrext.json")
            if ext.exists():
                atomic_write_json(dst / f.name, sep.merge_extension(json.loads(f.read_text()), json.loads(ext.read_text()))); n_m += 1
            else:
                shutil.copy2(f, dst / f.name); n_c += 1
        print(json.dumps({"merged": n_m, "copied": n_c})); return 0
    if a.cmd == "repro":
        return run_list([sep.reproduction_cell()], Path(a.out), _device(a.cpu), False)
    if a.cmd == "timing":
        cs = [sep.cell("T", "mlp256", "product", "independent", 0.3, 64, 8.0, 4096, 2000, sep.ARMS_ALL, sep.LRS_MLP, 100),
              sep.cell("T", "mlp256", "product", "independent", 0.3, 20, 4.0, 16384, 8000, sep.ARMS_PAIR, sep.LRS_MLP, 100)]
        return run_list(cs, Path(a.out), _device(a.cpu), False)
    blocks = set(a.blocks.replace("+", ",").split(",")); cs = [c for c in sep.cells_for_seed(a.seed) if c["block"] in blocks]
    if a.limit:
        cs = cs[: a.limit]
    print(f"[p156] seed {a.seed}: {len(cs)} cells ({a.blocks}) on {_device(a.cpu)}{' SMOKE' if a.smoke else ''}", flush=True)
    return run_list(cs, Path(a.out), _device(a.cpu), a.smoke)


if __name__ == "__main__":
    raise SystemExit(main())
