"""P134 (R1 contamination) aggregate: shifts of each estimator's native value in its own resolution units (P85 clean staircase I = 2 / 4 / 6),
the decomposition into target change and estimation-error change, and the pre-stated reading.

    python scripts/p134_aggregate.py --p134 /home/infres/yinwang/CS_QMI/outputs/P134_r1 \
        --p85 /home/infres/yinwang/CS_QMI/outputs/P85_estim_benchmark --out reports/P134_r1_aggregate
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vcs_estim import r1  # noqa: E402
from vcs_estim.p85_grid import cell as make_cell, cell_name as p85_name  # noqa: E402

PRIMARY = {"vcs": "product", "js": "product", "nwj": "inbatch", "infonce": "inbatch"}   # NWJ product: clean staircase not ordered (P85 V(I=6) = -60.9)
TARGET = {"vcs": "S", "js": "JS2", "infonce": "MI", "nwj": "MI"}


def selected(R: dict) -> dict:
    return {(r["kind"], r["negative_construction"]): r for r in R["rows"] if r.get("selected") and r["kind"] in r1.KINDS_R1}


def msd(v):
    v = np.asarray(v, float); return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--p134", required=True); ap.add_argument("--p85", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); P, Q = Path(a.p134), Path(a.p85)
    # clean staircase (P85) -> resolution units per (kind, construction)
    stair, missing = {}, []
    for I in (2.0, 4.0, 6.0):
        for s in r1.SEEDS:
            f = Q / (p85_name(make_cell("gaussian", 20, 20, I, 4096, 256, 2000, s)) + ".json")
            if not f.exists():
                missing.append(str(f)); continue
            for k, r in selected(json.loads(f.read_text())).items():
                stair.setdefault(k, {}).setdefault(I, {})[s] = r["native"]["value"]
    units = {}
    for k, d in stair.items():
        if all(len(d.get(I, {})) == len(r1.SEEDS) for I in (2.0, 4.0, 6.0)):
            v = {I: float(np.mean(list(d[I].values()))) for I in (2.0, 4.0, 6.0)}
            ru = r1.resolution_units(v[2.0], v[4.0], v[6.0])
            units[k] = {**ru, "V2": v[2.0], "V4": v[4.0], "V6": v[6.0], "ordered": ru["u_down"] > 0 and ru["u_up"] > 0}
    # P134 cells
    cells = {}
    for f in sorted(P.glob("P134_*.json")):
        R = json.loads(f.read_text()); c = R["cell"]["contamination"]; cells[(c["type"], c["eps"], R["cell"]["base"]["seed"])] = R
    clean = {s: cells.get(("clean", 0.0, s)) for s in r1.SEEDS}
    consistency = []
    for s, R in clean.items():
        if R is None:
            missing.append(f"P134 clean seed {s}"); continue
        for k, r in selected(R).items():
            ref = stair.get(k, {}).get(4.0, {}).get(s)
            consistency.append({"method": f"{k[0]}:{k[1]}", "seed": s, "p134_eps0": r["native"]["value"], "p85_I4": ref,
                                "diff": None if ref is None else r["native"]["value"] - ref})
    table, maxima = [], {}
    for t in r1.TYPES:
        for eps in [e for e in r1.EPS if e > 0]:
            per = {}
            for s in r1.SEEDS:
                R, R0 = cells.get((t, eps, s)), clean[s]
                if R is None or R0 is None:
                    missing.append(f"P134 {t} eps {eps} seed {s}"); continue
                S1, S0 = selected(R), selected(R0)
                for k in S1:
                    if k not in S0:
                        continue
                    tgt = TARGET[k[0]]; v1, v0 = S1[k]["native"]["value"], S0[k]["native"]["value"]
                    t1, t0 = R["truth"][tgt], R0["truth"][tgt]
                    per.setdefault(k, []).append((v1 - v0, t1 - t0, (v1 - t1) - (v0 - t0)))
            for k, lst in per.items():
                u = units.get(k); u = u if u and u["ordered"] else None      # no resolution unit when the clean staircase is not ordered around I = 4
                dm, dsd = msd([x[0] for x in lst]); tm, _ = msd([x[1] for x in lst]); em, esd = msd([x[2] for x in lst])
                row = {"type": t, "eps": eps, "method": f"{k[0]}:{k[1]}", "kind": k[0], "construction": k[1], "primary": PRIMARY.get(k[0]) == k[1],
                       "n_seeds": len(lst), "shift_mean": dm, "shift_sd": dsd, "target_change": tm, "error_change_mean": em, "error_change_sd": esd}
                if u:
                    ud = u["u_down"] if dm < 0 else u["u_up"]
                    row.update({"u": u["u"], "shift_units": dm / u["u"], "shift_units_sd": dsd / abs(u["u"]), "target_change_units": tm / u["u"],
                                "norm_shift": r1.normalized_shift(dm, u["u"], eps), "norm_shift_directional": r1.normalized_shift(dm, ud, eps)})
                    mk = (k, t); maxima[mk] = max(maxima.get(mk, 0.0), row["norm_shift"])
                table.append(row)
    # reading on the primary rows and primary types
    m = {kind: {t: maxima.get(((kind, PRIMARY[kind]), t), float("nan")) for t in r1.PRIMARY_TYPES} for kind in PRIMARY}
    complete = all(math.isfinite(v) for kind in ("vcs", "js") for v in m[kind].values())
    label = r1.reading_label(m["vcs"], m["js"]) if complete else "incomplete"
    out = {"units": {f"{k[0]}:{k[1]}": v for k, v in units.items()}, "consistency_eps0_vs_p85": consistency, "table": table,
           "max_norm_shift_primary": m, "reading": label, "missing": missing}
    Path(a.out + ".json").write_text(json.dumps(out, indent=1))
    L = ["# P134 — R1 contamination: aggregate", "", f"Reading (pre-stated; VCS product vs JS product, primary types {', '.join(r1.PRIMARY_TYPES)}): **{label}**", "",
         "## Resolution units (P85 clean staircase, selected rows, mean of seeds 0-2)", "", "| method | V(I=2) | V(I=4) | V(I=6) | u = (V6 - V2)/2 | u_down | u_up |", "|---|---|---|---|---|---|---|"]
    for k, v in sorted(units.items()):
        L.append(f"| {k[0]}:{k[1]} | {v['V2']:.4f} | {v['V4']:.4f} | {v['V6']:.4f} | {v['u']:.4f}{'' if v['ordered'] else ' (not ordered: no unit)'} | {v['u_down']:.4f} | {v['u_up']:.4f} |")
    L += ["", "## Max over eps of |shift| / u per 0.1 eps (primary rows; bound 1)", "", "| estimator | " + " | ".join(r1.PRIMARY_TYPES) + " |", "|---|" + "---|" * len(r1.PRIMARY_TYPES)]
    for kind in PRIMARY:
        L.append(f"| {kind}:{PRIMARY[kind]} | " + " | ".join(f"{m[kind][t]:.2f}" for t in r1.PRIMARY_TYPES) + " |")
    L += ["", "## Shifts per cell (mean ± sd over seeds; units = own resolution units)", "",
          "| type | eps | method | shift | shift (units) | target change (units) | error change | norm. shift | norm. shift (directional) |", "|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        if "u" not in r:
            L.append(f"| {r['type']} | {r['eps']:g} | {r['method']}{' *' if r['primary'] else ''} | {r['shift_mean']:+.4f} ± {r['shift_sd']:.4f} | — | — | {r['error_change_mean']:+.4f} | — | — |"); continue
        L.append(f"| {r['type']} | {r['eps']:g} | {r['method']}{' *' if r['primary'] else ''} | {r['shift_mean']:+.4f} ± {r['shift_sd']:.4f} | {r['shift_units']:+.2f} | "
                 f"{r['target_change_units']:+.2f} | {r['error_change_mean']:+.4f} | {r['norm_shift']:.2f} | {r['norm_shift_directional']:.2f} |")
    L += ["", "## eps = 0 consistency with P85 I = 4 (same roles, seeds, trainer)", "", "| method | seed | P134 eps 0 | P85 I 4 | diff |", "|---|---|---|---|---|"]
    for c in consistency:
        ref = "—" if c["p85_I4"] is None else f"{c['p85_I4']:.5f}"; diff = "—" if c["diff"] is None else f"{c['diff']:+.1e}"
        L.append(f"| {c['method']} | {c['seed']} | {c['p134_eps0']:.5f} | {ref} | {diff} |")
    if missing:
        L += ["", "## Missing", ""] + [f"- {x}" for x in missing]
    Path(a.out + ".md").write_text("\n".join(L) + "\n"); print(f"-> {a.out}.md / .json ({len(cells)} cells; reading: {label})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
