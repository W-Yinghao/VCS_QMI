"""P157 (SEP-2) aggregate (protocol reports/P157_SEP2_PROTOCOL_FROZEN_20261015.md).  Primary: Q1 (N 4,096), Q2 (N 16,384), Q3 (cos-2, both N):
per seed, the mean over the test's cells of (e_VCS - e_JS) / mean-over-seeds e_JS of that cell; paired t over the 8 seeds; Holm over 3."""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vcs_estim import sep_agg as A  # noqa: E402


def collect(dirpath: Path, which: str):
    S2, POP, qc = defaultdict(dict), {}, defaultdict(lambda: defaultdict(int))
    for f in sorted(dirpath.glob("P156_*_p157.json")):
        R = json.loads(f.read_text()); c = R["cell"]; key = (c["cls"], c["ctype"], c["eps"], c["d"], c["I"], c["N"])
        ej, ev = A.e_of(R, "js", which), A.e_of(R, "vcs", which)
        if c["block"] == "S2":
            S2[key][c["seed"]] = (ev, ej)
            for r in R["rows"]:
                if r["selected"]:
                    q = qc[(c["cls"], r["arm"])]; q["n"] += 1; q["top_lr"] += r["lr_edge"] == "high"; q["low_lr"] += r["lr_edge"] == "low"
                    q["not_converged"] += bool(r["select"].get("not_converged"))
        elif c["block"] == "S2P":
            POP[key[:5]] = {"e_js": ej, "e_vcs": ev, "delta": None if ej is None or ev is None else ev - ej,
                            "lr": {r["arm"]: r["lr"] for r in R["rows"] if r["selected"]}}
    return S2, POP, qc


def pooled(S2, keys):
    per_seed = defaultdict(list)
    for k in keys:
        v = S2.get(k, {}); ref = [x[1] for x in v.values() if x[1] is not None]
        if not ref:
            continue
        m = st.mean(ref)
        for s, (ev, ej) in v.items():
            if ev is not None and ej is not None:
                per_seed[s].append((ev - ej) / m)
    complete = all(len(per_seed[s]) == len(keys) for s in per_seed) and len(per_seed) == 8
    return {"stat": A.tstat([st.mean(x) for x in per_seed.values()]), "n_cells": len(keys), "complete": complete,
            "per_seed": {s: st.mean(x) for s, x in sorted(per_seed.items())}}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    res = {}
    for which in ("selected", "selected_commonJ"):
        S2, POP, qc = collect(Path(a.dir), which); keys = sorted(S2)
        tests = {"Q1_N4096": [k for k in keys if k[5] == 4096], "Q2_N16384": [k for k in keys if k[5] == 16384], "Q3_cos2_bothN": [k for k in keys if k[0] == "cos2"]}
        prim = {t: pooled(S2, ks) for t, ks in tests.items()}; ph = A.holm([prim[t]["stat"]["p"] for t in prim])
        for t, h in zip(prim, ph):
            prim[t]["p_holm"] = h; prim[t]["reading"] = A.reading(prim[t]["stat"], h)
        cells = []
        for k in keys:
            d = [ev - ej for ev, ej in S2[k].values() if ev is not None and ej is not None]
            cells.append({"cls": k[0], "ctype": k[1], "eps": k[2], "d": k[3], "I": k[4], "N": k[5], "n_seeds": len(d),
                          "e_js": st.mean(x[1] for x in S2[k].values()), "e_vcs": st.mean(x[0] for x in S2[k].values()), "delta": A.tstat(d)})
        hp = A.holm([c["delta"]["p"] for c in cells])
        for c, h in zip(cells, hp):
            c["p_holm"] = h; c["reading"] = A.reading(c["delta"], h)
        q1, q2 = prim["Q1_N4096"], prim["Q2_N16384"]
        neg = lambda q: q["p_holm"] is not None and q["p_holm"] < 0.05 and q["stat"]["mean"] < 0
        outcome = "CONFIRMED (persists at 4x N)" if neg(q1) and neg(q2) else "FINITE-SAMPLE ONLY" if neg(q1) else "NOT CONFIRMED"
        res[which] = {"primary": prim, "outcome": outcome, "cells": cells, "population_block": [{"cls": k[0], "ctype": k[1], "eps": k[2], "d": k[3], "I": k[4], **v} for k, v in sorted(POP.items())],
                      "qc": {f"{c}:{arm}": dict(v) for (c, arm), v in sorted(qc.items())}, "n_cells": len(keys)}
    Path(a.out + ".json").write_text(json.dumps(res, indent=1))
    L = ["# P157 (SEP-2) aggregate", "", "Effects: (e_VCS − e_JS) / mean e_JS of the cell (negative = VCS more accurate); paired t over 8 seeds.", ""]
    for which, r in res.items():
        L += [f"## Selection: {which} — outcome: **{r['outcome']}** ({r['n_cells']} cells)", "", "| test | mean [95 % t] | p | p Holm | cells | complete | reading |", "|---|---|---|---|---|---|---|"]
        for t, q in r["primary"].items():
            s = q["stat"]; ci = s["ci95"] or [float("nan")] * 2
            L.append(f"| {t} | {s['mean']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] | {s['p']:.3g} | {q['p_holm']:.3g} | {q['n_cells']} | {q['complete']} | {q['reading']} |")
        L.append("")
        if which == "selected":
            L += ["### Cells (absolute Δe, Holm over all cells)", "", "| class | type | ε | d | I | N | e_JS | e_VCS | Δe [95 % t] | p Holm | reading |", "|---|---|---|---|---|---|---|---|---|---|---|"]
            for c in r["cells"]:
                s = c["delta"]; ci = s["ci95"] or [float("nan")] * 2
                L.append(f"| {c['cls']} | {c['ctype']} | {c['eps']:g} | {c['d']} | {c['I']:g} | {c['N']} | {c['e_js']:.4f} | {c['e_vcs']:.4f} | {s['mean']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] | {c['p_holm']:.3g} | {c['reading']} |")
            L += ["", "### Population block S2P (N 131,072, 8,000 updates, seed 200; descriptive)", "", "| class | type | ε | d | I | e_JS | e_VCS | Δe | lr js / vcs |", "|---|---|---|---|---|---|---|---|---|"]
            for p in r["population_block"]:
                L.append(f"| {p['cls']} | {p['ctype']} | {p['eps']:g} | {p['d']} | {p['I']:g} | {p['e_js']:.4f} | {p['e_vcs']:.4f} | {p['delta']:+.4f} | {p['lr'].get('js')} / {p['lr'].get('vcs')} |")
            L += ["", "### QC (selected rows): top-lr (1.0) / low-lr / not converged", ""] + [f"- {k}: {v}" for k, v in r["qc"].items()] + [""]
    Path(a.out + ".md").write_text("\n".join(L) + "\n"); print("\n".join(L[:20])); return 0


if __name__ == "__main__":
    raise SystemExit(main())
