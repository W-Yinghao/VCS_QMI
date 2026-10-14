"""P156 aggregate: the pre-stated tests of protocol §5 on the cell JSONs written by `vcs_estim.sep.run_cell`.

Unit of inference = seed.  Effects are standardised by the clean JS error of the same (class, variant, d, I, N, updates) cell, mean over seeds.
Primary family (Holm over 3): P1 = mean over the 12 block-A cells of [Delta e(0.3) - Delta e(0)], P2 = mean of [Delta e(0.3) - Delta e(S-matched
to 0.3)], P3 = mean over the 8 restricted cells (cos2, mlp8 x d x I) of Delta e(0).  Delta e = e_vcs - e_js on the native-selected rows
(negative = VCS more accurate).  Secondary lists: per-cell one-sample t over seeds, Holm within each list.  Descriptive tables follow.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

KEYF = ("block", "cls", "variant", "ctype", "eps", "d", "I", "N", "updates")
TYPES_A = ("independent", "false_negative")
DS, IS = (20, 64), (2.0, 4.0, 8.0)


def key_of(c: dict) -> tuple:
    k = tuple(c[f] for f in KEYF)
    return k + ((c["match"]["ctype"], c["match"]["eps"], c["match"].get("by", "S")),) if c.get("match") else k + (None,)


def load(dirpath) -> dict:
    """{cell key (without seed): {seed: cell json}}; smoke cells are skipped unless every file is a smoke file (toy aggregation)."""
    files = sorted(Path(dirpath).glob("P156_*.json")); out = defaultdict(dict); smoke = []
    for f in files:
        R = json.loads(f.read_text()); c = R["cell"]
        if c.get("block") == "R":
            continue
        smoke.append(bool(c.get("smoke"))); out[key_of(c)][c["seed"]] = R
    return dict(out), (all(smoke) if smoke else False)


def sel_row(R: dict, arm: str, which: str = "selected") -> dict | None:
    for r in R["rows"]:
        if r["arm"] == arm and r.get(which):
            return r
    return None


def e_of(R: dict, arm: str, which: str = "selected", field: str = "posterior_mse") -> float | None:
    r = sel_row(R, arm, which)
    if r is None:
        return None
    src = r["readout"] if which == "selected" else r["readout_commonJ_selected"]
    return src.get(field)


def delta(R: dict, which: str = "selected") -> float | None:
    a, b = e_of(R, "vcs", which), e_of(R, "js", which)
    return None if a is None or b is None else a - b


def tstat(x) -> dict:
    x = np.asarray([v for v in x if v is not None and math.isfinite(v)], float); n = len(x)
    if n < 2:
        return {"n": n, "mean": float(x.mean()) if n else None, "ci95": None, "t": None, "p": None}
    m, sd = float(x.mean()), float(x.std(ddof=1)); se = sd / math.sqrt(n)
    if se == 0:
        return {"n": n, "mean": m, "sd": sd, "ci95": [m, m], "t": None, "p": 0.0 if m != 0 else 1.0}
    t = m / se; p = float(2 * stats.t.sf(abs(t), n - 1)); h = float(stats.t.ppf(0.975, n - 1) * se)
    return {"n": n, "mean": m, "sd": sd, "ci95": [m - h, m + h], "t": float(t), "p": p}


def holm(ps: list) -> list:
    idx = [i for i, p in enumerate(ps) if p is not None]; order = sorted(idx, key=lambda i: ps[i]); m = len(order); adj = [None] * len(ps); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj


def reading(s: dict, p_adj) -> str:
    if p_adj is None or s.get("mean") is None:
        return "not computable"
    if p_adj < 0.05:
        return "< 0 (VCS more accurate)" if s["mean"] < 0 else "> 0 (JS more accurate)"
    return "n.s."


class Agg:
    def __init__(self, cells: dict):
        self.C = cells

    def get(self, **kw) -> dict:
        """{seed: R} of the unique cell matching the key fields given (match=None by default)."""
        match = kw.pop("match", None); hits = [k for k in self.C if all(k[KEYF.index(f)] == v for f, v in kw.items()) and k[-1] == match]
        if len(hits) > 1:
            raise KeyError(f"ambiguous {kw}: {hits}")
        return self.C[hits[0]] if hits else {}

    def clean_ref(self, cls, variant, d, I, N=4096, U=2000, block=None) -> float | None:
        blk = block if block else ("Dp" if variant != "product" else {"mlp256": "A", "mlp8": "E", "cos2": "E", "quad3": "E"}[cls])
        cc = self.get(block=blk, cls=cls, variant=variant, ctype="clean", eps=0.0, d=d, I=I, N=N, updates=U)
        v = [e_of(R, "js") for R in cc.values()]; v = [x for x in v if x is not None]
        return float(np.mean(v)) if v else None

    def paired(self, a: dict, b: dict | None, scale: float | None, which="selected") -> dict:
        """Per-seed (Delta e(a) - Delta e(b)) / scale (b None: Delta e(a) / scale)."""
        out = {}
        for s, R in a.items():
            da = delta(R, which)
            if da is None or not scale:
                continue
            if b is None:
                out[s] = da / scale
            elif s in b and delta(b[s], which) is not None:
                out[s] = (da - delta(b[s], which)) / scale
        return out


def ref_for(A: "Agg", c0: dict) -> float | None:
    """The standardising clean JS error of a cell's family: A / B / D -> the block-A clean cell of (d, I); E -> the class's clean cell;
    C -> the block-C clean cell with the same (N, updates) (block A for 4 096 / 2 000); D' -> the cyclic clean cell."""
    b = c0["block"]
    if b in ("A", "B", "B2", "D"):
        return A.clean_ref("mlp256", "product", c0["d"], c0["I"])
    if b == "E":
        return A.clean_ref(c0["cls"], "product", c0["d"], c0["I"])
    if b == "C":
        return A.clean_ref("mlp256", "product", c0["d"], c0["I"], c0["N"], c0["updates"], block="C")
    if b == "Dp":
        return A.clean_ref("mlp256", "cyclic8", c0["d"], c0["I"])
    return None


def pooled(per_cell: list[dict]) -> dict:
    """Per-seed mean over cells (seeds present in every cell), then a one-sample t over seeds."""
    if not per_cell or any(not c for c in per_cell):
        return {"stat": tstat([]), "seeds": [], "n_cells": len(per_cell), "complete": False}
    seeds = sorted(set.intersection(*[set(c) for c in per_cell])); vals = [float(np.mean([c[s] for c in per_cell])) for s in seeds]
    return {"stat": tstat(vals), "seeds": seeds, "per_seed": vals, "n_cells": len(per_cell), "complete": True}


def aggregate(cells: dict) -> dict:
    A = Agg(cells); res = {"primary": {}, "secondary": {}, "descriptive": {}}
    # ---------------- primary
    P1, P2, P3, cellsP1, cellsP2, cellsP3, P2s, cellsP2s = [], [], [], [], [], [], [], []
    for ct in TYPES_A:
        for d in DS:
            for I in IS:
                sc = A.clean_ref("mlp256", "product", d, I)
                cont = A.get(block="A", cls="mlp256", variant="product", ctype=ct, eps=0.3, d=d, I=I, N=4096, updates=2000)
                clean = A.get(block="A", cls="mlp256", variant="product", ctype="clean", eps=0.0, d=d, I=I, N=4096, updates=2000)
                matched = A.get(block="B", cls="mlp256", variant="product", ctype="matched", eps=0.0, d=d, I=I, N=4096, updates=2000, match=(ct, 0.3, "S"))
                msat = A.get(block="B2", cls="mlp256", variant="product", ctype="matched", eps=0.0, d=d, I=I, N=4096, updates=2000, match=(ct, 0.3, "sat95"))
                g = A.paired(cont, clean, sc); h = A.paired(cont, matched, sc); hs = A.paired(cont, msat, sc); P1.append(g); P2.append(h); P2s.append(hs)
                cellsP2s.append({"ctype": ct, "d": d, "I": I, "H_sat": tstat(list(hs.values()))})
                cellsP1.append({"ctype": ct, "d": d, "I": I, "scale_js_clean": sc, "G": tstat(list(g.values())), "abs_delta_0.3": tstat(list(A.paired(cont, None, sc).values()))})
                cellsP2.append({"ctype": ct, "d": d, "I": I, "H": tstat(list(h.values()))})
    for cls in ("cos2", "mlp8"):
        for d in DS:
            for I in (4.0, 8.0):
                sc = A.clean_ref(cls, "product", d, I); cc = A.get(block="E", cls=cls, variant="product", ctype="clean", eps=0.0, d=d, I=I, N=4096, updates=2000)
                q = A.paired(cc, None, sc); P3.append(q); cellsP3.append({"cls": cls, "d": d, "I": I, "delta0": tstat(list(q.values()))})
    prim = {"P1_contamination_narrows_gap": pooled(P1), "P2_contamination_specific": pooled(P2), "P3_restricted_class_reversal": pooled(P3)}
    adj = holm([v["stat"]["p"] for v in prim.values()])
    for (k, v), pa in zip(prim.items(), adj):
        v["p_holm"] = pa; v["reading"] = reading(v["stat"], pa)
    res["primary"] = prim
    r1, r2, r3 = (prim[k]["reading"] for k in prim)
    cand = []
    if r1.startswith("<") and r2 == "n.s.":
        cand.append("O1 saturation region (check the saturation-model table: contaminated cells on the clean line)")
    if r1.startswith("<") and r2.startswith("<"):
        cand.append("O2 contamination-specific")
    if r1 == "n.s." and r3 == "n.s.":
        cand.append("O3 no region")
    if r3.startswith("<"):
        cand.append("O4 misspecification region")
    res["outcome_candidates"] = cand or ["none of O1-O4 by the mechanical rule (report the pattern as observed)"]
    # ---------------- secondary (Holm within each list)
    def holm_list(rows, key):
        adj = holm([r[key]["p"] for r in rows])
        for r, pa in zip(rows, adj):
            r["p_holm"] = pa; r["reading"] = reading(r[key], pa)
        return rows
    res["secondary"]["abs_delta_eps0.3_A"] = holm_list([{**{k: c[k] for k in ("ctype", "d", "I")}, "stat": c["abs_delta_0.3"]} for c in cellsP1], "stat")
    res["secondary"]["G_per_cell"] = holm_list([{**{k: c[k] for k in ("ctype", "d", "I")}, "stat": c["G"]} for c in cellsP1], "stat")
    res["secondary"]["H_per_cell"] = holm_list([{**{k: c[k] for k in ("ctype", "d", "I")}, "stat": c["H"]} for c in cellsP2], "stat")
    ps = pooled(P2s); ps["reading"] = reading(ps["stat"], ps["stat"]["p"]); res["secondary"]["P2_sat_pooled"] = ps
    res["secondary"]["H_sat_per_cell"] = holm_list([{**{k: c[k] for k in ("ctype", "d", "I")}, "stat": c["H_sat"]} for c in cellsP2s], "stat")
    rest = []
    for cls in ("cos2", "mlp8"):
        for d in DS:
            for I in (4.0, 8.0):
                sc = A.clean_ref(cls, "product", d, I)
                for e in (0.1, 0.3):
                    cc = A.get(block="E", cls=cls, variant="product", ctype="independent", eps=e, d=d, I=I, N=4096, updates=2000)
                    rest.append({"cls": cls, "d": d, "I": I, "eps": e, "stat": tstat(list(A.paired(cc, None, sc).values()))})
    res["secondary"]["restricted_delta_eps"] = holm_list(rest, "stat")
    quad = []
    for d in DS:
        for I in (4.0, 8.0):
            sc = A.clean_ref("quad3", "product", d, I); cc = A.get(block="E", cls="quad3", variant="product", ctype="clean", eps=0.0, d=d, I=I, N=4096, updates=2000)
            quad.append({"cls": "quad3", "d": d, "I": I, "stat": tstat(list(A.paired(cc, None, sc).values()))})
    res["secondary"]["quad3_clean_delta"] = holm_list(quad, "stat")
    res["secondary"]["P3_cells"] = cellsP3
    # ---------------- descriptive
    dose, sat, edges, blocks_cd, clean_t, sens = [], [], defaultdict(lambda: defaultdict(int)), [], [], []
    for k, seeds in sorted(cells.items(), key=lambda kv: str(kv[0])):
        c0 = next(iter(seeds.values()))["cell"]; arms = c0["arms"]; label = {f: c0[f] for f in KEYF}; label["match"] = c0.get("match")
        means = {a: float(np.mean([v for v in (e_of(R, a) for R in seeds.values()) if v is not None])) if seeds else None for a in arms}
        wins = defaultdict(int)
        for R in seeds.values():
            ev = {a: e_of(R, a) for a in arms}
            if all(v is not None for v in ev.values()):
                wins[min(ev, key=ev.get)] += 1
        dose.append({**label, "mean_e": means, "argmin_counts": dict(wins), "n_seeds": len(seeds)})
        tr = [R["truth"] for R in seeds.values()]
        dl = [delta(R) for R in seeds.values()]; dl = [x for x in dl if x is not None]
        sc = ref_for(A, c0)
        sat.append({**label, "S": float(np.mean([t["S"] for t in tr])), "sat95": float(np.mean([t["sat95"] for t in tr])),
                    "sat95_P_side": float(np.mean([t["sat95_P_side"] for t in tr])), "sat95_Q_side": float(np.mean([t["sat95_Q_side"] for t in tr])),
                    "rel_delta": (float(np.mean(dl)) / sc) if (dl and sc) else None, "delta": tstat(dl)})
        for R in seeds.values():
            for r in R["rows"]:
                if r.get("selected"):
                    edges[(c0["block"], r["arm"])]["n"] += 1
                    edges[(c0["block"], r["arm"])]["edge_low"] += r.get("lr_edge") == "low"
                    edges[(c0["block"], r["arm"])]["edge_high"] += r.get("lr_edge") == "high"
                    edges[(c0["block"], r["arm"])]["not_converged"] += bool(r["select"].get("not_converged"))
        if c0["block"] in ("C", "D", "Dp"):
            blocks_cd.append({**label, "delta": tstat(dl)})
        if c0["ctype"] in TYPES_A and c0["eps"] > 0 and "vcs" in arms:
            dc = [((sel_row(R, "vcs") or {}).get("readout", {}).get("clean_target", {}).get("posterior_mse_clean"),
                   (sel_row(R, "js") or {}).get("readout", {}).get("clean_target", {}).get("posterior_mse_clean")) for R in seeds.values()]
            dc = [a - b for a, b in dc if a is not None and b is not None]
            clean_t.append({**label, "delta_e": tstat(dl), "delta_e_clean_target": tstat(dc)})
        dj = [delta(R, "selected_commonJ") for R in seeds.values()]
        sens.append({**label, "delta_native_selection": tstat(dl), "delta_commonJ_selection": tstat([x for x in dj if x is not None])})
    res["descriptive"]["gamma_dose"] = dose
    res["descriptive"]["saturation"] = sat
    pts = [(r["sat95"], r["rel_delta"], 1.0 if (r["ctype"] in TYPES_A and r["block"] == "A" and r["eps"] > 0) else 0.0) for r in sat
           if r["block"] in ("A", "B", "B2") and r["cls"] == "mlp256" and r["rel_delta"] is not None]
    if len(pts) >= 4:
        X = np.array([[1.0, s, ind] for s, _, ind in pts]); y = np.array([v for _, v, _ in pts]); beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        res["descriptive"]["saturation_model"] = {"model": "rel_delta ~ 1 + sat95 + contaminated", "coef": {"intercept": float(beta[0]), "sat95": float(beta[1]),
                                                  "contaminated": float(beta[2])}, "n_cells": len(pts), "note": "descriptive (cell means; no inference)"}
    res["descriptive"]["lr_edges_and_convergence"] = {f"{b}:{a}": dict(v) for (b, a), v in sorted(edges.items())}
    res["descriptive"]["blocks_C_D_Dp"] = blocks_cd
    res["descriptive"]["clean_target_vs_contaminated"] = clean_t
    res["descriptive"]["selection_rule_sensitivity"] = sens
    return res


def fmt(s: dict | None) -> str:
    if not s or s.get("mean") is None:
        return "—"
    ci = s.get("ci95"); return f"{s['mean']:+.4f}" + (f" [{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "") + f" (n {s['n']})"


def markdown(res: dict, smoke: bool) -> str:
    L = ["# P156 (SEP-1) aggregate" + (" — TOY / SMOKE cells, not results" if smoke else ""), "",
         "Effects standardised by the clean JS error of the same cell family; Δe = e_vcs − e_js (negative = VCS more accurate).", "",
         "## Primary (Holm over 3)", "", "| test | mean [95 % t] (n seeds) | p | p Holm | cells | reading |", "|---|---|---|---|---|---|"]
    for k, v in res["primary"].items():
        s = v["stat"]; p = "—" if s.get("p") is None else f"{s['p']:.4g}"; ph = "—" if v.get("p_holm") is None else f"{v['p_holm']:.4g}"
        L.append(f"| {k} | {fmt(s)} | {p} | {ph} | {v['n_cells']} | {v['reading']} |")
    L += ["", "Outcome candidates (mechanical rule of protocol §6): " + "; ".join(res["outcome_candidates"]), ""]
    ps = res["secondary"].get("P2_sat_pooled")
    if ps:
        L += [f"Secondary pooled P2-sat (contaminated vs saturation-matched clean, ε 0.3; not in the primary family): {fmt(ps['stat'])} — {ps['reading']}", ""]
    for name, rows in res["secondary"].items():
        if name in ("P3_cells", "P2_sat_pooled"):
            continue
        L += [f"## Secondary: {name} (Holm within the list)", "", "| cell | mean [95 % t] | p Holm | reading |", "|---|---|---|---|"]
        for r in rows:
            lab = ", ".join(f"{k} {r[k]}" for k in ("ctype", "cls", "d", "I", "eps") if k in r)
            ph = "—" if r.get("p_holm") is None else f"{r['p_holm']:.4g}"; L.append(f"| {lab} | {fmt(r['stat'])} | {ph} | {r['reading']} |")
        L.append("")
    L += ["## Descriptive: lr edges and convergence (selected rows)", "", "| block:arm | n | low edge | high edge | not converged |", "|---|---|---|---|---|"]
    for k, v in res["descriptive"]["lr_edges_and_convergence"].items():
        L.append(f"| {k} | {v.get('n', 0)} | {v.get('edge_low', 0)} | {v.get('edge_high', 0)} | {v.get('not_converged', 0)} |")
    if "saturation_model" in res["descriptive"]:
        m = res["descriptive"]["saturation_model"]; L += ["", f"Saturation model ({m['model']}, {m['n_cells']} cells, descriptive): " + ", ".join(f"{k} {v:+.4f}" for k, v in m["coef"].items())]
    L += ["", "Full descriptive tables (γ dose, saturation, blocks C / D / D′, clean-target diagnostic, selection sensitivity) are in the JSON."]
    return "\n".join(L) + "\n"
