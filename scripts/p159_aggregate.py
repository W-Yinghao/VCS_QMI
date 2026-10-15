"""P159 (SEP-3) aggregate (protocol reports/P159_SEP3_NATURAL_PREREG_*_20261015.md).

Unit of inference = seed (draw).  Per cell (source, class, ctype, eps, N) and seed, the selected row of each arm gives the clean-EVAL readouts.
Effects per cell and seed (positive = VCS better):
    dJ_rel   = (J_vcs - J_js) / mean-over-seeds(J_js) of the cell            (the P157 standardisation)
    dA_rel   = (AUROC_vcs - AUROC_js) / mean-over-seeds(1 - AUROC_js)      (error-rate scale; AUROC itself is near 1)
    dECE     = ECE_js - ECE_vcs                                             (absolute; positive = VCS better calibrated)
Primary family (Holm over 3), paired t over the 8 seeds of the per-seed mean over the test's cells:
    Q1  dJ_rel over every eps > 0 cell (both sources, classes, types, eps, N)
    Q2  dJ_rel over the eps = 0 cells
    Q3  dA_rel over every eps > 0 cell
Outcome rule: SUPPORTED if Q1 > 0 (Holm) and Q2's 95 % interval contains 0; FIT-ONLY if Q1 > 0 and Q2 > 0 (Holm); CONTAMINATION-SPECIFIC WITH
CLEAN DEFICIT if Q1 > 0 and Q2 < 0 (Holm); NOT SUPPORTED if Q1 is not > 0.
Secondary: per-cell dJ, dA, dECE (Holm within each list); degradation ratios per rule J(eps) / J(0) on the same seed's clean cell and their
difference (R4-E0 reading) beside the absolute effects; QC (lr edges, not converged); descriptive by factor.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vcs_estim import sep_agg as A  # noqa: E402

KEY = ("source", "cls", "ctype", "eps", "N")


def sel_readout(R: dict, arm: str, which: str) -> dict | None:
    for r in R["rows"]:
        if r["arm"] == arm and r.get(which):
            return r["readout"] if which == "selected" else r["readout_commonJ_selected"]
    return None


def collect(dirpath: Path, which: str):
    C, qc = defaultdict(dict), defaultdict(lambda: defaultdict(int))
    for f in sorted(dirpath.glob("P159_*.json")):
        R = json.loads(f.read_text()); c = R["cell"]; k = tuple(c[x] for x in KEY)
        rj, rv = sel_readout(R, "js", which), sel_readout(R, "vcs", which)
        if rj is None or rv is None:
            continue
        C[k][c["seed"]] = {"js": rj, "vcs": rv}
        for r in R["rows"]:
            if r.get(which):
                q = qc[(c["source"], c["cls"], r["arm"])]; q["n"] += 1; q["top_lr"] += r["lr_edge"] == "high"; q["low_lr"] += r["lr_edge"] == "low"
                q["not_converged"] += bool(r["select"].get("not_converged"))
    return C, qc


def effects(C: dict) -> dict:
    """{cell key: {seed: {dJ_rel, dA_rel, dECE, raw}}} with the cell's JS references (mean over seeds)."""
    out = {}
    for k, seeds in C.items():
        mJ = st.mean(v["js"]["J_eval"] for v in seeds.values()); mA = st.mean(1 - v["js"]["auroc"] for v in seeds.values())
        out[k] = {}
        for s, v in seeds.items():
            dj = (v["vcs"]["J_eval"] - v["js"]["J_eval"]) / mJ if mJ != 0 else None
            da = (v["vcs"]["auroc"] - v["js"]["auroc"]) / mA if mA > 0 else None
            out[k][s] = {"dJ_rel": dj, "dA_rel": da, "dECE": v["js"]["ece"] - v["vcs"]["ece"], "dJ": v["vcs"]["J_eval"] - v["js"]["J_eval"],
                         "J_js": v["js"]["J_eval"], "J_vcs": v["vcs"]["J_eval"], "A_js": v["js"]["auroc"], "A_vcs": v["vcs"]["auroc"],
                         "ece_js": v["js"]["ece"], "ece_vcs": v["vcs"]["ece"]}
    return out


def pooled(E: dict, keys: list, field: str, n_seeds: int = 8) -> dict:
    per_seed = defaultdict(list)
    for k in keys:
        for s, v in E.get(k, {}).items():
            if v[field] is not None:
                per_seed[s].append(v[field])
    complete = len(per_seed) == n_seeds and all(len(x) == len(keys) for x in per_seed.values())
    return {"stat": A.tstat([st.mean(x) for x in per_seed.values()]) if per_seed else A.tstat([]), "n_cells": len(keys), "complete": complete,
            "per_seed": {s: st.mean(x) for s, x in sorted(per_seed.items())}}


def read_pos(s: dict, p_adj) -> str:
    if p_adj is None or s.get("mean") is None:
        return "not computable"
    if p_adj < 0.05:
        return "> 0 (VCS better)" if s["mean"] > 0 else "< 0 (JS better)"
    return "n.s."


def outcome(q1: dict, q2: dict) -> str:
    sig = lambda q: q["p_holm"] is not None and q["p_holm"] < 0.05
    if not (sig(q1) and q1["stat"]["mean"] > 0):
        return "NOT SUPPORTED"
    ci = q2["stat"].get("ci95")
    if not sig(q2) and ci is not None and ci[0] <= 0 <= ci[1]:
        return "SUPPORTED"
    if sig(q2) and q2["stat"]["mean"] > 0:
        return "FIT-ONLY"
    if sig(q2) and q2["stat"]["mean"] < 0:
        return "CONTAMINATION-SPECIFIC WITH CLEAN DEFICIT"
    return "SUPPORTED"                                                    # Q2 n.s. but interval not computable only with < 2 seeds; never reached at n = 8


def degradation(E: dict) -> list:
    """Per contaminated cell: per rule the ratio J(eps) / J(0) on the same seed's clean cell (same source, class, N), mean and 95 % t over seeds,
    and the paired difference of ratios (VCS - JS; positive = VCS degrades less), beside the absolute dJ_rel."""
    rows = []
    for k, seeds in E.items():
        if k[3] == 0.0:
            continue
        k0 = (k[0], k[1], "clean", 0.0, k[4]); base = E.get(k0, {})
        rj, rv, dd, dj = [], [], [], []
        for s, v in seeds.items():
            b = base.get(s)
            if b is None or b["J_js"] == 0 or b["J_vcs"] == 0:
                continue
            a, c = v["J_js"] / b["J_js"], v["J_vcs"] / b["J_vcs"]; rj.append(a); rv.append(c); dd.append(c - a); dj.append(v["dJ_rel"])
        rows.append({"source": k[0], "cls": k[1], "ctype": k[2], "eps": k[3], "N": k[4], "n_seeds": len(dd), "ratio_js": A.tstat(rj), "ratio_vcs": A.tstat(rv),
                     "ratio_diff_vcs_minus_js": A.tstat(dd), "dJ_rel": A.tstat(dj)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--out", required=True); ap.add_argument("--seeds", type=int, default=8); a = ap.parse_args()
    res = {}
    for which in ("selected", "selected_commonJ"):
        C, qc = collect(Path(a.dir), which); E = effects(C); keys = sorted(E)
        tests = {"Q1_dJ_eps_gt_0": ([k for k in keys if k[3] > 0], "dJ_rel"), "Q2_dJ_eps_0": ([k for k in keys if k[3] == 0], "dJ_rel"), "Q3_dA_eps_gt_0": ([k for k in keys if k[3] > 0], "dA_rel")}
        prim = {t: pooled(E, ks, f, a.seeds) for t, (ks, f) in tests.items()}; ph = A.holm([prim[t]["stat"]["p"] for t in prim])
        for t, h in zip(prim, ph):
            prim[t]["p_holm"] = h; prim[t]["reading"] = read_pos(prim[t]["stat"], h)
        cells = {}
        for field, lab in (("dJ_rel", "dJ"), ("dA_rel", "dA"), ("dECE", "dECE")):
            L = []
            for k in keys:
                vals = [v[field] for v in E[k].values() if v[field] is not None]
                L.append({"source": k[0], "cls": k[1], "ctype": k[2], "eps": k[3], "N": k[4], "n_seeds": len(vals), "stat": A.tstat(vals),
                          "js": st.mean(v[{"dJ_rel": "J_js", "dA_rel": "A_js", "dECE": "ece_js"}[field]] for v in E[k].values()),
                          "vcs": st.mean(v[{"dJ_rel": "J_vcs", "dA_rel": "A_vcs", "dECE": "ece_vcs"}[field]] for v in E[k].values())})
            hp = A.holm([c["stat"]["p"] for c in L])
            for c, h in zip(L, hp):
                c["p_holm"] = h; c["reading"] = read_pos(c["stat"], h)
            cells[lab] = L
        q1, q2 = prim["Q1_dJ_eps_gt_0"], prim["Q2_dJ_eps_0"]
        by = defaultdict(list)
        for k in keys:
            if k[3] > 0:
                for v in E[k].values():
                    if v["dJ_rel"] is not None:
                        for f, val in zip(KEY, k):
                            by[f"{f}={val}"].append(v["dJ_rel"])
        res[which] = {"primary": prim, "outcome": outcome(q1, q2), "cells": cells, "degradation": degradation(E),
                      "by_factor_dJ_rel_eps_gt_0": {k: st.mean(v) for k, v in sorted(by.items())},
                      "qc": {f"{s}:{c}:{arm}": dict(v) for (s, c, arm), v in sorted(qc.items())}, "n_cells": len(keys),
                      "missing_cells": [list(k) for k in keys if len(E[k]) < a.seeds]}
    Path(a.out + ".json").write_text(json.dumps(res, indent=1))
    L = ["# P159 (SEP-3) aggregate — clean-EVAL readouts on frozen natural features", "",
         "Effects (positive = VCS better): dJ_rel = (J_VCS − J_JS) / mean J_JS of the cell; dA_rel = (AUROC_VCS − AUROC_JS) / mean (1 − AUROC_JS); paired t over seeds.", ""]
    for which, r in res.items():
        L += [f"## Selection: {which} — outcome: **{r['outcome']}** ({r['n_cells']} cells; missing {len(r['missing_cells'])})", "",
              "| test | mean [95 % t] | p | p Holm | cells | complete | reading |", "|---|---|---|---|---|---|---|"]
        for t, q in r["primary"].items():
            s = q["stat"]; ci = s.get("ci95") or [float("nan")] * 2; p = s.get("p"); ph_ = q["p_holm"]
            L.append(f"| {t} | {s['mean'] if s['mean'] is None else f'{s['mean']:+.4f}'} [{ci[0]:+.4f}, {ci[1]:+.4f}] | {p if p is None else f'{p:.3g}'} | {ph_ if ph_ is None else f'{ph_:.3g}'} | {q['n_cells']} | {q['complete']} | {q['reading']} |")
        L.append("")
        if which == "selected":
            for lab, title in (("dJ", "Cells: dJ_rel (Holm over cells)"), ("dA", "Cells: dA_rel (Holm over cells)"), ("dECE", "Cells: ECE_JS − ECE_VCS (Holm over cells)")):
                L += [f"### {title}", "", "| source | class | type | ε | N | JS | VCS | effect [95 % t] | p Holm | reading |", "|---|---|---|---|---|---|---|---|---|---|"]
                for c in r["cells"][lab]:
                    s = c["stat"]; ci = s.get("ci95") or [float("nan")] * 2; ph_ = c["p_holm"]
                    L.append(f"| {c['source']} | {c['cls']} | {c['ctype']} | {c['eps']:g} | {c['N']} | {c['js']:.4f} | {c['vcs']:.4f} | {s['mean'] if s['mean'] is None else f'{s['mean']:+.4f}'} [{ci[0]:+.4f}, {ci[1]:+.4f}] | {ph_ if ph_ is None else f'{ph_:.3g}'} | {c['reading']} |")
                L.append("")
            L += ["### Degradation ratios J(ε) / J(0) per rule (same seed's clean cell) beside the absolute effect (R4-E0 reading: both, never one alone)", "",
                  "| source | class | type | ε | N | ratio JS | ratio VCS | ratio VCS − JS [95 % t] | dJ_rel [95 % t] |", "|---|---|---|---|---|---|---|---|---|"]
            for g in r["degradation"]:
                rd, dj = g["ratio_diff_vcs_minus_js"], g["dJ_rel"]; ci = rd.get("ci95") or [float("nan")] * 2; cj = dj.get("ci95") or [float("nan")] * 2
                L.append(f"| {g['source']} | {g['cls']} | {g['ctype']} | {g['eps']:g} | {g['N']} | {g['ratio_js']['mean']:.4f} | {g['ratio_vcs']['mean']:.4f} | {rd['mean']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] | {dj['mean']:+.4f} [{cj[0]:+.4f}, {cj[1]:+.4f}] |")
            L += ["", "### dJ_rel at ε > 0 by factor (descriptive)", ""] + [f"- {k}: {v:+.4f}" for k, v in r["by_factor_dJ_rel_eps_gt_0"].items()]
            L += ["", "### QC (selected rows): top-lr (1.0) / low-lr / not converged", ""] + [f"- {k}: {v}" for k, v in r["qc"].items()] + [""]
    Path(a.out + ".md").write_text("\n".join(L) + "\n"); print("\n".join(L[:16])); return 0


if __name__ == "__main__":
    raise SystemExit(main())
