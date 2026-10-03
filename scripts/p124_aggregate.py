"""P124 aggregate → <out>.{md,json}: per family (mean ± sd over encoder seeds) coarse / fine / conditional accuracy for every readout; seed-paired
contrasts (A-P3 − recipe VCS, A-P3 − SimCLR, G2 − recipe VCS) with 95 % t intervals and the frozen labels (close / clear / inconclusive); the pattern
statement on the PRIMARY readout; the 100-way head decomposition; the masked-head conditional readout; the anchor check against the stored
training-evaluation linear accuracy.
    python scripts/p124_aggregate.py --in reports/P124 --out reports/P124_v6_granularity_results
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

OUTPUT_ROOT = Path("/home/infres/yinwang/CS_QMI/outputs")
NAME = {"P91_c100_vcs_a5_views4_800ep": "recipe VCS", "P111_G2_c100_views4_800ep": "G2", "P107_AP3_c100_views4_800ep": "A-P3",
        "P91_c100_simclr_views4_800ep": "SimCLR"}
ORDER = ["A-P3", "G2", "recipe VCS", "SimCLR"]
CONTRASTS = [("A-P3", "recipe VCS"), ("A-P3", "SimCLR"), ("G2", "recipe VCS")]
READOUTS = ("recipe_raw", "raw_unstd", "raw_std", "l2_std", "knn")
PRIMARY = "recipe_raw"
CLOSE = 0.3  # points (as P114 / P120)


def fam_seed(run):
    m = re.match(r"(.*)_seed(\d+)$", run); return NAME.get(m.group(1), m.group(1)), int(m.group(2))


def value(rec, task, readout, which="macro_pct"):
    t = rec["tasks"][task][readout]
    return t[which] if task == "conditional" else t["acc_pct"]


def label(mean, lo, hi):
    if abs(mean) < CLOSE:
        return "close"
    if lo > 0 or hi < 0:
        return "clear +" if mean > 0 else "clear −"
    return "inconclusive"


def paired(a: dict, b: dict):
    seeds = sorted(set(a) & set(b)); d = np.array([a[s] - b[s] for s in seeds], float)
    if len(d) < 2:
        return None
    m, sd = float(d.mean()), float(d.std(ddof=1)); h = float(stats.t.ppf(0.975, len(d) - 1) * sd / np.sqrt(len(d)))
    return {"seeds": seeds, "per_seed": d.tolist(), "mean": m, "sd": sd, "ci95": [m - h, m + h], "label": label(m, m - h, m + h)}


def pattern(lc, lk):
    """Frozen pattern rule on the primary readout: lc = label of Δcoarse, lk = label of Δconditional (macro)."""
    if lc == "clear +" and lk in ("close", "clear −"):
        return "coarse-specific gain (coarse up, within-coarse not)"
    if lk == "clear +" and lc in ("close", "clear −"):
        return "within-coarse gain (within-coarse up, coarse not)"
    if lc == "clear +" and lk == "clear +":
        return "both up"
    if lc == "close" and lk == "close":
        return "neither (both close)"
    return "no pattern (inconclusive at 3 seeds)"


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="inp", default="reports/P124"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    recs = {}
    for f in sorted(glob.glob(f"{a.inp}/gran_*.json")):
        r = json.load(open(f)); fam, s = fam_seed(r["run"]); recs.setdefault(fam, {})[s] = r
    fams = [f for f in ORDER if f in recs]
    res = {"families": {f: sorted(recs[f]) for f in fams}, "tables": {}, "contrasts": {}, "patterns": {}, "anchor": {}, "decomposition": {}, "masked": {}}
    L = ["# P124 — v6 V6-GRANULARITY results (frozen CIFAR-100 SSL encoders; selection split; official test never opened)", "",
         "Readouts identical for every method; PRIMARY = `recipe_raw` (the original frozen-h linear readout).  Conditional = coarse label given at test",
         "time, 5-way within the coarse group (dedicated probe per group) — never an unconditional 100-way accuracy.  mean ± sd over encoder seeds.", ""]
    # anchor
    L += ["## Anchor: fine 100-way `recipe_raw` vs the stored training-evaluation linear accuracy (must agree within GPU noise, |Δ| ≤ 0.3)", "",
          "| run | stored | recomputed | Δ |", "|---|---|---|---|"]
    for f in fams:
        for s, r in sorted(recs[f].items()):
            ev = OUTPUT_ROOT / r["run"] / "evaluations" / "evaluation_epoch_800.json"
            if ev.is_file() and not r.get("smoke"):
                st = json.load(open(ev))["linear_val_top1_pct"]; rc = value(r, "fine", PRIMARY); d = rc - st
                res["anchor"][r["run"]] = {"stored": st, "recomputed": rc, "delta": d, "ok": abs(d) <= 0.3}
                L.append(f"| {r['run']} | {st:.2f} | {rc:.2f} | {d:+.2f}{'' if abs(d) <= 0.3 else ' **FLAG**'} |")
    L.append("")
    # family tables
    for task, which in (("coarse", None), ("fine", None), ("conditional", "macro_pct"), ("conditional", "overall_pct")):
        key = task if which is None else f"{task}_{which.split('_')[0]}"
        L += [f"## {key} accuracy (%)", "", "| readout | " + " | ".join(fams) + " |", "|---|" + "---|" * len(fams)]
        for ro in READOUTS:
            row = []
            for f in fams:
                v = np.array([value(recs[f][s], task, ro, which or "macro_pct") for s in sorted(recs[f]) if ro in recs[f][s]["tasks"][task]])
                res["tables"].setdefault(key, {}).setdefault(ro, {})[f] = v.tolist()
                row.append(f"{v.mean():.2f} ± {v.std(ddof=1):.2f}" if len(v) > 1 else (f"{v.mean():.2f}" if len(v) else "—"))
            L.append(f"| {ro}{' (primary)' if ro == PRIMARY else ''} | " + " | ".join(row) + " |")
        L.append("")
    # contrasts
    L += ["## Seed-paired contrasts (95 % t interval, df = n − 1; labels: close |Δ| < 0.3 / clear (|Δ| ≥ 0.3 and interval excludes 0) / inconclusive)", "",
          "| contrast | readout | Δ coarse | Δ fine | Δ conditional macro | Δ conditional overall |", "|---|---|---|---|---|---|"]
    for x, y in CONTRASTS:
        if x not in recs or y not in recs:
            continue
        for ro in READOUTS:
            cells = []
            for task, which in (("coarse", None), ("fine", None), ("conditional", "macro_pct"), ("conditional", "overall_pct")):
                p = paired({s: value(recs[x][s], task, ro, which or "macro_pct") for s in recs[x]}, {s: value(recs[y][s], task, ro, which or "macro_pct") for s in recs[y]})
                k = task if which is None else f"{task}_{which.split('_')[0]}"
                res["contrasts"].setdefault(f"{x} - {y}", {}).setdefault(ro, {})[k] = p
                cells.append("—" if p is None else f"{p['mean']:+.2f} [{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}] {p['label']}")
            L.append(f"| {x} − {y} | {ro}{' (primary)' if ro == PRIMARY else ''} | " + " | ".join(cells) + " |")
        c = res["contrasts"].get(f"{x} - {y}", {}).get(PRIMARY, {})
        if c.get("coarse") and c.get("conditional_macro"):
            res["patterns"][f"{x} - {y}"] = pattern(c["coarse"]["label"], c["conditional_macro"]["label"])
    L += ["", "## Pattern statement (frozen rule, PRIMARY readout only; Δcoarse vs Δconditional-macro labels)", ""]
    L += [f"- **{k}:** {v}" for k, v in res["patterns"].items()]
    # decomposition + masked
    L += ["", "## 100-way recipe head: implied-coarse accuracy × fine accuracy given implied coarse correct (= fine accuracy); masked-head conditional", "",
          "| family | implied coarse | fine given coarse correct | fine | masked-head conditional macro | masked-head overall |", "|---|---|---|---|---|---|"]
    for f in fams:
        dd = [recs[f][s].get("fine_head_decomposition") for s in sorted(recs[f])]; mm = [recs[f][s].get("conditional_masked_fine_head") for s in sorted(recs[f])]
        if all(dd) and all(mm):
            g = lambda xs, k: np.array([x[k] for x in xs])  # noqa: E731
            ms = lambda v: f"{v.mean():.2f} ± {v.std(ddof=1):.2f}" if len(v) > 1 else f"{v.mean():.2f}"  # noqa: E731
            res["decomposition"][f] = {k: g(dd, k).tolist() for k in dd[0]}; res["masked"][f] = {k: g(mm, k).tolist() for k in ("macro_pct", "overall_pct")}
            L.append(f"| {f} | {ms(g(dd, 'implied_coarse_acc_pct'))} | {ms(g(dd, 'fine_given_coarse_correct_pct'))} | {ms(g(dd, 'fine_acc_pct'))} | "
                     f"{ms(g(mm, 'macro_pct'))} | {ms(g(mm, 'overall_pct'))} |")
    # chosen lr
    L += ["", "## Chosen probe (lr, wd) per family and readout (selected readouts; grid lr 0.03 / 0.1 / 0.3 × wd 0 / 5e-4)", ""]
    for f in fams:
        for task in ("coarse", "fine"):
            ch = {ro: [str(recs[f][s]["tasks"][task][ro].get("chosen")) for s in sorted(recs[f])] for ro in ("raw_unstd", "raw_std", "l2_std") if ro in recs[f][min(recs[f])]["tasks"][task]}
            L.append(f"- {f} / {task}: " + "; ".join(f"{ro}: {', '.join(v)}" for ro, v in ch.items()))
    Path(a.out + ".json").write_text(json.dumps(res, indent=1, default=float)); Path(a.out + ".md").write_text("\n".join(L) + "\n")
    print(f"-> {a.out}.md / .json ({sum(len(v) for v in recs.values())} encoders)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
