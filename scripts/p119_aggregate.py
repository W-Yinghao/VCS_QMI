"""P119 aggregate → reports/P119_v5_probe_results.{md,json}.
  (1) label efficiency paired by encoder seed: P110 v1 draws 0–2 (primary, re-summary of reported data) and with P119 extra draws 3–5 (secondary);
  (2) transfer readouts (raw_unstd anchor, raw_std, l2_std, lognorm_std, l2_lognorm_std, kNN) per family, paired by seed vs SimCLR and vs recipe VCS;
  (3) float16 vs float32 check with the frozen rule ("matters" if any |Δacc| > 0.3 points or a family-mean ordering changes).
    python scripts/p119_aggregate.py [--p110 reports/P110] [--in reports/P119] [--out reports/P119_v5_probe_results]
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vcs_vtask.probe5 import READOUTS, paired_by_seed  # noqa: E402

NAME = {"P35_vcs_a5_views4_800ep": "recipe VCS", "P41_simclr_views4_800ep": "SimCLR", "P104_G2_views4_800ep": "G2", "P104_U2_views4_800ep": "U2",
        "P107_AP3_views4_800ep": "A-P3"}
ORDER = ["A-P3", "G2", "U2", "recipe VCS", "SimCLR"]


def fam_seed(run: str):
    m = re.match(r"(.*)_seed(\d+)$", run); return NAME.get(m.group(1), m.group(1)), int(m.group(2))


def ms(v):
    v = np.asarray(v, float); return f"{v.mean():.2f} ± {v.std(ddof=1):.2f}" if len(v) > 1 else (f"{v.mean():.2f}" if len(v) else "—")


def fmt_pair(p):
    return f"{p['mean']:+.2f} [{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}] (per seed {', '.join(f'{x:+.2f}' for x in p['per_seed'])}; draw sd {p['mean_within_seed_draw_sd']:.2f})"


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--p110", default="reports/P110"); ap.add_argument("--in", dest="inp", default="reports/P119")
    ap.add_argument("--out", default="reports/P119_v5_probe_results"); a = ap.parse_args()
    L = ["# P119 — v5 NEXT-V-PROBE results (frozen encoders; selection split; official test closed)", ""]; J = {}
    # ---------------------------------------------------------------- (1) label efficiency, paired by encoder seed
    t_p110 = defaultdict(lambda: defaultdict(lambda: defaultdict(list))); t_all = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for f in sorted(glob.glob(f"{a.p110}/v1_*.json")):
        d = json.load(open(f)); fam, s = fam_seed(d["run"])
        for row in d["datasets"]["cifar10"]["rows"]:
            t_p110[fam][s][row["fraction"]].append(row["acc_selected_pct"]); t_all[fam][s][row["fraction"]].append(row["acc_selected_pct"])
    for f in sorted(glob.glob(f"{a.inp}/c10extra_*.json")):
        d = json.load(open(f))
        if d.get("smoke"):
            continue
        fam, s = fam_seed(d["run"])
        for row in d["rows"]:
            t_all[fam][s][row["fraction"]].append(row["acc_selected_pct"])
    fams = [f for f in ORDER if f in t_p110]
    if fams:
        L += ["## 1. CIFAR-10 label efficiency, paired by encoder seed (FIT-selected probe; draws averaged within a seed; draws are not seeds)", ""]
        for label, tab in (("primary — P110 draws 0–2 (re-summary of reported data)", t_p110), ("secondary — with P119 extra draws 3–5", t_all)):
            if label.startswith("secondary") and not glob.glob(f"{a.inp}/c10extra_*.json"):
                continue
            L += [f"### {label}", "", "| family | 1 % | 10 % | 100 % |", "|---|---|---|---|"]
            for fam in fams:
                L.append(f"| {fam} | " + " | ".join(ms([np.mean(tab[fam][s][fr]) for s in sorted(tab[fam]) if fr in tab[fam][s]]) for fr in (0.01, 0.1, 1.0)) + " |")
            pr = {**paired_by_seed(tab, "SimCLR", fams), **paired_by_seed({k: tab[k] for k in ("A-P3", "G2") if k in tab}, "G2", ["A-P3", "G2"])}
            L += ["", "| paired difference (by encoder seed) | mean [95 % t CI over seeds] |", "|---|---|"] + [f"| {k} | {fmt_pair(v)} |" for k, v in pr.items()] + [""]
            J[f"label_efficiency::{label.split(' ')[0]}"] = pr
    # ---------------------------------------------------------------- (2) transfer readouts
    tr = [json.load(open(f)) for f in sorted(glob.glob(f"{a.inp}/transfer_*.json"))]
    tr = [d for d in tr if not d.get("smoke")]
    if tr:
        tabs = {k: defaultdict(lambda: defaultdict(lambda: defaultdict(list))) for k in list(READOUTS) + ["knn"]}
        for d in tr:
            fam, s = fam_seed(d["run"])
            for row in d["rows"]:
                tabs["knn"][fam][s][row["fraction"]].append(row["knn_pct"])
                for k in READOUTS:
                    tabs[k][fam][s][row["fraction"]].append(row["readouts"][k]["acc_pct"])
        fams_t = [f for f in ORDER if f in tabs["knn"]]
        L += ["## 2. CIFAR-10 → CIFAR-100 frozen transfer, identical readouts for every method (mean ± sd over encoder seeds; draws averaged within seed)", "",
              "Norm readouts (`lognorm_std`, `l2_lognorm_std`) are diagnostics and do not replace the standard table (`raw_unstd` = the P110 readout).", ""]
        for frac in (1.0, 0.1):
            L += [f"### {frac * 100:g} % labels", "", "| readout | " + " | ".join(fams_t) + " |", "|---|" + "---|" * len(fams_t)]
            for k in list(READOUTS) + ["knn"]:
                L.append(f"| {k} | " + " | ".join(ms([np.mean(tabs[k][f][s][frac]) for s in sorted(tabs[k][f]) if frac in tabs[k][f][s]]) for f in fams_t) + " |")
            L.append("")
        pj = {}
        L += ["### paired differences by encoder seed (100 % labels)", "", "| readout | contrast | mean [95 % t CI] |", "|---|---|---|"]
        for k in list(READOUTS) + ["knn"]:
            for ref in ("SimCLR", "recipe VCS"):
                pr = paired_by_seed({f: {s: {1.0: v[1.0]} for s, v in tabs[k][f].items() if 1.0 in v} for f in fams_t}, ref, fams_t)
                for c, v in pr.items():
                    L.append(f"| {k} | {c.replace(' @ 1.0', '')} | {v['mean']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}] |"); pj[f"{k}::{c}"] = v
        J["transfer"] = {"tables": {k: {f: {str(s): dict(v) for s, v in tabs[k][f].items()} for f in fams_t} for k in tabs}, "paired": pj}
        ch = defaultdict(lambda: defaultdict(int))
        for d in tr:
            fam, _ = fam_seed(d["run"])
            for row in d["rows"]:
                for k in READOUTS:
                    c = row["readouts"][k]["chosen"]; ch[(fam, k)][f"lr{c['lr']}"] += 1
        L += ["", "### chosen probe lr (grid 0.03 / 0.1 / 0.3; an edge choice means the grid bound binds)", "", "| family | " + " | ".join(READOUTS) + " |", "|---|" + "---|" * len(READOUTS)]
        for f in fams_t:
            L.append(f"| {f} | " + " | ".join(", ".join(f"{a_}×{n}" for a_, n in sorted(ch[(f, k)].items())) for k in READOUTS) + " |")
        L.append("")
    # ---------------------------------------------------------------- (3) float16 vs float32
    fp = [json.load(open(f)) for f in sorted(glob.glob(f"{a.inp}/fpcheck_*.json"))]
    if fp:
        L += ["## 3. float16 cache vs float32 re-extraction (small subset, same device)", "",
              "| encoder | dataset | h rel. Frobenius diff | h max abs diff | kNN fp16 / fp32 | raw_unstd fp16 / fp32 | raw_std fp16 / fp32 | max abs Δacc |",
              "|---|---|---|---|---|---|---|---|"]
        worst = 0.0; order_change = []
        for d in fp:
            for ds, r in d["datasets"].items():
                a16, a32 = r["readouts"]["fp16"], r["readouts"]["fp32"]; worst = max(worst, r["max_abs_acc_diff"])
                L.append(f"| {d['run']} | {ds} | {r['h_rel_frob_diff']:.2e} | {r['h_max_abs_diff']:.2e} | {a16['knn_pct']:.2f} / {a32['knn_pct']:.2f} | "
                         f"{a16['raw_unstd']:.2f} / {a32['raw_unstd']:.2f} | {a16['raw_std']:.2f} / {a32['raw_std']:.2f} | {r['max_abs_acc_diff']:.2f} |")
        for ds in ("cifar100", "cifar10"):
            for k in ("knn_pct", "raw_unstd", "raw_std"):
                means = {}
                for prec in ("fp16", "fp32"):
                    by = defaultdict(list)
                    for d in fp:
                        if ds in d["datasets"]:
                            by[fam_seed(d["run"])[0]].append(d["datasets"][ds]["readouts"][prec][k])
                    means[prec] = [f for f, _ in sorted(by.items(), key=lambda x: -np.mean(x[1]))]
                if means["fp16"] != means["fp32"]:
                    order_change.append(f"{ds}/{k}: {means['fp16']} vs {means['fp32']}")
        verdict = "MATTERS" if (worst > 0.3 or order_change) else "does not matter"
        L += ["", f"**Frozen rule:** max |Δacc| = {worst:.2f} (threshold 0.3); family-order changes: {order_change or 'none'} → float16 cache **{verdict}**.", ""]
        J["fpcheck"] = {"max_abs_acc_diff": worst, "order_changes": order_change, "verdict": verdict}
    Path(a.out + ".md").write_text("\n".join(L) + "\n"); Path(a.out + ".json").write_text(json.dumps(J, indent=1, default=float))
    print(f"-> {a.out}.md / .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
