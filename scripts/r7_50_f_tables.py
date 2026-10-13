"""Formatting only: R7-50 phase-F report tables from the committed results files.

Reads reports/R7/R7_50_results_F (official test, D1-FULL selectors), reports/R7/R7_50_results_L (official test,
D1-20 10k selectors) and reports/R7/R7_D1_FULL_results (DEV / val, both scales).  Computes nothing beyond seed means
and sign / verdict comparisons of numbers already in those files.
"""
import json
import sys
from pathlib import Path

ALPHA = 0.05
PRIMARY = ["S1-S2", "S1-S3", "S1-SparseVLM"]
POINTS = ["L25", "L50", "L75"]
BUDGETS = [72, 144, 288, 432]
DEV_OF = {"gqa_testdev": "gqa_dev", "chartqa_test": "chartqa_val"}


def key(c, pt):
    return f"{c}@{pt}[budget]" if c == "S1-SparseVLM" else f"{c}@{pt}"


def fmt_c(c):
    s = f"{c['delta']:+.2f} [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}]"
    return s if c["p_holm"] < ALPHA else s + " (n.s.)"


def family_mean(rows, fam, budget):
    vals = [r["quality"] for k, r in rows.items()
            if k.startswith(f"sel:{fam}_") and k.endswith(f":{budget}") and "_dtarget" not in k]
    if len(vals) != 3:
        raise SystemExit(f"expected 3 seeds for {fam}:{budget}, got {len(vals)}")
    return sum(vals) / 3


def lr_of(rows, fam):
    lrs = {k.split("_lr")[1].split("_")[0] for k in rows if k.startswith(f"sel:{fam}_")}
    return ",".join(sorted(lrs))


def main(root):
    root = Path(root)
    F = json.loads((root / "reports/R7/R7_50_results_F/r7_50_results.json").read_text())
    L = json.loads((root / "reports/R7/R7_50_results_L/r7_50_results.json").read_text())
    D = json.loads((root / "reports/R7/R7_D1_FULL_results/d1full_results.json").read_text())
    out = []
    p = out.append
    p("## 1. Primary contrasts (accuracy points; 95 % interval; n.s. = Holm-adjusted p ≥ 0.05)")
    p("| contrast | point | GQA testdev | ChartQA test |")
    p("|---|---|---|---|")
    for c in PRIMARY:
        for pt in POINTS:
            cells = [fmt_c(F["benches"][b]["contrasts"][key(c, pt)]) for b in ["gqa_testdev", "chartqa_test"]]
            p(f"| {c} | {pt} | {cells[0]} | {cells[1]} |")
    p("")
    p("Per-seed Δ (seed 0, 1, 2) of the primary cells:")
    p("")
    p("| contrast | point | GQA testdev | ChartQA test |")
    p("|---|---|---|---|")
    for c in PRIMARY:
        for pt in POINTS:
            cells = [", ".join(f"{x:+.2f}" for x in F["benches"][b]["contrasts"][key(c, pt)]["seed_deltas"])
                     for b in ["gqa_testdev", "chartqa_test"]]
            p(f"| {c} | {pt} | {cells[0]} | {cells[1]} |")
    p("")
    p("## 2. Accuracy at fixed budgets (seed means for S1–S3; tokens kept out of 576)")
    for b in ["gqa_testdev", "chartqa_test"]:
        rows = F["benches"][b]["rows"]
        p("")
        p(f"| **{b}** (n = {F['benches'][b]['n_items']}; full {rows['full']['quality']:.2f}) | 12.5 % | 25 % | 50 % | 75 % |")
        p("|---|---|---|---|---|")
        p("| uniform grid | " + " | ".join(f"{rows[f'static:grid:{k}']['quality']:.2f}" for k in BUDGETS) + " |")
        for fam, name in [("vcs", "S1 VCS"), ("logistic", "S2 logistic"), ("direct", "S3 direct risk")]:
            p(f"| {name} (lr {lr_of(rows, fam)}) | " + " | ".join(f"{family_mean(rows, fam, k):.2f}" for k in BUDGETS) + " |")
        p("| SparseVLM (adapted) | " + " | ".join(f"{rows[f'sparsevlm:{k}:adapted']['quality']:.2f}" for k in BUDGETS) + " |")
    p("")
    p("## 3. Agreement with the D1-FULL DEV / val reading (same full-FIT selectors, same rows)")
    p("| benchmark | contrast | point | official test | D1-FULL DEV / val | same sign | same Holm verdict |")
    p("|---|---|---|---|---|---|---|")
    ns = nv = n = 0
    for c in PRIMARY:
        for pt in POINTS:
            for b in ["gqa_testdev", "chartqa_test"]:
                t = F["benches"][b]["contrasts"][key(c, pt)]
                d = D["benches"][DEV_OF[b]]["full_scale"]["contrasts"][key(c, pt)]
                same_sign = (t["delta"] > 0) == (d["delta"] > 0)
                same_v = (t["p_holm"] < ALPHA) == (d["p_holm"] < ALPHA)
                n += 1; ns += same_sign; nv += same_v
                p(f"| {b} | {c} | {pt} | {t['delta']:+.2f} (p_holm {t['p_holm']:.3f}) | {d['delta']:+.3f} (p_holm {d['p_holm']:.3f})"
                  f" | {'yes' if same_sign else 'no'} | {'yes' if same_v else 'no'} |")
    p("")
    p(f"Same sign in {ns} of {n} cells; same Holm verdict in {nv} of {n} (Holm families differ: 9 per point in D1-FULL with TextVQA, 6 here).")
    p("")
    p("## 4. The two scales on the official test (phase L = 10k-FIT D1-20 selectors; phase F = full-FIT D1-FULL selectors)")
    p("Descriptive; no test of the difference between scales was pre-registered.")
    p("")
    p("| benchmark | contrast | point | 10k FIT (phase L) | full FIT (phase F) | same sign | both detected |")
    p("|---|---|---|---|---|---|---|")
    m = ms = mb = 0
    for b in ["gqa_testdev", "chartqa_test"]:
        for c in PRIMARY:
            for pt in POINTS:
                a = L["benches"][b]["contrasts"][key(c, pt)]
                f = F["benches"][b]["contrasts"][key(c, pt)]
                ss = (a["delta"] > 0) == (f["delta"] > 0)
                bd = ss and a["p_holm"] < ALPHA and f["p_holm"] < ALPHA
                m += 1; ms += ss; mb += bd
                p(f"| {b} | {c} | {pt} | {fmt_c(a)} | {fmt_c(f)} | {'yes' if ss else 'no'} | {'yes' if bd else 'no'} |")
    p("")
    p(f"Same sign in {ms} of {m} cells; same sign and detected at both scales in {mb} of {m}.")
    p("")
    p("## 5. Rows shared with phase L (not regenerated)")
    worst = 0.0
    shared = 0
    for b in ["gqa_testdev", "chartqa_test"]:
        for k, r in F["benches"][b]["rows"].items():
            if k == "full" or k.startswith("static:") or k.startswith("sparsevlm:"):
                worst = max(worst, abs(r["quality"] - L["benches"][b]["rows"][k]["quality"]))
                shared += 1
    p(f"{shared} rows (full, uniform grid, SparseVLM) on the two benchmarks; largest |phase F − phase L| quality difference {worst:.2e}.")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
