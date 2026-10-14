"""V7X-11 aggregation (formatting + Monte-Carlo summaries of the raw anchor rows; no selection).  Per anchor cell and N: each method's native estimate
against its OWN truth (bias, RMSE, MC SE over replicates), the common posterior MSE / J-gap, edge counts; tests: rejection rate with the exact
binomial (Clopper-Pearson) 95 % interval.  Duplicate rows of a (cell, N, rep) key keep the last complete one (resubmitted cells append).
    python -I scripts/v7x/x11_aggregate.py --dir outputs/V7/x11 --out x11_summary.json --md x11_summary.md"""
import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import beta

NATIVE = {"K-CS-NATIVE": "cs_lebesgue_sqratio", "K-RULSIF": "PE_half", "K-ULSIF": "pearson_chi2_half", "N-DV": "KL_PQ", "N-NWJ": "KL_PQ"}


def ms(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return (float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan"), len(v)) if len(v) else (float("nan"), float("nan"), 0)


def cp(k, n):
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0; hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--out", required=True); ap.add_argument("--md", required=True); a = ap.parse_args()
    truth, rows = {}, {}
    for f in sorted(glob.glob(str(Path(a.dir) / "est_*.jsonl"))):
        for l in open(f):
            r = json.loads(l); key = (r["anchor"], json.dumps(r["cell"], sort_keys=True))
            if "truth" in r:
                truth[key] = r["truth"]
            else:
                rows[key + (r["N"], r["rep"])] = r
    groups = defaultdict(list)
    for (an, cell, N, rep), r in rows.items():
        groups[(an, cell, N)].append(r)
    summary = []; md = ["# V7X-11 anchors — summary (Monte-Carlo over data replicates; each native estimate against its own truth)", ""]
    for (an, cell, N), rs in sorted(groups.items()):
        tr = dict(truth[(an, cell)]); tr["pearson_chi2_half"] = tr["pearson_chi2"] / 2; ent = {"anchor": an, "cell": json.loads(cell), "N": N, "reps": len(rs), "truth": tr, "methods": {}}
        cs = defaultdict(list)
        for r in rs:
            for v in r["K-CS-NATIVE"]:
                cs[v["variant"]].append(v["estimate"])
        ent["methods"]["K-CS-NATIVE"] = {var: dict(zip(("mean", "se", "n"), ms(x)), bias=ms(x)[0] - tr["cs_lebesgue_sqratio"],
                                                   rmse=float(np.sqrt(np.mean((np.asarray(x) - tr["cs_lebesgue_sqratio"]) ** 2)))) for var, x in cs.items()}
        for m in ("K-KDE-S", "K-NYSTROM", "N-VCS", "N-BCE", "N-DV", "N-NWJ"):
            ok = [r[m] for r in rs if m in r and r[m].get("status") != "failed"]
            e = {"posterior_mse": dict(zip(("mean", "se", "n"), ms([o["posterior_mse"] for o in ok]))), "J_gap": dict(zip(("mean", "se", "n"), ms([tr["mixture_S"] - o["J_eval"] for o in ok]))),
                 "failed": sum(1 for r in rs if m in r and r[m].get("status") == "failed")}
            if m in ("N-DV", "N-NWJ"):
                x = [o["native_KL"] for o in ok]; e["native_KL"] = dict(zip(("mean", "se", "n"), ms(x)), bias=ms(x)[0] - tr["KL_PQ"])
            if m in ("K-KDE-S", "K-NYSTROM"):
                e["at_edge"] = sum(1 for o in ok if o.get("edge") or o.get("at_edge_after_extension"))
            ent["methods"][m] = e
        for m, tgt in (("K-RULSIF", "PE_half"), ("K-ULSIF", "pearson_chi2_half")):
            for tag in ("native_cv", "common"):
                ok = [r[m][tag] for r in rs]; x = [o["native_estimate"] for o in ok]
                ent["methods"][f"{m}:{tag}"] = {"native": dict(zip(("mean", "se", "n"), ms(x)), bias=ms(x)[0] - tr[tgt]),
                                                "posterior_mse": dict(zip(("mean", "se", "n"), ms([o["posterior_mse"] for o in ok]))), "at_edge": sum(o["edge"] for o in ok),
                                                "clip_fraction": ms([o["clip_fraction"] for o in ok])[0]}
        summary.append(ent)
        md.append(f"## {an} {json.loads(cell)} N {N} ({len(rs)} replicates) — truths: CS {tr['cs_lebesgue_sqratio']:.4g}, KL {tr['KL_PQ']:.4g}, S {tr['mixture_S']:.4g}")
        md.append("| method | native (mean +- MC SE) | own truth | posterior MSE | at edge |"); md.append("|---|---|---|---|---|")
        for var, v in ent["methods"]["K-CS-NATIVE"].items():
            md.append(f"| K-CS-NATIVE {var} | {v['mean']:.4g} +- {v['se']:.2g} | {tr['cs_lebesgue_sqratio']:.4g} | — | — |")
        for m, v in ent["methods"].items():
            if m == "K-CS-NATIVE":
                continue
            nat = v.get("native") or v.get("native_KL"); own = {"K-RULSIF": tr["PE_half"], "K-ULSIF": tr["pearson_chi2_half"], "N-DV": tr["KL_PQ"], "N-NWJ": tr["KL_PQ"]}.get(m.split(":")[0])
            nat_s = f"{nat['mean']:.4g} +- {nat['se']:.2g}" if nat else "—"; own_s = f"{own:.4g}" if own is not None else "—"
            pm = v["posterior_mse"]; md.append(f"| {m} | {nat_s} | {own_s} | {pm['mean']:.4g} +- {pm['se']:.2g} | {v.get('at_edge', '—')} |")
        md.append("")
    tests = []
    for f in sorted(glob.glob(str(Path(a.dir) / "tests_*.jsonl"))):
        for l in open(f):
            r = json.loads(l); t = {"anchor": r["anchor"], "cell": r["cell"], "null": r["null"], "reps": r["reps"]}
            for k in ("T-MMD", "T-HSIC"):
                n_rej = r.get(f"{k}_rejections")
                if n_rej is not None:
                    t[k] = {"rate": n_rej / r["reps"], "ci95": cp(n_rej, r["reps"])}
            tests.append(t)
    if tests:
        md += ["## Tests (alpha 0.05; Clopper-Pearson 95 % interval over replicates)", "| anchor | cell | null | MMD rejection | HSIC rejection |", "|---|---|---|---|---|"]
        for t in tests:
            fmt = lambda k: f"{t[k]['rate']:.3f} [{t[k]['ci95'][0]:.3f}, {t[k]['ci95'][1]:.3f}]" if k in t else "—"
            md.append(f"| {t['anchor']} | {t['cell']} | {t['null']} | {fmt('T-MMD')} | {fmt('T-HSIC')} |")
    Path(a.out).write_text(json.dumps({"estimation": summary, "tests": tests}, indent=1)); Path(a.md).write_text("\n".join(md) + "\n"); print(f"{len(summary)} groups, {len(tests)} test rows")


if __name__ == "__main__":
    main()
