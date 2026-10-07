"""P150 aggregate (frozen reading, prereg §4): merge the two cell jobs (outputs/P150_c0/c0_<cell>.json) and summarise per site x critic over
the 20 draws: exact-Q common J (mean, 95 % t interval over draws, fraction of draws whose paired-bootstrap CI lies above 0), sampled-Q J, S_plugin,
own-statistic rejection rate (alpha 0.05, shared permutations), nested refit spread and step-0 fraction, and the paired exact-Q differences between
critics (mean over draws, fraction of draws whose CI excludes 0).
Readings: (a) every critic's J not clearly positive at h while the colour rejection rate at h stays well above the null cell's -> detection only;
(b) audit-class J clearly positive at h and nested-class J not -> the P144 reading was function-class-limited; (c) J clearly positive at h and not at O
-> h -> O loses the attribute dependence quantitatively.  "Clearly positive" = the t interval over the 20 draws lies above 0.
    python scripts/p150_aggregate.py [--out reports/P150_results.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

O = Path("/home/infres/yinwang/CS_QMI/outputs/P150_c0")
CELLS = ("colour_s0.1", "null")
SITES = ("h", "O")
CRITICS = ("AUD-VCS", "AUD-JS", "NEST-VCS", "NEST-JS")


def tci(x) -> dict:
    x = np.asarray(x, float); m = float(x.mean()); h = float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))
    return {"mean": m, "ci95": [m - h, m + h], "sd_over_draws": float(x.std(ddof=1)), "n": int(len(x)), "clearly_positive": bool(m - h > 0)}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="reports/P150_results.json"); a = ap.parse_args()
    cells, missing = {}, []
    for c in CELLS:
        f = O / f"c0_{c}.json"
        if not f.exists():
            missing.append(f.name); continue
        cells[c] = json.load(open(f))["cells"][c]
    summ = {}
    for c, d in cells.items():
        reps = d["repeats"]; summ[c] = {}
        for site in SITES:
            s = {}
            for nm in CRITICS:
                r = [x["sites"][site]["critics"][nm] for x in reps]
                s[nm] = {"J_exactQ": tci([x["J_common_exactQ"] for x in r]), "J_sampledQ": tci([x["J_common_sampledQ"] for x in r]),
                         "S_plugin_mean": float(np.mean([x["S_plugin"] for x in r])), "frac_draws_boot_ci_above_0": float(np.mean([x["boot_ci95_exactQ"][0] > 0 for x in r])),
                         "rejection_rate": float(np.mean([x["reject"] for x in r])), "sat_obs_mean": float(np.mean([x["sat_obs"] for x in r]))}
                if "nest" in r[0]:
                    s[nm]["refit_spread_mean"] = float(np.mean([x["refit_spread_fixed_pool"] for x in r]))
                    s[nm]["step0_fraction"] = float(np.mean([v for x in r for v in x["nest"]["step0_selected"]]))
                    s[nm]["picked"] = {k: int(v) for k, v in zip(*np.unique([p for x in r for p in x["nest"]["picked"]], return_counts=True))}
            pairs = {}
            for k in reps[0]["sites"][site]["paired_diff_exactQ"]:
                pd = [x["sites"][site]["paired_diff_exactQ"][k] for x in reps]
                pairs[k] = {"mean_over_draws": float(np.mean([p["mean"] for p in pd])), "frac_draws_ci_excludes_0": float(np.mean([p["ci95"][0] > 0 or p["ci95"][1] < 0 for p in pd]))}
            summ[c][site] = {"critics": s, "paired_diff_exactQ": pairs}
    reading = {}
    if "colour_s0.1" in summ:
        h, O_ = summ["colour_s0.1"]["h"]["critics"], summ["colour_s0.1"]["O"]["critics"]
        null_rate = {nm: summ["null"]["h"]["critics"][nm]["rejection_rate"] for nm in CRITICS} if "null" in summ else None
        pos_h = {nm: h[nm]["J_exactQ"]["clearly_positive"] for nm in CRITICS}; pos_O = {nm: O_[nm]["J_exactQ"]["clearly_positive"] for nm in CRITICS}
        reading = {"clearly_positive_h": pos_h, "clearly_positive_O": pos_O, "rejection_rate_h": {nm: h[nm]["rejection_rate"] for nm in CRITICS}, "null_rejection_rate_h": null_rate,
                   "(a)_detection_only": bool(not any(pos_h.values()) and max(h[nm]["rejection_rate"] for nm in CRITICS) >= 0.3),
                   "(b)_function_class_limited": bool((pos_h["AUD-VCS"] or pos_h["AUD-JS"]) and not (pos_h["NEST-VCS"] or pos_h["NEST-JS"])),
                   "(c)_h_to_O_loss": bool(any(pos_h.values()) and not any(pos_O.values()))}
    json.dump({"summary": summ, "reading": reading, "missing": missing}, open(a.out, "w"), indent=1)
    for c, d in summ.items():
        for site, s in d.items():
            print(f"== {c} / {site}")
            for nm, v in s["critics"].items():
                j = v["J_exactQ"]
                print(f"  {nm:9s} J_exactQ {j['mean']:+.4f} [{j['ci95'][0]:+.4f},{j['ci95'][1]:+.4f}] draws CI>0 {v['frac_draws_boot_ci_above_0']:.2f} reject {v['rejection_rate']:.2f} "
                      f"S_plug {v['S_plugin_mean']:.4f}" + (f" refit sd {v['refit_spread_mean']:.4f} step0 {v['step0_fraction']:.2f}" if "refit_spread_mean" in v else ""))
    print("reading:", json.dumps(reading))
    if missing:
        print("missing:", missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
