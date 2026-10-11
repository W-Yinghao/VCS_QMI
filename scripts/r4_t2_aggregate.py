"""R4-T2 readings (reports/R4/R4_T2_SAMPLING_PROTOCOL_FROZEN_20261011.yaml) over outputs/R4_T2/*.json (scripts/r4_t2_sampling.py).

Per cell: bias against the target (Monte Carlo SE of the draws plus the TRUTH SE in stages 2–3); Monte Carlo variance against the exact Hoeffding
variance (stage 1a); coverage with Wilson 95 % intervals — two-sided for J(T) (image-local: μ_G), one-sided for J(T), one-sided for S (image-local:
E_G S_G) — and mean two-sided widths; sparse shifts: conditional-variance formula against resampling, total variance against Var Ĵ + E conditional
variance, critic evaluations and wall time; stage 3: per trained critic coverage, training-variance decomposition and coverage of the pipeline
mean E_s J(T_s).  Implementation checks = the frozen gate for stage 3.  Writes reports/R4/R4_T2_results.json and R4_T2_tables.md.
    python scripts/r4_t2_aggregate.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

IN = Path("/home/infres/yinwang/CS_QMI/outputs/R4_T2"); OUT = Path(__file__).resolve().parents[1] / "reports" / "R4"
DELTA, Z2, Z1, KS, TOL = 0.05, stats.norm.ppf(0.975), stats.norm.ppf(0.95), (1, 4, 16), 1e-9
CROSS = ("naive", "jk", "blk_t", "blk_hoeff", "bd") + tuple(f"K{K}_jk" for K in KS); IMG = ("img_t", "pooled_naive", "img_hoeff", "boot")


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return None
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [p, c - h, c + h]


def arr(r, k):
    return np.array([np.nan if v is None else v for v in r["draws"][k]], dtype=float)


def cross_intervals(r):
    """name -> (centre, two-sided half-width, one-sided LCB) per draw."""
    n = r["n"]; J = arr(r, "J"); m = arr(r, "m_blk"); out = {
        "naive": (J, Z2 * arr(r, "se_naive"), J - Z1 * arr(r, "se_naive")), "jk": (J, Z2 * arr(r, "se_jk"), J - Z1 * arr(r, "se_jk")),
        "blk_t": (arr(r, "J_blk"), stats.t.ppf(0.975, m - 1) * arr(r, "sd_blk") / np.sqrt(m), arr(r, "J_blk") - stats.t.ppf(0.95, m - 1) * arr(r, "sd_blk") / np.sqrt(m)),
        "blk_hoeff": (arr(r, "J_blk"), np.sqrt(8 * math.log(2 / DELTA) / m), arr(r, "J_blk") - np.sqrt(8 * math.log(1 / DELTA) / m)),
        "bd": (J, np.full_like(J, math.sqrt(32 * math.log(2 / DELTA) / n)), J - math.sqrt(32 * math.log(1 / DELTA) / n))}
    for K in KS:
        if f"K{K}.J" in r["draws"]:
            c = arr(r, f"K{K}.J"); s = arr(r, f"K{K}.se_jk"); out[f"K{K}_jk"] = (c, Z2 * s, c - Z1 * s)
    return out


def image_intervals(r):
    N = r["n"]; J = arr(r, "J"); s = arr(r, "sd_img") / math.sqrt(N); lo, hi = arr(r, "boot_lo"), arr(r, "boot_hi")
    return {"img_t": (J, stats.t.ppf(0.975, N - 1) * s, J - stats.t.ppf(0.95, N - 1) * s), "pooled_naive": (J, Z2 * arr(r, "se_pooled"), J - Z1 * arr(r, "se_pooled")),
            "img_hoeff": (J, np.full_like(J, math.sqrt(8 * math.log(2 / DELTA) / N)), J - math.sqrt(8 * math.log(1 / DELTA) / N)),
            "boot": ((lo + hi) / 2, (hi - lo) / 2, arr(r, "boot_lcb"))}


def coverage(ints, target, s_target):
    res = {}
    for name, (c, h, lcb) in ints.items():
        ok = np.isfinite(c); N = int(ok.sum())
        two = int((np.abs(c[ok] - target) <= h[ok] + TOL).sum()); one = int((lcb[ok] <= target + TOL).sum()); s1 = int((np.maximum(0.0, lcb[ok]) <= s_target + TOL).sum())
        res[name] = {"two_sided_J": wilson(two, N), "one_sided_J": wilson(one, N), "one_sided_S": wilson(s1, N), "mean_width": float(2 * h[ok].mean()), "n_draws": N}
    return res


def bias_check(r, target, target_se):
    J = arr(r, "J"); se = math.hypot(J.std(ddof=1) / math.sqrt(len(J)), target_se); b = float(J.mean() - target)
    return {"bias": b, "se": se, "pass": abs(b) <= 3.3 * se if se > 0 else abs(b) < TOL}


def variance_check(r, gen):
    J = arr(r, "J"); ve = r["var_exact"]
    if r.get("degenerate") or ve <= 0:
        return {"ratio": None, "pass": True, "note": "degenerate / constant kernel"}
    boots = np.array([J[gen.integers(0, len(J), len(J))].var(ddof=1) for _ in range(2000)]) / ve; lo, hi = np.quantile(boots, [0.0005, 0.9995])
    return {"ratio": float(J.var(ddof=1) / ve), "ci999": [float(lo), float(hi)], "pass": bool(lo <= 1 <= hi)}


def sparse_checks(r):
    out = {}; vJ = arr(r, "J").var(ddof=1)
    for K in KS:
        if f"K{K}.J" not in r["draws"]:
            continue
        cv = arr(r, f"K{K}.cond_var"); rs = arr(r, f"K{K}.resampled_var"); ok = np.isfinite(rs) & (cv > 1e-15)
        ratio = float(np.mean(rs[ok] / cv[ok])) if ok.any() else None
        tot = arr(r, f"K{K}.J").var(ddof=1); pred = vJ + cv.mean()
        out[f"K{K}"] = {"formula_ratio": ratio, "formula_pass": ratio is None or 0.95 <= ratio <= 1.05, "total_var": float(tot), "pred_total_var": float(pred),
                        "total_ratio": float(tot / pred) if pred > 0 else None, "evals": int(np.nanmean(arr(r, f"K{K}.evals"))), "evals_full": int(r["n"] ** 2)}
    if "t_critic_K16" in r["draws"]:
        out["seconds_critic_full"] = float(np.mean(arr(r, "t_critic_full"))); out["seconds_critic_K16"] = float(np.mean(arr(r, "t_critic_K16")))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args(); src = IN / ("smoke" if a.smoke else "")
    gen = np.random.default_rng(20261011); res = {"protocol": "reports/R4/R4_T2_SAMPLING_PROTOCOL_FROZEN_20261011.yaml", "cells": [], "stage3": {}, "checks": {"failures": []}}
    truths = {(t["setting"], t["level"]): t for t in (json.load(open(src / "2_truths.json")) if (src / "2_truths.json").exists() else [])}
    for f in sorted(src.glob("1a_*.json")) + sorted(src.glob("1b_*.json")) + sorted(src.glob("2_*_n*.json")):
        r = json.load(open(f)); c = {k: r[k] for k in ("stage", "level", "critic", "n", "draws_done") if k in r}; c["setting"] = r.get("setting", "finite")
        if r["stage"] == "1b":
            c.update(target=r["mu_G"], S_target=r["S_G"], constant_kernel=r["constant_kernel"]); c["coverage"] = coverage(image_intervals(r), r["mu_G"], r["S_G"])
            c["bias"] = bias_check(r, r["mu_G"], 0.0)
        else:
            c.update(target=r["J"], S_target=r["S"], constant_kernel=r["constant_kernel"], degenerate=r.get("degenerate", r["constant_kernel"]))
            c["coverage"] = coverage(cross_intervals(r), r["J"], r["S"]); c["bias"] = bias_check(r, r["J"], r.get("J_se", 0.0)); c["sparse"] = sparse_checks(r)
            if r["stage"] == "1a":
                c["variance"] = variance_check(r, gen); c["zeta1"] = r["zeta1"]; c["E_h2"] = r["E_h2"]; c["var_exact"] = r["var_exact"]; c["rho"] = r["rho"]
        res["cells"].append(c); tag = f"{c['stage']} {c['setting']} {c['level']} {c['critic']} n{c['n']}"
        if not c["bias"]["pass"]:
            res["checks"]["failures"].append(f"bias {tag}: {c['bias']['bias']:+.5f} (se {c['bias']['se']:.5f})")
        if not c.get("variance", {"pass": True})["pass"]:
            res["checks"]["failures"].append(f"variance {tag}: ratio {c['variance']['ratio']:.3f} CI {c['variance']['ci999']}")
        for K, s in c.get("sparse", {}).items():
            if isinstance(s, dict) and not s["formula_pass"]:
                res["checks"]["failures"].append(f"sparse formula {tag} {K}: ratio {s['formula_ratio']:.3f}")
        for name in ("bd", "blk_hoeff", "img_hoeff"):
            cv = c["coverage"].get(name)
            if cv and cv["two_sided_J"][0] < 0.95 - 3.3 * math.sqrt(0.95 * 0.05 / cv["n_draws"]):
                res["checks"]["failures"].append(f"bounded radius {tag} {name}: coverage {cv['two_sided_J'][0]:.3f}")
    res["checks"]["stage3_allowed"] = not res["checks"]["failures"] and len(res["cells"]) > 0
    s3 = sorted(src.glob("3_*.json"))
    if s3:
        S_g = truths.get(("gaussian", "strong"), {}).get("S"); per = {}
        for f in s3:
            r = json.load(open(f)); key = (r["kind"], r["n"]); per.setdefault(key, []).append(r)
        for (kind, n), rs in sorted(per.items()):
            Js = np.array([r["J"] for r in rs]); mu = float(Js.mean()); ev = np.array([arr(r, "J").var(ddof=1) for r in rs])
            cov_s = {r["seed"]: coverage(cross_intervals(r), r["J"], S_g if S_g is not None else np.inf) for r in rs}
            pip = coverage({k: tuple(np.concatenate([cross_intervals(r)[k][i] for r in rs]) for i in range(3)) for k in ("jk", "naive", "bd")}, mu, S_g if S_g is not None else np.inf)
            res["stage3"][f"{kind}_n{n}"] = {"seeds": [r["seed"] for r in rs], "J_T_s": Js.tolist(), "J_T_s_se": [r["J_se"] for r in rs], "var_train": float(Js.var(ddof=1)) if len(Js) > 1 else None,
                                             "mean_var_eval": float(ev.mean()), "S": S_g, "per_seed_coverage": cov_s, "pipeline_mean_coverage": pip,
                                             "selected_update": [r["fit"].get("selected_update") for r in rs]}
    OUT.mkdir(parents=True, exist_ok=True); stem = "R4_T2_results" + ("_smoke" if a.smoke else ""); json.dump(res, open(OUT / f"{stem}.json", "w"), indent=1)
    pct = lambda w: "—" if w is None else f"{100 * w[0]:.1f}"
    L = [f"# R4-T2 — fixed-critic paired-sampling validation: results{' (SMOKE)' if a.smoke else ''}", "",
         f"Implementation checks (gate for stage 3): {'PASS' if res['checks']['stage3_allowed'] else 'FAIL'}; failures: {res['checks']['failures'] or 'none'}.", ""]
    for stage, cols, title in (("1a", CROSS, "Stage 1a — finite-state cross-pair"), ("2", CROSS, "Stage 2 — existing generators, cross-pair"), ("1b", IMG, "Stage 1b — finite-state image-local")):
        cs = [c for c in res["cells"] if c["stage"] == stage]
        if not cs:
            continue
        L += [f"## {title}: two-sided coverage of the target (%), Wilson-checked; [one-sided S coverage of the jk / img_t readout]", "",
              "| setting | level | critic | n | target | " + " | ".join(cols) + " | S one-sided | mean width jk / img_t | flags |", "|" + "---|" * (len(cols) + 8)]
        for c in cs:
            main_ro = "jk" if stage != "1b" else "img_t"; flags = []
            if c.get("degenerate") or c.get("constant_kernel"):
                flags.append("degenerate")
            for name in cols:
                w = c["coverage"].get(name, {}).get("two_sided_J")
                if w and w[2] < 0.95:
                    flags.append(f"{name} under")
            L.append(f"| {c['setting']} | {c['level']} | {c['critic']} | {c['n']} | {c['target']:.4f} | " + " | ".join(pct(c["coverage"].get(nm, {}).get("two_sided_J")) for nm in cols)
                     + f" | {pct(c['coverage'][main_ro]['one_sided_S'])} | {c['coverage'][main_ro]['mean_width']:.4f} | {', '.join(flags)} |")
        L.append("")
    v = [c for c in res["cells"] if "variance" in c and c["variance"]["ratio"] is not None]
    if v:
        L += ["## Stage 1a — Monte Carlo variance / exact Hoeffding variance (99.9 % bootstrap interval)", "", "| level | critic | n | ζ₁ | E h₂² | ratio | interval |", "|---|---|---|---|---|---|---|"]
        L += [f"| {c['level']} | {c['critic']} | {c['n']} | {c['zeta1']:.2e} | {c['E_h2']:.2e} | {c['variance']['ratio']:.3f} | [{c['variance']['ci999'][0]:.3f}, {c['variance']['ci999'][1]:.3f}] |" for c in v] + [""]
    sp = [c for c in res["cells"] if c.get("sparse")]
    if sp:
        L += ["## Sparse cyclic shifts — formula / resampled conditional variance, total / predicted variance, evaluations", "",
              "| stage | setting | level | critic | n | K | formula ratio | total ratio | evals (K) / full | critic s full / K16 |", "|---|---|---|---|---|---|---|---|---|---|"]
        for c in sp:
            for K in KS:
                s = c["sparse"].get(f"K{K}")
                if s:
                    fr = "—" if s["formula_ratio"] is None else f"{s['formula_ratio']:.3f}"; tr = "—" if s["total_ratio"] is None else f"{s['total_ratio']:.3f}"
                    tm = f"{c['sparse']['seconds_critic_full']:.4f} / {c['sparse']['seconds_critic_K16']:.4f}" if "seconds_critic_full" in c["sparse"] else "—"
                    L.append(f"| {c['stage']} | {c['setting']} | {c['level']} | {c['critic']} | {c['n']} | {K} | {fr} | {tr} | {s['evals']} / {s['evals_full']} | {tm} |")
        L.append("")
    if res["stage3"]:
        L += ["## Stage 3 — learned critics (gaussian d 20, I = 2 nats): per-seed J(T_s), training vs EVAL variance, coverage", "",
              "| kind | n | J(T_s) seeds 0–4 | Var_train | mean Var_EVAL | jk coverage of own J(T_s) (mean over seeds) | jk coverage of the pipeline mean | S one-sided (jk) |", "|---|---|---|---|---|---|---|---|"]
        for k, s in res["stage3"].items():
            own = np.mean([v["jk"]["two_sided_J"][0] for v in s["per_seed_coverage"].values()]); vt = "—" if s["var_train"] is None else f"{s['var_train']:.2e}"
            Js = ", ".join(f"{x:.4f}" for x in s["J_T_s"]); kind, n = k.split("_n")
            L.append(f"| {kind} | {n} | {Js} | {vt} | {s['mean_var_eval']:.2e} | {100 * own:.1f} | {pct(s['pipeline_mean_coverage']['jk']['two_sided_J'])} | {pct(s['pipeline_mean_coverage']['jk']['one_sided_S'])} |")
        L.append("")
    (OUT / f"{stem.replace('results', 'tables')}.md").write_text("\n".join(L)); print("\n".join(L[:40]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
