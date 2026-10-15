"""V9-G2 readings (reports/V9/V9_G2_PROTOCOL_*.yaml) over the unit results written by REG-P3's register_one (g2.cmd_register) and REG-P3's own
MI / NMI results (reused after the hash check).  Every statistic aggregates per independent UNIT (the EPFL scene with near-duplicates merged,
REG-P3's window -> unit map): a unit's success is the fraction of its registrations (its windows × 10 perturbations) below the threshold, DEV
success is the mean over DEV units (unit-macro).  Thresholds err_full < 42 / < 9 = 10.5 / 2.25 native px.  Per cell (method × fit_fraction ×
train_mismatch): mean ± sd over the 3 seeds; contrasts VCS − X at every cell (paired by seed for seeded X; VCS seed vs the deterministic X): 95 % t
over seeds and a unit bootstrap (2 000 resamples, indices shared across contrasts) with a two-sided bootstrap p.  Primary family (Holm over 4):
VCS − MI and VCS − CoMIR at ε = 0.2 for both fractions, success < 10.5 px; detection = both intervals exclude 0 and Holm p < 0.05.  Interaction
(descriptive): per method the change 0 → 0.2 at each fraction.  Ceiling readout (REG-P3's, pre-registered): if every method's CAL unit success
< 10.5 px is >= 0.95, the < 2.25 px success is the discriminating readout (with the truth caveat).  Wording: interval containing 0 -> no detectable
difference; VCS ≈ logistic ≈ InfoNCE -> a learned relational-energy class effect, never "VCS-only"."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import g2 as G
from .g1_aggregate import holm
from .r6_reg_aggregate import tint

THR = (42.0, 9.0); X_ALL = ("mi", "nmi", "kernel", "logistic", "infonce", "comir")


def per_unit(recs: dict, thr: float, unit_of: dict) -> dict:
    pooled = G.P3.pool_units(recs, unit_of)
    return {u: float(np.mean([r["err_full"] < thr for r in rs])) for u, rs in pooled.items() if rs}


def load_runs(smoke=False):
    runs, missing = {}, []
    for uid, m, f, e, s in G.all_units() + G.kernel_units():
        p = G.result_path(uid, m, s, smoke)
        if p.exists():
            runs[(m, f, e, s)] = json.load(open(p))
        else:
            missing.append(uid)
    chk = G.mi_nmi_reuse_check()
    for mth in ("mi", "nmi"):
        f = Path(chk["results"][mth]["file"])
        if f.exists() and not smoke:
            runs[(mth, None, None, None)] = json.load(open(f))
    return runs, missing, chk


def macro(run, thr, unit_of):
    return float(np.mean(list(per_unit(run["dev"], thr, unit_of).values())))


def contrast(runs, frac, eps, other, thr, g, unit_of, n_boot=2000):
    vk = sorted(k for k in runs if k[0] == "vcs" and k[1] == frac and k[2] == eps)
    if other in ("mi", "nmi"):
        ok = [(other, None, None, None)]
    elif other == "kernel":
        ok = [k for k in runs if k[0] == "kernel" and k[1] == frac and k[2] == eps]
    else:
        ok = [k for k in runs if k[0] == other and k[1] == frac and k[2] == eps]
    if not vk or not any(k in runs for k in ok):
        return None
    pairs = [(v, next((o for o in ok if o[3] == v[3]), ok[0]) if other not in ("mi", "nmi", "kernel") else ok[0]) for v in vk]; pairs = [(v, o) for v, o in pairs if o in runs]
    pu = lambda k: per_unit(runs[k]["dev"], thr, unit_of)
    units = sorted(set.intersection(*[set(pu(v)) & set(pu(o)) for v, o in pairs]))
    D = np.array([[pu(v)[u] - pu(o)[u] for u in units] for v, o in pairs])
    boot = np.array([D[:, g.integers(0, len(units), len(units))].mean() for _ in range(n_boot)]) if len(units) else np.array([])
    pb = float(2 * min((boot <= 0).mean(), (boot >= 0).mean())) if len(boot) else None
    return {**tint(D.mean(1)), "unit_boot95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))] if len(boot) else None, "boot_p": pb, "n_units": len(units), "pairs": [[v[3], o[3]] for v, o in pairs]}


def main(out: Path, smoke=False):
    m = G.load_manifest(G.P3_ROOT, False) if not smoke else G.load_manifest(G.V9_ROOT / "p3root", True)       # smoke: the CLI has already redirected V9_ROOT to <root>/smoke; unit_of = G.window_unit(m)
    runs, missing, chk = load_runs(smoke); res = {"protocol": "reports/V9/V9_G2_PROTOCOL_*.yaml", "dataset": G.DATASET, "thresholds": G.THRESHOLDS_NATIVE, "mi_nmi_reuse": chk, "missing": missing,
                                                 "cells": {}, "contrasts": {}, "primary": {}, "interaction": {}, "ceiling_readout": {}}
    for mth in G.METHODS + ("kernel",):
        for f in G.FRACTIONS:
            for e in G.EPSILONS:
                keys = sorted(k for k in runs if k[0] == mth and k[1] == f and k[2] == e)
                if not keys:
                    continue
                row = {"seeds": [k[3] for k in keys], "n_fit_units": [runs[k].get("n_fit_units") for k in keys]}
                for thr in THR:
                    row[f"success{int(thr)}"] = tint([macro(runs[k], thr, unit_of) for k in keys])
                row["cal_success42_unit"] = [runs[k].get("cal_success42_unit") for k in keys]
                for fld in ("dev_median_err", "dev_p90_err", "dev_median_rot_err_deg", "dev_median_trans_err", "dev_sec_per_registration"):
                    row[fld] = tint([runs[k][fld] for k in keys if runs[k].get(fld) is not None])
                row["failures42"] = [runs[k]["dev_failures42"] for k in keys]; row["failures42_energy"] = [runs[k]["dev_failures42_energy"] for k in keys]
                if mth in ("vcs", "logistic", "infonce"):
                    row["chosen_ckpt"] = [runs[k].get("chosen_ckpt") for k in keys]; row["realised_mismatch"] = [runs[k].get("realised_mismatch_fraction") for k in keys]
                res["cells"][f"{mth}|frac{f:g}|eps{e:g}"] = row
    for mth in ("mi", "nmi"):
        k = (mth, None, None, None)
        if k in runs:
            res["cells"][mth] = {"seeds": None, **{f"success{int(t)}": tint([macro(runs[k], t, unit_of)]) for t in THR}, "cal_success42_unit": [runs[k].get("cal_success42_unit")],
                                 "dev_median_err": tint([runs[k]["dev_median_err"]]), "failures42": [runs[k]["dev_failures42"]], "failures42_energy": [runs[k]["dev_failures42_energy"]]}
    cal = [v for r in res["cells"].values() for v in (r.get("cal_success42_unit") or []) if v is not None]
    res["ceiling_readout"] = {"all_cal_success42_ge_0.95": bool(cal) and all(v >= 0.95 for v in cal), "min_cal_success42": min(cal) if cal else None, "rule": "if every method's CAL unit success < 10.5 px >= 0.95 the < 2.25 px success is the discriminating readout (REG-P3's pre-registered readout)"}
    g = np.random.default_rng(20261016)
    for thr in THR:
        for f in G.FRACTIONS:
            for e in G.EPSILONS:
                for other in X_ALL:
                    c = contrast(runs, f, e, other, thr, g, unit_of)
                    if c:
                        res["contrasts"][f"vcs_minus_{other}|frac{f:g}|eps{e:g}@{int(thr)}"] = c
    prim = [f"vcs_minus_{o}|frac{f:g}|eps0.2@42" for f in G.FRACTIONS for o in ("mi", "comir")]; have = [k for k in prim if k in res["contrasts"]]
    if have:
        ph = holm([res["contrasts"][k]["boot_p"] for k in have])
        for k, p in zip(have, ph):
            c = res["contrasts"][k]; excl = lambda iv: iv is not None and (iv[0] > 0 or iv[1] < 0)
            res["primary"][k] = {**c, "p_holm": float(p), "detected": bool(excl(c.get("ci95")) and excl(c.get("unit_boot95")) and p < 0.05), "complete": len(c["pairs"]) == 3}
    for mth in G.METHODS + ("kernel",):
        for f in G.FRACTIONS:
            k0 = sorted(k for k in runs if k[0] == mth and k[1] == f and k[2] == 0.0); k2 = sorted(k for k in runs if k[0] == mth and k[1] == f and k[2] == 0.2)
            pairs = [(a, b) for a in k0 for b in k2 if a[3] == b[3]]
            if pairs:
                res["interaction"][f"{mth}|frac{f:g}|delta_0_to_0.2@42"] = tint([macro(runs[b], 42.0, unit_of) - macro(runs[a], 42.0, unit_of) for a, b in pairs])
    out.mkdir(parents=True, exist_ok=True); json.dump(res, open(out / ("g2_results_smoke.json" if smoke else "g2_results.json"), "w"), indent=1)
    p = lambda t: "—" if not t or t.get("mean") is None else f"{100 * t['mean']:.1f}" + (f" ± {100 * t['sd']:.1f}" if t.get("sd") is not None else "")
    q = lambda t: f"{100 * t['mean']:+.1f}" + (f" [{100 * t['ci95'][0]:+.1f}, {100 * t['ci95'][1]:+.1f}]" if t.get("ci95") else "") + (f"; unit bootstrap [{100 * t['unit_boot95'][0]:+.1f}, {100 * t['unit_boot95'][1]:+.1f}]" if t.get("unit_boot95") else "")
    L = [f"# V9-G2 — EPFL RGB–NIR rigid registration: training-sample fraction × training mismatch{' (SMOKE)' if smoke else ''}", "", f"Missing units: {missing or 'none'}.  Ceiling readout: {res['ceiling_readout']}", "",
         "| cell | seeds | n FIT units | success < 10.5 px (%) | success < 2.25 px (%) | median corner error (err_full px) | failures (energy) | s / registration |", "|---|---|---|---|---|---|---|---|"]
    for k, r in res["cells"].items():
        L.append(f"| {k} | {r.get('seeds')} | {r.get('n_fit_units')} | {p(r['success42'])} | {p(r['success9'])} | {r['dev_median_err']['mean']:.1f} | {sum(r['failures42'])} ({sum(r['failures42_energy'])}) | {r['dev_sec_per_registration']['mean']:.2f} |" if r.get("dev_sec_per_registration") else f"| {k} | — | — | {p(r['success42'])} | {p(r['success9'])} | {r['dev_median_err']['mean']:.1f} | {sum(r['failures42'])} ({sum(r['failures42_energy'])}) | — |")
    L += ["", "## Primary family (Holm over 4; detected = seed t and unit bootstrap exclude 0 and Holm p < 0.05)", "", "| contrast | value | boot p | Holm p | detected | complete |", "|---|---|---|---|---|---|"]
    L += [f"| {k} | {q(c)} | {c['boot_p']:.4f} | {c['p_holm']:.4f} | {c['detected']} | {c['complete']} |" for k, c in res["primary"].items()]
    L += ["", "## All contrasts", "", "| contrast | value |", "|---|---|"] + [f"| {k} | {q(c)} |" for k, c in res["contrasts"].items()]
    L += ["", "## Interaction: change of success < 10.5 px from ε 0 to 0.2 (seed-paired)", "", "| method, fraction | Δ (pp) |", "|---|---|"] + [f"| {k} | {q(v)} |" for k, v in res["interaction"].items()]
    (out / ("g2_tables_smoke.md" if smoke else "g2_tables.md")).write_text("\n".join(L) + "\n"); print("\n".join(L[:40])); return 0
