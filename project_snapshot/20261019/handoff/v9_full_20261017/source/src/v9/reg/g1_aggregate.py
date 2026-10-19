"""V9-G1 readings (reports/V9/V9_G1_PROTOCOL_*.yaml) over the unit results written by REG-P2's register (g1.cmd_register) and REG-P2's own
MI / NMI results (reused after the hash check).  Per cell (method × fit_fraction × train_mismatch): DEV unit-macro success at < 42 / < 9 px of
CoMIR's mean corner error, median / p90 corner error, median rotation / translation error, failures (energy vs search), s / registration;
mean ± sd over the 3 seeds.  Contrasts VCS − X at every cell (paired by seed for seeded X, VCS seed vs the deterministic X): 95 % t over seeds and
a unit (image) bootstrap with indices shared across contrasts, plus a two-sided bootstrap p.  Primary family (Holm over 4): VCS − MI and VCS − CoMIR
at ε = 0.2 for both fractions, success < 42 px; detection = both intervals exclude 0 and Holm p < 0.05.  Interaction (descriptive): per method the
change 0 → 0.2 at each fraction, and VCS − X of those changes.  Wording: interval containing 0 -> no detectable difference; VCS ≈ logistic ≈
InfoNCE -> a learned relational-energy class effect, never "VCS-only"."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

from . import g1 as G
from .r6_reg_aggregate import per_image, tint

THR = (42.0, 9.0); X_ALL = ("mi", "nmi", "kernel", "logistic", "infonce", "comir")


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
        if f.exists():
            runs[(mth, None, None, None)] = json.load(open(f))
    return runs, missing, chk


def macro(run, thr):
    return float(np.mean(list(per_image(run["dev"], thr).values())))


def holm(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p); out = np.empty(n); prev = 0.0
    for rank, i in enumerate(order):
        prev = max(prev, (n - rank) * p[i]); out[i] = min(1.0, prev)
    return out


def contrast(runs, frac, eps, other, thr, g, n_boot=2000):
    """VCS(frac, eps, seeds) − X: X seeded at the same cell (paired by seed), kernel at seed 901 of the same cell, MI / NMI deterministic."""
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
    imgs = sorted(set.intersection(*[set(per_image(runs[v]["dev"], thr)) & set(per_image(runs[o]["dev"], thr)) for v, o in pairs]))
    D = np.array([[per_image(runs[v]["dev"], thr)[i] - per_image(runs[o]["dev"], thr)[i] for i in imgs] for v, o in pairs])
    boot = np.array([D[:, g.integers(0, len(imgs), len(imgs))].mean() for _ in range(n_boot)]) if len(imgs) else np.array([])
    pb = float(2 * min((boot <= 0).mean(), (boot >= 0).mean())) if len(boot) else None
    return {**tint(D.mean(1)), "image_boot95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))] if len(boot) else None, "boot_p": pb,
            "n_units": len(imgs), "pairs": [[v[3], o[3]] for v, o in pairs]}


def main(out: Path, smoke=False):
    runs, missing, chk = load_runs(smoke); res = {"protocol": "reports/V9/V9_G1_PROTOCOL_*.yaml", "mi_nmi_reuse": chk, "missing": missing, "cells": {}, "contrasts": {}, "primary": {}, "interaction": {}}
    for m in G.METHODS + ("kernel",):
        for f in G.FRACTIONS:
            for e in G.EPSILONS:
                keys = sorted(k for k in runs if k[0] == m and k[1] == f and k[2] == e)
                if not keys:
                    continue
                row = {"seeds": [k[3] for k in keys], "n_fit": [len(runs[k].get("fit", runs[k].get("train", {}).get("fit", []))) for k in keys]}
                for thr in THR:
                    row[f"success{int(thr)}"] = tint([macro(runs[k], thr) for k in keys])
                for fld in ("dev_median_err", "dev_p90_err", "dev_median_rot_err_deg", "dev_median_trans_err", "dev_sec_per_registration"):
                    row[fld] = tint([runs[k][fld] for k in keys if runs[k].get(fld) is not None])
                row["failures42"] = [runs[k]["dev_failures42"] for k in keys]; row["failures42_energy"] = [runs[k]["dev_failures42_energy"] for k in keys]
                if m in ("vcs", "logistic", "infonce"):
                    row["chosen_ckpt"] = [runs[k].get("chosen_ckpt") for k in keys]; row["realised_mismatch"] = [runs[k].get("train", {}).get("realised_mismatch_fraction") for k in keys]
                res["cells"][f"{m}|frac{f:g}|eps{e:g}"] = row
    for mth in ("mi", "nmi"):
        k = (mth, None, None, None)
        if k in runs:
            res["cells"][mth] = {"seeds": None, **{f"success{int(t)}": tint([macro(runs[k], t)]) for t in THR}, "dev_median_err": tint([runs[k]["dev_median_err"]]), "failures42": [runs[k]["dev_failures42"]], "failures42_energy": [runs[k]["dev_failures42_energy"]]}
    g = np.random.default_rng(20261016)
    for thr in THR:
        for f in G.FRACTIONS:
            for e in G.EPSILONS:
                for other in X_ALL:
                    c = contrast(runs, f, e, other, thr, g)
                    if c:
                        res["contrasts"][f"vcs_minus_{other}|frac{f:g}|eps{e:g}@{int(thr)}"] = c
    prim = [f"vcs_minus_{o}|frac{f:g}|eps0.2@42" for f in G.FRACTIONS for o in ("mi", "comir")]; have = [k for k in prim if k in res["contrasts"]]
    if have:
        ph = holm([res["contrasts"][k]["boot_p"] for k in have])
        for k, p in zip(have, ph):
            c = res["contrasts"][k]; excl = lambda iv: iv is not None and (iv[0] > 0 or iv[1] < 0)
            res["primary"][k] = {**c, "p_holm": float(p), "detected": bool(excl(c.get("ci95")) and excl(c.get("image_boot95")) and p < 0.05), "complete": len(c["pairs"]) == 3}
    for m in G.METHODS + ("kernel",):                                             # interaction 0 -> 0.2 per method and fraction (seed-paired)
        for f in G.FRACTIONS:
            k0 = sorted(k for k in runs if k[0] == m and k[1] == f and k[2] == 0.0); k2 = sorted(k for k in runs if k[0] == m and k[1] == f and k[2] == 0.2)
            pairs = [(a, b) for a in k0 for b in k2 if a[3] == b[3]]
            if pairs:
                res["interaction"][f"{m}|frac{f:g}|delta_0_to_0.2@42"] = tint([macro(runs[b], 42.0) - macro(runs[a], 42.0) for a, b in pairs])
    out.mkdir(parents=True, exist_ok=True); json.dump(res, open(out / ("g1_results_smoke.json" if smoke else "g1_results.json"), "w"), indent=1)
    p = lambda t: "—" if not t or t.get("mean") is None else f"{100 * t['mean']:.1f}" + (f" ± {100 * t['sd']:.1f}" if t.get("sd") is not None else "")
    q = lambda t: f"{100 * t['mean']:+.1f}" + (f" [{100 * t['ci95'][0]:+.1f}, {100 * t['ci95'][1]:+.1f}]" if t.get("ci95") else "") + (f"; unit bootstrap [{100 * t['image_boot95'][0]:+.1f}, {100 * t['image_boot95'][1]:+.1f}]" if t.get("image_boot95") else "")
    L = [f"# V9-G1 — BF/SHG rigid registration: training-sample fraction × training mismatch{' (SMOKE)' if smoke else ''}", "", f"Missing units: {missing or 'none'}.", "",
         "| cell | seeds | n FIT | success < 42 px (%) | success < 9 px (%) | median corner error (px) | failures (energy) | s / registration |", "|---|---|---|---|---|---|---|---|"]
    for k, r in res["cells"].items():
        L.append(f"| {k} | {r.get('seeds')} | {r.get('n_fit')} | {p(r['success42'])} | {p(r['success9'])} | {r['dev_median_err']['mean']:.1f} | {sum(r['failures42'])} ({sum(r['failures42_energy'])}) | {r['dev_sec_per_registration']['mean']:.2f} |" if r.get("dev_sec_per_registration") else f"| {k} | — | — | {p(r['success42'])} | {p(r['success9'])} | {r['dev_median_err']['mean']:.1f} | {sum(r['failures42'])} ({sum(r['failures42_energy'])}) | — |")
    L += ["", "## Primary family (Holm over 4; detected = seed t and unit bootstrap exclude 0 and Holm p < 0.05)", "", "| contrast | value | boot p | Holm p | detected | complete |", "|---|---|---|---|---|---|"]
    L += [f"| {k} | {q(c)} | {c['boot_p']:.4f} | {c['p_holm']:.4f} | {c['detected']} | {c['complete']} |" for k, c in res["primary"].items()]
    L += ["", "## All contrasts", "", "| contrast | value |", "|---|---|"] + [f"| {k} | {q(c)} |" for k, c in res["contrasts"].items()]
    L += ["", "## Interaction: change of success < 42 px from ε 0 to 0.2 (seed-paired)", "", "| method, fraction | Δ (pp) |", "|---|---|"] + [f"| {k} | {q(v)} |" for k, v in res["interaction"].items()]
    (out / ("g1_tables_smoke.md" if smoke else "g1_tables.md")).write_text("\n".join(L) + "\n"); print("\n".join(L[:40])); return 0
