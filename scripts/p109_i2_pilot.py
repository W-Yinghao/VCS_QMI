"""P109 I2 pilot — nested fine-set critics and the exact binary-N product term on the P102 task (package v4 §I2).

    python scripts/p109_i2_pilot.py --out reports/P109_i2_pilot [--cells vcs4v800:colour:0,vcs4v800:colour:0.1,vcs4v800:blur:0.5] [--smoke]

Task, information sets, maps (PROBE-fitted standardisation / PCA / linear probes), FIT / EVAL / POOL draws, n = 3 000 and 5 repeats: exactly P102
(`scripts/t_line_increments.py`, imported).  Per repeat and objective (VCS, matched JS) four fitting variants on the same draws:
    indep_sampled   P102 baseline: every set fitted independently, sampled within-class POOL negatives
    indep_exact     independent fits, Q term by exact enumeration over N ∈ {0, 1} with the known P(N | Y)
    nested_sampled  fine critics = frozen coarse critic + zero-initialised residual (chain bottom-up: logit16 → pca16 → pca64 → h; main pair
                    logit_h → h), sampled negatives
    nested_exact    nested + exact enumeration
Every variant is read out twice on EVAL: with the sampled Q partner (P102's readout) and with the exact enumeration, so the fitting change and the
readout change are separable.  Recorded: J per set, main-pair Delta_J / D_T / r_BA, chain steps, R_orth, nesting violations (J_fine < J_coarse),
refit spread over repeats.  Nothing is clipped, made monotone or orthogonalised.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from cond_test_t1 import Family, draw_sample  # noqa: E402
from precheck_d_tests import DEVICE, within_class_pool  # noqa: E402
from t_line_increments import CHAIN, FEATURE_DIRS, SETS, feature_sets, fit_maps  # noqa: E402
from vcs_measure import nested as NS  # noqa: E402
from vcs_ssl.utils import atomic_write_json, utc_now  # noqa: E402

VARIANTS = ("indep_sampled", "indep_exact", "nested_sampled", "nested_exact")


def coarse_maps(maps: dict) -> dict:
    """Deterministic fine → coarse maps between the P102 information sets (device-aware copies of the PROBE maps)."""
    import copy
    basis, zc = maps["basis"].to(DEVICE), maps["zc"].to(DEVICE)
    lin_h, lin_16 = copy.deepcopy(maps["lin_h"]).to(DEVICE), copy.deepcopy(maps["lin_16"]).to(DEVICE)   # copies: feature_sets keeps the CPU maps
    for m in (lin_h, lin_16):
        for p in m.parameters():
            p.requires_grad_(False)
    return {("pca16", "logit16"): lambda x: lin_16(x), ("pca64", "pca16"): lambda x: x[:, :16], ("h", "pca64"): lambda x: (x - zc) @ basis,
            ("h", "logit_h"): lambda x: lin_h(x)}


def run_repeat(fam, maps, cm, probe_set, n, rng, *, strength, steps, seed, n_classes):
    mode = "planted" if strength > 0 else "null_label_only"; s_eff = strength if strength > 0 else min(fam.planted)
    used = set(probe_set)
    Hf, Yf, Nf, _ = draw_sample(fam, n, rng, mode, s_eff, used); He, Ye, Ne, _ = draw_sample(fam, n, rng, mode, s_eff, used)
    Hp, Yp, Np, _ = draw_sample(fam, n, rng, mode, s_eff, used)
    with torch.no_grad():
        Ff = {k: v.to(DEVICE) for k, v in feature_sets(Hf, maps).items()}; Fe = {k: v.to(DEVICE) for k, v in feature_sets(He, maps).items()}
    t = lambda a: torch.as_tensor(a).to(DEVICE)
    nf, yf, ne, ye = t(Nf), t(Yf), t(Ne), t(Ye)
    negf = {"n_neg": t(within_class_pool(Yf, Yp, Np, rng))}; nege = {"n_neg": t(within_class_pool(Ye, Yp, Np, rng))}
    p1 = lambda yy: torch.as_tensor(fam.p_c_vec(yy) if hasattr(fam, "p_c_vec") else np.vectorize(fam.p_c.get)(yy), dtype=torch.float32).to(DEVICE)
    exf, p1e = {"p1": p1(Yf)}, p1(Ye)
    res = {"n": n, "strength": strength, "mode": mode, "per_objective": {}}
    for obj in ("vcs", "js"):
        res["per_objective"][obj] = {}
        indep = {}
        for variant in VARIANTS:
            neg = exf if variant.endswith("exact") else negf
            crit, info = {}, {}
            k0 = 0
            if variant.startswith("indep"):
                for k, nm in enumerate(SETS):
                    crit[nm], pick, vals, bs = NS.fit_picked(Ff[nm], nf, yf, neg, objective=obj, n_classes=n_classes, steps=steps, seed=seed * 100 + k)
                    info[nm] = {"picked": pick, "val_J": vals, "best_step": bs}
                indep[variant] = crit
                chain_crit = {c: crit[c] for c in CHAIN}; main_B = crit["h"]; main_A = crit["logit_h"]
            else:
                base = indep["indep_" + variant.split("_")[1]]          # same negative representation's independent base critics
                chain_crit = {"logit16": base["logit16"]}
                for k, (fine, coarse) in enumerate((("pca16", "logit16"), ("pca64", "pca16"), ("h", "pca64"))):
                    chain_crit[fine], pick, vals, bs = NS.fit_picked(Ff[fine], nf, yf, neg, objective=obj, n_classes=n_classes, steps=steps,
                                                                     seed=seed * 100 + 50 + k, coarse=chain_crit[coarse], cmap=cm[(fine, coarse)])
                    info[f"{fine}<-{coarse}"] = {"picked": pick, "val_J": vals, "best_step": bs}
                main_A = base["logit_h"]
                main_B, pick, vals, bs = NS.fit_picked(Ff["h"], nf, yf, neg, objective=obj, n_classes=n_classes, steps=steps, seed=seed * 100 + 60,
                                                        coarse=main_A, cmap=cm[("h", "logit_h")])
                info["h<-logit_h"] = {"picked": pick, "val_J": vals, "best_step": bs}
            ro = {}
            for nm, c in list(chain_crit.items()) + [("main_B", main_B), ("main_A", main_A)]:
                feat = Fe["h"] if nm == "main_B" else (Fe["logit_h"] if nm == "main_A" else Fe[nm])
                ro[nm] = NS.readouts(c, feat, ne, ye, nege, p1e)
            out = {"critics": info}
            for rdname, j in (("readout_sampled", 1), ("readout_exact", 2)):
                wq = ro["main_B"][j][1]
                mainB, mainA = (ro["main_B"][0], ro["main_B"][j][0]), (ro["main_A"][0], ro["main_A"][j][0])
                chain = [(ro[c][0], ro[c][j][0]) for c in CHAIN]
                out[rdname] = {"main_pair": NS.increment_w(mainB, mainA, wq), "chain": NS.chain_w(chain, CHAIN, wq)}
            res["per_objective"][obj][variant] = out
    return res


def summarise(reps):
    out = {}
    for obj in ("vcs", "js"):
        for v in VARIANTS:
            for rd in ("readout_sampled", "readout_exact"):
                vals = {k: [r["per_objective"][obj][v][rd]["main_pair"][k] for r in reps] for k in ("delta_J", "D_T", "r_BA", "J_B", "J_A")}
                vals["R_orth"] = [r["per_objective"][obj][v][rd]["chain"]["R_orth"] for r in reps]
                vals["nesting_violations"] = [r["per_objective"][obj][v][rd]["chain"]["nesting_violations"] for r in reps]
                for c in CHAIN:
                    vals[f"J_{c}"] = [r["per_objective"][obj][v][rd]["chain"]["J"][c] for r in reps]
                out[f"{obj}/{v}/{rd}"] = {k: {"mean": float(np.mean(x)), "sd_refit": float(np.std(x, ddof=1)) if len(x) > 1 else None} for k, x in vals.items()}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--cells", default="vcs4v800:colour:0,vcs4v800:colour:0.1,vcs4v800:blur:0.5")
    ap.add_argument("--n", type=int, default=3000); ap.add_argument("--repeats", type=int, default=5); ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--probe", type=int, default=8000); ap.add_argument("--seed", type=int, default=20261003); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.n, a.repeats, a.steps, a.probe = 300, 2, 30, 2000
    R = {"utc": utc_now(), "device": str(DEVICE), "settings": vars(a), "cells": {}}
    t0 = time.time()
    for tok in a.cells.split(","):
        enc, famname, s = tok.split(":"); s = float(s)
        fam = Family(FEATURE_DIRS[(enc, famname)]); n_classes = int(len(np.unique(fam.y)))
        rng = np.random.default_rng(a.seed); probe_idx = rng.choice(len(fam.y), size=a.probe, replace=False)
        maps = fit_maps(fam, probe_idx, a.seed, n_classes); cm = coarse_maps(maps)
        reps = [run_repeat(fam, maps, cm, probe_idx.tolist(), a.n, rng, strength=s, steps=a.steps, seed=a.seed + r, n_classes=n_classes) for r in range(a.repeats)]
        key = f"{enc}/{famname}/s{s:g}"
        R["cells"][key] = {"summary": summarise(reps), "repeats": reps, "probe_acc": {"h": maps["probe_acc_h"], "pca16": maps["probe_acc_pca16"]}}
        sm = R["cells"][key]["summary"]
        for obj in ("vcs", "js"):
            print(f"[{key} {obj}] " + " | ".join(f"{v}/{rd[8:]}: dJ {sm[f'{obj}/{v}/{rd}']['delta_J']['mean']:+.4f} r {sm[f'{obj}/{v}/{rd}']['r_BA']['mean']:+.4f} "
                                                  f"Rorth {sm[f'{obj}/{v}/{rd}']['R_orth']['mean']:+.4f} viol {sm[f'{obj}/{v}/{rd}']['nesting_violations']['mean']:.1f}"
                                                  for v in VARIANTS for rd in ("readout_sampled", "readout_exact")) + f" ({time.time() - t0:.0f}s)", flush=True)
        atomic_write_json(Path(a.out + ".partial.json"), R)
    R["seconds"] = time.time() - t0
    atomic_write_json(Path(a.out + ".json"), R)
    return 0


if __name__ == "__main__":
    sys.exit(main())
