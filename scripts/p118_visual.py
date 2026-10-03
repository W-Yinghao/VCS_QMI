"""P118 §6.2 — visual confirmation of nested critics on the paired I1 feature caches (v5 NEXT-I-NEST).  No encoder training.

    python scripts/p118_visual.py --runs P107_AP3_views4_800ep_seed1,P41_simclr_views4_800ep_seed1 --out reports/P118_visual [--smoke]

Per encoder: the P102 information chain built from h of the cached P109 I1 features — h ⊇ pca64 ⊇ pca16 ⊇ logit16 and the main pair h vs logit_h —
with standardisation / PCA / linear probes fitted once on a fixed PROBE set of 4 000 FIT-split base images (same ids for every encoder; never drawn
afterwards).  Per repeat: FIT n from the remaining FIT split, EVAL n from the EVAL split, POOL from the VAL split (disjoint by base image ID);
N | Y ~ Bernoulli(½ ± 0.3) drawn afresh; the observed h is the planted version iff N = 1 ('planted') or clean for all ('null_label_only').
Variants and readouts exactly as P109 I2 (indep_sampled, indep_exact, nested_sampled, nested_exact) plus the constant-zero control.  Cells are
pre-named in the prereg (colour s 0.1, blur s 0.25, null) and not chosen on EVAL.  Units: encoder seed (reported separately), repeats = fresh N /
sample draws; base images are the sampling unit inside a repeat.
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
from p109_i2_pilot import coarse_maps  # noqa: E402
from precheck_d_tests import DEVICE, within_class_pool  # noqa: E402
from t_line_increments import CHAIN, SETS, feature_sets, fit_maps  # noqa: E402
from vcs_measure import nested as NS  # noqa: E402
from vcs_measure.audit import p_n1  # noqa: E402
from vcs_ssl.utils import atomic_write_json, utc_now  # noqa: E402

FEATURES = Path("/home/infres/yinwang/CS_QMI/outputs/P109_I1_features")
CELLS = {"colour_s0.1": ("planted", "colour_s0.1"), "blur_s0.25": ("planted", "blur_s0.25"), "null": ("null_label_only", None)}
VARIANTS = ("indep_sampled", "indep_exact", "nested_sampled", "nested_exact", "zero")
PROBE_N, PROBE_SEED = 4000, 118


class Store:
    def __init__(self, run: str):
        d = torch.load(FEATURES / run / "features.pt", map_location="cpu", weights_only=False)
        self.run, self.y, self.part, self.ids = run, d["y"].numpy(), d["part"].numpy(), d["ids"].numpy()
        self.H = {v: d["features"][v]["h"].float() for v in d["features"]}
        self.idx = {k: np.where(self.part == i)[0] for i, k in enumerate(("fit", "val", "eval"))}
        probe = np.random.default_rng(PROBE_SEED).choice(self.idx["fit"], size=PROBE_N, replace=False)
        self.probe, self.fit_pool = probe, np.setdiff1d(self.idx["fit"], probe)
        self.h0, self.ids_probe = self.H["clean"], self.ids[probe]      # fit_maps reads fam.h0 / fam.y


def draw(S: Store, rows_from: np.ndarray, n: int, rng, mode: str, version):
    rows = rng.choice(rows_from, size=n, replace=False); y = S.y[rows]; nn_ = (rng.random(n) < p_n1(y)).astype(np.int64)
    c = S.H["clean"][torch.as_tensor(rows)]
    H = torch.where(torch.as_tensor(nn_ == 1)[:, None], S.H[version][torch.as_tensor(rows)], c) if mode == "planted" else c
    return H, y, nn_, rows


def zero_ro(n, p1):
    return np.zeros(n), (np.zeros((n, 1)), np.ones((n, 1))), (np.zeros((n, 2)), np.stack([1 - p1, p1], 1))


def run_repeat(S, maps, cm, n, rng, mode, version, *, steps, seed, n_classes):
    Hf, Yf, Nf, rf = draw(S, S.fit_pool, n, rng, mode, version); He, Ye, Ne, re_ = draw(S, S.idx["eval"], n, rng, mode, version)
    _, Yp, Np, rp = draw(S, S.idx["val"], min(n, len(S.idx["val"])), rng, mode, version)
    assert not (set(rf) & set(re_)) and not (set(rf) & set(rp)) and not (set(re_) & set(rp)) and not (set(rf) & set(S.probe))
    with torch.no_grad():
        Ff = {k: v.to(DEVICE) for k, v in feature_sets(Hf, maps).items()}; Fe = {k: v.to(DEVICE) for k, v in feature_sets(He, maps).items()}
    t = lambda a: torch.as_tensor(a).to(DEVICE)
    nf, yf, ne, ye = t(Nf), t(Yf), t(Ne), t(Ye)
    negf = {"n_neg": t(within_class_pool(Yf, Yp, Np, rng))}; nege = {"n_neg": t(within_class_pool(Ye, Yp, Np, rng))}
    exf = {"p1": torch.as_tensor(p_n1(Yf), dtype=torch.float32).to(DEVICE)}; p1e_np = p_n1(Ye); p1e = torch.as_tensor(p1e_np, dtype=torch.float32).to(DEVICE)
    res = {"n": n, "mode": mode, "version": version, "n1_frac_eval": float(Ne.mean()), "per_objective": {}}
    for obj in ("vcs", "js"):
        res["per_objective"][obj] = {}; indep = {}
        for variant in VARIANTS:
            info = {}
            if variant == "zero":
                ro = {nm: zero_ro(n, p1e_np) for nm in CHAIN + ["main_B", "main_A"]}
            else:
                neg = exf if variant.endswith("exact") else negf
                if variant.startswith("indep"):
                    crit = {}
                    for k, nm in enumerate(SETS):
                        crit[nm], pick, vals, bs = NS.fit_picked(Ff[nm], nf, yf, neg, objective=obj, n_classes=n_classes, steps=steps, seed=seed * 100 + k)
                        info[nm] = {"picked": pick, "val_J": vals, "best_step": bs}
                    indep[variant] = crit; chain_crit = {c: crit[c] for c in CHAIN}; main_B, main_A = crit["h"], crit["logit_h"]
                else:
                    base = indep["indep_" + variant.split("_")[1]]; chain_crit = {"logit16": base["logit16"]}
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
            for rd, j in (("readout_sampled", 1), ("readout_exact", 2)):
                wq = ro["main_B"][j][1]
                mB, mA = (ro["main_B"][0], ro["main_B"][j][0]), (ro["main_A"][0], ro["main_A"][j][0])
                out[rd] = {"main_pair": NS.increment_w(mB, mA, wq), "chain": NS.chain_w([(ro[c][0], ro[c][j][0]) for c in CHAIN], CHAIN, wq)}
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
                for st in ("h->pca64", "pca64->pca16", "pca16->logit16"):
                    vals[f"step_{st}"] = [r["per_objective"][obj][v][rd]["chain"]["steps"][st]["delta_J"] for r in reps]
                out[f"{obj}/{v}/{rd}"] = {k: {"mean": float(np.mean(x)), "sd_refit": float(np.std(x, ddof=1)) if len(x) > 1 else None} for k, x in vals.items()}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True); ap.add_argument("--out", required=True); ap.add_argument("--cells", default="colour_s0.1,blur_s0.25,null")
    ap.add_argument("--n", type=int, default=3000); ap.add_argument("--repeats", type=int, default=5); ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261004); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.n, a.repeats, a.steps = 400, 2, 30
    R = {"utc": utc_now(), "device": str(DEVICE), "settings": vars(a), "runs": {}}; t0 = time.time()
    for run in a.runs.split(","):
        dst = Path(f"{a.out}_{run}.json")
        if dst.is_file():
            print(f"skip {run} (exists)", flush=True); continue
        S = Store(run); n_classes = int(len(np.unique(S.y)))
        maps = fit_maps(S, S.probe, a.seed, n_classes); cm = coarse_maps(maps); RR = {"run": run, "probe_acc": {"h": maps["probe_acc_h"], "pca16": maps["probe_acc_pca16"]}, "cells": {}}
        for cname in a.cells.split(","):
            mode, version = CELLS[cname]; rng = np.random.default_rng([a.seed, list(CELLS).index(cname)])
            reps = [run_repeat(S, maps, cm, a.n, rng, mode, version, steps=a.steps, seed=a.seed + r, n_classes=n_classes) for r in range(a.repeats)]
            RR["cells"][cname] = {"summary": summarise(reps), "repeats": reps}; sm = RR["cells"][cname]["summary"]
            for obj in ("vcs", "js"):
                print(f"[{run} {cname} {obj}] " + " | ".join(f"{v}/{rd[8:11]}: dJ {sm[f'{obj}/{v}/{rd}']['delta_J']['mean']:+.4f} Rorth {sm[f'{obj}/{v}/{rd}']['R_orth']['mean']:+.4f} "
                                                         f"viol {sm[f'{obj}/{v}/{rd}']['nesting_violations']['mean']:.1f}" for v in VARIANTS for rd in ("readout_exact",))
                      + f" ({time.time() - t0:.0f}s)", flush=True)
            atomic_write_json(Path(f"{a.out}_{run}.partial.json"), RR)
        RR["seconds"] = time.time() - t0; atomic_write_json(dst, RR); R["runs"][run] = str(dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
