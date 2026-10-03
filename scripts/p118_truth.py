"""P118 §6.1 — truth-compression check of nested critics (v5 NEXT-I-NEST) on a Gaussian conditional-binary oracle (scripts/p118_common.py).

    python scripts/p118_truth.py --out reports/P118_truth [--conds null,weak,moderate] [--repeats 5] [--smoke]

Chain B (64) ⊇ M (8: only irrelevant coordinates dropped, true increment 0) ⊇ A (4: half the relevant block dropped, true increment S_M − S_A > 0).
Per repeat (fresh FIT / EVAL / POOL draws, n each) and objective (VCS −J, matched JS), the fitting variants of P109 I2 on the same draws:
    indep_sampled   every set fitted independently, sampled within-class POOL negatives (P102 baseline)
    indep_exact     independent, Q term enumerated over N ∈ {0, 1} with the known P(N | Y)
    nested_sampled  A independent (sampled), M = frozen A + zero-initialised residual, B = frozen M + residual (sampled negatives)
    nested_exact    the same with exact enumeration
    zero            constant-zero critic on every set (control: perfect consistency residuals, wrong increments)
Each variant is read out with the sampled and with the exact Q term.  Endpoints: increment errors (step B→M vs 0, step M→A vs the oracle increment,
total B→A), posterior MSE of B, M and A against the exact eta of each set; auxiliary R_orth, ordering violations (J_fine < J_coarse), null floor.
Nothing on EVAL is clipped, made monotone or orthogonalised; EVAL selects nothing.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
import p118_common as C  # noqa: E402
from precheck_d_tests import within_class_pool  # noqa: E402
from vcs_measure import nested as NS  # noqa: E402
from vcs_ssl.utils import atomic_write_json, utc_now  # noqa: E402

CONDS = {"null": 0.0, "weak": 1.0, "moderate": 2.0}          # pre-named (oracle S_M ≈ 0, 0.036, 0.109; true M→A increment 0, 0.017, 0.044)
VARIANTS = ("indep_sampled", "indep_exact", "nested_sampled", "nested_exact", "zero")
CMAP = {("M", "A"): lambda z: z[:, :C.KEEP_A], ("B", "M"): lambda z: z[:, :C.R]}
DEVICE = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")


def feats(z):
    return {"B": z, "M": z[:, :C.R], "A": z[:, :C.KEEP_A]}


def run_repeat(delta, oracle, n, rng, mu, *, steps, seed):
    zf, yf, nf = C.sample(n, delta, rng, mu); ze, ye, ne = C.sample(n, delta, rng, mu); _, yp, np_ = C.sample(n, delta, rng, mu)
    nf_neg, ne_neg = within_class_pool(yf, yp, np_, rng), within_class_pool(ye, yp, np_, rng)
    t = lambda a, dt=None: torch.as_tensor(a, dtype=dt).to(DEVICE)
    Ff = {k: t(v, torch.float32) for k, v in feats(zf).items()}; Fe = {k: t(v, torch.float32) for k, v in feats(ze).items()}
    nf_t, yf_t, ne_t, ye_t = t(nf), t(yf), t(ne), t(ye)
    neg_s = {"n_neg": t(nf_neg)}; neg_x = {"p1": t(C.p_n1(yf), torch.float32)}; nege = {"n_neg": t(ne_neg)}; p1e = t(C.p_n1(ye), torch.float32)
    eta = {}
    for k, d in C.SETS.items():       # exact eta on EVAL: P units, sampled partner, enumerated n' ∈ {0, 1}
        eta[k] = (C.eta_set(ze, ne, ye, delta, mu, d), C.eta_set(ze, ne_neg, ye, delta, mu, d)[:, None],
                  np.stack([C.eta_set(ze, np.zeros_like(ne), ye, delta, mu, d), C.eta_set(ze, np.ones_like(ne), ye, delta, mu, d)], 1))
    true_inc = {"B->M": oracle["B"]["S"] - oracle["M"]["S"], "M->A": oracle["M"]["S"] - oracle["A"]["S"], "B->A": oracle["B"]["S"] - oracle["A"]["S"]}
    res = {"n": n, "delta": delta, "true_increment": true_inc, "per_objective": {}}
    for obj in ("vcs", "js"):
        res["per_objective"][obj] = {}; indep = {}
        for variant in VARIANTS:
            crit, info = {}, {}
            if variant == "zero":
                ro = {k: C.zero_readouts(n) for k in C.CHAIN}
                ro = {k: (v[0], v[1], (v[2][0], np.stack([1 - p1e.cpu().numpy(), p1e.cpu().numpy()], 1))) for k, v in ro.items()}
            else:
                neg = neg_x if variant.endswith("exact") else neg_s
                if variant.startswith("indep"):
                    for j, k in enumerate(C.CHAIN):
                        crit[k], pick, vals, bs = NS.fit_picked(Ff[k], nf_t, yf_t, neg, objective=obj, n_classes=C.N_CLASSES, steps=steps, seed=seed * 100 + j)
                        info[k] = {"picked": pick, "val_J": vals, "best_step": bs}
                    indep[variant] = crit
                else:
                    base = indep["indep_" + variant.split("_")[1]]; crit["A"] = base["A"]
                    for j, (fine, coarse) in enumerate((("M", "A"), ("B", "M"))):
                        crit[fine], pick, vals, bs = NS.fit_picked(Ff[fine], nf_t, yf_t, neg, objective=obj, n_classes=C.N_CLASSES, steps=steps,
                                                                   seed=seed * 100 + 50 + j, coarse=crit[coarse], cmap=CMAP[(fine, coarse)])
                        info[f"{fine}<-{coarse}"] = {"picked": pick, "val_J": vals, "best_step": bs}
                ro = {k: NS.readouts(crit[k], Fe[k], ne_t, ye_t, nege, p1e) for k in C.CHAIN}
            out = {"critics": info}
            for rd, j in (("readout_sampled", 1), ("readout_exact", 2)):
                wq = ro["B"][j][1]
                chain = [(ro[k][0], ro[k][j][0]) for k in C.CHAIN]; ch = NS.chain_w(chain, C.CHAIN, wq)
                post = {k: C.posterior_mse(ro[k][0], ro[k][j][0], wq, eta[k][0], eta[k][j]) for k in C.CHAIN}
                tot = NS.increment_w(chain[0], chain[-1], wq)
                inc = {s: ch["steps"][s]["delta_J"] for s in ("B->M", "M->A")} | {"B->A": tot["delta_J"]}
                out[rd] = {"J": ch["J"], "increment": inc, "increment_error": {s: inc[s] - true_inc[s] for s in inc}, "D_T": {s: ch["steps"][s]["D_T"] for s in ("B->M", "M->A")},
                           "R_orth": ch["R_orth"], "ordering_violations": ch["nesting_violations"], "posterior_mse": post}
            res["per_objective"][obj][variant] = out
    return res


def summarise(reps):
    out = {}
    for obj in ("vcs", "js"):
        for v in VARIANTS:
            for rd in ("readout_sampled", "readout_exact"):
                g = lambda f: [f(r["per_objective"][obj][v][rd]) for r in reps]
                vals = {f"inc_err_{s}": g(lambda x, s=s: x["increment_error"][s]) for s in ("B->M", "M->A", "B->A")}
                vals |= {f"inc_{s}": g(lambda x, s=s: x["increment"][s]) for s in ("B->M", "M->A", "B->A")}
                vals |= {f"post_{k}": g(lambda x, k=k: x["posterior_mse"][k]) for k in C.CHAIN}
                vals |= {"R_orth": g(lambda x: x["R_orth"]), "ordering_violations": g(lambda x: x["ordering_violations"])}
                vals |= {f"J_{k}": g(lambda x, k=k: x["J"][k]) for k in C.CHAIN}
                s = {k: {"mean": float(np.mean(x)), "sd": float(np.std(x, ddof=1)) if len(x) > 1 else None} for k, x in vals.items()}
                for st in ("B->M", "M->A", "B->A"):
                    s[f"inc_rmse_{st}"] = float(np.sqrt(np.mean(np.square(vals[f"inc_err_{st}"]))))
                out[f"{obj}/{v}/{rd}"] = s
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--conds", default="null,weak,moderate"); ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--repeats", type=int, default=5); ap.add_argument("--steps", type=int, default=300); ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.n, a.repeats, a.steps = 400, 2, 30
    mu = C.class_means(); R = {"utc": utc_now(), "device": str(DEVICE), "settings": vars(a), "generator": {"D": C.D, "R": C.R, "KEEP_A": C.KEEP_A}, "cells": {}}
    t0 = time.time()
    for cname in a.conds.split(","):
        delta = CONDS[cname]; oracle = C.oracle_S(delta, n=50_000 if a.smoke else 400_000); rng = np.random.default_rng([a.seed, int(delta * 100)])
        reps = [run_repeat(delta, oracle, a.n, rng, mu, steps=a.steps, seed=a.seed + r) for r in range(a.repeats)]
        R["cells"][cname] = {"delta": delta, "oracle": oracle, "summary": summarise(reps), "repeats": reps}
        sm = R["cells"][cname]["summary"]
        for obj in ("vcs", "js"):
            print(f"[{cname} d={delta} true M->A {oracle['M']['S'] - oracle['A']['S']:.4f} {obj}] " + " | ".join(
                f"{v}/{rd[8:11]}: errMA {sm[f'{obj}/{v}/{rd}']['inc_err_M->A']['mean']:+.4f} errBM {sm[f'{obj}/{v}/{rd}']['inc_err_B->M']['mean']:+.4f} "
                f"postB {sm[f'{obj}/{v}/{rd}']['post_B']['mean']:.4f} Rorth {sm[f'{obj}/{v}/{rd}']['R_orth']['mean']:+.4f}"
                for v in VARIANTS for rd in ("readout_exact",)) + f" ({time.time() - t0:.0f}s)", flush=True)
        atomic_write_json(Path(a.out + ".partial.json"), R)
    R["seconds"] = time.time() - t0
    atomic_write_json(Path(a.out + ".json"), R)
    return 0


if __name__ == "__main__":
    sys.exit(main())
