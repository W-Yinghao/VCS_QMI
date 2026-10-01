"""P105 — T1 critic-class ablation: does the VCS conditional test's advantage over the JS statistic come from the VCS objective, or from the closed-form
(exactly solved) linear critic that only VCS had in T1?

Every repeat re-uses the T1 construction unchanged (`cond_test_t1.draw_sample`: fresh N | Y, disjoint FIT / EVAL / POOL of n items, within-class POOL
negatives, B shared within-class permutations of N on EVAL).  On the same draws it fits six critics and runs a permutation test with each, plus the
VAL-picked combinations:

  VCS objective   vcs_lin, vcs_mlp (AdamW, 300 steps, early stop on VAL J — T1's fitter), vcs_closed (T1's exact ridge solution in the linear class
                  phi(z, n) = [z (2n − 1), 1], ridge and output scale on VAL J)
  JS objective    js_lin, js_mlp (T1's fitter), js_exact (NEW: the balanced logistic / JS objective solved exactly — convex, L-BFGS to convergence —
                  in the same class phi, ridge on the same grid chosen on VAL JS)
  picked tests    vcs3 = VAL-pick of {lin, mlp, closed} (= T1's vcs_perm), vcs2 = VAL-pick of {lin, mlp} (the ablation),
                  js2 = VAL-pick of {lin, mlp} (= T1's js_perm), js3 = VAL-pick of {lin, mlp, exact}

    python scripts/cond_test_t1_ablation.py --features <P45 dir> --family colour --cells 'colour_s0.05:1000;colour_s0.2:2000' --repeats 100 --seed 201 --out reports/P105_...
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cond_test_t1 import SOFTPLUS, Family, class_blocks, draw_sample, fit_js_critic, js_value  # noqa: E402
from precheck_d_tests import DEVICE, closed_form_critic, fit_vcs_critic, j_stat, within_class_pool  # noqa: E402
from vcs_ssl.utils import atomic_write_json, utc_now  # noqa: E402

FORCED = ("vcs_lin", "vcs_mlp", "vcs_closed", "js_lin", "js_mlp", "js_exact")
PICKED = ("vcs3", "vcs2", "js2", "js3")
TESTS = FORCED + PICKED


def _phi(zz, nn_):
    return torch.cat((zz * (2 * nn_.float() - 1)[:, None], torch.ones(len(zz), 1, device=zz.device)), 1).double()


def exact_js_critic(z, n, n_neg, seed, lams=(1e-3, 1e-2, 1e-1, 1.0), max_iter=200):
    """JS / balanced-logistic objective, linear class phi (the class of `closed_form_critic`), solved exactly: L-BFGS on the convex ridge-penalised
    loss E_P softplus(−f) + E_Q softplus(f) + lam·sc·|w|² / 2 (sc = mean diag of the second-moment matrix, as in the closed form), lam chosen on VAL JS."""
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(z), generator=g); nv = max(8, int(0.2 * len(z))); vi, ti = perm[:nv], perm[nv:]
    pp, pn = _phi(z[ti], n[ti]), _phi(z[ti], n_neg[ti]); vp, vn = _phi(z[vi], n[vi]), _phi(z[vi], n_neg[vi])
    sc = float(0.5 * ((pp ** 2).mean(0) + (pn ** 2).mean(0)).mean())
    best = (-9.0, None, None)
    for lam in lams:
        w = torch.zeros(pp.shape[1], dtype=torch.float64, device=pp.device, requires_grad=True)
        opt = torch.optim.LBFGS([w], lr=1.0, max_iter=max_iter, tolerance_grad=1e-10, tolerance_change=1e-12, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = SOFTPLUS(-(pp @ w)).mean() + SOFTPLUS(pn @ w).mean() + 0.5 * lam * sc * (w ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            jv = js_value((vp @ w).float(), (vn @ w).float())
        if jv > best[0]:
            best = (jv, w.detach().clone(), lam)

    class EX:
        def __init__(self, w):
            self.w = w
        def __call__(self, zz, nn_):
            return (_phi(zz, nn_) @ self.w).float()
    m = EX(best[1]); m.val_J = best[0]; m.lam = best[2]
    return m


def run_repeat(fam: Family, n, rng, *, mode, strength, delta, perms, steps, seed):
    used = set()
    Hf, Yf, Nf, _ = draw_sample(fam, n, rng, mode, strength, used)
    He, Ye, Ne, _ = draw_sample(fam, n, rng, mode, strength, used)
    Hp, Yp, Np, _ = draw_sample(fam, n, rng, mode, strength, used)
    mu, sd = Hf.mean(0), Hf.std(0) + 1e-6
    zf, ze = ((Hf - mu) / sd).to(DEVICE), ((He - mu) / sd).to(DEVICE)
    nf, ne = torch.as_tensor(Nf).to(DEVICE), torch.as_tensor(Ne).to(DEVICE)
    nf_neg = torch.as_tensor(within_class_pool(Yf, Yp, Np, rng)).to(DEVICE); ne_neg = torch.as_tensor(within_class_pool(Ye, Yp, Np, rng)).to(DEVICE)
    perm_N = []
    for _ in range(perms):
        pn = Ne.copy()
        for ii in class_blocks(Ye):
            pn[ii] = pn[ii][rng.permutation(len(ii))]
        perm_N.append(torch.as_tensor(pn).to(DEVICE))
    pval = lambda null, obs: float((1 + (np.asarray(null) >= obs).sum()) / (1 + len(null)))
    crit = {"vcs_lin": fit_vcs_critic(zf, nf, nf_neg, steps, seed, kind="linear"), "vcs_mlp": fit_vcs_critic(zf, nf, nf_neg, steps, seed, kind="mlp"),
            "vcs_closed": closed_form_critic(zf, nf, nf_neg, seed),
            "js_lin": fit_js_critic(zf, nf, nf_neg, steps, seed + 3, kind="linear"), "js_mlp": fit_js_critic(zf, nf, nf_neg, steps, seed + 3, kind="mlp"),
            "js_exact": exact_js_critic(zf, nf, nf_neg, seed + 3)}
    res = {"n": int(n), "n1_frac_eval": float(Ne.mean())}
    with torch.no_grad():
        for name, cr in crit.items():
            if name.startswith("vcs"):
                tn = torch.tanh(cr(ze, ne_neg)); obs = j_stat(torch.tanh(cr(ze, ne)), tn); null = [j_stat(torch.tanh(cr(ze, pn)), tn) for pn in perm_N]
            else:
                fneg = cr(ze, ne_neg); obs = js_value(cr(ze, ne), fneg); null = [js_value(cr(ze, pn), fneg) for pn in perm_N]
            p = pval(null, obs)
            res[name] = {"stat": float(obs), "p": p, "reject": p <= delta, "val": float(cr.val_J)}
    for name, pool in (("vcs3", ("vcs_lin", "vcs_mlp", "vcs_closed")), ("vcs2", ("vcs_lin", "vcs_mlp")),
                       ("js2", ("js_lin", "js_mlp")), ("js3", ("js_lin", "js_mlp", "js_exact"))):
        pick = max(pool, key=lambda k: res[k]["val"])
        res[name] = dict(res[pick], picked=pick)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True); ap.add_argument("--family", required=True, choices=("colour", "blur"))
    ap.add_argument("--cells", required=True, help="';'-separated case:n, case as in T1 (colour_s0.2, colour_null_label_only, colour_null_all_planted0.2)")
    ap.add_argument("--repeats", type=int, default=100); ap.add_argument("--perms", type=int, default=200); ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--delta", type=float, default=0.05); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))); print("device:", DEVICE, flush=True)
    fam = Family(Path(a.features))
    assert fam.name == a.family, (fam.name, a.family)
    R, perms, steps = (3, 20, 60) if a.smoke else (a.repeats, a.perms, a.steps)
    cells = []
    for tok in a.cells.split(";"):
        case, n = tok.split(":"); rest = case[len(a.family) + 1:]
        if rest.startswith("s"):
            mode, s = "planted", float(rest[1:])
        elif rest == "null_label_only":
            mode, s = "null_label_only", 0.0
        elif rest.startswith("null_all_planted"):
            mode, s = "null_all_planted", float(rest[len("null_all_planted"):])
        else:
            raise ValueError(tok)
        if mode != "null_label_only":
            assert s in fam.planted, f"strength {s} not in {fam.dir}"
        cells.append((case, mode, s, 200 if a.smoke else int(n)))
    rng = np.random.default_rng(a.seed)
    out = {"settings": vars(a), "device": str(DEVICE), "utc": utc_now(), "family_dir": str(fam.dir), "tests": TESTS, "cases": {}}
    t0 = time.time()
    for case, mode, s, n in cells:
        inst = [run_repeat(fam, n, rng, mode=mode, strength=s, delta=a.delta, perms=perms, steps=steps, seed=a.seed * 1000 + r) for r in range(R)]
        summ = {t: {"power": float(np.mean([i[t]["reject"] for i in inst]))} for t in TESTS}
        for t in PICKED:
            summ[t]["picked"] = {k: int(v) for k, v in zip(*np.unique([i[t]["picked"] for i in inst], return_counts=True))}
        c = out["cases"].setdefault(case, {"mode": mode, "strength": s, "by_n": {}})
        c["by_n"][str(n)] = {"repeats": R, "summary": summ, "instances": inst}
        print(f"[{case} n={n} R={R}] " + " ".join(f"{t}={summ[t]['power']:.2f}" for t in TESTS) + f" ({time.time() - t0:.0f}s)", flush=True)
        atomic_write_json(Path(a.out + ".partial.json"), out)
    atomic_write_json(Path(a.out + ".json"), out)
    print(f"wrote {a.out}.json ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
