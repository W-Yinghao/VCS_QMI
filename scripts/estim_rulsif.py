"""E4 — same-target comparison: closed-form VCS in a random-Fourier linear class vs RuLSIF (α = ½) on the Gaussian setting.

Relative density ratio r_½ = 2p/(p + q) = 1 + η, and the Pearson divergence PE_½ = ½ E_M[(r − 1)²] = S/2, so RuLSIF's own quantity is S up to the
constant 2.  (i) SAME FEATURES: both fitted by ridge least squares on identical random-Fourier features [φ(x, y), 1] with identical (σ, λ) →
Ŝ_VCS = 2ŵᵀd̂ − ŵᵀÂ_Mŵ from w = ½(Â_M + λI)⁻¹d̂ and Ŝ_RuLSIF = E_M[(r̂ − 1)²] from θ = (Ĥ + λI)⁻¹ĥ; agreement is reported per d.  (ii) each
method with its own model selection — RuLSIF: Gaussian-kernel basis on 100 centres, 5-fold CV over (σ, λ) on its LS objective; neural VCS: the
benchmark's JointMLP trained on the same n samples (K = 8 cyclic shifts, 3 000 steps, lr 5e-4), Ŝ = J on a held-out sample; truth S by MC.
    python scripts/estim_rulsif.py --out <prefix> [--dims 2,5,10,20,50] [--n 10000] [--mi 4]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vcs_estim.critics import JointMLP  # noqa: E402
from vcs_estim.data import Gaussian, rho_from_mi, truths_by_mc  # noqa: E402
from vcs_estim.estimators import estimate, scores  # noqa: E402


def rff(Z, W, b):
    return np.sqrt(2.0 / W.shape[1]) * np.cos(Z @ W + b)


def gauss_kernel(Z, C, sigma):
    d2 = (Z * Z).sum(1)[:, None] + (C * C).sum(1)[None, :] - 2 * Z @ C.T
    return np.exp(-d2 / (2 * sigma * sigma))


def fit_both_same_features(Zp, Zq, sigma, lam, D=500, rng=None):
    """Zp: joint pairs [n, 2d]; Zq: product pairs [n, 2d]; both methods on the same RFF + intercept features."""
    dim = Zp.shape[1]; W = rng.standard_normal((dim, D)) / sigma; b = rng.uniform(0, 2 * np.pi, D)
    Fp, Fq = np.c_[rff(Zp, W, b), np.ones(len(Zp))], np.c_[rff(Zq, W, b), np.ones(len(Zq))]
    A = 0.5 * (Fp.T @ Fp / len(Fp) + Fq.T @ Fq / len(Fq)); dvec = Fp.mean(0) - Fq.mean(0); I = np.eye(D + 1)
    w = 0.5 * np.linalg.solve(A + lam * I, dvec); S_vcs = 2 * w @ dvec - w @ A @ w
    theta = np.linalg.solve(A + lam * I, Fp.mean(0)); rp, rq = Fp @ theta, Fq @ theta
    S_rul = 0.5 * ((rp - 1) ** 2).mean() + 0.5 * ((rq - 1) ** 2).mean()
    return S_vcs, S_rul


def rulsif_cv(Zp, Zq, rng, n_centres=100, sigmas=(0.5, 1, 2, 4, 8), lams=(1e-3, 1e-2, 1e-1, 1), folds=5):
    C = Zp[rng.choice(len(Zp), n_centres, replace=False)]; best = None
    med = np.median(np.sqrt(((Zp[:500, None, :] - Zp[None, :500, :]) ** 2).sum(-1)))
    for sm in sigmas:
        sigma = sm * med; Kp, Kq = gauss_kernel(Zp, C, sigma), gauss_kernel(Zq, C, sigma)
        for lam in lams:
            errs = []
            for k in range(folds):
                tr = np.arange(len(Zp)) % folds != k; te = ~tr
                H = 0.5 * (Kp[tr].T @ Kp[tr] / tr.sum() + Kq[tr].T @ Kq[tr] / tr.sum()); h = Kp[tr].mean(0)
                th = np.linalg.solve(H + lam * np.eye(n_centres), h)
                Hte = 0.5 * (Kp[te].T @ Kp[te] / te.sum() + Kq[te].T @ Kq[te] / te.sum()); hte = Kp[te].mean(0)
                errs.append(0.5 * th @ Hte @ th - hte @ th)          # RuLSIF LS objective on the held-out fold
            e = float(np.mean(errs))
            if best is None or e < best[0]: best = (e, sigma, lam)
    _, sigma, lam = best; Kp, Kq = gauss_kernel(Zp, C, sigma), gauss_kernel(Zq, C, sigma)
    H = 0.5 * (Kp.T @ Kp / len(Kp) + Kq.T @ Kq / len(Kq)); th = np.linalg.solve(H + lam * np.eye(n_centres), Kp.mean(0))
    return th, C, sigma, lam


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--dims", default="2,5,10,20,50"); ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--mi", type=float, default=4.0); ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu"); dims = [int(x) for x in a.dims.split(",")]; seeds = [int(x) for x in a.seeds.split(",")]
    if a.smoke: dims, seeds, a.n, a.steps = [2, 5], [0], 2000, 300
    rows = []
    for d in dims:
        st = Gaussian("gaussian", d, 1, rho_from_mi(a.mi, d), a.mi); truth = truths_by_mc(st, 400_000, seed=7)["S"]
        for seed in seeds:
            rng = np.random.default_rng(seed); gen = torch.Generator().manual_seed(seed)
            x, y = st.sample(a.n, gen); xe, ye = st.sample(a.n, gen)
            Zp = torch.cat([x, y], 1).numpy(); Zq = torch.cat([x, y[torch.randperm(a.n, generator=gen)]], 1).numpy()
            Ze_p = torch.cat([xe, ye], 1).numpy(); Ze_q = torch.cat([xe, ye[torch.randperm(a.n, generator=gen)]], 1).numpy()
            med = float(np.median(np.sqrt(((Zp[:500, None, :] - Zp[None, :500, :]) ** 2).sum(-1))))
            same = {f"sigma{sm}_lam{lam}": fit_both_same_features(Zp, Zq, sm * med, lam, rng=rng) for sm in (1, 2) for lam in (1e-3, 1e-2)}
            th, C, sigma, lam = rulsif_cv(Zp, Zq, rng); rp, rq = gauss_kernel(Ze_p, C, sigma) @ th, gauss_kernel(Ze_q, C, sigma) @ th
            S_rul = 0.5 * ((rp - 1) ** 2).mean() + 0.5 * ((rq - 1) ** 2).mean()
            torch.manual_seed(seed); crit = JointMLP(d, d).to(device); opt = torch.optim.Adam(crit.parameters(), lr=5e-4); xd, yd = x.to(device), y.to(device)
            for step in range(a.steps):
                idx = torch.randint(0, a.n, (256,), generator=gen).to(device); fp, fn = scores(crit, xd[idx], yd[idx], "cyclic8", gen); loss, _ = estimate("vcs", fp, fn)
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                fp, fn = scores(crit, xe.to(device), ye.to(device), "cyclic8", gen); _, S_nn = estimate("vcs", fp, fn)
            rows.append({"d": d, "seed": seed, "truth_S": truth, "same_features": {k: {"S_vcs": float(v[0]), "S_rulsif": float(v[1]), "rel_diff": float(abs(v[0] - v[1]) / max(abs(v[1]), 1e-9))} for k, v in same.items()},
                         "rulsif_cv": {"S": float(S_rul), "sigma": float(sigma), "lam": float(lam), "abs_err": float(abs(S_rul - truth))}, "neural_vcs": {"S": float(S_nn), "abs_err": float(abs(float(S_nn) - truth))}})
            print(f"d={d} seed={seed} truth {truth:.4f} | same-features max rel diff {max(v['rel_diff'] for v in rows[-1]['same_features'].values()):.4f} | RuLSIF-CV {S_rul:.4f} (err {abs(S_rul-truth):.4f}) | neural VCS {float(S_nn):.4f} (err {abs(float(S_nn)-truth):.4f})", flush=True)
    json.dump(rows, open(a.out + ".json", "w"), indent=1)
    L = ["# E4 — closed-form VCS vs RuLSIF (α = ½), Gaussian setting at %g nats" % a.mi, "", "| d | truth S | same-feature max rel. diff (over σ, λ) | RuLSIF-CV |Ŝ − S| (mean over seeds) | neural VCS |Ŝ − S| |", "|---|---|---|---|---|"]
    for d in dims:
        rs = [r for r in rows if r["d"] == d]
        L.append(f"| {d} | {rs[0]['truth_S']:.4f} | {max(v['rel_diff'] for r in rs for v in r['same_features'].values()):.4f} | {np.mean([r['rulsif_cv']['abs_err'] for r in rs]):.4f} | {np.mean([r['neural_vcs']['abs_err'] for r in rs]):.4f} |")
    Path(a.out + ".md").write_text("\n".join(L) + "\n"); print("\n".join(L)); return 0


if __name__ == "__main__":
    sys.exit(main())
