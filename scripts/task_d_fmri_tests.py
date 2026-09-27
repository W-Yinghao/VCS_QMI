"""Task-level pre-check D-fMRI — step 2: motion-leakage detection with circular-shift-calibrated tests (P65).

    python scripts/task_d_fmri_tests.py --prep <dir from task_d_fmri_prep.py> --out <prefix> [--sizes 120,240,all] [--lags 0,1,2,4,8]
                                        [--alphas 0,0.5,1] [--shifts 200] [--min-shift 30] [--steps 300] [--delta 0.05] [--smoke]

Per subject, alpha, n (leading n TRs after lag alignment) and lag tau (z_t paired with N_{t-tau}): contiguous split — first 70 % = training
region (80 % fit / 20 % validation), last 30 % = EVAL block.  P-term = aligned pairs; Q-term = K = 4 circular shifts (>= min_shift) of N inside the
same region.  Critics on continuous n: closed-form linear class on phi(z, n) = [z*n, z, n, 1] (ridge + tanh scalar on validation) and a small
MLP on concat(z, n) (tanh output, early-stopped on validation); the reported critic has the better validation J.  Null: 200 circular shifts of
N (never i.i.d. permutations).  Tests at level delta: vcs_shift, vcs_hoeff, hsic_shift (Gaussian kernels on z and N), c2st (same MLP budget),
qcfc_corr (Pearson r of N with the global signal g, shift null).  Exact-null case: z of subject i with N of subject i+1 (mod S) at tau = 0.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from precheck_d_tests import DEVICE, gaussian_kernel, tau_hoeff  # noqa: E402

RIDGE = (1e-4, 1e-2, 1.0); CSCALE = (0.25, 0.5, 1.0, 2.0, 4.0)


def j_of(tp, tn):
    return float((tp - 0.5 * tp ** 2).mean() + (-tn - 0.5 * tn ** 2).mean())


def shifts_for(L, k, rng, min_shift):
    hi = L - min_shift
    if hi <= min_shift:  # region too short: fall back to any shift >= 1
        return rng.integers(1, L, size=k)
    return rng.integers(min_shift, hi + 1, size=k)


def rolled(n, s):
    return torch.roll(n, int(s), 0)


def phi(z, n):
    return torch.cat((z * n[:, None], z, n[:, None], torch.ones(len(z), 1, device=z.device)), 1).double()


class ClosedForm:
    """T = tanh(c * w^T phi), w = 0.5 (A_M + lam I)^-1 d; lam and c chosen on the validation region."""
    def __init__(self, zf, nf, negf, zv, nv, negv):
        best = (-1e9, None, None, None)
        Pf, Qf = phi(zf, nf), torch.cat([phi(zf, m) for m in negf])
        A = 0.5 * (Pf.T @ Pf / len(Pf) + Qf.T @ Qf / len(Qf)); d = Pf.mean(0) - Qf.mean(0); sc = float(A.diag().mean())
        Pv, Qv = phi(zv, nv), torch.cat([phi(zv, m) for m in negv])
        for lam in RIDGE:
            w = 0.5 * torch.linalg.solve(A + lam * sc * torch.eye(len(A), dtype=torch.float64, device=A.device), d)
            for c in CSCALE:
                j = j_of(torch.tanh(c * (Pv @ w)), torch.tanh(c * (Qv @ w)))
                if j > best[0]:
                    best = (j, w.clone(), c, lam)
        self.val_j, self.w, self.c, self.lam = best

    def __call__(self, z, n):
        return torch.tanh(self.c * (phi(z, n) @ self.w)).float()


class MLPCritic(nn.Module):
    def __init__(self, d, hidden=128):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d + 1, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, z, n):
        return torch.tanh(self.net(torch.cat((z, n[:, None]), 1)).squeeze(1))


def fit_mlp(zf, nf, negf, zv, nv, negv, steps, seed, lr=1e-3, wd=1e-2):
    torch.manual_seed(seed); m = MLPCritic(zf.shape[1]).to(zf.device); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    best, state = -1e9, {k: v.clone() for k, v in m.state_dict().items()}
    for step in range(1, steps + 1):
        opt.zero_grad(); tp = m(zf, nf); tn = torch.cat([m(zf, q) for q in negf]); loss = -((tp - 0.5 * tp ** 2).mean() + (-tn - 0.5 * tn ** 2).mean()); loss.backward(); opt.step()
        if step % 10 == 0 or step == steps:
            with torch.no_grad():
                j = j_of(m(zv, nv), torch.cat([m(zv, q) for q in negv]))
            if j > best:
                best, state = j, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(state); m.eval(); return m, best


class C2ST(nn.Module):
    def __init__(self, d, hidden=128):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d + 1, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, z, n):
        return self.net(torch.cat((z, n[:, None]), 1)).squeeze(1)


def fit_c2st(zf, nf, negf, zv, nv, negv, steps, seed, lr=1e-3, wd=1e-2):
    torch.manual_seed(seed); m = C2ST(zf.shape[1]).to(zf.device); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd); bce = nn.BCEWithLogitsLoss()
    def batch(z, n, negs):
        return torch.cat([z] * (1 + len(negs))), torch.cat([n] + list(negs)), torch.cat([torch.ones(len(z), device=z.device)] + [torch.zeros(len(z), device=z.device)] * len(negs))
    zt, nt, lt = batch(zf, nf, negf); zv_, nv_, lv_ = batch(zv, nv, negv); best, state = 1e9, {k: v.clone() for k, v in m.state_dict().items()}
    for step in range(1, steps + 1):
        opt.zero_grad(); loss = bce(m(zt, nt), lt); loss.backward(); opt.step()
        if step % 10 == 0 or step == steps:
            with torch.no_grad():
                l = float(bce(m(zv_, nv_), lv_))
            if l < best:
                best, state = l, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(state); m.eval(); return m


def c2st_acc(m, z, n, negs):
    with torch.no_grad():
        pos = (m(z, n) > 0).float().mean(); neg = torch.stack([(m(z, q) <= 0).float().mean() for q in negs]).mean()
    return float(0.5 * (pos + neg))


def hsic_cont(Kz, n):
    """HSIC_b with Gaussian kernels on z (precomputed) and on the scalar n (median heuristic)."""
    L = gaussian_kernel(n[:, None]); N = Kz.shape[0]; H = torch.eye(N, device=Kz.device) - 1.0 / N
    return float(torch.trace(Kz @ H @ L @ H) / N ** 2)


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def run_instance(z, n, g, *, rng, k_neg, n_shifts, min_shift, steps, delta, seed):
    """z [T,d], n [T], g [T] already aligned (lag applied) and truncated to the used length.  Returns per-test decisions and statistics."""
    T = len(n); n_eval = max(30, int(round(0.3 * T))); n_train = T - n_eval; n_fit = int(round(0.8 * n_train))
    zf, nf = z[:n_fit], n[:n_fit]; zv, nv = z[n_fit:n_train], n[n_fit:n_train]; ze, ne, ge = z[n_train:], n[n_train:], g[n_train:]
    negf = [rolled(nf, s) for s in shifts_for(len(nf), k_neg, rng, min_shift)]; negv = [rolled(nv, s) for s in shifts_for(len(nv), k_neg, rng, min_shift)]
    seval = shifts_for(len(ne), k_neg, rng, min_shift); nege = [rolled(ne, s) for s in seval]
    cf = ClosedForm(zf, nf, negf, zv, nv, negv); mlp, mlp_val = fit_mlp(zf, nf, negf, zv, nv, negv, steps, seed)
    picked = "closed" if cf.val_j >= mlp_val else "mlp"; crit = cf if picked == "closed" else mlp
    with torch.no_grad():
        def J_eval(nn_):
            return j_of(crit(ze, nn_), torch.cat([crit(ze, rolled(nn_, s)) for s in seval]))
        J = J_eval(ne)
        null_shifts = shifts_for(len(ne), n_shifts, rng, min_shift)
        null_J = np.array([J_eval(rolled(ne, s)) for s in null_shifts])
        Kz = gaussian_kernel(ze); h0 = hsic_cont(Kz, ne); null_h = np.array([hsic_cont(Kz, rolled(ne, s)) for s in null_shifts])
        r0 = pearson(ge, ne); null_r = np.array([abs(pearson(ge, rolled(ne, s))) for s in null_shifts])
    clf = fit_c2st(zf, nf, negf, zv, nv, negv, steps, seed + 1)
    acc = c2st_acc(clf, ze, ne, nege); null_acc = np.array([c2st_acc(clf, ze, rolled(ne, s), [rolled(rolled(ne, s), t) for t in seval]) for s in null_shifts])
    p = lambda null, obs: float((1 + (null >= obs).sum()) / (1 + len(null)))
    tau = tau_hoeff(n_eval, n_eval, delta)
    return {"n_used": int(T), "n_eval": int(n_eval), "picked": picked, "val_J": {"closed": cf.val_j, "mlp": mlp_val}, "J_eval": J, "tau": tau,
            "vcs_shift": {"p": p(null_J, J), "reject": p(null_J, J) <= delta}, "vcs_hoeff": {"reject": J - tau > 0, "margin": J - tau},
            "hsic_shift": {"stat": h0, "p": p(null_h, h0), "reject": p(null_h, h0) <= delta},
            "c2st": {"acc": acc, "p": p(null_acc, acc), "reject": p(null_acc, acc) <= delta},
            "qcfc_corr": {"r": r0, "p": p(null_r, abs(r0)), "reject": p(null_r, abs(r0)) <= delta}}


TESTS = ("vcs_shift", "vcs_hoeff", "hsic_shift", "c2st", "qcfc_corr")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="120,240,all"); ap.add_argument("--lags", default="0,1,2,4,8"); ap.add_argument("--alphas", default="0,0.5,1")
    ap.add_argument("--shifts", type=int, default=200); ap.add_argument("--min-shift", type=int, default=30); ap.add_argument("--k-neg", type=int, default=4)
    ap.add_argument("--steps", type=int, default=300); ap.add_argument("--delta", type=float, default=0.05); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--subjects", default=None); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    prep = Path(a.prep); man = json.load(open(prep / "manifest.json"))
    subs = sorted(s for s, q in man["subjects"].items() if "error" not in q and (prep / f"{s}.pt").exists())
    if a.subjects:
        subs = [s for s in a.subjects.split(",") if s in subs]
    sizes = [s.strip() for s in a.sizes.split(",")]; lags = [int(x) for x in a.lags.split(",")]; alphas = [x.strip() for x in a.alphas.split(",")]
    if a.smoke:
        subs, sizes, lags, a.shifts, a.steps = subs[:2], ["120"], [0, 1], 20, 50
    data = {s: torch.load(prep / f"{s}.pt", map_location="cpu", weights_only=False) for s in subs}
    akeys = {al: next(k for k in data[subs[0]]["z"] if float(k) == float(al)) for al in alphas}
    rng = np.random.default_rng(a.seed); t0 = time.time()
    results = {"settings": vars(a), "device": str(DEVICE), "subjects": subs, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cells": {}, "null": {}}

    def instance(z, n, g, sz, lag, seed):
        if lag > 0:
            z, n, g = z[lag:], n[:-lag], g[lag:]
        L = len(n) if sz == "all" else min(int(sz), len(n))
        return run_instance(z[:L].to(DEVICE), n[:L].to(DEVICE), g[:L].to(DEVICE), rng=rng, k_neg=a.k_neg, n_shifts=a.shifts, min_shift=a.min_shift, steps=a.steps, delta=a.delta, seed=seed)

    for al in alphas:
        for sz in sizes:
            for lag in lags:
                key = f"alpha{al}|n{sz}|lag{lag}"; per = {}
                for i, s in enumerate(subs):
                    d = data[s]; per[s] = instance(d["z"][akeys[al]], d["n"], d["g"][akeys[al]], sz, lag, seed=a.seed * 1000 + i)
                summ = {t: float(np.mean([per[s][t]["reject"] for s in subs])) for t in TESTS}
                summ["J_mean"] = float(np.mean([per[s]["J_eval"] for s in subs])); summ["J_sd"] = float(np.std([per[s]["J_eval"] for s in subs]))
                summ["qcfc_r_mean"] = float(np.mean([abs(per[s]["qcfc_corr"]["r"]) for s in subs])); summ["picked_closed_frac"] = float(np.mean([per[s]["picked"] == "closed" for s in subs]))
                results["cells"][key] = {"alpha": al, "n": sz, "lag": lag, "summary": summ, "per_subject": per}
                print(f"[{key}] " + " ".join(f"{t}={summ[t]:.2f}" for t in TESTS) + f" J={summ['J_mean']:.4f}±{summ['J_sd']:.4f} |r|={summ['qcfc_r_mean']:.3f} ({time.time() - t0:.0f}s)", flush=True)
            # exact-null case at lag 0: z of subject i with n (and g? no: g belongs to z's subject; the null pairs z_i, g_i with n_{i+1})
            key = f"alpha{al}|n{sz}|null"; per = {}
            for i, s in enumerate(subs):
                d, d2 = data[s], data[subs[(i + 1) % len(subs)]]; L = min(len(d["n"]), len(d2["n"]))
                per[s] = instance(d["z"][akeys[al]][:L], d2["n"][:L], d["g"][akeys[al]][:L], sz, 0, seed=a.seed * 7000 + i)
            summ = {t: float(np.mean([per[s][t]["reject"] for s in subs])) for t in TESTS}; summ["J_mean"] = float(np.mean([per[s]["J_eval"] for s in subs]))
            results["null"][key] = {"alpha": al, "n": sz, "summary": summ, "per_subject": per}
            print(f"[{key}] " + " ".join(f"{t}={summ[t]:.2f}" for t in TESTS) + f" J={summ['J_mean']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    Path(a.out + ".json").write_text(json.dumps(results, indent=1, default=float))
    S = len(subs)
    L = [f"# Task pre-check D-fMRI — motion leakage in cleaned resting-state BOLD (P65): {S} subjects, {a.shifts} circular shifts (≥ {a.min_shift} TRs), K = {a.k_neg} product shifts, δ = {a.delta} — {results['utc']}", "",
         "Rejection rate over subjects per test; Ĵ = held-out J of the picked critic on the last 30 % block; |r| = |Pearson(N, global signal)|.  alpha = motion-cleaning strength (0 none, 1 full 24-parameter regression).", "",
         "| alpha | n | lag (TR) | vcs_shift | vcs_hoeff | hsic_shift | c2st | qcfc_corr | Ĵ mean ± sd | |r| | closed-form picked |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, c in results["cells"].items():
        s = c["summary"]; L.append(f"| {c['alpha']} | {c['n']} | {c['lag']} | {s['vcs_shift']:.2f} | {s['vcs_hoeff']:.2f} | {s['hsic_shift']:.2f} | {s['c2st']:.2f} | {s['qcfc_corr']:.2f} | {s['J_mean']:.4f} ± {s['J_sd']:.4f} | {s['qcfc_r_mean']:.3f} | {s['picked_closed_frac']:.2f} |")
    L += ["", "## Exact-null calibration (z of subject i with N of subject i+1; lag 0)", "", "| alpha | n | vcs_shift | vcs_hoeff | hsic_shift | c2st | qcfc_corr | Ĵ mean |", "|---|---|---|---|---|---|---|---|"]
    for key, c in results["null"].items():
        s = c["summary"]; L.append(f"| {c['alpha']} | {c['n']} | {s['vcs_shift']:.2f} | {s['vcs_hoeff']:.2f} | {s['hsic_shift']:.2f} | {s['c2st']:.2f} | {s['qcfc_corr']:.2f} | {s['J_mean']:.4f} |")
    # monotonicity in alpha per subject (lag 0, n = all if present)
    szm = "all" if "all" in sizes else sizes[-1]
    if all(f"alpha{al}|n{szm}|lag0" in results["cells"] for al in alphas) and len(alphas) >= 2:
        mono = 0
        for s in subs:
            js = [results["cells"][f"alpha{al}|n{szm}|lag0"]["per_subject"][s]["J_eval"] for al in sorted(alphas, key=float)]
            mono += all(js[i] > js[i + 1] for i in range(len(js) - 1))
        results["monotone_fraction"] = mono / S
        L += ["", f"Strictly decreasing Ĵ with alpha (lag 0, n = {szm}): {mono} / {S} subjects = {mono / S:.2f}."]
        Path(a.out + ".json").write_text(json.dumps(results, indent=1, default=float))
    Path(a.out + ".md").write_text("\n".join(L) + "\n"); print("->", a.out + ".md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
