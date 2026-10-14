"""V7X-11 / V7X-12 controlled anchors (reports/V7/V7X_11_ANCHOR_PROTOCOL.yaml): analytic truths, the classical / kernel / neural estimators and the
MMD / HSIC tests on fresh data replicates.  TRUTH samples come from a separate random stream and are never used for selection (CAL only).
    est --anchor A1|A2|A3 --out DIR [--reps 10] [--sizes 256 4096] [--cells i,j]     estimation panel rows (A0 / A1 / B0)
    tests --anchor A1|A2|A3 --out DIR [--reps 200]                                   test panel rows (B1)"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import x_estimators as XE  # noqa: E402

ANCHORS = {"A1": [{"d": d, "delta": a} for d in (2, 20) for a in (0.0, 0.5, 2.0)],
           "A2": [{"d": d, "rho": r} for d in (2, 20) for r in (0.0, 0.5, 0.9)],
           "A3": [{"k": k, "rho": 0.8} for k in (1, 5)]}
CS_MULTS = (0.125, 0.25, 0.5, 1.0, 2.0); KDE_EXT = (0.0625, 4.0); KM = (0.25, 0.5, 1.0, 2.0, 4.0); RIDGES = (1e-6, 1e-4, 1e-2, 1.0)
N_EVAL, N_TRUTH, TEST_N, PERMS = 4096, 200000, 256, 200


def dim(anchor, c):
    return c["d"] if anchor in ("A1", "A2") else 2 * c["k"]


def sample(anchor, c, n, rng, which):
    d = dim(anchor, c)
    if which == "Q":
        return rng.normal(size=(n, d))
    if anchor == "A1":
        x = rng.normal(size=(n, d)); x[:, 0] += c["delta"]; return x
    k = d // 2; z1, z2 = rng.normal(size=(n, k)), rng.normal(size=(n, k))
    r = c["rho"] if anchor == "A2" else c["rho"] * rng.choice([-1.0, 1.0], size=(n, k))
    return np.concatenate([z1, r * z1 + np.sqrt(1 - np.square(r)) * z2], 1)


def log_ratio(anchor, c, x):
    """Analytic log p(x) / q(x)."""
    if anchor == "A1":
        return c["delta"] * x[:, 0] - c["delta"] ** 2 / 2
    k = x.shape[1] // 2; u, v = x[:, :k], x[:, k:]; r = c["rho"]

    def lg(rr):
        return -0.5 * np.log(1 - rr ** 2) - (u ** 2 - 2 * rr * u * v + v ** 2) / (2 * (1 - rr ** 2)) + (u ** 2 + v ** 2) / 2
    if anchor == "A2":
        return lg(r).sum(1)
    return np.logaddexp(lg(r), lg(-r)).sum(1) - k * math.log(2)


def truths(anchor, c, rng):
    """cs (squared ratio), KL, chi2 (Pearson), PE_half (relative Pearson, alpha 1/2) and S; closed forms where they exist, else Monte-Carlo."""
    xp, xq = sample(anchor, c, N_TRUTH, rng, "P"), sample(anchor, c, N_TRUTH, rng, "Q"); lp, lq = log_ratio(anchor, c, xp), log_ratio(anchor, c, xq)
    S = 0.5 * (np.tanh(lp / 2) ** 2).mean() + 0.5 * (np.tanh(lq / 2) ** 2).mean()
    up, uq = 2 / (1 + np.exp(-lp)), 2 / (1 + np.exp(-lq))                    # u* = p / (p/2 + q/2)
    pe_half = 0.5 * (0.5 * ((up - 1) ** 2).mean() + 0.5 * ((uq - 1) ** 2).mean())
    I2 = np.eye(2)
    if anchor == "A1":
        cs = XE.cs_mixture([1], [np.array([c["delta"]])], [np.eye(1)], [1], [np.zeros(1)], [np.eye(1)]); kl = c["delta"] ** 2 / 2; chi2 = math.expm1(c["delta"] ** 2)
    elif anchor == "A2":
        k = c["d"] // 2; Sg = np.array([[1, c["rho"]], [c["rho"], 1]])
        cs = k * XE.cs_mixture([1], [np.zeros(2)], [Sg], [1], [np.zeros(2)], [I2]); kl = -k / 2 * math.log(1 - c["rho"] ** 2); chi2 = (1 - c["rho"] ** 2) ** (-k) - 1
    else:
        k = c["k"]; S1, S2 = np.array([[1, c["rho"]], [c["rho"], 1]]), np.array([[1, -c["rho"]], [-c["rho"], 1]])
        cs = k * XE.cs_mixture([0.5, 0.5], [np.zeros(2), np.zeros(2)], [S1, S2], [1], [np.zeros(2)], [I2]); kl = float(lp.mean()); chi2 = float(np.exp(lp).mean() - 1)
    return {"cs_lebesgue_sqratio": float(cs), "KL_PQ": float(kl), "pearson_chi2": float(chi2), "PE_half": float(pe_half), "mixture_S": float(S),
            "provenance": "closed form (cs; KL / chi2 for A1 / A2) + Monte-Carlo (S, PE_half; KL / chi2 for A3)", "n_truth_mc": N_TRUTH}


# --------------------------------------------------------------------------------------------------------------------------------- helpers
def median_h(x, rng, m=2048):
    s = x[rng.choice(len(x), size=min(m, len(x)), replace=False)]; d2 = np.maximum(XE.sq_dist(s, s), 0); iu = np.triu_indices(len(s), 1)
    return float(np.sqrt(np.median(d2[iu])))


def quad_risk(Tp, Tq):
    """Common quadratic risk 1/2 E_P (1 - T)^2 + 1/2 E_Q (-1 - T)^2 (= (1 - J) / 2 of T in [-1, 1]); lower is better."""
    return float(0.5 * np.mean((1 - Tp) ** 2) + 0.5 * np.mean((1 + Tq) ** 2))


def common_eval(Tp, Tq, eta_p, eta_q):
    J = float(np.mean(Tp - Tp ** 2 / 2) + np.mean(-Tq - Tq ** 2 / 2))
    return {"posterior_mse": float(0.5 * np.mean((Tp - eta_p) ** 2) + 0.5 * np.mean((Tq - eta_q) ** 2)), "J_eval": J}


def gauss_feats(x, C, h):
    return np.exp(-0.5 * np.maximum(XE.sq_dist(x, C), 0) / h ** 2)


def kde_lr(Pf, Qf, x, h):
    return XE.KDE(Pf, h).logpdf(x) - XE.KDE(Qf, h).logpdf(x)


# --------------------------------------------------------------------------------------------------------------------------------- methods
def m_cs_native(Pf, Qf, Pc, Qc, h0, tr):
    rows = []; n, d = len(Pf) + len(Qf), Pf.shape[1]; sig = float(np.concatenate([Pf, Qf]).std(0).mean())
    hs = {f"mult{m}": m * h0 for m in CS_MULTS}; hs["scott"] = sig * n ** (-1 / (d + 4)); hs["silverman"] = sig * (4 / (d + 2)) ** (1 / (d + 4)) * n ** (-1 / (d + 4))
    ll = {}
    for name, h in hs.items():
        p, q = XE.KDE(Pf, h), XE.KDE(Qf, h); v = XE.cs_kde(p, q); ll[name] = float(p.logpdf(Pc).mean() + q.logpdf(Qc).mean())
        rows.append({"variant": name, "h": h, "estimate": v, "error": v - tr["cs_lebesgue_sqratio"]})
    cv = max((r for r in rows if r["variant"].startswith("mult")), key=lambda r: ll[r["variant"]])
    rows.append({**cv, "variant": "cv_" + cv["variant"]})
    return rows


def m_kde_s(Pf, Qf, Pc, Qc, Pe, Qe, h0, eta):
    grid = list(CS_MULTS); risk = {m: quad_risk(np.tanh(kde_lr(Pf, Qf, Pc, m * h0) / 2), np.tanh(kde_lr(Pf, Qf, Qc, m * h0) / 2)) for m in grid}
    sel = min(grid, key=lambda m: risk[m]); ext = None
    if sel in (grid[0], grid[-1]):
        ext = KDE_EXT[0] if sel == grid[0] else KDE_EXT[1]
        risk[ext] = quad_risk(np.tanh(kde_lr(Pf, Qf, Pc, ext * h0) / 2), np.tanh(kde_lr(Pf, Qf, Qc, ext * h0) / 2)); sel = min(risk, key=lambda m: risk[m])
    Tp, Tq = np.tanh(kde_lr(Pf, Qf, Pe, sel * h0) / 2), np.tanh(kde_lr(Pf, Qf, Qe, sel * h0) / 2)
    return {"selected_mult": sel, "extension": ext, "at_edge_after_extension": sel in (min(risk), max(risk)), **common_eval(Tp, Tq, *eta),
            "S_plugin": float(0.5 * np.mean(Tp ** 2) + 0.5 * np.mean(Tq ** 2))}


def m_rulsif(Pf, Qf, Pc, Qc, Pe, Qe, h0, eta, rng, alpha):
    """alpha 1/2 (RuLSIF) or 0 (uLSIF): native CV choice and the common-risk choice."""
    best_nat, best_com = None, None
    for nc in (100, 256):
        C = Pf[rng.choice(len(Pf), size=min(nc, len(Pf)), replace=False)]
        for m in KM:
            h = m * h0; fp, fq = gauss_feats(Pf, C, h), gauss_feats(Qf, C, h); cp, cq = gauss_feats(Pc, C, h), gauss_feats(Qc, C, h)
            for lam in RIDGES:
                th = XE.relative_ratio_fit(fp, fq, alpha, lam); up, uq = cp @ th, cq @ th
                nat = float(0.5 * (alpha * np.mean(up ** 2) + (1 - alpha) * np.mean(uq ** 2)) - np.mean(up))
                if alpha > 0:
                    Tp, Tq = np.clip(up, 0, 2) - 1, np.clip(uq, 0, 2) - 1
                else:
                    dp, dq = np.clip(up, 0, None) / (1 + np.clip(up, 0, None)), np.clip(uq, 0, None) / (1 + np.clip(uq, 0, None)); Tp, Tq = 2 * dp - 1, 2 * dq - 1
                com = quad_risk(Tp, Tq); cfg = (nc, m, lam, th, C, h)
                best_nat = (nat, cfg) if best_nat is None or nat < best_nat[0] else best_nat
                best_com = (com, cfg) if best_com is None or com < best_com[0] else best_com
    out = {}
    for tag, (_, (nc, m, lam, th, C, h)) in (("native_cv", best_nat), ("common", best_com)):
        up, uq = gauss_feats(Pe, C, h) @ th, gauss_feats(Qe, C, h) @ th
        if alpha > 0:
            Tp, Tq = np.clip(up, 0, 2) - 1, np.clip(uq, 0, 2) - 1; native = float(-0.5 * alpha * np.mean(up ** 2) - 0.5 * (1 - alpha) * np.mean(uq ** 2) + np.mean(up) - 0.5)
        else:
            dp, dq = np.clip(up, 0, None) / (1 + np.clip(up, 0, None)), np.clip(uq, 0, None) / (1 + np.clip(uq, 0, None)); Tp, Tq = 2 * dp - 1, 2 * dq - 1
            native = float(-0.5 * np.mean(uq ** 2) + np.mean(up) - 0.5)
        clip = float(np.mean(np.concatenate([up, uq]) < 0) + (np.mean(np.concatenate([up, uq]) > 2) if alpha > 0 else 0))
        out[tag] = {"centres": nc, "mult": m, "ridge": lam, "edge": m in (KM[0], KM[-1]) or lam in (RIDGES[0], RIDGES[-1]), "native_estimate": native,
                    "clip_fraction": clip, **common_eval(Tp, Tq, *eta)}
    return out


def m_nystrom(Pf, Qf, Pc, Qc, Pe, Qe, h0, eta, rng):
    best = None; pool = np.concatenate([Pf, Qf])
    for nc in (256, 1024):
        C = pool[rng.choice(len(pool), size=min(nc, len(pool)), replace=False)]
        for m in KM:
            nm = XE.nystrom_map(C, m * h0); fp, fq = XE.nystrom_features(Pf, nm), XE.nystrom_features(Qf, nm); cp, cq = XE.nystrom_features(Pc, nm), XE.nystrom_features(Qc, nm)
            for lam in RIDGES:
                th = XE.squared_posterior_fit(fp, fq, lam); com = quad_risk(np.clip(cp @ th, -1, 1), np.clip(cq @ th, -1, 1))
                best = (com, (nc, m, lam, th, nm)) if best is None or com < best[0] else best
    nc, m, lam, th, nm = best[1]; tp, tq = XE.nystrom_features(Pe, nm) @ th, XE.nystrom_features(Qe, nm) @ th
    return {"centres": nc, "mult": m, "ridge": lam, "edge": m in (KM[0], KM[-1]) or lam in (RIDGES[0], RIDGES[-1]),
            "clip_fraction": float(np.mean(np.abs(np.concatenate([tp, tq])) > 1)), **common_eval(np.clip(tp, -1, 1), np.clip(tq, -1, 1), *eta)}


def m_neural(Pf, Qf, Pc, Qc, Pe, Qe, eta, seed):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.set_num_threads(4); out = {}
    T = lambda a: torch.tensor(a, dtype=torch.float32)
    tPf, tQf, tPc, tQc, tPe, tQe = map(T, (Pf, Qf, Pc, Qc, Pe, Qe))
    for method in ("N-VCS", "N-BCE", "N-DV", "N-NWJ"):
        cands = []
        for lr in (1e-4, 1e-3):
            torch.manual_seed(seed); net = nn.Sequential(nn.Linear(Pf.shape[1], 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
            opt = torch.optim.Adam(net.parameters(), lr=lr); g = torch.Generator().manual_seed(seed); bad = 0
            for step in range(2000):
                ip, iq = torch.randint(len(tPf), (256,), generator=g), torch.randint(len(tQf), (256,), generator=g); fp, fq = net(tPf[ip]).squeeze(-1), net(tQf[iq]).squeeze(-1)
                if method == "N-VCS":
                    loss = 0.5 * (1 - torch.sigmoid(fp)).square().mean() + 0.5 * torch.sigmoid(fq).square().mean()
                elif method == "N-BCE":
                    loss = 0.5 * F.softplus(-fp).mean() + 0.5 * F.softplus(fq).mean()
                elif method == "N-DV":
                    loss = -(fp.mean() - (torch.logsumexp(fq, 0) - math.log(len(fq))))
                else:
                    loss = -(fp.mean() - torch.exp(fq - 1).mean())
                if not torch.isfinite(loss):
                    bad += 1; continue
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                f = {k: net(v).squeeze(-1).double().numpy() for k, v in (("pc", tPc), ("qc", tQc), ("pe", tPe), ("qe", tQe))}
            if not all(np.all(np.isfinite(v)) for v in f.values()):
                cands.append((math.inf, lr, None, bad)); continue
            shift = (logsumexp(f["qc"]) - math.log(len(f["qc"]))) if method == "N-DV" else (1.0 if method == "N-NWJ" else 0.0)
            com = quad_risk(np.tanh((f["pc"] - shift) / 2), np.tanh((f["qc"] - shift) / 2)); cands.append((com, lr, (f, shift), bad))
        com, lr, fs, bad = min(cands, key=lambda c: (c[0], c[1]))
        if fs is None:
            out[method] = {"status": "failed", "nonfinite_steps": bad}; continue
        f, shift = fs; rec = {"lr": lr, "nonfinite_steps": bad, **common_eval(np.tanh((f["pe"] - shift) / 2), np.tanh((f["qe"] - shift) / 2), *eta)}
        if method == "N-DV":
            rec["native_KL"] = float(f["pe"].mean() - (logsumexp(f["qe"]) - math.log(len(f["qe"]))))
        if method == "N-NWJ":
            rec["native_KL"] = float(f["pe"].mean() - np.exp(f["qe"] - 1).mean())
        out[method] = rec
    return out


# ------------------------------------------------------------------------------------------------------------------------------------ tests
def mmd_perm(X, Y, rng):
    Z = np.concatenate([X, Y]); h = median_h(Z, rng); K = np.exp(-0.5 * np.maximum(XE.sq_dist(Z, Z), 0) / h ** 2); n = len(X); np.fill_diagonal(K, 0)

    def stat(idx):
        a, b = idx[:n], idx[n:]; Kaa, Kbb, Kab = K[np.ix_(a, a)], K[np.ix_(b, b)], K[np.ix_(a, b)]
        return Kaa.sum() / (n * (n - 1)) + Kbb.sum() / (n * (n - 1)) - 2 * Kab.mean()
    obs = stat(np.arange(2 * n)); null = np.array([stat(rng.permutation(2 * n)) for _ in range(PERMS)])
    return float((1 + (null >= obs).sum()) / (PERMS + 1))


def hsic_perm(X, Y, rng):
    n = len(X); H = np.eye(n) - 1 / n
    Kx = np.exp(-0.5 * np.maximum(XE.sq_dist(X, X), 0) / median_h(X, rng) ** 2); Ky = np.exp(-0.5 * np.maximum(XE.sq_dist(Y, Y), 0) / median_h(Y, rng) ** 2)
    Kxc = H @ Kx @ H; obs = float((Kxc * Ky).sum()) / n ** 2; null = []
    for _ in range(PERMS):
        p = rng.permutation(n); null.append(float((Kxc * Ky[np.ix_(p, p)]).sum()) / n ** 2)
    return float((1 + (np.array(null) >= obs).sum()) / (PERMS + 1))


# --------------------------------------------------------------------------------------------------------------------------------------- cli
def cmd_est(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); cells = ANCHORS[a.anchor]
    idx = [int(i) for i in a.cells.split(",")] if a.cells else range(len(cells))
    f = open(out / (f"est_{a.anchor}_c{a.cells.replace(',', '-')}.jsonl" if a.cells else f"est_{a.anchor}.jsonl"), "a")     # one writer per file
    for ci in idx:
        c = cells[ci]; tr = truths(a.anchor, c, np.random.default_rng([7, ci, 999]))
        f.write(json.dumps({"anchor": a.anchor, "cell": c, "truth": tr}) + "\n"); f.flush()
        for N in a.sizes:
            for r in range(a.reps):
                rng = np.random.default_rng([11, ci, N, r]); t0 = time.time(); nc = max(64, N // 4)
                Pf, Qf, Pc, Qc = (sample(a.anchor, c, n, rng, w) for n, w in ((N, "P"), (N, "Q"), (nc, "P"), (nc, "Q")))
                Pe, Qe = sample(a.anchor, c, N_EVAL, rng, "P"), sample(a.anchor, c, N_EVAL, rng, "Q"); eta = (np.tanh(log_ratio(a.anchor, c, Pe) / 2), np.tanh(log_ratio(a.anchor, c, Qe) / 2))
                h0 = median_h(np.concatenate([Pf, Qf]), rng)
                rec = {"anchor": a.anchor, "cell": c, "N": N, "rep": r, "h0": h0,
                       "K-CS-NATIVE": m_cs_native(Pf, Qf, Pc, Qc, h0, tr), "K-KDE-S": m_kde_s(Pf, Qf, Pc, Qc, Pe, Qe, h0, eta),
                       "K-RULSIF": m_rulsif(Pf, Qf, Pc, Qc, Pe, Qe, h0, eta, rng, 0.5), "K-ULSIF": m_rulsif(Pf, Qf, Pc, Qc, Pe, Qe, h0, eta, rng, 0.0),
                       "K-NYSTROM": m_nystrom(Pf, Qf, Pc, Qc, Pe, Qe, h0, eta, rng), **m_neural(Pf, Qf, Pc, Qc, Pe, Qe, eta, r)}
                rec["seconds"] = time.time() - t0; f.write(json.dumps(rec) + "\n"); f.flush()
                print(f"[x11] {a.anchor} cell {ci} {c} N {N} rep {r}: {rec['seconds']:.0f}s", flush=True)
    return 0


def cmd_tests(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); f = open(out / f"tests_{a.anchor}.jsonl", "a")
    for ci, c in enumerate(ANCHORS[a.anchor]):
        rej = {"T-MMD": 0, "T-HSIC": 0}; n_h = 0; t0 = time.time()
        for r in range(a.reps):
            rng = np.random.default_rng([13, ci, r]); X, Y = sample(a.anchor, c, TEST_N, rng, "P"), sample(a.anchor, c, TEST_N, rng, "Q")
            rej["T-MMD"] += mmd_perm(X, Y, rng) <= 0.05
            if a.anchor in ("A2", "A3"):
                k = X.shape[1] // 2; rej["T-HSIC"] += hsic_perm(X[:, :k], X[:, k:], rng) <= 0.05; n_h += 1
        rec = {"anchor": a.anchor, "cell": c, "n_per_class": TEST_N, "reps": a.reps, "perms": PERMS, "alpha": 0.05,
               "null": (c.get("delta", c.get("rho")) == 0.0), "T-MMD_rejections": rej["T-MMD"], "T-HSIC_rejections": rej["T-HSIC"] if n_h else None, "seconds": time.time() - t0}
        f.write(json.dumps(rec) + "\n"); f.flush(); print(f"[x11] tests {a.anchor} {c}: MMD {rej['T-MMD']}/{a.reps}, HSIC {rej['T-HSIC'] if n_h else '-'}", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("est"); e.add_argument("--anchor", choices=sorted(ANCHORS), required=True); e.add_argument("--out", required=True)
    e.add_argument("--reps", type=int, default=10); e.add_argument("--sizes", type=int, nargs="+", default=[256, 4096]); e.add_argument("--cells", default="")
    t = sub.add_parser("tests"); t.add_argument("--anchor", choices=sorted(ANCHORS), required=True); t.add_argument("--out", required=True); t.add_argument("--reps", type=int, default=200)
    a = ap.parse_args(argv)
    return cmd_est(a) if a.cmd == "est" else cmd_tests(a)


if __name__ == "__main__":
    sys.exit(main())
