"""R6-02R — REG controlled validation (r6 package docs/04 §3–5, docs/03 §2 and §6; reference r6/package_r6_20261011/reference/r6_reference.py).

Independent 2-D scenes G: a latent field L_G (random Fourier features with Gaussian covariance, exact at continuous positions).  Modalities on the
integer grid: U_G(y) = L_G(y) + n_a;  V_G(y) = g(L_G(y − φ*_G)) + n_b, so V carries the known shift φ*_G.  Relation g: 'monotone' (−l) or
'nonmonotone' (l² − 1: zero linear correlation, still dependent).  Observation = the local value pair (u, v) only (no coordinates, scene id,
transform index or padding mask).
Within-scene laws at a candidate φ on a FIXED interior support Ω (every candidate keeps all samples inside the image; bilinear interpolation for
every φ, identity included): P_φ = (U(x_i), V(x_i + φ)); Q_φ = (U(x_i), V(x_π(i) + φ)) with a fixed within-scene permutation π — the reference term is
recomputed under each φ (Q is not a constant).
Critics T(u, v), fitted on independent TRAINING scenes at φ = id, then frozen: VCS (maximise J, T = tanh(w·feat), polynomial features of degree 4),
balanced logistic (T = tanh f), oracle (exact η of the pixel model by quadrature; generator-specific).  Matching energy F_T(φ) = J(T; P_φ, Q_φ)
on TEST scenes.  Baselines on the same samples: histogram MI, NMI, |Pearson correlation|.  One shared optimiser for every method: grid search
over integer shifts then a multi-scale compass search; capture range = compass search from initial offsets r ∈ {1, 2, 4, 6} px.
Checks: interpolation / sign convention; finite-difference vs analytic dF/dφ; counterexamples — repetitive (periodic) structure, flat (weak
relation), scene-identity leakage (scene offsets; cross-scene vs within-scene Q in training), coordinate leakage (a critic that sees the
coordinate difference), shrinking-overlap support vs fixed Ω.  Scenes are the independent units (pixels are not).
    python scripts/r6_reg_controlled.py --out /home/infres/yinwang/CS_QMI/outputs/R6_controlled [--seeds 3] [--workers 8] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

H = W = 96; MAXS = 8; MARGIN = MAXS + 2; N_POS = 2048; GRID = np.arange(-MAXS, MAXS + 1); RS = (1, 2, 4, 6)
CONDITIONS = {
    "main_nonmonotone": dict(relation="nonmonotone"), "main_monotone": dict(relation="monotone"),
    "periodic": dict(relation="nonmonotone", latent="periodic"), "weak": dict(relation="nonmonotone", sig_b=3.0),
    "scene_offset_within": dict(relation="nonmonotone", offset=1.5, train_q="within"), "scene_offset_cross": dict(relation="nonmonotone", offset=1.5, train_q="cross"),
    "coordinate_leak": dict(relation="nonmonotone", coord_leak=True),
    "frame_overlap": dict(relation="nonmonotone", support="overlap", frame=6), "frame_fixed": dict(relation="nonmonotone", frame=6),
}
METHODS = ("oracle", "vcs", "logistic", "mi", "nmi", "corr")


# --------------------------------------------------------------------------------------------- scenes
def g_rel(l, relation):
    return -l if relation == "monotone" else l * l - 1.0


def latent_fn(rng, kind, ell=4.0, n_feat=64, period=6.0):
    if kind == "periodic":
        ph = rng.uniform(0, 2 * np.pi, 2)
        return lambda yy, xx: np.cos(2 * np.pi * xx / period + ph[0]) + np.cos(2 * np.pi * yy / period + ph[1])
    Wf = rng.normal(0.0, 1.0 / ell, (n_feat, 2)); b = rng.uniform(0, 2 * np.pi, n_feat); amp = math.sqrt(2.0 / n_feat)
    return lambda yy, xx: amp * np.cos(yy[..., None] * Wf[:, 0] + xx[..., None] * Wf[:, 1] + b).sum(-1)


def make_scene(rng, phi_star, c):
    L = latent_fn(rng, c.get("latent", "rff")); yy, xx = np.mgrid[0:H, 0:W].astype(float)
    off = rng.normal(0.0, c.get("offset", 0.0)) if c.get("offset") else 0.0
    U = L(yy, xx) + c.get("sig_a", 0.3) * rng.normal(size=(H, W)) + off
    V = g_rel(L(yy - phi_star[0], xx - phi_star[1]), c["relation"]) + c.get("sig_b", 0.3) * rng.normal(size=(H, W)) + off
    f = c.get("frame", 0)
    if f:                                                                      # a bright image-border frame in both modalities (an acquisition artefact)
        for img in (U, V):
            img[:f, :] = img[-f:, :] = img[:, :f] = img[:, -f:] = 3.0
    return U, V


def positions(rng, full=False):
    """Fixed interior support Ω (every candidate keeps samples inside), or the full image for the shrinking-overlap counterexample."""
    lo = 0 if full else MARGIN; ys, xs = np.mgrid[lo:H - lo, lo:W - lo]; idx = rng.choice(ys.size, N_POS, replace=False)
    return ys.ravel()[idx].astype(float), xs.ravel()[idx].astype(float), rng.permutation(N_POS)


def bilinear(img, ys, xs):
    """Value and its y / x derivatives; raises if a sample leaves the image (fixed-support guard: no padding is ever read)."""
    if ys.min() < 0 or xs.min() < 0 or ys.max() > img.shape[0] - 1 - 1e-9 or xs.max() > img.shape[1] - 1 - 1e-9:
        raise ValueError("sample outside the image: the support must stay inside for every candidate transform")
    y0 = np.floor(ys).astype(int); x0 = np.floor(xs).astype(int); dy = ys - y0; dx = xs - x0
    y1 = np.minimum(y0 + 1, img.shape[0] - 1); x1 = np.minimum(x0 + 1, img.shape[1] - 1)
    a, b, c, d = img[y0, x0], img[y0, x1], img[y1, x0], img[y1, x1]
    v = (1 - dy) * (1 - dx) * a + (1 - dy) * dx * b + dy * (1 - dx) * c + dy * dx * d
    return v, (1 - dx) * (c - a) + dx * (d - b), (1 - dy) * (b - a) + dy * (d - c)


# --------------------------------------------------------------------------------------------- critics
def poly(zu, zv, deg=4):
    return np.stack([zu ** i * zv ** j for i in range(deg + 1) for j in range(deg + 1 - i)], -1)


def dpoly_dv(zu, zv, deg=4):
    return np.stack([(j * zu ** i * zv ** (j - 1)) if j else np.zeros_like(zu) for i in range(deg + 1) for j in range(deg + 1 - i)], -1)


class PolyCritic:
    def __init__(self, uP, vP, uQ, vQ, kind, extraP=None, extraQ=None, iters=1500):
        self.mu = (uP.mean(), vP.mean()); self.sd = (uP.std() + 1e-9, vP.std() + 1e-9); self.extra = extraP is not None
        FP, FQ = self.feat(uP, vP, extraP), self.feat(uQ, vQ, extraQ); w = np.zeros(FP.shape[1])
        if kind == "logistic":                                                 # balanced logistic σ(2f): damped Newton with backtracking (stable loss)
            X = np.concatenate([FP, FQ]); y = np.concatenate([np.ones(len(FP)), np.zeros(len(FQ))]); wt = np.concatenate([np.full(len(FP), 0.5 / len(FP)), np.full(len(FQ), 0.5 / len(FQ))])
            lam = 1e-3; sgn = np.where(y > 0, -1.0, 1.0)
            loss = lambda w_: float(wt @ np.logaddexp(0.0, sgn * 2 * (X @ w_)) + 0.5 * lam * w_ @ w_)
            for _ in range(100):
                f = X @ w; p = 0.5 * (1 + np.tanh(f)); gr = 2 * X.T @ (wt * (p - y)) + lam * w
                Hm = 4 * (X * (wt * p * (1 - p))[:, None]).T @ X + lam * np.eye(len(w)); st = np.linalg.solve(Hm, gr); L0 = loss(w); t = 1.0
                while loss(w - t * st) > L0 - 1e-4 * t * (gr @ st) and t > 1e-6:
                    t *= 0.5
                w = w - t * st
                if np.abs(t * st).max() < 1e-9:
                    break
        else:                                                                  # VCS: maximise J with T = tanh f (Adam)
            m = np.zeros_like(w); v2 = np.zeros_like(w)
            for t in range(1, iters + 1):
                TP, TQ = np.tanh(FP @ w), np.tanh(FQ @ w)
                gr = ((1 - TP) * (1 - TP ** 2)) @ FP / len(FP) + ((-1 - TQ) * (1 - TQ ** 2)) @ FQ / len(FQ) - 1e-4 * w
                m = 0.9 * m + 0.1 * gr; v2 = 0.999 * v2 + 0.001 * gr ** 2; w += 0.05 * (m / (1 - 0.9 ** t)) / (np.sqrt(v2 / (1 - 0.999 ** t)) + 1e-8)
        self.w = w

    def feat(self, u, v, extra=None):
        F = poly((u - self.mu[0]) / self.sd[0], (v - self.mu[1]) / self.sd[1])
        return F if extra is None else np.concatenate([F, extra], -1)

    def T_and_dv(self, u, v, extra=None):
        zu, zv = (u - self.mu[0]) / self.sd[0], (v - self.mu[1]) / self.sd[1]
        F = self.feat(u, v, extra); T = np.tanh(F @ self.w); nb = poly(zu[:1], zv[:1]).shape[1]
        return T, (1 - T ** 2) * (dpoly_dv(zu, zv) @ self.w[:nb]) / self.sd[1]


class OracleCritic:
    """Exact η of the generator's pixel model (L ~ N(0, 1) marginally): p(u, v) = ∫ φ_a(u − l) φ_b(v − g(l)) φ(l) dl vs the product of marginals."""
    def __init__(self, c, n=241):
        sa, sb = c.get("sig_a", 0.3), c.get("sig_b", 0.3); l = np.linspace(-6, 6, 601); wl = np.exp(-l ** 2 / 2); wl /= wl.sum()
        self.ug = np.linspace(-5, 5, n); gl = g_rel(l, c["relation"]); self.vg = np.linspace(gl.min() - 4 * sb, gl.max() + 4 * sb, n)
        nrm = lambda x, s: np.exp(-x ** 2 / (2 * s * s)) / (s * math.sqrt(2 * math.pi))
        A = nrm(self.ug[:, None] - l[None], sa) * wl; B = nrm(self.vg[:, None] - gl[None], sb)
        p = A @ B.T; q = np.outer(A.sum(1), B @ wl); self.eta = (p - q) / (p + q + 1e-300)

    def T_and_dv(self, u, v, extra=None):
        iu = np.clip((u - self.ug[0]) / (self.ug[1] - self.ug[0]), 0, len(self.ug) - 1.001); iv = np.clip((v - self.vg[0]) / (self.vg[1] - self.vg[0]), 0, len(self.vg) - 1.001)
        u0, v0 = iu.astype(int), iv.astype(int); du, dv = iu - u0, iv - v0; E = self.eta
        a, b, cc, d = E[u0, v0], E[u0, v0 + 1], E[u0 + 1, v0], E[u0 + 1, v0 + 1]
        T = (1 - du) * (1 - dv) * a + (1 - du) * dv * b + du * (1 - dv) * cc + du * dv * d
        return T, ((1 - du) * (b - a) + du * (d - cc)) / (self.vg[1] - self.vg[0])


# --------------------------------------------------------------------------------------------- objectives
def samples(U, V, pos, phi, support="fixed"):
    ys, xs, perm = pos
    if support == "overlap":                                                   # shrinking-overlap variant: drop samples whose partner leaves the image
        ok = (ys + phi[0] >= 0) & (ys + phi[0] <= H - 1.001) & (xs + phi[1] >= 0) & (xs + phi[1] <= W - 1.001); okq = ok & ok[perm]
        ys2, xs2, pq = ys[okq], xs[okq], perm[okq]
    else:
        ys2, xs2, pq = ys, xs, perm
    u = U[ys2.astype(int), xs2.astype(int)]
    vP, gyP, gxP = bilinear(V, ys2 + phi[0], xs2 + phi[1]); vQ, gyQ, gxQ = bilinear(V, ys[pq] + phi[0], xs[pq] + phi[1])
    return u, (vP, gyP, gxP), (vQ, gyQ, gxQ), (ys2, xs2, ys[pq], xs[pq])


def leak_feat(dy, dx):
    return np.stack([(dy / MAXS) ** 2, (dx / MAXS) ** 2], -1)


def F_critic(crit, U, V, pos, phi, support="fixed", leak=False, grad=False):
    u, (vP, gyP, gxP), (vQ, gyQ, gxQ), (yP, xP, yQ, xQ) = samples(U, V, pos, phi, support)
    eP = eQ = None
    if leak:                                                                   # forbidden input: the (squared) coordinate difference between the two samples
        eP = leak_feat(np.full_like(u, phi[0]), np.full_like(u, phi[1])); eQ = leak_feat(yQ + phi[0] - yP, xQ + phi[1] - xP)
    TP, dP = crit.T_and_dv(u, vP, eP); TQ, dQ = crit.T_and_dv(u, vQ, eQ)
    F = float(np.mean(TP - TP ** 2 / 2) + np.mean(-TQ - TQ ** 2 / 2))
    if not grad:
        return F
    gy = np.mean((1 - TP) * dP * gyP) + np.mean((-1 - TQ) * dQ * gyQ); gx = np.mean((1 - TP) * dP * gxP) + np.mean((-1 - TQ) * dQ * gxQ)
    return F, np.array([gy, gx])


def F_classic(kind, U, V, pos, phi, support="fixed", bins=32):
    u, (vP, _, _), _, _ = samples(U, V, pos, phi, support)
    if kind == "corr":
        return float(abs(np.corrcoef(u, vP)[0, 1]))
    h, _, _ = np.histogram2d(u, vP, bins=bins); p = h / h.sum(); pu, pv = p.sum(1), p.sum(0); nz = p > 0
    hu = -np.sum(pu[pu > 0] * np.log(pu[pu > 0])); hv = -np.sum(pv[pv > 0] * np.log(pv[pv > 0])); huv = -np.sum(p[nz] * np.log(p[nz]))
    return float(hu + hv - huv) if kind == "mi" else float((hu + hv) / huv)


def compass(fun, x0, step=2.0, tol=0.05, max_eval=300):
    x = np.clip(np.asarray(x0, float), -MAXS, MAXS); fx = fun(x); n = 1
    while step > tol and n < max_eval:
        for d in ((step, 0), (-step, 0), (0, step), (0, -step)):
            y = np.clip(x + d, -MAXS, MAXS)
            if np.allclose(y, x):
                continue
            fy = fun(y); n += 1
            if fy > fx:
                x, fx = y, fy; break
        else:
            step /= 2
    return x, fx, n


# --------------------------------------------------------------------------------------------- one condition
def run_condition(job):
    name, seed, n_train, n_test = job; c = CONDITIONS[name]; rng = np.random.default_rng([20261011, seed, sum(map(ord, name))])
    tr = []                                                                    # training scenes: aligned (φ = id)
    full = c.get("support") == "overlap"
    for _ in range(n_train):
        U, V = make_scene(rng, (0.0, 0.0), c); pos = positions(rng, full); tr.append((U, V, pos))
    uP = np.concatenate([U[p[0].astype(int), p[1].astype(int)] for U, V, p in tr]); vP = np.concatenate([V[p[0].astype(int), p[1].astype(int)] for U, V, p in tr])
    if c.get("train_q") == "cross":                                            # cross-scene reference (global product across scenes)
        vQ = np.concatenate([tr[(i + 1) % n_train][1][p[0][p[2]].astype(int), p[1][p[2]].astype(int)] for i, (U, V, p) in enumerate(tr)])
    else:
        vQ = np.concatenate([V[p[0][p[2]].astype(int), p[1][p[2]].astype(int)] for U, V, p in tr])
    extraP = extraQ = None
    if c.get("coord_leak"):
        extraP = leak_feat(np.zeros(len(uP)), np.zeros(len(uP))); extraQ = np.concatenate([leak_feat(p[0][p[2]] - p[0], p[1][p[2]] - p[1]) for U, V, p in tr])
    crit = {"vcs": PolyCritic(uP, vP, uP, vQ, "vcs", extraP, extraQ), "logistic": PolyCritic(uP, vP, uP, vQ, "logistic", extraP, extraQ)}
    if not c.get("coord_leak") and c.get("latent") != "periodic" and not c.get("offset") and not c.get("frame"):
        crit["oracle"] = OracleCritic(c)
    sup = c.get("support", "fixed"); leak = bool(c.get("coord_leak")); out = {m: {"grid_err": [], "final_err": [], "capture": {r: [] for r in RS}, "sharp": [], "evals": []} for m in METHODS}
    fd_checks = []; curves = None
    for t in range(n_test):
        phi_star = rng.uniform(-5, 5, 2); U, V = make_scene(rng, phi_star, c); pos = positions(rng, full)
        funs = {m: (lambda x, cr=crit[m]: F_critic(cr, U, V, pos, x, sup, leak)) for m in crit}
        funs.update({m: (lambda x, k=m: F_classic(k, U, V, pos, x, sup)) for m in ("mi", "nmi", "corr")})
        if t == 0 and "vcs" in crit and not leak:                              # finite differences vs analytic gradient at non-integer shifts
            for trial in range(3):
                x = phi_star + rng.uniform(-1.3, 1.3, 2); x = x + 0.37 * (np.abs(x - np.round(x)) < 0.05)
                F0, gr = F_critic(crit["vcs"], U, V, pos, x, sup, False, grad=True); h = 1e-4
                fd = np.array([(F_critic(crit["vcs"], U, V, pos, x + d, sup) - F_critic(crit["vcs"], U, V, pos, x - d, sup)) / (2 * h) for d in (np.array([h, 0]), np.array([0, h]))])
                fd_checks.append({"phi": x.tolist(), "analytic": gr.tolist(), "finite_difference": fd.tolist(), "max_abs_diff": float(np.abs(gr - fd).max())})
        if t == 0:
            curves = {"phi_star": phi_star.tolist(), "x_slice": {m: [funs[m](np.array([round(phi_star[0]), dx])) for dx in GRID] for m in funs}}
        for m, fun in funs.items():
            G = np.array([[fun(np.array([dy, dx])) for dx in GRID] for dy in GRID]); iy, ix = np.unravel_index(np.argmax(G), G.shape)
            g0 = np.array([GRID[iy], GRID[ix]]); out[m]["grid_err"].append(float(np.abs(g0 - phi_star).max()))
            nb = [G[jy, jx] for jy in range(max(0, iy - 1), min(len(GRID), iy + 2)) for jx in range(max(0, ix - 1), min(len(GRID), ix + 2)) if (jy, jx) != (iy, ix)]
            out[m]["sharp"].append(float((G[iy, ix] - np.mean(nb)) / (G.std() + 1e-12)))
            xf, _, ne = compass(fun, g0, step=0.5); out[m]["final_err"].append(float(np.abs(xf - phi_star).max())); out[m]["evals"].append(ne)
            for r in RS:
                th = rng.uniform(0, 2 * np.pi); x0 = phi_star + r * np.array([np.sin(th), np.cos(th)])
                xr, _, _ = compass(fun, x0); out[m]["capture"][r].append(bool(np.abs(xr - phi_star).max() <= 0.5))
    summ = {}
    for m in out:
        if not out[m]["grid_err"]:
            continue
        o = out[m]; ge, fe = np.array(o["grid_err"]), np.array(o["final_err"])
        summ[m] = {"n_scenes": len(ge), "grid_success_le1": float(np.mean(ge <= 1.0)), "final_success_le0.5": float(np.mean(fe <= 0.5)), "median_final_err": float(np.median(fe)),
                   "p90_final_err": float(np.quantile(fe, 0.9)), "capture_rate": {str(r): float(np.mean(o["capture"][r])) for r in RS}, "median_sharpness": float(np.median(o["sharp"])),
                   "mean_evals": float(np.mean(o["evals"])), "final_err_per_scene": [round(x, 3) for x in fe]}
    return {"condition": name, "seed": seed, "config": c, "methods": summ, "fd_checks": fd_checks, "curve_example": curves}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true"); a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    n_train, n_test = (3, 2) if a.smoke else (20, 20)
    jobs = [(n, s, n_train, n_test) for n in CONDITIONS for s in range(1 if a.smoke else a.seeds)]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(run_condition, jobs))
    (out / ("reg_controlled_smoke.json" if a.smoke else "reg_controlled.json")).write_text(json.dumps(res, indent=1)); print(f"[r6_reg] {len(res)} condition-seeds -> {out}", flush=True)
    for r in res:
        print(r["condition"], r["seed"], {m: (v["final_success_le0.5"], v["capture_rate"]) for m, v in r["methods"].items()}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
