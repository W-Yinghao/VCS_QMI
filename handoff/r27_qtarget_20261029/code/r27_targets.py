"""R27 pseudo-target generators on one fixed candidate graph (DESIGN_CN.md §3).

Inputs for one direction of one update (all stop-grad, float64 inside the solvers):
  A [B,M]  row-stochastic candidate weights (A_ij = #slots of query i pointing to atom j / K; merged atoms)
  Y [M,d]  unit candidate atoms (current-peer re-encoding, see r27_interface)
  r_i = 1/B, c = A^T r (background marginal over atoms), mu_i = A_i Y, mu_Q = c Y, delta_i = mu_i - mu_Q,
  H_i = sum_j (A_ij + c_j) y_j y_j^T, F_ij(z) = z . y_j, domain ||z|| <= 1 (so |F| <= 1: bounded cosine critic).

Generators (each row i independently, z in the unit ball):
  mean (A2): t_i = normalize(mu_i)                                     (no inner problem)
  vcs  (A3): max  delta_i.z - 1/2 z^T H_i z                             (= row-i cosine VCS J / r_i)
  bce  (A4): max  1/2 sum_j A_ij log((1+F_j)/2) + 1/2 sum_j c_j log((1-F_j)/2)   (row-i balanced BCE / r_i)
  snce (A5): max  sum_j A_ij F_j/tau - log sum_j c_j exp(F_j/tau), tau = 0.05     (row-i marginal-aware soft NCE / r_i)
All three have the same gradient direction at z = 0 (delta, delta/2, delta/tau).
VCS: exact solver (one shared M x M eigendecomposition of H_Q per direction + per-row rank-K Woodbury, secular
Newton on the KKT multiplier nu). BCE / sNCE (production): batched nonmonotone spectral projected gradient (SPG)
from z0 = 0 in exact reduced coordinates. solve_fista is kept only as an independent cross-check solver (CPU gate). Every solver reports the same KKT residual
  res_i = ||g_i - nu_hat_i z_i||, nu_hat_i = max(z_i.g_i, 0) if ||z_i|| = 1 else 0,  rel_i = res_i / ||grad f_i(0)||,
iterations and cap hits. Targets are normalize(z*); rows with non-finite / tiny ||z*|| fall back to the mean target.
"""
import math
import torch

TAU_SNCE = 0.05
TOL_REL = 1e-6        # relative KKT residual target, all generators
FISTA_CAP = 3000      # iteration cap (BCE / sNCE)
NEWTON_CAP = 100      # secular Newton cap (VCS)
EIG_TOL = 1e-12       # relative eigenvalue truncation of H_Q (null space of Y)
ZMIN = 1e-6           # ||z*|| below this -> fallback to the mean target
GENERATORS = ('mean', 'vcs', 'bce', 'snce')


# ----------------------------------------------------------------------------- graph moments

def moments(A, Y):
    c = A.mean(0)                 # r_i = 1/B
    mu = A @ Y
    muQ = c @ Y
    return c, mu, muQ, mu - muQ


def hmatvec(Z, A, c, Y):
    """Exact moment matvec, row i: H_i z_i = Y^T[(A_i + c) * (Y z_i)] (no d x d matrix)."""
    return ((A + c) * (Z @ Y.t())) @ Y


# ----------------------------------------------------------------------------- objectives (row-wise) and gradients

def f_vcs(Z, A, c, Y):
    T = Z @ Y.t()
    return ((A - c) * T).sum(1) - 0.5 * ((A + c) * T * T).sum(1)


def g_vcs(Z, A, c, Y):
    T = Z @ Y.t()
    return ((A - c) - (A + c) * T) @ Y


def f_bce(Z, A, c, Y):
    T = Z @ Y.t()
    lp = torch.where(A > 0, A * torch.log((1 + T) / 2), torch.zeros_like(T))
    lq = c * torch.log((1 - T) / 2)
    f = 0.5 * lp.sum(1) + 0.5 * lq.sum(1)
    bad = ~((T > -1) & (T < 1) | (A == 0) & (T < 1)).all(1)
    return torch.where(bad | ~torch.isfinite(f), torch.full_like(f, -math.inf), f)


def g_bce(Z, A, c, Y):
    T = Z @ Y.t()
    pa = torch.where(A > 0, A / (1 + T), torch.zeros_like(T))
    return (0.5 * (pa - c / (1 - T))) @ Y


def f_snce(Z, A, c, Y, tau=TAU_SNCE):
    T = Z @ Y.t()
    return (A * T).sum(1) / tau - torch.logsumexp(T / tau + torch.log(c), 1)


def g_snce(Z, A, c, Y, tau=TAU_SNCE):
    T = Z @ Y.t()
    p = torch.softmax(T / tau + torch.log(c), 1)
    return ((A - p) @ Y) / tau


OBJ = {'vcs': (f_vcs, g_vcs), 'bce': (f_bce, g_bce), 'snce': (f_snce, g_snce)}


def grad0_norm(gen, delta):
    s = {'vcs': 1.0, 'bce': 0.5, 'snce': 1.0 / TAU_SNCE}[gen]
    return s * delta.norm(dim=1)


def kkt(Z, G):
    """Ball-constrained KKT residual of a concave max: interior -> ||g||; boundary -> ||g - max(z.g,0) z||."""
    nz = Z.norm(dim=1)
    on = nz > 1 - 1e-9
    nu = torch.where(on, (Z * G).sum(1).clamp(min=0), torch.zeros_like(nz))
    return (G - nu.unsqueeze(1) * Z).norm(dim=1), nu


def proj_ball(Z):
    return Z / Z.norm(dim=1, keepdim=True).clamp(min=1.0)


# ----------------------------------------------------------------------------- VCS exact solver

def solve_vcs(A, Y, tol_rel=TOL_REL, cap=NEWTON_CAP):
    """Exact max delta.z - 1/2 z^T H z, ||z|| <= 1, for every row, float64.
    H_i = H_Q + V_i V_i^T, H_Q = Y^T C Y (shared), V_i = Y_{S_i}^T diag(sqrt(A_i,S_i)) (<= K columns).
    All of delta_i, H_i live in range(Y^T); work in an orthonormal eigenbasis U of H_Q (one M x M eigh)."""
    B, M = A.shape
    c, mu, muQ, delta = moments(A, Y)
    sc = c.sqrt()
    Gc = (sc.unsqueeze(1) * (Y @ Y.t())) * sc.unsqueeze(0)          # C^1/2 Y Y^T C^1/2
    lam, P = torch.linalg.eigh(Gc)
    keep = lam > EIG_TOL * lam.max()
    lam, P = lam[keep], P[:, keep]
    U = (Y.t() * sc.unsqueeze(0)) @ P / lam.sqrt().unsqueeze(0)      # d x r, orthonormal, H_Q = U diag(lam) U^T
    Yt = Y @ U                                                        # M x r (reduced atoms)
    dt = delta @ U                                                    # B x r
    lost = (delta - dt @ U.t()).norm(dim=1) / delta.norm(dim=1).clamp(min=1e-300)
    # per-row low-rank factors (support of A_i, padded to Kmax with zero columns)
    Kmax = int((A > 0).sum(1).max())
    vals, idx = torch.topk(A, Kmax, dim=1)                            # zero weights -> zero columns
    V = Yt[idx] * vals.sqrt().unsqueeze(2)                            # B x K x r  (rows of V_i^T)

    def apply_inv(nu, X):
        """(diag(lam) + nu + V_i V_i^T)^{-1} x_i for every row (Woodbury, exact)."""
        D = 1.0 / (lam.unsqueeze(0) + nu.unsqueeze(1))                # B x r
        DX = D * X
        DV = D.unsqueeze(1) * V                                       # B x K x r
        cap_ = torch.eye(Kmax, dtype=A.dtype, device=A.device) + V @ DV.transpose(1, 2)   # B x K x K
        rhs = (V @ DX.unsqueeze(2))                                   # B x K x 1
        sol = torch.linalg.solve(cap_, rhs)
        return DX - (DV.transpose(1, 2) @ sol).squeeze(2)

    zero = torch.zeros(B, dtype=A.dtype, device=A.device)
    z0 = apply_inv(zero, dt)
    n0 = z0.norm(dim=1)
    interior = n0 <= 1.0
    nu = zero.clone()
    lo = zero.clone()
    hi = dt.norm(dim=1).clamp(min=1e-300)                             # ||z(hi)|| <= ||delta||/hi = 1
    nu = torch.where(interior, zero, hi)
    it = torch.zeros(B, dtype=torch.long, device=A.device)
    active = ~interior
    z = torch.where(interior.unsqueeze(1), z0, apply_inv(nu, dt))
    for k in range(cap):
        if not bool(active.any()):
            break
        nz = z.norm(dim=1)
        done = ((nz - 1).abs() <= 1e-12) | ((hi - lo) <= 1e-15 * hi)
        active = active & ~done
        if not bool(active.any()):
            break
        w = apply_inv(nu, z)
        q = (z * w).sum(1).clamp(min=1e-300)
        step = nz * nz / q * (nz - 1)                                 # More-Sorensen Newton on 1/||z(nu)|| = 1
        lo = torch.where(active & (nz > 1), torch.maximum(lo, nu), lo)
        hi = torch.where(active & (nz < 1), torch.minimum(hi, nu), hi)
        nn = nu + step
        bad = (nn <= lo) | (nn >= hi) | ~torch.isfinite(nn)
        nn = torch.where(bad, 0.5 * (lo + hi), nn)
        nu = torch.where(active, nn, nu)
        it = it + active.long()
        z = torch.where(active.unsqueeze(1), apply_inv(nu, dt), z)
    cap_hit = active.clone()
    Z = proj_ball(z @ U.t())                                          # back to d dims (removes ~1e-13 overshoot)
    # exact residual with the moment matvec in the full space
    G = delta - hmatvec(Z, A, c, Y)
    res, nu_hat = kkt(Z, G)
    info = dict(nu=nu, nu_hat=nu_hat, iters=it, cap=cap_hit, interior=interior, res=res,
                rel=res / grad0_norm('vcs', delta).clamp(min=1e-300), rank=int(lam.numel()), lost=lost,
                kkt_eq=((hmatvec(Z, A, c, Y) + nu.unsqueeze(1) * Z) - delta).norm(dim=1))
    return Z, info


# ----------------------------------------------------------------------------- BCE / sNCE: batched FISTA

def solve_fista(gen, A, Y, tol_rel=TOL_REL, cap=FISTA_CAP, check_every=5, Z0=None):
    """Accelerated projected gradient ascent on the unit ball, z0 = 0 (common init), per-row Lipschitz backtracking
    (L doubles until the quadratic lower model holds and f is finite), L *= 0.9 after an accepted step, function
    restart (momentum reset) when f decreases. Stops per row when rel KKT residual <= tol_rel."""
    f, g = OBJ[gen]
    B, M = A.shape
    c, mu, muQ, delta = moments(A, Y)
    g0 = grad0_norm(gen, delta).clamp(min=1e-300)
    Z = torch.zeros_like(mu) if Z0 is None else Z0.clone()
    X = Z.clone()
    fZ = f(Z, A, c, Y)
    t = torch.ones(B, dtype=A.dtype, device=A.device)
    L = torch.full((B,), {'vcs': 2.0, 'bce': 1.0, 'snce': 1.0 / TAU_SNCE}[gen], dtype=A.dtype, device=A.device)
    active = torch.ones(B, dtype=torch.bool, device=A.device)
    it = torch.zeros(B, dtype=torch.long, device=A.device)
    nback = 0
    for k in range(cap):
        gX = g(X, A, c, Y)
        fX = f(X, A, c, Y)
        for _ in range(60):
            Zn = proj_ball(X + gX / L.unsqueeze(1))
            fn = f(Zn, A, c, Y)
            D = Zn - X
            okr = torch.isfinite(fn) & (fn >= fX + (gX * D).sum(1) - 0.5 * L * (D * D).sum(1) - 1e-12 * fX.abs())
            okr = okr | ~active
            if bool(okr.all()):
                break
            L = torch.where(okr, L, 2 * L)
            nback += 1
        restart = fn < fZ
        tn = 0.5 * (1 + torch.sqrt(1 + 4 * t * t))
        mom = ((t - 1) / tn).unsqueeze(1)
        Xn = torch.where(restart.unsqueeze(1), Zn, Zn + mom * (Zn - Z))
        Xn = proj_ball(Xn)
        # BCE: momentum point must stay in the log domain; fall back to Zn otherwise
        if gen == 'bce':
            okx = torch.isfinite(f(Xn, A, c, Y))
            Xn = torch.where(okx.unsqueeze(1), Xn, Zn)
        a = active.unsqueeze(1)
        Z = torch.where(a, Zn, Z)
        fZ = torch.where(active, fn, fZ)
        X = torch.where(a, Xn, X)
        t = torch.where(active, torch.where(restart, torch.ones_like(tn), tn), t)
        L = torch.where(active, 0.9 * L, L)
        it = it + active.long()
        if (k + 1) % check_every == 0:
            res, _ = kkt(Z, g(Z, A, c, Y))
            active = active & ~(res / g0 <= tol_rel)
            if not bool(active.any()):
                break
    G = g(Z, A, c, Y)
    res, nu_hat = kkt(Z, G)
    info = dict(nu_hat=nu_hat, iters=it, cap=active.clone(), res=res, rel=res / g0, backtracks=nback,
                interior=Z.norm(dim=1) < 1 - 1e-9)
    if gen == 'bce':
        T = Z @ Y.t()
        info['near_log_boundary'] = ((1 - T.abs()) < 2e-7).any(1)
    return Z, info


# ----------------------------------------------------------------------------- BCE / sNCE: spectral projected gradient (production)

SPG_CAP = 4000          # iteration cap
SPG_MEM = 10
SPG_CHECK = 10          # convergence check (host sync, exact recomputation of T = Z Y^T) every SPG_CHECK iterations
LAMS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125)


def fT(gen, T, A, c, tau=TAU_SNCE):
    """Row objectives from T = Z Y^T (identical formulas to f_vcs / f_bce / f_snce; T may carry a leading batch dim)."""
    if gen == 'vcs':
        return ((A - c) * T).sum(-1) - 0.5 * ((A + c) * T * T).sum(-1)
    if gen == 'bce':
        lp = torch.where(A > 0, A * torch.log((1 + T) / 2), torch.zeros_like(T))
        f = 0.5 * lp.sum(-1) + 0.5 * (c * torch.log((1 - T) / 2)).sum(-1)
        bad = ~(((T > -1) & (T < 1)) | ((A == 0) & (T < 1))).all(-1)
        return torch.where(bad | ~torch.isfinite(f), torch.full_like(f, -math.inf), f)
    if gen == 'snce':
        return (A * T).sum(-1) / tau - torch.logsumexp(T / tau + torch.log(c), -1)
    raise ValueError(gen)


def wT(gen, T, A, c, tau=TAU_SNCE):
    """Gradient weights: grad f(z) = wT(T) @ Y."""
    if gen == 'vcs':
        return (A - c) - (A + c) * T
    if gen == 'bce':
        pa = torch.where(A > 0, A / (1 + T), torch.zeros_like(T))
        return 0.5 * (pa - c / (1 - T))
    if gen == 'snce':
        return (A - torch.softmax(T / tau + torch.log(c), -1)) / tau
    raise ValueError(gen)


def solve_spg(gen, A, Y, tol_rel=TOL_REL, cap=SPG_CAP, mem=SPG_MEM):
    """Nonmonotone spectral projected gradient (Birgin-Martinez-Raydan) ascent on the unit ball, float64, z0 = 0
    (common init), alternating BB1/BB2 steps, Grippo nonmonotone Armijo line search over the last `mem` values that
    also rejects non-finite f (BCE outside the log domain).
    * Exact reduced coordinates: with z0 = 0 every gradient lies in range(Y^T) and the ball projection only rescales,
      so iterates stay in span(Y^T); an orthonormal basis Q (reduced QR of Y^T) is used and results mapped back.
    * Sync-light: the line search evaluates all step fractions in LAMS at once through T_c = T + lam * (D Y^T) (one
      matmul per iteration plus one for the gradient); the largest admissible fraction is taken, otherwise the step is
      rejected and the BB step halved. T is recomputed exactly and convergence (rel KKT <= tol_rel) checked every
      SPG_CHECK iterations. Residual reported in the full space."""
    B, M = A.shape
    c, mu, muQ, delta = moments(A, Y)
    g0 = grad0_norm(gen, delta).clamp(min=1e-300)
    Qb, _ = torch.linalg.qr(Y.t())                     # d x M
    Yr = Y @ Qb                                         # M x M reduced atoms (exact)
    nl = len(LAMS)
    lams = torch.tensor(LAMS, dtype=A.dtype, device=A.device).view(nl, 1, 1)
    ar = torch.arange(B, device=A.device)
    Z = torch.zeros(B, Yr.shape[1], dtype=A.dtype, device=A.device)
    T = Z @ Yr.t()
    fZ = fT(gen, T, A, c)
    G = wT(gen, T, A, c) @ Yr
    hist = fZ.unsqueeze(1).repeat(1, mem)
    alpha = 1.0 / G.norm(dim=1).clamp(min=1e-300)      # first trial step reaches the unit sphere
    active = torch.ones(B, dtype=torch.bool, device=A.device)
    it = torch.zeros(B, dtype=torch.long, device=A.device)
    k = 0
    while k < cap:
        for _ in range(SPG_CHECK):
            Dd = proj_ball(Z + alpha.unsqueeze(1) * G) - Z
            gd = (G * Dd).sum(1)
            TD = Dd @ Yr.t()
            Tc = T.unsqueeze(0) + lams * TD.unsqueeze(0)                         # nl x B x M
            fc = fT(gen, Tc, A, c)                                              # nl x B
            fmax = hist.max(1).values
            okc = torch.isfinite(fc) & (fc >= fmax.unsqueeze(0) + 1e-4 * lams.view(nl, 1) * gd.unsqueeze(0))
            first = torch.argmax(okc.to(torch.int8), dim=0)                     # largest admissible fraction
            upd = okc.any(0) & active
            lam = lams.view(nl)[first]
            Zn = torch.where(upd.unsqueeze(1), Z + lam.unsqueeze(1) * Dd, Z)
            Tn = torch.where(upd.unsqueeze(1), Tc[first, ar], T)
            fZ = torch.where(upd, fc[first, ar], fZ)
            Gn = wT(gen, Tn, A, c) @ Yr
            s_ = Zn - Z
            y_ = G - Gn                                                         # gradient of the convex -f
            sy = (s_ * y_).sum(1)
            bb = (s_ * s_).sum(1) / sy.clamp(min=1e-300) if k % 2 == 0 else sy / (y_ * y_).sum(1).clamp(min=1e-300)
            bb = torch.where(sy > 0, bb, 2 * alpha).clamp(1e-12, 1e12)
            alpha = torch.where(upd, bb, torch.where(active, 0.5 * alpha, alpha))
            Z, T, G = Zn, Tn, torch.where(upd.unsqueeze(1), Gn, G)
            hist = torch.where(upd.unsqueeze(1), torch.cat([hist[:, 1:], fZ.unsqueeze(1)], 1), hist)
            it = it + active.long()
            k += 1
        T = Z @ Yr.t()                                                          # exact refresh (no drift)
        fZ = fT(gen, T, A, c)
        G = wT(gen, T, A, c) @ Yr
        res, _ = kkt(Z, G)
        active = active & ~(res / g0 <= tol_rel)
        if not bool(active.any()):
            break
    Zf = Z @ Qb.t()
    Gf = wT(gen, Zf @ Y.t(), A, c) @ Y
    res, nu_hat = kkt(Zf, Gf)
    rel = res / g0
    info = dict(nu_hat=nu_hat, iters=it, cap=rel > tol_rel, res=res, rel=rel, interior=Zf.norm(dim=1) < 1 - 1e-9)
    if gen == 'bce':
        Tf = Zf @ Y.t()
        info['near_log_boundary'] = ((1 - Tf.abs()) < 2e-7).any(1)
    return Zf, info


# ----------------------------------------------------------------------------- dispatcher

@torch.no_grad()
def generate(gen, A, Y):
    """Returns (targets [B,d] same dtype as Y, unit rows; valid mask [B]; info dict of float64 tensors / ints).
    gen in GENERATORS. Inputs must be stop-grad (asserted)."""
    assert not A.requires_grad and not Y.requires_grad
    dt = Y.dtype
    A64, Y64 = A.double(), Y.double()
    c, mu, muQ, delta = moments(A64, Y64)
    mu_n = mu.norm(dim=1)
    mean_t = mu / mu_n.clamp(min=1e-300).unsqueeze(1)
    mean_ok = torch.isfinite(mean_t).all(1) & (mu_n > ZMIN)
    info = {'mu_norm': mu_n, 'delta_norm': delta.norm(dim=1)}
    if gen == 'mean':
        Z, zr = mu, None
        T = mean_t
        fb = torch.zeros_like(mean_ok)
    else:
        if gen == 'vcs':
            Z, zr = solve_vcs(A64, Y64)
        else:
            Z, zr = solve_spg(gen, A64, Y64)
        zn = Z.norm(dim=1)
        okz = torch.isfinite(Z).all(1) & (zn > ZMIN)
        fb = ~okz
        T = torch.where(okz.unsqueeze(1), Z / zn.clamp(min=1e-300).unsqueeze(1), mean_t)
        info.update({k: v for k, v in zr.items()})
        info['z_norm'] = zn
        f, _ = OBJ[gen]
        info['obj_raw'] = f(Z, A64, c, Y64)
        info['obj_norm'] = f(T, A64, c, Y64)
        info['obj_zero'] = f(torch.zeros_like(Z), A64, c, Y64)
        info['obj_mean'] = f(mean_t, A64, c, Y64)
    # a row is valid if its target exists: the generator solution, or (fallback) the mean target; else aux is off
    valid = mean_ok if gen == 'mean' else (~fb | mean_ok)
    valid = valid & torch.isfinite(T).all(1)
    info['fallback'] = fb
    info['Jv_raw'] = f_vcs(Z, A64, c, Y64)
    info['Jv_norm'] = f_vcs(T, A64, c, Y64)
    info['Jv_mean'] = f_vcs(mean_t, A64, c, Y64)
    info['cos_to_mean'] = (T * mean_t).sum(1)
    return T.to(dt), valid, info
