"""R27-S constrained single-target generators (code_v12 only; DESIGN_S_CN.md §3).

Feasible set per row i:  C_i = { z : ||z|| <= 1,  p_i.z >= gamma },  gamma = cos(15 deg), p_i = stopgrad unit
GraphRefiner prototype. Objectives are the frozen R27 row objectives (r27_targets: delta.z for S3, f_vcs S4,
f_bce S5, f_snce S6) on the same candidate graph (A, Y); J, P/Q unchanged, only the feasible set changes.

Exact reduced coordinates: the objectives depend on z only through Y z, and both constraints prefer removing any
component orthogonal to span(Y, p_i), so z* lies in span(Y, p_i). With Q an orthonormal basis of span(Y^T) (reduced
QR, float64) write p_i = Q a_i + b_i e_i, e_i a unit vector orthogonal to span(Y^T); then z = Q u + v e_i, Y z = Yr u,
||z||^2 = ||u||^2 + v^2 and p_i.z = [a_i, b_i].[u, v]. All rows share the (M+1)-dim space R^M x R (rows differ only
through p~_i = [a_i, b_i]). Results are mapped back to d dims exactly.

Solvers (float64):
  S3 delta: closed form max delta.z over C (direction of delta if it is feasible, else the cap corner).
  S4/S5/S6: nonmonotone SPG (same algorithm, line search, cap and check cadence as r27_targets.solve_spg) with the exact
            ball-cap projection proj_ball_half (copied unchanged from r27_audit_20261029/audit_constrained.py, KKT-verified
            in PROJECTION_KKT_20261030.json); common init z0 = (1 + gamma)/2 * p_i (feasible, ||z0|| < 1, |F| < 1 so BCE
            starts inside the log domain).
Residual (all four, same definition): active-set KKT residual  min_{nu,kappa >= 0} ||g - nu z + kappa p||  over the
active constraints (ball: | ||z||-1 | <= 1e-9, cap: |p.z - gamma| <= 1e-9), rel = res / ||grad f(0)||. Stopping:
rel <= TOL_REL = 1e-6 (same tolerance as the R27 generators). For a concave max over a convex set the certificate
f* - f(z) <= 2 * res holds (same argument as R27, with complementary multipliers).
Output target T = normalize(z*) (raw ||z*|| >= gamma, so T stays in the 15-degree cone). Failure (non-finite, ||z*||
tiny, cap violation) -> fallback T = p_i (the native target), counted.
"""
import math
import torch
from r27_targets import moments, grad0_norm, fT, wT, TOL_REL, SPG_CAP, SPG_MEM, SPG_CHECK, LAMS, f_vcs

GAMMA = math.cos(math.radians(15.0))
ACT_TOL = 1e-9


def proj_ball_half(X, P, c):
    """Copied unchanged from r27_audit_20261029/audit_constrained.py (exact projection onto ||z||<=1, p.z>=c)."""
    c = torch.as_tensor(c, dtype=X.dtype, device=X.device)
    c = c.expand(X.shape[0]) if c.dim() == 0 else c
    a = (X * P).sum(1)
    Xp = X - a.unsqueeze(1) * P
    r = Xp.norm(dim=1)
    n = torch.sqrt(a * a + r * r)
    rmax = (1 - c * c).clamp(min=0).sqrt()
    inf = torch.full_like(a, math.inf)
    tol = 1e-12
    cand = []
    cand.append((a, r, (a >= c) & (n <= 1)))                                    # x itself
    s1 = torch.maximum(a, c)
    cand.append((s1, r, s1 * s1 + r * r <= 1 + tol))                            # half-plane only
    sc = n.clamp(min=1.0)
    cand.append((a / sc, r / sc, a / sc >= c - tol))                            # ball only
    cand.append((c, rmax, torch.ones_like(a, dtype=torch.bool)))                # both (corner)
    best_d, best_s, best_r = inf.clone(), torch.zeros_like(a), torch.zeros_like(a)
    for s, rho, feas in cand:
        d = torch.where(feas, (s - a) ** 2 + (rho - r) ** 2, inf)
        take = d < best_d
        best_d = torch.where(take, d, best_d)
        best_s = torch.where(take, s, best_s)
        best_r = torch.where(take, rho, best_r)
    U = Xp / r.clamp(min=1e-300).unsqueeze(1)
    U = torch.where((r > 0).unsqueeze(1), U, torch.zeros_like(U))
    return best_s.unsqueeze(1) * P + best_r.unsqueeze(1) * U


def kkt_cap(Z, G, P, gamma):
    """Active-set KKT residual for max f s.t. ||z||<=1, p.z>=gamma:  grad f = nu z - kappa p, nu, kappa >= 0."""
    nz = Z.norm(dim=1)
    pz = (Z * P).sum(1)
    on_b = (nz - 1).abs() <= ACT_TOL
    on_h = (pz - gamma).abs() <= ACT_TOL
    big = torch.full_like(nz, math.inf)
    # candidate fits (each with nonnegative multipliers); take the smallest admissible residual
    r_none = G.norm(dim=1)
    nu1 = ((Z * G).sum(1) / (nz * nz).clamp(min=1e-300)).clamp(min=0)
    r_b = (G - nu1.unsqueeze(1) * Z).norm(dim=1)
    ka1 = (-(P * G).sum(1)).clamp(min=0)                                         # ||p|| = 1
    r_h = (G + ka1.unsqueeze(1) * P).norm(dim=1)
    zz, zp, pp = (Z * Z).sum(1), pz, (P * P).sum(1)
    gz, gp = (G * Z).sum(1), (G * P).sum(1)
    det = zz * pp - zp * zp
    nu2 = (gz * pp - gp * zp) / det.clamp(min=1e-300)        # solve [z, -p][nu, kappa] = g in least squares
    ka2 = (gz * zp - gp * zz) / det.clamp(min=1e-300)
    ok2 = (nu2 >= 0) & (ka2 >= 0) & (det > 1e-14)
    r_bh = torch.where(ok2, (G - nu2.unsqueeze(1) * Z + ka2.unsqueeze(1) * P).norm(dim=1), big)
    res = r_none.clone()
    res = torch.where(on_b, torch.minimum(res, r_b), res)
    res = torch.where(on_h, torch.minimum(res, r_h), res)
    res = torch.where(on_b & on_h, torch.minimum(res, r_bh), res)
    nu = torch.where(on_b, torch.where(on_h & ok2 & (r_bh <= r_b), nu2, nu1), torch.zeros_like(nz))
    ka = torch.where(on_h, torch.where(on_b & ok2 & (r_bh <= r_h), ka2, ka1), torch.zeros_like(nz))
    return res, nu, ka, on_b, on_h


def augmented_basis(Y, P):
    """Q (d x M) basis of span(Y^T); per-row p~ = [a, b] (M+1) with p = Q a + b e; e (B x d) unit, orthogonal to Q."""
    Qb, _ = torch.linalg.qr(Y.t())
    a = P @ Qb
    rest = P - a @ Qb.t()
    b = rest.norm(dim=1)
    e = rest / b.clamp(min=1e-300).unsqueeze(1)
    e = torch.where((b > 1e-14).unsqueeze(1), e, torch.zeros_like(e))
    b = torch.where(b > 1e-14, b, torch.zeros_like(b))
    Pt = torch.cat([a, b.unsqueeze(1)], 1)
    Pt = Pt / Pt.norm(dim=1, keepdim=True)
    return Qb, e, Pt


def to_full(Zt, Qb, e):
    return Zt[:, :-1] @ Qb.t() + Zt[:, -1:] * e


def solve_cap_linear(A, Y, P, gamma=GAMMA):
    """max delta.z over C: delta-hat if p.delta-hat >= gamma, else gamma p + sqrt(1-gamma^2) w, w = unit(delta perp p)."""
    c, mu, muQ, delta = moments(A, Y)
    dn = delta.norm(dim=1)
    dh = delta / dn.clamp(min=1e-300).unsqueeze(1)
    pd = (P * dh).sum(1)
    w = dh - pd.unsqueeze(1) * P
    wn = w.norm(dim=1)
    w = w / wn.clamp(min=1e-300).unsqueeze(1)
    corner = gamma * P + math.sqrt(1 - gamma * gamma) * w
    Z = torch.where((pd >= gamma).unsqueeze(1), dh, corner)
    Z = torch.where(((dn > 1e-14) & ((pd >= gamma) | (wn > 1e-14))).unsqueeze(1), Z, gamma * P)  # degenerate -> feasible point
    G = delta
    res, nu, ka, on_b, on_h = kkt_cap(Z, G, P, gamma)
    rel = res / dn.clamp(min=1e-300)
    it = torch.zeros(A.shape[0], dtype=torch.long, device=A.device)
    return Z, dict(iters=it, cap=rel > TOL_REL, res=res, rel=rel, nu_hat=nu, kappa=ka, ball_active=on_b, cap_active=on_h,
                   interior=~on_b, g0=dn)


def solve_cap_spg(gen, A, Y, P, gamma=GAMMA, tol_rel=TOL_REL, cap=SPG_CAP, mem=SPG_MEM):
    """SPG (as r27_targets.solve_spg) on C_i in exact augmented coordinates (u, v) in R^M x R, init (1+gamma)/2 p_i."""
    B, M = A.shape
    c, mu, muQ, delta = moments(A, Y)
    g0 = grad0_norm(gen, delta).clamp(min=1e-300)
    Qb, e, Pt = augmented_basis(Y, P)
    Yr = Y @ Qb                                                     # M x M
    Ya = torch.cat([Yr, torch.zeros(M, 1, dtype=Y.dtype, device=Y.device)], 1)   # M x (M+1): Y e_i = 0
    proj = lambda Z_: proj_ball_half(Z_, Pt, gamma)
    nl = len(LAMS)
    lams = torch.tensor(LAMS, dtype=A.dtype, device=A.device).view(nl, 1, 1)
    ar = torch.arange(B, device=A.device)
    Z = 0.5 * (1 + gamma) * Pt
    T = Z @ Ya.t()
    fZ = fT(gen, T, A, c)
    G = wT(gen, T, A, c) @ Ya
    hist = fZ.unsqueeze(1).repeat(1, mem)
    alpha = 1.0 / G.norm(dim=1).clamp(min=1e-300)
    active = torch.ones(B, dtype=torch.bool, device=A.device)
    it = torch.zeros(B, dtype=torch.long, device=A.device)
    k = 0
    while k < cap:
        for _ in range(SPG_CHECK):
            Dd = proj(Z + alpha.unsqueeze(1) * G) - Z
            gd = (G * Dd).sum(1)
            TD = Dd @ Ya.t()
            Tc = T.unsqueeze(0) + lams * TD.unsqueeze(0)
            fc = fT(gen, Tc, A, c)
            fmax = hist.max(1).values
            okc = torch.isfinite(fc) & (fc >= fmax.unsqueeze(0) + 1e-4 * lams.view(nl, 1) * gd.unsqueeze(0))
            first = torch.argmax(okc.to(torch.int8), dim=0)
            upd = okc.any(0) & active
            lam = lams.view(nl)[first]
            Zn = torch.where(upd.unsqueeze(1), Z + lam.unsqueeze(1) * Dd, Z)
            Tn = torch.where(upd.unsqueeze(1), Tc[first, ar], T)
            fZ = torch.where(upd, fc[first, ar], fZ)
            Gn = wT(gen, Tn, A, c) @ Ya
            s_ = Zn - Z
            y_ = G - Gn
            sy = (s_ * y_).sum(1)
            bb = (s_ * s_).sum(1) / sy.clamp(min=1e-300) if k % 2 == 0 else sy / (y_ * y_).sum(1).clamp(min=1e-300)
            bb = torch.where(sy > 0, bb, 2 * alpha).clamp(1e-12, 1e12)
            alpha = torch.where(upd, bb, torch.where(active, 0.5 * alpha, alpha))
            Z, T, G = Zn, Tn, torch.where(upd.unsqueeze(1), Gn, G)
            hist = torch.where(upd.unsqueeze(1), torch.cat([hist[:, 1:], fZ.unsqueeze(1)], 1), hist)
            it = it + active.long()
            k += 1
        Z = proj(Z)                                                 # exact feasibility (removes ~1e-16 drift)
        T = Z @ Ya.t()
        fZ = fT(gen, T, A, c)
        G = wT(gen, T, A, c) @ Ya
        res, _, _, _, _ = kkt_cap(Z, G, Pt, gamma)
        active = active & ~(res / g0 <= tol_rel)
        if not bool(active.any()):
            break
    Zf = to_full(Z, Qb, e)
    Gf = wT(gen, Zf @ Y.t(), A, c) @ Y
    res, nu, ka, on_b, on_h = kkt_cap(Zf, Gf, P, gamma)
    rel = res / g0
    return Zf, dict(iters=it, cap=rel > tol_rel, res=res, rel=rel, nu_hat=nu, kappa=ka, ball_active=on_b, cap_active=on_h,
                    interior=~on_b, g0=g0, p_outside_span=float((P - (P @ Qb) @ Qb.t()).norm(dim=1).max()))


SOLVERS = {'cap_delta': None, 'cap_vcs': 'vcs', 'cap_bce': 'bce', 'cap_snce': 'snce'}


@torch.no_grad()
def generate_cap(kind, A, Y, P, gamma=GAMMA):
    """Returns (T [B,d] unit targets in Y.dtype, valid [B] (always True: fallback is p), info)."""
    assert not A.requires_grad and not Y.requires_grad and not P.requires_grad
    dt = Y.dtype
    A64, Y64, P64 = A.double(), Y.double(), P.double()
    P64 = P64 / P64.norm(dim=1, keepdim=True)
    c, mu, muQ, delta = moments(A64, Y64)
    if kind == 'cap_delta':
        Z, info = solve_cap_linear(A64, Y64, P64, gamma)
        fobj = lambda Z_: (delta * Z_).sum(1)
    else:
        gen = SOLVERS[kind]
        Z, info = solve_cap_spg(gen, A64, Y64, P64, gamma)
        fobj = lambda Z_: fT(gen, Z_ @ Y64.t(), A64, c)
    zn = Z.norm(dim=1)
    viol = torch.maximum((zn - 1).clamp(min=0), (gamma - (Z * P64).sum(1)).clamp(min=0))
    okz = torch.isfinite(Z).all(1) & (zn > 1e-6) & (viol <= 1e-9)
    T = torch.where(okz.unsqueeze(1), Z / zn.clamp(min=1e-300).unsqueeze(1), P64)
    info['fallback'] = ~okz
    info['z_norm'] = zn
    info['violation'] = viol
    info['cos_to_p'] = (T * P64).sum(1)
    info['obj_raw'] = fobj(Z)
    info['obj_norm'] = fobj(T)
    info['obj_p'] = fobj(P64)
    info['Jv_raw'] = f_vcs(Z, A64, c, Y64)
    info['Jv_norm'] = f_vcs(T, A64, c, Y64)
    info['Jv_p'] = f_vcs(P64, A64, c, Y64)
    valid = torch.isfinite(T).all(1)
    return T.to(dt), valid, info
