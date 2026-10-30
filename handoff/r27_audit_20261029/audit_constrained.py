"""R27 audit: SERVER_ADJUSTMENTS_CN.md §5 offline comparison (no training) + task-identity scoring helpers.

1. proj_ball_half(X, P, c): exact Euclidean projection of each row x onto C = {z : ||z|| <= 1, p.z >= c}
   (p unit, 0 <= c < 1). Write x = a p + x_perp, r = ||x_perp||. The problem reduces to the 2-D set
   D = {(s, rho) : s >= c, s^2 + rho^2 <= 1, rho >= 0} (the perpendicular direction of x is kept). By KKT the projection
   is one of: x itself (no active constraint), the half-plane projection (max(a, c), r) (only p.z >= c active), the disk
   projection (a, r)/max(1, ||(a, r)||) (only the ball active), or the corner (c, sqrt(1 - c^2)) (both active). The
   projection is the feasible candidate closest to x (every feasible candidate is no closer than the projection).
   Verified against a generic solver and by the variational inequality in audit_check2.py.
2. solve_constrained(gen, A, Y, P, c): max f_gen(z) s.t. ||z|| <= 1, p_i.z >= c for every row i, with the SAME row
   objectives as production (r27_targets.OBJ: vcs / bce / snce; J and P/Q unchanged). Accelerated projected gradient
   (copy of r27_targets.solve_fista with proj_ball -> proj_ball_half), z0 = c p_i (= projection of 0; feasible, finite for
   BCE because |z0.y| <= c < 1). Stopping: projected-gradient fixed-point residual with step s = 1/L0,
   res = ||Proj(z + s g) - z|| / s, rel = res / ||grad f(0)|| <= tol (zero iff z is optimal for the constrained problem).
3. task_scores(Tg, E, truth_mask): offline task-identity evaluation of unit targets against a pool of re-encoded
   counterparts: score of the best true counterpart vs the strongest competitor (rank = #competitors scoring strictly
   higher, margin = true - best competitor).
"""
import math
import torch
from r27_targets import OBJ, moments, grad0_norm

L0 = {'vcs': 2.0, 'bce': 1.0, 'snce': 20.0}   # same initial Lipschitz guesses as r27_targets.solve_fista


def proj_ball_half(X, P, c):
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


@torch.no_grad()
def solve_constrained(gen, A, Y, P, cth, tol_rel=1e-6, cap=20000, check_every=5):
    f, g = OBJ[gen]
    B = A.shape[0]
    c, mu, muQ, delta = moments(A, Y)
    g0 = grad0_norm(gen, delta).clamp(min=1e-300)
    proj = lambda Z_: proj_ball_half(Z_, P, cth)
    Z = proj(torch.zeros_like(mu))                                   # = cth * P
    X = Z.clone()
    fZ = f(Z, A, c, Y)
    t = torch.ones(B, dtype=A.dtype, device=A.device)
    L = torch.full((B,), L0[gen], dtype=A.dtype, device=A.device)
    s_res = 1.0 / L0[gen]
    active = torch.ones(B, dtype=torch.bool, device=A.device)
    it = torch.zeros(B, dtype=torch.long, device=A.device)
    for k in range(cap):
        gX = g(X, A, c, Y)
        fX = f(X, A, c, Y)
        for _ in range(60):
            Zn = proj(X + gX / L.unsqueeze(1))
            fn = f(Zn, A, c, Y)
            D = Zn - X
            okr = torch.isfinite(fn) & (fn >= fX + (gX * D).sum(1) - 0.5 * L * (D * D).sum(1) - 1e-12 * fX.abs())
            okr = okr | ~active
            if bool(okr.all()):
                break
            L = torch.where(okr, L, 2 * L)
        restart = fn < fZ
        tn = 0.5 * (1 + torch.sqrt(1 + 4 * t * t))
        mom = ((t - 1) / tn).unsqueeze(1)
        Xn = proj(torch.where(restart.unsqueeze(1), Zn, Zn + mom * (Zn - Z)))
        if gen == 'bce':
            okx = torch.isfinite(f(Xn, A, c, Y))
            Xn = torch.where(okx.unsqueeze(1), Xn, Zn)
        a_ = active.unsqueeze(1)
        Z = torch.where(a_, Zn, Z); fZ = torch.where(active, fn, fZ); X = torch.where(a_, Xn, X)
        t = torch.where(active, torch.where(restart, torch.ones_like(tn), tn), t)
        L = torch.where(active, 0.9 * L, L)
        it = it + active.long()
        if (k + 1) % check_every == 0:
            res = (proj(Z + s_res * g(Z, A, c, Y)) - Z).norm(dim=1) / s_res
            active = active & ~(res / g0 <= tol_rel)
            if not bool(active.any()):
                break
    G = g(Z, A, c, Y)
    res = (proj(Z + s_res * G) - Z).norm(dim=1) / s_res
    rel = res / g0
    viol = torch.maximum((Z.norm(dim=1) - 1).clamp(min=0), (cth - (Z * P).sum(1)).clamp(min=0))
    return Z, dict(iters=it, cap=rel > tol_rel, rel=rel, res=res, violation=viol,
                   half_active=((Z * P).sum(1) - cth).abs() < 1e-9, ball_active=(Z.norm(dim=1) - 1).abs() < 1e-9)


def task_scores(Tg, E, truth):
    """Tg [B,d] unit targets, E [U,d] pool, truth [B,U] bool (true counterpart IDs of row i; >=1 per row).
    Returns rank (#competitors strictly above the best true score), margin (best true - best competitor), true score."""
    S = Tg @ E.t()
    ninf = torch.full_like(S, -math.inf)
    st = torch.where(truth, S, ninf).max(1).values
    sc = torch.where(truth, ninf, S)
    rank = (sc > st.unsqueeze(1)).sum(1)
    comp = sc.max(1).values
    return rank, st - comp, st
