"""R27 audit (supplement v1.1, SERVER_ADDENDUM 只读审计): SPG history-reference variant. NOT used in production.

Copy of production r27_targets.solve_spg (code_v11 == code, sha256 b94efbcf...) with one functional change:
  production  : fmax = hist.max(1).values   (concave MAXIMIZATION compared against the history MAX -> each accepted step
                must exceed the best of the last `mem` values, i.e. effectively monotone Armijo ascent)
  ref='min'   : reference = hist.min(1).values (the maximization analogue of Grippo/BMR nonmonotone minimization, which
                compares against the history max of the minimized function)
  ref='max'   : reproduces production exactly (checked bitwise in audit_check.py).
Other edits are diagnostics only (n_decrease_accepts, n_rejects, history_ref in info). Full diff: SPG_DIFF.patch.
"""
import math
import torch
from r27_targets import (TOL_REL, SPG_CAP, SPG_MEM, SPG_CHECK, LAMS, moments, grad0_norm, kkt, proj_ball, fT, wT)


def solve_spg_ref(gen, A, Y, tol_rel=TOL_REL, cap=SPG_CAP, mem=SPG_MEM, ref='min'):
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
    n_dec = torch.zeros(B, dtype=torch.long, device=A.device)   # AUDIT: accepted steps with f decrease
    n_rej = torch.zeros(B, dtype=torch.long, device=A.device)   # AUDIT: active rows with no admissible fraction
    while k < cap:
        for _ in range(SPG_CHECK):
            Dd = proj_ball(Z + alpha.unsqueeze(1) * G) - Z
            gd = (G * Dd).sum(1)
            TD = Dd @ Yr.t()
            Tc = T.unsqueeze(0) + lams * TD.unsqueeze(0)                         # nl x B x M
            fc = fT(gen, Tc, A, c)                                              # nl x B
            fmax = hist.max(1).values if ref == 'max' else hist.min(1).values   # AUDIT: 'max' = production; 'min' = nonmonotone ascent
            okc = torch.isfinite(fc) & (fc >= fmax.unsqueeze(0) + 1e-4 * lams.view(nl, 1) * gd.unsqueeze(0))
            first = torch.argmax(okc.to(torch.int8), dim=0)                     # largest admissible fraction
            upd = okc.any(0) & active
            lam = lams.view(nl)[first]
            n_dec += (upd & (fc[first, ar] < fZ)).long(); n_rej += (active & ~upd).long()   # AUDIT counters
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
    info = dict(nu_hat=nu_hat, iters=it, cap=rel > tol_rel, res=res, rel=rel, interior=Zf.norm(dim=1) < 1 - 1e-9,
                n_decrease_accepts=n_dec, n_rejects=n_rej, history_ref=ref)   # AUDIT: extra diagnostics
    if gen == 'bce':
        Tf = Zf @ Y.t()
        info['near_log_boundary'] = ((1 - Tf.abs()) < 2e-7).any(1)
    return Zf, info
