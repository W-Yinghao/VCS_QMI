"""R27 audit shared helpers (read-only; imports the frozen R27 code_v11 tree, never writes into it)."""
import os, sys, time, math, hashlib
import numpy as np
import torch

CODE = '/home/infres/yinwang/CS_QMI/handoff/r27_qtarget_20261029/code_v11'
if CODE not in sys.path:
    sys.path.insert(0, CODE)
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(1, HERE)

import r27_targets as T  # noqa: E402

DT = torch.float64


def sha(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def code_hashes():
    return {f: sha(os.path.join(CODE, f)) for f in sorted(os.listdir(CODE)) if f.endswith('.py')}


def own_hashes():
    return {f: sha(os.path.join(HERE, f)) for f in sorted(os.listdir(HERE)) if f.endswith('.py')}


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-300)


def angle(a, b):
    """Row-wise angle between directions (radians), 2*asin(||u-v||/2): accurate for tiny angles."""
    u, v = unit(a.double()), unit(b.double())
    return 2 * torch.asin(((u - v).norm(dim=-1) / 2).clamp(max=1.0))


def qstats(x):
    x = torch.as_tensor(x).double().flatten()
    m = torch.isfinite(x)
    n_nf = int((~m).sum())
    x = x[m]
    if x.numel() == 0:
        return dict(n=0, n_nonfinite=n_nf)
    qs = torch.quantile(x, torch.tensor([0.5, 0.9, 0.99], dtype=DT)) if x.numel() > 1 else x.repeat(3)
    return dict(n=int(x.numel()), n_nonfinite=n_nf, mean=float(x.mean()), median=float(qs[0]), p90=float(qs[1]),
                p99=float(qs[2]), min=float(x.min()), max=float(x.max()))


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def problem(kind, B, M, d, K=5, seed=0, scale_bank=1.5, dtype=DT):
    """Synthetic K-sparse candidate graph (same construction as code_v11/r27_cpu_check.problem)."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(8, d, generator=g, dtype=dtype)
    bank = unit(base[torch.randint(0, 8, (M,), generator=g)] + scale_bank * torch.randn(M, d, generator=g, dtype=dtype))
    q = unit(base[torch.randint(0, 8, (B,), generator=g)] + scale_bank * torch.randn(B, d, generator=g, dtype=dtype))
    idx = torch.topk(q @ bank.t(), K, 1).indices
    if kind == 'dup':
        idx[: B // 3] = idx[: B // 3, :1]
    cols, inv = torch.unique(idx.flatten(), return_inverse=True)
    inv = inv.view_as(idx)
    A = torch.zeros(B, cols.numel(), dtype=dtype)
    A.scatter_add_(1, inv, torch.full_like(inv, 1.0 / K, dtype=dtype))
    return A, bank[cols]


def slsqp_rows(gen, A, Y, rows, maxiter=3000):
    """scipy SLSQP for selected rows, in the exact reduced coordinates z = Q u (Q from QR of Y^T; the optimum lies in
    span(Y^T) because every gradient does). CPU float64, start u0 = 0 (production init). Out-of-domain (BCE log) values
    are mapped to -1e30. Returns list of dicts with z (torch, d) or None."""
    from scipy.optimize import minimize
    A = A.detach().double().cpu(); Y = Y.detach().double().cpu()
    c = A.mean(0)
    Qb, _ = torch.linalg.qr(Y.t())
    Yr = (Y @ Qb).numpy()
    out = []
    for i in rows:
        a = A[i].numpy(); cn = c.numpy()

        def fu(u):
            t = torch.from_numpy(Yr @ u).unsqueeze(0)
            v = float(T.fT(gen, t, torch.from_numpy(a).unsqueeze(0), torch.from_numpy(cn).unsqueeze(0))[0])
            return v if math.isfinite(v) else -1e30

        def gu(u):
            t = torch.from_numpy(Yr @ u).unsqueeze(0)
            w = T.wT(gen, t, torch.from_numpy(a).unsqueeze(0), torch.from_numpy(cn).unsqueeze(0))[0].numpy()
            g = w @ Yr
            return g if np.all(np.isfinite(g)) else np.zeros_like(u)

        t0 = time.time()
        r = minimize(lambda u: -fu(u), np.zeros(Yr.shape[1]), jac=lambda u: -gu(u), method='SLSQP',
                     constraints={'type': 'ineq', 'fun': lambda u: 1 - u @ u, 'jac': lambda u: -2 * u},
                     options={'ftol': 1e-14, 'maxiter': maxiter})
        dt = time.time() - t0
        feas = bool(r.x @ r.x <= 1 + 1e-9 and math.isfinite(fu(r.x)) and fu(r.x) > -1e29)
        z = (Qb @ torch.from_numpy(r.x)) if feas else None
        out.append(dict(row=int(i), z=z, f=fu(r.x) if feas else None, success=bool(r.success), nit=int(r.nit),
                        message=str(r.message), seconds=dt, feasible=feas))
    return out


# ----------------------------------------------------------------------------- ranks (audit 3)

def recalls_from_ranks(ranks):
    r = np.asarray(ranks)
    return (100.0 * (r < 1).sum() / len(r), 100.0 * (r < 5).sum() / len(r), 100.0 * (r < 10).sum() / len(r))


def ranks_independent(sims, per_captions):
    """Independent rank computation (no argsort): i2t rank = #captions scoring strictly above the best GT caption;
    t2i rank = #images scoring strictly above the GT image. Equals the host argsort rank when there are no ties."""
    sims = np.asarray(sims)
    N = sims.shape[0]
    gt = sims.reshape(N, N, per_captions)[np.arange(N), np.arange(N)]          # N x per  (GT caption scores)
    best = gt.max(1)
    r_i2t = (sims > best[:, None]).sum(1)
    st = sims.T                                                                   # captions x images
    own = st[np.arange(N * per_captions), np.arange(N * per_captions) // per_captions]
    r_t2i = (st > own[:, None]).sum(1)
    return r_i2t, r_t2i
