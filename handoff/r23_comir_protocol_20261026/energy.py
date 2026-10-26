"""Patch-restricted host energies (masked). With the mask disabled and the host positions/boxes/canvases they are the host
Q.RigidEnergy / A.MFRigidEnergy / MLPRigidEnergy / DescriptorRigidEnergy / soft-MI energies (checked in run.py check).

A position x contributes to candidate (theta, t) only if T(x) lies inside the moving patch by the method's receptive-field
margin; every average (positive term, reference/product term, histogram) is over valid positions / valid (x, pi(x)) pairs."""
import math
import numpy as np
from common import torch, F, Q, P, CEN, HALF_PIX, MOV_MARGIN

MIN_VALID = 64


def boxes_for(host):
    if host:
        return Q.BOX1, Q.BOX2
    lo, hi = int(CEN - HALF_PIX), int(CEN + HALF_PIX)                     # 152..359: positions +-86.5 plus 17 px
    return (lo, hi), (lo - 40, hi + 40)                                    # grid path: shifts up to 28*sqrt2 < 40


class PatchEnergy:
    def __init__(self, kind, x, perms, *, ref_feat, mov_map, scorer, eV=None, mov_margin=None, host=False):
        """kind: 'learned' (warp-then-encode with eV; ref_feat = reference descriptors at x) or 'sample' (encode-then-sample:
        mov_map is the moving descriptor/latent/intensity map; ref_feat = reference values at x).  scorer(ref_feat, v, m) -> [K]."""
        self.kind, self.eV, self.scorer, self.host = kind, eV, scorer, host
        dev = mov_map.device
        self.x = torch.as_tensor(x, dtype=torch.float32, device=dev)
        self.perms = torch.as_tensor(perms, device=dev)
        self.u, self.mov = ref_feat, mov_map
        b1, b2 = boxes_for(host)
        self.b1, self.b2 = b1, b2
        self.box1, self.box2 = Q.grid_coords(torch, *b1, dev), Q.grid_coords(torch, *b2, dev)
        self.half = math.inf if host else HALF_PIX - (mov_margin if mov_margin is not None else MOV_MARGIN[kind if kind != 'sample' else 'mi'])

    def mask(self, th, t):
        T = Q.apply_T(torch, th, t, self.x)                                # [K, N, 2]
        if math.isinf(self.half):
            return torch.ones(T.shape[:2], dtype=self.x.dtype, device=self.x.device)
        return ((T - CEN).abs() <= self.half).all(-1).to(self.x.dtype)

    def _finish(self, val, m):
        bad = m.sum(-1) < MIN_VALID
        return torch.where(bad, torch.full_like(val, -1e9), val) if bad.any() else val

    def batch(self, th, t):
        m = self.mask(th, t)
        if self.kind == 'learned':
            n = self.b1[1] - self.b1[0] + 1; dv = []
            for k in range(len(th)):
                W = Q.sample_pts(torch, F, self.mov, Q.apply_T(torch, th[k:k + 1], t[k:k + 1], self.box1))[0, :, 0].reshape(1, 1, n, n)
                dv.append(P.sample_map(torch, F, self.eV(W)[0], self.x - self.b1[0]))
            v = torch.stack(dv)
        else:
            v = Q.sample_pts(torch, F, self.mov, Q.apply_T(torch, th, t, self.x))
        return self._finish(self.scorer(self.u, v, m, self.perms), m)

    def grid_values(self, th_deg, ts, chunk=64):
        dev = ts.device; th = torch.full((1,), math.radians(th_deg), device=dev)
        if self.kind != 'learned':
            return torch.cat([self.batch(th.expand(len(b)), b) for b in ts.split(chunk)])
        zero = torch.zeros(1, 2, device=dev); n = self.b2[1] - self.b2[0] + 1
        W = Q.sample_pts(torch, F, self.mov, Q.apply_T(torch, th, zero, self.box2))[0, :, 0].reshape(1, 1, n, n); D = self.eV(W)[0]
        c, s = math.cos(math.radians(th_deg)), math.sin(math.radians(th_deg))
        out = []
        for b in ts.split(chunk):
            sh = torch.stack([c * b[:, 0] + s * b[:, 1], -s * b[:, 0] + c * b[:, 1]], -1)
            v = Q.sample_pts(torch, F, D, self.x[None] + sh[:, None] - self.b2[0])
            m = self.mask(th.expand(len(b)), b)
            out.append(self._finish(self.scorer(self.u, v, m, self.perms), m))
        return torch.cat(out)

    def __call__(self, p):
        return self.batch(p[0:1] / Q.RHO, p[1:3][None])[0]


# ------------------------------------------------------------------------------------------------------------ scorers
def mf_scorer(head):
    """Masked fixed-W MF moments energy (A.mf_energy_rows 'moments' with averages over valid positions)."""
    W = head.W
    def score(u, dv, m, perms):
        A_ = head.features_a(u); B = head.features_b(dv); Nv = m.sum(-1).clamp_min(1.0)
        AW = A_ @ W; t = (AW[None] * B).sum(-1)
        p1, p2 = (m * t).sum(-1) / Nv, (m * t.square()).sum(-1) / Nv
        Abar, Bbar = (m @ A_) / Nv[:, None], (m[..., None] * B).sum(1) / Nv[:, None]
        q1 = ((Abar @ W) * Bbar).sum(-1)
        GA = torch.einsum('kn,nr,nq->krq', m, A_, A_); GB = torch.einsum('kn,kns,knt->kst', m, B, B)
        q2 = ((GA @ W[None] @ GB) * W[None]).sum((-2, -1)) / Nv.square()
        return p1 - q1 - 0.5 * (p2 + q2)
    return score


def mlp_scorer(critic, draws):
    """Masked supplement mlp_energy_rows: J(tanh f), P over valid x, Q over pairs (x, pi(x)) with both ends valid."""
    def score(u, dv, m, perms):
        ue = u[None].expand(dv.shape[0], -1, -1); Nv = m.sum(-1).clamp_min(1.0)
        tp = torch.tanh(critic(torch.cat((ue, dv), -1)).squeeze(-1))
        pos = (m * (tp - 0.5 * tp.square())).sum(-1) / Nv
        neg = torch.zeros_like(pos)
        for perm in perms[:draws]:
            mq = m * m[:, perm]
            tq = torch.tanh(critic(torch.cat((ue, dv[:, perm]), -1)).squeeze(-1))
            neg = neg + (mq * (tq + 0.5 * tq.square())).sum(-1) / mq.sum(-1).clamp_min(1.0)
        return pos - neg / draws
    return score


def mse_scorer(channel0):
    """CoMIR (host 'comir': channel 0, -mean squared difference) or MIND (all channels) masked."""
    def score(u, v, m, perms):
        if channel0:
            d = (u[None, :, 0] - v[..., 0]).square()
        else:
            d = (u[None] - v).square().mean(-1)
        return -(m * d).sum(-1) / m.sum(-1).clamp_min(1.0)
    return score


def mi_scorer(ranges, bins=64, sigma=1.0):
    """Masked host soft_mi_batch (Parzen-smoothed 64-bin joint histogram)."""
    (ul, uh), (vl, vh) = ranges
    def score(u, v, m, perms):
        uu, vv = u[:, 0], v[..., 0]; c = torch.arange(bins, device=uu.device, dtype=uu.dtype)
        cu = (uu - ul) / (uh - ul) * (bins - 1); cv = (vv - vl) / (vh - vl) * (bins - 1)
        wu = torch.exp(-0.5 * ((cu[:, None] - c) / sigma) ** 2); wu = wu / wu.sum(-1, keepdim=True).clamp_min(1e-12)
        wv = torch.exp(-0.5 * ((cv[..., None] - c) / sigma) ** 2); wv = wv / wv.sum(-1, keepdim=True).clamp_min(1e-12)
        Pj = torch.einsum('kn,nb,knc->kbc', m, wu, wv) / m.sum(-1).clamp_min(1.0)[:, None, None]
        ent = lambda p: -(p * torch.log(p.clamp_min(1e-12))).sum(-1)
        return ent(Pj.sum(2)) + ent(Pj.sum(1)) - ent(Pj.reshape(Pj.shape[0], -1))
    return score


# ------------------------------------------------------------------------------------------------------------ search
def optimise(energy, dev, thetas, ts, t_clamp, th_clamp, n_starts=None, n_adam=None, lr=None):
    """Host Q.optimise_rigid with the patch grid and a square translation box: grid -> best n_starts -> Adam on (u, ty, tx)."""
    n_starts = n_starts or Q.P.N_STARTS; n_adam = n_adam or Q.P.N_ADAM; lr = lr or Q.P.ADAM_LR
    tt = torch.tensor(ts, dtype=torch.float32, device=dev); umax = math.radians(th_clamp) * Q.RHO
    with torch.no_grad():
        vals = np.concatenate([energy.grid_values(float(th), tt).float().cpu().numpy() for th in thetas])
    vals = np.nan_to_num(vals, nan=-np.inf); best = []
    for i in np.argsort(vals)[::-1][:n_starts]:
        th0, t0 = thetas[i // len(ts)], ts[i % len(ts)]
        p = torch.tensor([math.radians(th0) * Q.RHO, t0[0], t0[1]], dtype=torch.float32, device=dev, requires_grad=True)
        opt = torch.optim.Adam([p], lr=lr)
        for _ in range(n_adam):
            opt.zero_grad(); (-energy(p)).backward(); opt.step()
            with torch.no_grad():
                p[0].clamp_(-umax, umax); p[1:].clamp_(-t_clamp, t_clamp)
        with torch.no_grad():
            best.append((float(energy(p)), p.detach().cpu().numpy().astype(np.float64)))
    f, p = max(best, key=lambda z: z[0])
    return math.degrees(p[0] / Q.RHO), p[1:], len(vals), f
