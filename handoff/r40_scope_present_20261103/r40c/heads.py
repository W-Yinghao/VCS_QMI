"""R38: R37 heads.py + DirGHead (structure 'GD', R38 directional readout; GHead/build_heads('G') unchanged).
R36 (copied from R35 heads.py; NullSlots added for block N) per-peer critic heads. Every head starts from the common FIT-only native calibration g0 = a0*s + b0
(s = native H score of that peer = dot product of native unit vectors), so the step-0 peer-mean score is a
positive affine map of the native peer-mean score (rank identical).

A-ID  g = a * <u', v'> + b,  u = x + B_x A_x x (rank 64, B=0 init), u' = u * |x|/|u|   (|x| = 1 up to fp32;
      the rescale to the native vector's own norm makes the step-0 score bitwise a*s+b; == cos(u,v)*|x||y|).
A-G   g = a * s + b + out(h(phi)), phi = [x, y, x*y, |x-y|, s], h: 4097-256-GELU-128-GELU, out zero-init.
C-G   identical to A-G (fresh init).
C-L   g = a * s + b + out([h(phi), m_r, m_t]); m_r/m_t = masked means of the region/token cross-attention
      features (package reference LocalResidual, h=128, tau=0.1); out zero-init.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


class PositiveAffine(nn.Module):
    """a = softplus(raw_a) > 0 (package reference parametrisation), b free."""
    def __init__(self, a: float, b: float):
        super().__init__()
        if not a > 0:
            raise ValueError('a must be positive')
        raw = a + math.log(-math.expm1(-a))
        self.raw_a = nn.Parameter(torch.tensor(float(raw)))
        self.b = nn.Parameter(torch.tensor(float(b)))

    def ab(self):
        return F.softplus(self.raw_a), self.b

    def forward(self, s):
        a, b = self.ab()
        return a * s + b


class IDHead(nn.Module):
    kind = 'ID'

    def __init__(self, a0, b0, d=1024, rank=64):
        super().__init__()
        self.ax = nn.Linear(d, rank, bias=False); self.bx = nn.Linear(rank, d, bias=False)
        self.ay = nn.Linear(d, rank, bias=False); self.by = nn.Linear(rank, d, bias=False)
        nn.init.zeros_(self.bx.weight); nn.init.zeros_(self.by.weight)   # A: torch default (small) init
        self.affine = PositiveAffine(a0, b0)

    def tx(self, x):
        u = x + self.bx(self.ax(x))
        return u * (x.norm(dim=-1, keepdim=True) / u.norm(dim=-1, keepdim=True))

    def ty(self, y):
        v = y + self.by(self.ay(y))
        return v * (y.norm(dim=-1, keepdim=True) / v.norm(dim=-1, keepdim=True))

    def matrix(self, x, y):
        """[n_x, n_y] scores (training: in-batch matrix)."""
        return self.affine(self.tx(x) @ self.ty(y).T)


class GlobalMLP(nn.Module):
    def __init__(self, d=1024, h1=256, h2=128):
        super().__init__()
        self.d = d
        self.l1 = nn.Linear(4 * d + 1, h1)
        self.l2 = nn.Linear(h1, h2)

    def hidden(self, x, y, s):
        """paired x,y [N,d], s [N] -> [N,h2]. phi = [x, y, x*y, |x-y|, s]."""
        phi = torch.cat([x, y, x * y, (x - y).abs(), s[:, None]], -1)
        return F.gelu(self.l2(F.gelu(self.l1(phi))))


class LocalBranch(nn.Module):
    """Package reference LocalResidual body (projection 128 + L2, bidirectional cross-attention tau 0.1,
    [p, q, p*q, |p-q|] -> Linear(512,128)+GELU, masked means). Split into projection (per unique item)
    and pair features (per pair) so projections are computed once per item."""
    def __init__(self, d=1024, h=128, tau=.1):
        super().__init__()
        if tau <= 0:
            raise ValueError('tau must be positive')
        self.tau = tau
        self.vi = nn.Linear(d, h); self.wt = nn.Linear(d, h)
        self.mix = nn.Sequential(nn.Linear(4 * h, h), nn.GELU())

    def proj_regions(self, regions):
        return F.normalize(self.vi(regions), dim=-1)

    def proj_words(self, words):
        return F.normalize(self.wt(words), dim=-1)

    def pair_features(self, v, w, region_mask, word_mask):
        if not bool(region_mask.any(1).all()) or not bool(word_mask.any(1).all()):
            raise ValueError('Each sequence needs a valid element')
        sim = torch.einsum('brd,btd->brt', v, w) / self.tau
        ar = sim.masked_fill(~word_mask[:, None, :], -torch.inf).softmax(-1)
        aw = sim.transpose(1, 2).masked_fill(~region_mask[:, None, :], -torch.inf).softmax(-1)
        vw = ar @ w; wv = aw @ v
        vr = self.mix(torch.cat([v, vw, v * vw, (v - vw).abs()], -1))
        wt = self.mix(torch.cat([w, wv, w * wv, (w - wv).abs()], -1))
        rm = region_mask[:, :, None].to(vr.dtype); tm = word_mask[:, :, None].to(wt.dtype)
        mr = (vr * rm).sum(1) / rm.sum(1)
        mt = (wt * tm).sum(1) / tm.sum(1)
        return torch.cat([mr, mt], -1)


class GHead(nn.Module):
    """A-G / C-G (local=False) and C-L (local=True)."""
    def __init__(self, a0, b0, d=1024, local=False):
        super().__init__()
        self.kind = 'L' if local else 'G'
        self.affine = PositiveAffine(a0, b0)
        self.glob = GlobalMLP(d)
        self.local = LocalBranch(d) if local else None
        self.out = nn.Linear(128 + (256 if local else 0), 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def paired(self, x, y, s, local_feats=None):
        """x,y [N,d] native unit vectors, s [N] native score of the pair; local_feats [N,256] for C-L."""
        h = self.glob.hidden(x, y, s)
        if self.local is not None:
            h = torch.cat([h, local_feats], -1)
        return self.affine(s) + self.out(h).squeeze(-1)

    def matrix(self, x, y):
        """in-batch [n,m] scores for A-G training (s from the same matmul as native)."""
        n, m = len(x), len(y)
        s = x @ y.T
        xx = x[:, None, :].expand(n, m, x.shape[1]).reshape(n * m, -1)
        yy = y[None, :, :].expand(n, m, y.shape[1]).reshape(n * m, -1)
        h = self.glob.hidden(xx, yy, s.reshape(-1))
        return self.affine(s) + self.out(h).reshape(n, m)


def build_heads(structure, a0, b0, seed):
    """Two independent per-peer heads, same structure and init rule. Peer k is built after
    torch.manual_seed(10*seed + k) (k=0 for A, 1 for B): identical across objectives, different between peers,
    and the global MLP of C-L equals that of C-G (it is created before the local branch)."""
    mk = {'ID': lambda: IDHead(a0, b0), 'G': lambda: GHead(a0, b0), 'L': lambda: GHead(a0, b0, local=True), 'GD': lambda: DirGHead(a0, b0)}[structure]
    heads = {}
    for k, p in enumerate(('A', 'B')):
        torch.manual_seed(10 * seed + k)
        heads[p] = mk()
    return nn.ModuleDict(heads)


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


class NullSlots(nn.Module):
    """R36 block N: one learnable scalar score per peer and per direction (index 0 = i2t, 1 = t2i) for the symbolic
    slot "no OBSERVED positive in this list". Initialised to 0; takes no input (never sees observed labels, noise
    flags or truth). Built without consuming RNG, so the real-slot heads are bitwise those of the same-seed C head."""
    def __init__(self):
        super().__init__()
        self.v = nn.ParameterDict({p: nn.Parameter(torch.zeros(2)) for p in ('A', 'B')})

    def score(self, peer, direction, n):
        return self.v[peer][direction].expand(n)


class DirGHead(GHead):
    """R38 directional G: g_d = a_d * s + b_d + w_d^T h(phi) + c_d, d in {0: i2t, 1: t2i}. Shared trunk h (GlobalMLP, exactly the
    GHead trunk); d = 0 uses the inherited GHead out/affine, d = 1 adds one 128->1 output (out_t2i) and one positive affine
    (affine_t2i) = 131 extra parameters per peer. GHead is built first (same RNG consumption as structure 'G', so trunk, out and
    affine are bitwise those of the same-seed shared head); the extra output is created under a forked CPU RNG (no RNG
    consumed) and zeroed, affine_t2i = PositiveAffine(a0, b0) (no RNG). At init both directions equal the shared head."""
    directional = True

    def __init__(self, a0, b0, d=1024):
        super().__init__(a0, b0, d, local=False)
        self.kind = 'GD'
        with torch.random.fork_rng(devices=[]):
            self.out_t2i = nn.Linear(128, 1)
        nn.init.zeros_(self.out_t2i.weight); nn.init.zeros_(self.out_t2i.bias)
        self.affine_t2i = PositiveAffine(a0, b0)

    def ab_dir(self, direction):
        return (self.affine if direction == 0 else self.affine_t2i).ab()

    def rows(self, s, hid, dd, s_exact=None):
        """per-row direction dd (bool/long tensor, 0 = i2t, 1 = t2i). s_exact (float64 tensor) replaces s in the affine part
        (evaluation), as GHead in train_unit.g_pairs_lists."""
        m = dd.bool()
        r = torch.where(m, self.out_t2i(hid).squeeze(-1), self.out(hid).squeeze(-1))
        if s_exact is None:
            a0, b0 = self.affine.ab(); a1, b1 = self.affine_t2i.ab()
            return torch.where(m, a1, a0) * s + torch.where(m, b1, b0) + r
        (a0, b0), (a1, b1) = ((float(x) for x in self.affine.ab()), (float(x) for x in self.affine_t2i.ab()))
        a = torch.where(m, torch.tensor(a1, dtype=torch.float64, device=s_exact.device), torch.tensor(a0, dtype=torch.float64, device=s_exact.device))
        b = torch.where(m, torch.tensor(b1, dtype=torch.float64, device=s_exact.device), torch.tensor(b0, dtype=torch.float64, device=s_exact.device))
        return a * s_exact + b + r.double()
