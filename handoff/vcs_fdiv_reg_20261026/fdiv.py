"""Histogram dependence measures on the host Parzen joint histogram (REG-P1 soft_mi construction, r6_reg_p2.soft_mi_batch).

All measures see the same joint histogram p (K x B x B) of the reference intensities u [N] against the warped moving intensities
v [K, N]; q = p_a (x) p_b is the product of its marginals.  Only the divergence D(p || q) changes:
  mi        KL(p || q) = H(a) + H(b) - H(a, b)                           (host MI, identical operations)
  nmi       (H(a) + H(b)) / H(a, b)                                      (host NMI, identical operations)
  vcs       sup_T J = 1/2 sum (p - q)^2 / (p + q)                        (VCS with the per-bin critic; T* = (p - q)/(p + q) in [-1, 1])
  js        JS(p, q)                                                     (optimum of the balanced BCE critic on the same bins)
  chi2      sum (p - q)^2 / q                                            (Pearson chi^2, unbounded quadratic)
  cs_qmi    log((sum p^2)(sum q^2) / (sum p q)^2)                        (classic Cauchy-Schwarz quadratic MI)
  hellinger sum (sqrt p - sqrt q)^2
"""
DIVERGENCES = ("mi", "nmi", "vcs", "js", "chi2", "cs_qmi", "hellinger")
DEFAULT = (64, 1.0)
CONFIGS = ((32, 1.0), (32, 2.0), (64, 1.0), (64, 2.0))
EPS = 1e-30


def joint_hist(torch, u, v, ranges, bins=64, sigma=1.0):
    """Exactly the operations of r6_reg_p2.soft_mi_batch up to the joint histogram (bins / sigma as parameters)."""
    (ul, uh), (vl, vh) = ranges; c = torch.arange(bins, device=u.device, dtype=u.dtype)
    cu = (u - ul) / (uh - ul) * (bins - 1); cv = (v - vl) / (vh - vl) * (bins - 1)
    wu = torch.exp(-0.5 * ((cu[:, None] - c) / sigma) ** 2); wu = wu / wu.sum(-1, keepdim=True).clamp_min(1e-12)
    wv = torch.exp(-0.5 * ((cv[..., None] - c) / sigma) ** 2); wv = wv / wv.sum(-1, keepdim=True).clamp_min(1e-12)
    return torch.einsum("nb,knc->kbc", wu, wv) / u.shape[0]


def divergence(torch, Pj, name):
    """D(p || p_a p_b) for a batch of joint histograms Pj [K, B, B] -> [K] (larger = more dependent)."""
    if name in ("mi", "nmi"):
        ent = lambda p: -(p * torch.log(p.clamp_min(1e-12))).sum(-1)
        ha, hb, hab = ent(Pj.sum(2)), ent(Pj.sum(1)), ent(Pj.reshape(Pj.shape[0], -1))
        return (ha + hb) / hab if name == "nmi" else ha + hb - hab
    p = Pj; q = Pj.sum(2)[:, :, None] * Pj.sum(1)[:, None, :]
    if name == "vcs":
        return 0.5 * ((p - q) ** 2 / (p + q).clamp_min(EPS)).sum((1, 2))
    if name == "js":
        m = 0.5 * (p + q); lm = torch.log(m.clamp_min(EPS))
        return 0.5 * (p * (torch.log(p.clamp_min(EPS)) - lm)).sum((1, 2)) + 0.5 * (q * (torch.log(q.clamp_min(EPS)) - lm)).sum((1, 2))
    if name == "chi2":
        return ((p - q) ** 2 / q.clamp_min(EPS)).sum((1, 2))
    if name == "cs_qmi":
        return torch.log((p * p).sum((1, 2)) * (q * q).sum((1, 2)) / ((p * q).sum((1, 2)) ** 2).clamp_min(EPS))
    if name == "hellinger":
        return ((torch.sqrt(p + 1e-20) - torch.sqrt(q + 1e-20)) ** 2).sum((1, 2))
    raise ValueError(name)


def similarity(torch, u, v, ranges, name, bins=64, sigma=1.0):
    return divergence(torch, joint_hist(torch, u, v, ranges, bins, sigma), name)


def make_energy(Q, torch, F, base, name, bins, sigma):
    """A host RigidEnergy (kind 'mi', same positions x / reference samples u / moving image / ranges) whose batch() uses
    D(p || q) on the (bins, sigma) Parzen joint histogram.  grid_values / __call__ / value are the host's, unchanged."""

    class FDivEnergy(Q.RigidEnergy):
        def __init__(self):
            self.__dict__.update(base.__dict__)
            assert self.kind == "mi", "base must be the host MI energy"
            self.div, self.bins, self.sigma = name, bins, sigma

        def batch(self, th, t):
            v = Q.sample_pts(torch, F, self.mov, Q.apply_T(torch, th, t, self.x))[..., 0]
            return similarity(torch, self.u, v, self.kw["ranges"], self.div, self.bins, self.sigma)

    return FDivEnergy()
