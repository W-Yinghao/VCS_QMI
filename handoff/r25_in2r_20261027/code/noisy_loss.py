"""R25 noisy-prototype objectives (04_MATH_INTERFACE_CN.md §3–4).

S is the query/prototype cosine matrix of one branch (rows = queries, cols = prototypes).
P_B = uniform diagonal pairs; Q_B = product of the empirical marginals = all B^2 pairs INCLUDING the diagonal.
"""
import torch
import torch.nn.functional as F

from vse import infonce


def critic(S, link, a=None, b=None):
    if link == 'c0':
        return S
    if link == 'c1':
        return torch.tanh(a * (S - b))
    raise ValueError(link)


def vcs_J_dense(Fm):
    d = Fm.diagonal()
    return (d - 0.5 * d * d).mean() - (Fm + 0.5 * Fm * Fm).mean()


def vcs_J_moment(X, Y):
    """Exact cosine (C0) form: mean(p-p^2/2) - xbar.ybar - 0.5 <X^T X/B, Y^T Y/B>_F (no detach)."""
    B = X.shape[0]
    p = (X * Y).sum(1)
    return (p - 0.5 * p * p).mean() - X.mean(0) @ Y.mean(0) - 0.5 * ((X.t() @ X / B) * (Y.t() @ Y / B)).sum()


def balanced_bce(S, link, a=None, b=None, clip=1e-7):
    """-0.5 E_P log D - 0.5 E_Q log(1-D), D=(1+F)/2; returns (loss, clip_fraction)."""
    B = S.shape[0]
    eye = torch.eye(B, device=S.device, dtype=torch.bool)
    if link == 'c0':
        D = (1 + S) / 2
        clipped = ((D < clip) | (D > 1 - clip)).float().mean()
        D = D.clamp(clip, 1 - clip)
        lp = -torch.log(D[eye]).mean()
        lq = -torch.log(1 - D).mean()
    else:
        z = 2 * a * (S - b)  # (1+tanh(u))/2 = sigmoid(2u)
        clipped = torch.zeros((), device=S.device)
        lp = F.softplus(-z[eye]).mean()
        lq = F.softplus(z).mean()
    return 0.5 * lp + 0.5 * lq, clipped


def noisy_loss(S, obj, link='c0', a=None, b=None):
    """Unscaled branch loss and diagnostics. Native SCE is the author's infonce() unchanged."""
    if obj == 'sce':
        return infonce(S), {}
    Fm = critic(S, link, a, b)
    J = vcs_J_dense(Fm)
    if obj == 'vcs':
        oob = (S.detach().abs() > 1 + 1e-5).float().mean()
        return -J, {'raw_J': J.detach(), 'oob_frac': oob}
    if obj == 'bce':
        loss, clipped = balanced_bce(S, link, a, b)
        return loss, {'raw_J': J.detach(), 'clip_frac': clipped.detach()}
    raise ValueError(obj)
