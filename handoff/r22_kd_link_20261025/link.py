"""Bounded critic link for the CE decision-coordinate KD aux: T = tanh(a*s + b), s = dot(centered_L2 student, centered_L2 teacher).

vcs_tanh : a=1, b=0 fixed (buffers)      -> separates the effect of the tanh map
vcs_link : a, b learnable, init 1, 0     -> same as vcs_tanh at initialisation
bce_link : same learnable map, balanced BCE with P(positive)=(1+T)/2=sigmoid(2f)
Q is the ordered off-diagonal within-batch set (as in the original 'vcs'/'bce'), now from an explicit n x n score matrix because the
nonlinear link removes the first/second-moment shortcut. P = diagonal pairs. Theory (J, P, Q) unchanged; only the critic class changes.
With f = atanh(s) the bce_link loss equals the original 'bce' (checked in run.py check()).
"""
import torch
import torch.nn.functional as F
from kd import decision as D
from kd import losses as L

METHODS = ('vcs_tanh', 'vcs_link', 'bce_link')
ORIGINAL_DECISION = D.decision_loss
ORIGINAL_INIT = L.DistillationLoss.__init__
CURRENT = {}


class Link(torch.nn.Module):
    def __init__(self, learnable):
        super().__init__()
        if learnable:
            self.a = torch.nn.Parameter(torch.tensor(1.0)); self.b = torch.nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer('a', torch.tensor(1.0)); self.register_buffer('b', torch.tensor(0.0))

    def forward(self, s):
        return self.a * s + self.b


def link_terms(u, v, link, method):
    n = len(u)
    f = link(u @ v.T)
    mask = ~torch.eye(n, dtype=torch.bool, device=u.device)
    fp, fq = f.diagonal(), f[mask]
    if method == 'bce_link':
        loss = F.softplus(-2 * fp).mean() + F.softplus(2 * fq).mean()
    else:
        tp, tq = torch.tanh(fp), torch.tanh(fq)
        j = (tp - .5 * tp.square()).mean() - (tq + .5 * tq.square()).mean()
        loss = -j
    return loss, f


def decision_loss(student, teacher, method):
    if method not in METHODS:
        return ORIGINAL_DECISION(student, teacher, method)
    link = CURRENT['link']
    u, v = D.coordinates(student), D.coordinates(teacher.detach())
    loss, f = link_terms(u, v, link, method)
    with torch.no_grad():
        n = len(u); mask = ~torch.eye(n, dtype=torch.bool, device=u.device)
        t = torch.tanh(f); s = u @ v.T
        diagnostics = dict(method=method, a=float(link.a), b=float(link.b), positive_cosine=float(s.diagonal().mean()),
                           product_cosine=float(s[mask].mean()), positive_T=float(t.diagonal().mean()), product_T=float(t[mask].mean()),
                           positive_abs_T_ge099=float((t.diagonal().abs() >= .99).float().mean()), raw_auxiliary=float(loss),
                           raw_j_identity=float(D.quadratic_u(u, v)), q='off-diagonal ordered cross-instance, explicit n x n')
    return loss, diagnostics


def patched_init(self, student_dim, teacher_dim, config):
    ORIGINAL_INIT(self, student_dim, teacher_dim, config)
    method = config.get('decision', {}).get('method', 'none')
    if method in METHODS:
        self.decision_link = Link(learnable=method != 'vcs_tanh')
        CURRENT['link'] = self.decision_link


def install():
    D.decision_loss = decision_loss
    L.DistillationLoss.__init__ = patched_init
