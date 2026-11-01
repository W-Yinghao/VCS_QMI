"""v3 additions (supplementary plan §四-§六): hardest-competitor identity audit, fixed reference list objects (raw J, oracle
S, S-J, independent squared gap), wrong-top1 rates. Truth only on the evaluator side. margin_core / margin_core_v2 unchanged."""
import math
import numpy as np
import torch
import torch.nn.functional as F
import margin_core as MC
import margin_core_v2 as M2

TAU = 0.05
AUDIT_GROUPS = ('all', 'correct_easy', 'correct_hard', 'mismatch', 'mismatch&true_unavailable', 'mismatch&true_available',
                'mismatch&true_available&competitor_is_true', 'mismatch&true_available&competitor_not_true')


def audit_rows(S, obs, true):
    """Per-direction per-query quantities at one score matrix: observed margin, observed hardest index k, true partner mask,
    best-true-partner margin over true negatives, true rank (# true negatives above the best true partner), wrong top-1."""
    pos = obs.view(-1, 1).eq(obs.view(1, -1))
    tp = M2.true_partner_masks(obs, true)
    out = {}
    for d, M in (('i2t', S), ('t2i', S.t())):
        P = tp[d]
        nmax, k = M.masked_fill(pos, float('-inf')).max(1)
        pmax, _ = M.masked_fill(~P, float('-inf')).max(1)
        tnmax, _ = M.masked_fill(P, float('-inf')).max(1)
        avail = P.any(1)
        trank = ((M > pmax.view(-1, 1)) & ~P).sum(1)
        top = M.argmax(1)
        ar = torch.arange(M.shape[0], device=M.device)
        out[d] = dict(M=M, m_obs=M.diag() - nmax, k=k, avail=avail, P=P,
                      m_true=torch.where(avail, pmax - tnmax, torch.zeros_like(pmax)), rank_true=trank,
                      wrong_top_obs=~pos[ar, top], wrong_top_true=~P[ar, top])
    return out


def audit_groups(ref_cls_d, a0_d):
    """Boolean group masks (numpy) for one direction; reference classes from the frozen H parent, competitor identity from
    the measured state before the update (a0)."""
    g = {k: ref_cls_d[k] for k in ('all',)}
    g['correct_easy'] = ref_cls_d['correct_ref_first']
    g['correct_hard'] = ref_cls_d['correct_ref_not_first']
    mis = ref_cls_d['mismatch']
    av = a0_d['avail'].cpu().numpy()
    ar = torch.arange(a0_d['k'].shape[0], device=a0_d['k'].device)
    sel_true = a0_d['P'][ar, a0_d['k']].cpu().numpy()
    g['mismatch'] = mis
    g['mismatch&true_unavailable'] = mis & ~av
    g['mismatch&true_available'] = mis & av
    g['mismatch&true_available&competitor_is_true'] = mis & av & sel_true
    g['mismatch&true_available&competitor_not_true'] = mis & av & ~sel_true
    return g, sel_true


class AuditAcc:
    F = ('n', 'n_true_available', 'n_competitor_is_true', 'sum_dobs', 'n_obs_down', 'sum_dtrue', 'n_true_down', 'sum_drank',
         'n_rank_worse', 'n_comp_changed', 'sum_doldcomp', 'n_wt_obs0', 'n_wt_obs1', 'n_wt_true0', 'n_wt_true1')

    def __init__(self):
        self.d = {}

    def add(self, key, a0, a1, groups, sel_true):
        cv = lambda t: t.detach().double().cpu().numpy() if t.dtype != torch.bool else t.cpu().numpy()
        ar = torch.arange(a0['k'].shape[0], device=a0['k'].device)
        dobs = cv(a1['m_obs'] - a0['m_obs']); dtrue = cv(a1['m_true'] - a0['m_true'])
        drank = cv(a1['rank_true'].double() - a0['rank_true'].double())
        changed = cv(a1['k'] != a0['k'])
        doldc = cv((a1['M'].diag() - a1['M'][ar, a0['k']]) - (a0['M'].diag() - a0['M'][ar, a0['k']]))
        av = cv(a0['avail'])
        w0o, w1o, w0t, w1t = cv(a0['wrong_top_obs']), cv(a1['wrong_top_obs']), cv(a0['wrong_top_true']), cv(a1['wrong_top_true'])
        for gname, m in groups.items():
            a = self.d.setdefault(key + (gname,), dict.fromkeys(self.F, 0.0))
            ma = m & av
            a['n'] += m.sum(); a['n_true_available'] += ma.sum(); a['n_competitor_is_true'] += (m & sel_true).sum()
            a['sum_dobs'] += dobs[m].sum(); a['n_obs_down'] += ((dobs < 0) & m).sum()
            a['sum_dtrue'] += dtrue[ma].sum(); a['n_true_down'] += ((dtrue < 0) & ma).sum()
            a['sum_drank'] += drank[ma].sum(); a['n_rank_worse'] += ((drank > 0) & ma).sum()
            a['n_comp_changed'] += (changed & m).sum(); a['sum_doldcomp'] += doldc[m].sum()
            a['n_wt_obs0'] += (w0o & m).sum(); a['n_wt_obs1'] += (w1o & m).sum()
            a['n_wt_true0'] += (w0t & ma).sum(); a['n_wt_true1'] += (w1t & ma).sum()

    @staticmethod
    def finish(a):
        n, na = float(a['n']), float(a['n_true_available'])
        r = lambda x, d: float(x) / d if d else None
        return {'group_count': int(n), 'true_available_count': int(na), 'unavailable_count': int(n - na),
                'competitor_is_true_match_count': int(a['n_competitor_is_true']),
                'mean_d_observed_margin': r(a['sum_dobs'], n), 'frac_observed_margin_down': r(a['n_obs_down'], n),
                'mean_d_true_margin': r(a['sum_dtrue'], na), 'frac_true_margin_down': r(a['n_true_down'], na),
                'mean_d_true_rank': r(a['sum_drank'], na), 'frac_true_rank_worse': r(a['n_rank_worse'], na),
                'frac_top_competitor_changed': r(a['n_comp_changed'], n), 'mean_d_old_competitor_margin': r(a['sum_doldcomp'], n),
                'wrong_top1_observed_before': r(a['n_wt_obs0'], n), 'wrong_top1_observed_after': r(a['n_wt_obs1'], n),
                'wrong_top1_true_before': r(a['n_wt_true0'], na), 'wrong_top1_true_after': r(a['n_wt_true1'], na)}

    def result(self):
        return {k: self.finish(v) for k, v in self.d.items()}


# ---------------------------------------------------------------------------------------------------- fixed reference list

def list_object(M, pos, tau=TAU):
    """Rows = queries, columns = the K fixed candidates; pos = observed positives. float64. Returns J, S, gap, m."""
    M = M.double(); K = M.shape[1]
    m = pos.sum(1).double()
    a = pos.double() / m.view(-1, 1)
    q = 1.0 / K
    T = torch.tanh(0.5 * (math.log(K) + F.log_softmax(M / tau, dim=1)))
    J = (a * (T - 0.5 * T * T)).sum(1) - (T + 0.5 * T * T).mean(1)
    S = (K - m) / (K + m)
    eta = (a - q) / (a + q)
    gap = ((a + q) / 2 * (T - eta) ** 2).sum(1)
    return J, S, gap, m


def pair_object(s_pos, s_comp, tau=TAU):
    """Binary object (labelled positive, competitor): K = 2, m = 1, S = 1/3."""
    z = torch.stack(((s_pos.double() - s_comp.double()) / tau, torch.zeros_like(s_pos, dtype=torch.float64)), 1)
    pos = torch.zeros_like(z, dtype=torch.bool); pos[:, 0] = True
    T = torch.tanh(0.5 * (math.log(2) + F.log_softmax(z, dim=1)))
    a = pos.double()
    J = (a * (T - 0.5 * T * T)).sum(1) - (T + 0.5 * T * T).mean(1)
    eta = (a - 0.5) / (a + 0.5)
    gap = ((a + 0.5) / 2 * (T - eta) ** 2).sum(1)
    return J, torch.full_like(J, 1.0 / 3.0), gap


def fixed_list_rows(Sbar, obs, k_fixed):
    """Per direction: list J/S/gap/m, pair J/S/gap with the FIXED competitor, fixed margin, rank of the labelled positive among
    observed non-positives, plus dynamic-hardest readouts (separate)."""
    pos = obs.view(-1, 1).eq(obs.view(1, -1))
    out = {}
    for d, M in (('i2t', Sbar), ('t2i', Sbar.t())):
        ar = torch.arange(M.shape[0], device=M.device)
        kf = k_fixed[d]
        Jl, Sl, gl, m = list_object(M, pos)
        sp, sc = M.diag(), M[ar, kf]
        Jp, Sp, gp = pair_object(sp, sc)
        nmax, kd = M.masked_fill(pos, float('-inf')).max(1)
        Jd, _, gd = pair_object(sp, nmax)
        rank = ((M > sp.view(-1, 1)) & ~pos).sum(1)
        out[d] = dict(J_list=Jl, S_list=Sl, gap_list=gl, resid_list=(Sl - Jl - gl).abs(), m=m,
                      J_pair=Jp, S_pair=Sp, gap_pair=gp, resid_pair=(Sp - Jp - gp).abs(),
                      margin_fixed=(sp - sc).double(), rank_pos=rank.double(), pos_first=(rank == 0),
                      margin_dyn=(sp - nmax).double(), J_pair_dyn=Jd, gap_pair_dyn=gd, dyn_equals_fixed=(kd == kf))
    return out
