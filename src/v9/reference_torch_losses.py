"""Shared posterior coordinates and detached risk weighting for server tests.

Not a substitute for the upstream benchmark loss/negative-sampling implementation.
"""
import torch
import torch.nn.functional as F


def balanced_fit_loss(logits_p,logits_q,kind='vcs'):
    """Common natural logit ell; T=tanh(ell/2) for BOTH rules.
    VCS = 1-J, matched BCE = balanced class log loss.
    """
    if logits_p.numel()==0 or logits_q.numel()==0:raise ValueError('empty role')
    if kind=='vcs':
        tp=torch.tanh(logits_p/2);tq=torch.tanh(logits_q/2)
        return .5*(1-tp).square().mean()+.5*(1+tq).square().mean()
    if kind=='bce':return .5*F.softplus(-logits_p).mean()+.5*F.softplus(logits_q).mean()
    raise ValueError('unknown objective')


def detached_weighted_loss(per_pair_loss,weights,global_weight_mean):
    """Fixed dataset-scale normalizer, not the current batch's weight mean."""
    if per_pair_loss.ndim!=1 or per_pair_loss.shape!=weights.shape:
        raise ValueError('expect equal one-dimensional arrays')
    if per_pair_loss.numel()==0 or not torch.isfinite(per_pair_loss).all() or not torch.isfinite(weights).all():
        raise ValueError('empty or nonfinite values')
    if torch.any(weights<0) or not 0<float(global_weight_mean)<=1:
        raise ValueError('invalid weights or normalizer')
    return (per_pair_loss*weights.detach()).mean()/float(global_weight_mean)
