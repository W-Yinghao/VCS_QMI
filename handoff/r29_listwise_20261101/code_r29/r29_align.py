"""R29 main-alignment objectives (NEXT_EXPERIMENTS_CN.md of VCS_Retrieval_Global_Review_20261101).

The only term that differs between arms is the image-text alignment term of the clean main learning:
  warmup  : the cross-modal term of VSE.train(mode='warmup')      (native: infonce(sims) = SCE, tau 0.05, batch mean)
  co-train: loss_1 of the reliable (labeled / clean-split) branch   (native: ContrastiveLoss hinge, margin 0.2, max-violation, batch sum)
Intra-modal terms (loss2/loss3), the noisy branch (R27 A1 interface), GMM scoring (native eval_loss hinge), validation and the
test-time cosine score are the same for every arm.

Candidate list of a query = the in-batch candidates of the original batch (i2t: the B captions; t2i: the B image rows), K = B.
Target a = training-visible multi-positive identity: candidate j is a positive of query i iff its OBSERVED image id equals the
query's observed image id (dataset.t2i_index after the noise file; injected-noise truth is never read); a is uniform over the
positives (contains the diagonal). Q = P_Z x Unif(K): the uniform candidate-slot permutation is applied analytically (mean
over the K slots), so no K-fold training.

All four non-native arms share the temperature TAU = 0.05 of the backbone's own InfoNCE (vse.infonce default):
  N  n_infonce        row loss  -sum_j a_j log pi_j,                        pi = softmax(s / TAU)
  P  p_pointwise_vcs  row loss  -J(T, a),  T_j = tanh(s_j / TAU + b)        (reads only its own score; b one learnable scalar per peer)
  L  l_listwise_vcs   row loss  -J(T, a),  T_j = (K pi_j - 1)/(K pi_j + 1) = tanh(u_j / 2),  u_j = log(K pi_j)
  B  b_listwise_bce   row loss  -sum_j a_j log D_j - (1/K) sum_j log(1 - D_j),  D = (1+T)/2 = sigmoid(u)
  J(T, a) = sum_j a_j (T_j - T_j^2/2) - (1/K) sum_j (T_j + T_j^2/2)        (original J; listwise_vcs_reference.py)
Reduction rule (one rule, all arms, no per-arm tuning): the replacement uses the native term's own reduction in that phase,
  warmup  : 0.5 * (mean_rows i2t + mean_rows t2i)      (= native SCE's 0.5 per direction, batch mean)
  co-train: sum_rows i2t + sum_rows t2i                (= native hinge's sum over rows of both directions)
"""
import math
import torch
import torch.nn.functional as F
from vse import infonce

TAU = 0.05
P_B0 = 0.0          # initial b of the pointwise critic (T = tanh(s/TAU + b); b = 0 -> T(0) = 0, unsaturated at the untrained init)
ARMS = ('h_native', 'n_infonce', 'p_pointwise_vcs', 'l_listwise_vcs', 'b_listwise_bce')
KIND = {'n_infonce': 'n', 'p_pointwise_vcs': 'p', 'l_listwise_vcs': 'l', 'b_listwise_bce': 'b'}
PHASE_COEF = {'warmup': 0.5, 'cotrain': 1.0}
PHASE_REDUCE = {'warmup': 'mean', 'cotrain': 'sum'}


def target_from_obs(obs, dtype=torch.float32):
    """Row-stochastic multi-positive target from observed image ids [B] (symmetric support, diagonal included)."""
    M = (obs.view(-1, 1) == obs.view(1, -1)).to(dtype)
    return M / M.sum(1, keepdim=True)


def vcs_J_rows(T, A):
    return (A * (T - 0.5 * T * T)).sum(1) - (T + 0.5 * T * T).mean(1)


def log_kpi(S, tau=TAU):
    """u = log(K pi), pi = softmax(S / tau) along the candidate axis (dim 1)."""
    return math.log(S.shape[1]) + F.log_softmax(S / tau, dim=1)


def row_losses(kind, S, A, b=None, tau=TAU):
    """Per-query losses [n_rows]; queries = rows of S, candidates = columns, A row-stochastic target."""
    if kind == 'n':
        return -(A * F.log_softmax(S / tau, dim=1)).sum(1)
    if kind == 'l':
        return -vcs_J_rows(torch.tanh(0.5 * log_kpi(S, tau)), A)
    if kind == 'b':
        u = log_kpi(S, tau)
        sp = F.softplus(u)
        return -(A * (u - sp)).sum(1) + sp.mean(1)
    if kind == 'p':
        return -vcs_J_rows(torch.tanh(S / tau + b), A)
    raise ValueError(kind)


def critic_T(kind, S, b=None, tau=TAU):
    if kind == 'p':
        return torch.tanh(S / tau + b)
    return torch.tanh(0.5 * log_kpi(S, tau))


def two_way(kind, sims, A, b=None, tau=TAU):
    """(i2t rows [B], t2i rows [B]) for the image x caption score matrix sims."""
    return row_losses(kind, sims, A, b, tau), row_losses(kind, sims.t(), A.t(), b, tau)


def align(opt, net, phase, sims, obs):
    """The alignment term for arm opt.r29_arm in phase 'warmup' / 'cotrain'. H returns the native expression unchanged."""
    arm = opt.r29_arm
    if arm == 'h_native':
        if phase == 'warmup':
            return infonce(sims)
        return net.criterion(sims, labels=None)
    kind = KIND[arm]
    A = target_from_obs(obs, sims.dtype)
    r1, r2 = two_way(kind, sims, A, getattr(net, 'r29_b', None))
    if PHASE_REDUCE[phase] == 'mean':
        return PHASE_COEF[phase] * (r1.mean() + r2.mean())
    return PHASE_COEF[phase] * (r1.sum() + r2.sum())


@torch.no_grad()
def readout(sims, obs, b=None):
    """Report-only batch diagnostics common to all arms (detached): listwise J (L critic), positive-top1 rate, positive mass."""
    A = target_from_obs(obs, sims.dtype)
    out = {}
    for name, S, At in (('i2t', sims, A), ('t2i', sims.t(), A.t())):
        T = critic_T('l', S)
        out[f'Jlist_{name}'] = float(vcs_J_rows(T, At).mean())
        out[f'top1pos_{name}'] = float((At.gather(1, S.argmax(1, keepdim=True)) > 0).float().mean())
        out[f'pimass_{name}'] = float((F.softmax(S / TAU, dim=1) * (At > 0)).sum(1).mean())
        if b is not None:
            out[f'Jpoint_{name}'] = float(vcs_J_rows(critic_T('p', S, b), At).mean())
    out['multi_pos_rows'] = float(((A > 0).sum(1) > 1).float().mean())
    return out
