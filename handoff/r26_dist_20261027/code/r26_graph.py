"""R26 candidate-distribution graph and objectives (DESIGN_CN.md §1-2; package 01_ARCHITECTURE §3, 02_MATH_AUDIT).

Graph for one direction of one update of peer `net`:
  * the OTHER peer (net2, eval mode, no grad) encodes the current noisy batch;
  * its own high-confidence FIFO is searched with same-modality cosine (same peer coordinates), top-K slots;
  * candidates are the opposite-modality cached embeddings of those slots, cloned (frozen snapshot, no grad);
  * repeated slots are merged into one atom column and their mass accumulated (within one update the queue
    snapshot is fixed, so (peer, slot) identifies the atom; write version is constant inside a snapshot);
  * A[i, j] = 1/K per selected slot, W = A / B (r_i = 1/B), c = W.sum(0), Q = r c^T (all positions, diagonal of
    W included; no uniform-queue reference).
Arms: d_vcs / d_bce / d_snce on (W, Q, F = X Y^T); m_vcs / n_vcs on point targets ybar_i = sum_j A_ij y_j
(unnormalized / renormalized) with the point-pair P = diag/B and Q = all B^2 pairs incl. diagonal.
"""
import torch
import torch.nn.functional as Fn

ARMS = ('d_vcs', 'd_bce', 'd_snce', 'm_vcs', 'n_vcs')
TAU_SNCE = 0.05  # host temperature of the native SCE (vse.infonce default)


@torch.no_grad()
def build_graph(query_peer, img_bank, txt_bank, k, branch):
    """query_peer: [B,d] other-peer embeddings of the noisy queries (same modality as the search bank side).
    branch 'i2t': search img_bank, candidates = txt_bank rows; 't2i': search txt_bank, candidates = img_bank rows.
    Returns dict(idx [B,K] slots, cols [M] unique slots, Y [M,d] frozen unit candidates, A [B,M]) or None."""
    search, support = (img_bank, txt_bank) if branch == 'i2t' else (txt_bank, img_bank)
    n = search.shape[0]
    if n < k:
        return None
    sims = query_peer.mm(search.t())
    idx = torch.topk(sims, k, dim=1).indices  # [B,K]
    cols, inv = torch.unique(idx.flatten(), return_inverse=True)
    inv = inv.view_as(idx)
    B, M = idx.shape[0], cols.numel()
    A = torch.zeros(B, M, device=query_peer.device, dtype=query_peer.dtype)
    A.scatter_add_(1, inv, torch.full_like(inv, 1.0 / k, dtype=query_peer.dtype))
    Y = support[cols].clone()
    norm_dev = float((Y.norm(dim=1) - 1).abs().max())
    Y = Fn.normalize(Y, dim=-1)
    return dict(idx=idx, cols=cols, Y=Y, A=A, norm_dev=norm_dev,
                n_atoms=M, max_col_mass=float(A.sum(0).max() / B), shared_frac=float((A > 0).sum(0).gt(1).float().mean()))


def coupling(A):
    B = A.shape[0]
    W = A / B
    c = W.sum(0)
    Q = c.unsqueeze(0).expand(B, -1) / B
    return W, c, Q


def j_graph(W, Q, Fm):
    return ((W - Q) * Fm - 0.5 * (W + Q) * Fm * Fm).sum()


def bce_graph(W, Q, Fm, clip=1e-7):
    D = ((1 + Fm) / 2).clamp(clip, 1 - clip)
    return -0.5 * (W * torch.log(D)).sum() - 0.5 * (Q * torch.log(1 - D)).sum()


def snce_graph(W, c, Fm, tau=TAU_SNCE):
    B = Fm.shape[0]
    return -(W * Fm).sum() / tau + (torch.logsumexp(Fm / tau + torch.log(c).unsqueeze(0), dim=1) / B).sum()


def j_point(S):
    d = S.diagonal()
    return (d - 0.5 * d * d).mean() - (S + 0.5 * S * S).mean()


def arm_loss(arm, X, G):
    """X: [B,d] current-peer unit queries (with grad). Returns (loss to minimize, diagnostics)."""
    A, Y = G['A'], G['Y']
    if arm in ('d_vcs', 'd_bce', 'd_snce'):
        W, c, Q = coupling(A)
        Fm = X.mm(Y.t())
        J = j_graph(W, Q, Fm)
        if arm == 'd_vcs':
            loss = -J
        elif arm == 'd_bce':
            loss = bce_graph(W, Q, Fm)
        else:
            loss = snce_graph(W, c, Fm)
        return loss, {'raw_J': J.detach()}
    if arm in ('m_vcs', 'n_vcs'):
        ybar = A.mm(Y)
        if arm == 'n_vcs':
            ybar = Fn.normalize(ybar, dim=-1)
        S = X.mm(ybar.t())
        J = j_point(S)
        return -J, {'raw_J': J.detach(), 'ybar_norm': ybar.norm(dim=1).mean().detach()}
    raise ValueError(arm)


def moment_j(X, A, Y):
    """Exact cosine moment form (checking only): sum_i r_i [x_i.(mu_i-mu_Q) - 0.5 x_i^T (M_i+M_Q) x_i]."""
    B = X.shape[0]
    mu = A.mm(Y)                       # [B,d]
    muQ = mu.mean(0)
    proj = X.mm(Y.t())                 # x_i . y_j
    xMx = (A * proj * proj).sum(1)     # x_i^T M_i x_i
    W, c, _ = coupling(A)
    xMQx = (proj * proj * c.unsqueeze(0)).sum(1)  # x_i^T M_Q x_i
    return ((X * (mu - muQ)).sum(1) - 0.5 * (xMx + xMQx)).sum() / B
