"""R27-Q (NEXT_DESIGN_CN.md, review f3640384, 2026-10-31): query-conditioned candidate pooling on the C-native host.

Only the t2i repair module changes; i2t keeps the full native A1 path. Candidates: the C-native fixed 3A+2B rule, K=5
unique image IDs (r27cs_sources.search_t2i mode 'C_mix', unchanged). The current peer re-encodes the candidates exactly as
in C-native; the native GraphRefiner self-attention block is kept and ONLY its final uniform mean over neighbours is
replaced by p_i = normalize(sum_j alpha_ij h_ij) (co_train.GraphRefiner.forward(neighbors, alpha)). alpha is
stop-gradient and is produced from the OTHER peer's (teacher's) scalar query-candidate scores; the two peers' raw vectors
are never dot-producted with each other. One refiner forward per direction; the training loss stays the single native SCE.

Arms (alpha = softmax(beta * score) over the K candidates of each row; beta fixed in r27q_config, one value per score family):
  q1_cos_pool  : score = teacher cosine q.y (direct cross-modal; no head)
  q2_vcs_pool  : score = g of a relation head trained with the original VCS objective
                 J(T) = E_P[T - T^2/2] - E_Q[T + T^2/2], T = tanh(g)
  q3_bce_pool  : same head/inputs/params/P/budget, balanced BCE with logit 2g (same nonparametric optimum g = log(p/q)/2)
  q4_snce_pool : same, soft-NCE over the in-batch images with logits 2g (t2i competition; soft targets = observed-image-ID
                 duplicates within the batch). The logit scale is not capped.

Teacher coordinate convention (identical when fitting and applying the head; also used by Q1):
  teacher = the other peer net2 in eval mode, no grad; raw unaugmented inputs (captions: Ctx.tokens, drop=False; images:
  all 36 regions); each modality centered by the mean of a batch of <= 128 same-modality training samples encoded in the
  same call (fit: each 128-pair encoding chunk of P; apply: query captions by the noisy batch's own unaugmented captions,
  candidates by the noisy batch's observed images), then l2-normalized (host l2norm).
P / Q (per peer stage): P = the stage's labeled split (the teacher's GMM clean split, as handed to the stage) rows whose
  teacher clean probability > the teacher's avg_clean_prob (the host's high-confidence rule), observed caption ->
  OBSERVED image (training-visible), uniformly weighted within the stage; random subset of <= N_FIT (+ <= N_VAL held out
  for diagnostics). Q = the empirical product of P's two marginals INCLUDING its diagonal (caption and image drawn
  independently and uniformly from the fitted P rows).
Stage schedule: at the start of every peer stage (R.main: model A trains with teacher B, then model B with teacher A) the
  head is fitted from scratch on the frozen teacher with a stage seed (SEED_BASE*seed + 2*epoch + peer, identical for
  all arms) inside torch.random.fork_rng (the global CPU/CUDA RNG streams are restored; numpy/python RNG are not used).
  The head is then frozen for the whole stage and discarded; the next stage re-estimates it. No head state is carried
  across stages, so a resume at an epoch boundary refits the same stage deterministically.
Training truth (the pre-injection pairing, the caption-index -> image-index arithmetic, the dataset's clean-label
arrays) is never read here; it is used only by the separate offline evaluator tools/fit_eval_offline.py.
"""
from __future__ import annotations
import hashlib, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from r27_interface import Ctx
from vse import l2norm
import r27q_config as C

Q_ARMS = {'q1_cos_pool': 'cos', 'q2_vcs_pool': 'vcs', 'q3_bce_pool': 'bce', 'q4_snce_pool': 'snce'}
HEAD_KINDS = ('vcs', 'bce', 'snce')
SOURCE = 'C_mix'            # C-native candidate rule (r27cs_sources)
K = 5
GATE_UNIFORM = None         # gate only: None (normal) | 'mean' (alpha computed, then the native mean path) | 'weighted'
STAGE = {}                  # the current stage's frozen scorer
DIM = 1024


# ----------------------------------------------------------------------------- relation head

class PairHead(nn.Module):
    """Small pair head on teacher features: g(q, y) = l2(relu(l1([q*y, |q-y|, q.y]))). Pre-link score g."""

    def __init__(self, dim=DIM, hidden=C.HIDDEN):
        super().__init__()
        self.l1 = nn.Linear(2 * dim + 1, hidden)
        self.l2 = nn.Linear(hidden, 1)

    @staticmethod
    def features(q, y):
        return torch.cat([q * y, (q - y).abs(), (q * y).sum(-1, keepdim=True)], -1)

    def forward(self, q, y):
        return self.l2(F.relu(self.l1(self.features(q, y)))).squeeze(-1)

    def grid(self, q, y):
        """q [a,d] (captions / queries), y [b,d] (images) -> g [a,b]."""
        a, b = q.shape[0], y.shape[0]
        return self.forward(q[:, None, :].expand(a, b, q.shape[1]), y[None, :, :].expand(a, b, y.shape[1]))


def vcs_J(g_pos, g_q):
    """Original VCS objective (maximized): J(T) = E_P[T - T^2/2] - E_Q[T + T^2/2], T = tanh(g)."""
    tp, tq = torch.tanh(g_pos), torch.tanh(g_q)
    return (tp - 0.5 * tp * tp).mean() - (tq + 0.5 * tq * tq).mean()


def bce_loss(g_pos, g_q):
    """Balanced BCE with logit 2g: -E_P log s(2g) - E_Q log(1 - s(2g))."""
    return F.softplus(-2.0 * g_pos).mean() + F.softplus(2.0 * g_q).mean()


def snce_loss(G, soft):
    """Soft-NCE, t2i competition: rows = captions, cols = in-batch images, logits 2g; soft targets (rows sum 1)."""
    return -(soft * torch.log_softmax(2.0 * G, dim=1)).sum(1).mean()


def soft_targets(obs):
    s = (obs[None, :] == obs[:, None]).to(torch.float32)
    return s / s.sum(1, keepdim=True)


def head_loss(kind, head, cap, img, obs, ip, ia, ib):
    """Training loss of one step. P = (cap[ip], img[ip]); Q grid = cap[ib] x img[ia] (independent marginal draws)."""
    if kind == 'vcs':
        return -vcs_J(head(cap[ip], img[ip]), head.grid(cap[ib], img[ia]))
    if kind == 'bce':
        return bce_loss(head(cap[ip], img[ip]), head.grid(cap[ib], img[ia]))
    if kind == 'snce':
        return snce_loss(head.grid(cap[ip], img[ip]), soft_targets(obs[ip]))
    raise ValueError(kind)


# ----------------------------------------------------------------------------- teacher coordinates

@torch.no_grad()
def t_txt_raw(net2, gidx):
    toks = [Ctx.tokens(int(c)) for c in (gidx.tolist() if torch.is_tensor(gidx) else list(gidx))]
    lengths = [len(t) for t in toks]
    cap = torch.zeros(len(toks), max(lengths), dtype=torch.long)
    for j, t in enumerate(toks):
        cap[j, :len(t)] = t
    return net2.txt_enc(cap.cuda(), lengths)


@torch.no_grad()
def t_img_raw(net2, img_idx):
    idx = torch.as_tensor(img_idx, dtype=torch.long, device='cuda')
    ims = Ctx.gpu_images()[idx]
    lens = torch.full((ims.shape[0],), float(ims.shape[1]), device=ims.device)
    return net2.img_enc(ims, lens)


def center(raw):
    return l2norm(raw - raw.mean(dim=0, keepdim=True))


@torch.no_grad()
def encode_pairs(net2, gidx, obs):
    """Teacher features of observed pairs, centered per CHUNK-pair encoding batch (len must be a multiple of CHUNK)."""
    assert len(gidx) % C.CHUNK == 0 and len(gidx) == len(obs)
    caps, ims = [], []
    for s in range(0, len(gidx), C.CHUNK):
        caps.append(center(t_txt_raw(net2, gidx[s:s + C.CHUNK])))
        ims.append(center(t_img_raw(net2, obs[s:s + C.CHUNK])))
    return torch.cat(caps), torch.cat(ims)


def teacher_mode_ok(net2):
    return (not net2.training) and (not net2.img_enc.training) and (not net2.txt_enc.training)


def fingerprint(net2):
    h = hashlib.sha256()
    for m in (net2.img_enc, net2.txt_enc):
        for p in m.state_dict().values():
            h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def head_sha(head):
    h = hashlib.sha256()
    for p in head.state_dict().values():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------- P / Q

def select_P(ld, thr):
    """High-confidence observed pairs of the stage's labeled split (teacher prob > teacher avg_clean_prob).
    Returns caption gidx, OBSERVED image idx (both training-visible) and the labeled-row index."""
    prob = np.asarray(ld.probability, dtype=np.float64).reshape(-1)
    rows = np.nonzero(prob > float(thr))[0]
    gidx = np.asarray(ld.r27_gidx, dtype=np.int64)[rows]
    obs = np.asarray(ld.t2i_index, dtype=np.int64)[rows]
    return gidx, obs, rows


def split_sizes(n):
    n_val = min(C.N_VAL, (n // 10) // C.CHUNK * C.CHUNK)
    n_fit = min(C.N_FIT, (n - n_val) // C.CHUNK * C.CHUNK)
    return n_fit, n_val


def stage_seed(seed, epoch, peer):
    return int(C.SEED_BASE * int(seed) + 2 * int(epoch) + int(peer))


# ----------------------------------------------------------------------------- held-out diagnostics

@torch.no_grad()
def heldout_metrics(head, cap, img, obs, gen_seed):
    """Held-out objectives of one head on the held-out P: VCS J, BCE, soft-NCE (CHUNK blocks), and the observed-image
    in-block top1 (training-visible). Q = 512x512 independent marginal draws (fixed generator)."""
    n = cap.shape[0]
    g = torch.Generator().manual_seed(gen_seed)
    ia = torch.randint(n, (512,), generator=g).to(cap.device); ib = torch.randint(n, (512,), generator=g).to(cap.device)
    gp = torch.cat([head(cap[s:s + 1024], img[s:s + 1024]) for s in range(0, n, 1024)])
    gq = torch.cat([head.grid(cap[ib[s:s + 64]], img[ia]) for s in range(0, 512, 64)])
    nce, top1 = [], []
    for s in range(0, n, C.CHUNK):
        G = head.grid(cap[s:s + C.CHUNK], img[s:s + C.CHUNK])
        o = obs[s:s + C.CHUNK]
        nce.append(float(snce_loss(G, soft_targets(o))))
        top1.append(float((o[G.argmax(1)] == o).float().mean()))
    return dict(J=float(vcs_J(gp, gq)), bce=float(bce_loss(gp, gq)), snce=float(np.mean(nce)), top1_obs=float(np.mean(top1)),
                g_pos_mean=float(gp.mean()), g_q_mean=float(gq.mean()), T_pos_mean=float(torch.tanh(gp).mean()),
                T_q_mean=float(torch.tanh(gq).mean()))


def own_objective(kind, m):
    return {'vcs': -m['J'], 'bce': m['bce'], 'snce': m['snce']}[kind]


# ----------------------------------------------------------------------------- fit

def fit_head(kind, cap, img, obs, seed, steps, val=None, eval_at=()):
    """Fit one head from scratch (call inside a forked RNG). Same init / index sequence for every kind given the seed."""
    torch.manual_seed(seed)
    head = PairHead(dim=cap.shape[1]).to(cap.device)
    opt_h = torch.optim.Adam(head.parameters(), lr=C.LR)
    n = cap.shape[0]
    curve = {}
    if val is not None and 0 in eval_at:
        curve[0] = heldout_metrics(head, *val, gen_seed=seed + 1)
    first = last = None
    for s in range(1, steps + 1):
        ip, ia, ib = (torch.randint(n, (C.BATCH,)).to(cap.device) for _ in range(3))
        loss = head_loss(kind, head, cap, img, obs, ip, ia, ib)
        opt_h.zero_grad(set_to_none=True)
        loss.backward()
        opt_h.step()
        if first is None:
            first = float(loss.detach())
        if val is not None and s in eval_at:
            curve[s] = heldout_metrics(head, *val, gen_seed=seed + 1)
    last = float(loss.detach()) if steps else None
    opt_h.zero_grad(set_to_none=True)          # no leftover fit gradients on the frozen head
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return head, dict(train_loss_first=first, train_loss_last=last, curve=curve)


def stage_data(net2, ld, seed):
    """P selection + teacher encoding (inside the caller's forked RNG; torch CPU RNG seeded by the caller)."""
    gidx, obs, rows = select_P(ld, net2.avg_clean_prob)
    n_fit, n_val = split_sizes(len(rows))
    perm = torch.randperm(len(rows)).numpy()
    fit_i, val_i = perm[:n_fit], perm[n_fit:n_fit + n_val]
    cap, img = encode_pairs(net2, gidx[fit_i], obs[fit_i])
    vcap, vimg = encode_pairs(net2, gidx[val_i], obs[val_i]) if n_val else (None, None)
    o = torch.as_tensor(obs[fit_i], device='cuda'); vo = torch.as_tensor(obs[val_i], device='cuda')
    info = dict(nP_avail=int(len(rows)), nP_fit=int(n_fit), nP_val=int(n_val), thr=float(net2.avg_clean_prob),
                n_labeled=int(len(ld.probability)), fit_unique_images=int(np.unique(obs[fit_i]).size))
    return (cap, img, o), ((vcap, vimg, vo) if n_val else None), info, (gidx[fit_i], obs[fit_i], rows[fit_i])


def begin_stage(opt, net, net2, ld, epoch, peer, stats):
    """Fit (or, for Q1, only register) this stage's frozen scorer. Global RNG streams are untouched."""
    kind = Q_ARMS[opt.arm]
    STAGE.clear()
    t0 = time.time()
    assert teacher_mode_ok(net2), 'teacher must be in eval mode'
    seed = stage_seed(opt.seed, epoch, peer)
    fp = fingerprint(net2)
    STAGE.update(arm=opt.arm, kind=kind, epoch=int(epoch), peer=int(peer), student=id(net), teacher=id(net2), seed=seed,
                 beta=C.BETA_COS if kind == 'cos' else C.BETA_HEAD, teacher_fp=fp, head=None)
    stats['qstage_seed'] = seed
    stats['qstage_teacher_fp'] = fp[:16]
    stats['qstage_beta'] = STAGE['beta']
    stats['qstage_kind'] = kind
    stats['qstage_config_frozen'] = bool(C.FROZEN)
    if kind == 'cos':
        gidx, obs, rows = select_P(ld, net2.avg_clean_prob)
        stats['qstage_nP_avail'] = int(len(rows))
        STAGE['version'] = f'e{epoch}p{peer}-cos-beta{C.BETA_COS:g}-t{fp[:12]}'
        stats['qstage_version'] = STAGE['version']
        stats['qstage_fit_seconds'] = time.time() - t0
        return
    dev = torch.cuda.current_device()
    with torch.random.fork_rng(devices=[dev]):
        torch.manual_seed(seed)
        fitd, vald, info, _ = stage_data(net2, ld, seed)
        head, fi = fit_head(kind, *fitd, seed=seed + 17, steps=C.STEPS)
        hm = heldout_metrics(head, *vald, gen_seed=seed + 1) if vald is not None else {}
    hs = head_sha(head)
    STAGE.update(head=head, head_sha=hs, version=f'e{epoch}p{peer}-{kind}-s{seed}-h{hs[:12]}-t{fp[:12]}')
    for k, v in info.items():
        stats[f'qstage_{k}'] = v
    stats['qstage_train_loss_first'] = fi['train_loss_first']
    stats['qstage_train_loss_last'] = fi['train_loss_last']
    for k, v in hm.items():
        stats[f'qstage_val_{k}'] = v
    stats['qstage_head_params'] = int(sum(p.numel() for p in head.parameters()))
    stats['qstage_head_sha'] = hs[:16]
    stats['qstage_version'] = STAGE['version']
    torch.cuda.synchronize()
    stats['qstage_fit_seconds'] = time.time() - t0


# ----------------------------------------------------------------------------- apply (one t2i step)

@torch.no_grad()
def teacher_scores(net2, S, cap_gidx, obs_img, kind=None, head=None):
    """Teacher scores [B,K] of every (query, candidate) pair of the C-native graph, in the teacher coordinates."""
    q = center(t_txt_raw(net2, cap_gidx))                                   # query captions (unaugmented)
    mimg = t_img_raw(net2, obs_img).mean(dim=0, keepdim=True)                # noisy batch's observed images
    yc = l2norm(t_img_raw(net2, S['cols']) - mimg)                           # candidate images
    Yn = yc[S['inv']]                                                        # [B,K,d]
    qq = q[:, None, :].expand_as(Yn)
    if kind == 'cos':
        return (qq * Yn).sum(-1), q, yc
    return head(qq, Yn), q, yc


@torch.no_grad()
def alpha_t2i(opt, net, net2, S, uids, stats, timed=False):
    """alpha [B,K] for the current t2i step (stop-grad), or None when the gate routes uniform through the native mean."""
    st = STAGE
    assert st and st['arm'] == opt.arm and st['epoch'] == int(opt.r27_epoch) and st['student'] == id(net) and st['teacher'] == id(net2), 'stale R27-Q scorer'
    assert S['inv'].shape[1] == K and teacher_mode_ok(net2)
    t0 = time.time()
    s, _, _ = teacher_scores(net2, S, uids[0], uids[1], st['kind'], st['head'])
    alpha = torch.softmax(st['beta'] * s.float(), dim=1)
    ent = -(alpha * alpha.clamp_min(1e-30).log()).sum(1)
    b = 't2i'
    stats[f'{b}_q_alpha_entropy'] += float(ent.mean())
    stats[f'{b}_q_alpha_max'] += float(alpha.max(1).values.mean())
    stats[f'{b}_q_alpha_mass_slot0'] += float(alpha[:, 0].mean())
    if 'route_b' in S:
        stats[f'{b}_q_alpha_mass_routeB'] += float((alpha * S['route_b'].float()).sum(1).mean())
    stats[f'{b}_q_score_mean'] += float(s.mean())
    stats[f'{b}_q_score_rowstd'] += float(s.std(1).mean())
    stats[f'{b}_q_frac_rows_alpha_over_half'] += float((alpha.max(1).values > 0.5).float().mean())
    stats['qmax_alpha_t2i'] = max(stats.get('qmax_alpha_t2i', 0.0), float(alpha.max()))
    stats['qmin_entropy_t2i'] = min(stats.get('qmin_entropy_t2i', 9.0), float(ent.min()))
    stats['cnt_t2i_q_rows'] += int(alpha.shape[0])
    if timed:
        torch.cuda.synchronize()
        stats['time_qscore_t2i'] += time.time() - t0
    if GATE_UNIFORM == 'mean':
        return None
    if GATE_UNIFORM == 'weighted':
        return torch.full_like(alpha, 1.0 / K)
    return alpha


# ----------------------------------------------------------------------------- options

def validate_options(opt):
    """R27-Q v1: Flickr, seed1001, K=5, noise 0.4/0.6, native 40-epoch SCE host, lam 0; separate <=2-epoch smoke allowed."""
    if opt.arm not in Q_ARMS:
        return
    if not opt.i2t or not opt.t2i:
        raise ValueError('Both native directions must stay ON; do not use --no_i2t/--no_t2i.')
    if opt.k != K or opt.seed != 1001 or opt.data_name != 'f30k_precomp':
        raise ValueError('R27-Q v1 is Flickr, seed1001, K=5 only.')
    if not any(abs(opt.noise_ratio - x) < 1e-12 for x in (0.4, 0.6)):
        raise ValueError('Noise ratio must be 0.4 or 0.6.')
    if opt.lam != 0.0:
        raise ValueError('R27-Q arms keep the single native SCE: lam must be 0.')
    max_iters = int(getattr(opt, 'max_iters', 0))
    smoke = (0 < max_iters <= 30 and 1 <= opt.num_epochs <= 2 and 'smoke' in str(getattr(opt, 'output_dir', '')).lower())
    if max_iters and not smoke:
        raise ValueError('Partial-iteration runs are allowed only in a separate <=2-epoch smoke directory.')
    if (opt.num_epochs != 40 and not smoke) or opt.noisy_obj != 'sce' or opt.link != 'c0':
        raise ValueError('Keep the native 40-epoch SCE host and c0 flag (separate smoke excepted).')
    if not smoke and not C.FROZEN:
        raise ValueError('r27q_config is not frozen: formal units need the frozen protocol.')
    opt.r27q_smoke_only = smoke
    opt.r27q_scorer = Q_ARMS[opt.arm]
    opt.r27q_source = SOURCE
    opt.r27q_config = {k: getattr(C, k) for k in ('FROZEN', 'HIDDEN', 'N_FIT', 'N_VAL', 'CHUNK', 'BATCH', 'LR', 'STEPS',
                                                  'BETA_HEAD', 'BETA_COS', 'SEED_BASE')}
