"""NRCS-v1 production GPO adapter (server side; the package does not include one).

Pointwise two-tower: upstream IN2R EncoderImage / EncoderText (+GPO) at 909e5329 (vse.py imported unchanged except the
R25 GloVe shim import), NOT VSE.forward_emb (that subtracts the current batch mean, vse.py:452-466). BatchNorm running
statistics are frozen for every arm (BN modules kept in eval mode after every .train() call; affine params trainable).
Encoder outputs keep upstream per-sample L2; the head input is sqrt(1024)*h for all arms; ratio heads read out
r = (u/sqrt(pi)).(v/sqrt(pi)) with no extra L2. Package critics (native_ratio.models) are reused unchanged.
No GMM, peer, GraphRefiner or queue. Training sampler sees observed maps only.
"""
import json, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vse import EncoderImage, EncoderText          # upstream (UPSTREAM_DIFF.patch: GloVe shim import only)
from native_ratio.models import SimplexRatioCritic, AffineCosineCritic

EMBED, K, REGIONS, REGION_DIM = 1024, 256, 36, 2048
HEAD_SCALE = math.sqrt(EMBED)
ARMS = ('cosine_J', 'ratio_J', 'ratio_logistic', 'cosine_hinge')
VOCAB_PATH = '/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/vocab/f30k_precomp_vocab.json'


def load_vocab():
    """IN2R co_train.py:211-214: deserialize vocab, then add '<mask>' (index 8481)."""
    w2i = dict(json.load(open(VOCAB_PATH))['word2idx'])
    assert '<mask>' not in w2i
    w2i['<mask>'] = len(w2i)
    return w2i


class _Opt:
    dropout_rate = 0.3


def build_critic(arm):
    if arm in ('cosine_J', 'cosine_hinge'):
        return AffineCosineCritic(EMBED, EMBED, K, None, 2.0, 0.5, True)
    if arm in ('ratio_J', 'ratio_logistic'):
        return SimplexRatioCritic(EMBED, EMBED, K, None, 'uniform')
    raise ValueError(arm)


class GPOTwoTower(nn.Module):
    def __init__(self, arm, word2idx):
        super().__init__()
        self.arm = arm
        self.img_enc = EncoderImage(REGION_DIM, EMBED, dropout_rate=0.3, no_imgnorm=False)
        self.txt_enc = EncoderText(_Opt(), len(word2idx), 300, EMBED, 1, use_bi_gru=True, no_txtnorm=False, word2idx=word2idx)
        self.critic = build_critic(arm)
        self.freeze_bn()

    # --- BN policy: frozen running statistics for all arms ---
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()

    def train(self, mode=True):
        super().train(mode)
        self.freeze_bn()
        return self

    def encode_image(self, images, lengths):
        return self.img_enc(images, lengths)

    def encode_text(self, tokens, lengths):
        return self.txt_enc(tokens, lengths)

    def heads_in(self, h):
        return HEAD_SCALE * h

    def param_groups(self, enc_lr, head_lr, wd):
        enc = list(self.img_enc.parameters()) + list(self.txt_enc.parameters())
        head = list(self.critic.parameters())
        assert len(enc) + len(head) == len(list(self.parameters()))
        return [dict(params=enc, lr=enc_lr, weight_decay=wd, name='encoder'),
                dict(params=head, lr=head_lr, weight_decay=wd, name='head')]


def transfer_state(parent_state, child):
    """E2 transfer: encoders (incl. BN buffers) + linear heads only; child critic extras (cosine affine) stay fresh."""
    own = child.state_dict()
    keep = {k: v for k, v in parent_state.items()
            if k.startswith(('img_enc.', 'txt_enc.', 'critic.image_head.', 'critic.text_head.'))}
    missing = [k for k in own if k not in keep]
    assert all(k in ('critic.raw_scale', 'critic.offset_value', 'critic.uniform_prior') for k in missing), missing
    for k, v in keep.items():
        assert own[k].shape == v.shape, k
    child.load_state_dict(keep, strict=False)
    return sorted(keep), sorted(missing)


# ---------------------------------------------------------------- data
class RolePool:
    """Role-local tables. Train/select pools expose observed_image_index only (no clean map is loaded)."""

    def __init__(self, npz_path, ims, device):
        role_key = None
        with np.load(npz_path, allow_pickle=False) as z:
            files = set(z.files)
            if 'observed_image_index' in files:
                assert not files & {'caption_image_index', 'true_caption_image_index', 'clean_image_index'}
                role_key = 'observed_image_index'
            else:
                role_key = 'caption_image_index'
            self.mapping = np.array(z[role_key], dtype=np.int64)
            self.mapping_kind = role_key
            self.image_rows = np.array(z['image_rows'])
            self.image_ids = np.array(z['image_ids']).astype(str)
            self.caption_ids = np.array(z['caption_ids']).astype(str)
            self.tokens = np.array(z['tokens'], dtype=np.int64)
            self.offsets = np.array(z['token_offsets'], dtype=np.int64)
        self.n_img, self.n_cap = len(self.image_rows), len(self.caption_ids)
        assert self.mapping.shape == (self.n_cap,) and self.mapping.min() >= 0 and self.mapping.max() < self.n_img
        self.feats = torch.from_numpy(np.ascontiguousarray(ims[self.image_rows])).to(device)  # [N,36,2048]
        self.device = device
        # integer identity codes for intra duplicate masks (strings never parsed)
        _, self.image_code = np.unique(self.image_ids, return_inverse=True)
        _, self.caption_code = np.unique(self.caption_ids, return_inverse=True)


def image_batch(pool, img_idx, aug_rng=None):
    """Upstream train-time region augmentation (keep region iff U>0.2) from a dedicated RNG; zero-length guarded.
    Region order is irrelevant for GPO with frozen BN (pool weights depend on lengths; features sorted per dim)."""
    x = pool.feats[torch.as_tensor(img_idx, device=pool.device)]
    B = x.shape[0]
    guard = 0
    if aug_rng is None:
        lengths = torch.full((B,), REGIONS, dtype=torch.long, device=pool.device)
        return x, lengths, guard
    keep = aug_rng.random((B, REGIONS)) > 0.20
    empty = ~keep.any(1)
    guard = int(empty.sum())
    keep[empty] = True
    order = np.argsort(~keep, axis=1, kind='stable')
    lengths_np = keep.sum(1)
    o = torch.as_tensor(order, device=pool.device)
    x = x.gather(1, o[:, :, None].expand(-1, -1, REGION_DIM))
    lengths = torch.as_tensor(lengths_np, dtype=torch.long, device=pool.device)
    L = int(lengths_np.max())
    x = x[:, :L]
    valid = torch.arange(L, device=pool.device)[None] < lengths[:, None]
    x = x * valid[:, :, None]
    return x, lengths, guard


def text_batch(pool, cap_idx, vocab_len, mask_id, aug_rng=None):
    """Upstream process_caption: drop=False for eval; drop=True for training (20%: 50% <mask>, 10% random, 40% delete;
    keep all tokens if every token would be deleted - R25 fix), from a dedicated RNG."""
    seqs, guard = [], 0
    for c in cap_idx:
        ids = pool.tokens[pool.offsets[c]:pool.offsets[c + 1]].copy()
        if aug_rng is not None:
            p = aug_rng.random(len(ids))
            rnd = aug_rng.integers(0, vocab_len, len(ids))
            sel = p < 0.20
            q = p / 0.20
            ids = np.where(sel & (q < 0.5), mask_id, ids)
            ids = np.where(sel & (q >= 0.5) & (q < 0.6), rnd, ids)
            dele = sel & (q >= 0.6)
            nd = int(dele.sum())
            if nd and nd < len(ids):
                ids = ids[~dele]
            elif nd == len(ids):
                guard += 1
        seqs.append(ids)
    lengths = [len(s) for s in seqs]
    assert min(lengths) >= 1
    T = np.zeros((len(seqs), max(lengths)), dtype=np.int64)
    for i, s in enumerate(seqs):
        T[i, :len(s)] = s
    return torch.as_tensor(T, device=pool.device), lengths, guard


@torch.no_grad()
def encode_pool(model, pool, vocab_len, mask_id, chunk=256):
    """Eval-mode, dropout off, frozen BN, no augmentation, no batch centering: pointwise h for every image/caption."""
    was = model.training
    model.eval()
    hi, ht = [], []
    for lo in range(0, pool.n_img, chunk):
        idx = np.arange(lo, min(lo + chunk, pool.n_img))
        x, l, _ = image_batch(pool, idx)
        hi.append(model.encode_image(x, l))
    for lo in range(0, pool.n_cap, chunk):
        idx = np.arange(lo, min(lo + chunk, pool.n_cap))
        t, l, _ = text_batch(pool, idx, vocab_len, mask_id)
        ht.append(model.encode_text(t, l))
    model.train(was)
    return torch.cat(hi), torch.cat(ht)


# ---------------------------------------------------------------- losses
def hardest_triplet_2B(scores, margin, neg_mask=None):
    """Bidirectional hardest-negative hinge (IN2R ContrastiveLoss mode='train' formula), sum over rows + columns
    divided by 2B. neg_mask[i,j]=True marks entries that may NOT act as negatives (diagonal always excluded)."""
    B = scores.shape[0]
    d = scores.diag().view(B, 1)
    cost_s = (margin + scores - d).clamp(min=0)          # row i: image i vs other captions
    cost_im = (margin + scores - d.t()).clamp(min=0)     # column j: caption j vs other images
    excl = torch.eye(B, dtype=torch.bool, device=scores.device)
    if neg_mask is not None:
        excl = excl | neg_mask
    cost_s = cost_s.masked_fill(excl, 0)
    cost_im = cost_im.masked_fill(excl, 0)
    return (cost_s.max(1)[0].sum() + cost_im.max(0)[0].sum()) / (2 * B)


def intra_loss(h1_img, h2_img, img_code, h1_txt, h2_txt, cap_code, margin=0.3):
    """Same-modality two-dropout-view hardest triplet; duplicate identities never act as intra negatives (aux only)."""
    ic = torch.as_tensor(img_code, device=h1_img.device)
    cc = torch.as_tensor(cap_code, device=h1_img.device)
    li = hardest_triplet_2B(h1_img @ h2_img.t(), margin, ic[:, None] == ic[None, :])
    lt = hardest_triplet_2B(h1_txt @ h2_txt.t(), margin, cc[:, None] == cc[None, :])
    return li + lt, li, lt
