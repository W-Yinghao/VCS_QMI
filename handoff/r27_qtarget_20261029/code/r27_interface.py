"""R27 shared candidate-identity + consistent-encoding interface (DESIGN_CN.md §2). Identical for A1-A5.

1. Queue identities: a parallel ID buffer on every native MemoryQueue (q.r27_ids: cap = global train caption index,
   img = OBSERVED (training-visible, possibly noisy) image index of that caption, wepoch / witer = co-training epoch and
   iteration of the write, peer = 0/1 owner). Written in lock-step with the native enqueue (same pointer arithmetic, asserted).
   Prefix-era slots (written by the native prefix epoch, no IDs) carry cap = -1 and are EXCLUDED from search until
   overwritten. Injected-noise truth (dataset._t2i_index / _labels) is never read here.
2. Identity search (as R26): the OTHER peer net2 (eval mode, no grad) encodes the current noisy batch and searches its
   OWN queue in the same modality (masked to ID-bearing slots), top-K = 5 uniform; atoms = unique sample IDs
   (i2t: caption IDs; t2i: observed image IDs), duplicates merged with accumulated mass.
3. Consistent encoding: the CURRENT peer net re-encodes every atom from raw data in eval mode under no_grad
   (images: all 36 regions, no region drop; captions: no token augmentation), subtracts the SAME current noisy batch's
   pre-centering mean of that modality (detached, taken from the native forward_emb of the query batch) and
   unit-normalizes (host l2norm). The snapshot is stop-grad. Queries keep the native forward_emb values.
"""
import time
import numpy as np
import torch
from vse import l2norm
from data import process_caption

KEYS = ('cap', 'img', 'wepoch', 'witer', 'peer')


class Ctx:
    """Global training data handles (training-visible only): caption strings, observed image features, vocab."""
    captions = None
    images = None          # numpy mmap (N_img, 36, 2048)
    vocab = None
    images_gpu = None
    tok = {}

    @classmethod
    def set(cls, captions, images, vocab):
        if cls.captions is None:
            cls.captions, cls.images, cls.vocab = captions, images, vocab

    @classmethod
    def gpu_images(cls):
        if cls.images_gpu is None:
            n = cls.images.shape[0]
            t = torch.empty(cls.images.shape, dtype=torch.float32, device='cuda')
            for s in range(0, n, 1000):
                t[s:s + 1000] = torch.from_numpy(np.ascontiguousarray(cls.images[s:s + 1000])).cuda()
            cls.images_gpu = t
        return cls.images_gpu

    @classmethod
    def tokens(cls, cid):
        t = cls.tok.get(cid)
        if t is None:
            t = process_caption(cls.vocab, cls.captions[cid], False).long()  # drop=False: no augmentation, no RNG
            cls.tok[cid] = t
        return t


# ----------------------------------------------------------------------------- ID buffer

def ensure_ids(q):
    if not hasattr(q, 'r27_ids'):
        dev = q.img_queue.device
        q.r27_ids = {k: torch.full((q.queue_size,), -1, dtype=torch.long, device=dev) for k in KEYS}
    return q.r27_ids


def enqueue_with_ids(q, img_feats, txt_feats, cap_ids, img_ids, wepoch, witer, peer):
    """Native MemoryQueue.enqueue + the same pointer arithmetic on the ID buffer (asserted)."""
    ids = ensure_ids(q)
    n = img_feats.shape[0]
    p0, Q = q.ptr, q.queue_size
    q.enqueue(img_feats, txt_feats)
    dev = ids['cap'].device
    vals = {'cap': torch.as_tensor(cap_ids, dtype=torch.long, device=dev),
            'img': torch.as_tensor(img_ids, dtype=torch.long, device=dev),
            'wepoch': torch.full((n,), int(wepoch), dtype=torch.long, device=dev),
            'witer': torch.full((n,), int(witer), dtype=torch.long, device=dev),
            'peer': torch.full((n,), int(peer), dtype=torch.long, device=dev)}
    if p0 + n <= Q:
        for k, v in vals.items():
            ids[k][p0:p0 + n] = v
        assert q.ptr == p0 + n
    else:
        tail = Q - p0
        rem = n - tail
        for k, v in vals.items():
            ids[k][p0:] = v[:tail]
            ids[k][:rem] = v[tail:]
        assert q.ptr == rem and q.is_full


def ids_state(q):
    return {k: v.detach().cpu().clone() for k, v in q.r27_ids.items()} if hasattr(q, 'r27_ids') else None


def load_ids(q, s):
    ids = ensure_ids(q)
    if s is None:
        for v in ids.values():
            v.fill_(-1)
        return False
    for k in ids:
        ids[k].copy_(s[k].to(ids[k].device))
    return True


def usable(q):
    n = q.queue_size if q.is_full else q.ptr
    ids = ensure_ids(q)
    v = ids['cap'][:n] >= 0
    return n, v


# ----------------------------------------------------------------------------- forward with centering context

def forward_emb_ctx(net, images, captions, lengths, imgae_lengths):
    """Line-for-line VSE.forward_emb, additionally returning the detached pre-centering modality means."""
    if torch.cuda.is_available():
        images = images.cuda()
        captions = captions.cuda()
        imgae_lengths = imgae_lengths.cuda()
    img_embs = net.img_enc(images, imgae_lengths)
    cap_embs = net.txt_enc(captions, lengths)
    mi = img_embs.mean(dim=0, keepdim=True)
    mt = cap_embs.mean(dim=0, keepdim=True)
    img_embs = img_embs - mi
    img_embs = l2norm(img_embs)
    cap_embs = cap_embs - mt
    cap_embs = l2norm(cap_embs)
    return img_embs, cap_embs, lengths, mi.detach(), mt.detach()


# ----------------------------------------------------------------------------- search + re-encode

@torch.no_grad()
def search(q2, query2, branch, k):
    """Identity search of the other peer's own queue. Returns dict or None (fewer than k ID-bearing slots)."""
    n, valid = usable(q2)
    nv = int(valid.sum())
    if nv < k:
        return None, dict(n=n, nv=nv)
    ids = q2.r27_ids
    bank = q2.img_queue[:n] if branch == 'i2t' else q2.txt_queue[:n]
    sims = query2.mm(bank.t())
    sims.masked_fill_(~valid.unsqueeze(0), float('-inf'))
    slots = torch.topk(sims, k, dim=1).indices                       # [B,K]
    atom = (ids['cap'] if branch == 'i2t' else ids['img'])[slots]    # sample IDs [B,K]
    cols, inv = torch.unique(atom.flatten(), return_inverse=True)
    inv = inv.view_as(atom)
    B, M = slots.shape[0], cols.numel()
    A = torch.zeros(B, M, device=query2.device, dtype=torch.float32)
    A.scatter_add_(1, inv, torch.full_like(inv, 1.0 / k, dtype=torch.float32))
    return dict(slots=slots, cols=cols, inv=inv, A=A, wepoch=ids['wepoch'][slots], cap=ids['cap'][slots],
                img=ids['img'][slots]), dict(n=n, nv=nv)


@torch.no_grad()
def reencode(net, branch, cols, mean):
    """Current-peer eval-mode re-encoding of atom IDs, centered with the current noisy batch's modality mean."""
    was = net.training
    net.val_start()
    try:
        if branch == 'i2t':
            toks = [Ctx.tokens(int(c)) for c in cols.tolist()]
            lengths = [len(t) for t in toks]
            cap = torch.zeros(len(toks), max(lengths), dtype=torch.long)
            for j, t in enumerate(toks):
                cap[j, :len(t)] = t
            raw = net.txt_enc(cap.cuda(), lengths)
        else:
            ims = Ctx.gpu_images()[cols]
            lens = torch.full((ims.shape[0],), float(ims.shape[1]), device=ims.device)
            raw = net.img_enc(ims, lens)
    finally:
        if was:
            net.train_start()
    return l2norm(raw - mean)
