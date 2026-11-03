"""R41: R40 r39ext.py with per-cond JOINT parents (C.JOINT_PARENTS: checkpoint, JOINT prep dir, native code; as R39X r39xf). R40: R39 r39ext.py + update scope (Student(modalities): the frozen modality keeps the parent weights with requires_grad=False, so it is
not in trainable() / the optimizer / the clip, and its whole state is part of frozen_state(); param_distance split per modality).
R39 additions, byte-identical in both code trees (r39f Flickr60, r39c CC152K); dataset specifics come from the tree's common.py.

COMPLETE (package docs/SERVER_PLAN_CN.md section 2): fixed, cross-checked completion of OBSERVED-MISSING rows of the R38 N lists.
  Two fixed same-series teacher parents (their own peer-mean native score) each give a full-FIT-gallery top-1 for every FIT query
  and the reverse rank of the query for that proposal (tools/prep_complete.py, written once, hash-bound). A row is completed iff
  (i) all its N label mass is on the null slot, (ii) both teachers propose the same ID, (iii) that ID is already among the target
  H's natural top-32 of the row (no insertion), (iv) both reverse ranks <= 5. Completed row: A = 0.5 one_hot(agreed) + 0.5 null.
  The rule is the package reference r39_targets.complete_missing (byte copy of reference/targets.py), applied per sampled batch;
  every other row keeps its N label bitwise. No truth is read.
JOINT (section 3): per peer a student copy of the parent's GLOBAL encoders (native vse.EncoderImage: fc + MLP + GPO + l2norm;
  native vse.EncoderText: frozen word embedding + BiGRU + fc + MLP + GPO + l2norm), re-run from the stored region features and
  tokens. Native forward_emb then subtracts the per-batch mean of each modality and L2-normalises; that batch mean is held FIXED
  at the parent's value of the item's native encoding batch (128 sequential rows; tools/prep_joint.py) - registered
  compatibility patch: an arbitrary training batch cannot reproduce the native 128-row batch, so the student uses the parent's
  centring vector. BN uses running statistics (modules in eval), dropout is off; the GRUs (caption BiGRU, both GPO GRUs; one
  layer, dropout 0) are put in train mode only because cuDNN RNN backward requires it (forward semantics unchanged; equivalence
  to the native cache is measured in prep and in every unit). The skip term s_H and the candidates stay those of the frozen H.
"""
from __future__ import annotations
import hashlib, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn

import common as C
from common import PEERS, N_TRAIN_IMG, N_TRAIN_CAP, N_DEV_IMG, N_DEV_CAP, sha256, read_json
from r39_targets import complete_missing

PER_CAP = C.PER_CAP
NATIVE_BATCH = 128
NULL_ID = -1
REAL_MASS = 0.5
RECIPROCAL_MAX = 5


# ============================================================================ COMPLETE

def completion_path(ds=None):
    return C.R39_PREP / (ds or C.DS_KEY) / f'completion_{ds or C.DS_KEY}.npz'


class Completion:
    """per-query proposal tables of the two fixed teachers (+ precomputed acceptance, re-derived per batch and asserted)."""
    def __init__(self, cond, d):
        assert cond == C.TARGET_COND, f'completion tables exist for the target parent {C.TARGET_COND} only'
        man = read_json(C.R39_PREP / C.DS_KEY / 'COMPLETE_PREP_MANIFEST.json')
        p = completion_path()
        assert sha256(p) == man['files'][p.name]['sha256'], 'completion table changed since prep'
        assert man['target']['candidate_table_sha256'] == d['cand_sha256'] and man['target']['cache_manifest_sha256'] == d['manifest_sha256']
        z = np.load(p)
        self.sha256 = man['files'][p.name]['sha256']; self.manifest = man
        self.pa_i, self.pb_i = z['t1_i2t_top1'].astype(np.int64), z['t2_i2t_top1'].astype(np.int64)
        self.rr_i = np.stack([z['t1_i2t_rr'], z['t2_i2t_rr']], 1).astype(np.float64)
        self.pa_t, self.pb_t = z['t1_t2i_top1'].astype(np.int64), z['t2_t2i_top1'].astype(np.int64)
        self.rr_t = np.stack([z['t1_t2i_rr'], z['t2_t2i_rr']], 1).astype(np.float64)
        self.acc_i, self.acc_t = z['i2t_accept'].astype(bool), z['t2i_accept'].astype(bool)
        assert self.pa_i.shape == (N_TRAIN_IMG,) and self.pa_t.shape == (N_TRAIN_CAP,)

    def apply_dir(self, ids, a, q, pa, pb, rr, acc_pre):
        out, acc = complete_missing(a, ids, pa[q], pb[q], rr[q], real_mass=REAL_MASS, reciprocal_max=RECIPROCAL_MAX, null_id=NULL_ID)
        assert np.array_equal(acc, acc_pre[q]), 'per-batch acceptance differs from the prep table'
        out = out.astype(a.dtype)   # keep the N label dtype (data36.null_lists gives float64 rows, e.g. 1/3 for 3 observed positives): unchanged rows stay bitwise
        # invariants: unchanged rows bitwise; changed rows were all-null and now hold 0.5 null + 0.5 on one real slot
        assert np.array_equal(out[~acc], a[~acc])
        if acc.any():
            nul = ids[acc] == NULL_ID
            assert np.all(a[acc][nul] == 1.0) and np.all(out[acc][nul] == REAL_MASS)
            assert np.all((out[acc] == REAL_MASS).sum(1) == 2) and np.all((out[acc] > 0).sum(1) == 2)
        return out, acc

    def apply(self, lists):
        (qi, caps, ai), (qc, imgs, at) = lists
        ni, acc_i = self.apply_dir(caps, ai, qi, self.pa_i, self.pb_i, self.rr_i, self.acc_i)
        nt, acc_t = self.apply_dir(imgs, at, qc, self.pa_t, self.pb_t, self.rr_t, self.acc_t)
        null_i = ai[caps == NULL_ID] == 1.0; null_t = at[imgs == NULL_ID] == 1.0
        st = dict(complete_null_rows_i2t=float(null_i.sum()), complete_null_rows_t2i=float(null_t.sum()),
                  complete_completed_rows_i2t=float(acc_i.sum()), complete_completed_rows_t2i=float(acc_t.sum()))
        return ((qi, caps, ni), (qc, imgs, nt)), st


# ============================================================================ JOINT: student global encoders

def native_vse():
    if str(C.NATIVE_CODE) not in sys.path:
        sys.path.insert(1, str(C.NATIVE_CODE))
    import vse as V
    return V


class Student(nn.Module):
    """one peer: native EncoderImage + EncoderText (parent weights), see module docstring for modes."""
    def __init__(self, img_state, txt_state, opt, modalities=('img', 'txt')):
        super().__init__()
        assert set(modalities) <= {'img', 'txt'} and len(modalities) > 0, modalities
        self.modalities = tuple(m for m in ('img', 'txt') if m in modalities)   # R40: updated modalities (the rest stays the parent)
        V = native_vse()

        class _Text(V.EncoderText):
            def init_weights(self, word2idx):   # parent weights are loaded right after construction (no GloVe file access)
                return None
        with torch.random.fork_rng(devices=[]):   # construction draws no RNG of the run
            self.img = V.EncoderImage(opt.img_dim, opt.embed_size, dropout_rate=opt.dropout_rate, no_imgnorm=opt.no_imgnorm)
            self.txt = _Text(opt, opt.vocab_size, opt.word_dim, opt.embed_size, opt.num_layers, use_bi_gru=opt.bi_gru,
                             no_txtnorm=opt.no_txtnorm, word2idx=None)
        self.img.load_state_dict(img_state); self.txt.load_state_dict(txt_state)
        self.txt.embed.weight.requires_grad_(False)
        # native vse.MLP builds one BatchNorm per layer incl. the output layer, but its forward applies BN only to the hidden
        # layers: the output BN (mlp.bns[num_layers-1]) is never used; kept frozen (no gradient exists for it either way)
        for enc in (self.img, self.txt):
            for q in enc.mlp.bns[enc.mlp.num_layers - 1].parameters():
                q.requires_grad_(False)
        for m in ('img', 'txt'):   # R40: frozen modality = parent parameters, requires_grad False (never in the optimizer)
            if m not in self.modalities:
                for q in getattr(self, m).parameters():
                    q.requires_grad_(False)
        self.l2norm = V.l2norm
        self.train()

    def train(self, mode=True):   # BN running stats + dropout off always; one-layer GRUs in train mode (cuDNN RNN backward)
        super().train(False)
        for g in (self.txt.cap_rnn, self.txt.gpool.gru, self.img.gpool.gru):
            g.training = True
        return self

    def trainable(self):
        return [q for n, q in self.named_parameters() if q.requires_grad]

    def frozen_state(self):
        """tensors that must never change: word embedding, BN running statistics, the unused output BN of the native MLP."""
        fm = tuple(m + '.' for m in ('img', 'txt') if m not in self.modalities)   # R40: + every tensor of the frozen modality
        return {n: t for n, t in self.state_dict().items() if n == 'txt.embed.weight' or 'running_' in n or 'num_batches_tracked' in n or '.mlp.bns.1.' in n or n.startswith(fm)}

    def enc_img(self, ims):
        lengths = torch.full((ims.shape[0],), float(ims.shape[1]), device=ims.device)   # native collate: float image lengths (all 36)
        return self.img(ims, lengths)

    def enc_cap(self, toks, lens):
        lb = int(lens.max())
        return self.txt(toks[:, :lb], lens.cpu())


def load_parent(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    assert int(ck['opt'].batch_size) == NATIVE_BATCH
    return ck


def build_students(ck, device, modalities=('img', 'txt')):
    s = nn.ModuleDict({p: Student(ck[f'model_{p}'][0], ck[f'model_{p}'][1], ck['opt'], modalities) for p in PEERS}).to(device)
    for p in PEERS:
        s[p].train()
    return s


def batch_index(kind, ids):
    """native encoding batch of an item: row r = caption r = image r // PER_CAP; batch = row // 128 (image j: row PER_CAP*j)."""
    return (PER_CAP * ids) // NATIVE_BATCH if kind == 'img' else ids // NATIVE_BATCH


def state_digest(tensors):
    h = hashlib.sha256()
    for k in sorted(tensors):
        h.update(k.encode()); h.update(tensors[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class JointInputs:
    """raw region features, tokens and the fixed parent centring vectors of the target parent (hash-verified against
    JOINT_PREP_MANIFEST.json), plus the encoder of explicit (image, caption) pairs."""
    def __init__(self, cond, d, device):
        jp = C.JOINT_PARENTS[cond]   # R41 (as R39X): per-cond JOINT parent (checkpoint, prep dir, native code dir)
        self.dev = device; self.d = d; self.ckpt_path = jp['ckpt']
        pdir = Path(jp['prep_dir'])
        assert sha256(Path(jp['code']) / 'vse.py') == sha256(C.NATIVE_CODE / 'vse.py'), 'parent code vse.py differs from the student module source'
        self.manifest = read_json(pdir / 'JOINT_PREP_MANIFEST.json'); man = self.manifest
        assert man['parent']['model_best_sha256'] == sha256(jp['ckpt']) == d['manifest']['parent']['model_best_sha256']
        assert man['target_cache_manifest_sha256'] == d['manifest_sha256'] and man.get('cond', cond) == cond
        self.ck = load_parent(jp['ckpt'])
        tr = np.load(C.RAW_FIT_IMS, mmap_mode='r'); dv = np.load(C.RAW_DEV_IMS, mmap_mode='r')
        assert tr.shape == (N_TRAIN_IMG, 36, 2048) and dv.shape[0] == N_DEV_IMG * C.DEV_IMS_ROW_STRIDE
        # GPU: copy from the mmap; host: force a RAM copy (np.array), else every per-step gather reads the NFS mmap (CC152K prep check:
        # 14 s/update on the mmap)
        fit = torch.from_numpy(np.ascontiguousarray(tr)).to(device) if C.RAW_FIT_ON_GPU else torch.from_numpy(np.array(tr))
        self.ims = dict(fit=fit,
                        dev=torch.from_numpy(np.ascontiguousarray(dv[::C.DEV_IMS_ROW_STRIDE])).to(device))
        assert sha256(C.TOKENS_FILE) == man['tokens']['sha256']
        T = np.load(C.TOKENS_FILE)
        self.tok = dict(fit=torch.from_numpy(T['train_tok']).to(device), dev=torch.from_numpy(T['dev_tok']).to(device))
        self.len = dict(fit=torch.from_numpy(T['train_len']).to(device), dev=torch.from_numpy(T['dev_len']).to(device))
        assert len(self.len['fit']) == N_TRAIN_CAP and len(self.len['dev']) == N_DEV_CAP
        self.mu = {}
        for split in ('fit', 'dev'):
            for kind in ('img', 'cap'):
                for p in PEERS:
                    f = pdir / f'center_{split}_{kind}_peer{p}.npy'
                    assert sha256(f) == man['files'][f.name]['sha256'], f'centring file changed: {f}'
                    self.mu[(split, kind, p)] = torch.from_numpy(np.load(f)).to(device)
        self.n = dict(fit=dict(img=N_TRAIN_IMG, cap=N_TRAIN_CAP), dev=dict(img=N_DEV_IMG, cap=N_DEV_CAP))
        self._eval = {}
        self.l2norm = native_vse().l2norm

    # ------------------------------------------------------------------ encoding
    def raw_encode(self, stud_p, split, kind, ids_t):
        """encoder output before the native centring (already l2-normalised by the encoder), ids_t: long tensor (any device)."""
        if kind == 'img':
            src = self.ims[split]
            x = src.index_select(0, ids_t.to(src.device)).to(self.dev)
            return stud_p.enc_img(x)
        ii = ids_t.to(self.dev)
        return stud_p.enc_cap(self.tok[split][ii], self.len[split][ii])

    def encode(self, stud_p, p, split, kind, ids, grad=False, chunk=None):
        ids_t = torch.as_tensor(np.asarray(ids), dtype=torch.long)
        chunk = chunk or (1 << 30 if grad else (512 if kind == 'img' else 2048))
        outs = []
        with (torch.enable_grad() if grad else torch.no_grad()):
            for i in range(0, len(ids_t), chunk):
                c = ids_t[i:i + chunk]
                e = self.raw_encode(stud_p, split, kind, c)
                mu = self.mu[(split, kind, p)][batch_index(kind, c.to(self.dev))]
                outs.append(self.l2norm(e - mu))
        return torch.cat(outs)

    def table(self, stud_p, p, split, kind):
        key = (p, split, kind)
        if key not in self._eval:
            self._eval[key] = self.encode(stud_p, p, split, kind, np.arange(self.n[split][kind]), grad=False)
        return self._eval[key]

    def clear_eval(self):
        self._eval = {}

    def pairs(self, stud_p, p, split, img, cap, grad=False):
        """(U, inv_img, V, inv_cap): student vectors of the unique IDs and the per-pair index into them (dedup by ID)."""
        if not grad:   # evaluation: full split tables (cached until clear_eval)
            return (self.table(stud_p, p, split, 'img'), torch.as_tensor(np.asarray(img), device=self.dev),
                    self.table(stud_p, p, split, 'cap'), torch.as_tensor(np.asarray(cap), device=self.dev))
        ui, inv_i = np.unique(img, return_inverse=True); uc, inv_c = np.unique(cap, return_inverse=True)
        return (self.encode(stud_p, p, split, 'img', ui, grad=True), torch.as_tensor(inv_i.ravel(), device=self.dev),
                self.encode(stud_p, p, split, 'cap', uc, grad=True), torch.as_tensor(inv_c.ravel(), device=self.dev))

    # ------------------------------------------------------------------ checks / readouts
    @torch.no_grad()
    def cache_equivalence(self, stud, img, cap, split='fit', grad=True):
        """student vectors (forward WITH grad enabled, as in training, if grad) vs the frozen native cache, on real items."""
        out = {}
        for p in PEERS:
            for kind, ids in (('img', np.unique(img)), ('cap', np.unique(cap))):
                with torch.enable_grad() if grad else torch.no_grad():
                    u = self.encode(stud[p], p, split, kind, ids, grad=grad)
                u = u.detach()
                ref = self.d[(split, kind, p)][torch.as_tensor(ids, device=self.dev)]
                dlt = (u - ref).abs()
                out[f'{kind}_{p}'] = dict(n=int(len(ids)), max_abs=float(dlt.max()), mean_abs=float(dlt.mean()), frac_bitwise=float((dlt == 0).all(1).float().mean()),
                                          min_cos=float((u * ref).sum(1).min()))
        return out

    @torch.no_grad()
    def feature_change(self, stud):
        """student vs frozen-H (cache) global vectors over ALL items of FIT and DEV (eval tables), per peer and modality."""
        out = {}
        for p in PEERS:
            for split in ('fit', 'dev'):
                for kind in ('img', 'cap'):
                    u = self.table(stud[p], p, split, kind); ref = self.d[(split, kind, p)]
                    cs = (u * ref).sum(1)
                    out[f'{split}_{kind}_{p}'] = dict(mean_cos=float(cs.double().mean()), p05_cos=float(torch.quantile(cs.double()[:200000], .05)),
                                                      min_cos=float(cs.min()), mean_l2=float((u - ref).norm(dim=1).double().mean()))
        return out

    def param_distance(self, stud, ref_states):
        out = {}
        for p in PEERS:
            sq = 0.0; sq0 = 0.0; mod = {m: [0.0, 0.0] for m in ('img', 'txt')}
            for n, q in stud[p].named_parameters():
                if q.requires_grad:
                    r = ref_states[p][n]
                    a_, b_ = float(((q.detach() - r) ** 2).sum()), float((r ** 2).sum())
                    sq += a_; sq0 += b_; mod[n.split('.')[0]][0] += a_; mod[n.split('.')[0]][1] += b_
            out[p] = dict(l2_from_parent=sq ** .5, rel_l2_from_parent=(sq / sq0) ** .5,
                          per_modality={m: dict(l2_from_parent=v[0] ** .5, rel_l2_from_parent=(v[0] / v[1]) ** .5 if v[1] > 0 else None) for m, v in mod.items()})
        return out
