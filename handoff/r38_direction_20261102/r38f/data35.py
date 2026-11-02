"""R37 data (copied from R36/R35 data35.py; load_cond generalised to the bound parents): R34 raw (unstandardised) native global caches (hash-verified), R35 prep locks, FIT candidate
lists, local (pre-GPO) encoders. Only train/dev arrays are ever opened (TEST guard active in every entry)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CONDS, PEERS, PREP, R35_PREP, R34_OUT, R30_RUNS, CODE_R30, DATA_SRC, N_TRAIN_CAP, N_TRAIN_IMG, N_DEV_IMG,
                    N_DEV_CAP, sha256, read_json)
import evals as EV

N_SLOTS, N_Q = 32, 16


# ----------------------------------------------------------------------------- global caches

def shard_mm(x, y, shard=100):
    """native evaluation.shard_attn_scores arithmetic: float32 mm on 100x100 shards, stored as float64."""
    out = torch.zeros(len(x), len(y), dtype=torch.float64, device=x.device)
    for i in range(0, len(x), shard):
        for j in range(0, len(y), shard):
            out[i:i + shard, j:j + shard] = (x[i:i + shard] @ y[j:j + shard].T).double()
    return out


def load_cond(cond, device, verify=True, need_prep=True):
    """Cache of the bound parent (raw native unit vectors; R34 cache for 'n60', R37 cache for the new parents) +
    R34 DEV-observed lock (evaluation-only) + prep: record locks/tokens always from R35 prep (hash-verified against the
    R35 manifest); calibration a0/b0, top-32 tables and the B-TRUST subset from the parent's own prep dir."""
    c = CONDS[cond]
    cdir = Path(c['cache_dir'])
    man = read_json(cdir / 'CACHE_MANIFEST.json'); locks = read_json(R34_OUT / 'locks' / 'LOCKS.json')
    dl = locks['dev_observed_maps'][c['dev_lock']]
    if verify:
        for name, rec in man['files'].items():
            assert sha256(cdir / name) == rec['sha256'], f'cache hash mismatch {name}'
        assert sha256(dl['path']) == dl['sha256']
        assert sha256(man['observed_map']['path']) == man['observed_map']['sha256'] == c['noise_sha256']
        assert man['parent']['unit'] == c['h_unit'], (man['parent']['unit'], c['h_unit'])
    d = dict(cond=cond, manifest=man, manifest_sha256=sha256(cdir / 'CACHE_MANIFEST.json'))
    obs = np.load(man['observed_map']['path']).astype(np.int64); assert obs.shape == (N_TRAIN_CAP,)
    d['obs_np'] = obs; d['obs'] = torch.from_numpy(obs).to(device)
    for p in PEERS:
        for split in ('fit', 'dev'):
            d[(split, 'img', p)] = torch.from_numpy(np.load(cdir / f'{split}_img_peer{p}.npy')).to(device)
            d[(split, 'cap', p)] = torch.from_numpy(np.load(cdir / f'{split}_cap_peer{p}.npy')).to(device)
    d['native_dev'] = np.load(cdir / 'native_dev_sims.npy')
    d['dev_obs'] = np.load(dl['path']).astype(np.int64)
    d['dev_clean'] = np.arange(N_DEV_CAP, dtype=np.int64) // 5
    d['dev_s'] = {p: shard_mm(d[('dev', 'img', p)], d[('dev', 'cap', p)]) for p in PEERS}
    d['dev_tab'] = EV.dev_tables(d['native_dev'])
    if need_prep:
        r35m = read_json(R35_PREP / 'PREP_MANIFEST.json')
        pdir = Path(c['prep_dir']); pm = read_json(pdir / 'PREP_MANIFEST.json')
        if verify:
            for name in ('locks_records.npz', 'tokens.npz'):
                assert sha256(R35_PREP / name) == r35m['files'][name]['sha256'], f'R35 prep hash mismatch {name}'
            for name in (f"candidates_{c['prep_key']}.npz", c['trust_file']):
                assert sha256(pdir / name) == pm['files'][name]['sha256'], f'prep hash mismatch {name}'
            if pdir != R35_PREP:
                assert pm['r35_prep_manifest_sha256'] == sha256(R35_PREP / 'PREP_MANIFEST.json')
        d['prep_manifest_sha256'] = sha256(pdir / 'PREP_MANIFEST.json')
        d['r35_prep_manifest_sha256'] = sha256(R35_PREP / 'PREP_MANIFEST.json')
        d['a0b0'] = pm['calibration_init'][c['prep_key']]
        L = np.load(R35_PREP / 'locks_records.npz')
        d['locks'] = {k: L[k] for k in L.files}
        cand = np.load(pdir / f"candidates_{c['prep_key']}.npz")
        d['cand_i2t'] = cand['i2t_top32']; d['cand_t2i'] = cand['t2i_top32']
        d['cand_sha256'] = pm['files'][f"candidates_{c['prep_key']}.npz"]['sha256']
        T = np.load(pdir / c['trust_file']); d['trust_subset'] = T['subset'].astype(np.int64)
        d['trust_sha256'] = pm['files'][c['trust_file']]['sha256']
    return d


# ----------------------------------------------------------------------------- candidate lists (C/D)

def build_lists(rec_i, rec_t, obs, cand_i2t, cand_t2i, perm_i, perm_t):
    """rec_i: records whose observed image is the i2t query (positive caption = the record);
    rec_t: records whose caption is the t2i query (positive image = observed image of the record).
    Slots: observed positive + first 31 frozen-H candidates other than it (from a per-query top-32 table, so
    always 31 distinct others), then permuted by perm_*. A = observed positives (i2t: every slot caption whose
    observed image equals the query image) normalised per row. Truth is never read.
    Returns (q_img_i [n], caps_i [n,32], A_i), (q_cap_t [n], imgs_t [n,32], A_t)."""
    n_i, n_t = len(rec_i), len(rec_t)
    qi = obs[rec_i]
    caps = np.zeros((n_i, N_SLOTS), np.int64)
    for r in range(n_i):
        oth = cand_i2t[qi[r]]; oth = oth[oth != rec_i[r]][:N_SLOTS - 1]
        caps[r, 0] = rec_i[r]; caps[r, 1:] = oth
    caps = np.take_along_axis(caps, perm_i, 1)
    ai = (obs[caps] == qi[:, None]).astype(np.float32); ai /= ai.sum(1, keepdims=True)
    imgs = np.zeros((n_t, N_SLOTS), np.int64)
    pi = obs[rec_t]
    for r in range(n_t):
        oth = cand_t2i[rec_t[r]]; oth = oth[oth != pi[r]][:N_SLOTS - 1]
        imgs[r, 0] = pi[r]; imgs[r, 1:] = oth
    imgs = np.take_along_axis(imgs, perm_t, 1)
    at = (imgs == pi[:, None]).astype(np.float32); at /= at.sum(1, keepdims=True)
    return (qi, caps, ai), (rec_t.copy(), imgs, at)


def sample_lists(rng, obs, cand_i2t, cand_t2i, n=N_Q):
    rec_i = rng.integers(0, N_TRAIN_CAP, n); rec_t = rng.integers(0, N_TRAIN_CAP, n)
    perm_i = np.stack([rng.permutation(N_SLOTS) for _ in range(n)]); perm_t = np.stack([rng.permutation(N_SLOTS) for _ in range(n)])
    return build_lists(rec_i, rec_t, obs, cand_i2t, cand_t2i, perm_i, perm_t), (rec_i, rec_t, perm_i, perm_t)


# ----------------------------------------------------------------------------- local (pre-GPO) encoders

def _vse():
    if str(CODE_R30) not in sys.path:
        sys.path.insert(1, str(CODE_R30))
    import vse as V
    return V


class LocalEncoder(nn.Module):
    """Copy of one H peer's local path: image fc + mlp (pre-GPO region vectors, 36 x 1024) and text
    embedding (GloVe-initialised, trained H weights; frozen) + BiGRU (token vectors = mean of the two
    directions, as EncoderText before fc/mlp/GPO). Dropout is not applied (eval-equivalent); MLP BatchNorm
    uses running statistics (module kept in eval mode, never updates them). The GRU module is kept in
    train mode only because cuDNN RNN backward requires it; one layer, so no dropout is involved."""
    def __init__(self, img_state, txt_state, vocab_size):
        super().__init__()
        V = _vse()
        self.fc = nn.Linear(2048, 1024)
        self.mlp = V.MLP(2048, 512, 1024, 2)
        self.embed = nn.Embedding(vocab_size, 300)
        self.rnn = nn.GRU(300, 1024, 1, batch_first=True, bidirectional=True)
        self.fc.load_state_dict({k[3:]: v for k, v in img_state.items() if k.startswith('fc.')})
        self.mlp.load_state_dict({k[4:]: v for k, v in img_state.items() if k.startswith('mlp.')})
        self.embed.load_state_dict({k[6:]: v for k, v in txt_state.items() if k.startswith('embed.')})
        self.rnn.load_state_dict({k[8:]: v for k, v in txt_state.items() if k.startswith('cap_rnn.')})
        self.embed.weight.requires_grad_(False)
        self.set_modes()

    def set_modes(self):
        self.eval(); self.rnn.train()

    def train(self, mode=True):  # keep BN in eval, GRU in train (see docstring) regardless of caller
        super().train(False)
        self.rnn.training = True
        return self

    def trainable(self):
        return [p for n, p in self.named_parameters() if not n.startswith('embed.')]

    def regions(self, images):
        """images [n,36,2048] -> [n,36,1024] = fc(images) + mlp(images) (EncoderImage before gpool)."""
        return self.fc(images) + self.mlp(images)

    def tokens(self, toks, lens):
        """toks [n,L] int64 (padded), lens [n] -> words [n,Lb,1024], mask [n,Lb]."""
        lb = int(lens.max())
        emb = self.embed(toks[:, :lb])
        packed = nn.utils.rnn.pack_padded_sequence(emb, lens.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=lb)
        out = (out[:, :, :out.size(2) // 2] + out[:, :, out.size(2) // 2:]) / 2
        mask = torch.arange(lb, device=toks.device)[None, :] < lens.to(toks.device)[:, None]
        return out, mask


def load_local_encoders(cond, device):
    c = CONDS[cond]
    ck_path = R30_RUNS / c['h_unit'] / 'model_best.pth.tar'
    ck = torch.load(ck_path, map_location='cpu', weights_only=False)
    vocab_size = ck['model_A'][1]['embed.weight'].shape[0]
    enc = nn.ModuleDict({p: LocalEncoder(ck[f'model_{p}'][0], ck[f'model_{p}'][1], vocab_size) for p in PEERS}).to(device)
    for p in PEERS:
        enc[p].set_modes()
    return enc, dict(path=str(ck_path), sha256=sha256(ck_path), epoch=int(ck['epoch']), best_rsum=float(ck['best_rsum']))


def load_raw(device):
    """raw region features (train 29000x36x2048, dev images = dev_ims[::5]) and prep token arrays."""
    tr = np.load(DATA_SRC / 'train_ims.npy', mmap_mode='r'); assert tr.shape == (N_TRAIN_IMG, 36, 2048)
    dv = np.load(DATA_SRC / 'dev_ims.npy', mmap_mode='r'); assert dv.shape == (N_DEV_CAP, 36, 2048)
    raw = dict(train_ims=torch.from_numpy(np.ascontiguousarray(tr)).to(device),
               dev_ims=torch.from_numpy(np.ascontiguousarray(dv[::5])).to(device))
    T = np.load(PREP / 'tokens.npz')
    raw['train_tok'] = torch.from_numpy(T['train_tok']).to(device); raw['train_len'] = torch.from_numpy(T['train_len']).to(device)
    raw['dev_tok'] = torch.from_numpy(T['dev_tok']).to(device); raw['dev_len'] = torch.from_numpy(T['dev_len']).to(device)
    return raw
