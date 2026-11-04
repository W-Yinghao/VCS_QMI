"""R37C data (port of R37 data35.py to CC152K): R37C native global caches (hash-verified), R37C prep locks, FIT candidate
lists. Only train/dev arrays are ever opened (TEST guard active in every entry). G units only: the local (pre-GPO)
encoder path of R35/R36 is not ported (load_local_encoders / load_raw refuse). Observed relation = native identity
(observed_map.npy in the cache, re-derived from the native loader); DEV clean/observed relation = identity (per_captions 1)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CONDS, PEERS, PREP, N_TRAIN_CAP, N_TRAIN_IMG, N_DEV_IMG, N_DEV_CAP, PER_CAP, sha256, read_json)
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
    """Cache of the bound parent (raw native unit vectors, R37C cache) + prep (record locks, calibration a0/b0, top-32
    tables and the B-TRUST subset, all R37C prep). DEV observed = DEV clean = identity (CC152K per_captions 1)."""
    c = CONDS[cond]
    cdir = Path(c['cache_dir'])
    man = read_json(cdir / 'CACHE_MANIFEST.json')
    if verify:
        for name, rec in man['files'].items():
            assert sha256(cdir / name) == rec['sha256'], f'cache hash mismatch {name}'
        assert man['parent']['unit'] == c['h_unit'], (man['parent']['unit'], c['h_unit'])
        assert man['native_dev']['reproduces_checkpoint_best_rsum_exactly'] and man['observed_map']['is_identity']
    d = dict(cond=cond, manifest=man, manifest_sha256=sha256(cdir / 'CACHE_MANIFEST.json'))
    obs = np.load(cdir / 'observed_map.npy').astype(np.int64)
    assert obs.shape == (N_TRAIN_CAP,) and np.array_equal(obs, np.arange(N_TRAIN_CAP)) and N_TRAIN_IMG == N_TRAIN_CAP
    d['obs_np'] = obs; d['obs'] = torch.from_numpy(obs).to(device)
    for p in PEERS:
        for split in ('fit', 'dev'):
            d[(split, 'img', p)] = torch.from_numpy(np.load(cdir / f'{split}_img_peer{p}.npy')).to(device)
            d[(split, 'cap', p)] = torch.from_numpy(np.load(cdir / f'{split}_cap_peer{p}.npy')).to(device)
    d['native_dev'] = np.load(cdir / 'native_dev_sims.npy')
    d['dev_obs'] = np.arange(N_DEV_CAP, dtype=np.int64) // PER_CAP
    d['dev_clean'] = np.arange(N_DEV_CAP, dtype=np.int64) // PER_CAP
    d['dev_s'] = {p: shard_mm(d[('dev', 'img', p)], d[('dev', 'cap', p)]) for p in PEERS}
    d['dev_tab'] = EV.dev_tables(d['native_dev'])
    if need_prep:
        pdir = Path(c['prep_dir']); pm = read_json(pdir / 'PREP_MANIFEST.json')
        if verify:
            for name in ('locks_records.npz', f"candidates_{c['prep_key']}.npz", c['trust_file']):
                assert sha256(pdir / name) == pm['files'][name]['sha256'], f'prep hash mismatch {name}'
        d['prep_manifest_sha256'] = sha256(pdir / 'PREP_MANIFEST.json')
        d['r35_prep_manifest_sha256'] = None   # R35 (Flickr) prep is not used by R37C
        d['a0b0'] = pm['calibration_init'][c['prep_key']]
        L = np.load(pdir / 'locks_records.npz')
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


# ----------------------------------------------------------------------------- local (pre-GPO) encoders: not ported

def load_local_encoders(cond, device):
    raise NotImplementedError('R37C runs G units only (no local encoder path for CC152K)')


def load_raw(device):
    raise NotImplementedError('R37C runs G units only (no raw region features loaded)')
