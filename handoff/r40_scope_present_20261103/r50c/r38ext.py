"""R38 additions, byte-identical in both code trees (r38f Flickr, r38c CC152K):
 - lr_multiplier: the package reference schedule (reference/directional_reference.py lr_multiplier, copied verbatim);
   common cosine for J/BCE/NCE: lr(t) = lr_head * [floor + (1-floor)/2 * (1 + cos(pi t / (T-1)))], t = 0..T-1 = update index.
 - state_hash_shared: the R36 init-state hash restricted to the keys of the shared head (directional extras out_t2i/affine_t2i
   excluded): equals the R36/R37C reference init hash iff the trunk/out/affine/null init is bitwise the same.
 - reference(cfg): the R36 / R37C unit used as the read-only shared-constant reference of the same parent and route (init-state
   hash and batch-index hash chain; files sha256-bound in the config).
 - dir_grads: per-direction gradients of the training loss (L = L_i2t + L_t2i, each 0.5 * mean over peers of 0.5 * list loss)
   w.r.t. each peer's head parameters, grouped (trunk/out_i2t/affine_i2t/out_t2i/affine_t2i/null): routing check and
   direction-gradient angle diagnostic (training batch; does not touch .grad)."""
from __future__ import annotations
import hashlib, math
from pathlib import Path
import numpy as np
import torch
from common import read_json, sha256, state_hash


def lr_multiplier(update: int, total: int, schedule: str, floor: float = 0.1) -> float:
    """update is zero-based, before the optimizer update; total is 8000 in R38."""
    if total < 2 or not 0 <= update < total or not 0 <= floor <= 1:
        raise ValueError('Invalid schedule arguments.')
    if schedule == 'constant': return 1.0
    if schedule == 'cosine':
        return floor + (1-floor)*.5*(1+math.cos(math.pi*update/(total-1)))
    raise ValueError(schedule)


EXTRA_KEYS = ('.out_t2i.', '.affine_t2i.')


def state_hash_shared(heads, nulls):
    h = hashlib.sha256()
    for k, v in sorted(heads.state_dict().items()):
        if any(x in '.' + k for x in EXTRA_KEYS):
            continue
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest() + (state_hash(nulls) if nulls is not None else '')


def reference(cfg):
    ref = cfg.get('reference_unit')
    if not ref:
        return None
    for k in ('result_json', 'train_history_json'):
        assert sha256(ref[k]) == ref[k + '_sha256'], f'reference file changed: {ref[k]}'
    res = read_json(ref['result_json']); hist = read_json(ref['train_history_json'])
    assert res['config']['ckpt_steps'] == cfg['ckpt_steps'] and res['config']['sampler_seed'] == cfg['sampler_seed'] and res['config']['init_seed'] == cfg['init_seed']
    return dict(unit=ref['unit_id'], init=res['phases'][0]['identity']['init_state_sha256'], chain=list(hist['batch_hash']))


def group_of(name):
    for g in ('out_t2i', 'affine_t2i'):
        if name.startswith(g + '.'):
            return g
    if name.startswith('glob.'):
        return 'trunk'
    if name.startswith('out.'):
        return 'out_i2t'
    if name.startswith('affine.'):
        return 'affine_i2t'
    return 'other'


def dir_grads(heads, nulls, peers, L_i, L_t):
    """returns {peer: {group: dict(norm_i2t, norm_t2i)}, cos_trunk, cos_all} and the flat per-parameter grads for the sum check."""
    named = []
    for p in peers:
        named += [(p, n, q) for n, q in heads[p].named_parameters()]
        if nulls is not None:
            named.append((p, 'null', nulls.v[p]))
    ps = [q for _, _, q in named]
    gi = torch.autograd.grad(L_i, ps, retain_graph=True, allow_unused=True)
    gt = torch.autograd.grad(L_t, ps, retain_graph=True, allow_unused=True)
    z = lambda g, q: torch.zeros_like(q) if g is None else g
    gi = [z(g, q) for g, q in zip(gi, ps)]; gt = [z(g, q) for g, q in zip(gt, ps)]
    out = {}
    for p in peers:
        rec = {}
        idx = [k for k, (pp, _, _) in enumerate(named) if pp == p]
        for k in idx:
            g = 'null' if named[k][1] == 'null' else group_of(named[k][1])
            r = rec.setdefault(g, dict(sq_i2t=0.0, sq_t2i=0.0, dot=0.0))
            r['sq_i2t'] += float((gi[k].double() ** 2).sum()); r['sq_t2i'] += float((gt[k].double() ** 2).sum()); r['dot'] += float((gi[k].double() * gt[k].double()).sum())
        for g, r in rec.items():
            r['norm_i2t'] = math.sqrt(r['sq_i2t']); r['norm_t2i'] = math.sqrt(r['sq_t2i'])
            r['cos'] = r['dot'] / (r['norm_i2t'] * r['norm_t2i']) if r['norm_i2t'] > 0 and r['norm_t2i'] > 0 else float('nan')
        tot = {k2: sum(r[k2] for r in rec.values()) for k2 in ('sq_i2t', 'sq_t2i', 'dot')}
        rec['all'] = dict(norm_i2t=math.sqrt(tot['sq_i2t']), norm_t2i=math.sqrt(tot['sq_t2i']),
                          cos=tot['dot'] / math.sqrt(tot['sq_i2t'] * tot['sq_t2i']) if tot['sq_i2t'] > 0 and tot['sq_t2i'] > 0 else float('nan'))
        out[p] = rec
    return out, (named, gi, gt)
