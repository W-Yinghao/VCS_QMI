"""R44 audit (zero training, no TEST; main-session request). Hypothesis under test: the JOINT student's vectors are u = l2norm(e - mu[batch]) with mu =
the PARENT's per-native-batch mean (128 sequential rows of the native eval layout, where row r holds image r//5 and caption r, so every image and its
own captions fall in the same batch). A student whose own output e loses item-specific signal can identify an image-caption pair through the shared
batch mean, i.e. through the data LAYOUT (DEV: true pairs; FIT: the clean record grouping k//5), not through content.
For each checkpoint (step 8000; parent = fresh student) and DEV centring variant:
  native     : the training / readout centring (parent per-native-batch means)                       [reproduces eval_step8000]
  global     : one global DEV mean per modality (no batch information)
  decoupled  : per-batch means of the parent outputs under independent random batch assignments of image rows and caption rows (layout broken)
report the two-stage DEV rSum (trainer readout_r36, frozen-H top-100 reranked by the unit's head) and the student-cosine-only DEV rSum, plus
collapse statistics of the raw student outputs e on DEV (mean cosine of random distinct pairs, mean cos(e, mu[batch]), |mu[batch]| / |e|).
Usage: python -B tools/r44_audit.py <tree> <cond> <label=unit_dir,...> (one condition per process)."""
import json, sys, time
from pathlib import Path
import numpy as np
W = Path(__file__).resolve().parents[1]
TREE, COND, ITEMS = sys.argv[1], sys.argv[2], [x.split('=', 1) for x in sys.argv[3].split(',')]
sys.path.insert(0, str(W / TREE))
import torch
import common as C; C.install_guards('trainer'); import train_unit as TU, evals as EV, r39ext as R9
from data35 import shard_mm
from heads import build_heads, NullSlots
TU.set_determinism()
dev = torch.device('cuda'); PEERS = C.PEERS
ctx = TU.Ctx(COND, dev, need_local=False); ctx.ensure_joint(); J = ctx.joint; d = ctx.d
orig_mu = dict(J.mu); orig_bi = R9.batch_index
NB = 128


def raw_dev(stud):
    out = {}
    with torch.no_grad():
        for p in PEERS:
            out[('img', p)] = J.raw_encode(stud[p], 'dev', 'img', torch.arange(C.N_DEV_IMG))   # J.ims['dev'] already holds one row per image
            out[('cap', p)] = J.raw_encode(stud[p], 'dev', 'cap', torch.arange(C.N_DEV_CAP))
    return out


def per_item_mu(e_parent, kind, perm):
    """per-item centring: rows r of the native layout permuted by perm (row -> position), batch = position // 128, mean of the parent outputs of the
    rows in each batch; an item's centring = the batch of its native row (image j: row PER_CAP*j)."""
    n_rows = C.N_DEV_CAP
    rows = np.arange(n_rows); item_of_row = rows // C.PER_CAP if kind == 'img' else rows
    pos = np.empty(n_rows, np.int64); pos[perm] = rows
    b = pos // NB; nb = b.max() + 1
    E = e_parent[torch.as_tensor(item_of_row, device=dev)].double()
    S = torch.zeros(nb, E.shape[1], dtype=torch.float64, device=dev).index_add_(0, torch.as_tensor(b, device=dev), E)
    cnt = torch.bincount(torch.as_tensor(b, device=dev), minlength=nb).double()[:, None]
    mub = (S / cnt).float()
    item_rows = np.arange(C.N_DEV_IMG) * C.PER_CAP if kind == 'img' else np.arange(C.N_DEV_CAP)
    return mub[torch.as_tensor(b[item_rows], device=dev)]


def set_variant(name, e_par):
    J.mu = dict(orig_mu); R9.batch_index = orig_bi
    if name == 'native':
        return
    R9.batch_index = lambda kind, ids: ids   # per-item centring tables for every split (FIT keeps its native values, expanded per item)
    for p in PEERS:
        for kind in ('img', 'cap'):
            nf = C.N_TRAIN_IMG if kind == 'img' else C.N_TRAIN_CAP
            J.mu[('fit', kind, p)] = orig_mu[('fit', kind, p)][orig_bi(kind, torch.arange(nf, device=dev))]
            if name == 'global':
                g = e_par[(kind, p)].double().mean(0) if kind == 'cap' else e_par[(kind, p)].double().mean(0)
                J.mu[('dev', kind, p)] = g.float()[None].expand(C.N_DEV_IMG if kind == 'img' else C.N_DEV_CAP, -1).contiguous()
            else:
                rng = np.random.default_rng(44 if kind == 'img' else 4444)
                J.mu[('dev', kind, p)] = per_item_mu(e_par[(kind, p)], kind, rng.permutation(C.N_DEV_CAP))
    J.clear_eval()


res = dict(cond=COND, tree=TREE, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), gpu=torch.cuda.get_device_name(0), units={})
par_stud = R9.build_students(J.ck, dev, ('img', 'txt')); par_stud.train()
e_par = raw_dev(par_stud)
for label, udir in [('parent_fresh_student', None)] + ITEMS:
    udir = Path(udir) if udir else None
    a0, b0 = d['a0b0']['a0'], d['a0b0']['b0']
    heads = build_heads('G', a0, b0, 1001).to(dev); nulls = NullSlots().to(dev)
    stud = R9.build_students(J.ck, dev, ('img', 'txt'))
    rec = dict(unit=udir.name if udir else 'parent')
    if udir:
        ck = torch.load(udir / 'ckpt_step8000.pt', map_location=dev, weights_only=False); assert ck['step'] == 8000
        heads.load_state_dict(ck['heads']); nulls.load_state_dict(ck['nulls']); stud.load_state_dict(ck['student'])
        rec['logged_rsum_step8000'] = json.loads((udir / 'eval_step8000.json').read_text())['dev_retrieval']['rsum']
    stud.train()
    if udir:   # parameter audit: which student tensors moved vs the parent; frozen tensors (word embedding, BN statistics) unchanged?
        rec['frozen_state_equal_parent'] = {p: R9.state_digest(stud[p].frozen_state()) == R9.state_digest(par_stud[p].frozen_state()) for p in PEERS}
        moved = {}; tot = {}
        for p in PEERS:
            ps = dict(par_stud[p].named_parameters()); bs = dict(par_stud[p].named_buffers())
            moved[p] = sorted(n for n, q in stud[p].named_parameters() if not torch.equal(q.detach(), ps[n].detach()))
            moved[p + '_buffers'] = sorted(n for n, q in stud[p].named_buffers() if not torch.equal(q, bs[n]))
            tot[p] = dict(trainable=sorted(n for n, q in stud[p].named_parameters() if q.requires_grad), not_trainable_moved=sorted(n for n, q in stud[p].named_parameters() if not q.requires_grad and not torch.equal(q.detach(), ps[n].detach())))
        rec['moved_tensors'] = {k: len(v) for k, v in moved.items()}; rec['moved_buffers'] = {k: v for k, v in moved.items() if k.endswith('_buffers')}
        rec['non_trainable_moved'] = {p: tot[p]['not_trainable_moved'] for p in PEERS}; rec['n_trainable'] = {p: len(tot[p]['trainable']) for p in PEERS}
        rec['param_distance_vs_parent'] = J.param_distance(stud, {p: {n: q.detach().clone() for n, q in par_stud[p].named_parameters()} for p in PEERS})
    e = raw_dev(stud)
    with torch.no_grad():
        col = {}
        for p in PEERS:
            for kind in ('img', 'cap'):
                x = e[(kind, p)]; n = len(x); g = torch.Generator(device='cpu').manual_seed(7)
                i1 = torch.randint(0, n, (20000,), generator=g); i2 = torch.randint(0, n, (20000,), generator=g); keep = i1 != i2
                xn = x / x.norm(dim=1, keepdim=True)
                mu = orig_mu[('dev', kind, p)][orig_bi(kind, torch.arange(n, device=dev))]
                col[f'{kind}_{p}'] = dict(mean_cos_random_pairs=float((xn[i1[keep]] * xn[i2[keep]]).sum(1).mean()), mean_norm_e=float(x.norm(dim=1).mean()),
                                          mean_norm_mu=float(mu.norm(dim=1).mean()), mean_cos_e_mu=float(((x * mu).sum(1) / (x.norm(dim=1) * mu.norm(dim=1))).mean()),
                                          mean_norm_e_minus_mu=float((x - mu).norm(dim=1).mean()))
        rec['collapse'] = col
    for var in ('native', 'global', 'decoupled'):
        set_variant(var, e_par)
        r_, _, _ = TU.readout_r36(heads, nulls, None, ctx, final=False, stud=stud)
        sims = sum(shard_mm(J.table(stud[p], p, 'dev', 'img'), J.table(stud[p], p, 'dev', 'cap')) for p in PEERS).cpu().numpy() / 2
        rec[var] = dict(two_stage_rsum=r_['dev_retrieval']['rsum'], two_stage=r_['dev_retrieval'], student_cosine_rsum=EV.full_metrics(sims)[0]['rsum'])
        print(label, var, json.dumps(rec[var]['two_stage_rsum']), rec[var]['student_cosine_rsum'], flush=True)
    set_variant('native', e_par)
    res['units'][label] = rec
    print('R44_AUDIT', label, json.dumps({k: (v if not isinstance(v, dict) else v.get('two_stage_rsum', '')) for k, v in rec.items() if k != 'collapse'}), json.dumps(rec['collapse']), flush=True)
out = W / f'R44_AUDIT_{COND}.json'; out.write_text(json.dumps(res, indent=1, default=float)); print('wrote', out)
