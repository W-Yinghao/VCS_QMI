"""Corrected (layout-free) DEV readouts of completed critic units, one group (tree, cond, GPU type) per process; zero training, no TEST, evaluator role.
Background (R44_AUDIT.md): the native IN2R embeddings are centred by the mean of 128 SEQUENTIAL rows of the native eval layout (row r = image r//5 and
caption r), so an image and its own captions share the centring vector. This holds for the frozen-H cache (all frozen-encoder units: x, y, s_H and the
stage-1 top-100) and for the JOINT student (fixed parent per-native-batch means).
Variants (DEV two-stage = top-100 of the stage-1 score reranked by the unit's peer-mean g, rest in stage-1 order; the trainer's g / composite metric code):
  native          logged readout path (must reproduce the logged step-final rSum; difference recorded)
  student_global  JOINT only: student centring = one global DEV mean per modality; frozen-H cache native
  full_global     parent cache vectors, s_H and stage-1 recomputed from the parent outputs with one global DEV mean per modality; JOINT student as student_global
  full_decoupled  as full_global with per-batch means over independent random batch assignments of image rows and caption rows (layout broken)
plus, per condition, the parent H rSum under each centring, and for JOINT units the student-cosine-only rSum (native / global).
Usage: python -B tools/corrected_readouts.py <group.json>"""
import json, sys, time, traceback
from pathlib import Path
import numpy as np
W = Path(__file__).resolve().parents[1]
G = json.loads(Path(sys.argv[1]).read_text()); TREE, COND = G[0]['tree'], G[0]['cond']
sys.path.insert(0, str(W / TREE))
import torch
import common as C; C.install_guards('audit'); import train_unit as TU, evals as EV, r39ext as R9
from data35 import shard_mm
from heads import build_heads, NullSlots
TU.set_determinism()
dev = torch.device('cuda'); PEERS = C.PEERS; NB = 128
ctx = TU.Ctx(COND, dev, need_local=False); ctx.ensure_joint(); J = ctx.joint; d = ctx.d
orig_mu = dict(J.mu); orig_bi = R9.batch_index
l2 = lambda x: x / x.norm(dim=1, keepdim=True)
NI, NC = C.N_DEV_IMG, C.N_DEV_CAP


def raw_dev(stud):
    with torch.no_grad():
        return {(k, p): J.raw_encode(stud[p], 'dev', k, torch.arange(NI if k == 'img' else NC)) for p in PEERS for k in ('img', 'cap')}


def item_mu(e, kind, variant):
    n_items = NI if kind == 'img' else NC
    if variant == 'global':
        return e.double().mean(0).float()[None].expand(n_items, -1).contiguous() if kind == 'cap' else e.double().mean(0).float()[None].expand(n_items, -1).contiguous()
    rows = np.arange(NC); item_of_row = rows // C.PER_CAP if kind == 'img' else rows
    perm = np.random.default_rng(44 if kind == 'img' else 4444).permutation(NC); pos = np.empty(NC, np.int64); pos[perm] = rows; b = pos // NB; nb = b.max() + 1
    E = e[torch.as_tensor(item_of_row, device=dev)].double()
    S = torch.zeros(nb, E.shape[1], dtype=torch.float64, device=dev).index_add_(0, torch.as_tensor(b, device=dev), E)
    mub = (S / torch.bincount(torch.as_tensor(b, device=dev), minlength=nb).double()[:, None]).float()
    item_rows = np.arange(NI) * C.PER_CAP if kind == 'img' else np.arange(NC)
    return mub[torch.as_tensor(b[item_rows], device=dev)]


def set_student_centring(variant, e_par):
    J.mu = dict(orig_mu); R9.batch_index = orig_bi; J.clear_eval()
    if variant == 'native':
        return
    R9.batch_index = lambda kind, ids: ids
    for p in PEERS:
        for kind in ('img', 'cap'):
            nf = C.N_TRAIN_IMG if kind == 'img' else C.N_TRAIN_CAP
            J.mu[('fit', kind, p)] = orig_mu[('fit', kind, p)][orig_bi(kind, torch.arange(nf, device=dev))]
            J.mu[('dev', kind, p)] = item_mu(e_par[(kind, p)], kind, variant)
    J.clear_eval()


par = R9.build_students(J.ck, dev, ('img', 'txt')); par.train(); e_par = raw_dev(par)
DV = {'native': d}
for v in ('global', 'decoupled'):
    d2 = dict(d)
    for p in PEERS:
        for k in ('img', 'cap'):
            d2[('dev', k, p)] = l2(e_par[(k, p)] - item_mu(e_par[(k, p)], k, v))
    d2['dev_s'] = {p: shard_mm(d2[('dev', 'img', p)], d2[('dev', 'cap', p)]) for p in PEERS}
    d2['native_dev'] = (sum(d2['dev_s'][p] for p in PEERS) / 2).cpu().numpy(); d2['dev_tab'] = EV.dev_tables(d2['native_dev'])
    DV[v] = d2
out = dict(group=Path(sys.argv[1]).stem, tree=TREE, cond=COND, gpu=torch.cuda.get_device_name(0), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           parent_H={v: EV.full_metrics(DV[v]['native_dev'])[0]['rsum'] for v in DV}, units={})
print('PARENT_H', COND, json.dumps(out['parent_H']), flush=True)


def two_stage(heads, stud, dd):
    tab = dd['dev_tab']; i_top, t_top = tab['i_top'], tab['t_top']
    img = np.concatenate([np.repeat(np.arange(NI), i_top.shape[1]), t_top.ravel()]); cap = np.concatenate([i_top.ravel(), np.repeat(np.arange(NC), t_top.shape[1])])
    dirs = np.concatenate([np.zeros(i_top.size, np.int64), np.ones(t_top.size, np.int64)])
    cx = TU._shallow(ctx, d=dd); gs = 0
    with torch.no_grad():
        for p in PEERS:
            s_ex = dd['dev_s'][p][torch.as_tensor(img, device=dev), torch.as_tensor(cap, device=dev)].cpu().numpy()
            stu = J.pairs(stud[p], p, 'dev', img, cap) if stud is not None else None
            gs = gs + TU.g_pairs_lists(heads[p], None, cx, p, 'dev', img, cap, s_exact=s_ex, direction=dirs, stu=stu)
    g = (gs / 2).cpu().numpy(); gi = g[:i_top.size].reshape(NI, -1); gt = g[i_top.size:].reshape(NC, -1)
    return EV.composite_metrics(dd['native_dev'], tab, gi, gt)[0]['rsum']


def stud_cos(stud):
    sims = sum(shard_mm(J.table(stud[p], p, 'dev', 'img'), J.table(stud[p], p, 'dev', 'cap')) for p in PEERS).cpu().numpy() / 2
    return EV.full_metrics(sims)[0]['rsum']


for e in G:
    t0 = time.time(); rec = dict(e)
    try:
        cfg = json.loads(Path(e['config']).read_text()); ck = torch.load(e['ckpt'], map_location=dev, weights_only=False); assert ck['step'] == e['steps']
        heads = build_heads(e['structure'], d['a0b0']['a0'], d['a0b0']['b0'], e['init_seed'] or 1001).to(dev); heads.load_state_dict(ck['heads']); heads.eval()
        stud = None
        if e['encoder'] == 'joint':
            stud = R9.build_students(J.ck, dev, C.ADAPTATIONS[cfg.get('adaptation') or 'both_joint'] if hasattr(C, 'ADAPTATIONS') else ('img', 'txt'))
            stud.load_state_dict(ck['student']); stud.train()
        set_student_centring('native', e_par)
        rec['native'] = two_stage(heads, stud, DV['native']); rec['native_minus_logged'] = rec['native'] - e['logged_rsum']
        if stud is not None:
            rec['student_cos_native'] = stud_cos(stud)
            set_student_centring('global', e_par); rec['student_global'] = two_stage(heads, stud, DV['native']); rec['student_cos_global'] = stud_cos(stud)
        set_student_centring('global', e_par); rec['full_global'] = two_stage(heads, stud, DV['global'])
        set_student_centring('decoupled', e_par); rec['full_decoupled'] = two_stage(heads, stud, DV['decoupled'])
        set_student_centring('native', e_par)
        rec['H_native'], rec['H_global'], rec['H_decoupled'] = out['parent_H']['native'], out['parent_H']['global'], out['parent_H']['decoupled']
        rec['status'] = 'EVALUATED'
    except Exception:
        rec['status'] = 'ERROR'; rec['error'] = traceback.format_exc()[-1500:]
    rec['seconds'] = time.time() - t0; out['units'][e['unit']] = rec
    print('UNIT', e['unit'], rec['status'], json.dumps({k: rec.get(k) for k in ('logged_rsum', 'native', 'student_global', 'full_global', 'full_decoupled', 'student_cos_native', 'student_cos_global')}), flush=True)
    del ck
od = W / 'corrected' / 'results'; od.mkdir(parents=True, exist_ok=True)
(od / f"{out['group']}.json").write_text(json.dumps(out, indent=1, default=float)); print('wrote', od / f"{out['group']}.json")
