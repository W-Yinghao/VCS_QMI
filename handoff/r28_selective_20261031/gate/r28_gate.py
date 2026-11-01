"""R28 real-data GPU gate (modeled on the R27-Q gate gate/r27q_gate.py). Run from code_r28 (cwd).
Usage: r28_gate.py SMOKE_DIR NOISE_FILE DATA_PATH GATE_STATE CODE_DIR -> SMOKE_DIR/GPU_GATE.json

From one restored real R27-S smoke state (both queues ID-bearing): one real training step of model A per arm/route (stage
fit included), with a probe taking per-direction gradients (torch.autograd.grad, retain_graph; production backward
untouched). References run in this snapshot: a1_iface, cn_mix_t2i (C-native), q1_cos_pool (Q1), q2_vcs_pool (Q2).
R28 arms g1/g2/r1/r2 normal and with the gate-only routes r28_select.GATE_U in {'zero','one'} and (R arms)
GATE_RESIDUAL='zero'. Checks:
  u0_equiv      every R28 arm with u = 0 == C-native bitwise (encoder/refiner grads, params after the step, AdamW state,
                RNG, per-direction probe grads, prototypes, graphs) although the stage fit + calibration ran
  u1_equiv      u = 1: g1 == Q1, g2 == Q2 bitwise (incl. the same stage head); r1/r2 with zero residual and u = 1 == Q1
  zres_equiv    r1/r2 with zero residual (normal u) == g1 bitwise (zero residual recovers the cosine readout, one update)
  shared_u      u identical across the four arms (same calibrator, same teacher), calibrator status, u non-constant
  i2t_identical normal R28 arms: i2t == A1 bitwise; t2i candidate graph == C-native bitwise
  mechanics     alpha / u stop-grad, rows sum 1, u in [0,1], passed to the t2i refiner only; head / residual / calibrator not
                in the optimizer, no grad, unchanged by the step; other peer no grad, queue/IDs unchanged; one optimizer
                step; one refiner forward per direction; t2i prototype differs from C-native; residual active and non-zero
  calibration   calibration rows = high-confidence labeled rows outside the head fit/held-out rows, observed images of the
                fit/report splits disjoint, own-record exclusion == production C_mix on a queue without the own slots,
                weak-label class counts non-degenerate; anchor (a, b) reproducible and stationary; evaluator-only counts
  pq_truth_free stage P recomputed independently; static truth check on r28_select / r28_config; hook lines
  resume        snapshot restore twice bitwise (stage fit + calibration incl.); disk reload; smoke state arms; runner identity
  timing        ms/step (30 iterations incl. the stage fit) and peak memory per arm
  logs          smoke arms: finite losses, rstage_* and t2i_r28_* logs per peer stage
"""
import os, sys, json, copy, hashlib, time, inspect, subprocess, tempfile, shutil, types
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(sys.argv[5]))
import r25_run
import r25_co_train as R
import r27_co_train as R27
import r27_interface as I
import r27cs_sources as CS
import r27q_pool as QP
import r27q_config as QC
import r28_select as SEL
import r28_config as C
from data import get_dataset
from vse import VSE
from vocab import deserialize_vocab
from co_train import eval_train, split_prob

res = {}
RA = list(SEL.R28_ARMS)
RES_ARMS = [a for a in RA if SEL.R28_ARMS[a] in SEL.RES_KINDS]
TOL = 1e-6
K = 5
HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(os.path.dirname(HERE), 'tools', 'run_unit_r28.py')
CELLS = '/home/infres/yinwang/CS_QMI/handoff/r27_qtarget_20261029/CELLS.json'
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'tests'))


def ok(name, cond, **info):
    res[name] = dict(pass_=bool(cond), **info)
    print(name, 'PASS' if cond else 'FAIL', json.dumps(info, default=float)[:4000], flush=True)


def h(t):
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()


def fsha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def gdiff(a, b):
    m = 0.0
    for x, y in zip(a, b):
        if (x is None) != (y is None):
            return float('inf')
        if x is not None:
            m = max(m, float((x - y).abs().max()))
    return m


def main():
    smoke, noise, dpath, S0, code_dir = sys.argv[1:6]
    manifest = json.load(open(os.path.join(code_dir, 'INSTALL_MANIFEST.json')))
    code_now = {f: fsha(os.path.join(code_dir, f)) for f in manifest['files_sha256']}
    opt = r25_run.parse(['--noise_file', noise, '--output_dir', f'{smoke}/gate', '--resume', S0, '--data_path', dpath, '--max_iters', '30'])
    opt.arm, opt.lam = 'a1_iface', 0.0
    os.makedirs(opt.output_dir, exist_ok=True)
    R27.install()
    r25_run.seed_all(opt.seed)
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, "%s_vocab.json" % opt.data_name)); vocab.add_word('<mask>')
    opt.vocab_size = len(vocab)
    caps, ims = get_dataset(opt.data_path, opt.data_name, "train")
    noisy_loader, data_size, clean_labels = R.get_loader(caps, ims, vocab, "warmup", opt.batch_size, 0, opt.noise_ratio, opt.noise_file)
    A = VSE(opt, vocab.word2idx); B = VSE(opt, vocab.word2idx)
    st = R.load_full_state(S0, opt, A, B)
    ep = st['epoch'] + 1
    prob_A, prob_B, _ = eval_train(opt, A, B, noisy_loader, data_size, copy.deepcopy(st['all_loss']), clean_labels, ep)
    pred_A = split_prob(prob_A, opt.p_threshold); pred_B = split_prob(prob_B, opt.p_threshold)
    for m, prob, pred in ((A, prob_A, pred_A), (B, prob_B, pred_B)):
        m.avg_clean_prob = prob[pred].mean()
    lt, ut = R.get_loader(caps, ims, vocab, "train", opt.batch_size, 0, opt.noise_ratio, opt.noise_file, pred=pred_B, prob=prob_B)
    snap = copy.deepcopy(R.model_full_state(A)), copy.deepcopy(R.model_full_state(B)), R.rng_state()
    NOISE = np.load(noise)

    def restore():
        for m, s in ((A, snap[0]), (B, snap[1])):
            m.img_enc.load_state_dict(s['img_enc']); m.txt_enc.load_state_dict(s['txt_enc'])
            m.optimizer.load_state_dict(copy.deepcopy(s['optimizer']))
            m.i2t_refiner_node.load_state_dict(s['i2t_refiner']); m.t2i_refiner_node.load_state_dict(s['t2i_refiner'])
            q = m.memory_queue; q.img_queue.copy_(s['queue']['img']); q.txt_queue.copy_(s['queue']['txt']); q.ptr, q.is_full = s['queue']['ptr'], s['queue']['is_full']
            I.load_ids(q, s['r27_queue_ids'])
            for p in list(m.params) + list(m.i2t_refiner_node.parameters()) + list(m.t2i_refiner_node.parameters()):
                p.grad = None
        R.set_rng_state(snap[2])
        R27._PEER.clear(); R27._PEER[id(A)] = 0; R27._PEER[id(B)] = 1

    own_ref = lambda: list(A.i2t_refiner_node.parameters()) + list(A.t2i_refiner_node.parameters())
    peer_all = lambda: list(B.params) + list(B.i2t_refiner_node.parameters()) + list(B.t2i_refiner_node.parameters())
    ids_of = lambda q: {k: v.clone() for k, v in getattr(q, 'r27_ids', {}).items()}

    # recorders (gate only)
    al_rec, ref_rec = [], []
    orig_alpha_q, orig_alpha_r = QP.alpha_t2i, SEL.alpha_u_t2i

    def rec_q(opt_, net, net2, S, uids, stats, timed=False):
        a = orig_alpha_q(opt_, net, net2, S, uids, stats, timed)
        hd = QP.STAGE.get('head')
        al_rec.append(dict(src='Q', alpha=None if a is None else a.detach().clone(), u=None, req=None if a is None else bool(a.requires_grad),
                           ureq=None, head=hd, head_sha=None if hd is None else QP.head_sha(hd), teacher_is_B=net2 is B))
        return a

    def rec_r(opt_, net, net2, S, uids, stats, timed=False):
        a, u = orig_alpha_r(opt_, net, net2, S, uids, stats, timed)
        st_ = SEL.STAGE
        hd = st_.get('head')
        hm = None if hd is None else (hd.res if st_['kind'] in SEL.RES_KINDS else hd)
        al_rec.append(dict(src='R', alpha=a.detach().clone(), u=u.detach().clone(), req=bool(a.requires_grad), ureq=bool(u.requires_grad),
                           head=hm, head_sha=None if hm is None else QP.head_sha(hm), teacher_is_B=net2 is B,
                           w=st_['w'].detach().clone(), c=st_['c'], a=st_.get('a'), b=st_.get('b'), res_active=st_.get('res_active'),
                           cal_status=st_.get('cal_status'), log=dict(st_.get('log', {}))))
        return a, u
    QP.alpha_t2i = rec_q
    SEL.alpha_u_t2i = rec_r

    def hook(name):
        def f(mod, args, kwargs, out):
            al = kwargs.get('alpha', args[1] if len(args) > 1 else None)
            uu = kwargs.get('u', args[2] if len(args) > 2 else None)
            ref_rec.append(dict(name=name, alpha=None if al is None else al.detach().clone(), u=None if uu is None else uu.detach().clone(),
                                out=out.detach().clone()))
        return f
    hooks = [A.i2t_refiner_node.register_forward_hook(hook('i2t'), with_kwargs=True),
             A.t2i_refiner_node.register_forward_hook(hook('t2i'), with_kwargs=True)]

    def one_step(arm, gate_u=None, gate_res=None, probe_grads=True, do_restore=True):
        if do_restore:
            restore()
        opt.arm, opt.lam = arm, 0.0
        SEL.GATE_U, SEL.GATE_RESIDUAL = gate_u, gate_res
        cap, br = {'nstep': 0}, {}
        al_rec.clear(); ref_rec.clear()
        params = list(A.params) + own_ref()

        def probe(branch, dd):
            e = dict(loss=float(dd['loss']), l_nat=float(dd['l_nat']), graphA=dd['S']['A'].detach().clone(),
                     cols=dd['S']['cols'].detach().clone(), img=dd['S']['img'].detach().clone(), inv=dd['S']['inv'].detach().clone(),
                     proto=dd['proto'].detach().clone(), Y_req=bool(dd['Y'].requires_grad), neigh_req=bool(dd['neigh'].requires_grad),
                     X_req=bool(dd['X'].requires_grad), finite=bool(torch.isfinite(dd['loss'])), aux_none=dd['l_aux'] is None)
            if probe_grads:
                g = torch.autograd.grad(dd['loss'], params, retain_graph=True, allow_unused=True)
                e['grads'] = [None if x is None else x.detach().clone() for x in g]
            br[branch] = e
        orig = A.optimizer.step

        def cur_heads():
            out = []
            for stg in (QP.STAGE, SEL.STAGE):
                hd = stg.get('head') if stg else None
                if hd is not None:
                    out.append(hd)
            return out

        def step_probe(*a, **k):
            cap['nstep'] += 1
            cap['enc'] = [None if p.grad is None else p.grad.detach().clone() for p in A.params]
            cap['ref_i2t'] = [None if p.grad is None else p.grad.detach().clone() for p in A.i2t_refiner_node.parameters()]
            cap['ref_t2i'] = [None if p.grad is None else p.grad.detach().clone() for p in A.t2i_refiner_node.parameters()]
            cap['peer_with_grad'] = sum(1 for p in peer_all() if p.grad is not None)
            hds = cur_heads() if arm in RA or arm in QP.Q_ARMS else []
            hd = [x for x in hds if (arm in RA and x is SEL.STAGE.get('head')) or (arm in QP.Q_ARMS and x is QP.STAGE.get('head'))]
            hd = hd[0] if hd else None
            cap['head_sha_before_step'] = None if hd is None else QP.head_sha(hd)
            cap['head_with_grad'] = 0 if hd is None else sum(1 for p in hd.parameters() if p.grad is not None)
            cap['head_requires_grad'] = 0 if hd is None else sum(1 for p in hd.parameters() if p.requires_grad)
            opt_ids = {id(p) for g in A.optimizer.param_groups for p in g['params']}
            cap['head_in_optimizer'] = 0 if hd is None else sum(1 for p in hd.parameters() if id(p) in opt_ids)
            wt = SEL.STAGE.get('w') if arm in RA else None
            cap['calib_w_requires_grad'] = None if wt is None else bool(wt.requires_grad)
            cap['calib_w_in_optimizer'] = None if wt is None else id(wt) in opt_ids
            out = orig(*a, **k)
            cap['head_sha_after_step'] = None if hd is None else QP.head_sha(hd)
            cap['params'] = torch.cat([p.detach().flatten().double() for p in params])
            cap['adam'] = torch.cat([v.detach().flatten().double() for p in params for kk, v in A.optimizer.state.get(p, {}).items() if torch.is_tensor(v) and v.dim() > 0])
            cap['adam_step'] = [float(v) for p in params for kk, v in A.optimizer.state.get(p, {}).items() if kk == 'step']
            return out
        A.optimizer.step = step_probe
        qh = (h(B.memory_queue.img_queue), h(B.memory_queue.txt_queue), {k: h(v) for k, v in ids_of(B.memory_queue).items()})
        bfp = QP.fingerprint(B)
        stats = R.new_stats()
        try:
            R27.train(opt, A, B, lt, ut, ep, stats, max_iters=1, probe=probe)
        finally:
            A.optimizer.step = orig
            SEL.GATE_U, SEL.GATE_RESIDUAL = None, None
        cap.update(rng=R.rng_state(), stats=dict(stats), br=br, al=list(al_rec), refs=list(ref_rec),
                   peer_queue_same=(h(B.memory_queue.img_queue), h(B.memory_queue.txt_queue), {k: h(v) for k, v in ids_of(B.memory_queue).items()}) == qh,
                   peer_params_same=QP.fingerprint(B) == bfp,
                   own_ids={k: v.clone() for k, v in ids_of(A.memory_queue).items()},
                   sel_stage=dict(SEL.STAGE) if arm in RA else None)
        return cap

    def rng_eq(a, b):
        na, nb_ = a['rng']['numpy'], b['rng']['numpy']
        return (bool(torch.equal(a['rng']['torch'], b['rng']['torch'])) and all(torch.equal(x, y) for x, y in zip(a['rng']['cuda'], b['rng']['cuda']))
                and na[0] == nb_[0] and np.array_equal(na[1], nb_[1]) and tuple(na[2:]) == tuple(nb_[2:]) and a['rng']['python'] == b['rng']['python'])

    def full_cmp(a, b):
        e = dict(enc_grad_maxdiff=gdiff(a['enc'], b['enc']), ref_i2t_grad_maxdiff=gdiff(a['ref_i2t'], b['ref_i2t']),
                 ref_t2i_grad_maxdiff=gdiff(a['ref_t2i'], b['ref_t2i']),
                 param_maxdiff=float((a['params'] - b['params']).abs().max()), adam_maxdiff=float((a['adam'] - b['adam']).abs().max()),
                 adam_len_equal=a['adam'].numel() == b['adam'].numel(), adam_step_equal=a['adam_step'] == b['adam_step'], rng_equal=rng_eq(a, b),
                 dir_grad_maxdiff={d: gdiff(a['br'][d]['grads'], b['br'][d]['grads']) for d in ('i2t', 't2i')},
                 dir_lnat_absdiff={d: abs(a['br'][d]['l_nat'] - b['br'][d]['l_nat']) for d in ('i2t', 't2i')},
                 proto_maxdiff={d: float((a['br'][d]['proto'] - b['br'][d]['proto']).abs().max()) for d in ('i2t', 't2i')},
                 graph_equal={d: bool(torch.equal(a['br'][d]['graphA'], b['br'][d]['graphA']) and torch.equal(a['br'][d]['cols'], b['br'][d]['cols'])
                                      and torch.equal(a['br'][d]['inv'], b['br'][d]['inv'])) for d in ('i2t', 't2i')})
        e['bitwise'] = (e['enc_grad_maxdiff'] == 0 and e['ref_i2t_grad_maxdiff'] == 0 and e['ref_t2i_grad_maxdiff'] == 0
                        and e['param_maxdiff'] == 0 and e['adam_maxdiff'] == 0 and e['adam_len_equal'] and e['adam_step_equal'] and e['rng_equal']
                        and all(v == 0 for v in e['dir_grad_maxdiff'].values()) and all(v == 0 for v in e['dir_lnat_absdiff'].values())
                        and all(v == 0 for v in e['proto_maxdiff'].values()) and all(e['graph_equal'].values()))
        return e

    steps = {a: one_step(a) for a in ('a1_iface', 'cn_mix_t2i', 'q1_cos_pool', 'q2_vcs_pool')}
    for arm in RA:
        steps[arm] = one_step(arm)
        steps[arm + '/u0'] = one_step(arm, gate_u='zero')
        steps[arm + '/u1'] = one_step(arm, gate_u='one')
    for arm in RES_ARMS:
        steps[arm + '/zres'] = one_step(arm, gate_res='zero')
        steps[arm + '/zres_u1'] = one_step(arm, gate_u='one', gate_res='zero')
    cn = steps['cn_mix_t2i']
    slim = lambda e: {k: v for k, v in e.items() if k != 'bitwise'} | {'bitwise': e['bitwise']}

    # ---- (1) u = 0 == C-native
    r0, good = {}, True
    for arm in RA:
        a = steps[arm + '/u0']
        e = full_cmp(a, cn)
        t2r = [r for r in a['refs'] if r['name'] == 't2i']
        e.update(stage_fitted=bool(a['stats'].get('rstage_version')), rows=a['stats'].get('cnt_t2i_r28_rows', 0),
                 refiner_got_u_zero=bool(t2r) and t2r[0]['u'] is not None and bool((t2r[0]['u'] == 0).all()))
        e['pass'] = e['bitwise'] and e['stage_fitted'] and e['rows'] > 0 and e['refiner_got_u_zero']
        good &= e['pass']; r0[arm] = slim(e)
    ok('u0_equiv', good, **r0)

    # ---- (2) u = 1 == R27-Q references
    r1, good = {}, True
    pairs = [('g1_cos_sel/u1', 'q1_cos_pool'), ('g2_vcs_sel/u1', 'q2_vcs_pool')] + [(a + '/zres_u1', 'q1_cos_pool') for a in RES_ARMS]
    for x, y in pairs:
        e = full_cmp(steps[x], steps[y])
        ax, ay = steps[x]['al'][0], steps[y]['al'][0]
        e['alpha_equal'] = bool(torch.equal(ax['alpha'], ay['alpha']))
        e['refiner_got_u_one'] = all(bool((r['u'] == 1).all()) for r in steps[x]['refs'] if r['name'] == 't2i')
        if y == 'q2_vcs_pool':
            e['same_head_as_q2'] = ax['head_sha'] == ay['head_sha'] and ax['head_sha'] is not None
        e['pass'] = e['bitwise'] and e['alpha_equal'] and e['refiner_got_u_one'] and e.get('same_head_as_q2', True)
        good &= e['pass']; r1[f'{x}=={y}'] = slim(e)
    ok('u1_equiv', good, **r1)

    # ---- (3) zero residual == G1 (normal u)
    r2, good = {}, True
    for arm in RES_ARMS:
        e = full_cmp(steps[arm + '/zres'], steps['g1_cos_sel'])
        ax, ay = steps[arm + '/zres']['al'][0], steps['g1_cos_sel']['al'][0]
        e.update(alpha_equal=bool(torch.equal(ax['alpha'], ay['alpha'])), u_equal=bool(torch.equal(ax['u'], ay['u'])),
                 residual_was_fitted=bool(ax['res_active']))
        e['pass'] = e['bitwise'] and e['alpha_equal'] and e['u_equal'] and e['residual_was_fitted']
        good &= e['pass']; r2[arm] = slim(e)
    ok('zres_equiv', good, **r2)

    # ---- (4) shared u
    us = {arm: steps[arm]['al'][0] for arm in RA}
    u0 = us['g1_cos_sel']
    lg = u0['log']
    e = dict(u_equal_all=all(torch.equal(v['u'], u0['u']) for v in us.values()),
             w_equal_all=all(torch.equal(v['w'], u0['w']) and v['c'] == u0['c'] for v in us.values()),
             cal_status=u0['cal_status'], u_mean=float(u0['u'].mean()), u_std=float(u0['u'].std()), u_min=float(u0['u'].min()),
             u_max=float(u0['u'].max()), weights=u0['w'].tolist(), intercept=u0['c'],
             report={k: v for k, v in lg.items() if k.startswith('rstage_cal')})
    e['pass'] = e['u_equal_all'] and e['w_equal_all'] and e['cal_status'] == 'ok' and e['u_std'] > 0
    ok('shared_u', e['pass'], **e)

    # ---- (5) i2t identical to A1, t2i candidate graph == C-native
    r3, good = {}, True
    for arm in RA:
        a, b = steps[arm], steps['a1_iface']
        ai, bi = a['br']['i2t'], b['br']['i2t']
        e = dict(grad_maxdiff=gdiff(ai['grads'], bi['grads']), l_nat_absdiff=abs(ai['l_nat'] - bi['l_nat']),
                 proto_equal=bool(torch.equal(ai['proto'], bi['proto'])),
                 graph_equal=bool(torch.equal(ai['graphA'], bi['graphA']) and torch.equal(ai['cols'], bi['cols']) and torch.equal(ai['img'], bi['img'])),
                 step_ref_i2t_grad_maxdiff=gdiff(a['ref_i2t'], b['ref_i2t']),
                 t2i_graph_equal_cnative=bool(torch.equal(a['br']['t2i']['img'], cn['br']['t2i']['img']) and torch.equal(a['br']['t2i']['graphA'], cn['br']['t2i']['graphA'])
                                              and torch.equal(a['br']['t2i']['inv'], cn['br']['t2i']['inv'])),
                 cs_rows=a['stats'].get('cnt_t2i_cs_rows', 0))
        e['pass'] = (e['grad_maxdiff'] == 0 and e['l_nat_absdiff'] == 0 and e['proto_equal'] and e['graph_equal']
                     and e['step_ref_i2t_grad_maxdiff'] == 0 and e['t2i_graph_equal_cnative'] and e['cs_rows'] > 0)
        good &= e['pass']; r3[arm] = e
    ok('i2t_identical', good, **r3)

    # ---- (6) mechanics
    r4, good = {}, True
    nz = lambda L: sum(1 for g in L if g is not None and float(g.abs().sum()) > 0)
    for arm in ['a1_iface', 'cn_mix_t2i'] + RA:
        c = steps[arm]; s_ = c['stats']
        e = dict(i2t_n=s_.get('i2t_n', 0), t2i_n=s_.get('t2i_n', 0), enc_grad_nonzero=nz(c['enc']), ref_i2t_with_grad=nz(c['ref_i2t']),
                 ref_t2i_with_grad=nz(c['ref_t2i']), peer_params_with_grad=c['peer_with_grad'], peer_queue_ids_same=c['peer_queue_same'],
                 peer_params_same=c['peer_params_same'], optimizer_steps=c['nstep'], refiner_forwards=[r['name'] for r in c['refs']],
                 stopgrad_graph=all((not v['Y_req']) and (not v['neigh_req']) and v['X_req'] and v['finite'] and v['aux_none'] for v in c['br'].values()))
        base = (e['i2t_n'] == 1 and e['t2i_n'] == 1 and e['enc_grad_nonzero'] > 0 and e['ref_i2t_with_grad'] > 0 and e['ref_t2i_with_grad'] > 0
                and e['peer_params_with_grad'] == 0 and e['peer_queue_ids_same'] and e['peer_params_same'] and e['optimizer_steps'] == 1
                and e['refiner_forwards'] == ['i2t', 't2i'] and e['stopgrad_graph'])
        if arm in RA:
            r_ = c['al'][0]; al, u = r_['alpha'], r_['u']
            t2r = [r for r in c['refs'] if r['name'] == 't2i'][0]; i2r = [r for r in c['refs'] if r['name'] == 'i2t'][0]
            kind = SEL.R28_ARMS[arm]
            e.update(alpha_calls=len(c['al']), alpha_requires_grad=r_['req'], u_requires_grad=r_['ureq'], alpha_shape=list(al.shape),
                     alpha_rowsum_maxdev=float((al.sum(1) - 1).abs().max()), u_in_unit=bool(((u >= 0) & (u <= 1)).all()),
                     u_shape=list(u.shape), alpha_row_range_mean=float((al.max(1).values - al.min(1).values).mean()),
                     t2i_refiner_got_alpha_u=t2r['alpha'] is not None and bool(torch.equal(t2r['alpha'], al)) and t2r['u'] is not None and bool(torch.equal(t2r['u'], u)),
                     i2t_refiner_plain=i2r['alpha'] is None and i2r['u'] is None, teacher_is_other_peer=r_['teacher_is_B'],
                     t2i_proto_differs_from_cnative=not bool(torch.equal(c['br']['t2i']['proto'], cn['br']['t2i']['proto'])),
                     head_in_optimizer=c['head_in_optimizer'], head_with_grad=c['head_with_grad'], head_requires_grad=c['head_requires_grad'],
                     head_unchanged_by_step=c['head_sha_before_step'] == c['head_sha_after_step'],
                     head_present=c['head_sha_before_step'] is not None, calib_w_requires_grad=c['calib_w_requires_grad'],
                     calib_w_in_optimizer=c['calib_w_in_optimizer'])
            ok_r = (e['alpha_calls'] == 1 and e['alpha_requires_grad'] is False and e['u_requires_grad'] is False and e['alpha_shape'][1] == K
                    and e['alpha_rowsum_maxdev'] <= 1e-5 and e['u_in_unit'] and e['u_shape'] == [al.shape[0]] and e['alpha_row_range_mean'] > 1e-4
                    and e['t2i_refiner_got_alpha_u'] and e['i2t_refiner_plain'] and e['teacher_is_other_peer'] and e['t2i_proto_differs_from_cnative']
                    and e['head_in_optimizer'] == 0 and e['head_with_grad'] == 0 and e['head_requires_grad'] == 0 and e['head_unchanged_by_step']
                    and e['head_present'] == (kind != 'cos') and e['calib_w_requires_grad'] is False and e['calib_w_in_optimizer'] is False)
            if kind in SEL.RES_KINDS:
                lgr = r_['log']
                e.update(res_active=bool(r_['res_active']), a=r_['a'], b=r_['b'], beta=lgr.get('rstage_beta'),
                         res_rowstd_step=s_.get('t2i_r28_res_rowstd', 0.0), aff_converged=lgr.get('rstage_aff_converged'),
                         aff_grad_maxabs=lgr.get('rstage_aff_grad_maxabs'),
                         alpha_differs_from_g1=not bool(torch.equal(al, steps['g1_cos_sel']['al'][0]['alpha'])),
                         residual_train_loss=(lgr.get('rstage_train_loss_first'), lgr.get('rstage_train_loss_last')),
                         val_J_g0_vs_res=(lgr.get('rstage_g0_val_J'), lgr.get('rstage_val_J')),
                         val_bce_g0_vs_res=(lgr.get('rstage_g0_val_bce'), lgr.get('rstage_val_bce')),
                         hard_top1_cos_vs_arm=(lgr.get('rstage_hard_cos_top1_obs'), lgr.get('rstage_hard_arm_top1_obs')))
                ok_r = ok_r and e['res_active'] and e['a'] >= C.A_MIN and e['res_rowstd_step'] > 0 and e['alpha_differs_from_g1']
            base = base and ok_r
        e['pass'] = base
        good &= base; r4[arm] = e
    ok('mechanics', good, **r4)

    # ---- (7) calibration set / anchor
    ld = lt.dataset
    prob = np.asarray(ld.probability, dtype=np.float64).reshape(-1)
    thr = float(B.avg_clean_prob)
    seed = QP.stage_seed(opt.seed, ep, 0)
    B.val_start()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed)
        fitd, vald, info, (g_fit, o_fit, r_fit) = QP.stage_data(B, ld, seed)
        torch.manual_seed(seed)
        _, _, rows_all = QP.select_P(ld, B.avg_clean_prob)
        perm = torch.randperm(len(rows_all)).numpy()
        used = rows_all[perm[:info['nP_fit'] + info['nP_val']]]
        gen = torch.Generator().manual_seed(seed + C.CAL_SEED_OFFSET)
        rows, pinfo = SEL.cal_rows(ld, B.avg_clean_prob, used, gen)
        a1, b1, aff1 = SEL.calib_affine(fitd[0], fitd[1])
        a2, b2, aff2 = SEL.calib_affine(fitd[0], fitd[1])
    lo = np.asarray(ld.t2i_index, dtype=np.int64); lg_ = np.asarray(ld.r27_gidx, dtype=np.int64)
    lgr = steps['r1_res_vcs_sel']['al'][0]['log']
    # own-record exclusion == production C_mix on a queue view without the own slots (first 64 calibration-fit rows)
    q2 = B.memory_queue
    rr = rows['fit'][:64]
    with torch.no_grad():
        qv = QP.center(QP.t_txt_raw(B, lg_[rows['fit'][:QC.CHUNK]]))[:64]
        own = torch.as_tensor(lg_[rr], device=qv.device)
        fin, _, _ = SEL.cal_search(q2, qv, own)
        excl_eq, own_in_queue = 0, 0
        for i in range(len(rr)):
            ids2 = {k: v.clone() for k, v in q2.r27_ids.items()}
            msk = ids2['cap'] == own[i]
            own_in_queue += int(msk.any())
            ids2['cap'][msk] = -1; ids2['img'][msk] = -1
            view = types.SimpleNamespace(queue_size=q2.queue_size, is_full=q2.is_full, ptr=q2.ptr, img_queue=q2.img_queue,
                                         txt_queue=q2.txt_queue, r27_ids=ids2)
            Si, _ = CS.search_t2i(view, qv[i:i + 1], 'C_mix', K)
            excl_eq += int(Si is not None and torch.equal(Si['img'][0], fin[i]))
    e = dict(stage_seed=seed, thr=thr, n_cal_fit=int(len(rows['fit'])), n_cal_rep=int(len(rows['rep'])),
             logged_cal_fit_rows_plus_short=(lgr.get('rstage_cal_fit_rows', 0) + lgr.get('rstage_cal_fit_short', 0)),
             rows_high_conf=bool((prob[rows['fit']] > thr).all() and (prob[rows['rep']] > thr).all()),
             rows_disjoint_from_head_sets=bool(not np.isin(rows['fit'], used).any() and not np.isin(rows['rep'], used).any()),
             fit_rep_images_disjoint=not bool(set(lo[rows['fit']].tolist()) & set(lo[rows['rep']].tolist())),
             own_record_exclusion_equal_production=excl_eq, own_record_rows_checked=int(len(rr)), own_records_present_in_queue=own_in_queue,
             cal_fit_pos=lgr.get('rstage_cal_fit_pos'), cal_fit_neg=lgr.get('rstage_cal_fit_neg'), cal_status=lgr.get('rstage_cal_status'),
             calfit_weak_rate=lgr.get('rstage_calfit_weak_rate'), calrep_weak_rate=lgr.get('rstage_calrep_weak_rate'),
             calrep_auc=lgr.get('rstage_calrep_auc'), calrep_ece=lgr.get('rstage_calrep_ece'),
             calrep_u_with_pair=lgr.get('rstage_calrep_u_mean_with_pair'), calrep_u_without_pair=lgr.get('rstage_calrep_u_mean_without_pair'),
             anchor=dict(a=a1, b=b1, reproducible=(a1 == a2 and b1 == b2), stage_logged=(lgr.get('rstage_a'), lgr.get('rstage_b')),
                         equals_stage=(a1 == lgr.get('rstage_a') and b1 == lgr.get('rstage_b')), **aff1),
             evaluator_only_cal_fit_observed_is_injected_clean=float((lo[rows['fit']] == lg_[rows['fit']] // 5).mean()),
             evaluator_only_cal_rep_observed_is_injected_clean=float((lo[rows['rep']] == lg_[rows['rep']] // 5).mean()),
             note='evaluator_only_* use the injected-noise pairing inside this gate only; training never reads it')
    e['pass'] = (e['n_cal_fit'] > 0 and e['n_cal_rep'] > 0 and e['logged_cal_fit_rows_plus_short'] == e['n_cal_fit'] and e['rows_high_conf']
                 and e['rows_disjoint_from_head_sets'] and e['fit_rep_images_disjoint'] and excl_eq == len(rr) and own_in_queue > 0
                 and e['cal_status'] == 'ok' and min(e['cal_fit_pos'] or 0, e['cal_fit_neg'] or 0) >= C.MIN_CLASS
                 and e['anchor']['reproducible'] and e['anchor']['equals_stage'] and aff1['aff_ok'] and aff1['aff_grad_maxabs'] < 1e-4)
    ok('calibration', e['pass'], **e)

    # ---- (8) P / Q truth-free
    rows_ref = np.nonzero(prob > thr)[0]
    s2 = steps['r1_res_vcs_sel']['stats']
    lab_g = np.asarray(ld.r27_gidx); lab_o = np.asarray(ld.t2i_index)
    pred_rows_global = np.nonzero(np.asarray(pred_B))[0]
    e = dict(thr=thr, thr_equals_teacher_mean=abs(thr - float(np.asarray(ld.probability, dtype=np.float32).mean())) <= 1e-6, nP_avail=int(len(rows_ref)),
             stage_log_nP_avail=s2.get('rstage_nP_avail'), stage_log_nP_fit=s2.get('rstage_nP_fit'), nP_fit=int(len(r_fit)),
             rows_subset=bool(np.isin(r_fit, rows_ref).all()), rows_unique=len(np.unique(r_fit)) == len(r_fit),
             gidx_is_labeled_observed=bool(np.array_equal(g_fit, lab_g[r_fit]) and np.array_equal(o_fit, lab_o[r_fit])),
             labeled_is_teacher_clean_split=bool(np.array_equal(lab_g, pred_rows_global)),
             observed_consistent_with_noise_lock=bool((NOISE[g_fit] == o_fit).all()),
             evaluator_only_P_clean_fraction=float((o_fit == g_fit // 5).mean()))
    import test_r28 as TR
    hits = {f: [w for w in TR.TRUTH_WORDS if w in TR.code_only(os.path.join(code_dir, f))] for f in ('r28_select.py', 'r28_config.py')}
    src_nb = inspect.getsource(R27.noisy_branches)
    hook_lines = [l.strip() for l in src_nb.splitlines() if 'R28_ARMS' in l or 'SEL.' in l]
    st_lines = [l.strip() for l in inspect.getsource(R27.train).splitlines() if 'SEL.' in l]
    e.update(static_truth_hits=hits, hook_lines=hook_lines, stage_hook_lines=st_lines)
    e['pass'] = (e['thr_equals_teacher_mean'] and e['nP_avail'] == e['stage_log_nP_avail'] and e['nP_fit'] == e['stage_log_nP_fit'] and e['nP_fit'] > 0
                 and e['rows_subset'] and e['rows_unique'] and e['gidx_is_labeled_observed'] and e['labeled_is_teacher_clean_split']
                 and e['observed_consistent_with_noise_lock'] and not any(hits.values()) and len(hook_lines) == 4 and len(st_lines) == 1)
    ok('pq_truth_free', e['pass'], **e)

    # ---- (9) resume
    rs = {}
    arm_r = 'r1_res_vcs_sel'
    a = one_step(arm_r, probe_grads=False); b = one_step(arm_r, probe_grads=False)
    rs['snapshot_twice'] = dict(param_maxdiff=float((a['params'] - b['params']).abs().max()), adam_maxdiff=float((a['adam'] - b['adam']).abs().max()),
                                rng_equal=rng_eq(a, b), head_sha_equal=a['al'][0]['head_sha'] == b['al'][0]['head_sha'],
                                alpha_maxdiff=float((a['al'][0]['alpha'] - b['al'][0]['alpha']).abs().max()),
                                u_maxdiff=float((a['al'][0]['u'] - b['al'][0]['u']).abs().max()),
                                anchor_equal=(a['al'][0]['a'], a['al'][0]['b']) == (b['al'][0]['a'], b['al'][0]['b']))
    R.load_full_state(S0, opt, A, B)
    for m, prob_, pred in ((A, prob_A, pred_A), (B, prob_B, pred_B)):
        m.avg_clean_prob = prob_[pred].mean()
    for m in (A, B):
        for p in list(m.params) + list(m.i2t_refiner_node.parameters()) + list(m.t2i_refiner_node.parameters()):
            p.grad = None
    R.set_rng_state(snap[2]); R27._PEER.clear(); R27._PEER[id(A)] = 0; R27._PEER[id(B)] = 1
    c = one_step(arm_r, probe_grads=False, do_restore=False)
    rs['disk_reload'] = dict(param_maxdiff=float((a['params'] - c['params']).abs().max()), adam_maxdiff=float((a['adam'] - c['adam']).abs().max()),
                             rng_equal=rng_eq(a, c), head_sha_equal=a['al'][0]['head_sha'] == c['al'][0]['head_sha'],
                             u_maxdiff=float((a['al'][0]['u'] - c['al'][0]['u']).abs().max()),
                             queue_ids_equal=all(torch.equal(a['own_ids'][k], c['own_ids'][k]) for k in a['own_ids']) and set(a['own_ids']) == set(c['own_ids']))
    saved = {}
    for arm in RA:
        dst = torch.load(f'{smoke}/{arm}/state_last.pt', map_location='cpu', weights_only=False)
        so = dst.get('opt')
        g_ = (lambda k_: so.get(k_) if isinstance(so, dict) else getattr(so, k_, None))
        saved[arm] = dict(saved_arm=g_('arm'), saved_lam=g_('lam'), saved_kind=g_('r28_kind'), saved_source=g_('r28_source'),
                          has_queue_ids=dst['model_B'].get('r27_queue_ids') is not None)
    rs['smoke_states'] = saved
    tmp = tempfile.mkdtemp(prefix='r28_gate_id_')
    sys.path.insert(0, os.path.dirname(RUNNER)); import run_unit_r28 as RU
    base = [sys.executable, '-B', RUNNER, '--cells', CELLS, '--code-dir', code_dir, '--data-path', dpath, '--output-root', tmp]

    def ru(unit):
        p = subprocess.run(base + ['--unit', unit], capture_output=True, text=True)
        return p.returncode, (p.stdout + p.stderr).strip()
    code0, out0 = ru('f40_s1001__r1_res_vcs_sel')
    lam_zero_in_cmd = '--arm r1_res_vcs_sel --lam 0.0 ' in out0
    ident = RU.identity_for('f40_s1001__r1_res_vcs_sel', CELLS, code_dir)[-1]
    for arm_dir in ('r1_res_vcs_sel', 'r2_res_bce_sel'):
        os.makedirs(f'{tmp}/f40_s1001/{arm_dir}')
        json.dump(ident, open(f'{tmp}/f40_s1001/{arm_dir}/R28_IDENTITY.json', 'w'), indent=2)
        open(f'{tmp}/f40_s1001/{arm_dir}/state_last.pt', 'w').close()
    os.makedirs(f'{tmp}/f60_s1001/r1_res_vcs_sel'); open(f'{tmp}/f60_s1001/r1_res_vcs_sel/state_last.pt', 'w').close()
    code1, out1 = ru('f40_s1001__r1_res_vcs_sel')
    code2, out2 = ru('f40_s1001__r2_res_bce_sel')
    code3, out3 = ru('f60_s1001__r1_res_vcs_sel')
    code4, out4 = ru('f40_s1001__q2_vcs_pool')
    shutil.rmtree(tmp)
    rs['run_unit_identity'] = dict(dry_fresh_rc=code0, same_arm_rc=code1, same_arm_resumes_state_last='state_last.pt' in out1,
                                   other_arm_rc=code2, other_arm_msg=out2[-200:], unidentified_state_rc=code3, unidentified_msg=out3[-200:],
                                   non_r28_unit_rc=code4, non_r28_msg=out4[-200:], lam_zero_in_cmd=lam_zero_in_cmd, dry_fresh_cmd=out0[-1200:],
                                   identity_has_protocol_sha=ident.get('protocol_frozen_sha256') is not None)
    sn, dr = rs['snapshot_twice'], rs['disk_reload']
    good = (sn['param_maxdiff'] == 0 and sn['adam_maxdiff'] == 0 and sn['rng_equal'] and sn['head_sha_equal'] and sn['alpha_maxdiff'] == 0
            and sn['u_maxdiff'] == 0 and sn['anchor_equal']
            and dr['param_maxdiff'] <= TOL and dr['adam_maxdiff'] <= TOL and dr['rng_equal'] and dr['queue_ids_equal'] and dr['head_sha_equal'] and dr['u_maxdiff'] == 0
            and all(v['saved_arm'] == k and v['saved_lam'] == 0.0 and v['saved_kind'] == SEL.R28_ARMS[k] and v['saved_source'] == 'C_mix' and v['has_queue_ids']
                    for k, v in saved.items())
            and code0 == 0 and lam_zero_in_cmd and code1 == 0 and rs['run_unit_identity']['same_arm_resumes_state_last']
            and code2 != 0 and 'identity mismatch' in out2 and code3 != 0 and 'Unidentified' in out3 and code4 != 0
            and rs['run_unit_identity']['identity_has_protocol_sha'])
    ok('resume', good, **rs)

    for hk in hooks:
        hk.remove()
    QP.alpha_t2i, SEL.alpha_u_t2i = orig_alpha_q, orig_alpha_r

    # ---- (10) timing and memory
    tim = {}
    for arm in ['a1_iface', 'cn_mix_t2i', 'q1_cos_pool', 'q2_vcs_pool'] + RA:
        restore(); opt.arm, opt.lam = arm, 0.0
        s = R.new_stats()
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
        R27.train(opt, A, B, lt, ut, ep, s, max_iters=30)
        torch.cuda.synchronize()
        tot = time.time() - t0
        fit = float(s.get('rstage_fit_seconds', s.get('qstage_fit_seconds', 0.0)))
        tim[arm] = dict(seconds_30it_incl_fit=tot, stage_fit_seconds=fit, ms_per_step_excl_fit=1000 * (tot - fit) / 30,
                        peak_mem_GiB=torch.cuda.max_memory_allocated() / 2**30, t2i_n=s.get('t2i_n', 0))
    for k in tim:
        tim[k]['ratio_vs_cnative_excl_fit'] = tim[k]['ms_per_step_excl_fit'] / tim['cn_mix_t2i']['ms_per_step_excl_fit']
    steps_per_stage = len(lt)
    est = {k: (v['stage_fit_seconds'] * 80 + (v['ms_per_step_excl_fit'] - tim['cn_mix_t2i']['ms_per_step_excl_fit']) / 1000 * steps_per_stage * 80) / 3600
           for k, v in tim.items() if k in RA}
    ok('timing', True, gpu=torch.cuda.get_device_name(0), note='30 iterations incl. data loading and the stage fit; model A stage only',
       est_extra_hours_per_unit_vs_cnative=est, steps_per_stage=steps_per_stage, **tim)

    # ---- (11) logs of the smoke arms
    arms_ok = {}
    for arm in RA:
        ee = json.load(open(f'{smoke}/{arm}/EPOCHS.json'))['cotrain']
        last = ee[-1]; tr = last['train']
        ran = {f'{p}/{b}': tr[p].get(f'{b}_n') or 0 for p in ('A', 'B') for b in ('i2t', 't2i')}
        fin = all(np.isfinite(tr[p].get(f'{b}_raw_loss_mean', np.nan)) for p in ('A', 'B') for b in ('i2t', 't2i') if ran[f'{p}/{b}'] > 0)
        both_dirs = all(any(ran[f'{p}/{b}'] > 0 for p in ('A', 'B')) for b in ('i2t', 't2i'))
        stage = {p: {k: v for k, v in tr[p].items() if k.startswith('rstage_')} for p in ('A', 'B')}
        rlog = {p: {k: v for k, v in tr[p].items() if k.startswith('t2i_r28_') or k.startswith('cnt_t2i_r28')} for p in ('A', 'B')}
        need = ['rstage_version', 'rstage_seed', 'rstage_beta', 'rstage_nP_avail', 'rstage_fit_seconds', 'rstage_cal_status',
                'rstage_cal_fit_rows', 'rstage_calrep_n', 'rstage_u_fallback']
        if SEL.R28_ARMS[arm] in SEL.RES_KINDS:
            need += ['rstage_a', 'rstage_b', 'rstage_aff_fallback', 'rstage_aff_converged']
        logged = all(all(k in stage[p] for k in need) for p in ('A', 'B'))
        r_logged = all('t2i_r28_u_mean' in rlog[p] and 't2i_r28_alpha_final_max_mean' in rlog[p] for p in ('A', 'B') if ran[f'{p}/t2i'] > 0) and any(ran[f'{p}/t2i'] > 0 for p in ('A', 'B'))
        arms_ok[arm] = dict(finite=fin and both_dirs, stage_logged=logged, r28_logged=r_logged, stage=stage, rlog=rlog, noisy_steps=ran,
                            dev_rsum=last['dev']['rsum'], epochs=[x['epoch'] for x in ee])
    ok('logs', all(v['finite'] and v['stage_logged'] and v['r28_logged'] for v in arms_ok.values()), **arms_ok)

    checks = {k: v for k, v in res.items() if isinstance(v, dict)}
    pf = os.path.join(os.path.dirname(HERE), 'PROTOCOL_FROZEN.json')
    out = {'all_pass': all(v['pass_'] for v in checks.values()) and code_now == manifest['files_sha256'] and bool(C.FROZEN),
           'config_frozen': bool(C.FROZEN), 'config': {k: getattr(C, k) for k in dir(C) if k.isupper()},
           'code_files_sha256': code_now, 'code_matches_install_manifest': code_now == manifest['files_sha256'],
           'checks': checks, 'gpu': torch.cuda.get_device_name(0), 'job': os.environ.get('SLURM_JOB_ID'),
           'source_state': S0, 'source_state_sha256': fsha(S0), 'gate_script_sha256': fsha(os.path.abspath(__file__)),
           'runner_sha256': fsha(RUNNER), 'protocol_frozen_sha256': fsha(pf) if os.path.exists(pf) else None,
           'raw_report_paths': [f'{smoke}/GPU_GATE.json'] + [f'{smoke}/{a}' for a in RA]}
    json.dump(out, open(f'{smoke}/GPU_GATE.json', 'w'), indent=1, default=float)
    print('ALL_PASS', out['all_pass'], flush=True)
    sys.exit(0 if out['all_pass'] else 1)


if __name__ == '__main__':
    main()
