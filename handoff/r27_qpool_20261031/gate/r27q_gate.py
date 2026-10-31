"""R27-Q real-data GPU gate (modeled on R27-CS gate/r27cs_gate.py). Run from code_q (cwd).
Usage: r27q_gate.py SMOKE_DIR NOISE_FILE DATA_PATH GATE_STATE CODE_DIR -> SMOKE_DIR/GPU_GATE.json

From one restored real R27-S smoke state (both queues ID-bearing): one real training step of model A per arm (stage fit
included), with a probe taking per-direction gradients (torch.autograd.grad, retain_graph; production backward untouched).
Arms: a1_iface, cn_mix_t2i (references) and q1..q4 normal, plus q1..q4 with the gate-only uniform routes
(r27q_pool.GATE_UNIFORM = 'mean': alpha computed then the native mean path; 'weighted': alpha = 1/5 through the weighted path).
  uniform_equiv   'mean' route == C-native bitwise (full-step encoder/refiner grads, params after the step, AdamW state,
                  RNG, per-direction probe grads, graphs) although the stage fit and teacher scoring ran;
                  'weighted' route vs C-native: max diffs reported, pass within float tolerance
  i2t_identical   normal Q arms: i2t direction == A1 bitwise; t2i candidate graph == C-native bitwise (same C-mix IDs)
  pooling         alpha stop-grad, rows sum 1, non-uniform, passed to the t2i refiner only (i2t refiner alpha=None); head
                  params not in the optimizer, no grad, unchanged by the step; other peer no grad, queue/IDs unchanged;
                  one optimizer.step; one refiner forward per direction (= A1); t2i prototype differs from C-native
  pq_truth_free   stage P == teacher high-confidence labeled rows, observed pairing (recomputed independently); counts
                  match the stage log; static truth check on r27q_pool / hook lines; P clean fraction (evaluator only)
  resume          snapshot restore twice bitwise (incl. stage fit); disk reload; smoke state arms; runner identity
  timing          ms/step (30 iterations incl. the stage fit) and peak memory per arm; stage-fit seconds
  logs            smoke arms: finite losses, qstage_* and t2i_q_* logs per peer stage
"""
import os, sys, json, copy, hashlib, time, inspect, subprocess, tempfile, shutil, re
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(sys.argv[5]))
import r25_run
import r25_co_train as R
import r27_co_train as R27
import r27_interface as I
import r27cs_sources as CS
import r27q_pool as QP
import r27q_config as C
from data import get_dataset
from vse import VSE
from vocab import deserialize_vocab
from co_train import eval_train, split_prob

res = {}
QA = list(QP.Q_ARMS)
TOL = 1e-6
K = 5
HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(os.path.dirname(HERE), 'tools', 'run_unit_q.py')
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


def gnorm(a):
    return max([float(x.abs().max()) for x in a if x is not None] + [0.0])


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
    al_rec, uids_rec, ref_rec = [], [], []
    orig_alpha = QP.alpha_t2i

    def rec_alpha(opt_, net, net2, S, uids, stats, timed=False):
        a = orig_alpha(opt_, net, net2, S, uids, stats, timed)
        st_ = QP.STAGE
        al_rec.append(dict(alpha=None if a is None else a.detach().clone(), req=None if a is None else bool(a.requires_grad),
                           head=st_.get('head'), head_sha=None if st_.get('head') is None else QP.head_sha(st_['head']),
                           S={kk: (v.clone() if torch.is_tensor(v) else v) for kk, v in S.items()}, uids=tuple(u.clone() for u in uids),
                           teacher_is_B=net2 is B))
        return a
    QP.alpha_t2i = rec_alpha
    orig_nb = R27.noisy_branches

    def rec_nb(opt_, net, net2, *a, **k):
        uids_rec.append(tuple(u.clone() for u in a[-2]))
        return orig_nb(opt_, net, net2, *a, **k)
    R27.noisy_branches = rec_nb

    def hook(name):
        def f(mod, args, kwargs, out):
            al = kwargs.get('alpha', args[1] if len(args) > 1 else None)
            ref_rec.append(dict(name=name, alpha=None if al is None else al.detach().clone(), out=out.detach().clone()))
        return f
    hooks = [A.i2t_refiner_node.register_forward_hook(hook('i2t'), with_kwargs=True),
             A.t2i_refiner_node.register_forward_hook(hook('t2i'), with_kwargs=True)]

    def one_step(arm, uniform=None, probe_grads=True, do_restore=True):
        if do_restore:
            restore()
        opt.arm, opt.lam = arm, 0.0
        QP.GATE_UNIFORM = uniform
        cap, br = {'nstep': 0}, {}
        al_rec.clear(); uids_rec.clear(); ref_rec.clear()
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

        def step_probe(*a, **k):
            cap['nstep'] += 1
            cap['enc'] = [None if p.grad is None else p.grad.detach().clone() for p in A.params]
            cap['ref_i2t'] = [None if p.grad is None else p.grad.detach().clone() for p in A.i2t_refiner_node.parameters()]
            cap['ref_t2i'] = [None if p.grad is None else p.grad.detach().clone() for p in A.t2i_refiner_node.parameters()]
            cap['peer_with_grad'] = sum(1 for p in peer_all() if p.grad is not None)
            hd = QP.STAGE.get('head')
            cap['head_sha_before_step'] = None if hd is None else QP.head_sha(hd)
            cap['head_with_grad'] = 0 if hd is None else sum(1 for p in hd.parameters() if p.grad is not None)
            opt_ids = {id(p) for g in A.optimizer.param_groups for p in g['params']}
            cap['head_in_optimizer'] = 0 if hd is None else sum(1 for p in hd.parameters() if id(p) in opt_ids)
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
            QP.GATE_UNIFORM = None
        cap.update(rng=R.rng_state(), stats=dict(stats), br=br, al=list(al_rec), refs=list(ref_rec), uids=list(uids_rec),
                   peer_queue_same=(h(B.memory_queue.img_queue), h(B.memory_queue.txt_queue), {k: h(v) for k, v in ids_of(B.memory_queue).items()}) == qh,
                   peer_params_same=QP.fingerprint(B) == bfp,
                   own_ids={k: v.clone() for k, v in ids_of(A.memory_queue).items()})
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

    steps = {'a1_iface': one_step('a1_iface'), 'cn_mix_t2i': one_step('cn_mix_t2i')}
    for arm in QA:
        steps[arm] = one_step(arm)
        steps[arm + '/mean'] = one_step(arm, uniform='mean')
        steps[arm + '/weighted'] = one_step(arm, uniform='weighted')

    # ---- (1) uniform routes == C-native
    r0, good = {}, True
    cn = steps['cn_mix_t2i']
    for arm in QA:
        a = steps[arm + '/mean']
        e = full_cmp(a, cn)
        e['stage_fitted'] = bool(a['stats'].get('qstage_version')) and (QP.Q_ARMS[arm] == 'cos' or a['stats'].get('qstage_nP_fit', 0) > 0)
        e['alpha_computed_rows'] = a['stats'].get('cnt_t2i_q_rows', 0)
        e['refiner_alpha_none'] = all(r['alpha'] is None for r in a['refs'])
        w = steps[arm + '/weighted']
        ew = full_cmp(w, cn)
        rel = lambda x, y: gdiff(x, y) / max(gnorm(y), 1e-30)
        ew_s = dict(param_maxdiff=ew['param_maxdiff'], adam_maxdiff=ew['adam_maxdiff'], rng_equal=ew['rng_equal'],
                    proto_t2i_maxdiff=ew['proto_maxdiff']['t2i'], proto_i2t_maxdiff=ew['proto_maxdiff']['i2t'],
                    enc_grad_rel=rel(w['enc'], cn['enc']), ref_t2i_grad_rel=rel(w['ref_t2i'], cn['ref_t2i']),
                    t2i_dir_grad_rel=rel(w['br']['t2i']['grads'], cn['br']['t2i']['grads']), i2t_bitwise=ew['dir_grad_maxdiff']['i2t'] == 0,
                    graph_equal=ew['graph_equal'], t2i_alpha_passed_uniform=all(torch.equal(r['alpha'], torch.full_like(r['alpha'], 0.2)) for r in w['refs'] if r['name'] == 't2i'))
        # v2 (after 1078152): the param criterion was 1e-6; the observed 1.08e-6 is fp32 rounding of sum_j 0.2*h_j vs mean
        # (proto diff 6e-8, grad rel <= 7.5e-6) passed through one AdamW step (lr 5e-4, m/sqrt(v) normalisation). The
        # exact equivalence is carried by the bitwise 'mean' route; the weighted route is held to float tolerance 1e-5.
        ew_s['pass'] = (ew_s['proto_t2i_maxdiff'] <= 1e-6 and ew_s['enc_grad_rel'] <= 1e-4 and ew_s['ref_t2i_grad_rel'] <= 1e-4
                        and ew_s['t2i_dir_grad_rel'] <= 1e-4 and ew_s['param_maxdiff'] <= 1e-5 and ew_s['rng_equal'] and ew_s['i2t_bitwise']
                        and all(ew_s['graph_equal'].values()) and ew_s['t2i_alpha_passed_uniform'])
        e['weighted_route'] = ew_s
        e['pass'] = e['bitwise'] and e['stage_fitted'] and e['alpha_computed_rows'] > 0 and e['refiner_alpha_none'] and ew_s['pass']
        good &= e['pass']
        r0[arm] = e
    ok('uniform_equiv', good, **r0)

    # ---- (2) i2t identical to A1, t2i candidate graph == C-native
    r1, good = {}, True
    for arm in QA:
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
        good &= e['pass']
        r1[arm] = e
    ok('i2t_identical', good, **r1)

    # ---- (3) pooling mechanics
    r2, good = {}, True
    for arm in ['a1_iface', 'cn_mix_t2i'] + QA:
        c = steps[arm]; s_ = c['stats']
        nz = lambda L: sum(1 for g in L if g is not None and float(g.abs().sum()) > 0)
        e = dict(i2t_n=s_.get('i2t_n', 0), t2i_n=s_.get('t2i_n', 0), enc_grad_nonzero=nz(c['enc']), ref_i2t_with_grad=nz(c['ref_i2t']),
                 ref_t2i_with_grad=nz(c['ref_t2i']), peer_params_with_grad=c['peer_with_grad'], peer_queue_ids_same=c['peer_queue_same'],
                 peer_params_same=c['peer_params_same'], optimizer_steps=c['nstep'], refiner_forwards=[r['name'] for r in c['refs']],
                 stopgrad_graph=all((not v['Y_req']) and (not v['neigh_req']) and v['X_req'] and v['finite'] and v['aux_none'] for v in c['br'].values()))
        base = (e['i2t_n'] == 1 and e['t2i_n'] == 1 and e['enc_grad_nonzero'] > 0 and e['ref_i2t_with_grad'] > 0 and e['ref_t2i_with_grad'] > 0
                and e['peer_params_with_grad'] == 0 and e['peer_queue_ids_same'] and e['peer_params_same'] and e['optimizer_steps'] == 1
                and e['refiner_forwards'] == ['i2t', 't2i'] and e['stopgrad_graph'])
        if arm in QA:
            al = c['al'][0]['alpha']; t2r = [r for r in c['refs'] if r['name'] == 't2i'][0]; i2r = [r for r in c['refs'] if r['name'] == 'i2t'][0]
            e.update(alpha_calls=len(c['al']), alpha_requires_grad=c['al'][0]['req'], alpha_shape=list(al.shape),
                     alpha_rowsum_maxdev=float((al.sum(1) - 1).abs().max()), alpha_min=float(al.min()), alpha_max=float(al.max()),
                     alpha_row_range_mean=float((al.max(1).values - al.min(1).values).mean()),
                     alpha_entropy_mean=float(-(al * al.clamp_min(1e-30).log()).sum(1).mean()),
                     t2i_refiner_got_alpha=t2r['alpha'] is not None and bool(torch.equal(t2r['alpha'], al)),
                     i2t_refiner_alpha_none=i2r['alpha'] is None, teacher_is_other_peer=c['al'][0]['teacher_is_B'],
                     t2i_proto_differs_from_cnative=not bool(torch.equal(c['br']['t2i']['proto'], cn['br']['t2i']['proto'])),
                     head_in_optimizer=c['head_in_optimizer'], head_with_grad=c['head_with_grad'],
                     head_unchanged_by_step=c['head_sha_before_step'] == c['head_sha_after_step'],
                     head_present=c['head_sha_before_step'] is not None)
            ok_q = (e['alpha_calls'] == 1 and e['alpha_requires_grad'] is False and e['alpha_shape'][1] == K and e['alpha_rowsum_maxdev'] <= 1e-5
                    and e['alpha_row_range_mean'] > 1e-4 and e['t2i_refiner_got_alpha'] and e['i2t_refiner_alpha_none'] and e['teacher_is_other_peer']
                    and e['t2i_proto_differs_from_cnative'] and e['head_in_optimizer'] == 0 and e['head_with_grad'] == 0 and e['head_unchanged_by_step']
                    and e['head_present'] == (QP.Q_ARMS[arm] != 'cos'))
            base = base and ok_q
        e['pass'] = base
        good &= base
        r2[arm] = e
    ok('pooling', good, **r2)

    # ---- (4) P / Q truth-free (independent recomputation of the stage P)
    r3 = {}
    ld = lt.dataset
    prob = np.asarray(ld.probability, dtype=np.float64).reshape(-1)
    thr = float(B.avg_clean_prob)
    rows_ref = np.nonzero(prob > thr)[0]
    seed = QP.stage_seed(opt.seed, ep, 0)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed)
        fitd, vald, info, (g_fit, o_fit, r_fit) = QP.stage_data(B, ld, seed)
    s2 = steps['q2_vcs_pool']['stats']
    lab_g = np.asarray(ld.r27_gidx); lab_o = np.asarray(ld.t2i_index)
    pred_rows_global = np.nonzero(np.asarray(pred_B))[0]
    # v3: threshold compared with the fp32 mean as computed in R.main (v1: float64 mean at 1e-9)
    e = dict(thr=thr, thr_equals_teacher_mean=abs(thr - float(np.asarray(ld.probability, dtype=np.float32).mean())) <= 1e-6, nP_avail=int(len(rows_ref)), stage_log_nP_avail=s2.get('qstage_nP_avail'),
             stage_log_nP_fit=s2.get('qstage_nP_fit'), nP_fit=int(len(r_fit)), rows_subset=bool(np.isin(r_fit, rows_ref).all()),
             rows_unique=len(np.unique(r_fit)) == len(r_fit),
             gidx_is_labeled_observed=bool(np.array_equal(g_fit, lab_g[r_fit]) and np.array_equal(o_fit, lab_o[r_fit])),
             labeled_is_teacher_clean_split=bool(np.array_equal(lab_g, pred_rows_global)),
             observed_consistent_with_noise_lock=bool((NOISE[g_fit] == o_fit).all()),
             all_prob_above_thr=bool((prob[r_fit] > thr).all()),
             evaluator_only_P_clean_fraction=float((o_fit == g_fit // 5).mean()),
             evaluator_only_all_labeled_clean_fraction=float((lab_o == lab_g // 5).mean()),
             note='clean fractions use the injected-noise truth inside this gate only (evaluator); training never reads it')
    import test_qpool as TQ
    code_txt = TQ.code_only(os.path.join(code_dir, 'r27q_pool.py'))
    hits = [w for w in TQ.TRUTH_WORDS if w in code_txt]
    src_nb = inspect.getsource(orig_nb)
    hook_lines = [l.strip() for l in src_nb.splitlines() if 'Q_ARMS' in l or 'QP.' in l]
    st_lines = [l.strip() for l in inspect.getsource(R27.train).splitlines() if 'QP.' in l]
    e.update(static_truth_hits=hits, hook_lines=hook_lines, stage_hook_lines=st_lines)
    e['pass'] = (e['thr_equals_teacher_mean'] and e['nP_avail'] == e['stage_log_nP_avail'] and e['nP_fit'] == e['stage_log_nP_fit'] and e['nP_fit'] > 0
                 and e['rows_subset'] and e['rows_unique'] and e['gidx_is_labeled_observed'] and e['labeled_is_teacher_clean_split']
                 and e['observed_consistent_with_noise_lock'] and e['all_prob_above_thr'] and not hits and len(hook_lines) == 4 and len(st_lines) == 1)
    r3 = e
    ok('pq_truth_free', e['pass'], **r3)

    # ---- (5) resume
    rs = {}
    a = one_step('q2_vcs_pool', probe_grads=False); b = one_step('q2_vcs_pool', probe_grads=False)
    rs['snapshot_twice'] = dict(param_maxdiff=float((a['params'] - b['params']).abs().max()), adam_maxdiff=float((a['adam'] - b['adam']).abs().max()),
                                rng_equal=rng_eq(a, b), head_sha_equal=a['al'][0]['head_sha'] == b['al'][0]['head_sha'],
                                alpha_maxdiff=float((a['al'][0]['alpha'] - b['al'][0]['alpha']).abs().max()))
    R.load_full_state(S0, opt, A, B)
    for m, prob_, pred in ((A, prob_A, pred_A), (B, prob_B, pred_B)):
        m.avg_clean_prob = prob_[pred].mean()
    for m in (A, B):
        for p in list(m.params) + list(m.i2t_refiner_node.parameters()) + list(m.t2i_refiner_node.parameters()):
            p.grad = None
    R.set_rng_state(snap[2]); R27._PEER.clear(); R27._PEER[id(A)] = 0; R27._PEER[id(B)] = 1
    c = one_step('q2_vcs_pool', probe_grads=False, do_restore=False)
    rs['disk_reload'] = dict(param_maxdiff=float((a['params'] - c['params']).abs().max()), adam_maxdiff=float((a['adam'] - c['adam']).abs().max()),
                             rng_equal=rng_eq(a, c), head_sha_equal=a['al'][0]['head_sha'] == c['al'][0]['head_sha'],
                             queue_ids_equal=all(torch.equal(a['own_ids'][k], c['own_ids'][k]) for k in a['own_ids']) and set(a['own_ids']) == set(c['own_ids']))
    saved = {}
    for arm in QA:
        dst = torch.load(f'{smoke}/{arm}/state_last.pt', map_location='cpu', weights_only=False)
        so = dst.get('opt')
        g_ = (lambda k_: so.get(k_) if isinstance(so, dict) else getattr(so, k_, None))
        saved[arm] = dict(saved_arm=g_('arm'), saved_lam=g_('lam'), saved_scorer=g_('r27q_scorer'), saved_source=g_('r27q_source'),
                          has_queue_ids=dst['model_B'].get('r27_queue_ids') is not None)
    rs['smoke_states'] = saved
    tmp = tempfile.mkdtemp(prefix='r27q_gate_id_')
    sys.path.insert(0, os.path.dirname(RUNNER)); import run_unit_q as RU
    base = [sys.executable, '-B', RUNNER, '--cells', CELLS, '--code-dir', code_dir, '--data-path', dpath, '--output-root', tmp]

    def ru(unit):
        p = subprocess.run(base + ['--unit', unit], capture_output=True, text=True)
        return p.returncode, (p.stdout + p.stderr).strip()
    code0, out0 = ru('f40_s1001__q2_vcs_pool')
    lam_zero_in_cmd = '--arm q2_vcs_pool --lam 0.0 ' in out0
    ident = RU.identity_for('f40_s1001__q2_vcs_pool', CELLS, code_dir)[-1]
    for arm_dir in ('q2_vcs_pool', 'q3_bce_pool'):
        os.makedirs(f'{tmp}/f40_s1001/{arm_dir}')
        json.dump(ident, open(f'{tmp}/f40_s1001/{arm_dir}/R27Q_IDENTITY.json', 'w'), indent=2)
        open(f'{tmp}/f40_s1001/{arm_dir}/state_last.pt', 'w').close()
    os.makedirs(f'{tmp}/f60_s1001/q2_vcs_pool'); open(f'{tmp}/f60_s1001/q2_vcs_pool/state_last.pt', 'w').close()
    code1, out1 = ru('f40_s1001__q2_vcs_pool')
    code2, out2 = ru('f40_s1001__q3_bce_pool')
    code3, out3 = ru('f60_s1001__q2_vcs_pool')
    code4, out4 = ru('f40_s1001__cn_mix_t2i')
    shutil.rmtree(tmp)
    rs['run_unit_identity'] = dict(dry_fresh_rc=code0, same_arm_rc=code1, same_arm_resumes_state_last='state_last.pt' in out1,
                                   other_arm_rc=code2, other_arm_msg=out2[-200:], unidentified_state_rc=code3, unidentified_msg=out3[-200:],
                                   non_q_unit_rc=code4, non_q_msg=out4[-200:], lam_zero_in_cmd=lam_zero_in_cmd, dry_fresh_cmd=out0[-1200:],
                                   identity_has_protocol_sha=ident.get('protocol_frozen_sha256') is not None)
    good = (rs['snapshot_twice']['param_maxdiff'] == 0 and rs['snapshot_twice']['adam_maxdiff'] == 0 and rs['snapshot_twice']['rng_equal']
            and rs['snapshot_twice']['head_sha_equal'] and rs['snapshot_twice']['alpha_maxdiff'] == 0
            and rs['disk_reload']['param_maxdiff'] <= TOL and rs['disk_reload']['adam_maxdiff'] <= TOL and rs['disk_reload']['rng_equal']
            and rs['disk_reload']['queue_ids_equal'] and rs['disk_reload']['head_sha_equal']
            and all(v['saved_arm'] == k and v['saved_lam'] == 0.0 and v['saved_scorer'] == QP.Q_ARMS[k] and v['saved_source'] == 'C_mix' and v['has_queue_ids']
                    for k, v in saved.items())
            and code0 == 0 and lam_zero_in_cmd and code1 == 0 and rs['run_unit_identity']['same_arm_resumes_state_last']
            and code2 != 0 and 'identity mismatch' in out2 and code3 != 0 and 'Unidentified' in out3 and code4 != 0
            and rs['run_unit_identity']['identity_has_protocol_sha'])
    ok('resume', good, **rs)

    for hk in hooks:
        hk.remove()
    QP.alpha_t2i = orig_alpha
    R27.noisy_branches = orig_nb

    # ---- (6) timing and memory
    tim = {}
    for arm in ['a1_iface', 'cn_mix_t2i'] + QA:
        restore(); opt.arm, opt.lam = arm, 0.0
        s = R.new_stats()
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
        R27.train(opt, A, B, lt, ut, ep, s, max_iters=30)
        torch.cuda.synchronize()
        tot = time.time() - t0
        fit = float(s.get('qstage_fit_seconds', 0.0))
        tim[arm] = dict(seconds_30it_incl_fit=tot, stage_fit_seconds=fit, ms_per_step_excl_fit=1000 * (tot - fit) / 30,
                        peak_mem_GiB=torch.cuda.max_memory_allocated() / 2**30, t2i_n=s.get('t2i_n', 0),
                        qscore_s_per_timed_step=s.get('time_qscore_t2i', 0.0) / max(s.get('time_n', 1), 1))
    for k in tim:
        tim[k]['ratio_vs_cnative_excl_fit'] = tim[k]['ms_per_step_excl_fit'] / tim['cn_mix_t2i']['ms_per_step_excl_fit']
    steps_per_stage = len(lt)   # labeled batches of this stage (informative extrapolation only)
    est = {k: (v['stage_fit_seconds'] * 80 + (v['ms_per_step_excl_fit'] - tim['cn_mix_t2i']['ms_per_step_excl_fit']) / 1000 * steps_per_stage * 80) / 3600
           for k, v in tim.items() if k in QA}
    ok('timing', True, gpu=torch.cuda.get_device_name(0), note='30 iterations incl. data loading and the stage fit; model A stage only',
       est_extra_hours_per_unit_vs_cnative=est, steps_per_stage=steps_per_stage, **tim)

    # ---- (7) logs of the smoke arms
    arms_ok = {}
    for arm in QA:
        e = json.load(open(f'{smoke}/{arm}/EPOCHS.json'))['cotrain']
        last = e[-1]; tr = last['train']
        ran = {f'{p}/{b}': tr[p].get(f'{b}_n') or 0 for p in ('A', 'B') for b in ('i2t', 't2i')}
        fin = all(np.isfinite(tr[p].get(f'{b}_raw_loss_mean', np.nan)) for p in ('A', 'B') for b in ('i2t', 't2i') if ran[f'{p}/{b}'] > 0)
        both_dirs = all(any(ran[f'{p}/{b}'] > 0 for p in ('A', 'B')) for b in ('i2t', 't2i'))
        stage = {p: {k: v for k, v in tr[p].items() if k.startswith('qstage_')} for p in ('A', 'B')}
        qlog = {p: {k: v for k, v in tr[p].items() if k.startswith('t2i_q_') or k.startswith('qm') or k == 'cnt_t2i_q_rows'} for p in ('A', 'B')}
        need = ['qstage_version', 'qstage_seed', 'qstage_beta', 'qstage_nP_avail', 'qstage_fit_seconds']
        if QP.Q_ARMS[arm] != 'cos':
            need += ['qstage_nP_fit', 'qstage_val_J', 'qstage_val_bce', 'qstage_val_snce', 'qstage_head_sha', 'qstage_train_loss_last']
        logged = all(all(k in stage[p] for k in need) for p in ('A', 'B'))
        alpha_logged = all('t2i_q_alpha_entropy_mean' in qlog[p] and 't2i_q_alpha_max_mean' in qlog[p] for p in ('A', 'B') if ran[f'{p}/t2i'] > 0) and any(ran[f'{p}/t2i'] > 0 for p in ('A', 'B'))
        arms_ok[arm] = dict(finite=fin and both_dirs, stage_logged=logged, alpha_logged=alpha_logged, stage=stage, qlog=qlog, noisy_steps=ran,
                            dev_rsum=last['dev']['rsum'], epochs=[x['epoch'] for x in e])
    ok('logs', all(v['finite'] and v['stage_logged'] and v['alpha_logged'] for v in arms_ok.values()), **arms_ok)

    checks = {k: v for k, v in res.items() if isinstance(v, dict)}
    out = {'all_pass': all(v['pass_'] for v in checks.values()) and code_now == manifest['files_sha256'] and bool(C.FROZEN),
           'config_frozen': bool(C.FROZEN), 'config': {k: getattr(C, k) for k in dir(C) if k.isupper()},
           'code_files_sha256': code_now, 'code_matches_install_manifest': code_now == manifest['files_sha256'],
           'checks': checks, 'gpu': torch.cuda.get_device_name(0), 'job': os.environ.get('SLURM_JOB_ID'),
           'source_state': S0, 'source_state_sha256': fsha(S0), 'gate_script_sha256': fsha(os.path.abspath(__file__)),
           'runner_sha256': fsha(RUNNER), 'protocol_frozen_sha256': fsha(os.path.join(os.path.dirname(HERE), 'PROTOCOL_FROZEN.json'))
           if os.path.exists(os.path.join(os.path.dirname(HERE), 'PROTOCOL_FROZEN.json')) else None,
           'raw_report_paths': [f'{smoke}/GPU_GATE.json'] + [f'{smoke}/{a}' for a in QA]}
    json.dump(out, open(f'{smoke}/GPU_GATE.json', 'w'), indent=1, default=float)
    print('ALL_PASS', out['all_pass'], flush=True)
    sys.exit(0 if out['all_pass'] else 1)


if __name__ == '__main__':
    main()
