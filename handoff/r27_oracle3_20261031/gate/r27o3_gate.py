"""R27-O3 (ORACLE direct true-target diagnostic) real-data GPU gate, modeled on R27-O gate/r27o_gate.py. Run from code_o3.
Usage: r27o3_gate.py SMOKE_DIR NOISE_FILE DATA_PATH GATE_STATE CODE_DIR -> SMOKE_DIR/GPU_GATE.json

From one restored real R27-S smoke state (both queues partly ID-bearing): one real training step of model A per arm with a
probe taking the per-direction gradient of that direction's loss w.r.t. the encoder + current-peer refiners
(torch.autograd.grad, retain_graph; the production backward is untouched). Arms: a1_iface, o1_rescue_native_t2i,
o3_true_target_t2i, and o3 with the true target switched OFF (r27o_rescue.O3_TARGET = False, gate only -> must be O1).
  i2t_identical       O3 i2t direction (loss, graph, encoder+refiner grads) == A1 and == O1 bitwise
  t2i_true_target     O3 t2i: refiner gradient == O1 t2i refiner gradient bitwise (direction probe and full step), same
                      rescued graph / P input; encoder target T == an independent r27_interface.reencode of the true image
                      (truth taken from the dataset's evaluator-side clean index, gate only) by the current peer with the
                      current batch image mean (bitwise), == the rescued candidate's re-encoding (tolerance), unit norm, no
                      grad; T != O1 target (sg P); encoder grad == autograd of 0.5*SCE(X, sg T) (tolerance); dup counters
  o3_target_off_equiv O3 with the target off == O1 bitwise: full-step encoder/refiner grads, params after the step, AdamW
                      state, RNG, per-direction grads (the switch is gate-only)
  both_branches       both directions trained once, both own refiners get grads, other peer no grad, its queue/IDs
                      unchanged, one optimizer.step, refiner forwards equal to A1, stop-grad targets
  resume              snapshot restore twice bitwise; reload from disk; smoke checkpoint arm/flags; runner identity
  truth_isolation     truth tokens only in r27o_rescue among the noisy-branch sources; the two hooks gated by arm
  timing              ms/step (30 iterations, same state) and peak memory per arm
  arms_ran            the O3 smoke arm finished its smoke epoch with finite losses; rescue + duplicate-target counters logged
"""
import os, sys, json, copy, hashlib, time, inspect, subprocess, tempfile, shutil
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(sys.argv[5]))
import r25_run
import r25_co_train as R
import r27_co_train as R27
import r27_interface as I
import r27s_targets as ST
import r27d_policy as P
import r27o_rescue as RX
from data import get_dataset
from vse import VSE, infonce
from vocab import deserialize_vocab
from co_train import eval_train, split_prob

res = {}
O3 = 'o3_true_target_t2i'
LAM = 1.1448122442850819
TOL = 1e-6
HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(os.path.dirname(HERE), 'tools', 'run_unit_o3.py')
CELLS = '/home/infres/yinwang/CS_QMI/handoff/r27_qtarget_20261029/CELLS.json'
LAMF = '/home/infres/yinwang/CS_QMI/handoff/r27_qtarget_20261029/LAMBDA_flickr.json'


def ok(name, cond, **info):
    res[name] = dict(pass_=bool(cond), **info)
    print(name, 'PASS' if cond else 'FAIL', json.dumps(info, default=float)[:3000], flush=True)


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
    CLEAN_T2I = np.asarray(noisy_loader.dataset._t2i_index, dtype=np.int64)   # evaluator-side truth (gate only)
    NENC = len(A.params)

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
    NI2T = len(list(A.i2t_refiner_node.parameters()))
    peer_all = lambda: list(B.params) + list(B.i2t_refiner_node.parameters()) + list(B.t2i_refiner_node.parameters())
    ids_of = lambda q: {k: v.clone() for k, v in getattr(q, 'r27_ids', {}).items()}

    # record the noisy-branch call (uids, current batch image mean) and the true-target helper calls
    ctx = {}
    orig_nb = R27.noisy_branches

    def rec_nb(opt_, net, net2, *a, **k):
        ctx['uids'] = tuple(u.clone() for u in a[-2])      # positional: ..., mi, mt, stats, uids, timed
        ctx['mi'] = a[-5].clone()
        return orig_nb(opt_, net, net2, *a, **k)
    R27.noisy_branches = rec_nb
    tt_calls = []
    orig_tt = RX.true_target_t2i

    def rec_tt(net, cap_gidx, mean, stats=None):
        T, info = orig_tt(net, cap_gidx, mean, stats)
        tt_calls.append(dict(net_is_A=net is A, cap=cap_gidx.clone(), mean=mean.clone(), T=T.clone(), info={k: v.clone() for k, v in info.items()}))
        return T, info
    RX.true_target_t2i = rec_tt
    rx_calls = []
    orig_rx = RX.rescue_t2i

    def rec_rx(S, cap_gidx, k, epoch, stats=None):
        S2, info = orig_rx(S, cap_gidx, k, epoch, stats)
        rx_calls.append(1)
        return S2, info
    RX.rescue_t2i = rec_rx

    ref_calls = [0]
    hooks = [m.register_forward_hook(lambda *a: ref_calls.__setitem__(0, ref_calls[0] + 1))
             for m in (A.i2t_refiner_node, A.t2i_refiner_node)]

    def one_step(arm, o3_target=True, probe_grads=True, do_restore=True, indep=False):
        if do_restore:
            restore()
        opt.arm, opt.lam = arm, (0.0 if arm == 'a1_iface' else LAM)
        RX.O3_TARGET = o3_target
        cap, br = {'nstep': 0}, {}
        ref_calls[0] = 0; tt_calls.clear(); rx_calls.clear(); ctx.clear()
        params = list(A.params) + own_ref()

        def probe(branch, dd):
            Pd = dd.get('Pd')                         # None on the A1 (non-S) path
            e = dict(loss=float(dd['loss']), l_nat=float(dd['l_nat']), l_enc=None if dd['l_aux'] is None else float(dd['l_aux']),
                     target=None if dd['target'] is None else dd['target'].detach().clone(), Pd=None if Pd is None else Pd.detach().clone(),
                     graphA=dd['S']['A'].detach().clone(), cols=dd['S']['cols'].detach().clone(), img=dd['S']['img'].detach().clone(),
                     inv=dd['S']['inv'].detach().clone(), Y=dd['Y'].detach().clone(), proto=dd['proto'].detach().clone(),
                     Pd_req=None if Pd is None else bool(Pd.requires_grad), target_req=bool(dd['target'].requires_grad) if dd['target'] is not None else None,
                     X_req=bool(dd['X'].requires_grad), finite=bool(torch.isfinite(dd['loss'])))
            if probe_grads:
                g = torch.autograd.grad(dd['loss'], params, retain_graph=True, allow_unused=True)
                e['grads'] = [None if x is None else x.detach().clone() for x in g]
            if indep and branch == 't2i':
                # independent truth (dataset clean index, gate only) and an independent production re-encode call
                u0 = ctx['uids'][0]
                tru = torch.as_tensor(CLEAN_T2I[u0.cpu().numpy()], device=u0.device)
                cols_t, inv_t = torch.unique(tru, return_inverse=True)
                Yt = I.reencode(A, 't2i', cols_t, ctx['mi'])
                e['T_indep'] = Yt[inv_t].clone()
                e['truth_indep'] = tru.clone()
                Ti = e['T_indep']
                gi = torch.autograd.grad(0.5 * infonce(dd['X'].mm(Ti.t())), list(A.params), retain_graph=True, allow_unused=True)
                e['enc_grad_indep'] = [None if x is None else x.detach().clone() for x in gi]
            br[branch] = e
        orig = A.optimizer.step

        def step_probe(*a, **k):
            cap['nstep'] += 1
            cap['enc'] = [None if p.grad is None else p.grad.detach().clone() for p in A.params]
            cap['ref_i2t'] = [None if p.grad is None else p.grad.detach().clone() for p in A.i2t_refiner_node.parameters()]
            cap['ref_t2i'] = [None if p.grad is None else p.grad.detach().clone() for p in A.t2i_refiner_node.parameters()]
            cap['peer_with_grad'] = sum(1 for p in peer_all() if p.grad is not None)
            out = orig(*a, **k)
            cap['params'] = torch.cat([p.detach().flatten().double() for p in params])
            cap['adam'] = torch.cat([v.detach().flatten().double() for p in params for kk, v in A.optimizer.state.get(p, {}).items() if torch.is_tensor(v) and v.dim() > 0])
            cap['adam_step'] = [float(v) for p in params for kk, v in A.optimizer.state.get(p, {}).items() if kk == 'step']
            return out
        A.optimizer.step = step_probe
        qh = (h(B.memory_queue.img_queue), h(B.memory_queue.txt_queue), {k: h(v) for k, v in ids_of(B.memory_queue).items()})
        stats = R.new_stats()
        try:
            R27.train(opt, A, B, lt, ut, ep, stats, max_iters=1, probe=probe)
        finally:
            A.optimizer.step = orig
            RX.O3_TARGET = True
        cap.update(rng=R.rng_state(), stats=stats, br=br, ref_calls=ref_calls[0], tt_calls=list(tt_calls), rescue_calls=len(rx_calls),
                   ctx=dict(ctx), peer_queue_same=(h(B.memory_queue.img_queue), h(B.memory_queue.txt_queue), {k: h(v) for k, v in ids_of(B.memory_queue).items()}) == qh,
                   own_ids={k: v.clone() for k, v in ids_of(A.memory_queue).items()})
        return cap

    def rng_eq(a, b):
        na, nb_ = a['rng']['numpy'], b['rng']['numpy']
        return (bool(torch.equal(a['rng']['torch'], b['rng']['torch'])) and all(torch.equal(x, y) for x, y in zip(a['rng']['cuda'], b['rng']['cuda']))
                and na[0] == nb_[0] and np.array_equal(na[1], nb_[1]) and tuple(na[2:]) == tuple(nb_[2:]) and a['rng']['python'] == b['rng']['python'])

    steps = {'a1_iface': one_step('a1_iface'), 'o1_rescue_native_t2i': one_step('o1_rescue_native_t2i'),
             O3: one_step(O3, indep=True), O3 + '/off': one_step(O3, o3_target=False)}
    o3, o1, a1 = steps[O3], steps['o1_rescue_native_t2i'], steps['a1_iface']

    # ---- (1) i2t identical to A1 and O1
    r1, good = {}, True
    for ref in ('a1_iface', 'o1_rescue_native_t2i'):
        # vs A1: gradients / native SCE value / graph (A1 has no separate encoder target; its loss is 0.5*SCE(X,P), gradient-
        # identical to 0.5*SCE(sgX,P) + 0.5*SCE(X,sgP)); vs O1 additionally loss, encoder loss and target bitwise
        a, b = o3['br']['i2t'], steps[ref]['br']['i2t']
        full = ref != 'a1_iface'
        e = dict(grad_maxdiff=gdiff(a['grads'], b['grads']), l_nat_absdiff=abs(a['l_nat'] - b['l_nat']),
                 graph_equal=bool(torch.equal(a['graphA'], b['graphA']) and torch.equal(a['cols'], b['cols']) and torch.equal(a['img'], b['img'])),
                 Y_equal=bool(torch.equal(a['Y'], b['Y'])), proto_equal=bool(torch.equal(a['proto'], b['proto'])),
                 none_pattern_equal=[x is None for x in a['grads']] == [x is None for x in b['grads']])
        if full:
            e.update(l_enc_absdiff=abs(a['l_enc'] - b['l_enc']), loss_absdiff=abs(a['loss'] - b['loss']),
                     target_equal=bool(torch.equal(a['target'], b['target'])))
        else:
            e.update(a1_loss_vs_o3_loss=(b['loss'], a['loss']))
        e['bitwise'] = (e['grad_maxdiff'] == 0 and e['l_nat_absdiff'] == 0 and e['graph_equal'] and e['Y_equal'] and e['proto_equal']
                        and e['none_pattern_equal'] and (not full or (e['l_enc_absdiff'] == 0 and e['loss_absdiff'] == 0 and e['target_equal'])))
        good &= e['bitwise']
        r1['vs_' + ref] = e
    r1['full_step_ref_i2t_grad_maxdiff_vs_a1'] = gdiff(o3['ref_i2t'], a1['ref_i2t'])
    r1['full_step_ref_i2t_grad_maxdiff_vs_o1'] = gdiff(o3['ref_i2t'], o1['ref_i2t'])
    good &= r1['full_step_ref_i2t_grad_maxdiff_vs_a1'] == 0 and r1['full_step_ref_i2t_grad_maxdiff_vs_o1'] == 0
    ok('i2t_identical', good, **r1)

    # ---- (2) t2i true target
    a, b = o3['br']['t2i'], o1['br']['t2i']
    ref_sl = slice(NENC + NI2T, None)
    tc = o3['tt_calls']
    T, Ti = a['target'], a['T_indep']
    u0, u1 = o3['ctx']['uids']
    tru_rx = u0 // 5
    # T vs the rescued candidate's own re-encoding (Y gathered at the truth's column)
    pos = torch.searchsorted(a['cols'], a['truth_indep'])
    in_cols = bool(torch.equal(a['cols'][pos.clamp(max=a['cols'].numel() - 1)], a['truth_indep']))
    Ycand = a['Y'][pos]
    eg_ind = a['enc_grad_indep']; eg = a['grads'][:NENC]
    gnorm = max(float(x.abs().max()) for x in eg if x is not None)
    cnt = o3['stats']
    tru_np = a['truth_indep'].cpu().numpy()
    _, inv_np, c_np = np.unique(tru_np, return_inverse=True, return_counts=True)
    dup_gate = int((c_np[inv_np] > 1).sum())
    e = dict(rows=int(T.shape[0]), dim=int(T.shape[1]),
             refiner_dir_grad_maxdiff_vs_o1=gdiff(a['grads'][ref_sl], b['grads'][ref_sl]),
             refiner_full_step_t2i_grad_maxdiff_vs_o1=gdiff(o3['ref_t2i'], o1['ref_t2i']),
             l_ref_absdiff_vs_o1=abs(a['l_nat'] - b['l_nat']),
             graph_equal_vs_o1=bool(torch.equal(a['graphA'], b['graphA']) and torch.equal(a['cols'], b['cols']) and torch.equal(a['img'], b['img'])),
             Y_equal_vs_o1=bool(torch.equal(a['Y'], b['Y'])), proto_equal_vs_o1=bool(torch.equal(a['proto'], b['proto'])),
             helper_calls=len(tc), helper_net_is_current_peer=bool(tc and tc[0]['net_is_A']),
             helper_mean_is_batch_image_mean=bool(tc and torch.equal(tc[0]['mean'], o3['ctx']['mi'])),
             helper_cap_is_query_uids0=bool(tc and torch.equal(tc[0]['cap'], u0)),
             target_is_helper_output=bool(tc and torch.equal(tc[0]['T'], T)),
             target_vs_independent_reencode_maxdiff=float((T - Ti).abs().max()),
             truth_rx_equals_dataset_clean_index=bool(torch.equal(tru_rx, a['truth_indep'])),
             noise_lock_equals_loader_pairing=bool((NOISE[u0.cpu().numpy()] == u1.cpu().numpy()).all()),
             noisy_query_frac=float((NOISE[u0.cpu().numpy()] != u0.cpu().numpy() // 5).mean()),
             truth_in_rescued_cols=in_cols, target_vs_rescued_candidate_reencode_maxdiff=float((T - Ycand).abs().max()),
             target_vs_rescued_candidate_note=('same IDs re-encoded inside a different (larger) candidate batch: fp32 kernel-shape '
                                               'differences only; tolerance 1e-4 abs (fp32 T01 convention); the same-call comparison '
                                               'target_vs_independent_reencode_maxdiff is required bitwise'),
             target_vs_rescued_candidate_cos_min=float((T * Ycand).sum(1).min()),
             norm_maxdev=float((T.norm(dim=1) - 1).abs().max()), target_requires_grad=a['target_req'],
             target_vs_o1_target_maxdiff=float((T - b['target']).abs().max()),
             o1_target_is_sgP=bool(torch.equal(b['target'], b['Pd'])),
             cos_T_vs_P_mean=float((T * (b['Pd'] / b['Pd'].norm(dim=1, keepdim=True))).sum(1).mean()),
             l_enc_o3=a['l_enc'], l_enc_o1=b['l_enc'],
             loss_is_half_ref_plus_half_enc=abs(a['loss'] - (0.5 * a['l_nat'] + 0.5 * a['l_enc'])),
             enc_grad_vs_indep_sce_maxdiff=gdiff(eg, eg_ind), enc_grad_absmax=gnorm,
             enc_grad_maxdiff_vs_o1=gdiff(eg, b['grads'][:NENC]),
             dup_counters=dict(rows=cnt['cnt_t2i_o3_rows'], dup_rows=cnt['cnt_t2i_o3_dup_rows'], unique=cnt['cnt_t2i_o3_unique_targets'],
                               dup_frac=cnt['t2i_o3_dup_frac']),
             dup_rows_gate=dup_gate, unique_gate=int(len(c_np)),
             rescue_counters=dict(rows=cnt['cnt_t2i_rescue_rows'], present=cnt['cnt_t2i_truth_present_rows'], rescued=cnt['cnt_t2i_rescued_rows']),
             rescue_counters_o1=dict(rows=o1['stats']['cnt_t2i_rescue_rows'], present=o1['stats']['cnt_t2i_truth_present_rows'], rescued=o1['stats']['cnt_t2i_rescued_rows']),
             rescue_calls=o3['rescue_calls'])
    e['pass'] = (e['refiner_dir_grad_maxdiff_vs_o1'] == 0 and e['refiner_full_step_t2i_grad_maxdiff_vs_o1'] == 0 and e['l_ref_absdiff_vs_o1'] == 0
                 and e['graph_equal_vs_o1'] and e['Y_equal_vs_o1'] and e['proto_equal_vs_o1']
                 and e['helper_calls'] == 1 and e['helper_net_is_current_peer'] and e['helper_mean_is_batch_image_mean'] and e['helper_cap_is_query_uids0']
                 and e['target_is_helper_output'] and e['target_vs_independent_reencode_maxdiff'] == 0
                 and e['truth_rx_equals_dataset_clean_index'] and e['noise_lock_equals_loader_pairing'] and in_cols
                 and e['target_vs_rescued_candidate_reencode_maxdiff'] <= 1e-4 and e['norm_maxdev'] <= 1e-5 and e['target_requires_grad'] is False
                 and e['target_vs_o1_target_maxdiff'] > 1e-3 and e['o1_target_is_sgP'] and e['loss_is_half_ref_plus_half_enc'] <= 1e-5
                 and e['enc_grad_vs_indep_sce_maxdiff'] <= 1e-5 * max(gnorm, 1e-12) + 1e-9 and e['enc_grad_maxdiff_vs_o1'] > 0
                 and e['dup_counters']['rows'] == e['rows'] and e['dup_counters']['dup_rows'] == dup_gate and e['dup_counters']['unique'] == len(c_np)
                 and abs(e['dup_counters']['dup_frac'] - dup_gate / e['rows']) < 1e-12
                 and e['rescue_counters'] == e['rescue_counters_o1'] and e['rescue_counters']['rescued'] > 0 and e['rescue_calls'] == 1)
    ok('t2i_true_target', e['pass'], **e)

    # ---- (3) O3 target off == O1 bitwise
    a, b = steps[O3 + '/off'], o1
    e = dict(helper_calls=len(a['tt_calls']), enc_grad_maxdiff=gdiff(a['enc'], b['enc']),
             ref_i2t_grad_maxdiff=gdiff(a['ref_i2t'], b['ref_i2t']), ref_t2i_grad_maxdiff=gdiff(a['ref_t2i'], b['ref_t2i']),
             param_maxdiff=float((a['params'] - b['params']).abs().max()), adam_maxdiff=float((a['adam'] - b['adam']).abs().max()),
             adam_len_equal=a['adam'].numel() == b['adam'].numel(), adam_step_equal=a['adam_step'] == b['adam_step'], rng_equal=rng_eq(a, b),
             dir_grad_maxdiff={d: gdiff(a['br'][d]['grads'], b['br'][d]['grads']) for d in ('i2t', 't2i')},
             dir_target_equal={d: bool(torch.equal(a['br'][d]['target'], b['br'][d]['target'])) for d in ('i2t', 't2i')})
    e['bitwise'] = (e['helper_calls'] == 0 and e['enc_grad_maxdiff'] == 0 and e['ref_i2t_grad_maxdiff'] == 0 and e['ref_t2i_grad_maxdiff'] == 0
                    and e['param_maxdiff'] == 0 and e['adam_maxdiff'] == 0 and e['adam_len_equal'] and e['adam_step_equal'] and e['rng_equal']
                    and all(v == 0 for v in e['dir_grad_maxdiff'].values()) and all(e['dir_target_equal'].values()))
    e['note'] = 'O3_TARGET is a gate-only switch; production never changes it (a missing target raises)'
    ok('o3_target_off_equiv', e['bitwise'], **e)

    # ---- (4) both branches
    bb, good = {}, True
    for arm in ('a1_iface', 'o1_rescue_native_t2i', O3):
        c = steps[arm]; s_ = c['stats']
        nz = lambda L: sum(1 for g in L if g is not None and float(g.abs().sum()) > 0)
        e = dict(opt_i2t=bool(opt.i2t), opt_t2i=bool(opt.t2i), i2t_n=s_['i2t_n'], t2i_n=s_['t2i_n'], enc_grad_nonzero=nz(c['enc']),
                 ref_i2t_with_grad=nz(c['ref_i2t']), ref_t2i_with_grad=nz(c['ref_t2i']), peer_params_with_grad=c['peer_with_grad'],
                 peer_queue_ids_same=c['peer_queue_same'], optimizer_steps=c['nstep'], refiner_forwards=c['ref_calls'],
                 refiner_forwards_a1=a1['ref_calls'],
                 stopgrad=all((v['Pd_req'] in (False, None)) and (v['target_req'] in (False, None)) and v['X_req'] and v['finite'] for v in c['br'].values()))
        good &= (e['opt_i2t'] and e['opt_t2i'] and e['i2t_n'] == 1 and e['t2i_n'] == 1 and e['enc_grad_nonzero'] > 0 and e['ref_i2t_with_grad'] > 0
                 and e['ref_t2i_with_grad'] > 0 and e['peer_params_with_grad'] == 0 and e['peer_queue_ids_same'] and e['optimizer_steps'] == 1
                 and e['refiner_forwards'] == e['refiner_forwards_a1'] and e['stopgrad'])
        bb[arm] = e
    ok('both_branches', good, **bb)

    # ---- (5) resume
    rs = {}
    a = one_step(O3, probe_grads=False); b = one_step(O3, probe_grads=False)
    rs['snapshot_twice'] = dict(param_maxdiff=float((a['params'] - b['params']).abs().max()), adam_maxdiff=float((a['adam'] - b['adam']).abs().max()), rng_equal=rng_eq(a, b))
    R.load_full_state(S0, opt, A, B)
    for m, prob, pred in ((A, prob_A, pred_A), (B, prob_B, pred_B)):
        m.avg_clean_prob = prob[pred].mean()
    for m in (A, B):
        for p in list(m.params) + list(m.i2t_refiner_node.parameters()) + list(m.t2i_refiner_node.parameters()):
            p.grad = None
    R.set_rng_state(snap[2]); R27._PEER.clear(); R27._PEER[id(A)] = 0; R27._PEER[id(B)] = 1
    c = one_step(O3, probe_grads=False, do_restore=False)
    rs['disk_reload'] = dict(param_maxdiff=float((a['params'] - c['params']).abs().max()), adam_maxdiff=float((a['adam'] - c['adam']).abs().max()),
                             rng_equal=rng_eq(a, c),
                             queue_ids_equal=all(torch.equal(a['own_ids'][k], c['own_ids'][k]) for k in a['own_ids']) and set(a['own_ids']) == set(c['own_ids']))
    dst = torch.load(f'{smoke}/{O3}/state_last.pt', map_location='cpu', weights_only=False)
    so = dst.get('opt')
    g_ = (lambda k: so.get(k)) if isinstance(so, dict) else (lambda k: getattr(so, k, None))
    rs['smoke_state'] = dict(saved_arm=g_('arm'), saved_rescue_flag=g_('r27o_rescue_t2i'), saved_o3_flag=g_('r27o3_true_target_t2i'),
                             policy_from_arm={bb_: R27.direction_kind(O3, bb_, R27.S_ARMS) for bb_ in ('i2t', 't2i')})
    del dst
    tmp = tempfile.mkdtemp(prefix='r27o3_gate_id_')
    sys.path.insert(0, os.path.dirname(RUNNER)); import run_unit_o3 as RU
    base = [sys.executable, '-B', RUNNER, '--cells', CELLS, '--lambda-file', LAMF, '--code-dir', code_dir, '--data-path', dpath, '--output-root', tmp]

    def ru(unit):
        p = subprocess.run(base + ['--unit', unit], capture_output=True, text=True)
        return p.returncode, (p.stdout + p.stderr).strip()[-400:]
    U40, U60 = f'f40_s1001__{O3}', f'f60_s1001__{O3}'
    code0, out0 = ru(U40)
    ident40 = RU.identity_for(U40, CELLS, LAMF, code_dir)[-1]
    ident60 = RU.identity_for(U60, CELLS, LAMF, code_dir)[-1]
    os.makedirs(f'{tmp}/f40_s1001/{O3}')
    json.dump(ident40, open(f'{tmp}/f40_s1001/{O3}/R27O3_IDENTITY.json', 'w'), indent=2)
    open(f'{tmp}/f40_s1001/{O3}/state_last.pt', 'w').close()
    code1, out1 = ru(U40)                                           # same unit: resumes its own state_last
    os.makedirs(f'{tmp}/f60_s1001/{O3}')
    json.dump(ident40, open(f'{tmp}/f60_s1001/{O3}/R27O3_IDENTITY.json', 'w'), indent=2)   # foreign identity (other cell)
    code2, out2 = ru(U60)
    os.remove(f'{tmp}/f60_s1001/{O3}/R27O3_IDENTITY.json'); open(f'{tmp}/f60_s1001/{O3}/state_last.pt', 'w').close()
    code3, out3 = ru(U60)                                           # unidentified state
    code4, out4 = ru('f40_s1001__o1_rescue_native_t2i')             # not in the O3 matrix
    shutil.rmtree(tmp)
    rs['run_unit_identity'] = dict(dry_fresh_rc=code0, dry_fresh_tail=out0[-300:], same_unit_rc=code1, same_unit_resumes_state_last='state_last.pt' in out1,
                                   foreign_identity_rc=code2, foreign_identity_msg=out2[-200:], unidentified_state_rc=code3, unidentified_msg=out3[-200:],
                                   non_o3_unit_rc=code4, non_o3_msg=out4[-200:], identities_differ=ident40 != ident60)
    good = (rs['snapshot_twice']['param_maxdiff'] == 0 and rs['snapshot_twice']['adam_maxdiff'] == 0 and rs['snapshot_twice']['rng_equal']
            and rs['disk_reload']['param_maxdiff'] <= TOL and rs['disk_reload']['adam_maxdiff'] <= TOL and rs['disk_reload']['rng_equal']
            and rs['disk_reload']['queue_ids_equal']
            and rs['smoke_state']['saved_arm'] == O3 and rs['smoke_state']['saved_rescue_flag'] is True and rs['smoke_state']['saved_o3_flag'] is True
            and rs['smoke_state']['policy_from_arm'] == RX.O_ARMS[O3]
            and code0 == 0 and code1 == 0 and rs['run_unit_identity']['same_unit_resumes_state_last'] and code2 != 0 and 'identity mismatch' in out2
            and code3 != 0 and 'Unidentified' in out3 and code4 != 0 and ident40 != ident60)
    ok('resume', good, **rs)

    # ---- (6) truth isolation (static)
    srcs = {n: inspect.getsource(f) for n, f in (('s_branch', R27.s_branch), ('noisy_branches', orig_nb),
                                                 ('search', I.search), ('reencode', I.reencode))}
    srcs['r27s_targets'] = inspect.getsource(ST); srcs['r27d_policy'] = inspect.getsource(P)
    words = ('clean_labels', 'clean_label', 'gt_index', 'true_index', 'noise_file', '_t2i_index', '_labels', 'IM_DIV', 'true_image', '// 5', '//5')
    bad = {n: [w for w in words if w in s] for n, s in srcs.items()}
    hook_lines = [l.strip() for l in srcs['noisy_branches'].splitlines() if 'RX.' in l]
    rx_src = inspect.getsource(RX)
    tt_src = inspect.getsource(orig_tt)
    good = (not any(bad.values()) and hook_lines == [
        "if branch == 't2i' and opt.arm in O_ARMS and RX.ENABLED:",
        "S, _ = RX.rescue_t2i(S, uids[0], opt.k, opt.r27_epoch, stats)   # uids[0] = global caption IDs of the query batch",
        "if branch == 't2i' and opt.arm in RX.O3_ARMS and RX.O3_TARGET:",
        "oT, _ = RX.true_target_t2i(net, uids[0], mi, stats)   # R27-O3: truth read inside the oracle module only"]
            and 'IM_DIV' in rx_src and 'true_image(' in tt_src and 'reencode' in tt_src and RX.O3_ARMS == (O3,)
            and opt.data_name == 'f30k_precomp' and RX.ENABLED is True and RX.O3_TARGET is True)
    ok('truth_isolation', good, source_term_hits=bad, hook_lines=hook_lines,
       truth_reader='r27o_rescue.true_image (caption j -> image j // 5), used by rescue_t2i and true_target_t2i only',
       split='train (FIT) only; smoke DEV for arms_ran only; TEST never loaded')

    for hk in hooks:
        hk.remove()
    RX.rescue_t2i = orig_rx
    RX.true_target_t2i = orig_tt
    R27.noisy_branches = orig_nb

    # ---- (7) timing and memory
    tim = {}
    for arm in ('a1_iface', 'o1_rescue_native_t2i', O3):
        restore(); opt.arm, opt.lam = arm, (0.0 if arm == 'a1_iface' else LAM)
        s = R.new_stats()
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
        R27.train(opt, A, B, lt, ut, ep, s, max_iters=30)
        torch.cuda.synchronize()
        tim[arm] = dict(ms_per_step=1000 * (time.time() - t0) / 30, peak_mem_GiB=torch.cuda.max_memory_allocated() / 2**30,
                        rescued_rows=s.get('cnt_t2i_rescued_rows', 0), rescue_rows=s.get('cnt_t2i_rescue_rows', 0),
                        o3_rows=s.get('cnt_t2i_o3_rows', 0), o3_dup_rows=s.get('cnt_t2i_o3_dup_rows', 0))
    for k in tim:
        tim[k]['ratio_vs_a1'] = tim[k]['ms_per_step'] / tim['a1_iface']['ms_per_step']
    ok('timing', True, gpu=torch.cuda.get_device_name(0), note='30 iterations incl. data loading; model A step only', **tim)

    # ---- (8) O3 smoke arm ran; counters logged
    ep_ = json.load(open(f'{smoke}/{O3}/EPOCHS.json'))['cotrain']
    last = ep_[-1]; tr = last['train']
    ran = {f'{p}/{b}': tr[p].get(f'{b}_n') or 0 for p in ('A', 'B') for b in ('i2t', 't2i')}
    fin = all(np.isfinite(tr[p].get(f'{b}_raw_loss_mean', np.nan)) and np.isfinite(tr[p].get(f'{b}_enc_loss_mean', np.nan))
              for p in ('A', 'B') for b in ('i2t', 't2i') if ran[f'{p}/{b}'] > 0)
    both_dirs = all(any(ran[f'{p}/{b}'] > 0 for p in ('A', 'B')) for b in ('i2t', 't2i'))
    keys = ('cnt_t2i_rescue_rows', 'cnt_t2i_truth_present_rows', 'cnt_t2i_rescued_rows', 'cnt_t2i_o3_rows', 'cnt_t2i_o3_dup_rows',
            'cnt_t2i_o3_unique_targets', 't2i_o3_dup_frac_mean')
    cnt = {p: {k: tr[p].get(k) for k in keys} for p in ('A', 'B')}
    t2i_peers = [p for p in ('A', 'B') if ran[f'{p}/t2i'] > 0]
    logged = bool(t2i_peers) and all(cnt[p]['cnt_t2i_rescue_rows'] and cnt[p]['cnt_t2i_o3_rows'] and cnt[p]['t2i_o3_dup_frac_mean'] is not None for p in t2i_peers)
    cons = all((cnt[p]['cnt_t2i_truth_present_rows'] or 0) + (cnt[p]['cnt_t2i_rescued_rows'] or 0) == (cnt[p]['cnt_t2i_rescue_rows'] or 0)
               and (cnt[p]['cnt_t2i_o3_rows'] or 0) == (cnt[p]['cnt_t2i_rescue_rows'] or 0) for p in ('A', 'B'))
    no_proto = {f'{p}/{b}': tr[p].get(f'{b}_no_proto') for p in ('A', 'B') for b in ('i2t', 't2i')}
    ok('arms_ran', fin and both_dirs and logged and cons, finite=fin and both_dirs, counters=cnt, counters_logged=logged, counters_consistent=cons,
       noisy_steps=ran, no_proto=no_proto, epochs=[x['epoch'] for x in ep_], dev_rsum=last['dev']['rsum'],
       note='1-epoch smoke from the common prefix: peer A has no ID-bearing candidates (known), so only peer B runs t2i')

    checks = {k: v for k, v in res.items() if isinstance(v, dict)}
    out = {'all_pass': all(v['pass_'] for v in checks.values()) and code_now == manifest['files_sha256'],
           'oracle_diagnostic': 'R27-O3 uses injected-pairing truth for the t2i candidate rescue AND the t2i encoder target; never part of a SOTA table',
           'code_files_sha256': code_now, 'code_matches_install_manifest': code_now == manifest['files_sha256'],
           'checks': checks, 'gpu': torch.cuda.get_device_name(0), 'job': os.environ.get('SLURM_JOB_ID'),
           'source_state': S0, 'source_state_sha256': fsha(S0), 'gate_script_sha256': fsha(os.path.abspath(__file__)),
           'runner_sha256': fsha(RUNNER), 'raw_report_paths': [f'{smoke}/GPU_GATE.json', f'{smoke}/{O3}']}
    json.dump(out, open(f'{smoke}/GPU_GATE.json', 'w'), indent=1, default=float)
    print('ALL_PASS', out['all_pass'], flush=True)
    sys.exit(0 if out['all_pass'] else 1)


if __name__ == '__main__':
    main()
