"""R27-CS real-data GPU gate (t2i candidate-source units), modeled on R27-O gate/r27o_gate.py. Run from code_cs (cwd).
Usage: r27cs_gate.py SMOKE_DIR NOISE_FILE DATA_PATH GATE_STATE CODE_DIR PKG_DIR -> SMOKE_DIR/GPU_GATE.json

From one restored real R27-S smoke state (both queues partly ID-bearing): one real training step of model A per arm with a
probe taking the per-direction gradient of that direction's loss w.r.t. the encoder + current-peer refiners
(torch.autograd.grad, retain_graph; the production backward is untouched). Arms: a1_iface, bn_cross_t2i, cn_mix_t2i with
the new t2i source ON, and bn/cn with the source switched OFF (r27cs_sources.ENABLED = False, gate only).
  switch_off_equiv  bn/cn with the source OFF == A1 bitwise (full-step grads, params after the step, AdamW state, RNG,
                    per-direction probe grads and graphs; no CS call)
  i2t_identical     source ON: i2t direction (loss value, graph, encoder + refiner probe grads) == A1 bitwise; i2t refiner
                    step grads == A1; exactly one CS call (t2i)
  t2i_candidates    source ON: the query handed to the CS search == the production A1 t2i query (net2 val forward_emb text)
                    bitwise, bank = net2 queue (unchanged); IDs == independent recomputation with the PACKAGE functions
                    (candidate_sources.unique_topk / mix_3a_2b, loaded from the package) on features exported to CPU
                    and moved back; K=5 distinct IDs per row; cols[inv] == img; A == production 1/K scatter rule, rows
                    sum 1; img == queue img IDs of the representative slots; the refiner consumed exactly this graph;
                    A1 slot route deduplicated in order is a prefix of A_unique (same route); offline identity counters
                    (truth caption j -> image j // 5, evaluator only, in this gate, never in training) reported
  both_branches     both directions trained once, both own refiners get grads, other peer no grad and its queue/IDs
                    unchanged, one optimizer.step, refiner forwards equal to A1, stop-grad graph
  resume            snapshot restore twice bitwise; reload from disk; smoke checkpoint arm; runner identity
  truth_isolation   no truth tokens in the noisy-branch sources incl. r27cs_sources; train split only
  timing            ms/step (30 iterations, same state) and peak memory per arm
  arms_ran          the two smoke arms finished their smoke epoch with finite losses; CS counters logged
"""
import os, sys, json, copy, hashlib, time, inspect, subprocess, tempfile, shutil, re, importlib.util
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(sys.argv[5]))
import r25_run
import r25_co_train as R
import r27_co_train as R27
import r27_interface as I
import r27cs_sources as CS
from data import get_dataset
from vse import VSE
from vocab import deserialize_vocab
from co_train import eval_train, split_prob

res = {}
CS_RUN = ['bn_cross_t2i', 'cn_mix_t2i']
TOL = 1e-6
K = 5
HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(os.path.dirname(HERE), 'tools', 'run_unit_cs.py')
CELLS = '/home/infres/yinwang/CS_QMI/handoff/r27_qtarget_20261029/CELLS.json'


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
    smoke, noise, dpath, S0, code_dir, pkg = sys.argv[1:7]
    spec = importlib.util.spec_from_file_location('pkg_candidate_sources', os.path.join(pkg, 'src', 'candidate_sources.py'))
    PK = importlib.util.module_from_spec(spec); spec.loader.exec_module(PK)
    pkg_sha = fsha(os.path.join(pkg, 'src', 'candidate_sources.py'))
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

    # recorders (gate only): CS search calls, production search calls, query-batch uids
    cs_rec, prod_rec, uids_rec = [], [], []
    orig_cs = CS.search_t2i

    def rec_cs(q2, query2, mode, k, stats=None):
        n, valid = I.usable(q2)
        S, qs = orig_cs(q2, query2, mode, k, stats)
        cs_rec.append(dict(query=query2.detach().clone(), mode=mode, n=n, valid=valid.clone(),
                           img_queue=q2.img_queue[:n].detach().clone(), txt_queue=q2.txt_queue[:n].detach().clone(),
                           ids={kk: v[:n].clone() for kk, v in q2.r27_ids.items()}, q2_is_peer_queue=q2 is B.memory_queue,
                           S=None if S is None else {kk: (v.clone() if torch.is_tensor(v) else v) for kk, v in S.items()}, qs=qs))
        return S, qs
    CS.search_t2i = rec_cs
    orig_search = R27.search

    def rec_search(q2, query2, branch, k):
        S, qs = orig_search(q2, query2, branch, k)
        prod_rec.append(dict(branch=branch, query=query2.detach().clone(),
                             S=None if S is None else {kk: v.clone() for kk, v in S.items()}))
        return S, qs
    R27.search = rec_search
    orig_nb = R27.noisy_branches

    def rec_nb(opt_, net, net2, *a, **k):
        uids_rec.append(tuple(u.clone() for u in a[-2]))      # positional: ..., mi, mt, stats, uids, timed
        return orig_nb(opt_, net, net2, *a, **k)
    R27.noisy_branches = rec_nb

    ref_calls = [0]
    hooks = [m.register_forward_hook(lambda *a: ref_calls.__setitem__(0, ref_calls[0] + 1))
             for m in (A.i2t_refiner_node, A.t2i_refiner_node)]

    def one_step(arm, enabled=True, probe_grads=True, do_restore=True):
        if do_restore:
            restore()
        opt.arm, opt.lam = arm, 0.0
        CS.ENABLED = enabled
        cap, br = {'nstep': 0}, {}
        ref_calls[0] = 0; cs_rec.clear(); prod_rec.clear(); uids_rec.clear()
        params = list(A.params) + own_ref()

        def probe(branch, dd):
            e = dict(loss=float(dd['loss']), l_nat=float(dd['l_nat']), graphA=dd['S']['A'].detach().clone(),
                     cols=dd['S']['cols'].detach().clone(), img=dd['S']['img'].detach().clone(), inv=dd['S']['inv'].detach().clone(),
                     Y_req=bool(dd['Y'].requires_grad), neigh_req=bool(dd['neigh'].requires_grad),
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
            CS.ENABLED = True
        cap.update(rng=R.rng_state(), stats=dict(stats), br=br, ref_calls=ref_calls[0], cs_calls=list(cs_rec), prod_calls=list(prod_rec),
                   uids=list(uids_rec),
                   peer_queue_same=(h(B.memory_queue.img_queue), h(B.memory_queue.txt_queue), {k: h(v) for k, v in ids_of(B.memory_queue).items()}) == qh,
                   own_ids={k: v.clone() for k, v in ids_of(A.memory_queue).items()})
        return cap

    def rng_eq(a, b):
        na, nb_ = a['rng']['numpy'], b['rng']['numpy']
        return (bool(torch.equal(a['rng']['torch'], b['rng']['torch'])) and all(torch.equal(x, y) for x, y in zip(a['rng']['cuda'], b['rng']['cuda']))
                and na[0] == nb_[0] and np.array_equal(na[1], nb_[1]) and tuple(na[2:]) == tuple(nb_[2:]) and a['rng']['python'] == b['rng']['python'])

    steps = {'a1_iface': one_step('a1_iface')}
    for arm in CS_RUN:
        steps[arm] = one_step(arm, enabled=True)
        steps[arm + '/off'] = one_step(arm, enabled=False)

    # ---- (0) source OFF == A1, bitwise, full step
    r0, good = {}, True
    for arm in CS_RUN:
        a, b = steps[arm + '/off'], steps['a1_iface']
        e = dict(cs_calls=len(a['cs_calls']), enc_grad_maxdiff=gdiff(a['enc'], b['enc']),
                 ref_i2t_grad_maxdiff=gdiff(a['ref_i2t'], b['ref_i2t']), ref_t2i_grad_maxdiff=gdiff(a['ref_t2i'], b['ref_t2i']),
                 param_maxdiff=float((a['params'] - b['params']).abs().max()), adam_maxdiff=float((a['adam'] - b['adam']).abs().max()),
                 adam_len_equal=a['adam'].numel() == b['adam'].numel(), adam_step_equal=a['adam_step'] == b['adam_step'], rng_equal=rng_eq(a, b),
                 dir_grad_maxdiff={d: gdiff(a['br'][d]['grads'], b['br'][d]['grads']) for d in ('i2t', 't2i')},
                 dir_lnat_absdiff={d: abs(a['br'][d]['l_nat'] - b['br'][d]['l_nat']) for d in ('i2t', 't2i')},
                 graph_equal={d: bool(torch.equal(a['br'][d]['graphA'], b['br'][d]['graphA']) and torch.equal(a['br'][d]['cols'], b['br'][d]['cols'])) for d in ('i2t', 't2i')})
        e['bitwise'] = (e['cs_calls'] == 0 and e['enc_grad_maxdiff'] == 0 and e['ref_i2t_grad_maxdiff'] == 0 and e['ref_t2i_grad_maxdiff'] == 0
                        and e['param_maxdiff'] == 0 and e['adam_maxdiff'] == 0 and e['adam_len_equal'] and e['adam_step_equal'] and e['rng_equal']
                        and all(v == 0 for v in e['dir_grad_maxdiff'].values()) and all(v == 0 for v in e['dir_lnat_absdiff'].values())
                        and all(e['graph_equal'].values()))
        good &= e['bitwise']
        r0[arm] = e
    ok('switch_off_equiv', good, **r0)

    # ---- (1) source ON: i2t identical to A1
    r1, good = {}, True
    for arm in CS_RUN:
        a, b = steps[arm], steps['a1_iface']
        ai, bi = a['br']['i2t'], b['br']['i2t']
        e = dict(grad_maxdiff=gdiff(ai['grads'], bi['grads']), l_nat_absdiff=abs(ai['l_nat'] - bi['l_nat']), loss_absdiff=abs(ai['loss'] - bi['loss']),
                 graph_equal=bool(torch.equal(ai['graphA'], bi['graphA']) and torch.equal(ai['cols'], bi['cols']) and torch.equal(ai['img'], bi['img'])),
                 none_pattern_equal=[x is None for x in ai['grads']] == [x is None for x in bi['grads']],
                 step_ref_i2t_grad_maxdiff=gdiff(a['ref_i2t'], b['ref_i2t']),
                 cs_calls=len(a['cs_calls']), prod_calls=[c['branch'] for c in a['prod_calls']],
                 t2i_graph_differs_from_a1=not bool(torch.equal(a['br']['t2i']['img'], b['br']['t2i']['img'])))
        e['bitwise'] = (e['grad_maxdiff'] == 0 and e['l_nat_absdiff'] == 0 and e['loss_absdiff'] == 0 and e['graph_equal']
                        and e['none_pattern_equal'] and e['step_ref_i2t_grad_maxdiff'] == 0)
        good &= e['bitwise'] and e['cs_calls'] == 1 and e['prod_calls'] == ['i2t']
        r1[arm] = e
    ok('i2t_identical', good, **r1)

    # ---- (2,3) t2i candidates == package recomputation on exported features; K=5 unique; A rule
    r2, good = {}, True
    a1_t2i = [c for c in steps['a1_iface']['prod_calls'] if c['branch'] == 't2i'][0]
    for arm in CS_RUN:
        c = steps[arm]; rc = c['cs_calls'][0]; S = rc['S']
        mode = CS.CS_ARMS[arm]
        # export (as the offline audit): eligible paired slots, CPU numpy features, then recompute with the package code
        n = rc['n']; elig = (rc['ids']['cap'] >= 0) & (rc['ids']['img'] >= 0)
        ex = dict(query_txt=rc['query'].float().cpu().numpy(), bank_img=rc['img_queue'][elig].float().cpu().numpy(),
                  bank_txt=rc['txt_queue'][elig].float().cpu().numpy(), bank_observed_img_ids=rc['ids']['img'][elig].cpu().numpy())
        dev = 'cuda'; ten = lambda kk: torch.as_tensor(ex[kk], device=dev)
        pa, _, _ = PK.unique_topk(ten('query_txt'), ten('bank_txt'), ex['bank_observed_img_ids'], K)
        pb, _, _ = PK.unique_topk(ten('query_txt'), ten('bank_img'), ex['bank_observed_img_ids'], K)
        pa, pb = pa.cpu().numpy(), pb.cpu().numpy()
        pkg_final = pb if mode == 'B_direct' else PK.mix_3a_2b(pa, pb)
        # CPU float32 package run (info: hardware/kernel near-tie robustness)
        ca, _, _ = PK.unique_topk(torch.as_tensor(ex['query_txt']), torch.as_tensor(ex['bank_txt']), ex['bank_observed_img_ids'], K)
        cb, _, _ = PK.unique_topk(torch.as_tensor(ex['query_txt']), torch.as_tensor(ex['bank_img']), ex['bank_observed_img_ids'], K)
        cpu_final = cb.numpy() if mode == 'B_direct' else PK.mix_3a_2b(ca.numpy(), cb.numpy())
        img = S['img']; Bn = img.shape[0]
        Aref = torch.zeros_like(S['A']); Aref.scatter_add_(1, S['inv'], torch.full_like(S['inv'], 1.0 / K, dtype=torch.float32))
        srt = img.sort(1).values
        consumed = bool(torch.equal(c['br']['t2i']['graphA'], S['A']) and torch.equal(c['br']['t2i']['cols'], S['cols'])
                        and torch.equal(c['br']['t2i']['img'], img) and torch.equal(c['br']['t2i']['inv'], S['inv']))
        # A1 production slot route, distinct IDs in slot order, must be a prefix of A_unique (same text-bank route)
        a1img = a1_t2i['S']['img'].cpu().numpy()
        pref = []
        for r in range(Bn):
            d = list(dict.fromkeys(int(x) for x in a1img[r]))
            pref.append(list(pa[r, :len(d)]) == d)
        # offline identity counters (evaluator only; truth caption j -> image j // 5; never fed back)
        u0, u1 = c['uids'][0]
        capg = u0.cpu().numpy(); tru = capg // 5
        hit = lambda P: int(sum(int(t) in set(map(int, p)) for p, t in zip(P, tru)))
        mrr = lambda P: float(np.mean([1.0 / (list(map(int, p)).index(int(t)) + 1) if int(t) in set(map(int, p)) else 0.0 for p, t in zip(P, tru)]))
        e = dict(mode=mode, rows=Bn, K=int(img.shape[1]), n_slots=n, eligible_slots=int(elig.sum()),
                 query_equals_production_t2i_query=bool(torch.equal(rc['query'], a1_t2i['query'])),
                 bank_is_other_peer_queue=rc['q2_is_peer_queue'],
                 ids_equal_package_gpu=bool(np.array_equal(img.cpu().numpy(), pkg_final)),
                 ids_equal_package_cpu_frac=float((cpu_final == img.cpu().numpy()).all(1).mean()),
                 k_distinct_all_rows=bool((srt[:, 1:] != srt[:, :-1]).all()) and int(img.shape[1]) == K and bool((img >= 0).all()),
                 cols_inv_img=bool(torch.equal(S['cols'][S['inv']], img)), A_rowsum_maxdev=float((S['A'].sum(1) - 1).abs().max()),
                 A_equals_rule=bool(torch.equal(Aref, S['A'])), A_row_nnz=sorted(set((S['A'] > 0).sum(1).tolist())),
                 img_equals_slot_ids=bool(torch.equal(rc['ids']['img'][S['slots']], img)),
                 refiner_consumed_cs_graph=consumed,
                 a1_slot_dedup_is_prefix_of_A_unique_frac=float(np.mean(pref)),
                 a1_slot_rows_with_duplicate_ids=int(sum(len(set(map(int, r))) < K for r in a1img)),
                 n_from_b_route=int(S['route_b'].sum()), short_rows=rc['qs']['short_rows'], bank_unique_ids=rc['qs']['bank_unique_ids'],
                 cs_stats={kk: v for kk, v in c['stats'].items() if kk.startswith('cnt_t2i_cs')},
                 offline_identity=dict(note='evaluator only, this gate batch, gate state (not A1 last)', rows=Bn,
                                       truth_from_noise_lock_consistent=bool((NOISE[capg] == u1.cpu().numpy()).all()),
                                       noisy_query_frac=float((u1.cpu().numpy() != tru).mean()),
                                       hit_A1_slot=hit(a1img), hit_A_unique=hit(pa), hit_B_direct=hit(pb), hit_C_mix=hit(PK.mix_3a_2b(pa, pb)),
                                       hit_this_arm=hit(img.cpu().numpy()), mrr5_B=mrr(pb), mrr5_C=mrr(PK.mix_3a_2b(pa, pb))))
        e['pass'] = (e['query_equals_production_t2i_query'] and e['bank_is_other_peer_queue'] and e['ids_equal_package_gpu']
                     and e['k_distinct_all_rows'] and e['cols_inv_img'] and e['A_rowsum_maxdev'] <= 1e-6 and e['A_equals_rule']
                     and e['A_row_nnz'] == [K] and e['img_equals_slot_ids'] and consumed and e['short_rows'] == 0
                     and e['a1_slot_dedup_is_prefix_of_A_unique_frac'] >= 0.95)
        good &= e['pass']
        r2[arm] = e
    ok('t2i_candidates', good, **r2)

    # ---- (4) both branches
    bb, good = {}, True
    for arm in ['a1_iface'] + CS_RUN:
        c = steps[arm]; s_ = c['stats']
        nz = lambda L: sum(1 for g in L if g is not None and float(g.abs().sum()) > 0)
        e = dict(opt_i2t=bool(opt.i2t), opt_t2i=bool(opt.t2i), i2t_n=s_.get('i2t_n', 0), t2i_n=s_.get('t2i_n', 0), enc_grad_nonzero=nz(c['enc']),
                 ref_i2t_with_grad=nz(c['ref_i2t']), ref_t2i_with_grad=nz(c['ref_t2i']), peer_params_with_grad=c['peer_with_grad'],
                 peer_queue_ids_same=c['peer_queue_same'], optimizer_steps=c['nstep'], refiner_forwards=c['ref_calls'],
                 refiner_forwards_a1=steps['a1_iface']['ref_calls'],
                 stopgrad_graph=all((not v['Y_req']) and (not v['neigh_req']) and v['X_req'] and v['finite'] and v['aux_none'] for v in c['br'].values()))
        good &= (e['opt_i2t'] and e['opt_t2i'] and e['i2t_n'] == 1 and e['t2i_n'] == 1 and e['enc_grad_nonzero'] > 0 and e['ref_i2t_with_grad'] > 0
                 and e['ref_t2i_with_grad'] > 0 and e['peer_params_with_grad'] == 0 and e['peer_queue_ids_same'] and e['optimizer_steps'] == 1
                 and e['refiner_forwards'] == e['refiner_forwards_a1'] and e['stopgrad_graph'])
        bb[arm] = e
    ok('both_branches', good, **bb)

    # ---- (5) resume
    rs = {}
    a = one_step('cn_mix_t2i', probe_grads=False); b = one_step('cn_mix_t2i', probe_grads=False)
    rs['snapshot_twice'] = dict(param_maxdiff=float((a['params'] - b['params']).abs().max()), adam_maxdiff=float((a['adam'] - b['adam']).abs().max()), rng_equal=rng_eq(a, b))
    R.load_full_state(S0, opt, A, B)
    for m, prob, pred in ((A, prob_A, pred_A), (B, prob_B, pred_B)):
        m.avg_clean_prob = prob[pred].mean()
    for m in (A, B):
        for p in list(m.params) + list(m.i2t_refiner_node.parameters()) + list(m.t2i_refiner_node.parameters()):
            p.grad = None
    R.set_rng_state(snap[2]); R27._PEER.clear(); R27._PEER[id(A)] = 0; R27._PEER[id(B)] = 1
    c = one_step('cn_mix_t2i', probe_grads=False, do_restore=False)
    rs['disk_reload'] = dict(param_maxdiff=float((a['params'] - c['params']).abs().max()), adam_maxdiff=float((a['adam'] - c['adam']).abs().max()),
                             rng_equal=rng_eq(a, c),
                             queue_ids_equal=all(torch.equal(a['own_ids'][k], c['own_ids'][k]) for k in a['own_ids']) and set(a['own_ids']) == set(c['own_ids']))
    saved = {}
    for arm in CS_RUN:
        dst = torch.load(f'{smoke}/{arm}/state_last.pt', map_location='cpu', weights_only=False)
        so = dst.get('opt')
        g_ = (lambda k_: so.get(k_) if isinstance(so, dict) else getattr(so, k_, None))
        saved[arm] = dict(saved_arm=g_('arm'), saved_lam=g_('lam'), saved_source=g_('r27cs_source'), has_queue_ids=dst['model_B'].get('r27_queue_ids') is not None)
    rs['smoke_states'] = saved
    tmp = tempfile.mkdtemp(prefix='r27cs_gate_id_')
    sys.path.insert(0, os.path.dirname(RUNNER)); import run_unit_cs as RU
    base = [sys.executable, '-B', RUNNER, '--cells', CELLS, '--code-dir', code_dir, '--data-path', dpath, '--output-root', tmp]

    def ru(unit):
        p = subprocess.run(base + ['--unit', unit], capture_output=True, text=True)
        return p.returncode, (p.stdout + p.stderr).strip()
    code0, out0 = ru('f40_s1001__bn_cross_t2i')
    lam_zero_in_cmd = '--arm bn_cross_t2i --lam 0.0 ' in out0      # full printed driver command (v1 checked a 400-char tail)
    ident = RU.identity_for('f40_s1001__bn_cross_t2i', CELLS, code_dir)[-1]
    for arm_dir in CS_RUN:
        os.makedirs(f'{tmp}/f40_s1001/{arm_dir}')
        json.dump(ident, open(f'{tmp}/f40_s1001/{arm_dir}/R27CS_IDENTITY.json', 'w'), indent=2)
        open(f'{tmp}/f40_s1001/{arm_dir}/state_last.pt', 'w').close()
    os.makedirs(f'{tmp}/f60_s1001/bn_cross_t2i'); open(f'{tmp}/f60_s1001/bn_cross_t2i/state_last.pt', 'w').close()
    code1, out1 = ru('f40_s1001__bn_cross_t2i')
    code2, out2 = ru('f40_s1001__cn_mix_t2i')
    code3, out3 = ru('f60_s1001__bn_cross_t2i')
    code4, out4 = ru('f40_s1001__a1_iface')
    shutil.rmtree(tmp)
    rs['run_unit_identity'] = dict(dry_fresh_rc=code0, dry_fresh_tail=out0[-300:], same_arm_rc=code1, same_arm_resumes_state_last='state_last.pt' in out1,
                                   other_arm_rc=code2, other_arm_msg=out2[-200:], unidentified_state_rc=code3, unidentified_msg=out3[-200:],
                                   non_cs_unit_rc=code4, non_cs_msg=out4[-200:], lam_zero_in_cmd=lam_zero_in_cmd, dry_fresh_cmd=out0[-1200:])
    good = (rs['snapshot_twice']['param_maxdiff'] == 0 and rs['snapshot_twice']['adam_maxdiff'] == 0 and rs['snapshot_twice']['rng_equal']
            and rs['disk_reload']['param_maxdiff'] <= TOL and rs['disk_reload']['adam_maxdiff'] <= TOL and rs['disk_reload']['rng_equal']
            and rs['disk_reload']['queue_ids_equal']
            and all(v['saved_arm'] == k and v['saved_lam'] == 0.0 and v['saved_source'] == CS.CS_ARMS[k] and v['has_queue_ids'] for k, v in saved.items())
            and code0 == 0 and lam_zero_in_cmd and code1 == 0 and rs['run_unit_identity']['same_arm_resumes_state_last']
            and code2 != 0 and 'identity mismatch' in out2 and code3 != 0 and 'Unidentified' in out3 and code4 != 0)
    ok('resume', good, **rs)

    # ---- truth isolation
    srcs = {n: inspect.getsource(f) for n, f in (('noisy_branches', orig_nb), ('search', orig_search), ('reencode', I.reencode))}
    srcs['r27cs_sources'] = inspect.getsource(CS)
    words = ('clean_labels', 'clean_label', 'gt_index', 'true_index', 'noise_file', '_t2i_index', '_labels', 'IM_DIV', 'true_image', '// 5', '//5', 'gt_', 'truth[')
    bad = {n: [w for w in words if w in s] + re.findall(r'(?i)\btest\b', s) for n, s in srcs.items()}
    hook_lines = [l.strip() for l in srcs['noisy_branches'].splitlines() if 'CS.' in l or 'CS_ARMS' in l]
    good = (not any(bad.values()) and opt.data_name == 'f30k_precomp' and hook_lines == [
        "if branch == 't2i' and opt.arm in CS_ARMS and CS.ENABLED:",
        "S, qs = CS.search_t2i(net2.memory_queue, xt2, CS_ARMS[opt.arm], opt.k, stats)"])
    ok('truth_isolation', good, source_term_hits=bad, hook_lines=hook_lines,
       note='truth used only inside this gate for offline identity counters; training code never reads it',
       split='train (FIT) only; smoke DEV for arms_ran only; TEST never loaded')

    for hk in hooks:
        hk.remove()
    CS.search_t2i = orig_cs
    R27.search = orig_search
    R27.noisy_branches = orig_nb

    # ---- timing and memory
    tim = {}
    for arm in ['a1_iface'] + CS_RUN:
        restore(); opt.arm, opt.lam = arm, 0.0
        s = R.new_stats()
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
        R27.train(opt, A, B, lt, ut, ep, s, max_iters=30)
        torch.cuda.synchronize()
        tim[arm] = dict(ms_per_step=1000 * (time.time() - t0) / 30, peak_mem_GiB=torch.cuda.max_memory_allocated() / 2**30,
                        cs_rows=s.get('cnt_t2i_cs_rows', 0), cs_short_rows=s.get('cnt_t2i_cs_short_rows', 0), t2i_n=s.get('t2i_n', 0),
                        time_search_t2i_s=s.get('time_search_t2i', 0.0) / max(s.get('time_n', 1), 1))
    for k in tim:
        tim[k]['ratio_vs_a1'] = tim[k]['ms_per_step'] / tim['a1_iface']['ms_per_step']
    ok('timing', True, gpu=torch.cuda.get_device_name(0), note='30 iterations incl. data loading; model A step only', **tim)

    # ---- arms ran (smoke)
    arms_ok = {}
    for arm in CS_RUN:
        e = json.load(open(f'{smoke}/{arm}/EPOCHS.json'))['cotrain']
        last = e[-1]; tr = last['train']
        ran = {f'{p}/{b}': tr[p].get(f'{b}_n') or 0 for p in ('A', 'B') for b in ('i2t', 't2i')}
        fin = all(np.isfinite(tr[p].get(f'{b}_raw_loss_mean', np.nan)) for p in ('A', 'B') for b in ('i2t', 't2i') if ran[f'{p}/{b}'] > 0)
        both_dirs = all(any(ran[f'{p}/{b}'] > 0 for p in ('A', 'B')) for b in ('i2t', 't2i'))
        cnt = {p: {k: v for k, v in tr[p].items() if k.startswith('cnt_t2i_cs')} for p in ('A', 'B')}
        logged = all(cnt[p].get('cnt_t2i_cs_rows') for p in ('A', 'B') if ran[f'{p}/t2i'] > 0) and any(ran[f'{p}/t2i'] > 0 for p in ('A', 'B'))
        no_proto = {f'{p}/{b}': tr[p].get(f'{b}_no_proto') for p in ('A', 'B') for b in ('i2t', 't2i')}
        arms_ok[arm] = dict(finite=fin and both_dirs, cs_counters=cnt, counters_logged=logged, noisy_steps=ran, no_proto=no_proto,
                            epochs=[x['epoch'] for x in e], dev_rsum=last['dev']['rsum'])
    ok('arms_ran', all(v['finite'] and v['counters_logged'] for v in arms_ok.values()), **arms_ok)

    checks = {k: v for k, v in res.items() if isinstance(v, dict)}
    out = {'all_pass': all(v['pass_'] for v in checks.values()) and code_now == manifest['files_sha256'],
           'code_files_sha256': code_now, 'code_matches_install_manifest': code_now == manifest['files_sha256'],
           'package_candidate_sources_sha256': pkg_sha, 'checks': checks, 'gpu': torch.cuda.get_device_name(0),
           'job': os.environ.get('SLURM_JOB_ID'), 'source_state': S0, 'source_state_sha256': fsha(S0),
           'gate_script_sha256': fsha(os.path.abspath(__file__)), 'runner_sha256': fsha(RUNNER),
           'raw_report_paths': [f'{smoke}/GPU_GATE.json'] + [f'{smoke}/{a}' for a in CS_RUN]}
    json.dump(out, open(f'{smoke}/GPU_GATE.json', 'w'), indent=1, default=float)
    print('ALL_PASS', out['all_pass'], flush=True)
    sys.exit(0 if out['all_pass'] else 1)


if __name__ == '__main__':
    main()
