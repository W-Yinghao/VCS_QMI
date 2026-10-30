"""R27-S real-data GPU gate (DESIGN_S_CN.md §5.2). Usage: r27s_gate.py SMOKE_DIR NOISE_FILE DATA_PATH -> SMOKE_DIR/GATE_S.json

Runs after the six smoke arms (real f40_s1001 prefix, 2 co-train epochs x 30 iterations each). Starts from the
s4_single_cap_vcs smoke state_last.pt (both queues partly ID-bearing). Technical checks only:
  a1_identity  from the same restored state, one real training step of A1 and of the gate-only arm s0_identity (T = sg P):
               every encoder and refiner gradient at optimizer.step, parameters after the step, AdamW state and RNG
               (torch + cuda) must match A1 (bitwise reported; pass at max abs diff <= 1e-6, cudnn.benchmark host)
  grad_paths   one real step per S arm: encoder + current-peer refiner get gradients, other peer none, other-peer
               queue/IDs unchanged, targets / P_detached carry no grad, finite losses; cap arms: p.T >= cos15 - 1e-6,
               solver rel KKT <= 1e-6, no cap hits, no fallback
  solver_real  real graph (B x M) with the real refiner prototypes: S4/S5/S6 production SPG vs the independent full-
               dimensional reference audit_constrained.solve_constrained (tight); certified gap <= 2*res; S3 closed form
               KKT residual
  timing       ms/step over the same 30 iterations from the same state: A1 and S1-S6
  arms_ran     every smoke arm finished 2 co-train epochs with finite losses
"""
import os, sys, json, copy, hashlib, time
import numpy as np
import torch

sys.path.insert(1, '/home/infres/yinwang/CS_QMI/handoff/r27_audit_20261029')
import r25_run
import r25_co_train as R
import r27_co_train as R27
import r27_interface as I
import r27_targets as T
import r27s_targets as ST
import audit_constrained as AC
from data import get_dataset
from vse import VSE
from vocab import deserialize_vocab
from co_train import eval_train, split_prob

res = {}
S_RUN = ['s1_balanced_dual', 's2_single_free_vcs', 's3_single_cap_delta', 's4_single_cap_vcs', 's5_single_cap_bce', 's6_single_cap_snce']
LAM = 1.1448122442850819


def ok(name, cond, **info):
    res[name] = dict(pass_=bool(cond), **info)
    print(name, 'PASS' if cond else 'FAIL', json.dumps(info, default=float)[:3000], flush=True)


def h(t):
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()


def main():
    smoke, noise, dpath = sys.argv[1], sys.argv[2], sys.argv[3]
    S0 = f'{smoke}/s4_single_cap_vcs/state_last.pt'
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

    def restore():
        for m, s in ((A, snap[0]), (B, snap[1])):
            m.img_enc.load_state_dict(s['img_enc']); m.txt_enc.load_state_dict(s['txt_enc'])
            m.optimizer.load_state_dict(copy.deepcopy(s['optimizer']))  # deepcopy: load_state_dict aliases same-device tensors, step() would mutate the snapshot
            m.i2t_refiner_node.load_state_dict(s['i2t_refiner']); m.t2i_refiner_node.load_state_dict(s['t2i_refiner'])
            q = m.memory_queue; q.img_queue.copy_(s['queue']['img']); q.txt_queue.copy_(s['queue']['txt']); q.ptr, q.is_full = s['queue']['ptr'], s['queue']['is_full']
            I.load_ids(q, s['r27_queue_ids'])
            for p in list(m.params) + list(m.i2t_refiner_node.parameters()) + list(m.t2i_refiner_node.parameters()):
                p.grad = None
        R.set_rng_state(snap[2])
        R27._PEER.clear(); R27._PEER[id(A)] = 0; R27._PEER[id(B)] = 1

    own_ref = lambda: list(A.i2t_refiner_node.parameters()) + list(A.t2i_refiner_node.parameters())

    def one_step(arm, lam, probe=None):
        restore()
        opt.arm, opt.lam = arm, lam
        cap = {}
        orig = A.optimizer.step

        def step_probe(*a, **k):
            cap['enc'] = [None if p.grad is None else p.grad.detach().clone() for p in A.params]
            cap['ref'] = [None if p.grad is None else p.grad.detach().clone() for p in own_ref()]
            cap['peer_with_grad'] = sum(1 for p in list(B.params) + list(B.i2t_refiner_node.parameters()) + list(B.t2i_refiner_node.parameters()) if p.grad is not None)
            out = orig(*a, **k)
            cap['params'] = torch.cat([p.detach().flatten().double() for p in list(A.params) + own_ref()])
            cap['adam'] = torch.cat([v.detach().flatten().double() for p in list(A.params) + own_ref() for kk, v in A.optimizer.state.get(p, {}).items() if torch.is_tensor(v) and v.dim() > 0])
            return out
        A.optimizer.step = step_probe
        stats = R.new_stats()
        R27.train(opt, A, B, lt, ut, ep, stats, max_iters=1, probe=probe)
        A.optimizer.step = orig
        cap['rng'] = R.rng_state(); cap['stats'] = stats
        return cap

    def gdiff(a, b):
        m = 0.0
        for x, y in zip(a, b):
            if (x is None) != (y is None):
                return float('inf')
            if x is not None:
                m = max(m, float((x - y).abs().max()))
        return m

    # ---- A1 identity (gradients, step, optimizer state, RNG)
    a1 = one_step('a1_iface', 0.0)
    s0 = one_step('s0_identity', LAM)
    d = dict(enc_grad_maxdiff=gdiff(a1['enc'], s0['enc']), refiner_grad_maxdiff=gdiff(a1['ref'], s0['ref']),
             refiner_grad_none_pattern_equal=[x is None for x in a1['ref']] == [x is None for x in s0['ref']],
             param_after_step_maxdiff=float((a1['params'] - s0['params']).abs().max()),
             adam_state_maxdiff=float((a1['adam'] - s0['adam']).abs().max()),
             rng_torch_equal=bool(torch.equal(a1['rng']['torch'], s0['rng']['torch'])),
             rng_cuda_equal=all(torch.equal(x, y) for x, y in zip(a1['rng']['cuda'], s0['rng']['cuda'])),
             noisy_steps=[a1['stats']['i2t_n'], s0['stats']['i2t_n'], a1['stats']['t2i_n'], s0['stats']['t2i_n']])
    d['bitwise_all'] = d['enc_grad_maxdiff'] == 0 and d['refiner_grad_maxdiff'] == 0 and d['param_after_step_maxdiff'] == 0
    okid = (d['enc_grad_maxdiff'] <= 1e-6 and d['refiner_grad_maxdiff'] <= 1e-6 and d['refiner_grad_none_pattern_equal']
            and d['param_after_step_maxdiff'] <= 1e-6 and d['adam_state_maxdiff'] <= 1e-6 and d['rng_torch_equal'] and d['rng_cuda_equal']
            and min(d['noisy_steps']) == 1)
    ok('a1_identity', okid, **d)

    # ---- grad paths per S arm
    info_all, good, real = {}, True, None
    for arm in S_RUN:
        qh = (h(B.memory_queue.img_queue), h(B.memory_queue.txt_queue))
        seen = {}

        def probe(branch, dd, arm=arm):
            tg = dd['target']
            e = dict(X_req=bool(dd['X'].requires_grad), Pd_req=bool(dd['Pd'].requires_grad), proto_req=bool(dd['proto'].requires_grad),
                     target_req=bool(tg.requires_grad), finite=bool(torch.isfinite(dd['loss'])), loss=float(dd['loss']))
            if 'cap' in arm:
                e['min_cos_to_p'] = float((tg * dd['Pd']).sum(1).min())
                e['cone_ok'] = e['min_cos_to_p'] >= ST.GAMMA - 1e-6
            seen[branch] = e
            nonlocal real
            if arm == 's4_single_cap_vcs' and branch == 'i2t':
                real = (dd['S']['A'].clone(), dd['Y'].clone(), dd['Pd'].clone())
        cap = one_step(arm, LAM, probe=probe)
        st_ = cap['stats']
        e = dict(branches=seen, enc_grad_nonzero=sum(1 for g in cap['enc'] if g is not None and float(g.abs().sum()) > 0),
                 own_refiner_with_grad=sum(1 for g in cap['ref'] if g is not None and float(g.abs().sum()) > 0),
                 peer_params_with_grad=cap['peer_with_grad'],
                 peer_queue_same=(h(B.memory_queue.img_queue), h(B.memory_queue.txt_queue)) == qh,
                 solver={k: st_.get(k) for k in ('max_rel_res_i2t', 'max_rel_res_t2i', 'cnt_i2t_cap_hits', 'cnt_t2i_cap_hits',
                                                 'cnt_i2t_fallback', 'cnt_t2i_fallback', 'max_iters_i2t', 'max_iters_t2i',
                                                 'max_violation_i2t', 'max_violation_t2i', 'i2t_cap_active_frac', 't2i_cap_active_frac')})
        g = (all(seen.get(b, {}).get('X_req') and not seen[b]['Pd_req'] and seen[b]['proto_req'] and not seen[b]['target_req'] and seen[b]['finite']
                 for b in ('i2t', 't2i')) and e['enc_grad_nonzero'] > 0 and e['own_refiner_with_grad'] > 0 and e['peer_params_with_grad'] == 0
             and e['peer_queue_same'] and st_['i2t_n'] == 1 and st_['t2i_n'] == 1)
        if True:
            g &= all((st_.get(f'max_rel_res_{b}') or 0) <= T.TOL_REL and not st_.get(f'cnt_{b}_cap_hits') for b in ('i2t', 't2i'))
        if 'cap' in arm:
            g &= all(seen[b]['cone_ok'] and not st_.get(f'cnt_{b}_fallback') for b in ('i2t', 't2i'))
        info_all[arm] = e
        good &= g
    ok('grad_paths', good, **info_all)

    # ---- solver accuracy on a real graph with real prototypes
    Ag, Yg, Pg = (x.double() for x in real)
    Pg = Pg / Pg.norm(dim=1, keepdim=True)
    c, mu, muQ, delta = T.moments(Ag, Yg)
    sol, good = {}, True
    Td, _, idl = ST.generate_cap('cap_delta', Ag, Yg, Pg)
    sol['cap_delta'] = dict(max_rel=float(idl['rel'].max()), cap_active=float(idl['cap_active'].double().mean()))
    good &= sol['cap_delta']['max_rel'] <= T.TOL_REL
    for k in ('cap_vcs', 'cap_bce', 'cap_snce'):
        gen = ST.SOLVERS[k]
        torch.cuda.synchronize(); t0 = time.time()
        Tg, _, inf = ST.generate_cap(k, Ag, Yg, Pg)
        torch.cuda.synchronize(); tt = time.time() - t0
        Z = Tg * inf['z_norm'].unsqueeze(1)
        Zr, ir = AC.solve_constrained(gen, Ag, Yg, Pg, ST.GAMMA, tol_rel=1e-9, cap=40000)
        f = lambda Z_: T.fT(gen, Z_ @ Yg.t(), Ag, c)
        gap = f(Zr) - f(Z)
        sol[k] = dict(seconds=tt, max_rel=float(inf['rel'].max()), max_iters=int(inf['iters'].max()), mean_iters=float(inf['iters'].double().mean()),
                      caps=int(inf['cap'].sum()), cap_active=float(inf['cap_active'].double().mean()), ball_active=float(inf['ball_active'].double().mean()),
                      ref_rel_max=float(ir['rel'].max()), gap_max=float(gap.max()), gap_excess=float((gap - 2 * inf['res']).max()),
                      max_angle_vs_ref=float(torch.arccos(((Z * Zr).sum(1) / (Z.norm(dim=1) * Zr.norm(dim=1))).clamp(-1, 1)).max()),
                      p_outside_spanY=inf['p_outside_span'])
        good &= sol[k]['gap_excess'] <= 1e-9 and sol[k]['caps'] == 0 and sol[k]['max_rel'] <= T.TOL_REL
    ok('solver_real', good, B=int(Ag.shape[0]), M=int(Ag.shape[1]), **sol)

    # ---- timing
    tim = {}
    for arm in ['a1_iface'] + S_RUN:
        restore(); opt.arm, opt.lam = arm, (0.0 if arm == 'a1_iface' else LAM)
        s = R.new_stats()
        torch.cuda.synchronize(); t0 = time.time()
        R27.train(opt, A, B, lt, ut, ep, s, max_iters=30)
        torch.cuda.synchronize()
        tim[arm] = dict(ms_per_step=1000 * (time.time() - t0) / 30)
    for k in tim:
        tim[k]['ratio_vs_a1'] = tim[k]['ms_per_step'] / tim['a1_iface']['ms_per_step']
    ok('timing', True, gpu=torch.cuda.get_device_name(0), note='30 iterations incl. data loading; model A step only', **tim)

    # ---- arms ran
    arms_ok = {}
    for arm in S_RUN:
        e = json.load(open(f'{smoke}/{arm}/EPOCHS.json'))['cotrain']
        last = e[-1]
        fin = all(np.isfinite(last['train'][p].get(f'{b}_raw_loss_mean', np.nan)) and np.isfinite(last['train'][p].get(f'{b}_enc_loss_mean', np.nan))
                  for p in ('A', 'B') for b in ('i2t', 't2i'))
        arms_ok[arm] = dict(finite=fin, epochs=[x['epoch'] for x in e], dev_rsum=last['dev']['rsum'], last_A=last['train']['A'])
    ok('arms_ran', all(v['finite'] for v in arms_ok.values()), **arms_ok)

    res['all_pass'] = all(v['pass_'] for v in res.values() if isinstance(v, dict))
    json.dump(res, open(f'{smoke}/GATE_S.json', 'w'), indent=1, default=float)
    sys.exit(0 if res['all_pass'] else 1)


if __name__ == '__main__':
    main()
