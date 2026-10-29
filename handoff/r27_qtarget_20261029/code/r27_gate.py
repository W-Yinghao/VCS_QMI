"""R27 real-data GPU gate (DESIGN_CN.md §6.2). Usage: r27_gate.py SMOKE_DIR NOISE_FILE DATA_PATH. Writes SMOKE_DIR/GATE.json.

Runs after the smoke arms (each: real f40_s1001 prefix, 2 co-train epochs x 30 iterations, lam = 1 placeholder).
Starts from the a3_vcs smoke state_last.pt (both queues partly ID-bearing).
  forward_ctx       forward_emb_ctx == native VSE.forward_emb bitwise (same RNG state)
  grad_paths        one real step per arm: queries carry grad; re-encoded candidates / neighbours / targets do not;
                    current-peer encoders AND current-peer GraphRefiners get gradients (native path kept); other-peer
                    encoders/refiners get none; other-peer queue + ID buffer bitwise unchanged; A1 has no aux term,
                    A2-A5 do; losses finite
  current_peer      re-encoding the probe's atom IDs with the CURRENT peer (eval, same centering mean) reproduces the
                    candidates bitwise; the other peer's re-encoding differs; the centering mean equals the current
                    noisy batch's pre-centering mean
  truth_isolation   queue img IDs equal the OBSERVED (noisy) t2i_index for every ID-bearing slot, including injected
                    mismatches where it differs from the true index (truth read here only)
  state_roundtrip   state_last -> load -> save -> load: all tensors incl. queues, ID buffers, optimizer, RNG identical
  resume            first step after (load state) vs after (load, save, load): parameters within 1e-4 (host tolerance,
                    cudnn.benchmark non-deterministic)
  solver_real       real graphs: production solvers vs tighter independent ones (gap <= 2*KKT residual; angles)
  timing            ms per training step over the same 30 iterations from the same state: R25 native, R26 d_vcs, R27 arms
  arms_ran          every smoke arm finished 2 co-train epochs with finite losses; logs usable fractions/solver stats
"""
import os, sys, json, copy, hashlib, time
import numpy as np
import torch

import r25_run
import r25_co_train as R
import r27_co_train as R27
import r27_interface as I
import r27_targets as T
from data import get_dataset
from vse import VSE
from vocab import deserialize_vocab
from co_train import eval_train, split_prob

res = {}
ARMS = list(R27.ARMS)


def ok(name, cond, **info):
    res[name] = dict(pass_=bool(cond), **info)
    print(name, 'PASS' if cond else 'FAIL', json.dumps(info, default=float)[:3000], flush=True)


def h(t):
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()


def flat(m):
    return torch.cat([p.detach().flatten().double() for p in list(m.img_enc.parameters()) + list(m.txt_enc.parameters())])


def main():
    smoke, noise, dpath = sys.argv[1], sys.argv[2], sys.argv[3]
    S0 = f'{smoke}/a3_vcs/state_last.pt'
    opt = r25_run.parse(['--noise_file', noise, '--output_dir', f'{smoke}/gate', '--resume', S0, '--data_path', dpath, '--max_iters', '30'])
    opt.arm, opt.lam = 'a3_vcs', 1.0
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
            m.img_enc.load_state_dict(s['img_enc']); m.txt_enc.load_state_dict(s['txt_enc']); m.optimizer.load_state_dict(s['optimizer'])
            m.i2t_refiner_node.load_state_dict(s['i2t_refiner']); m.t2i_refiner_node.load_state_dict(s['t2i_refiner'])
            q = m.memory_queue; q.img_queue.copy_(s['queue']['img']); q.txt_queue.copy_(s['queue']['txt']); q.ptr, q.is_full = s['queue']['ptr'], s['queue']['is_full']
            I.load_ids(q, s['r27_queue_ids'])
        R.set_rng_state(snap[2])
        R27._PEER.clear(); R27._PEER[id(A)] = 0; R27._PEER[id(B)] = 1

    nA, vA = I.usable(A.memory_queue); nB, vB = I.usable(B.memory_queue)
    qinfo = dict(A_usable=int(vA.sum()), A_n=nA, B_usable=int(vB.sum()), B_n=nB)

    # ---- forward_ctx equivalence
    batch = next(iter(ut))
    A.train_start()
    rng = R.rng_state()
    a1, a2, _ = A.forward_emb(batch[0], batch[2], batch[3], imgae_lengths=batch[1])
    R.set_rng_state(rng)
    b1, b2, _, mi, mt = I.forward_emb_ctx(A, batch[0], batch[2], batch[3], imgae_lengths=batch[1])
    ok('forward_ctx', torch.equal(a1, b1) and torch.equal(a2, b2), img_equal=torch.equal(a1, b1), txt_equal=torch.equal(a2, b2))
    del a1, a2, b1, b2

    # ---- grad paths + current peer, one real step per arm
    arm_info, allgood, cp_info, cp_good = {}, True, {}, True
    for arm in ARMS:
        restore()
        opt.arm, opt.lam = arm, (0.0 if arm == 'a1_iface' else 1.0)
        for m in (A, B):
            for p in list(m.params) + list(m.i2t_refiner_node.parameters()) + list(m.t2i_refiner_node.parameters()):
                p.grad = None
        qh = (h(B.memory_queue.img_queue), h(B.memory_queue.txt_queue), tuple(h(v) for v in B.memory_queue.r27_ids.values()))
        seen = {}

        def probe(branch, d):
            Y, neigh, tgt = d['Y'], d['neigh'], d['target']
            S = d['S']
            mean = d['mean']
            # current-peer check: re-encode the same atom IDs now (A is in train mode inside the step)
            Ya = I.reencode(A, branch, S['cols'], mean)
            Yb = I.reencode(B, branch, S['cols'], mean)
            seen[branch] = dict(X_req=bool(d['X'].requires_grad), Y_req=bool(Y.requires_grad), neigh_req=bool(neigh.requires_grad),
                                target_req=(None if tgt is None else bool(tgt.requires_grad)), proto_req=bool(d['proto'].requires_grad),
                                aux=d['l_aux'] is not None, loss=float(d['loss']), finite=bool(torch.isfinite(d['loss'])),
                                reenc_current_equal=bool(torch.equal(Ya, Y)), reenc_current_maxdiff=float((Ya - Y).abs().max()),
                                reenc_other_maxdiff=float((Yb - Y).abs().max()),
                                mean_req=bool(mean.requires_grad), n_atoms=int(S['cols'].numel()),
                                cos_reenc_vs_other_cached=float((neigh * (B.memory_queue.txt_queue if branch == 'i2t' else B.memory_queue.img_queue)[S['slots']]).sum(-1).mean()),
                                queue_same=(h(B.memory_queue.img_queue), h(B.memory_queue.txt_queue), tuple(h(v) for v in B.memory_queue.r27_ids.values())) == qh)
            if branch == 'i2t' and arm == 'a3_vcs':
                seen['_graph'] = (S['A'].clone(), Y.clone())
        grads = {}
        orig = A.optimizer.step

        def step_probe(*a, **k):
            grads['enc'] = sum(float(p.grad.abs().sum()) for p in A.params if p.grad is not None)
            grads['own_refiner_params_with_grad'] = sum(1 for p in list(A.i2t_refiner_node.parameters()) + list(A.t2i_refiner_node.parameters()) if p.grad is not None and float(p.grad.abs().sum()) > 0)
            grads['own_refiner_params_total'] = len(list(A.i2t_refiner_node.parameters()) + list(A.t2i_refiner_node.parameters()))
            grads['peer_params_with_grad'] = sum(1 for p in list(B.params) + list(B.i2t_refiner_node.parameters()) + list(B.t2i_refiner_node.parameters()) if p.grad is not None)
            grads['peer_queue_same_after'] = (h(B.memory_queue.img_queue), h(B.memory_queue.txt_queue), tuple(h(v) for v in B.memory_queue.r27_ids.values())) == qh
            return orig(*a, **k)
        A.optimizer.step = step_probe
        stats = R.new_stats()
        R27.train(opt, A, B, lt, ut, ep, stats, max_iters=1, probe=probe)
        A.optimizer.step = orig
        aux_expected = arm != 'a1_iface'
        g = (all(seen.get(b, {}).get('X_req') and not seen[b]['Y_req'] and not seen[b]['neigh_req'] and seen[b]['proto_req']
                 and seen[b]['target_req'] in (None, False) and seen[b]['aux'] == aux_expected and seen[b]['finite'] and seen[b]['queue_same']
                 and not seen[b]['mean_req'] for b in ('i2t', 't2i'))
             and grads['enc'] > 0 and grads['own_refiner_params_with_grad'] > 0 and grads['peer_params_with_grad'] == 0
             and grads['peer_queue_same_after'] and stats['i2t_n'] == 1 and stats['t2i_n'] == 1)
        c = all(seen[b]['reenc_current_maxdiff'] < 1e-6 and seen[b]['reenc_other_maxdiff'] > 1e-3 for b in ('i2t', 't2i'))
        graph = seen.pop('_graph', None)
        if graph is not None:
            real_graph = graph
        arm_info[arm] = dict(branches=seen, **grads)
        cp_info[arm] = {b: dict(bitwise_equal=seen[b]['reenc_current_equal'], current_maxdiff=seen[b]['reenc_current_maxdiff'],
                                other_maxdiff=seen[b]['reenc_other_maxdiff']) for b in ('i2t', 't2i')}
        allgood &= g
        cp_good &= c
    ok('grad_paths', allgood, queues=qinfo, **arm_info)
    ok('current_peer', cp_good, note='mean = detached pre-centering modality mean of the same noisy batch (forward_ctx check)', **cp_info)

    # ---- truth isolation (truth read only here)
    good, nnoisy = True, 0
    full_obs = np.load(noise)               # observed (noisy) caption -> image index used by the trainer
    true_idx = np.arange(len(full_obs)) // 5
    for m in (A, B):
        n, v = I.usable(m.memory_queue)
        cap = m.memory_queue.r27_ids['cap'][:n][v].cpu().numpy(); img = m.memory_queue.r27_ids['img'][:n][v].cpu().numpy()
        good &= bool((img == full_obs[cap]).all())
        nnoisy += int((full_obs[cap] != true_idx[cap]).sum())
        good &= bool((img[full_obs[cap] != true_idx[cap]] != true_idx[cap][full_obs[cap] != true_idx[cap]]).all())
    ok('truth_isolation', good and nnoisy > 0, id_bearing_slots_with_injected_mismatch=nnoisy)

    # ---- state roundtrip incl. IDs
    A2 = VSE(opt, vocab.word2idx); B2 = VSE(opt, vocab.word2idx)
    st2 = R.load_full_state(S0, opt, A2, B2)
    rt = f'{opt.output_dir}/rt.pt'
    R.save_full_state(rt, opt, A2, B2, st2['epoch'], st2['best_rsum'], st2['all_loss'], st2['history'])
    s1 = torch.load(S0, weights_only=False); s2 = torch.load(rt, weights_only=False)
    bad = []
    for side in ('model_A', 'model_B'):
        for k in ('img_enc', 'txt_enc', 'i2t_refiner', 't2i_refiner'):
            for kk in s1[side][k]:
                if not torch.equal(s1[side][k][kk].cpu(), s2[side][k][kk].cpu()):
                    bad.append(f'{side}/{k}/{kk}')
        for kk in ('img', 'txt'):
            if not torch.equal(s1[side]['queue'][kk].cpu(), s2[side]['queue'][kk].cpu()):
                bad.append(f'{side}/queue/{kk}')
        for kk in I.KEYS:
            if not torch.equal(s1[side]['r27_queue_ids'][kk], s2[side]['r27_queue_ids'][kk]):
                bad.append(f'{side}/ids/{kk}')
        o1, o2 = s1[side]['optimizer']['state'], s2[side]['optimizer']['state']
        for pk in o1:
            for kk in o1[pk]:
                if torch.is_tensor(o1[pk][kk]) and not torch.equal(o1[pk][kk].cpu(), o2[pk][kk].cpu()):
                    bad.append(f'{side}/opt/{pk}/{kk}')
    for kk in ('torch', 'cuda'):
        a_, b_ = s1['rng'][kk], s2['rng'][kk]
        if (not torch.equal(a_, b_)) if torch.is_tensor(a_) else not all(torch.equal(x, y) for x, y in zip(a_, b_)):
            bad.append(f'rng/{kk}')
    ok('state_roundtrip', not bad, mismatches=bad[:10], ids_present=all('r27_queue_ids' in s1[s] for s in ('model_A', 'model_B')))
    del s1, s2

    # ---- resume: first step from (load) vs (load, save, load)
    outs = []
    for path in (S0, rt):
        A3 = VSE(opt, vocab.word2idx); B3 = VSE(opt, vocab.word2idx)
        R.load_full_state(path, opt, A3, B3)
        A3.avg_clean_prob, B3.avg_clean_prob = A.avg_clean_prob, B.avg_clean_prob
        R27._PEER.clear()
        opt.arm, opt.lam = 'a3_vcs', 1.0
        lt3, ut3 = R.get_loader(caps, ims, vocab, "train", opt.batch_size, 0, opt.noise_ratio, opt.noise_file, pred=pred_B, prob=prob_B)
        R27.train(opt, A3, B3, lt3, ut3, ep, R.new_stats(), max_iters=1)
        outs.append(flat(A3))
        del A3, B3
    pd = float((outs[0] - outs[1]).abs().max())
    ok('resume', pd < 1e-4, first_step_param_max_abs_diff=pd, note='host tolerance (R26 gate): cudnn.benchmark=True is not bitwise deterministic')
    os.remove(rt)

    # ---- solver accuracy on a real graph
    Ag, Yg = real_graph
    sol = {}
    good = True
    Zv, iv = T.solve_vcs(Ag.double(), Yg.double())
    Zs, is_ = T.solve_spg('vcs', Ag.double(), Yg.double(), tol_rel=1e-10, cap=20000)
    fv = T.f_vcs
    c = Ag.double().mean(0)
    gap = fv(Zs, Ag.double(), c, Yg.double()) - fv(Zv, Ag.double(), c, Yg.double())
    sol['vcs'] = dict(exact_max_rel=float(iv['rel'].max()), exact_newton_max=int(iv['iters'].max()), rank=iv['rank'],
                      spg_vs_exact_max_angle=float(torch.arccos((Zs * Zv).sum(1) / (Zs.norm(dim=1) * Zv.norm(dim=1))).clamp(max=1).nan_to_num(0).max()),
                      gap_excess=float((gap - 2 * iv['res']).max()), interior=int(iv['interior'].sum()), max_range_loss=float(iv['lost'].max()))
    good &= sol['vcs']['gap_excess'] <= 1e-9 and sol['vcs']['exact_max_rel'] <= T.TOL_REL
    for gname in ('bce', 'snce'):
        torch.cuda.synchronize(); t = time.time()
        Zp, ip = T.solve_spg(gname, Ag.double(), Yg.double())
        torch.cuda.synchronize(); tp = time.time() - t
        Zf, if_ = T.solve_fista(gname, Ag.double(), Yg.double(), tol_rel=1e-9, cap=40000)
        f = T.OBJ[gname][0]
        gap = f(Zf, Ag.double(), c, Yg.double()) - f(Zp, Ag.double(), c, Yg.double())
        cosang = ((Zp * Zf).sum(1) / (Zp.norm(dim=1) * Zf.norm(dim=1))).clamp(-1, 1)
        sol[gname] = dict(spg_max_rel=float(ip['rel'].max()), spg_max_iters=int(ip['iters'].max()), spg_mean_iters=float(ip['iters'].double().mean()),
                          spg_cap=int(ip['cap'].sum()), spg_seconds=tp, fista_tight_max_rel=float(if_['rel'].max()), fista_iters=int(if_['iters'].max()),
                          gap_max=float(gap.max()), gap_excess=float((gap - 2 * ip['res']).max()), max_angle_vs_tight=float(torch.arccos(cosang).max()),
                          interior=int(ip['interior'].sum()))
        good &= sol[gname]['gap_excess'] <= 1e-9 and sol[gname]['spg_cap'] == 0
    ok('solver_real', good, B=int(Ag.shape[0]), M=int(Ag.shape[1]), **sol)

    # ---- timing over the same 30 iterations from the same state
    import r26_co_train as R26
    tim = {}
    for name in ['native_r25', 'r26_d_vcs'] + ARMS:
        restore()
        s = R.new_stats()
        torch.cuda.synchronize(); t = time.time()
        if name == 'native_r25':
            opt.noisy_obj = 'sce'
            R._orig_train(opt, A, B, lt, ut, ep, s, max_iters=30)
        elif name == 'r26_d_vcs':
            opt.arm, opt.c_J = 'd_vcs', 63.07442577797713
            R26.train(opt, A, B, lt, ut, ep, s, max_iters=30)
        else:
            opt.arm, opt.lam = name, (0.0 if name == 'a1_iface' else 1.0)
            R27.train(opt, A, B, lt, ut, ep, s, max_iters=30)
        torch.cuda.synchronize()
        tim[name] = dict(ms_per_step=1000 * (time.time() - t) / 30, steps=s.get('steps'))
    for k in tim:
        tim[k]['ratio_vs_r26'] = tim[k]['ms_per_step'] / tim['r26_d_vcs']['ms_per_step']
        tim[k]['ratio_vs_native'] = tim[k]['ms_per_step'] / tim['native_r25']['ms_per_step']
    ok('timing', True, gpu=torch.cuda.get_device_name(0), note='30 iterations incl. data loading; model A step only', **tim)

    # ---- smoke arms ran
    arms_ok = {}
    for arm in ARMS:
        e = json.load(open(f'{smoke}/{arm}/EPOCHS.json'))['cotrain']
        last = e[-1]
        fin = all(np.isfinite(last['train'][p].get(f'{b}_raw_loss_mean', np.nan)) for p in ('A', 'B') for b in ('i2t', 't2i'))
        if arm != 'a1_iface':
            fin &= all(np.isfinite(last['train'][p].get(f'{b}_aux_loss_mean', np.nan)) for p in ('A', 'B') for b in ('i2t', 't2i'))
        arms_ok[arm] = dict(finite=fin, epochs=[x['epoch'] for x in e], dev_rsum=last['dev']['rsum'],
                            first_epoch_A=e[1]['train']['A'] if len(e) > 1 else None, last_A=last['train']['A'], last_B=last['train']['B'])
    ok('arms_ran', all(v['finite'] for v in arms_ok.values()), **arms_ok)

    res['all_pass'] = all(v['pass_'] for v in res.values() if isinstance(v, dict))
    json.dump(res, open(f'{smoke}/GATE.json', 'w'), indent=1, default=float)
    sys.exit(0 if res['all_pass'] else 1)


if __name__ == '__main__':
    R._orig_train = R.train
    main()
