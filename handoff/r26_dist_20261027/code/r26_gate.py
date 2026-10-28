"""R26 real-data GPU gate (DESIGN §4.2). Usage: r26_gate.py SMOKE_DIR NOISE_FILE [data_path]. Writes SMOKE_DIR/GATE.json.

checks (on a native smoke prefix, real Flickr batches):
  grad_paths      one real training step per arm: candidates/A/queue carry no grad; queries do; current-peer
                  encoders get gradients; NO GraphRefiner parameter of either peer and no other-peer parameter gets a
                  gradient; the other peer's FIFO snapshot is bitwise unchanged across the noisy branch; raw J finite
  graph_real      real graph: W sums to 1, Q = r c^T includes every W edge, merged atoms counted, candidates unit
  dense_vs_moment D-VCS dense graph J == exact cosine moment form on the real graph (value and query gradient)
  state_roundtrip after an R26 epoch: save -> reload identical (all tensors, queues, RNG)
  resume          two runs from the same R26 state, first step: parameters equal within the cudnn tolerance
  arms_ran        every arm finished one smoke co-training epoch with finite losses and logged skip counts
"""
import os, sys, json, copy, hashlib
import numpy as np
import torch

import r25_run
import r25_co_train as R
import r26_co_train as R26
from data import get_loader, get_dataset
from vse import VSE
from vocab import deserialize_vocab
from co_train import eval_train, split_prob
from r26_graph import ARMS, build_graph, coupling, j_graph, moment_j
from r26_co_train import bank

res = {}


def ok(name, cond, **info):
    res[name] = dict(pass_=bool(cond), **info)
    print(name, 'PASS' if cond else 'FAIL', json.dumps(info, default=float)[:2000])


def h(t):
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()


def flat(m):
    return torch.cat([p.detach().flatten().double() for p in list(m.img_enc.parameters()) + list(m.txt_enc.parameters())])


def main():
    smoke, noise = sys.argv[1], sys.argv[2]
    extra = ['--data_path', sys.argv[3]] if len(sys.argv) > 3 else []
    opt = r25_run.parse(['--noise_file', noise, '--output_dir', f'{smoke}/gate', '--resume', f'{smoke}/prefix/prefix_state.pt', '--max_iters', '30'] + extra)
    opt.arm, opt.c_J = 'd_vcs', 1.0
    os.makedirs(opt.output_dir, exist_ok=True)
    r25_run.seed_all(opt.seed)
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, "%s_vocab.json" % opt.data_name)); vocab.add_word('<mask>')
    opt.vocab_size = len(vocab)
    caps, ims = get_dataset(opt.data_path, opt.data_name, "train")
    noisy_loader, data_size, clean_labels = get_loader(caps, ims, vocab, "warmup", opt.batch_size, 0, opt.noise_ratio, opt.noise_file)
    A = VSE(opt, vocab.word2idx); B = VSE(opt, vocab.word2idx)
    st = R.load_full_state(opt.resume, opt, A, B)
    prob_A, prob_B, _ = eval_train(opt, A, B, noisy_loader, data_size, copy.deepcopy(st['all_loss']), clean_labels, st['epoch'] + 1)
    pred_A = split_prob(prob_A, opt.p_threshold); pred_B = split_prob(prob_B, opt.p_threshold)
    for m, prob, pred in ((A, prob_A, pred_A), (B, prob_B, pred_B)):
        m.avg_clean_prob = prob[pred].mean()
    lt, ut = get_loader(caps, ims, vocab, "train", opt.batch_size, 0, opt.noise_ratio, opt.noise_file, pred=pred_B, prob=prob_B)
    init = copy.deepcopy(R.model_full_state(A)), copy.deepcopy(R.model_full_state(B)), R.rng_state()

    def restore():
        for m, s in ((A, init[0]), (B, init[1])):
            m.img_enc.load_state_dict(s['img_enc']); m.txt_enc.load_state_dict(s['txt_enc']); m.optimizer.load_state_dict(s['optimizer'])
            q = m.memory_queue; q.img_queue.copy_(s['queue']['img']); q.txt_queue.copy_(s['queue']['txt']); q.ptr, q.is_full = s['queue']['ptr'], s['queue']['is_full']
        R.set_rng_state(init[2])

    # ---- real graph + dense/moment
    batch = next(iter(ut))
    A.train_start(); B.val_start()
    xi, xt, _ = A.forward_emb(batch[0], batch[2], batch[3], imgae_lengths=batch[1])
    with torch.no_grad():
        xi2, xt2, _ = B.forward_emb(batch[0], batch[2], batch[3], imgae_lengths=batch[1])
    ib, tb = bank(B.memory_queue)
    info = {}
    good = True
    for br, X, Xp in (('i2t', xi, xi2), ('t2i', xt, xt2)):
        G = build_graph(Xp, ib, tb, opt.k, br)
        W, c, Q = coupling(G['A'])
        Xd = X.detach().double().requires_grad_(True); Ad = G['A'].double(); Yd = G['Y'].double()
        Wd, cd, Qd = coupling(Ad)
        jd = j_graph(Wd, Qd, Xd.mm(Yd.t())); gd = torch.autograd.grad(jd, Xd)[0]
        Xm = X.detach().double().requires_grad_(True); jm = moment_j(Xm, Ad, Yd); gm = torch.autograd.grad(jm, Xm)[0]
        info[br] = dict(B=int(X.shape[0]), n_atoms=G['n_atoms'], shared_frac=G['shared_frac'], max_col_mass=G['max_col_mass'],
                        W_sum=float(W.sum()), Q_covers_W=bool((Q[W > 0] > 0).all()), cand_norm_dev=G['norm_dev'],
                        J=float(jd), J_moment_abs=float(abs(jd - jm)), grad_max_abs=float((gd - gm).abs().max()),
                        bank_size=int(ib.shape[0]))
        good &= abs(float(W.sum()) - 1) < 1e-5 and info[br]['Q_covers_W'] and G['norm_dev'] < 1e-4 and info[br]['J_moment_abs'] < 1e-12 and info[br]['grad_max_abs'] < 1e-12
    ok('graph_real', good, **{k: {kk: v[kk] for kk in ('B', 'n_atoms', 'shared_frac', 'max_col_mass', 'W_sum', 'Q_covers_W', 'cand_norm_dev', 'bank_size')} for k, v in info.items()})
    ok('dense_vs_moment', good, **{k: {kk: v[kk] for kk in ('J', 'J_moment_abs', 'grad_max_abs')} for k, v in info.items()})
    del xi, xt, xi2, xt2

    # ---- gradient paths, one real step per arm
    arm_info = {}
    allgood = True
    for arm in ARMS:
        restore()
        opt.arm = arm
        for p in list(A.params) + list(B.params) + list(A.i2t_refiner_node.parameters()) + list(A.t2i_refiner_node.parameters()) \
                + list(B.i2t_refiner_node.parameters()) + list(B.t2i_refiner_node.parameters()):
            p.grad = None
        qh = (h(B.memory_queue.img_queue), h(B.memory_queue.txt_queue))
        seen = {}

        def probe(branch, X, G, loss):
            seen[branch] = dict(X_req=bool(X.requires_grad), Y_req=bool(G['Y'].requires_grad), A_req=bool(G['A'].requires_grad),
                                loss=float(loss), finite=bool(torch.isfinite(loss)),
                                queue_same=(h(B.memory_queue.img_queue), h(B.memory_queue.txt_queue)) == qh)
        grads = {}
        orig = A.optimizer.step

        def step_probe(*a, **k):
            grads['enc'] = sum(float(p.grad.abs().sum()) for p in A.params if p.grad is not None)
            grads['refiner_params_with_grad'] = sum(1 for m in (A, B) for p in list(m.i2t_refiner_node.parameters()) + list(m.t2i_refiner_node.parameters()) if p.grad is not None)
            grads['peer_params_with_grad'] = sum(1 for p in B.params if p.grad is not None)
            grads['peer_queue_requires_grad'] = bool(B.memory_queue.img_queue.requires_grad or B.memory_queue.txt_queue.requires_grad)
            grads['peer_queue_same_after'] = (h(B.memory_queue.img_queue), h(B.memory_queue.txt_queue)) == qh
            return orig(*a, **k)
        A.optimizer.step = step_probe
        stats = R.new_stats()
        R26.train(opt, A, B, lt, ut, st['epoch'] + 1, stats, max_iters=1, probe=probe)
        A.optimizer.step = orig
        g = (all(seen.get(b, {}).get('X_req') and not seen[b]['Y_req'] and not seen[b]['A_req'] and seen[b]['finite'] and seen[b]['queue_same'] for b in ('i2t', 't2i'))
             and grads['enc'] > 0 and grads['refiner_params_with_grad'] == 0 and grads['peer_params_with_grad'] == 0
             and not grads['peer_queue_requires_grad'] and grads['peer_queue_same_after'] and stats['i2t_n'] == 1 and stats['t2i_n'] == 1)
        arm_info[arm] = dict(branches=seen, **grads, skips=stats.get('skip_small_or_empty', 0))
        allgood &= g
    ok('grad_paths', allgood, **arm_info)

    # ---- resume: two runs from the same restored state, first step parameters
    outs = []
    for rep in range(2):
        restore(); opt.arm = 'd_vcs'
        R26.train(opt, A, B, lt, ut, st['epoch'] + 1, R.new_stats(), max_iters=1)
        outs.append(flat(A))
    pd = float((outs[0] - outs[1]).abs().max())
    ok('resume', pd < 1e-4, first_step_param_max_abs_diff=pd, note='cudnn.benchmark=True (native) is not bitwise deterministic')

    # ---- state roundtrip + arms ran (from the smoke arm runs)
    bad, arms_ok = [], {}
    for arm in ARMS:
        d = f'{smoke}/{arm}'
        s1 = torch.load(f'{d}/state_last.pt', weights_only=False)
        A2 = VSE(opt, vocab.word2idx); B2 = VSE(opt, vocab.word2idx); st2 = R.load_full_state(f'{d}/state_last.pt', opt, A2, B2)
        R.save_full_state(f'{opt.output_dir}/rt_{arm}.pt', opt, A2, B2, st2['epoch'], st2['best_rsum'], st2['all_loss'], st2['history'])
        s2 = torch.load(f'{opt.output_dir}/rt_{arm}.pt', weights_only=False)
        for side in ('model_A', 'model_B'):
            for k in ('img_enc', 'txt_enc', 'i2t_refiner', 't2i_refiner'):
                for kk in s1[side][k]:
                    if not torch.equal(s1[side][k][kk].cpu(), s2[side][k][kk].cpu()):
                        bad.append(f'{arm}/{side}/{k}/{kk}')
            for kk in ('img', 'txt'):
                if not torch.equal(s1[side]['queue'][kk].cpu(), s2[side]['queue'][kk].cpu()):
                    bad.append(f'{arm}/{side}/queue/{kk}')
        if not torch.equal(s1['rng']['torch'], s2['rng']['torch']):
            bad.append(f'{arm}/rng')
        ep = json.load(open(f'{d}/EPOCHS.json'))['cotrain'][-1]
        tr = ep['train']
        fin = all(np.isfinite(tr[p].get(f'{b}_raw_loss_mean', np.nan)) for p in ('A', 'B') for b in ('i2t', 't2i'))
        arms_ok[arm] = dict(finite=fin, dev_rsum=ep['dev']['rsum'], train_A={k: v for k, v in tr['A'].items()})
        os.remove(f'{opt.output_dir}/rt_{arm}.pt')
    ok('state_roundtrip', not bad, mismatches=bad[:10])
    ok('arms_ran', all(v['finite'] for v in arms_ok.values()), **arms_ok)

    res['all_pass'] = all(v['pass_'] for v in res.values() if isinstance(v, dict))
    json.dump(res, open(f'{smoke}/GATE.json', 'w'), indent=1, default=float)
    sys.exit(0 if res['all_pass'] else 1)


if __name__ == '__main__':
    main()
