"""R27-A1X real-data GPU gate (A1 only, COCO 40% / CC152K). Writes SMOKE_DIR/GATE.json.
Usage: r27x_gate.py SMOKE_DIR DATA_PATH DATA_NAME NOISE_FILE NOISE_RATIO SEED PREFIX

Runs after the a1_iface smoke (real R26 prefix, 2 co-train epochs x 30 iterations); starts from its state_last.pt.
  forward_ctx      forward_emb_ctx == native VSE.forward_emb bitwise (same RNG state)
  grad_paths       one real A1 step: queries carry grad; re-encoded candidates / neighbours do not; current-peer encoders and
                   current-peer GraphRefiners get gradients; other peer gets none; other-peer queue + ID buffer unchanged;
                   no aux term; losses finite
  current_peer     re-encoding the probe's atom IDs with the CURRENT peer (eval, same centering mean) reproduces the
                   candidates (bitwise reported); the other peer's re-encoding differs
  candidate_ids    over 10 real steps: every selected slot is ID-bearing (cap >= 0, no prefix slot); i2t atoms = caption IDs
                   < n_caps, t2i atoms = observed image IDs < n_imgs; slot img == observed[slot cap] (dataset's own map);
                   gathered image rows == per-row memmap reads bitwise (and GPU-resident == host gather when it fits)
  queue_ids        every ID-bearing slot of both queues: cap < n_caps, img < n_imgs, img == observed[cap], peer == owner,
                   wepoch in the smoke epochs; injected mismatches present iff noise_ratio > 0 (truth read only here);
                   R26 prefix -> all slots pre-ID, search returns None
  state_roundtrip  state_last -> load -> save -> load: all tensors incl. queues, ID buffers, optimizer, RNG identical
  resume           first step after (load) vs after (load, save, load): parameters within 1e-4 (host tolerance)
  timing_memory    ms/step over the same 30 iterations from the same state: R25 native vs A1 (production gather mode,
                   plus host-gather mode when the production mode is GPU-resident); peak GPU memory
  arm_ran          smoke finished co-train epochs 1-2 with finite native losses for both peers / directions
"""
import os, sys, json, copy, hashlib, time, glob
import numpy as np
import torch

import r25_run
import r25_co_train as R
import r27_co_train as R27
import r27_interface as I
from r27_interface import Ctx
from data import get_dataset
from vse import VSE
from vocab import deserialize_vocab
from co_train import eval_train, split_prob

res = {}


def ok(name, cond, **info):
    res[name] = dict(pass_=bool(cond), **info)
    print(name, 'PASS' if cond else 'FAIL', json.dumps(info, default=float)[:3000], flush=True)


def h(t):
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()


def flat(m):
    return torch.cat([p.detach().flatten().double() for p in list(m.img_enc.parameters()) + list(m.txt_enc.parameters())])


def main():
    smoke, dpath, dname, noise, nratio, seed, prefix = sys.argv[1:8]
    S0 = f'{smoke}/a1_iface/state_last.pt'
    opt = r25_run.parse(['--noise_file', noise, '--noise_ratio', nratio, '--seed', seed, '--data_name', dname,
                         '--output_dir', f'{smoke}/gate', '--resume', S0, '--data_path', dpath, '--max_iters', '30'])
    opt.arm, opt.lam = 'a1_iface', 0.0
    os.makedirs(opt.output_dir, exist_ok=True)
    R27.install()
    r25_run.seed_all(opt.seed)
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, "%s_vocab.json" % opt.data_name)); vocab.add_word('<mask>')
    opt.vocab_size = len(vocab)
    caps, ims = get_dataset(opt.data_path, opt.data_name, "train")
    n_caps, n_imgs = len(caps), ims.shape[0]
    noisy_loader, data_size, clean_labels = R.get_loader(caps, ims, vocab, "warmup", opt.batch_size, 0, opt.noise_ratio, opt.noise_file)
    obs = np.asarray(noisy_loader.dataset.t2i_index, dtype=np.int64)
    true = np.asarray(noisy_loader.dataset._t2i_index, dtype=np.int64)
    obs_t = torch.as_tensor(obs, device='cuda'); true_t = torch.as_tensor(true, device='cuda')
    div = 5 if n_imgs != n_caps else 1
    A = VSE(opt, vocab.word2idx); B = VSE(opt, vocab.word2idx)
    st = R.load_full_state(S0, opt, A, B)
    ep = st['epoch'] + 1
    prob_A, prob_B, _ = eval_train(opt, A, B, noisy_loader, data_size, copy.deepcopy(st['all_loss']), clean_labels, ep)
    pred_A = split_prob(prob_A, opt.p_threshold); pred_B = split_prob(prob_B, opt.p_threshold)
    for m, prob, pred in ((A, prob_A, pred_A), (B, prob_B, pred_B)):
        m.avg_clean_prob = prob[pred].mean()
    lt, ut = R.get_loader(caps, ims, vocab, "train", opt.batch_size, 0, opt.noise_ratio, opt.noise_file, pred=pred_B, prob=prob_B)
    snap = copy.deepcopy(R.model_full_state(A)), copy.deepcopy(R.model_full_state(B)), R.rng_state()

    def restore():  # deep copies at every restore: optimizer.load_state_dict keeps references to the given tensors
        for m, s0 in ((A, snap[0]), (B, snap[1])):
            s = copy.deepcopy(s0)
            m.img_enc.load_state_dict(s['img_enc']); m.txt_enc.load_state_dict(s['txt_enc']); m.optimizer.load_state_dict(s['optimizer'])
            m.i2t_refiner_node.load_state_dict(s['i2t_refiner']); m.t2i_refiner_node.load_state_dict(s['t2i_refiner'])
            q = m.memory_queue; q.img_queue.copy_(s['queue']['img']); q.txt_queue.copy_(s['queue']['txt']); q.ptr, q.is_full = s['queue']['ptr'], s['queue']['is_full']
            I.load_ids(q, s['r27_queue_ids'])
        R.set_rng_state(copy.deepcopy(snap[2]))
        R27._PEER.clear(); R27._PEER[id(A)] = 0; R27._PEER[id(B)] = 1

    # ---- forward_ctx equivalence
    batch = next(iter(ut))
    A.train_start()
    rng = R.rng_state()
    a1, a2, _ = A.forward_emb(batch[0], batch[2], batch[3], imgae_lengths=batch[1])
    R.set_rng_state(rng)
    b1, b2, _, mi, mt = I.forward_emb_ctx(A, batch[0], batch[2], batch[3], imgae_lengths=batch[1])
    ok('forward_ctx', torch.equal(a1, b1) and torch.equal(a2, b2), img_equal=torch.equal(a1, b1), txt_equal=torch.equal(a2, b2))
    del a1, a2, b1, b2

    mode = Ctx.mode()
    gpu_fits = Ctx.images.nbytes + Ctx.GATHER_HEADROOM <= torch.cuda.get_device_properties(0).total_memory

    # ---- grad paths + current peer, one real A1 step
    restore()
    for m in (A, B):
        for p in list(m.params) + list(m.i2t_refiner_node.parameters()) + list(m.t2i_refiner_node.parameters()):
            p.grad = None
    qh = (h(B.memory_queue.img_queue), h(B.memory_queue.txt_queue), tuple(h(v) for v in B.memory_queue.r27_ids.values()))
    seen = {}

    def probe(branch, d):
        Y, neigh, S, mean = d['Y'], d['neigh'], d['S'], d['mean']
        Ya = I.reencode(A, branch, S['cols'], mean)
        Yb = I.reencode(B, branch, S['cols'], mean)
        seen[branch] = dict(X_req=bool(d['X'].requires_grad), Y_req=bool(Y.requires_grad), neigh_req=bool(neigh.requires_grad),
                            target_none=d['target'] is None, proto_req=bool(d['proto'].requires_grad),
                            aux=d['l_aux'] is not None, loss=float(d['loss']), finite=bool(torch.isfinite(d['loss'])),
                            reenc_current_equal=bool(torch.equal(Ya, Y)), reenc_current_maxdiff=float((Ya - Y).abs().max()),
                            reenc_other_maxdiff=float((Yb - Y).abs().max()), mean_req=bool(mean.requires_grad), n_atoms=int(S['cols'].numel()),
                            queue_same=(h(B.memory_queue.img_queue), h(B.memory_queue.txt_queue), tuple(h(v) for v in B.memory_queue.r27_ids.values())) == qh)
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
    g = (all(seen.get(b, {}).get('X_req') and not seen[b]['Y_req'] and not seen[b]['neigh_req'] and seen[b]['proto_req']
             and seen[b]['target_none'] and not seen[b]['aux'] and seen[b]['finite'] and seen[b]['queue_same'] and not seen[b]['mean_req']
             for b in ('i2t', 't2i'))
         and grads['enc'] > 0 and grads['own_refiner_params_with_grad'] > 0
         and grads['peer_params_with_grad'] == 0 and grads['peer_queue_same_after'] and stats['i2t_n'] == 1 and stats['t2i_n'] == 1)
    ok('grad_paths', g, branches=seen, **grads)
    ok('current_peer', all(seen[b]['reenc_current_maxdiff'] < 1e-6 and seen[b]['reenc_other_maxdiff'] > 1e-3 for b in ('i2t', 't2i')),
       gather_mode=mode, **{b: dict(bitwise_equal=seen[b]['reenc_current_equal'], current_maxdiff=seen[b]['reenc_current_maxdiff'],
                                    other_maxdiff=seen[b]['reenc_other_maxdiff']) for b in ('i2t', 't2i')})

    # ---- candidate IDs + gather over 10 real steps
    restore()
    cid = dict(steps=0, bad_prefix_slot=0, bad_range=0, bad_partner=0, bad_cols=0, gather_rowref_bad=0, gather_gpu_vs_host_bad=0,
               gather_gpu_vs_host_checked=0, atoms_i2t=0, atoms_t2i=0, injected_mismatch_slots=0)

    def probe2(branch, d):
        S = d['S']
        cap, img, cols = S['cap'], S['img'], S['cols']
        cid['steps'] += branch == 'i2t'
        cid['bad_prefix_slot'] += int((cap < 0).sum())
        cid['bad_partner'] += int((img != obs_t[cap.clamp(min=0)]).sum())
        cid['injected_mismatch_slots'] += int((obs_t[cap.clamp(min=0)] != true_t[cap.clamp(min=0)]).sum())
        if branch == 'i2t':
            cid['bad_range'] += int(((cols < 0) | (cols >= n_caps)).sum()); cid['atoms_i2t'] += int(cols.numel())
            cid['bad_cols'] += int(not torch.equal(cols, torch.unique(cap)))
        else:
            cid['bad_range'] += int(((cols < 0) | (cols >= n_imgs)).sum()); cid['atoms_t2i'] += int(cols.numel())
            cid['bad_cols'] += int(not torch.equal(cols, torch.unique(img)))
            got = Ctx.image_rows(cols)
            ref = torch.from_numpy(np.stack([np.array(Ctx.images[int(c)]) for c in cols.tolist()])).cuda()
            cid['gather_rowref_bad'] += int(not torch.equal(got, ref))
            if mode == 'gpu':
                cid['gather_gpu_vs_host_bad'] += int(not torch.equal(Ctx.gpu_images()[cols], torch.from_numpy(Ctx.rows_np(cols.cpu().numpy())).cuda()))
                cid['gather_gpu_vs_host_checked'] += 1
    R27.train(opt, A, B, lt, ut, ep, R.new_stats(), max_iters=10, probe=probe2)
    ok('candidate_ids', cid['steps'] >= 9 and all(cid[k] == 0 for k in ('bad_prefix_slot', 'bad_range', 'bad_partner', 'bad_cols', 'gather_rowref_bad', 'gather_gpu_vs_host_bad'))
       and (cid['gather_gpu_vs_host_checked'] > 0 or mode == 'cpu'), gather_mode=mode, gpu_resident_fits=gpu_fits, n_caps=n_caps, n_imgs=n_imgs, im_div=div, **cid)

    # ---- queue IDs (truth read only here) + prefix exclusion, on the smoke end state (gate steps undone)
    restore()
    good, info = True, {}
    for name, m, owner in (('A', A, 0), ('B', B, 1)):
        n, v = I.usable(m.memory_queue)
        ids = {k: x[:n][v].cpu().numpy() for k, x in m.memory_queue.r27_ids.items()}
        cap, img = ids['cap'], ids['img']
        gq = (cap.max() < n_caps and img.max() < n_imgs and cap.min() >= 0 and bool((img == obs[cap]).all())
              and bool((ids['peer'] == owner).all()) and set(np.unique(ids['wepoch']).tolist()) <= {1, 2})
        mism = int((obs[cap] != true[cap]).sum())
        info[name] = dict(filled=n, id_bearing=int(v.sum()), pre_id_slots=int((~v).sum()), injected_mismatch=mism,
                          wepochs=sorted(set(np.unique(ids['wepoch']).tolist())), ok=bool(gq))
        good &= bool(gq) and (mism > 0 if opt.noise_ratio > 0 else mism == 0)
    good &= (opt.noise_ratio > 0) or bool(np.array_equal(obs, true))
    Ap = VSE(opt, vocab.word2idx); Bp = VSE(opt, vocab.word2idx)
    R27.load_full_state(prefix, opt, Ap, Bp)
    pre = {}
    for name, m in (('A', Ap), ('B', Bp)):
        n, v = I.usable(m.memory_queue)
        S, _ = I.search(m.memory_queue, torch.nn.functional.normalize(torch.randn(8, opt.embed_size, device='cuda'), dim=1), 'i2t', opt.k)
        pre[name] = dict(filled=n, usable=int(v.sum()), search_none=S is None)
        good &= int(v.sum()) == 0 and S is None
    del Ap, Bp
    ok('queue_ids', good, noise_ratio=opt.noise_ratio, observed_equals_true=bool(np.array_equal(obs, true)), prefix=pre, **info)

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
        for kk in ('ptr', 'is_full'):
            if s1[side]['queue'][kk] != s2[side]['queue'][kk]:
                bad.append(f'{side}/queue/{kk}')
        for kk in I.KEYS:
            if not torch.equal(s1[side]['r27_queue_ids'][kk], s2[side]['r27_queue_ids'][kk]):
                bad.append(f'{side}/ids/{kk}')
        o1, o2 = s1[side]['optimizer']['state'], s2[side]['optimizer']['state']
        for pk in o1:
            for kk in o1[pk]:
                if torch.is_tensor(o1[pk][kk]) and not torch.equal(o1[pk][kk].cpu(), o2[pk][kk].cpu()):
                    bad.append(f'{side}/opt/{pk}/{kk}')
        if s1[side]['Eiters'] != s2[side]['Eiters'] or s1[side]['avg_clean_prob'] != s2[side]['avg_clean_prob']:
            bad.append(f'{side}/scalars')
    for kk in ('torch', 'cuda'):
        a_, b_ = s1['rng'][kk], s2['rng'][kk]
        if (not torch.equal(a_, b_)) if torch.is_tensor(a_) else not all(torch.equal(x, y) for x, y in zip(a_, b_)):
            bad.append(f'rng/{kk}')
    n1, n2 = s1['rng']['numpy'], s2['rng']['numpy']
    if not (n1[0] == n2[0] and np.array_equal(n1[1], n2[1]) and tuple(n1[2:]) == tuple(n2[2:])) or s1['rng']['python'] != s2['rng']['python']:
        bad.append('rng/numpy_or_python')
    ok('state_roundtrip', not bad, mismatches=bad[:10], ids_present=all('r27_queue_ids' in s1[s] for s in ('model_A', 'model_B')),
       epoch=s1['epoch'])
    del s1, s2, A2, B2

    # ---- resume: first step from (load) vs (load, save, load)
    outs = []
    for path in (S0, rt):
        A3 = VSE(opt, vocab.word2idx); B3 = VSE(opt, vocab.word2idx)
        R.load_full_state(path, opt, A3, B3)
        A3.avg_clean_prob, B3.avg_clean_prob = A.avg_clean_prob, B.avg_clean_prob
        R27._PEER.clear()
        lt3, ut3 = R.get_loader(caps, ims, vocab, "train", opt.batch_size, 0, opt.noise_ratio, opt.noise_file, pred=pred_B, prob=prob_B)
        R27.train(opt, A3, B3, lt3, ut3, ep, R.new_stats(), max_iters=1)
        outs.append(flat(A3))
        del A3, B3
    pd = float((outs[0] - outs[1]).abs().max())
    ok('resume', pd < 1e-4, first_step_param_max_abs_diff=pd, bitwise=pd == 0.0, note='host tolerance (R26/R27 gates): cudnn.benchmark=True')
    os.remove(rt)

    # ---- timing over the same 30 iterations from the same state + memory
    tim = {}
    runs = [('native_r25', None), ('a1_iface_' + mode, mode)]
    if mode == 'gpu':
        runs.append(('a1_iface_cpu', 'cpu'))
    for name, gm in runs:
        restore()
        s = R.new_stats()
        torch.cuda.synchronize(); t = time.time()
        if gm is None:
            opt.noisy_obj = 'sce'
            R._orig_train(opt, A, B, lt, ut, ep, s, max_iters=30)
        else:
            Ctx.gather_mode = gm
            R27.train(opt, A, B, lt, ut, ep, s, max_iters=30)
        torch.cuda.synchronize()
        tim[name] = dict(ms_per_step=1000 * (time.time() - t) / 30, steps=s.get('steps'),
                         reenc_t2i_s=s.get('time_reenc_t2i', 0) / max(s.get('time_n', 1), 1), reenc_i2t_s=s.get('time_reenc_i2t', 0) / max(s.get('time_n', 1), 1))
    Ctx.gather_mode = mode
    for k in tim:
        tim[k]['ratio_vs_native'] = tim[k]['ms_per_step'] / tim['native_r25']['ms_per_step']
    peak = [json.load(open(p)) for p in glob.glob(f'{smoke}/a1_iface/PEAK_*.json')]
    ok('timing_memory', True, gpu=torch.cuda.get_device_name(0), gpu_total_GiB=torch.cuda.get_device_properties(0).total_memory / 2**30,
       image_array_GiB=Ctx.images.nbytes / 2**30, gather_mode=mode, gate_peak_alloc_GiB=torch.cuda.max_memory_allocated() / 2**30,
       smoke_peak=peak, note='30 iterations incl. data loading; model A step only', **tim)

    # ---- smoke arm ran
    e = json.load(open(f'{smoke}/a1_iface/EPOCHS.json'))['cotrain']
    last = e[-1]
    fin = all(np.isfinite(last['train'][p].get(f'{b}_raw_loss_mean', np.nan)) for p in ('A', 'B') for b in ('i2t', 't2i'))
    fin &= all(np.isfinite(x['train'][p]['clean_loss_mean']) for x in e[1:] for p in ('A', 'B'))
    ok('arm_ran', fin and [x['epoch'] for x in e] == [0, 1, 2], epochs=[x['epoch'] for x in e], dev_rsum=[x['dev']['rsum'] for x in e],
       last_A=last['train']['A'], last_B=last['train']['B'], epoch1_A_no_proto=e[1]['train']['A'].get('i2t_no_proto'))

    here = os.path.dirname(os.path.abspath(__file__))
    res['code_sha256'] = {os.path.basename(f): hashlib.sha256(open(f, 'rb').read()).hexdigest() for f in sorted(glob.glob(os.path.join(here, '*.py')))}
    res['data_name'] = dname
    res['job'] = os.environ.get('SLURM_JOB_ID')
    res['all_pass'] = all(v['pass_'] for v in res.values() if isinstance(v, dict) and 'pass_' in v)
    json.dump(res, open(f'{smoke}/GATE.json', 'w'), indent=1, default=float)
    print('GATE all_pass', res['all_pass'])
    sys.exit(0 if res['all_pass'] else 1)


if __name__ == '__main__':
    R._orig_train = R.train
    main()
