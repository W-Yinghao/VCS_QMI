"""R27 common lambda calibration (DESIGN_CN.md §5), once, at the common native prefix (Flickr f40_s1001, as R26).

Problem: prefix queues carry no IDs, so the shared interface has no usable candidates at the prefix. For calibration
only, each peer's queue is rebuilt in memory WITH IDs by one pass of that peer's own epoch-(prefix+1) labeled loader
(train-mode forward as in the native enqueue, no grad, high-confidence rule batch_prob > avg_clean_prob, generator
seed 20261030); the peer's encoder state (BN running stats) is restored afterwards. Nothing is written back.

Then, for each peer as the trained network (A then B), the first NB=32 unlabeled FIT batches (generator seed 20261029,
B >= 2) give, wrt the current peer's ENCODER parameters (net.params; refiners excluded):
  g_nat  = grad of 0.5*SCE(x_i, refiner(K re-encoded neighbours))_i2t + 0.5*SCE(...)_t2i      (A1-interface native noisy branch)
  g_mean = grad of 0.5*SCE(x, stopgrad normalize(mu))_i2t + 0.5*SCE(...)_t2i                 (mean-target auxiliary, A2)
  lambda = median over the 2*NB (peer, batch) of ||g_nat|| / ||g_mean||, so that lam*aux matches the native noisy
           branch's encoder-gradient magnitude; the same lambda is used for A2-A5 (frozen).
Also logged (not used for lambda): the same ratio for the original native A0 noisy branch (net query searching the
other peer's ORIGINAL prefix queue, cached embeddings), the vcs/bce/snce target auxiliaries, query-gradient ratios,
cos(g_nat, g_aux), clean-loss encoder gradient, generator residuals/iterations/time on real FIT graphs.
No parameter update, no DEV/TEST read. Usage: r27_calibrate.py OUT.json <r25_run args incl --resume>.
"""
import os, sys, json, copy, time
import numpy as np
import torch

import r25_run
import r25_co_train as R
import r27_co_train as R27
import r27_interface as I
from data import get_dataset
from vse import VSE, infonce
from vocab import deserialize_vocab
from co_train import eval_train, split_prob, MemoryQueue
from r27_targets import generate

NB = 32


def gflat(loss, params):
    g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat([(x if x is not None else torch.zeros_like(p)).flatten().double() for x, p in zip(g, params)])


def gq(loss, tensors):
    g = torch.autograd.grad(loss, tensors, retain_graph=True)
    return float(torch.sqrt(sum((x.double() ** 2).sum() for x in g)))


@torch.no_grad()
def rebuild_queue(opt, net2, lt):
    """In-memory ID-bearing queue for net2 from one pass of net2's own labeled loader (train-mode forward, no grad)."""
    keep = {'img': copy.deepcopy(net2.img_enc.state_dict()), 'txt': copy.deepcopy(net2.txt_enc.state_dict())}
    q = MemoryQueue(queue_size=opt.queue_size, dim=opt.embed_size)
    I.ensure_ids(q)
    gidx, obs = lt.dataset.r27_gidx, np.asarray(lt.dataset.t2i_index, dtype=np.int64)
    net2.train_start()
    n_batches = 0
    for b in lt:
        if b[0].size(0) < 2:
            continue
        il, cl, _ = net2.forward_emb(b[0], b[2], b[3], imgae_lengths=b[1])
        prob = b[6].cuda()
        hc = (prob > net2.avg_clean_prob).nonzero(as_tuple=True)[0]
        if hc.numel():
            ids = np.asarray(b[4], dtype=np.int64)[hc.cpu().numpy()]
            I.enqueue_with_ids(q, il[hc], cl[hc], gidx[ids], obs[ids], 1, n_batches, 9)
        n_batches += 1
    net2.img_enc.load_state_dict(keep['img']); net2.txt_enc.load_state_dict(keep['txt'])
    net2.val_start()
    return q, n_batches


def main():
    out_path = sys.argv[1]
    opt = r25_run.parse(sys.argv[2:])
    assert opt.resume
    R27.install()
    r25_run.seed_all(opt.seed)
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, "%s_vocab.json" % opt.data_name)); vocab.add_word('<mask>')
    opt.vocab_size = len(vocab)
    caps, ims = get_dataset(opt.data_path, opt.data_name, "train")
    noisy_loader, data_size, clean_labels = R.get_loader(caps, ims, vocab, "warmup", opt.batch_size, opt.workers, opt.noise_ratio, opt.noise_file)
    A = VSE(opt, vocab.word2idx); B = VSE(opt, vocab.word2idx)
    st = R.load_full_state(opt.resume, opt, A, B)
    epoch = st['epoch'] + 1
    prob_A, prob_B, _ = eval_train(opt, A, B, noisy_loader, data_size, copy.deepcopy(st['all_loss']), clean_labels, epoch)
    pred_A = split_prob(prob_A, opt.p_threshold); pred_B = split_prob(prob_B, opt.p_threshold)
    for m, prob, pred in ((A, prob_A, pred_A), (B, prob_B, pred_B)):
        cp = prob[pred]; m.avg_clean_prob = cp.mean() if len(cp) > 0 else 0.0
    loaders = {}
    for name, pred, prob in (('A', pred_B, prob_B), ('B', pred_A, prob_A)):   # peer X trains on the other peer's split
        loaders[name] = R.get_loader(caps, ims, vocab, "train", opt.batch_size, opt.workers, opt.noise_ratio, opt.noise_file, pred=pred, prob=prob)
    rows, qinfo = [], {}
    for name, net, net2 in (('A', A, B), ('B', B, A)):
        g0 = torch.Generator(); g0.manual_seed(20261030)
        ltd = loaders['B' if name == 'A' else 'A'][0].dataset           # net2's own labeled split
        lq = torch.utils.data.DataLoader(ltd, batch_size=opt.batch_size, shuffle=True, generator=g0, pin_memory=True, collate_fn=loaders[name][0].collate_fn, num_workers=0)
        t = time.time()
        cq, nbq = rebuild_queue(opt, net2, lq)
        n_ids = int((cq.r27_ids['cap'] >= 0).sum())
        qinfo[name] = dict(other_peer_queue_rebuilt_entries=n_ids, batches=nbq, seconds=time.time() - t, is_full=bool(cq.is_full))
        orig_q = net2.memory_queue
        lt, ut = loaders[name]
        gen = torch.Generator(); gen.manual_seed(20261029)
        utl = torch.utils.data.DataLoader(ut.dataset, batch_size=opt.batch_size, shuffle=True, generator=gen, pin_memory=True, collate_fn=ut.collate_fn, num_workers=0)
        gen2 = torch.Generator(); gen2.manual_seed(20261028)
        lti = iter(torch.utils.data.DataLoader(lt.dataset, batch_size=opt.batch_size, shuffle=True, generator=gen2, pin_memory=True, collate_fn=lt.collate_fn, num_workers=0))
        net.train_start(); net2.val_start()
        params = list(net.params)
        n_ok = 0
        for batch in utl:
            if n_ok >= NB:
                break
            imgs, ilen, txt, tlen = batch[0], batch[1], batch[2], batch[3]
            if imgs.size(0) < 2:
                continue
            xi, xt, _, mi, mt = I.forward_emb_ctx(net, imgs, txt, tlen, imgae_lengths=ilen)
            with torch.no_grad():
                xi2, xt2, _ = net2.forward_emb(imgs, txt, tlen, imgae_lengths=ilen)
            r = {'peer': name, 'batch': n_ok, 'B': int(imgs.size(0))}
            L = {k: [] for k in ('nat', 'mean', 'vcs', 'bce', 'snce', 'native0')}
            for br, X, Xp, mean in (('i2t', xi, xi2, mt), ('t2i', xt, xt2, mi)):
                S, qs = I.search(cq, Xp, br, opt.k)
                Y = I.reencode(net, br, S['cols'], mean)
                ref = net.i2t_refiner_node if br == 'i2t' else net.t2i_refiner_node
                L['nat'].append(infonce(X.mm(ref(Y[S['inv']]).t())))
                for gname in ('mean', 'vcs', 'bce', 'snce'):
                    torch.cuda.synchronize(); tg0 = time.time()
                    Tg, valid, info = generate(gname, S['A'], Y)
                    torch.cuda.synchronize()
                    r[f'{br}_{gname}_sec'] = time.time() - tg0
                    assert bool(valid.all())
                    L[gname].append(infonce(X.mm(Tg.t())))
                    if gname != 'mean':
                        r[f'{br}_{gname}_maxrel'] = float(info['rel'].max()); r[f'{br}_{gname}_maxit'] = int(info['iters'].max())
                        r[f'{br}_{gname}_cap'] = int(info['cap'].sum()); r[f'{br}_{gname}_fallback'] = int(info['fallback'].sum())
                        r[f'{br}_{gname}_cos_to_mean'] = float(info['cos_to_mean'].mean())
                r[f'{br}_n_atoms'] = int(S['cols'].numel())
                # original native A0 path: current query searches the other peer's ORIGINAL prefix queue (cached embeddings)
                p0 = orig_q.retrieve_nearest_text(X, k=opt.k, refiner_node=net.i2t_refiner_node) if br == 'i2t' else \
                    orig_q.retrieve_nearest_img(X, k=opt.k, refiner_node=net.t2i_refiner_node)
                L['native0'].append(infonce(X.mm(p0.t())))
            loss = {'nat': 0.5 * L['nat'][0] + 0.5 * L['nat'][1], 'native0': 0.5 * L['native0'][0] + 0.5 * L['native0'][1]}
            for gname in ('mean', 'vcs', 'bce', 'snce'):
                loss[gname] = 0.5 * (L[gname][0] + L[gname][1])
            G = {k: gflat(v, params) for k, v in loss.items()}
            for k, v in G.items():
                r[f'enc_{k}'] = float(v.norm())
                r[f'qgrad_{k}'] = gq(loss[k], [xi, xt])
                r[f'loss_{k}'] = float(loss[k])
            for k in ('mean', 'vcs', 'bce', 'snce', 'native0'):
                r[f'cos_nat_{k}'] = float((G['nat'] @ G[k]) / (G['nat'].norm() * G[k].norm()))
            bl = next(lti)
            il, cl, cll = net.forward_emb(bl[0], bl[2], bl[3], imgae_lengths=bl[1])
            il2, cl2, _ = net.forward_emb(bl[0], bl[2], bl[3], imgae_lengths=bl[1])
            crit = net.criterion
            lclean = crit(net.forward_sim(il, cl, cll), labels=None) + crit(il.mm(il2.t()), margin=0.3) + crit(cl.mm(cl2.t()), margin=0.3)
            r['enc_clean'] = float(gflat(lclean, params).norm())
            rows.append(r); n_ok += 1
            print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items() if k.startswith(('enc_', 'cos_nat', 'peer', 'batch'))}), flush=True)
            del xi, xt, xi2, xt2, il, cl, il2, cl2, G
        del cq
        torch.cuda.empty_cache()
    br_ = [r for r in rows if 'batch' in r]
    ratio = np.array([r['enc_nat'] / r['enc_mean'] for r in br_])
    lam = float(np.median(ratio))
    q = lambda a: {str(p): float(np.quantile(a, p)) for p in (0, .1, .25, .5, .75, .9, 1)}
    out = {'prefix': opt.resume, 'prefix_epoch': st['epoch'], 'split_epoch': epoch, 'data_name': opt.data_name, 'noise_file': opt.noise_file,
           'fit_batches_per_peer': NB, 'n_rows': len(br_), 'lambda': lam, 'lambda_rule': 'median ||g_enc(native noisy, A1 interface)|| / ||g_enc(mean-target aux)||',
           'ratio_quantiles': q(ratio), 'queue_rebuild': qinfo,
           'diagnostic_ratios_native_over_aux': {k: q(np.array([r['enc_nat'] / r[f'enc_{k}'] for r in br_])) for k in ('vcs', 'bce', 'snce')},
           'diagnostic_ratio_native0_over_mean': q(np.array([r['enc_native0'] / r['enc_mean'] for r in br_])),
           'diagnostic_ratio_nat_over_native0': q(np.array([r['enc_nat'] / r['enc_native0'] for r in br_])),
           'diagnostic_query_grad_ratio_nat_over_mean': q(np.array([r['qgrad_nat'] / r['qgrad_mean'] for r in br_])),
           'lam_aux_to_clean_enc_median': float(np.median([lam * r['enc_mean'] / r['enc_clean'] for r in br_])),
           'nat_to_clean_enc_median': float(np.median([r['enc_nat'] / r['enc_clean'] for r in br_])),
           'cos_nat_median': {k: float(np.median([r[f'cos_nat_{k}'] for r in br_])) for k in ('mean', 'vcs', 'bce', 'snce', 'native0')},
           'generator_sec_median': {g: float(np.median([r[f'{b}_{g}_sec'] for r in br_ for b in ('i2t', 't2i')])) for g in ('mean', 'vcs', 'bce', 'snce')},
           'generator_max_rel_kkt': {g: float(max(r[f'{b}_{g}_maxrel'] for r in br_ for b in ('i2t', 't2i'))) for g in ('vcs', 'bce', 'snce')},
           'generator_max_iters': {g: int(max(r[f'{b}_{g}_maxit'] for r in br_ for b in ('i2t', 't2i'))) for g in ('vcs', 'bce', 'snce')},
           'generator_cap_hits': {g: int(sum(r[f'{b}_{g}_cap'] for r in br_ for b in ('i2t', 't2i'))) for g in ('vcs', 'bce', 'snce')},
           'gpu': torch.cuda.get_device_name(0), 'rows': rows}
    json.dump(out, open(out_path, 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, indent=1))
    sys.exit(0 if np.isfinite(lam) and lam > 0 and len(br_) == 2 * NB else 1)


if __name__ == '__main__':
    main()
