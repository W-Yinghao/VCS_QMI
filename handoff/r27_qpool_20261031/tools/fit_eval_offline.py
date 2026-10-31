#!/usr/bin/env python3
"""R27-Q offline FIT correspondence-quality evaluator (NEXT_DESIGN_CN.md §6 items 1-5). EVALUATOR ONLY: this is the one
R27-Q file that reads the injected-noise truth (true image of train caption j = j // 5, Flickr). Never imported by training.

On one saved full state (e.g. a unit's state_last.pt, or state_epoch10/20.pt) the next stage is reconstructed exactly as in
training (eval_train -> GMM split, teacher = the other peer in eval mode, C-native C_mix candidates from the teacher's own
queue, the stage's frozen scorers refitted with the protocol's stage seed / config). Every scorer (cos, vcs, bce, snce) is
applied to the SAME candidate graphs (fixed-graph comparison), and the current peer's GraphRefiner (eval mode) gives the
uniform (C-native) and alpha-weighted prototypes. Queries = the first --n-batches batches of the stage's noisy (unlabeled)
split in a fixed order (torch seed 0 for the loader; evaluation only).

Reported per scorer:
 1. true image in the teacher's eligible bank; in the Top5 (C_mix) candidates;
 2. truth in Top5: alpha mass on the true candidate, alpha-argmax hit, rank of the truth by alpha;
 3. truth not in Top5: max alpha, entropy, and target change cos(p_alpha, p_uniform) / ||p_alpha - p_uniform||;
 4. candidate-order permutation with identity-aligned alpha: max |alpha_perm - alpha[perm]| and target diff (must be ~0);
    query swap (row i's candidates scored with another row's query): mean L1 change of alpha (allowed to change);
 5. source influence: alpha mass on route-A vs route-B candidates, mass by slot, and truth-mass by the truth's route;
    plus the target change vs uniform for all rows.
Usage (cwd anywhere): fit_eval_offline.py --state S --noise-file N --data-path DP --code-dir CODE_Q --out OUT.json
                      [--stage A|B] [--n-batches 50] [--scorers cos,vcs,bce,snce]
"""
import argparse, copy, json, os, sys, time, hashlib


def main():
    p = argparse.ArgumentParser()
    for n in ('state', 'noise-file', 'data-path', 'code-dir', 'out'):
        p.add_argument('--' + n, required=True)
    p.add_argument('--noise-ratio', type=float, default=None)
    p.add_argument('--stage', default='A', choices=['A', 'B'])
    p.add_argument('--n-batches', type=int, default=50)
    p.add_argument('--scorers', default='cos,vcs,bce,snce')
    a = p.parse_args()
    sys.path.insert(0, os.path.abspath(a.code_dir))
    import numpy as np
    import torch
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

    st0 = torch.load(a.state, map_location='cpu', weights_only=False)
    so = st0['opt']; g_ = (lambda k: so.get(k) if isinstance(so, dict) else getattr(so, k, None))
    nr = a.noise_ratio if a.noise_ratio is not None else float(g_('noise_ratio'))
    del st0
    opt = r25_run.parse(['--noise_file', a.noise_file, '--noise_ratio', str(nr), '--output_dir', os.path.dirname(os.path.abspath(a.out)),
                         '--resume', a.state, '--data_path', a.data_path])
    opt.arm, opt.lam = 'q2_vcs_pool', 0.0
    R27.install()
    r25_run.seed_all(opt.seed)
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, "%s_vocab.json" % opt.data_name)); vocab.add_word('<mask>')
    opt.vocab_size = len(vocab)
    caps, ims = get_dataset(opt.data_path, opt.data_name, "train")
    noisy_loader, data_size, clean_labels = R.get_loader(caps, ims, vocab, "warmup", opt.batch_size, 0, opt.noise_ratio, opt.noise_file)
    A = VSE(opt, vocab.word2idx); B = VSE(opt, vocab.word2idx)
    st = R.load_full_state(a.state, opt, A, B)
    ep = st['epoch'] + 1
    prob_A, prob_B, _ = eval_train(opt, A, B, noisy_loader, data_size, copy.deepcopy(st['all_loss']), clean_labels, ep)
    pred_A = split_prob(prob_A, opt.p_threshold); pred_B = split_prob(prob_B, opt.p_threshold)
    for m, prob, pred in ((A, prob_A, pred_A), (B, prob_B, pred_B)):
        m.avg_clean_prob = prob[pred].mean()
    if a.stage == 'A':
        net, net2, pred, prob, peer = A, B, pred_B, prob_B, 0
    else:
        net, net2, pred, prob, peer = B, A, pred_A, prob_A, 1
    lt, ut = R.get_loader(caps, ims, vocab, "train", opt.batch_size, 0, opt.noise_ratio, opt.noise_file, pred=pred, prob=prob)
    net.val_start(); net2.val_start()
    ref = net.t2i_refiner_node; ref.eval()
    seed = QP.stage_seed(opt.seed, ep, peer)
    kinds = a.scorers.split(',')
    heads = {}
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed)
        fitd, vald, info, _ = QP.stage_data(net2, lt.dataset, seed)
        for k in kinds:
            if k != 'cos':
                heads[k], _ = QP.fit_head(k, *fitd, seed=seed + 17, steps=C.STEPS)
    beta = {k: (C.BETA_COS if k == 'cos' else C.BETA_HEAD) for k in kinds}

    ud = ut.dataset
    u_gidx, u_obs = ud.r27_gidx, np.asarray(ud.t2i_index, dtype=np.int64)
    torch.manual_seed(0)
    acc = {k: dict(rows=0, in_bank=0, in_top5=0, mass_true_sum=0.0, hit_true=0, rank_true_sum=0.0, absent_rows=0, absent_maxw_sum=0.0,
                   absent_ent_sum=0.0, absent_cos_sum=0.0, absent_l2_sum=0.0, all_cos_sum=0.0, mass_routeB_sum=0.0,
                   slot_mass=[0.0] * 5, true_routeA_n=0, true_routeA_mass=0.0, true_routeB_n=0, true_routeB_mass=0.0,
                   perm_alpha_maxdiff=0.0, perm_target_maxdiff=0.0, qswap_l1_sum=0.0, ent_sum=0.0, maxw_sum=0.0,
                   uniform_hit_true_in_top5=0) for k in kinds}
    nb = 0
    gperm = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for bi, batch in enumerate(ut):
            if nb >= a.n_batches:
                break
            imgs, ilen, txt, tlen, ids, _cl = batch
            if imgs.size(0) < 2:
                continue
            iu = np.asarray(ids, dtype=np.int64)
            cg = torch.as_tensor(u_gidx[iu], device='cuda'); co = torch.as_tensor(u_obs[iu], device='cuda')
            truth = cg // 5                                                          # EVALUATOR ONLY
            xi2, xt2, _ = net2.forward_emb(imgs, txt, tlen, imgae_lengths=ilen)
            S, qs = CS.search_t2i(net2.memory_queue, xt2, QP.SOURCE, QP.K, None)
            if S is None:
                continue
            nb += 1
            n, valid = I.usable(net2.memory_queue)
            bank_ids = net2.memory_queue.r27_ids['img'][:n][valid]
            in_bank = torch.isin(truth, bank_ids)
            img = S['img']; Bn = img.shape[0]
            tpos = (img == truth[:, None])
            in5 = tpos.any(1)
            _, _, _, mi, mt = I.forward_emb_ctx(net, imgs, txt, tlen, imgae_lengths=ilen)
            Y = I.reencode(net, 't2i', S['cols'], mi)
            neigh = Y[S['inv']]
            p_uni = ref(neigh)
            q = QP.center(QP.t_txt_raw(net2, cg)); mimg = QP.t_img_raw(net2, co).mean(0, keepdim=True)
            yc = QP.l2norm(QP.t_img_raw(net2, S['cols']) - mimg); Yn = yc[S['inv']]
            perm = torch.stack([torch.randperm(5, generator=gperm) for _ in range(Bn)]).cuda()
            qsw = torch.roll(torch.arange(Bn, device='cuda'), 1)
            for k in kinds:
                def score(qv, Yv):
                    qq = qv[:, None, :].expand_as(Yv)
                    return (qq * Yv).sum(-1) if k == 'cos' else heads[k](qq, Yv)
                s = score(q, Yn)
                al = torch.softmax(beta[k] * s.float(), 1)
                p_al = ref(neigh, alpha=al)
                A_ = acc[k]
                A_['rows'] += Bn; A_['in_bank'] += int(in_bank.sum()); A_['in_top5'] += int(in5.sum())
                ent = -(al * al.clamp_min(1e-30).log()).sum(1)
                A_['ent_sum'] += float(ent.sum()); A_['maxw_sum'] += float(al.max(1).values.sum())
                cosu = (p_al * p_uni).sum(1)
                A_['all_cos_sum'] += float(cosu.sum())
                if bool(in5.any()):
                    mt_ = (al * tpos.float()).sum(1)[in5]
                    A_['mass_true_sum'] += float(mt_.sum())
                    A_['hit_true'] += int((al.argmax(1)[in5] == tpos.float().argmax(1)[in5]).sum())
                    rk = (al > (al * tpos.float()).sum(1, keepdim=True)).sum(1)[in5].float() + 1
                    A_['rank_true_sum'] += float(rk.sum())
                    rb = S['route_b'] if 'route_b' in S else torch.zeros_like(tpos)
                    tb = (rb & tpos).any(1)
                    for flag, key in ((~tb & in5, 'true_routeA'), (tb & in5, 'true_routeB')):
                        A_[key + '_n'] += int(flag.sum()); A_[key + '_mass'] += float((al * tpos.float()).sum(1)[flag].sum())
                ab = ~in5
                if bool(ab.any()):
                    A_['absent_rows'] += int(ab.sum()); A_['absent_maxw_sum'] += float(al.max(1).values[ab].sum())
                    A_['absent_ent_sum'] += float(ent[ab].sum()); A_['absent_cos_sum'] += float(cosu[ab].sum())
                    A_['absent_l2_sum'] += float((p_al - p_uni).norm(dim=1)[ab].sum())
                if 'route_b' in S:
                    A_['mass_routeB_sum'] += float((al * S['route_b'].float()).sum(1).sum())
                for j in range(5):
                    A_['slot_mass'][j] += float(al[:, j].sum())
                # 4. permutation with identity-aligned alpha; query swap
                Yp = torch.gather(Yn, 1, perm[:, :, None].expand_as(Yn)); np_ = torch.gather(neigh, 1, perm[:, :, None].expand_as(neigh))
                alp = torch.softmax(beta[k] * score(q, Yp).float(), 1)
                A_['perm_alpha_maxdiff'] = max(A_['perm_alpha_maxdiff'], float((alp - torch.gather(al, 1, perm)).abs().max()))
                A_['perm_target_maxdiff'] = max(A_['perm_target_maxdiff'], float((ref(np_, alpha=alp) - p_al).abs().max()))
                als = torch.softmax(beta[k] * score(q[qsw], Yn).float(), 1)
                A_['qswap_l1_sum'] += float((als - al).abs().sum(1).sum())
    out = dict(state=a.state, saved_arm=g_('arm'), stage=a.stage, epoch_reconstructed=ep, n_batches=nb, stage_seed=seed, P=info, beta=beta,
               config={k: getattr(C, k) for k in dir(C) if k.isupper()}, note='evaluator only (truth = caption index // 5)',
               eval_mode_notes='current peer re-encoding mean from an eval-mode forward of the query batch; refiner in eval mode (no dropout)')
    res = {}
    for k, A_ in acc.items():
        r = max(A_['rows'], 1); t5 = max(A_['in_top5'], 1); ab = max(A_['absent_rows'], 1)
        res[k] = dict(rows=A_['rows'], frac_true_in_bank=A_['in_bank'] / r, frac_true_in_top5=A_['in_top5'] / r,
                      in_top5_alpha_mass_on_true=A_['mass_true_sum'] / t5, in_top5_alpha_argmax_hit=A_['hit_true'] / t5,
                      in_top5_mean_rank_of_true=A_['rank_true_sum'] / t5, uniform_mass_on_true=0.2,
                      absent_max_alpha=A_['absent_maxw_sum'] / ab, absent_entropy=A_['absent_ent_sum'] / ab,
                      absent_cos_target_vs_uniform=A_['absent_cos_sum'] / ab, absent_l2_target_vs_uniform=A_['absent_l2_sum'] / ab,
                      all_cos_target_vs_uniform=A_['all_cos_sum'] / r, mean_entropy=A_['ent_sum'] / r, mean_max_alpha=A_['maxw_sum'] / r,
                      perm_alpha_maxdiff=A_['perm_alpha_maxdiff'], perm_target_maxdiff=A_['perm_target_maxdiff'],
                      query_swap_alpha_l1=A_['qswap_l1_sum'] / r, mass_routeB=A_['mass_routeB_sum'] / r,
                      slot_mass=[x / r for x in A_['slot_mass']],
                      truth_routeA=dict(n=A_['true_routeA_n'], alpha_mass=A_['true_routeA_mass'] / max(A_['true_routeA_n'], 1)),
                      truth_routeB=dict(n=A_['true_routeB_n'], alpha_mass=A_['true_routeB_mass'] / max(A_['true_routeB_n'], 1)))
    out['scorers'] = res
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1, default=float)
    print('FIT_EVAL_DONE', a.out, json.dumps({k: {kk: round(v, 4) for kk, v in d.items() if isinstance(v, float)} for k, d in res.items()})[:3000], flush=True)


if __name__ == '__main__':
    main()
