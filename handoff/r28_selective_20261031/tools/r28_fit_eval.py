#!/usr/bin/env python3
"""R28 read-only FIT diagnostics (NEXT_EXPERIMENTS_CN.md §4 items 1-4). EVALUATOR ONLY: this is the one R28 file that reads
the injected-noise pairing (true image of train caption j = j // 5, Flickr). Never imported by training; no training step.

On one saved full state (e.g. the C-native or Q2 final state_last.pt) the next stage is reconstructed exactly as in
training (eval_train -> GMM split, teacher = the other peer in eval mode, C-native C_mix candidates from the teacher's own
queue). The R28 stage objects are refitted with the frozen R28 rules and stage seed (r28_select: Q2 VCS head, cosine anchor
(a, b), residual VCS / BCE heads, shared support calibrator u). Queries = the first --n-batches batches of the stage's
noisy split (fixed order, torch seed 0 for the loader). Current peer's GraphRefiner in eval mode; queries X = the current
peer's eval-mode query embeddings (training uses train-mode embeddings; noted).

 1. oracle routing: alpha_final with u = 1[true image among the 5 candidates] (evaluator) vs C-native uniform, vs u = 1
    (forced weighting, = R27-Q) and vs the learned u; per scorer: target change cos(p, p_uniform), gradient of the native
    t2i SCE w.r.t. the queries (cos / norm ratio vs the uniform gradient) and the first-order change of x . y_true along
    the negative gradient (d_true), split by truth present / absent; fraction of rows whose d_true is below the uniform one.
 2. truth present: alpha = one-hot on the true ID -> h_true (refiner output) vs the raw re-encoded y_true: cos(h_true, y_true),
    x . y_true, x . h_true, x . p_uniform, margins vs the 4 hard negatives in raw (y_j) and refined (h_j) coordinates.
 3. scorers (cos / vcs head / g0 anchor / residual VCS / residual BCE): held-out global P/Q objectives vs argmax / mass on
    the true candidate among the real C_mix candidates; argmax change vs the cosine; fixed hard-candidate set top-1.
 4. support u: stage calibration report (training-visible, with / without the observed pair) and, on the application rows,
    u by truth presence (mean, AUC, mis-fallback = truth present & u < .5, mis-enable = truth absent & u >= .5), and by the
    training-visible observed-pair-in-candidates label.
Usage: r28_fit_eval.py --state S --noise-file N --data-path DP --code-dir CODE_R28 --out OUT.json [--stage A|B] [--n-batches 50]
"""
import argparse, copy, json, os, sys, time


def main():
    p = argparse.ArgumentParser()
    for n in ('state', 'noise-file', 'data-path', 'code-dir', 'out'):
        p.add_argument('--' + n, required=True)
    p.add_argument('--noise-ratio', type=float, default=None)
    p.add_argument('--stage', default='A', choices=['A', 'B'])
    p.add_argument('--n-batches', type=int, default=50)
    a = p.parse_args()
    sys.path.insert(0, os.path.abspath(a.code_dir))
    import numpy as np
    import torch
    import torch.nn.functional as F
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
    from vse import VSE, infonce
    from vocab import deserialize_vocab
    from co_train import eval_train, split_prob

    t_start = time.time()
    st0 = torch.load(a.state, map_location='cpu', weights_only=False)
    so = st0['opt']; g_ = (lambda k: so.get(k) if isinstance(so, dict) else getattr(so, k, None))
    nr = a.noise_ratio if a.noise_ratio is not None else float(g_('noise_ratio'))
    del st0
    opt = r25_run.parse(['--noise_file', a.noise_file, '--noise_ratio', str(nr), '--output_dir', os.path.dirname(os.path.abspath(a.out)),
                         '--resume', a.state, '--data_path', a.data_path])
    opt.arm, opt.lam = 'r1_res_vcs_sel', 0.0
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
    ld = lt.dataset
    seed = SEL.stage_seed(opt.seed, ep, peer)

    # ---- stage objects (the frozen R28 rules)
    t0 = time.time()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed)
        fitd, vald, info, (_, _, r_fit) = QP.stage_data(net2, ld, seed)
        head_vcs, _ = QP.fit_head('vcs', *fitd, seed=seed + 17, steps=QC.STEPS)
        av, bv, aff = SEL.calib_affine(fitd[0], fitd[1])
        res = {}
        for k in SEL.RES_KINDS:
            res[k], _ = SEL.fit_residual(k, *fitd, a=av, b=bv, seed=seed + 17, steps=C.STEPS)
        g0s = SEL.AnchoredScorer(SEL.zero_last(QP.PairHead(dim=fitd[0].shape[1]).to(fitd[0].device)), av, bv).eval()
        torch.manual_seed(seed)
        _, _, rows_all = QP.select_P(ld, net2.avg_clean_prob)
        perm = torch.randperm(len(rows_all)).numpy()
        used = rows_all[perm[:info['nP_fit'] + info['nP_val']]]
        (w, c), cal, _ = SEL.fit_support(net2, ld, used, seed)
    w = w.cuda()
    stage_seconds = time.time() - t0
    glob = {}
    if vald is not None:
        for name, sc in (('vcs', head_vcs), ('g0', g0s), ('res_vcs', res['res_vcs']), ('res_bce', res['res_bce'])):
            glob[name] = QP.heldout_metrics(sc, *vald, gen_seed=seed + 1)
        vcap, vimg, vobs = vald
        bank, cand = SEL.hard_set(vcap, vimg, vobs)
        with torch.no_grad():
            Yh = bank[cand]; qq = vcap[:, None, :].expand_as(Yh); sh = (qq * Yh).sum(-1)
            hl = {'cos': C.BETA_COS * sh, 'vcs': C.BETA_HEAD * head_vcs(qq, Yh),
                  'res_vcs': C.BETA_COS * sh + (C.BETA_COS / av) * res['res_vcs'].res(qq, Yh),
                  'res_bce': C.BETA_COS * sh + (C.BETA_COS / av) * res['res_bce'].res(qq, Yh)}
        hard = {k: SEL.hard_metrics(v.float()) for k, v in hl.items()}
    else:
        hard = {}

    # ---- batches
    ud = ut.dataset
    u_gidx, u_obs = ud.r27_gidx, np.asarray(ud.t2i_index, dtype=np.int64)
    torch.manual_seed(0)
    SC = ('cos', 'vcs', 'res_vcs', 'res_bce')
    ROUTES = ('always', 'oracle', 'support')
    rec = {k: [] for k in ['in5', 'obs_in5', 'u', 'in_bank']}
    per = {f'{k}/{r}': {m: [] for m in ('cos_p', 'gcos', 'gnorm_ratio', 'dtrue', 'p_ytrue')} for k in SC for r in ROUTES}
    uni = {m: [] for m in ('dtrue', 'p_ytrue')}
    sc_rec = {k: {m: [] for m in ('mass_true', 'hit_true', 'maxw', 'argmax_diff_cos')} for k in SC}
    two = {m: [] for m in ('cos_htrue_ytrue', 'x_ytrue', 'x_htrue', 'x_puni', 'margin_raw', 'margin_h', 'cos_hj_yj')}
    nb = 0
    for bi, batch in enumerate(ut):
        if nb >= a.n_batches:
            break
        imgs, ilen, txt, tlen, ids, _cl = batch
        if imgs.size(0) < 2:
            continue
        iu = np.asarray(ids, dtype=np.int64)
        cg = torch.as_tensor(u_gidx[iu], device='cuda'); co = torch.as_tensor(u_obs[iu], device='cuda')
        truth = cg // 5                                                          # EVALUATOR ONLY
        with torch.no_grad():
            xi2, xt2, _ = net2.forward_emb(imgs, txt, tlen, imgae_lengths=ilen)
            S, qs = CS.search_t2i(net2.memory_queue, xt2, SEL.SOURCE, SEL.K, None)
        if S is None:
            continue
        nb += 1
        with torch.no_grad():
            n_, valid = I.usable(net2.memory_queue)
            in_bank = torch.isin(truth, net2.memory_queue.r27_ids['img'][:n_][valid])
            tpos = S['img'] == truth[:, None]; in5 = tpos.any(1)
            obs5 = (S['img'] == co[:, None]).any(1)
            _, X, _, mi, mt = I.forward_emb_ctx(net, imgs, txt, tlen, imgae_lengths=ilen)
            Y = I.reencode(net, 't2i', S['cols'], mi); neigh = Y[S['inv']]
            ytrue = I.reencode(net, 't2i', truth, mi)
            s, q, yc = QP.teacher_scores(net2, S, cg, co, 'cos')
            Yn = yc[S['inv']]; qq = q[:, None, :].expand_as(Yn)
            logits = {'cos': C.BETA_COS * s.float(), 'vcs': C.BETA_HEAD * head_vcs(qq, Yn).float(),
                      'res_vcs': C.BETA_COS * s.float() + (C.BETA_COS / av) * res['res_vcs'].res(qq, Yn).float(),
                      'res_bce': C.BETA_COS * s.float() + (C.BETA_COS / av) * res['res_bce'].res(qq, Yn).float()}
            ov, t1 = SEL.ab_features(S['a_ids'], S['b_ids'])
            u = SEL.u_of(torch.stack([s.max(1).values.double(), ov, t1], 1), w, c).float()
        Bn = X.shape[0]
        for kk, v in (('in5', in5), ('obs_in5', obs5), ('u', u), ('in_bank', in_bank)):
            rec[kk].append(v.float().cpu())

        def grad_of(alpha=None, uu=None):
            with torch.no_grad():
                pp = ref(neigh) if alpha is None else ref(neigh, alpha=alpha, u=uu)
            Xg = X.detach().clone().requires_grad_(True)
            gg, = torch.autograd.grad(infonce(Xg.mm(pp.t())), [Xg])
            return pp, gg
        p_uni, g_uni = grad_of()
        d_uni = -(g_uni * ytrue).sum(1)
        uni['dtrue'].append(d_uni.cpu()); uni['p_ytrue'].append((p_uni * ytrue).sum(1).cpu())
        for k in SC:
            al = torch.softmax(logits[k], 1)
            R_ = sc_rec[k]
            R_['mass_true'].append((al * tpos.float()).sum(1).cpu())
            R_['hit_true'].append(((al.argmax(1) == tpos.float().argmax(1)) & in5).float().cpu())
            R_['maxw'].append(al.max(1).values.cpu())
            R_['argmax_diff_cos'].append((al.argmax(1) != s.argmax(1)).float().cpu())
            for r in ROUTES:
                uu = torch.ones(Bn, device='cuda') if r == 'always' else (in5.float() if r == 'oracle' else u)
                pp, gg = grad_of(al, uu)
                P_ = per[f'{k}/{r}']
                P_['cos_p'].append((pp * p_uni).sum(1).cpu())
                P_['gcos'].append(F.cosine_similarity(gg, g_uni, dim=1).cpu())
                P_['gnorm_ratio'].append((gg.norm(dim=1) / g_uni.norm(dim=1).clamp_min(1e-30)).cpu())
                P_['dtrue'].append((-(gg * ytrue).sum(1)).cpu())
                P_['p_ytrue'].append((pp * ytrue).sum(1).cpu())
        # 2. one-hot alpha -> refined tokens
        with torch.no_grad():
            H = torch.stack([ref(neigh, alpha=F.one_hot(torch.full((Bn,), j, device='cuda'), SEL.K).float()) for j in range(SEL.K)], 1)
            tidx = tpos.float().argmax(1)
            ar = torch.arange(Bn, device='cuda')
            sel = in5
            if bool(sel.any()):
                htrue = H[ar, tidx]
                xr = (X[:, None, :] * neigh).sum(-1); xh = (X[:, None, :] * H).sum(-1)
                oth = ~tpos
                mr = (X * ytrue).sum(1) - xr.masked_fill(~oth, -9.0).max(1).values
                mh = (X * htrue).sum(1) - xh.masked_fill(~oth, -9.0).max(1).values
                for kk, v in (('cos_htrue_ytrue', (htrue * ytrue).sum(1)), ('x_ytrue', (X * ytrue).sum(1)), ('x_htrue', (X * htrue).sum(1)),
                              ('x_puni', (X * p_uni).sum(1)), ('margin_raw', mr), ('margin_h', mh)):
                    two[kk].append(v[sel].cpu())
            two['cos_hj_yj'].append((H * neigh).sum(-1).mean(1).cpu())
    cat = lambda L: torch.cat(L) if L else torch.zeros(0)
    in5 = cat(rec['in5']).bool(); obs5 = cat(rec['obs_in5']).bool(); u = cat(rec['u'])
    pres, absn = in5, ~in5
    mean = lambda v, m=None: (float(v[m].mean()) if m is not None and bool(m.any()) else (float(v.mean()) if m is None and len(v) else None))
    d_uni = cat(uni['dtrue'])
    out1 = {}
    for key, P_ in per.items():
        d = cat(P_['dtrue'])
        e = {}
        for grp, m in (('present', pres), ('absent', absn), ('all', None)):
            e[grp] = dict(cos_target_vs_uniform=mean(cat(P_['cos_p']), m), grad_cos_vs_uniform=mean(cat(P_['gcos']), m),
                          grad_norm_ratio=mean(cat(P_['gnorm_ratio']), m), dtrue=mean(d, m), dtrue_uniform=mean(d_uni, m),
                          dtrue_minus_uniform=mean(d - d_uni, m),
                          frac_dtrue_below_uniform=mean((d < d_uni).float(), m),
                          target_ytrue=mean(cat(P_['p_ytrue']), m), target_ytrue_uniform=mean(cat(uni['p_ytrue']), m))
        out1[key] = e
    out3 = {}
    for k, R_ in sc_rec.items():
        out3[k] = dict(present_mass_on_true=mean(cat(R_['mass_true']), pres), present_argmax_hit=mean(cat(R_['hit_true']), pres),
                       absent_max_alpha=mean(cat(R_['maxw']), absn), argmax_differs_from_cos=mean(cat(R_['argmax_diff_cos'])),
                       hard_set=hard.get(k), heldout_global=glob.get(k))
    out3['g0'] = dict(heldout_global=glob.get('g0'), note='g0 = a s + b; its argmax equals the cosine argmax (a > 0)')
    out2 = {k: (float(cat(v).mean()) if len(cat(v)) else None) for k, v in two.items()}
    mr_, mh_ = cat(two['margin_raw']), cat(two['margin_h'])
    out2.update(frac_margin_raw_pos=float((mr_ > 0).float().mean()) if len(mr_) else None,
                frac_margin_h_pos=float((mh_ > 0).float().mean()) if len(mh_) else None, rows=int(len(mr_)))

    def auc(score, lab):
        lab = lab.bool()
        if not bool(lab.any()) or bool(lab.all()):
            return None
        o = torch.argsort(score, stable=True); rk = torch.empty(len(score), dtype=torch.float64); rk[o] = torch.arange(1, len(score) + 1, dtype=torch.float64)
        uu, inv = torch.unique(score, return_inverse=True)
        rk = (torch.zeros(len(uu), dtype=torch.float64).index_add_(0, inv, rk) / torch.zeros(len(uu), dtype=torch.float64).index_add_(0, inv, torch.ones(len(score), dtype=torch.float64)))[inv]
        npos, nneg = float(lab.sum()), float((~lab).sum())
        return float((rk[lab].sum() - npos * (npos + 1) / 2) / (npos * nneg))
    out4 = dict(stage_calibration={k: v for k, v in cal.items()},
                application=dict(rows=int(len(u)), frac_truth_in_top5=mean(in5.float()), frac_observed_in_top5=mean(obs5.float()),
                                 u_mean=mean(u), u_mean_truth_present=mean(u, pres), u_mean_truth_absent=mean(u, absn),
                                 auc_u_truth_present=auc(u, in5), misfallback_truth_present_u_lt_half=mean((u < 0.5).float(), pres),
                                 misenable_truth_absent_u_ge_half=mean((u >= 0.5).float(), absn),
                                 u_mean_observed_present=mean(u, obs5), u_mean_observed_absent=mean(u, ~obs5), auc_u_observed_present=auc(u, obs5),
                                 u_hist=[int(((u >= j / 10) & ((u < (j + 1) / 10) if j < 9 else (u <= 1))).sum()) for j in range(10)]))
    out = dict(state=a.state, saved_arm=g_('arm'), stage=a.stage, epoch_reconstructed=ep, n_batches=nb, stage_seed=seed, P=info,
               anchor=dict(a=av, b=bv, **aff), stage_seconds=stage_seconds, wall_seconds=time.time() - t_start,
               config={k: getattr(C, k) for k in dir(C) if k.isupper() and isinstance(getattr(C, k), (int, float, str, bool))},
               note='evaluator only (truth = caption index // 5); stage objects refitted with the frozen R28 rules on the reconstructed next stage',
               eval_mode_notes='queries and candidates from eval-mode forwards; refiner in eval mode (no dropout); u from the stage calibrator',
               item1_routing=out1, item2_onehot_refiner=out2, item3_scorers=out3, item4_support=out4)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1, default=float)
    print('R28_FIT_EVAL_DONE', a.out, json.dumps(out4['application'])[:1500], flush=True)


if __name__ == '__main__':
    main()
