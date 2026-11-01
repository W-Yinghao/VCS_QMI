#!/usr/bin/env python3
"""R30 v1.1 probe v3 (DIAG_PROTOCOL_V3.json; supplementary plan §四-§六), read-only.
Layers A (score space), B (main term only, inherited AdamW, real clip), C (main + native intra/noisy, real clip, inherited
AdamW), each B/C step evaluated on the update batch and on a disjoint held-out FIT batch; groups and the fixed reference list
defined ONCE on the common H parent; hardest-competitor identity audit; per-query score-gradient strengths saved for raw /
calibrated quantiles. v1/v2 outputs and modules are unchanged.
Usage: margin_diag_v3.py --cond f00|f60 --seed S --ref PARENT_STATE --ref-id PARENT_ID --states LABEL=PATH[=UNIT_ID] ...
                         --out-dir DIR --data-path DP [--smoke]"""
import argparse, copy, hashlib, json, math, os, sys, time
from collections import defaultdict
from types import SimpleNamespace
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import margin_diag as V1
import margin_diag_v2 as V2
sys.path.insert(1, V1.CODE_R29)

SEED_TRAIN, CLIP = V2.SEED_TRAIN, V2.CLIP


def fingerprint(models):
    h = hashlib.sha256()
    for m in models:
        for enc in (m.img_enc, m.txt_enc):
            for k, v in enc.state_dict().items():
                h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def dev_rsum_for(path, raw_epoch):
    """full-DEV rSum (peer-averaged) of the checkpoint, from the run's own records; NA if not recorded."""
    d = os.path.dirname(path)
    try:
        if os.path.basename(path) == 'parent_state.pt':
            return json.load(open(os.path.join(d, 'PARENT.json')))['warmup_dev'][-1]['rsum'], 'PARENT.json warmup_dev[-1]'
        if os.path.basename(path).startswith('state_completed'):
            r = json.load(open(os.path.join(d, os.path.basename(path).replace('state_completed', 'E').replace('.pt', '_RECORD.json'))))
            return r['dev']['rsum'], 'E*_RECORD.json'
        h = json.load(open(os.path.join(d, 'EPOCHS.json')))
        e = [x for x in h['cotrain'] if x['epoch'] == raw_epoch]
        return (e[0]['dev']['rsum'], 'EPOCHS.json cotrain raw epoch') if e else (None, 'NA')
    except Exception as ex:
        return None, f'NA ({ex!r})'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cond', required=True, choices=('f00', 'f60'))
    ap.add_argument('--seed', required=True, type=int)
    ap.add_argument('--ref', required=True); ap.add_argument('--ref-id', required=True)
    ap.add_argument('--states', nargs='+', required=True)
    ap.add_argument('--out-dir', required=True); ap.add_argument('--data-path', required=True)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ref = V1.load_reference()
    import r29_align as AL
    import vse
    import r25_co_train as R
    import r27_co_train as R27
    import r27_interface as I
    from vocab import deserialize_vocab
    from data import get_dataset
    import margin_core as MC
    import margin_core_v2 as M2
    import margin_core_v3 as M3
    assert torch.get_default_dtype() == torch.float32
    proto_path = os.path.join(os.path.dirname(HERE), 'DIAG_PROTOCOL_V3.json')
    pr, psrc, pinfo = V1.pair_source(ref)
    OBJ = MC.Objectives(AL, pr, psrc)
    vocab = deserialize_vocab(os.path.join(V1.VOCAB, 'f30k_precomp_vocab.json')); vocab.add_word('<mask>')
    # update batches (v1 rule) and the disjoint held-out batches (next NB*BS indices of the same permutation)
    U, binfo = V1.build_batches(a.cond, a.data_path, vocab)
    caps, ims = get_dataset(a.data_path, 'f30k_precomp', 'train')
    noise = np.load(V1.NOISE[a.cond]).astype(np.int64); true_all = np.arange(len(caps)) // 5
    from data import process_caption
    perm = np.random.default_rng(V1.SEED).permutation(len(caps))
    hidx = perm[V1.NB * V1.BS: 2 * V1.NB * V1.BS].reshape(V1.NB, V1.BS)
    Hb = []
    for row in hidx:
        obs = noise[row]
        toks = [process_caption(vocab, caps[c], False) for c in row]
        lengths = [len(t) for t in toks]
        text = torch.zeros(V1.BS, max(lengths)).long()
        for i, t in enumerate(toks):
            text[i, :len(t)] = t
        img = torch.from_numpy(np.ascontiguousarray(ims[obs], dtype=np.float32)).pin_memory()
        Hb.append(dict(img=img, il=torch.full((V1.BS,), float(img.shape[1])), text=text, lengths=lengths, obs=torch.as_tensor(obs),
                       true=torch.as_tensor(true_all[row]), cap_idx=row))
    for b in U:
        b['img'] = b['img'].pin_memory(); b['gidx'] = torch.as_tensor(b['cap_idx'], dtype=torch.long)
    binfo['heldout_caption_indices_sha256'] = hashlib.sha256(hidx.astype(np.int64).tobytes()).hexdigest()
    binfo['heldout_rule'] = 'next NB*BS indices of the same permutation (disjoint)'
    I.Ctx.set(caps, ims, vocab)
    env = {'gpu': torch.cuda.get_device_name(0), 'host': os.uname().nodename, 'job': os.environ.get('SLURM_JOB_ID'),
           'torch': torch.__version__, 'cudnn': torch.backends.cudnn.version(), 'cudnn_deterministic': True, 'dtype': 'float32'}
    code = {f: V1.sha(os.path.join(HERE, f)) for f in ('margin_core.py', 'margin_core_v2.py', 'margin_core_v3.py', 'margin_diag.py', 'margin_diag_v2.py', 'margin_diag_v3.py')}
    for f in ('r29_align.py', 'vse.py', 'r27_co_train.py', 'r27_interface.py', 'r25_co_train.py', 'co_train.py'):
        code[f] = V1.sha(os.path.join(V1.CODE_R29, f))
    code['objective_checks.py'] = V1.sha(V1.REF_PY); code['DIAG_PROTOCOL_V3.json'] = V1.sha(proto_path)

    st0 = torch.load(a.ref, map_location='cpu', mmap=True, weights_only=False)
    o0 = st0['opt'] if isinstance(st0['opt'], dict) else vars(st0['opt'])
    assert abs(float(o0['noise_ratio']) - (0.0 if a.cond == 'f00' else 0.6)) < 1e-9 and int(o0['seed']) == a.seed or a.smoke, 'reference identity mismatch'
    opt0 = SimpleNamespace(**o0)
    A = vse.VSE(opt0, vocab.word2idx); B = vse.VSE(opt0, vocab.word2idx)

    def refs_needed():
        R.ensure_refiners(opt0, A, B)
        for m in (A, B):
            I.ensure_ids(m.memory_queue)

    # ------------------------------------------------ reference (H parent): groups for U and H batches, fixed list on U
    t0 = time.time()
    SrefU, SrefH = [None] * V1.NB, [None] * V1.NB
    for peer in ('A', 'B'):
        A.img_enc.load_state_dict(st0['model_' + peer]['img_enc']); A.txt_enc.load_state_dict(st0['model_' + peer]['txt_enc'])
        A.val_start()
        for bi in range(V1.NB):
            su = V1.scores(A, U[bi], False).detach(); sh = V1.scores(A, Hb[bi], False).detach()
            SrefU[bi] = su / 2 if SrefU[bi] is None else SrefU[bi] + su / 2
            SrefH[bi] = sh / 2 if SrefH[bi] is None else SrefH[bi] + sh / 2
    clsU, clsH, kfix, counts = [], [], [], defaultdict(int)
    for bi in range(V1.NB):
        for lst, Sref, b, tag in ((clsU, SrefU, U, 'update'), (clsH, SrefH, Hb, 'heldout')):
            c = M2.reference_classes(Sref[bi], b[bi]['obs'].cuda(), b[bi]['true'].cuda())
            lst.append(M2.masks_np(c))
            for d in ('i2t', 't2i'):
                for k, v in c[d].items():
                    counts[f'{tag}:{d}:{k}'] += int(v.sum())
        mg = MC.margins(SrefU[bi], U[bi]['obs'].cuda())
        kfix.append({d: mg[d]['k'] for d in ('i2t', 't2i')})
    reflist = {'update_caption_indices': np.stack([b['cap_idx'] for b in U]).astype(np.int64),
               'k_fixed_i2t': np.stack([k['i2t'].cpu().numpy() for k in kfix]).astype(np.int64),
               'k_fixed_t2i': np.stack([k['t2i'].cpu().numpy() for k in kfix]).astype(np.int64),
               'observed_ids': np.stack([b['obs'].numpy() for b in U]).astype(np.int64)}
    hsh = hashlib.sha256()
    for k in sorted(reflist):
        hsh.update(k.encode()); hsh.update(reflist[k].tobytes())
    reflist_sha = hsh.hexdigest()
    os.makedirs(a.out_dir, exist_ok=True)
    rl_path = os.path.join(a.out_dir, f'REFLIST_{a.ref_id}.npz')
    if not os.path.isfile(rl_path):
        tmp = rl_path[:-4] + f".tmp{os.environ.get('SLURM_JOB_ID', os.getpid())}.npz"
        np.savez(tmp, **reflist); os.replace(tmp, rl_path)     # atomic: parallel jobs of the same parent write identical content
    else:
        old = np.load(rl_path)
        assert all(np.array_equal(old[k], reflist[k]) for k in reflist), 'fixed reference list differs from the saved one'
    ref_sha = V2.sha_big(a.ref)
    pj = os.path.join(os.path.dirname(a.ref), 'PARENT.json')
    parent_rec = json.load(open(pj)) if os.path.isfile(pj) else None
    if parent_rec is not None:
        assert parent_rec['sha256'] == ref_sha and parent_rec['parent_id'] == a.ref_id and parent_rec['warmup_source'] == 'H', 'reference parent identity mismatch'
    ref_meta = {'ref_id': a.ref_id, 'state': a.ref, 'state_sha256': ref_sha, 'smoke_stand_in': a.smoke,
                'parent_json_sha256_match': None if parent_rec is None else parent_rec['sha256'] == ref_sha,
                'parent_training_state_fingerprint': None if parent_rec is None else parent_rec.get('training_state_fingerprint'),
                'rule': 'peer-averaged eval-path scores of the common H parent; groups + fixed list chosen once', 'counts': dict(counts),
                'fixed_list_file': rl_path, 'fixed_list_sha256': reflist_sha, 'seconds': time.time() - t0}
    del SrefU, SrefH, st0
    print('reference', json.dumps({k: v for k, v in ref_meta.items() if k != 'counts'}), flush=True)

    for item in a.states:
        parts = item.split('=')
        label, path = parts[0], parts[1]
        unit_id = parts[2] if len(parts) > 2 else label
        dest = os.path.join(a.out_dir, f'{a.cond}_s{a.seed}__{label}.json')
        if os.path.isfile(dest):
            print('skip existing', dest, flush=True); continue
        t1 = time.time()
        state_sha = V2.sha_big(path)
        pj_ = os.path.join(os.path.dirname(path), 'PARENT.json')
        rec_sha = (json.load(open(pj_))['sha256'] if os.path.basename(path) == 'parent_state.pt' and os.path.isfile(pj_) else None)
        st = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        so = st['opt'] if isinstance(st['opt'], dict) else vars(st['opt'])
        sopt = SimpleNamespace(**so)
        raw_epoch = int(st['epoch'])
        lr, next_epoch = V2.lr_next(sopt, raw_epoch)
        has = {p: V2.load_peer(m, st['model_' + p], R, I, opt0, refs_needed) for p, m in (('A', A), ('B', B))}
        qinfo = {p: V2.queue_info(st['model_' + p], I) for p in ('A', 'B')}
        if any(has.values()):
            I.Ctx.gpu_images()
        fp0 = fingerprint((A, B))
        dsum, dsrc = dev_rsum_for(path, raw_epoch)
        meta = {'unit_id': unit_id, 'label': label, 'path': path, 'state_sha256': state_sha, 'recorded_sha256_match': None if rec_sha is None else rec_sha == state_sha,
                'raw_epoch': raw_epoch,
                'completed_cotrain_epochs': raw_epoch + 1, 'cond': a.cond, 'seed': a.seed, 'state_opt_seed': sopt.seed,
                'state_opt_noise_ratio': sopt.noise_ratio, 'state_opt_r29_arm': getattr(sopt, 'r29_arm', None),
                'state_opt_r30': {k: v for k, v in so.items() if str(k).startswith('r30_')}, 'lr_next': lr, 'next_epoch': next_epoch,
                'noisy_branch_available': has, 'queue': qinfo, 'full_dev_rsum': dsum, 'full_dev_rsum_source': dsrc}
        acc, aud = M2.Acc2(), M3.AuditAcc()
        comp_rows = defaultdict(list)
        gstr = {o: [] for o in MC.OBJECTIVES}; glab = defaultdict(list)
        checks = {'gru_train_vs_eval_forward_maxabs': 0.0, 'restore_recompute_maxabs': 0.0, 'nonfinite_events': 0, 'opt_state': {}}
        Sbar = [None] * V1.NB
        for peer, net, net2 in (('A', A, B), ('B', B, A)):
            s = st['model_' + peer]
            params = net.params
            moments, oinfo = V2.opt_moments(s, params)
            checks['opt_state'][peer] = oinfo
            sd_backup = {'img': copy.deepcopy(net.img_enc.state_dict()), 'txt': copy.deepcopy(net.txt_enc.state_dict())}
            backup = [p.detach().clone() for p in params]
            nopt = copy.copy(sopt); nopt.r27_epoch = next_epoch
            assert getattr(nopt, 'arm', 'a1_iface') == 'a1_iface' and float(getattr(nopt, 'lam', 0.0)) == 0.0
            nopt.arm, nopt.lam, nopt.i2t, nopt.t2i, nopt.k = 'a1_iface', 0.0, True, True, 5
            for bi in range(V1.NB):
                bu, bh = U[bi], Hb[bi]
                obsU, trueU = bu['obs'].cuda(), bu['true'].cuda()
                obsH, trueH = bh['obs'].cuda(), bh['true'].cuda()
                mU, mH = clsU[bi], clsH[bi]
                net.val_start(); net2.val_start()
                if bi == 0:
                    V1.gru_train(net, False)
                    S_eval = V1.scores(net, bu, False).detach()
                V1.gru_train(net, True)
                S = V1.scores(net, bu, True)
                S0 = S.detach()
                if bi == 0:
                    checks['gru_train_vs_eval_forward_maxabs'] = max(checks['gru_train_vs_eval_forward_maxabs'], float((S0 - S_eval).abs().max()))
                with torch.no_grad():
                    S0H = V1.scores(net, bh, False).detach()
                Sbar[bi] = S0 / 2 if Sbar[bi] is None else Sbar[bi] + S0 / 2
                mgU = MC.margins(S0, obsU); tmU = M2.true_margins(S0, obsU, trueU)
                compU = {d: tmU[d]['computable'].cpu().numpy() for d in ('i2t', 't2i')}
                mgH = MC.margins(S0H, obsH); tmH = M2.true_margins(S0H, obsH, trueH)
                compH = {d: tmH[d]['computable'].cpu().numpy() for d in ('i2t', 't2i')}
                a0U, a0H = M3.audit_rows(S0, obsU, trueU), M3.audit_rows(S0H, obsH, trueH)
                grpU, grpH = {}, {}
                for d in ('i2t', 't2i'):
                    grpU[d] = M3.audit_groups(mU[d], a0U[d]); grpH[d] = M3.audit_groups(mH[d], a0H[d])
                    for sname in M2.SUBSETS:
                        glab[sname].append(mU[d][sname])
                l0U, l0H, pg = {}, {}, {}
                # ---------------- layer A (score space) + eval-mode main-term gradients for the X_* extras
                for name in MC.OBJECTIVES:
                    x = S0.clone().requires_grad_()
                    r1, r2 = OBJ.rows(name, x, obsU)
                    G1 = torch.autograd.grad(r1.sum(), x, retain_graph=True)[0]
                    G2 = torch.autograd.grad(r2.sum(), x)[0]
                    l0U[name] = (r1.detach(), r2.detach())
                    with torch.no_grad():
                        h1, h2 = OBJ.rows(name, S0H, obsH)
                    l0H[name] = (h1, h2)
                    gstr[name].append(torch.cat([G1.norm(dim=1), G2.norm(dim=0)]).float().cpu().numpy())
                    for mode in ('joint', 'own'):
                        M2.accumulate(acc, ('A_score_' + mode + '@update', name), MC.first_order(G1, G2, mgU, mode),
                                      M2.first_order_true(G1, G2, tmU, mode), compU, mU, peer)
                    e1, e2 = OBJ.rows(name, S, obsU)
                    g = torch.autograd.grad(e1.sum() + e2.sum(), params, retain_graph=True, allow_unused=True)
                    pg[name] = [x_ if x_ is not None else torch.zeros_like(p) for x_, p in zip(g, params)]
                del S

                def measure(name, deltas, layer, heldout):
                    with torch.no_grad():
                        for p, dd in zip(params, deltas):
                            p.add_(dd)
                        S1 = V1.scores(net, bu, False).detach()
                        S1H = V1.scores(net, bh, False).detach() if heldout else None
                        for p, bk in zip(params, backup):
                            p.copy_(bk)
                    for tag, S1_, S0_, obs_, true_, mg_, tm_, comp_, msk_, l0_, a0_, grp_ in (
                            ('update', S1, S0, obsU, trueU, mgU, tmU, compU, mU, l0U, a0U, grpU),
                            ('heldout', S1H, S0H, obsH, trueH, mgH, tmH, compH, mH, l0H, a0H, grpH)):
                        if S1_ is None:
                            continue
                        if not torch.isfinite(S1_).all():
                            checks['nonfinite_events'] += 1
                        m1 = MC.margins(S1_, obs_); t1_ = M2.true_margins(S1_, obs_, true_)
                        x1, x2 = OBJ.rows(name, S1_, obs_)
                        gz = torch.zeros(S0_.shape[0], dtype=torch.bool, device=S0_.device)
                        per = {'i2t': (m1['i2t']['m'] - mg_['i2t']['m'], x1 - l0_[name][0], gz), 't2i': (m1['t2i']['m'] - mg_['t2i']['m'], x2 - l0_[name][1], gz)}
                        pert = {d: t1_[d]['m'] - tm_[d]['m'] for d in ('i2t', 't2i')}
                        M2.accumulate(acc, (f'{layer}@{tag}', name), per, pert, comp_, msk_, peer)
                        if layer in ('B_main_inherited', 'C_full'):
                            a1_ = M3.audit_rows(S1_, obs_, true_)
                            for d in ('i2t', 't2i'):
                                g_, sel_ = grp_[d]
                                aud.add((f'{layer}@{tag}', name, d), a0_[d], a1_[d], g_, sel_)
                                aud.add((f'{layer}@{tag}', name, 'both'), a0_[d], a1_[d], g_, sel_)

                for name in MC.OBJECTIVES:
                    da, _ = MC.adam1_delta(pg[name], V1.ETA)
                    na = MC.delta_norm(da)
                    measure(name, da, 'X_adam1_fresh', False)
                    measure(name, MC.gd_delta(pg[name], na), 'X_gd', False)
                    del da
                pg.clear()
                # ---------------- train-mode gradients (shared dropout seed) for layers B and C
                net.train_start(); net2.val_start()
                torch.manual_seed(SEED_TRAIN + bi)
                ie, ce, cl = net.forward_emb(bu['img'], bu['text'], bu['lengths'], imgae_lengths=bu['il'])
                sims_l = net.forward_sim(ie, ce, cl)
                ie2, ce2, _ = net.forward_emb(bu['img'], bu['text'], bu['lengths'], imgae_lengths=bu['il'])
                intra = net.criterion(ie.mm(ie2.t()), margin=0.3) + net.criterion(ce.mm(ce2.t()), margin=0.3)
                gi = [x_ if x_ is not None else torch.zeros_like(p) for x_, p in zip(torch.autograd.grad(intra, params, retain_graph=True, allow_unused=True), params)]
                gm = {}
                for name in MC.OBJECTIVES:
                    g = torch.autograd.grad(OBJ.total(name, sims_l, obsU), params, retain_graph=True, allow_unused=True)
                    gm[name] = [x_ if x_ is not None else torch.zeros_like(p) for x_, p in zip(g, params)]
                del ie, ce, ie2, ce2, sims_l, intra
                gn, nst = None, None
                if has[peer]:
                    for p in list(params) + [p for r_ in (net.i2t_refiner_node, net.t2i_refiner_node) for p in r_.parameters()]:
                        p.grad = None
                    torch.manual_seed(SEED_TRAIN + 100000 + bi)
                    iu, cu, _, mi, mt = I.forward_emb_ctx(net, bu['img'], bu['text'], bu['lengths'], imgae_lengths=bu['il'])
                    nst = defaultdict(float)
                    R27.noisy_branches(nopt, net, net2, bu['img'], bu['il'], bu['text'], bu['lengths'], iu, cu, mi, mt, nst, (bu['gidx'].cuda(), obsU), False)
                    gn = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
                    for p in list(params) + [p for r_ in (net.i2t_refiner_node, net.t2i_refiner_node) for p in r_.parameters()]:
                        p.grad = None
                    del iu, cu
                net.img_enc.load_state_dict(sd_backup['img']); net.txt_enc.load_state_dict(sd_backup['txt'])
                net.val_start()
                for name in MC.OBJECTIVES:
                    cB, preB = MC.clip_coef(gm[name], CLIP)
                    dB = M2.adamw_actual_delta(params, [t_ * cB for t_ in gm[name]], moments, lr, wd=oinfo['weight_decay'], betas=tuple(oinfo['betas']), eps=oinfo['eps'])
                    nB = MC.delta_norm(dB)
                    measure(name, dB, 'B_main_inherited', True)
                    del dB
                    tot = [m_ + i_ + (n_ if gn is not None else 0) for m_, i_, n_ in zip(gm[name], gi, gn if gn is not None else gi)]
                    c_mi, nm, ni = M2.flat_cos(gm[name], gi)
                    c_mn, _, nn_ = M2.flat_cos(gm[name], gn) if gn is not None else (None, None, None)
                    c_in = M2.flat_cos(gi, gn)[0] if gn is not None else None
                    cC, preC = MC.clip_coef(tot, CLIP)
                    dC = M2.adamw_actual_delta(params, [t_ * cC for t_ in tot], moments, lr, wd=oinfo['weight_decay'], betas=tuple(oinfo['betas']), eps=oinfo['eps'])
                    nC = MC.delta_norm(dC)
                    comp_rows[name].append({'peer': peer, 'batch': bi, 'g_main': nm, 'g_intra': ni, 'g_noisy': nn_, 'cos_main_intra': c_mi,
                                            'cos_main_noisy': c_mn, 'cos_intra_noisy': c_in,
                                            'B_preclip': preB, 'B_postclip': preB * cB, 'B_clipped': bool(cB < 1), 'B_update_norm': nB,
                                            'C_preclip': preC, 'C_postclip': preC * cC, 'C_clipped': bool(cC < 1), 'C_update_norm': nC,
                                            'noisy_steps': None if nst is None else {k: nst.get(k, 0) for k in ('i2t_n', 't2i_n', 'i2t_no_proto', 't2i_no_proto')}})
                    measure(name, dC, 'C_full', True)
                    del tot, dC
                del gm, gi, gn
                with torch.no_grad():
                    Sc = V1.scores(net, bu, False).detach()
                checks['restore_recompute_maxabs'] = max(checks['restore_recompute_maxabs'], float((Sc - S0).abs().max()))
            net.img_enc.load_state_dict(sd_backup['img']); net.txt_enc.load_state_dict(sd_backup['txt'])
            del moments
        fp1 = fingerprint((A, B))
        checks['model_fingerprint_before'] = fp0; checks['model_fingerprint_after'] = fp1; checks['fingerprint_unchanged'] = fp0 == fp1
        del st
        # ---------------- fixed reference list readouts (peer-averaged eval-path scores of this state)
        fl = defaultdict(lambda: defaultdict(list))
        for bi in range(V1.NB):
            rows = M3.fixed_list_rows(Sbar[bi], U[bi]['obs'].cuda(), kfix[bi])
            for d in ('i2t', 't2i'):
                for k, v in rows[d].items():
                    fl[d][k].append(v.detach().double().cpu().numpy() if v.dtype != torch.bool else v.cpu().numpy())
        flg = {}
        for d in ('i2t', 't2i', 'both'):
            dd = ('i2t', 't2i') if d == 'both' else (d,)
            cat = {k: np.concatenate([np.concatenate(fl[x][k]) for x in dd]) for k in fl['i2t']}
            masks = {sname: np.concatenate([np.concatenate([clsU[bi][x][sname] for bi in range(V1.NB)]) for x in dd]) for sname in M2.SUBSETS}
            for sname, m in masks.items():
                if not m.any():
                    flg[f'{d}|{sname}'] = {'n': 0}; continue
                flg[f'{d}|{sname}'] = {'n': int(m.sum()), **{f'mean_{k}': float(cat[k][m].mean()) for k in cat if k not in ('resid_list', 'resid_pair')},
                                       'max_abs_S_minus_J_minus_gap_list': float(cat['resid_list'][m].max()),
                                       'max_abs_S_minus_J_minus_gap_pair': float(cat['resid_pair'][m].max())}
        out = {}
        for (prefix, p, d, s_), v in acc.result().items():
            mode, name = prefix
            out.setdefault(mode, {}).setdefault(name, {}).setdefault(p, {}).setdefault(d, {})[s_] = v
        audit = {}
        for (mode, name, d, g), v in aud.result().items():
            audit.setdefault(mode, {}).setdefault(name, {}).setdefault(d, {})[g] = v
        comp_sum = {}
        for name, rows in comp_rows.items():
            arr = lambda k: np.array([r[k] for r in rows if r[k] is not None], dtype=np.float64)
            q = lambda x: {f'p{int(t * 100)}': float(np.quantile(x, t)) for t in (0.05, 0.5, 0.95)} if x.size else None
            comp_sum[name] = {'n_batches': len(rows), **{f'mean_{k}': (float(arr(k).mean()) if arr(k).size else None) for k in
                              ('g_main', 'g_intra', 'g_noisy', 'cos_main_intra', 'cos_main_noisy', 'cos_intra_noisy', 'B_preclip', 'B_postclip',
                               'B_update_norm', 'C_preclip', 'C_postclip', 'C_update_norm')},
                              'B_clip_rate': float(np.mean([r['B_clipped'] for r in rows])), 'C_clip_rate': float(np.mean([r['C_clipped'] for r in rows])),
                              'n_batches_g_intra_zero': int((arr('g_intra') == 0).sum()),
                              'n_batches_g_noisy_zero': int((arr('g_noisy') == 0).sum()) if arr('g_noisy').size else None,
                              'ratio_main_intra_quantiles': q(np.array([r['g_main'] / r['g_intra'] for r in rows if r['g_intra']], dtype=np.float64)),
                              'ratio_main_noisy_quantiles': q(np.array([r['g_main'] / r['g_noisy'] for r in rows if r['g_noisy']], dtype=np.float64))}
        npz = os.path.join(a.out_dir, f'{a.cond}_s{a.seed}__{label}__gradstrength.npz')
        np.savez_compressed(npz, **{f'g_{o}': np.concatenate(v) for o, v in gstr.items()}, **{f'grp_{k}': np.concatenate(v) for k, v in glab.items()})
        meta.update(seconds=time.time() - t1, checks=checks, n_params=sum(p.numel() for p in A.params), eta_extra=V1.ETA,
                    pair_source=psrc, pair_source_info=pinfo, batches=binfo, env=env, code_sha256=code, reference=ref_meta,
                    seed_train=SEED_TRAIN, clip=CLIP, protocol_sha256=code['DIAG_PROTOCOL_V3.json'], gradstrength_file=npz)
        r = {'meta': meta, 'results': out, 'audit': audit, 'fixed_list': flg, 'grad_composition': comp_sum, 'grad_composition_per_batch': comp_rows}
        json.dump(r, open(dest + '.tmp', 'w')); os.replace(dest + '.tmp', dest)
        print('done', label, f"{meta['seconds']:.0f}s", json.dumps({k: v for k, v in checks.items() if k not in ('opt_state', 'model_fingerprint_before', 'model_fingerprint_after')}), flush=True)


if __name__ == '__main__':
    main()
