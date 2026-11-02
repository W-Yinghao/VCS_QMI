#!/usr/bin/env python3
"""R30 probe v4 = v3 layers B and C with the ACTUAL-CELL static coefficient (review package 2026-11-02, SERVER_PLAN §1 R30).
Read-only. Imports the v3 modules from handoff/r30_diag_20261101/tools unchanged; nothing there is modified.

v3 (margin_diag_v3.py) computes B (main term only + inherited AdamW + real clip 2.0) and C (main + native intra + noisy, same
optimizer/clip) with the RAW main-term gradient for every objective (no static coefficient). Production training uses
loss_1 = c * raw (r30_align.main_term, fit_static) before the clip, so clipping, the AdamW step and the C-layer composition
depend on c. Here the main-term gradient is multiplied by c_actual(o, state) before clip / AdamW / composition:
  c_actual(o, state) = the state's own unit coefficient (opt.r30_coef) when o is that unit's objective;
                       otherwise the COEFFICIENTS.json entry of the state's OWN (cond, seed) H parent (L -> Ls), N/B/H = 1.
(v3's 'calibrated' gradient-strength column instead used the s1001 parent of the condition for every seed.)
Objectives with c_actual = 1 are numerically the v3 computation and are not rerun, except 'N' on the first state of each job
as a reproduction control. Layers A / X of v3 are step-normalized (scale-free) and are not rerun. Same fixed FIT update /
held-out batches, same reference groups (common H parent of cond/seed), same dropout seeds, same audit as v3.
Usage: margin_diag_v4_actualscale.py --cond --seed --ref --ref-id --coef COEFFICIENTS.json --out-dir DIR --data-path DP
                                     --states LABEL=PATH[=UNIT_ID] ...
"""
import argparse, copy, hashlib, json, os, sys, time
from collections import defaultdict
from types import SimpleNamespace
import numpy as np
import torch

V3DIR = '/home/infres/yinwang/CS_QMI/handoff/r30_diag_20261101/tools'
sys.path.insert(0, V3DIR)
import margin_diag as V1
import margin_diag_v2 as V2
sys.path.insert(1, V1.CODE_R29)

SEED_TRAIN, CLIP = V2.SEED_TRAIN, V2.CLIP
OBJ_OF_R30 = {'H': 'H_hinge', 'L': 'L', 'N': 'N', 'B': 'B', 'PN': 'PN', 'PV': 'PV', 'PB': 'PB'}
COEF_KEY = {'L': 'Ls', 'PN': 'PN', 'PV': 'PV', 'PB': 'PB'}


def coef_map(so, cell_coefs, objectives):
    own_obj = OBJ_OF_R30.get(so.get('r30_objective')) if so.get('r30_stage') == 'unit' else None
    own_c = so.get('r30_coef')
    out = {}
    for o in objectives:
        if o == own_obj:
            out[o] = {'c': 1.0 if own_c is None else float(own_c), 'source': f"unit_own r30_scale={so.get('r30_scale')} r30_coef={own_c}"}
        elif o in COEF_KEY:
            out[o] = {'c': float(cell_coefs[COEF_KEY[o]]['c']), 'source': f'cell_H_parent_calibration {COEF_KEY[o]}'}
        else:
            out[o] = {'c': 1.0, 'source': 'fixed 1 (N/B/H by design)'}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cond', required=True, choices=('f00', 'f60'))
    ap.add_argument('--seed', required=True, type=int)
    ap.add_argument('--ref', required=True); ap.add_argument('--ref-id', required=True)
    ap.add_argument('--coef', required=True)
    ap.add_argument('--states', nargs='+', required=True)
    ap.add_argument('--out-dir', required=True); ap.add_argument('--data-path', required=True)
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
    from data import get_dataset, process_caption
    import margin_core as MC
    import margin_core_v2 as M2
    import margin_core_v3 as M3
    assert torch.get_default_dtype() == torch.float32
    COEF = json.load(open(a.coef))
    cell_parent = f'wp_{a.cond}_s{a.seed}_H_e5'
    cell_coefs = COEF['parents'][cell_parent]['coefficients']
    pr, psrc, pinfo = V1.pair_source(ref)
    OBJ = MC.Objectives(AL, pr, psrc)
    vocab = deserialize_vocab(os.path.join(V1.VOCAB, 'f30k_precomp_vocab.json')); vocab.add_word('<mask>')
    U, binfo = V1.build_batches(a.cond, a.data_path, vocab)
    caps, ims = get_dataset(a.data_path, 'f30k_precomp', 'train')
    noise = np.load(V1.NOISE[a.cond]).astype(np.int64); true_all = np.arange(len(caps)) // 5
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
    I.Ctx.set(caps, ims, vocab)
    env = {'gpu': torch.cuda.get_device_name(0), 'host': os.uname().nodename, 'job': os.environ.get('SLURM_JOB_ID'),
           'torch': torch.__version__, 'cudnn': torch.backends.cudnn.version()}
    code = {f: V1.sha(os.path.join(V3DIR, f)) for f in ('margin_core.py', 'margin_core_v2.py', 'margin_core_v3.py', 'margin_diag.py', 'margin_diag_v2.py', 'margin_diag_v3.py')}
    code['margin_diag_v4_actualscale.py'] = V1.sha(os.path.abspath(__file__)); code['COEFFICIENTS.json'] = V1.sha(a.coef)

    st0 = torch.load(a.ref, map_location='cpu', mmap=True, weights_only=False)
    o0 = st0['opt'] if isinstance(st0['opt'], dict) else vars(st0['opt'])
    assert abs(float(o0['noise_ratio']) - (0.0 if a.cond == 'f00' else 0.6)) < 1e-9 and int(o0['seed']) == a.seed
    opt0 = SimpleNamespace(**o0)
    A = vse.VSE(opt0, vocab.word2idx); B = vse.VSE(opt0, vocab.word2idx)

    def refs_needed():
        R.ensure_refiners(opt0, A, B)
        for m in (A, B):
            I.ensure_ids(m.memory_queue)

    # reference groups exactly as v3 (common H parent of cond/seed; peer-averaged eval-path scores); fixed list checked
    SrefU, SrefH = [None] * V1.NB, [None] * V1.NB
    for peer in ('A', 'B'):
        A.img_enc.load_state_dict(st0['model_' + peer]['img_enc']); A.txt_enc.load_state_dict(st0['model_' + peer]['txt_enc'])
        A.val_start()
        for bi in range(V1.NB):
            su = V1.scores(A, U[bi], False).detach(); sh = V1.scores(A, Hb[bi], False).detach()
            SrefU[bi] = su / 2 if SrefU[bi] is None else SrefU[bi] + su / 2
            SrefH[bi] = sh / 2 if SrefH[bi] is None else SrefH[bi] + sh / 2
    clsU, clsH, kfix = [], [], []
    for bi in range(V1.NB):
        clsU.append(M2.masks_np(M2.reference_classes(SrefU[bi], U[bi]['obs'].cuda(), U[bi]['true'].cuda())))
        clsH.append(M2.masks_np(M2.reference_classes(SrefH[bi], Hb[bi]['obs'].cuda(), Hb[bi]['true'].cuda())))
        kfix.append(MC.margins(SrefU[bi], U[bi]['obs'].cuda()))
    v3list = np.load(os.path.join(os.path.dirname(V3DIR), 'states_out_v3', f'REFLIST_{a.ref_id}.npz'))
    kf_same = all(np.array_equal(v3list[f'k_fixed_{d}'], np.stack([k[d]['k'].cpu().numpy() for k in kfix])) for d in ('i2t', 't2i'))
    ref_meta = {'ref_id': a.ref_id, 'state_sha256': V2.sha_big(a.ref), 'v3_fixed_list_k_identical': bool(kf_same)}
    del SrefU, SrefH, st0
    print('reference', json.dumps(ref_meta), flush=True)
    os.makedirs(a.out_dir, exist_ok=True)

    for si, item in enumerate(a.states):
        parts = item.split('=')
        label, path = parts[0], parts[1]
        unit_id = parts[2] if len(parts) > 2 else label
        dest = os.path.join(a.out_dir, f'{a.cond}_s{a.seed}__{label}.json')
        if os.path.isfile(dest):
            print('skip existing', dest, flush=True); continue
        t1 = time.time()
        st = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        so = st['opt'] if isinstance(st['opt'], dict) else vars(st['opt'])
        sopt = SimpleNamespace(**so)
        raw_epoch = int(st['epoch'])
        lr, next_epoch = V2.lr_next(sopt, raw_epoch)
        cm_all = coef_map(so, cell_coefs, MC.OBJECTIVES)
        run_objs = [o for o in MC.OBJECTIVES if cm_all[o]['c'] != 1.0]
        control = si == 0 and 'N' not in run_objs
        if control:
            run_objs = ['N'] + run_objs
        has = {p: V2.load_peer(m, st['model_' + p], R, I, opt0, refs_needed) for p, m in (('A', A), ('B', B))}
        if any(has.values()):
            I.Ctx.gpu_images()
        meta = {'unit_id': unit_id, 'label': label, 'path': path, 'state_sha256': V2.sha_big(path), 'raw_epoch': raw_epoch,
                'completed_cotrain_epochs': raw_epoch + 1, 'cond': a.cond, 'seed': a.seed, 'lr_next': lr, 'next_epoch': next_epoch,
                'state_opt_r30': {k: v for k, v in so.items() if str(k).startswith('r30_')}, 'noisy_branch_available': has,
                'cell_coefficient_parent': cell_parent, 'coefficients_actual': cm_all, 'objectives_run': run_objs,
                'n_control_objective': control,
                'v3_counterpart': os.path.join(os.path.dirname(V3DIR), 'states_out_v3', f'{a.cond}_s{a.seed}__{label}.json')}
        acc, aud = M2.Acc2(), M3.AuditAcc()
        comp_rows = defaultdict(list)
        checks = {'restore_recompute_maxabs': 0.0, 'nonfinite_events': 0}
        for peer, net, net2 in (('A', A, B), ('B', B, A)):
            s = st['model_' + peer]
            params = net.params
            moments, oinfo = V2.opt_moments(s, params)
            sd_backup = {'img': copy.deepcopy(net.img_enc.state_dict()), 'txt': copy.deepcopy(net.txt_enc.state_dict())}
            backup = [p.detach().clone() for p in params]
            nopt = copy.copy(sopt); nopt.r27_epoch = next_epoch
            nopt.arm, nopt.lam, nopt.i2t, nopt.t2i, nopt.k = 'a1_iface', 0.0, True, True, 5
            for bi in range(V1.NB):
                bu, bh = U[bi], Hb[bi]
                obsU, trueU = bu['obs'].cuda(), bu['true'].cuda()
                obsH, trueH = bh['obs'].cuda(), bh['true'].cuda()
                mU, mH = clsU[bi], clsH[bi]
                net.val_start(); net2.val_start()
                with torch.no_grad():
                    V1.gru_train(net, False)
                    S0 = V1.scores(net, bu, False).detach()
                    S0H = V1.scores(net, bh, False).detach()
                mgU = MC.margins(S0, obsU); tmU = M2.true_margins(S0, obsU, trueU)
                compU = {d: tmU[d]['computable'].cpu().numpy() for d in ('i2t', 't2i')}
                mgH = MC.margins(S0H, obsH); tmH = M2.true_margins(S0H, obsH, trueH)
                compH = {d: tmH[d]['computable'].cpu().numpy() for d in ('i2t', 't2i')}
                a0U, a0H = M3.audit_rows(S0, obsU, trueU), M3.audit_rows(S0H, obsH, trueH)
                grpU = {d: M3.audit_groups(mU[d], a0U[d]) for d in ('i2t', 't2i')}
                grpH = {d: M3.audit_groups(mH[d], a0H[d]) for d in ('i2t', 't2i')}
                l0U, l0H = {}, {}
                with torch.no_grad():
                    for name in run_objs:
                        l0U[name] = OBJ.rows(name, S0, obsU); l0H[name] = OBJ.rows(name, S0H, obsH)

                def measure(name, deltas, layer):
                    with torch.no_grad():
                        for p, dd in zip(params, deltas):
                            p.add_(dd)
                        S1 = V1.scores(net, bu, False).detach()
                        S1H = V1.scores(net, bh, False).detach()
                        for p, bk in zip(params, backup):
                            p.copy_(bk)
                    for tag, S1_, S0_, obs_, true_, mg_, tm_, comp_, msk_, l0_, a0_, grp_ in (
                            ('update', S1, S0, obsU, trueU, mgU, tmU, compU, mU, l0U, a0U, grpU),
                            ('heldout', S1H, S0H, obsH, trueH, mgH, tmH, compH, mH, l0H, a0H, grpH)):
                        if not torch.isfinite(S1_).all():
                            checks['nonfinite_events'] += 1
                        m1 = MC.margins(S1_, obs_); t1_ = M2.true_margins(S1_, obs_, true_)
                        x1, x2 = OBJ.rows(name, S1_, obs_)
                        gz = torch.zeros(S0_.shape[0], dtype=torch.bool, device=S0_.device)
                        per = {'i2t': (m1['i2t']['m'] - mg_['i2t']['m'], x1 - l0_[name][0], gz), 't2i': (m1['t2i']['m'] - mg_['t2i']['m'], x2 - l0_[name][1], gz)}
                        pert = {d: t1_[d]['m'] - tm_[d]['m'] for d in ('i2t', 't2i')}
                        M2.accumulate(acc, (f'{layer}@{tag}', name), per, pert, comp_, msk_, peer)
                        a1_ = M3.audit_rows(S1_, obs_, true_)
                        for d in ('i2t', 't2i'):
                            g_, sel_ = grp_[d]
                            aud.add((f'{layer}@{tag}', name, d), a0_[d], a1_[d], g_, sel_)
                            aud.add((f'{layer}@{tag}', name, 'both'), a0_[d], a1_[d], g_, sel_)

                # train-mode gradients (shared dropout seed, identical to v3)
                net.train_start(); net2.val_start()
                torch.manual_seed(SEED_TRAIN + bi)
                ie, ce, cl = net.forward_emb(bu['img'], bu['text'], bu['lengths'], imgae_lengths=bu['il'])
                sims_l = net.forward_sim(ie, ce, cl)
                ie2, ce2, _ = net.forward_emb(bu['img'], bu['text'], bu['lengths'], imgae_lengths=bu['il'])
                intra = net.criterion(ie.mm(ie2.t()), margin=0.3) + net.criterion(ce.mm(ce2.t()), margin=0.3)
                gi = [x_ if x_ is not None else torch.zeros_like(p) for x_, p in zip(torch.autograd.grad(intra, params, retain_graph=True, allow_unused=True), params)]
                gm = {}
                for name in run_objs:
                    g = torch.autograd.grad(OBJ.total(name, sims_l, obsU), params, retain_graph=True, allow_unused=True)
                    gm[name] = [x_ if x_ is not None else torch.zeros_like(p) for x_, p in zip(g, params)]
                del ie, ce, ie2, ce2, sims_l, intra
                gn = None
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
                for name in run_objs:
                    c = cm_all[name]['c']
                    gs = [t_ * c for t_ in gm[name]]
                    cB, preB = MC.clip_coef(gs, CLIP)
                    dB = M2.adamw_actual_delta(params, [t_ * cB for t_ in gs], moments, lr, wd=oinfo['weight_decay'], betas=tuple(oinfo['betas']), eps=oinfo['eps'])
                    nB = MC.delta_norm(dB)
                    measure(name, dB, 'B_main_actualscale')
                    del dB
                    tot = [m_ + i_ + (n_ if gn is not None else 0) for m_, i_, n_ in zip(gs, gi, gn if gn is not None else gi)]
                    c_mi, nm, ni = M2.flat_cos(gs, gi)
                    c_mn, _, nn_ = M2.flat_cos(gs, gn) if gn is not None else (None, None, None)
                    cC, preC = MC.clip_coef(tot, CLIP)
                    dC = M2.adamw_actual_delta(params, [t_ * cC for t_ in tot], moments, lr, wd=oinfo['weight_decay'], betas=tuple(oinfo['betas']), eps=oinfo['eps'])
                    nC = MC.delta_norm(dC)
                    comp_rows[name].append({'peer': peer, 'batch': bi, 'c': c, 'g_main_raw': nm / c if c else None, 'g_main_scaled': nm, 'g_intra': ni,
                                            'g_noisy': nn_, 'cos_main_intra': c_mi, 'cos_main_noisy': c_mn,
                                            'B_preclip': preB, 'B_postclip': preB * cB, 'B_clipped': bool(cB < 1), 'B_update_norm': nB,
                                            'C_preclip': preC, 'C_postclip': preC * cC, 'C_clipped': bool(cC < 1), 'C_update_norm': nC})
                    measure(name, dC, 'C_full_actualscale')
                    del tot, dC, gs
                del gm, gi, gn
                with torch.no_grad():
                    Sc = V1.scores(net, bu, False).detach()
                checks['restore_recompute_maxabs'] = max(checks['restore_recompute_maxabs'], float((Sc - S0).abs().max()))
            net.img_enc.load_state_dict(sd_backup['img']); net.txt_enc.load_state_dict(sd_backup['txt'])
            del moments
        del st
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
            comp_sum[name] = {'n_batches': len(rows), 'c': rows[0]['c'],
                              **{f'mean_{k}': (float(arr(k).mean()) if arr(k).size else None) for k in
                                 ('g_main_raw', 'g_main_scaled', 'g_intra', 'g_noisy', 'cos_main_intra', 'cos_main_noisy', 'B_preclip', 'B_postclip',
                                  'B_update_norm', 'C_preclip', 'C_postclip', 'C_update_norm')},
                              'B_clip_rate': float(np.mean([r['B_clipped'] for r in rows])), 'C_clip_rate': float(np.mean([r['C_clipped'] for r in rows])),
                              'median_ratio_scaledmain_intra': float(np.median([r['g_main_scaled'] / r['g_intra'] for r in rows if r['g_intra']])),
                              'median_ratio_scaledmain_noisy': (float(np.median([r['g_main_scaled'] / r['g_noisy'] for r in rows if r['g_noisy']]))
                                                                if any(r['g_noisy'] for r in rows) else None)}
        meta.update(seconds=time.time() - t1, checks=checks, env=env, code_sha256=code, reference=ref_meta, batches=binfo,
                    pair_source=psrc, seed_train=SEED_TRAIN, clip=CLIP)
        r = {'meta': meta, 'results': out, 'audit': audit, 'grad_composition': comp_sum, 'grad_composition_per_batch': comp_rows}
        json.dump(r, open(dest + '.tmp', 'w')); os.replace(dest + '.tmp', dest)
        print('done', label, run_objs, f"{meta['seconds']:.0f}s", json.dumps(checks), flush=True)


if __name__ == '__main__':
    main()
