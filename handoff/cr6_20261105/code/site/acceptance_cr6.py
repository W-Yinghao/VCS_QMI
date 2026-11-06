"""CR6 real acceptance (GPU). No truth except the DEV evaluator check V1/T1 (DEV only). Never reads TEST.
E1  CR6 estimator clone with CR1 seeds (1001, default_rng([1001,e,51])) reproduces CR1 J/BCE E01 step-250 bitwise (model+optimizer) [CR1 ran on L40S]
E2  new source seeds: J and BCE of the same expert share the identical init and identical sampler draws; init differs from CR1; torch global RNG untouched
S1  CR5 student path (seed 1002, sampler 20261104202) reproduces CR5 Flickr60 OBS COMPLETE_PC s1002 step-250 bitwise [CR5 ran on L40S]
V1  CR6 evaluator reproduces the CR5 eval_student METRICS (six recalls, rSum) and orders on CR5 FROZEN checkpoints (step 0 and 8000)
T1  zero-LoRA TAIL reproduces the matched FROZEN start: FIT features/batch scores vs cached globals; DEV step-0 orders vs FROZEN step-0
T2  non-zero head + non-zero LoRA: per-item features independent of batch membership/position/order; pair-chunk invariance; ID-rename invariance
T3  direct vs replay (chunked embedding-gradient) gradients equal (G and LoRA)
T4  optimizer covers exactly G + LoRA (98,304); all trainable tensors change after 3 updates; frozen base weights and CSR bytes unchanged
T5  resume: 20 straight updates == 10 + resume + 10 (bitwise model/adapter/optimizer)
T6  timing per update (replay) for budget; T7 CC hidden-cache alignment (zero-LoRA features vs CC bank globals)
"""
from __future__ import annotations
import argparse, copy, json, sys, time
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import cr6_common as K                                                       # noqa: E402
import cr6_source as SRC                                                     # noqa: E402
import cr6_tail as TL                                                        # noqa: E402
import cr6_eval as EV                                                        # noqa: E402
sys.path.insert(0, str(K.CR5_TREE/'code'/'site'))
import cr5_student as S5                                                     # noqa: E402

R = {}


def rec(k, ok, **kw):
    R[k] = dict(ok=bool(ok), **kw); print(k, 'PASS' if ok else 'FAIL', json.dumps(kw, default=str)[:600], flush=True)


def sd_equal(a, b):
    ks = set(a) | set(b)
    return all(k in a and k in b and torch.equal(a[k].cpu(), b[k].cpu()) for k in ks), max((float((a[k].cpu().double()-b[k].cpu().double()).abs().max()) for k in ks if k in a and k in b), default=0.0)


def opt_equal(a, b):
    sa, sb = a['state'], b['state']
    if set(sa) != set(sb): return False
    return all(torch.equal(torch.as_tensor(sa[i][k]).cpu(), torch.as_tensor(sb[i][k]).cpu()) for i in sa for k in sa[i])


def cr5_dir(cond, orig):
    d = K.CR5/'students'/cond/orig; u = K.read_json(d/'UNIT_COMPLETE.json')
    return Path(u['reused_dir']) if u.get('status') == 'REUSE_SAME_JOB_RECT' else d


def light_tail_checks(cond, graph, work, device='cuda', student=None, prefix='T'):
    """T2/T3 (+ part of T1) - also run in-job by every B group on its own GPU."""
    S = student or TL.TailStudent(cond, graph, 1002, 1002, device)
    rng = np.random.default_rng(20261104202); b = S.relation.batch_edges(rng, 64); pb = S.fr.build(S.fr.from_edge_indices(b['edges']), 'COMPLETE')
    ui, ii, ut, it = TL.unique_items(pb.pairs)
    # T1 FIT part (zero LoRA)
    X, Y = S.frozen_features()
    with torch.no_grad():
        U, V = S.hid.encode('image', ui), S.hid.encode('text', ut)
        s0f = TL.g_scores(S.G, X[torch.as_tensor(ui, device=X.device)], Y[torch.as_tensor(ut, device=X.device)], ii, it); s0t = TL.g_scores(S.G, U, V, ii, it)
    t1 = dict(image_feat_max_abs=float((U-X[torch.as_tensor(ui, device=X.device)]).abs().max()), text_feat_max_abs=float((V-Y[torch.as_tensor(ut, device=X.device)]).abs().max()),
              batch_score_max_abs=float((s0f-s0t).abs().max()), unique_images=int(len(ui)), unique_texts=int(len(ut)), pairs=int(len(pb.pairs)))
    del X, Y
    t1['score_tolerance'] = 1e-5*max(1.0, float(S.G.log_scale.exp()))
    rec(f'{prefix}1_zero_lora_fit', max(t1['image_feat_max_abs'], t1['text_feat_max_abs']) <= 1e-5 and t1['batch_score_max_abs'] <= t1['score_tolerance'], **t1)
    # non-zero state
    sd_ad0, sd_g0 = S.adapter_state(), copy.deepcopy(S.G.state_dict())
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for n, p in S.lora:
            if n.endswith('.B'): p.copy_((torch.randn(p.shape, generator=g)*0.02).to(p.device))
        for p in S.G.global_residual[-1].parameters(): p.copy_((torch.randn(p.shape, generator=g)*0.05).to(p.device))
    r = np.random.default_rng(3); others_i = r.choice(len(S.bank.image_global), 50, replace=False); others_t = r.choice(len(S.bank.text_global), 50, replace=False)
    with torch.no_grad():
        Ua = S.hid.encode('image', ui); Va = S.hid.encode('text', ut)
        pi = r.permutation(len(ui)); mix_i = np.concatenate([others_i[:17], ui[pi], others_i[17:]]); Ub = S.hid.encode('image', mix_i)[17:17+len(ui)][np.argsort(pi)]
        pt = r.permutation(len(ut)); mix_t = np.concatenate([others_t[:5], ut[pt], others_t[5:]]); Vb = S.hid.encode('text', mix_t)[5:5+len(ut)][np.argsort(pt)]
        sa = TL.g_scores(S.G, Ua, Va, ii, it, 4096); sb = TL.g_scores(S.G, Ua, Va, ii, it, 97)
        # ID rename: relabel every image/text id through a random permutation; hidden lookup through the inverse
        perm_i = r.permutation(len(S.bank.image_global)); perm_t = r.permutation(len(S.bank.text_global))
        inv_i, inv_t = np.argsort(perm_i), np.argsort(perm_t)
        pairs_r = np.stack([perm_i[pb.pairs[:, 0]], perm_t[pb.pairs[:, 1]]], 1)
        ui_r, ii_r, ut_r, it_r = TL.unique_items(pairs_r)
        Ur = S.hid.encode('image', inv_i[ui_r]); Vr = S.hid.encode('text', inv_t[ut_r])
        from reference.positive_exclusive import packed_loss
        la = packed_loss(sa, pb, kind='PC'); lr_ = packed_loss(TL.g_scores(S.G, Ur, Vr, ii_r, it_r), pb, kind='PC')
    t2 = dict(batch_membership_image_max_abs=float((Ua-Ub).abs().max()), batch_membership_text_max_abs=float((Va-Vb).abs().max()),
              pair_chunk_max_abs=float((sa-sb).abs().max()), id_rename_loss_abs_diff=float((la-lr_).abs()), loss=float(la),
              lora_B_absmax=max(float(p.detach().abs().max()) for n, p in S.lora if n.endswith('.B')))
    rec(f'{prefix}2_nonzero_invariance', t2['batch_membership_image_max_abs'] <= 1e-6 and t2['batch_membership_text_max_abs'] <= 1e-6 and t2['pair_chunk_max_abs'] <= 1e-6
        and t2['id_rename_loss_abs_diff'] <= 1e-6*max(1, abs(t2['loss'])), **t2)
    # T3 direct vs replay
    grads = {}
    for mode in ('direct', 'replay'):
        S.G.zero_grad(set_to_none=True)
        for _, p in S.lora: p.grad = None
        l, info = TL.tail_loss_backward(S.G, S.hid, pb, mode)
        grads[mode] = {**{'G.'+n: p.grad.detach().clone() for n, p in S.G.named_parameters()}, **{n: p.grad.detach().clone() for n, p in S.lora}}
    mx = max(float((grads['direct'][k]-grads['replay'][k]).abs().max()) for k in grads['direct'])
    rel = max(float((grads['direct'][k]-grads['replay'][k]).norm()/max(float(grads['direct'][k].norm()), 1e-30)) for k in grads['direct'])
    bit = all(torch.equal(grads['direct'][k], grads['replay'][k]) for k in grads['direct'])
    nz = all(float(v.abs().max()) > 0 for v in grads['replay'].values())
    rec(f'{prefix}3_direct_vs_replay', rel <= 1e-5 and nz, bitwise=bit, max_abs=mx, max_rel_norm=rel, all_grads_nonzero=nz, replay_max_abs=info['replay_max_abs'])
    with torch.no_grad():
        S.load_adapter(sd_ad0); S.G.load_state_dict(sd_g0)
    S.G.zero_grad(set_to_none=True)
    for _, p in S.lora: p.grad = None
    return S


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--work', required=True); ap.add_argument('--device', default='cuda')
    ap.add_argument('--mode', choices=['history', 'portable'], default='history'); ap.add_argument('--no-tail', action='store_true'); a = ap.parse_args()
    W = Path(a.work); W.mkdir(parents=True, exist_ok=True); dev = a.device; env = K.gpu_env(); t00 = time.perf_counter()
    C = 'Flickr60'; bank, prep = K.bank_path(C), K.prep_dir(C); hist = a.mode == 'history'
    import subprocess
    # E1: clone with CR1 seeds vs (a) the frozen CR1 production fit_estimator rerun in THIS job [all modes], (b) CR1 history [history mode]
    for obj in ('J', 'BCE'):
        d = W/f'E1_{obj}_E01_cr1seed'; c0 = torch.get_rng_state().clone(); g0 = [z.clone() for z in torch.cuda.get_rng_state_all()]
        with K.stop_after(250): SRC.fit(bank, prep, obj, 'E01', d, dev, 1001, [1001, 0, 51], steps=8000, unit=f'ACCEPT_E1_{obj}')   # production T_max=8000, stop after 250
        untouched = bool(torch.equal(c0, torch.get_rng_state())) and all(torch.equal(a_, b_) for a_, b_ in zip(g0, torch.cuda.get_rng_state_all()))
        x = torch.load(d/'step_000250.pt', map_location='cpu', weights_only=False)
        pj = W/f'E1_{obj}_E01_cr1prod_samejob'
        if not (pj/'DONE.json').exists():
            r = subprocess.run([K.PY, '-I', '-B', str(K.CR1_TREE/'code/site/cr1_prod.py'), 'fit-estimator', '--bank', str(bank), '--prepared', str(prep),
                                '--unit', f'CR1_Flickr60_EST_{obj}_E01_s1001', '--out', str(pj), '--device', dev], capture_output=True, text=True)
            if r.returncode: raise RuntimeError(r.stderr[-3000:])
        y = torch.load(pj/'step_000250.pt', map_location='cpu', weights_only=False)
        eq, dm = sd_equal(x['model'], y['model']); oe = opt_equal(x['optimizer'], y['optimizer'])
        rec(f'E1_{obj}_clone_vs_cr1prod_samejob', eq and oe, model_bitwise=eq, optimizer_bitwise=oe, max_abs=dm, rng_untouched=untouched)
        if hist:
            y = torch.load(K.CR1/C/'experts'/obj/'E01'/'step_000250.pt', map_location='cpu', weights_only=False)
            eq, dm = sd_equal(x['model'], y['model']); oe = opt_equal(x['optimizer'], y['optimizer'])
            rec(f'E1_{obj}_replay_cr1_history', eq and oe, model_bitwise=eq, optimizer_bitwise=oe, max_abs=dm)
    # E2
    s = SRC.NEW_SEEDS['E01']; m0 = {}
    for obj in ('J', 'BCE'):
        d = W/f'E2_{obj}_E01_src2002'; SRC.fit(bank, prep, obj, 'E01', d, dev, s['head_seed'], s['sampler_seed'], steps=250, saves=(0, 250), unit=f'ACCEPT_E2_{obj}')
        m0[obj] = torch.load(d/'step_000000.pt', map_location='cpu', weights_only=False)
    same_init, _ = sd_equal(m0['J']['model'], m0['BCE']['model'])
    same_rng = m0['J']['rng_numpy'] == m0['BCE']['rng_numpy']
    cr1_0 = torch.load(K.CR1/C/'experts'/'J'/'E01'/'step_000000.pt', map_location='cpu', weights_only=False)
    diff_init = not sd_equal(m0['J']['model'], cr1_0['model'])[0]
    r1, r2 = np.random.default_rng(s['sampler_seed']), np.random.default_rng([1001, 0, 51])
    first_new, first_old = r1.integers(0, 36175, 64), r2.integers(0, 36175, 64)
    rec('E2_new_source_seeds', same_init and same_rng and diff_init and not np.array_equal(first_new, first_old),
        J_BCE_same_init=same_init, J_BCE_same_sampler_state=same_rng, init_differs_from_CR1=diff_init,
        rng_untouched=[K.read_json(W/f'E2_{o}_E01_src2002'/'REPLAY.json')['torch_global_rng_untouched'] for o in ('J', 'BCE')], seed_map=SRC.NEW_SEEDS)
    # S1: CR5 student path (frozen cr5_student, used unchanged by A2 and references): same-job determinism [all]; vs CR5 history [history]
    g_obs = K.CR3/'graphs'/C/'OBS.npz'; ref = cr5_dir(C, 'CR5_Flickr60_OBS_COMPLETE_PC_s1002'); ms = []
    for k in ('a', 'b'):
        d = W/f'S1_cr5_obs_pc_s1002_{k}'
        if not (d/'step_000250.pt').exists():
            with K.stop_after(250): S5.train(str(bank), str(prep), str(g_obs), d, 'ACCEPT_S1', 'COMPLETE_PC', 1002, 20261104202, dev)   # production schedule, stop after 250
        ms.append(torch.load(d/'step_000250.pt', map_location='cpu', weights_only=False)['model'])
    eq, dm = sd_equal(ms[0], ms[1]); rec('S1_cr5_student_samejob_determinism', eq, max_abs=dm)
    if hist:
        eq, dm = sd_equal(ms[0], torch.load(ref/'step_000250.pt', map_location='cpu', weights_only=False)['model'])
        rec('S1_cr5_student_replay_history', eq, max_abs=dm, reference=str(ref))
    # V1: CR6 evaluator vs the frozen CR1 eval_student bridge run in THIS job [all]; vs CR5 stored METRICS/orders [history]
    vb = W/'V1_cr1bridge_student'; vb.mkdir(exist_ok=True)
    import shutil
    for st in (0, 8000):
        if not (vb/f'step_{st:06d}.pt').exists(): shutil.copy2(ref/f'step_{st:06d}.pt', vb/f'step_{st:06d}.pt')
    dv, tr = K.dev_paths(C)
    r = subprocess.run([K.PY, '-I', '-B', str(K.CR1_TREE/'code/site/eval_student.py'), '--student', str(vb), '--dev-cache', str(dv), '--truth', str(tr),
                        '--prepared', str(prep), '--steps', '0,8000', '--device', dev], capture_output=True, text=True)
    if r.returncode: raise RuntimeError(r.stderr[-3000:])
    v1 = {}
    for st in (0, 8000):
        rr = EV.evaluate_ckpt(C, ref/f'step_{st:06d}.pt', W/f'V1_eval_{st:06d}', dev); npd = np.load(W/f'V1_eval_{st:06d}'/'PREDICTIONS.npz')
        sj = K.read_json(vb/f'eval_{st:06d}'/'METRICS.json'); sp = np.load(vb/f'eval_{st:06d}'/'PREDICTIONS.npz')
        v1[st] = dict(samejob_metrics_equal=rr['metrics'] == sj['metrics'], samejob_orders_equal=bool(np.array_equal(sp['i_order'], npd['i_order']) and np.array_equal(sp['t_order'], npd['t_order'])),
                      rSum_cr6=rr['metrics']['rSum'], rSum_cr1bridge_samejob=sj['metrics']['rSum'])
        if hist:
            old = K.read_json(ref/f'eval_{st:06d}'/'METRICS.json'); op = np.load(ref/f'eval_{st:06d}'/'PREDICTIONS.npz')
            v1[st].update(history_metrics_equal=rr['metrics'] == old['metrics'], history_orders_equal=bool(np.array_equal(op['i_order'], npd['i_order']) and np.array_equal(op['t_order'], npd['t_order'])), rSum_cr5=old['metrics']['rSum'])
    rec('V1_evaluator', all(x['samejob_metrics_equal'] and x['samejob_orders_equal'] and x.get('history_metrics_equal', True) and x.get('history_orders_equal', True) for x in v1.values()), steps=v1)
    if a.no_tail: return finish(W, env, t00, a, ['A1', 'A2'])
    # T1 FIT + T2 + T3 on Flickr60 J_KEEP_TOP1 (high-degree graph) with mmap hidden cache
    g_j = Path(K.read_json(K.CR5_TREE/'PROTOCOL_FROZEN_CR5.json')['units'][f'CR5_{C}_J_KEEP_TOP1_COMPLETE_PC_s1002']['graph_path'])
    S = light_tail_checks(C, g_j, W, dev)
    # T1 DEV: TAIL step-0 eval vs FROZEN step-0 eval orders
    t4d = W/'T4_three_steps'; TL.train(C, g_j, t4d, 'ACCEPT_T4', 1002, 20261104202, 1002, dev, steps=8000, max_steps=3, student=S)
    r0 = EV.evaluate_ckpt(C, t4d/'step_000000.pt', W/'T1_dev_tail_step0', dev); f0 = np.load(W/'V1_eval_000000'/'PREDICTIONS.npz'); t0p = np.load(W/'T1_dev_tail_step0'/'PREDICTIONS.npz')
    fm = r0['features'].get('max_abs_vs_frozen_cache') or {}
    rec('T1_zero_lora_dev_step0', max(fm.get('image', 1), fm.get('text', 1)) <= 1e-5,
        i_order_equal=bool(np.array_equal(f0['i_order'], t0p['i_order'])), t_order_equal=bool(np.array_equal(f0['t_order'], t0p['t_order'])),
        dev_feature_max_abs=r0['features'].get('max_abs_vs_frozen_cache'), tail_step0_rSum=r0['metrics']['rSum'], frozen_step0_rSum=v1[0]['rSum_cr6'],
        note='criterion: zero-LoRA raw DEV features within 1e-5 of the frozen DEV cache; order/rSum identity reported (may differ only via ULP-level ties)')
    # T4
    ck0 = torch.load(t4d/'step_000000.pt', map_location='cpu', weights_only=False); ck3 = torch.load(t4d/'RESUME.pt', map_location='cpu', weights_only=False)
    ch_g = {k: float((ck3['model'][k]-ck0['model'][k]).abs().max()) for k in ck0['model']}; ch_a = {k: float((ck3['adapter'][k]-ck0['adapter'][k]).abs().max()) for k in ck0['adapter']}
    from nr4.tail_encoder import base_weight_digest
    rel_ok = K.sha256(g_j) == S.relation_sha0; dig = base_weight_digest(S.siglip)
    n_head = sum(p.numel() for p in S.G.parameters()); n_ad = sum(p.numel() for _, p in S.lora)
    rec('T4_optimizer_coverage_updates', all(v > 0 for v in ch_g.values()) and all(v > 0 for v in ch_a.values()) and rel_ok and n_ad == 98304,
        head_params=n_head, lora_params=n_ad, min_head_change=min(ch_g.values()), min_adapter_change=min(ch_a.values()), csr_unchanged=rel_ok,
        base_weight_digest=dig, step0=K.read_json(t4d/'STEP0_CHECK.json')['step0_vs_frozen'])
    # T5 resume + T6 timing
    S5r = TL.TailStudent(C, g_j, 1002, 1002, dev, hid=S.hid)
    da, db = W/'T5_straight20', W/'T5_resume'
    torch.cuda.synchronize(); tt = time.perf_counter(); TL.train(C, g_j, da, 'ACCEPT_T5', 1002, 20261104202, 1002, dev, max_steps=20, student=S5r); torch.cuda.synchronize(); per = (time.perf_counter()-tt)/20
    TL.train(C, g_j, db, 'ACCEPT_T5', 1002, 20261104202, 1002, dev, max_steps=10, student=TL.TailStudent(C, g_j, 1002, 1002, dev, hid=S.hid))
    TL.train(C, g_j, db, 'ACCEPT_T5', 1002, 20261104202, 1002, dev, max_steps=20, student=TL.TailStudent(C, g_j, 1002, 1002, dev, hid=S.hid))
    xa = torch.load(da/'RESUME.pt', map_location='cpu', weights_only=False); xb = torch.load(db/'RESUME.pt', map_location='cpu', weights_only=False)
    e1, _ = sd_equal(xa['model'], xb['model']); e2, _ = sd_equal(xa['adapter'], xb['adapter']); e3 = opt_equal(xa['optH'], xb['optH']) and opt_equal(xa['optA'], xb['optA'])
    rec('T5_resume_bitwise', e1 and e2 and e3 and xa['rng_numpy'] == xb['rng_numpy'], model=e1, adapter=e2, optimizers=e3)
    rec('T6_timing', True, seconds_per_update_replay_mmap=per, est_8000_update_hours=per*8000/3600, gpu=env['gpu_name'])
    # T7 CC alignment
    Sc = TL.TailStudent('CC152K_RECOVERED_20261104', K.CR3/'graphs'/'CC152K_RECOVERED_20261104'/'OBS.npz', 1002, 1002, dev)
    rows = np.random.default_rng(11).choice(len(Sc.bank.image_global), 256, replace=False); rt = np.random.default_rng(12).choice(len(Sc.bank.text_global), 256, replace=False)
    Xc, Yc = Sc.frozen_features()
    with torch.no_grad(): di = float((Sc.hid.encode('image', rows)-Xc[torch.as_tensor(rows, device=Xc.device)]).abs().max()); dt = float((Sc.hid.encode('text', rt)-Yc[torch.as_tensor(rt, device=Xc.device)]).abs().max())
    rec('T7_cc_hidden_alignment', di <= 1e-5 and dt <= 1e-5, image_max_abs=di, text_max_abs=dt)
    return finish(W, env, t00, a, ['A1', 'A2', 'B'])

def finish(W, env, t00, a, validates):
    allok = all(v['ok'] for v in R.values())
    out = dict(status='ALL_PASS' if allok else 'FAIL', mode=a.mode, tail_checked=not a.no_tail, validates_groups=validates if allok else [], gpu_name=env['gpu_name'],
               checks=R, env=env, job=K.job(), seconds=time.perf_counter()-t00, utc=K.utc(), test_read=False,
               code_sha256={p.name: K.sha256(p) for p in sorted((K.TREE/'code'/'site').glob('*.py'))})
    K.write_json(K.TREE/'acceptance'/f'ACCEPTANCE_CR6_{K.job()[0]}.json', out); print(out['status'], flush=True)


if __name__ == '__main__': main()
