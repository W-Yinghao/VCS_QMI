from pathlib import Path
R = Path(__file__).resolve().parents[1]
REP = [
# docstring
('"""R39: R38 train_unit.py + R39 edits',
 '"""R40: R39 train_unit.py + R40 edits (update scope image_only/text_only/both_joint via cfg adaptation -> Student(modalities); N_PRESENT_ONLY\n'
 'in-unit checks (p_d recomputed from the full FIT observed relation, masked-row zero gradient every update, valid-row/unique-query/empty-batch\n'
 'logs); per-modality/head/null pre-clip gradient norms + shared clip multiplier; R40 one-step active-path rule (all units); read-only\n'
 'reused references R39 N-JOINT / R39 ORACLE-FROZEN asserted in-unit (init hashes, student init, batch chain); R39X A1 direction-sum\n'
 'tolerance 1e-5 adopted before any unit; R40 freeze file). Training path otherwise byte-identical to R39.\n'
 'R39: R38 train_unit.py + R39 edits'),
# imports
('import r39ext as R9\n', 'import r39ext as R9\nimport r40ext as R40\nfrom common import FROZEN_FILE, ADAPTATIONS\n'),
# PRESENT stats
("""        stats.update(valid_rows_i2t=float(mi.sum()), valid_rows_t2i=float(mt.sum()), empty_valid_batch_i2t=float(mi.sum() == 0),
                     empty_valid_batch_t2i=float(mt.sum() == 0))""",
 """        stats.update(valid_rows_i2t=float(mi.sum()), valid_rows_t2i=float(mt.sum()), empty_valid_batch_i2t=float(mi.sum() == 0),
                     empty_valid_batch_t2i=float(mt.sum() == 0))
        stats.update(valid_unique_queries_i2t=float(len(np.unique(qi[mi]))), valid_unique_queries_t2i=float(len(np.unique(qc[mt]))),   # R40 logs
                     unique_queries_i2t=float(len(np.unique(qi))), unique_queries_t2i=float(len(np.unique(qc))),
                     present_weight_i2t=float(mi.sum()) / (len(mi) * ctx.present['p_i2t']), present_weight_t2i=float(mt.sum()) / (len(mt) * ctx.present['p_t2i']))"""),
("    ctxs = dict(ai=ai, at=at, ri=ri, rt=rt, dir_losses=(dl_i, dl_t), dirs=dirs)",
 "    ctxs = dict(ai=ai, at=at, ri=ri, rt=rt, dir_losses=(dl_i, dl_t), dirs=dirs, present_masks=(mi, mt) if present_only else None)"),
# student build
("""        stud = R9.build_students(ctx.joint.ck, dev)
        stud_ref = {p: {n: q.detach().clone() for n, q in stud[p].named_parameters()} for p in PEERS}""",
 """        stud = R9.build_students(ctx.joint.ck, dev, ADAPTATIONS[cfg['adaptation']])   # R40: enabled modalities only (frozen modality: parent, requires_grad False)
        stud_ref = {p: {n: q.detach().clone() for n, q in stud[p].named_parameters()} for p in PEERS}"""),
("""                                  trainable_params=int(sum(q.numel() for p in PEERS for q in stud[p].trainable())))""",
 """                                  trainable_params=int(sum(q.numel() for p in PEERS for q in stud[p].trainable())),
                                  adaptation=cfg['adaptation'], modalities=list(ADAPTATIONS[cfg['adaptation']]),
                                  trainable_params_by_modality={m: int(sum(q.numel() for p in PEERS for n, q in stud[p].named_parameters() if q.requires_grad and n.startswith(m + '.'))) for m in ('img', 'txt')},
                                  frozen_modality_params=int(sum(q.numel() for p in PEERS for n, q in stud[p].named_parameters() if not q.requires_grad and n.split('.')[0] not in stud[p].modalities)))"""),
# references (after R38 ref check)
("""        assert r38['reference_init_equal'], f"init differs from reference {ref['unit']}"
""",
 """        assert r38['reference_init_equal'], f"init differs from reference {ref['unit']}"
    assert (cfg['adaptation'] != 'frozen') == joint, (cfg['adaptation'], cfg.get('encoder'))
    xrefs = R40.extra_references(cfg)   # R40: read-only reused references (R39 N-JOINT / R39 ORACLE-FROZEN): init hashes + batch chain
    r40 = dict(adaptation=cfg['adaptation'], references={k: dict(unit=v['unit'], init_equal=init_hash == v['init'], chain_equal_at=[],
                                                               student_init_equal=(state_hash(stud) == v['student_init']) if v['student_init'] else None) for k, v in xrefs.items()},
               grad_split_windows=[])
    for k, v in r40['references'].items():
        assert v['init_equal'], f'head/null init differs from {k} {v["unit"]}'
        assert v['student_init_equal'] in (None, True), f'student init differs from {k} {v["unit"]}'
    if cfg['supervision'] == 'N_PRESENT_ONLY':   # R40: p_d fixed from the full FIT observed relation (recomputed independently, asserted)
        r40['present_check'] = R40.present_recheck(d, ctx.present, D36.null_lists, D36.valid_rows)
        write_json(out / 'PRESENT_CHECK.json', r40['present_check'])
        assert r40['present_check']['passed'], r40['present_check']
        r40['masked_grad'] = dict(updates_checked=0, violations=0, masked_rows=0, valid_rows_nonzero_grad=0)
"""),
# hist r40
("""    hist.setdefault('r39', dict(joint=joint, supervision=cfg['supervision'], lr_encoder_at_readout={}, param_distance={}, grad_split_windows=[]))""",
 """    hist.setdefault('r39', dict(joint=joint, supervision=cfg['supervision'], lr_encoder_at_readout={}, param_distance={}, grad_split_windows=[]))
    hist.setdefault('r40', {})
    for k_, v_ in r40.items():
        if k_ == 'references':
            for rk_, rv_ in v_.items():
                hist['r40'].setdefault('references', {}).setdefault(rk_, rv_)
        elif k_ != 'grad_split_windows' or k_ not in hist['r40']:
            hist['r40'].setdefault(k_, v_)
    gs40 = []"""),
# grad split readout window
("""                    hist['r39']['grad_split_windows'].append(dict(end_step=step, **rec['train_window']['r39_grad_split'])); gs_win = []""",
 """                    hist['r39']['grad_split_windows'].append(dict(end_step=step, **rec['train_window']['r39_grad_split'])); gs_win = []
                if gs40:
                    a_ = np.array(gs40)
                    rec['train_window']['r40_grad_split'] = {nm_: float(a_[:, j_].mean()) for j_, nm_ in enumerate(('head_norm_mean', 'null_norm_mean', 'img_norm_mean', 'txt_norm_mean', 'total_norm_mean', 'clip_coef_mean'))}
                    rec['train_window']['r40_grad_split']['frac_clipped'] = float((a_[:, 5] < 1).mean())
                    hist['r40']['grad_split_windows'].append(dict(end_step=step, **rec['train_window']['r40_grad_split'])); gs40 = []"""),
# extra chain check
("""                if step not in hist['r38']['reference_chain_equal_at']:
                    hist['r38']['reference_chain_equal_at'].append(step)""",
 """                if step not in hist['r38']['reference_chain_equal_at']:
                    hist['r38']['reference_chain_equal_at'].append(step)
            for rk_, rv_ in xrefs.items():   # R40: same batch-index hash chain as the reused R39 N-JOINT / ORACLE-FROZEN unit
                kk = len(hist['batch_hash']) - 1
                assert hist['batch_hash'][-1] == rv_['chain'][kk], f"batch chain differs from {rk_} {rv_['unit']} at step {step}"
                if step not in hist['r40']['references'][rk_]['chain_equal_at']:
                    hist['r40']['references'][rk_]['chain_equal_at'].append(step)"""),
# direction sum tolerance
("""            gc['sum_of_direction_grads_equals_total'] = dict(max_abs_err=err, max_abs_grad=scale, passed=bool(err <= 1e-6 * scale + 1e-12))""",
 """            gc['sum_of_direction_grads_equals_total'] = dict(max_abs_err=err, max_abs_grad=scale, passed=bool(err <= 1e-5 * scale + 1e-12), tol_rel=1e-5,
                                                             note='R40: R39X amendment A1 tolerance adopted before any R40 unit (fp32 reduction order)')"""),
# after loss.backward: present masked check + update-1 grad record (insert right after 'loss.backward()')
("""        opt.zero_grad(set_to_none=True)
        loss.backward()
""",
 """        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cx['present_masks'] is not None:   # R40: masked (absent) rows must get exactly zero dL/dg on every slot, every update
            v_, nm_, nz_ = R40.masked_rows_grad(gl, cx['present_masks'])
            mg = hist['r40']['masked_grad']; mg['updates_checked'] += 1; mg['violations'] += v_; mg['masked_rows'] += nm_; mg['valid_rows_nonzero_grad'] += nz_
            assert v_ == 0, f'N_PRESENT_ONLY: {v_} masked rows with nonzero gradient at step {step}'
        if step == 1 and cfg.get('phase', 1) == 1:   # R40 one-step rule: update-1 gradients (pre-clip) of the declared-active tensors
            act40 = R40.active_named(heads, nulls, stud); grads40 = R40.grad_record(act40)
"""),
# grad split per modality: inside per-peer clip loop
("""            gns.append(float(torch.nn.utils.clip_grad_norm_(ps, cfg['clip_per_peer'])))
            if stud is not None:
                gs_win.append((hn, en, gns[-1], min(1.0, cfg['clip_per_peer'] / (gns[-1] + 1e-6))))""",
 """            nrm = lambda qs: float(torch.sqrt(sum(((q.grad.double() ** 2).sum() for q in qs if q.grad is not None), torch.zeros((), dtype=torch.float64, device=dev))))
            hn40 = nrm(list(heads[p].parameters())); nn40 = nrm([nulls.v[p]]) if nulls is not None else 0.0   # R40 split (pre-clip)
            in40 = nrm([q for n_, q in stud[p].named_parameters() if q.requires_grad and n_.startswith('img.')]) if stud is not None else 0.0
            tn40 = nrm([q for n_, q in stud[p].named_parameters() if q.requires_grad and n_.startswith('txt.')]) if stud is not None else 0.0
            gns.append(float(torch.nn.utils.clip_grad_norm_(ps, cfg['clip_per_peer'])))
            gs40.append((hn40, nn40, in40, tn40, gns[-1], min(1.0, cfg['clip_per_peer'] / (gns[-1] + 1e-6))))
            if stud is not None:
                gs_win.append((hn, en, gns[-1], min(1.0, cfg['clip_per_peer'] / (gns[-1] + 1e-6))))"""),
# snapshot before opt.step + rule after
("""        last_lr_enc = opt.param_groups[1]['lr'] if len(opt.param_groups) > 1 else None
        opt.step()
""",
 """        last_lr_enc = opt.param_groups[1]['lr'] if len(opt.param_groups) > 1 else None
        if step == 1 and cfg.get('phase', 1) == 1:
            before40 = R40.snapshot(act40)
        opt.step()
        if step == 1 and cfg.get('phase', 1) == 1:   # R40 one-step active-path rule (every unit; frozen modality bit-unchanged)
            fz40 = R40.frozen_modality_named(stud)
            o40 = R40.one_step_rule(opt, act40, before40, grads40, fz40, {n_: stud_ref[n_.split('.')[1]][n_.split('.', 2)[2]] for n_, _ in fz40})
            if stud is not None:
                o40['frozen_state_unchanged'] = {p: R9.state_digest(stud[p].frozen_state()) == frozen0[p] for p in PEERS}
                o40['passed'] = bool(o40['passed'] and all(o40['frozen_state_unchanged'].values()))
            o40['frozen_H_cache_unchanged'] = R9.state_digest({f'{s}_{k}_{p}': d[(s, k, p)] for s in ('fit', 'dev') for k in ('img', 'cap') for p in PEERS}) == cache40
            o40['passed'] = bool(o40['passed'] and o40['frozen_H_cache_unchanged'])
            hist['r40']['one_step_check'] = {k_: v_ for k_, v_ in o40.items() if k_ not in ('per_tensor', 'frozen_modality')}
            write_json(out / 'ONE_STEP_CHECK_R40.json', o40)
            assert o40['passed'], f"R40 one-step active-path check failed: { {k_: v_ for k_, v_ in o40.items() if k_ not in ('per_tensor', 'frozen_modality')} }"
"""),
# R39 one-step check: replace the all-changed rule by the R40 rule result (keep R39 record)
("""            oc['passed'] = bool(all(ch[p] == ntr[p] for p in PEERS) and all(fr.values()) and ca and oc['parent_ckpt_sha256_unchanged'])""",
 """            oc['all_trainable_changed_vs_parent_R39_rule'] = bool(all(ch[p] == ntr[p] for p in PEERS))   # R40: recorded only; the R40 rule (ONE_STEP_CHECK_R40.json) decides
            oc['passed'] = bool(o40['passed'] and all(fr.values()) and ca and oc['parent_ckpt_sha256_unchanged'])"""),
# finalize
("""                  r39=phases[-1]['train_history'].get('r39'),""",
 """                  r39=phases[-1]['train_history'].get('r39'), r40=phases[-1]['train_history'].get('r40'), adaptation=cfg['adaptation'],"""),
# check_frozen
("""    frozen = read_json(ROOT / 'PROTOCOL_FROZEN.json')""", """    frozen = read_json(ROOT / FROZEN_FILE)   # R40 freeze file"""),
("""            print('R39_UNIT_DONE', cfg['unit_id']""", """            print('R40_UNIT_DONE', cfg['unit_id']"""),
("""    print('R39_JOB_END failed', failed)""", """    print('R40_JOB_END failed', failed)"""),
]
for t in ('r40f', 'r40c'):
    p = R / t / 'train_unit.py'; s = p.read_text()
    for a, b in REP:
        assert s.count(a) == 1, (t, a[:80], s.count(a))
        s = s.replace(a, b)
    # cache digest before the loop (all units): insert after init_hash line
    a = "    init_hash = state_hash(heads) + (state_hash(nulls) if nulls is not None else '')\n"
    assert s.count(a) == 1
    s = s.replace(a, a + "    cache40 = R9.state_digest({f'{s_}_{k_}_{p_}': d[(s_, k_, p_)] for s_ in ('fit', 'dev') for k_ in ('img', 'cap') for p_ in PEERS})   # R40: frozen-H cache digest (all units)\n")
    p.write_text(s); print(t, 'ok')
