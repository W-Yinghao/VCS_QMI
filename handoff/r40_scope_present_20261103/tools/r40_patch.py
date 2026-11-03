"""R40 code-derivation patch data (extracted from the one-off edit scripts tools/_edit_*.py that produced r40f/r40c from r39f/r39c).
cpu_test re-applies these edits to the frozen R39 sources and requires equality with the R40 files (common.py: line allowlist)."""
TRAIN_UNIT_REP = [('"""R39: R38 train_unit.py + R39 edits',
  '"""R40: R39 train_unit.py + R40 edits (update scope image_only/text_only/both_joint via cfg adaptation -> Student(modalities); N_PRESENT_ONLY\n'
  'in-unit checks (p_d recomputed from the full FIT observed relation, masked-row zero gradient every update, valid-row/unique-query/empty-batch\n'
  'logs); per-modality/head/null pre-clip gradient norms + shared clip multiplier; R40 one-step active-path rule (all units); read-only\n'
  'reused references R39 N-JOINT / R39 ORACLE-FROZEN asserted in-unit (init hashes, student init, batch chain); R39X A1 direction-sum\n'
  'tolerance 1e-5 adopted before any unit; R40 freeze file). Training path otherwise byte-identical to R39.\n'
  'R39: R38 train_unit.py + R39 edits'),
 ('import r39ext as R9\n', 'import r39ext as R9\nimport r40ext as R40\nfrom common import FROZEN_FILE, ADAPTATIONS\n'),
 ('        stats.update(valid_rows_i2t=float(mi.sum()), valid_rows_t2i=float(mt.sum()), empty_valid_batch_i2t=float(mi.sum() == 0),\n                     empty_valid_batch_t2i=float(mt.sum() == 0))',
  '        stats.update(valid_rows_i2t=float(mi.sum()), valid_rows_t2i=float(mt.sum()), empty_valid_batch_i2t=float(mi.sum() == 0),\n'
  '                     empty_valid_batch_t2i=float(mt.sum() == 0))\n'
  '        stats.update(valid_unique_queries_i2t=float(len(np.unique(qi[mi]))), valid_unique_queries_t2i=float(len(np.unique(qc[mt]))),   # R40 logs\n'
  '                     unique_queries_i2t=float(len(np.unique(qi))), unique_queries_t2i=float(len(np.unique(qc))),\n'
  "                     present_weight_i2t=float(mi.sum()) / (len(mi) * ctx.present['p_i2t']), present_weight_t2i=float(mt.sum()) / (len(mt) * ctx.present['p_t2i']))"),
 ('    ctxs = dict(ai=ai, at=at, ri=ri, rt=rt, dir_losses=(dl_i, dl_t), dirs=dirs)',
  '    ctxs = dict(ai=ai, at=at, ri=ri, rt=rt, dir_losses=(dl_i, dl_t), dirs=dirs, present_masks=(mi, mt) if present_only else None)'),
 ('        stud = R9.build_students(ctx.joint.ck, dev)\n        stud_ref = {p: {n: q.detach().clone() for n, q in stud[p].named_parameters()} for p in PEERS}',
  "        stud = R9.build_students(ctx.joint.ck, dev, ADAPTATIONS[cfg['adaptation']])   # R40: enabled modalities only (frozen modality: parent, requires_grad False)\n"
  '        stud_ref = {p: {n: q.detach().clone() for n, q in stud[p].named_parameters()} for p in PEERS}'),
 ('                                  trainable_params=int(sum(q.numel() for p in PEERS for q in stud[p].trainable())))',
  '                                  trainable_params=int(sum(q.numel() for p in PEERS for q in stud[p].trainable())),\n'
  "                                  adaptation=cfg['adaptation'], modalities=list(ADAPTATIONS[cfg['adaptation']]),\n"
  "                                  trainable_params_by_modality={m: int(sum(q.numel() for p in PEERS for n, q in stud[p].named_parameters() if q.requires_grad and n.startswith(m + '.'))) for m in "
  "('img', 'txt')},\n"
  "                                  frozen_modality_params=int(sum(q.numel() for p in PEERS for n, q in stud[p].named_parameters() if not q.requires_grad and n.split('.')[0] not in "
  'stud[p].modalities)))'),
 ('        assert r38[\'reference_init_equal\'], f"init differs from reference {ref[\'unit\']}"\n',
  '        assert r38[\'reference_init_equal\'], f"init differs from reference {ref[\'unit\']}"\n'
  "    assert (cfg['adaptation'] != 'frozen') == joint, (cfg['adaptation'], cfg.get('encoder'))\n"
  '    xrefs = R40.extra_references(cfg)   # R40: read-only reused references (R39 N-JOINT / R39 ORACLE-FROZEN): init hashes + batch chain\n'
  "    r40 = dict(adaptation=cfg['adaptation'], references={k: dict(unit=v['unit'], init_equal=init_hash == v['init'], chain_equal_at=[],\n"
  "                                                               student_init_equal=(state_hash(stud) == v['student_init']) if v['student_init'] else None) for k, v in xrefs.items()},\n"
  '               grad_split_windows=[])\n'
  "    for k, v in r40['references'].items():\n"
  '        assert v[\'init_equal\'], f\'head/null init differs from {k} {v["unit"]}\'\n'
  '        assert v[\'student_init_equal\'] in (None, True), f\'student init differs from {k} {v["unit"]}\'\n'
  "    if cfg['supervision'] == 'N_PRESENT_ONLY':   # R40: p_d fixed from the full FIT observed relation (recomputed independently, asserted)\n"
  "        r40['present_check'] = R40.present_recheck(d, ctx.present, D36.null_lists, D36.valid_rows)\n"
  "        write_json(out / 'PRESENT_CHECK.json', r40['present_check'])\n"
  "        assert r40['present_check']['passed'], r40['present_check']\n"
  "        r40['masked_grad'] = dict(updates_checked=0, violations=0, masked_rows=0, valid_rows_nonzero_grad=0)\n"),
 ("    hist.setdefault('r39', dict(joint=joint, supervision=cfg['supervision'], lr_encoder_at_readout={}, param_distance={}, grad_split_windows=[]))",
  "    hist.setdefault('r39', dict(joint=joint, supervision=cfg['supervision'], lr_encoder_at_readout={}, param_distance={}, grad_split_windows=[]))\n"
  "    hist.setdefault('r40', {})\n"
  '    for k_, v_ in r40.items():\n'
  "        if k_ == 'references':\n"
  '            for rk_, rv_ in v_.items():\n'
  "                hist['r40'].setdefault('references', {}).setdefault(rk_, rv_)\n"
  "        elif k_ != 'grad_split_windows' or k_ not in hist['r40']:\n"
  "            hist['r40'].setdefault(k_, v_)\n"
  '    gs40 = []'),
 ("                    hist['r39']['grad_split_windows'].append(dict(end_step=step, **rec['train_window']['r39_grad_split'])); gs_win = []",
  "                    hist['r39']['grad_split_windows'].append(dict(end_step=step, **rec['train_window']['r39_grad_split'])); gs_win = []\n"
  '                if gs40:\n'
  '                    a_ = np.array(gs40)\n'
  "                    rec['train_window']['r40_grad_split'] = {nm_: float(a_[:, j_].mean()) for j_, nm_ in enumerate(('head_norm_mean', 'null_norm_mean', 'img_norm_mean', 'txt_norm_mean', "
  "'total_norm_mean', 'clip_coef_mean'))}\n"
  "                    rec['train_window']['r40_grad_split']['frac_clipped'] = float((a_[:, 5] < 1).mean())\n"
  "                    hist['r40']['grad_split_windows'].append(dict(end_step=step, **rec['train_window']['r40_grad_split'])); gs40 = []"),
 ("                if step not in hist['r38']['reference_chain_equal_at']:\n                    hist['r38']['reference_chain_equal_at'].append(step)",
  "                if step not in hist['r38']['reference_chain_equal_at']:\n"
  "                    hist['r38']['reference_chain_equal_at'].append(step)\n"
  '            for rk_, rv_ in xrefs.items():   # R40: same batch-index hash chain as the reused R39 N-JOINT / ORACLE-FROZEN unit\n'
  "                kk = len(hist['batch_hash']) - 1\n"
  '                assert hist[\'batch_hash\'][-1] == rv_[\'chain\'][kk], f"batch chain differs from {rk_} {rv_[\'unit\']} at step {step}"\n'
  "                if step not in hist['r40']['references'][rk_]['chain_equal_at']:\n"
  "                    hist['r40']['references'][rk_]['chain_equal_at'].append(step)"),
 ("            gc['sum_of_direction_grads_equals_total'] = dict(max_abs_err=err, max_abs_grad=scale, passed=bool(err <= 1e-6 * scale + 1e-12))",
  "            gc['sum_of_direction_grads_equals_total'] = dict(max_abs_err=err, max_abs_grad=scale, passed=bool(err <= 1e-5 * scale + 1e-12), tol_rel=1e-5,\n"
  "                                                             note='R40: R39X amendment A1 tolerance adopted before any R40 unit (fp32 reduction order)')"),
 ('        opt.zero_grad(set_to_none=True)\n        loss.backward()\n',
  '        opt.zero_grad(set_to_none=True)\n'
  '        loss.backward()\n'
  "        if cx['present_masks'] is not None:   # R40: masked (absent) rows must get exactly zero dL/dg on every slot, every update\n"
  "            v_, nm_, nz_ = R40.masked_rows_grad(gl, cx['present_masks'])\n"
  "            mg = hist['r40']['masked_grad']; mg['updates_checked'] += 1; mg['violations'] += v_; mg['masked_rows'] += nm_; mg['valid_rows_nonzero_grad'] += nz_\n"
  "            assert v_ == 0, f'N_PRESENT_ONLY: {v_} masked rows with nonzero gradient at step {step}'\n"
  "        if step == 1 and cfg.get('phase', 1) == 1:   # R40 one-step rule: update-1 gradients (pre-clip) of the declared-active tensors\n"
  '            act40 = R40.active_named(heads, nulls, stud); grads40 = R40.grad_record(act40)\n'),
 ("            gns.append(float(torch.nn.utils.clip_grad_norm_(ps, cfg['clip_per_peer'])))\n"
  '            if stud is not None:\n'
  "                gs_win.append((hn, en, gns[-1], min(1.0, cfg['clip_per_peer'] / (gns[-1] + 1e-6))))",
  '            nrm = lambda qs: float(torch.sqrt(sum(((q.grad.double() ** 2).sum() for q in qs if q.grad is not None), torch.zeros((), dtype=torch.float64, device=dev))))\n'
  '            hn40 = nrm(list(heads[p].parameters())); nn40 = nrm([nulls.v[p]]) if nulls is not None else 0.0   # R40 split (pre-clip)\n'
  "            in40 = nrm([q for n_, q in stud[p].named_parameters() if q.requires_grad and n_.startswith('img.')]) if stud is not None else 0.0\n"
  "            tn40 = nrm([q for n_, q in stud[p].named_parameters() if q.requires_grad and n_.startswith('txt.')]) if stud is not None else 0.0\n"
  "            gns.append(float(torch.nn.utils.clip_grad_norm_(ps, cfg['clip_per_peer'])))\n"
  "            gs40.append((hn40, nn40, in40, tn40, gns[-1], min(1.0, cfg['clip_per_peer'] / (gns[-1] + 1e-6))))\n"
  '            if stud is not None:\n'
  "                gs_win.append((hn, en, gns[-1], min(1.0, cfg['clip_per_peer'] / (gns[-1] + 1e-6))))"),
 ("        last_lr_enc = opt.param_groups[1]['lr'] if len(opt.param_groups) > 1 else None\n        opt.step()\n",
  "        last_lr_enc = opt.param_groups[1]['lr'] if len(opt.param_groups) > 1 else None\n"
  "        if step == 1 and cfg.get('phase', 1) == 1:\n"
  '            before40 = R40.snapshot(act40)\n'
  '        opt.step()\n'
  "        if step == 1 and cfg.get('phase', 1) == 1:   # R40 one-step active-path rule (every unit; frozen modality bit-unchanged)\n"
  '            fz40 = R40.frozen_modality_named(stud)\n'
  "            o40 = R40.one_step_rule(opt, act40, before40, grads40, fz40, {n_: stud_ref[n_.split('.')[1]][n_.split('.', 2)[2]] for n_, _ in fz40})\n"
  '            if stud is not None:\n'
  "                o40['frozen_state_unchanged'] = {p: R9.state_digest(stud[p].frozen_state()) == frozen0[p] for p in PEERS}\n"
  "                o40['passed'] = bool(o40['passed'] and all(o40['frozen_state_unchanged'].values()))\n"
  "            o40['frozen_H_cache_unchanged'] = R9.state_digest({f'{s}_{k}_{p}': d[(s, k, p)] for s in ('fit', 'dev') for k in ('img', 'cap') for p in PEERS}) == cache40\n"
  "            o40['passed'] = bool(o40['passed'] and o40['frozen_H_cache_unchanged'])\n"
  "            hist['r40']['one_step_check'] = {k_: v_ for k_, v_ in o40.items() if k_ not in ('per_tensor', 'frozen_modality')}\n"
  "            write_json(out / 'ONE_STEP_CHECK_R40.json', o40)\n"
  '            assert o40[\'passed\'], f"R40 one-step active-path check failed: { {k_: v_ for k_, v_ in o40.items() if k_ not in (\'per_tensor\', \'frozen_modality\')} }"\n'),
 ("            oc['passed'] = bool(all(ch[p] == ntr[p] for p in PEERS) and all(fr.values()) and ca and oc['parent_ckpt_sha256_unchanged'])",
  "            oc['all_trainable_changed_vs_parent_R39_rule'] = bool(all(ch[p] == ntr[p] for p in PEERS))   # R40: recorded only; the R40 rule (ONE_STEP_CHECK_R40.json) decides\n"
  "            oc['passed'] = bool(o40['passed'] and all(fr.values()) and ca and oc['parent_ckpt_sha256_unchanged'])"),
 ("                  r39=phases[-1]['train_history'].get('r39'),",
  "                  r39=phases[-1]['train_history'].get('r39'), r40=phases[-1]['train_history'].get('r40'), adaptation=cfg['adaptation'],"),
 ("    frozen = read_json(ROOT / 'PROTOCOL_FROZEN.json')", '    frozen = read_json(ROOT / FROZEN_FILE)   # R40 freeze file'),
 ("            print('R39_UNIT_DONE', cfg['unit_id']", "            print('R40_UNIT_DONE', cfg['unit_id']"),
 ("    print('R39_JOB_END failed', failed)", "    print('R40_JOB_END failed', failed)")]
TRAIN_UNIT_CACHE_ANCHOR = "    init_hash = state_hash(heads) + (state_hash(nulls) if nulls is not None else '')\n"
TRAIN_UNIT_CACHE_LINE = "    cache40 = R9.state_digest({f'{s_}_{k_}_{p_}': d[(s_, k_, p_)] for s_ in ('fit', 'dev') for k_ in ('img', 'cap') for p_ in PEERS})   # R40: frozen-H cache digest (all units)\n"
R39EXT_REP = [('"""R39 additions,',
  '"""R40: R39 r39ext.py + update scope (Student(modalities): the frozen modality keeps the parent weights with requires_grad=False, so it is\n'
  'not in trainable() / the optimizer / the clip, and its whole state is part of frozen_state(); param_distance split per modality).\n'
  'R39 additions,'),
 ('    def __init__(self, img_state, txt_state, opt):\n        super().__init__()',
  "    def __init__(self, img_state, txt_state, opt, modalities=('img', 'txt')):\n"
  '        super().__init__()\n'
  "        assert set(modalities) <= {'img', 'txt'} and len(modalities) > 0, modalities\n"
  "        self.modalities = tuple(m for m in ('img', 'txt') if m in modalities)   # R40: updated modalities (the rest stays the parent)"),
 ('        self.l2norm = V.l2norm\n        self.train()',
  "        for m in ('img', 'txt'):   # R40: frozen modality = parent parameters, requires_grad False (never in the optimizer)\n"
  '            if m not in self.modalities:\n'
  '                for q in getattr(self, m).parameters():\n'
  '                    q.requires_grad_(False)\n'
  '        self.l2norm = V.l2norm\n'
  '        self.train()'),
 ("        return {n: t for n, t in self.state_dict().items() if n == 'txt.embed.weight' or 'running_' in n or 'num_batches_tracked' in n or '.mlp.bns.1.' in n}",
  "        fm = tuple(m + '.' for m in ('img', 'txt') if m not in self.modalities)   # R40: + every tensor of the frozen modality\n"
  "        return {n: t for n, t in self.state_dict().items() if n == 'txt.embed.weight' or 'running_' in n or 'num_batches_tracked' in n or '.mlp.bns.1.' in n or n.startswith(fm)}"),
 ("def build_students(ck, device):\n    s = nn.ModuleDict({p: Student(ck[f'model_{p}'][0], ck[f'model_{p}'][1], ck['opt']) for p in PEERS}).to(device)",
  "def build_students(ck, device, modalities=('img', 'txt')):\n    s = nn.ModuleDict({p: Student(ck[f'model_{p}'][0], ck[f'model_{p}'][1], ck['opt'], modalities) for p in PEERS}).to(device)"),
 ('            sq = 0.0; sq0 = 0.0\n'
  '            for n, q in stud[p].named_parameters():\n'
  '                if q.requires_grad:\n'
  '                    r = ref_states[p][n]\n'
  '                    sq += float(((q.detach() - r) ** 2).sum()); sq0 += float((r ** 2).sum())\n'
  '            out[p] = dict(l2_from_parent=sq ** .5, rel_l2_from_parent=(sq / sq0) ** .5)',
  "            sq = 0.0; sq0 = 0.0; mod = {m: [0.0, 0.0] for m in ('img', 'txt')}\n"
  '            for n, q in stud[p].named_parameters():\n'
  '                if q.requires_grad:\n'
  '                    r = ref_states[p][n]\n'
  '                    a_, b_ = float(((q.detach() - r) ** 2).sum()), float((r ** 2).sum())\n'
  "                    sq += a_; sq0 += b_; mod[n.split('.')[0]][0] += a_; mod[n.split('.')[0]][1] += b_\n"
  '            out[p] = dict(l2_from_parent=sq ** .5, rel_l2_from_parent=(sq / sq0) ** .5,\n'
  '                          per_modality={m: dict(l2_from_parent=v[0] ** .5, rel_l2_from_parent=(v[0] / v[1]) ** .5 if v[1] > 0 else None) for m, v in mod.items()})')]


def apply(src, rep):
    for a, b in rep:
        assert src.count(a) == 1, a[:80]
        src = src.replace(a, b)
    return src


def derive_train_unit(src):
    s = apply(src, TRAIN_UNIT_REP)
    assert s.count(TRAIN_UNIT_CACHE_ANCHOR) == 1
    return s.replace(TRAIN_UNIT_CACHE_ANCHOR, TRAIN_UNIT_CACHE_ANCHOR + TRAIN_UNIT_CACHE_LINE)
