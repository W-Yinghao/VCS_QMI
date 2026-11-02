"""R39: derive r39f/train_unit.py and r39c/train_unit.py from the byte copies of R38 r38f/r38c train_unit.py by fixed textual edits
(each target string must occur exactly once; the script refuses to run twice). Edits: COMPLETE label hook (r39ext.Completion),
JOINT student global encoders (r39ext.JointInputs: phi takes student vectors, s_H stays frozen-H), two AdamW groups with the common
cosine multiplier, per-peer joint clip, student state in checkpoints, in-unit checks (step-0 cache equivalence of the student
forward-with-grad; update-1 student change / frozen-tensor invariance; COMPLETE row invariants per batch), readouts (completed-label
FIT list J, student feature change, parameter distance, clip/gradient split). Frozen N/COMPLETE units keep the R38 numerics.
Usage: python3 tools/patch_train_unit.py"""
from pathlib import Path
W = Path(__file__).resolve().parents[1]

E = []
def edit(old, new):
    E.append((old, new))

edit('from __future__ import annotations\n',
     'from __future__ import annotations\n')   # placeholder (keeps the edit list non-empty if the docstring changes)

edit('import r38ext as X\n', 'import r38ext as X\nimport r39ext as R9\n')

edit("""        self.present = D36.presence_probs(self.d)   # R37: fixed p_d per direction (N_PRESENT_ONLY normalisation)
""", """        self.present = D36.presence_probs(self.d)   # R37: fixed p_d per direction (N_PRESENT_ONLY normalisation)
        self.completion = None; self.joint = None   # R39: loaded on demand (COMPLETE units / JOINT units)

    def ensure_completion(self):
        if self.completion is None:
            self.completion = R9.Completion(self.cond, self.d)
        return self.completion

    def ensure_joint(self):
        if self.joint is None:
            self.joint = R9.JointInputs(self.cond, self.d, self.dev)
        return self.joint
""")

edit("def g_pairs_lists(h, enc, ctx, p, split, img_idx, cap_idx, grad=False, enc_grad=False, s_exact=None, chunk=8192, proj=None, direction=None):",
     "def g_pairs_lists(h, enc, ctx, p, split, img_idx, cap_idx, grad=False, enc_grad=False, s_exact=None, chunk=8192, proj=None, direction=None, stu=None):")
edit("""        x, y = X[ii], Y[cc]; s = (x * y).sum(-1)
        lf = None
""", """        x, y = X[ii], Y[cc]; s = (x * y).sum(-1)
        if stu is not None:   # R39 JOINT: phi takes the current student's global vectors; the skip term s stays the frozen-H native score
            x, y = stu[0][stu[1][i:i + chunk]], stu[2][stu[3][i:i + chunk]]
        lf = None
""")

edit("""def dev_pairs(tab):""", """def stu_pairs(stud, ctx, p, split, img, cap, grad=False):
    \"\"\"R39 JOINT: student vectors of explicit pairs (unique IDs encoded once, re-expanded by index); None for frozen units.\"\"\"
    if stud is None:
        return None
    return ctx.joint.pairs(stud[p], p, split, img, cap, grad)


def dev_pairs(tab):""")

edit("def readout_lists(heads, encs, ctx, final=False):", "def readout_lists(heads, encs, ctx, final=False, stud=None):")
edit("            gsum = gsum + g_pairs_lists(h, encs[p] if encs else None, ctx, p, 'dev', img, cap, s_exact=s_ex, proj=proj, direction=dirs)",
     "            gsum = gsum + g_pairs_lists(h, encs[p] if encs else None, ctx, p, 'dev', img, cap, s_exact=s_ex, proj=proj, direction=dirs, stu=stu_pairs(stud, ctx, p, 'dev', img, cap))")
edit("                gs = gs + g_pairs_lists(heads[p], encs[p] if encs else None, ctx, p, 'fit', im, cp, direction=fd).double()",
     "                gs = gs + g_pairs_lists(heads[p], encs[p] if encs else None, ctx, p, 'fit', im, cp, direction=fd, stu=stu_pairs(stud, ctx, p, 'fit', im, cp)).double()")

edit("def fit_pair_scores(heads, encs, ctx, img, cap, direction=None):", "def fit_pair_scores(heads, encs, ctx, img, cap, direction=None, stud=None):")
edit("        gs = gs + g_pairs_lists(heads[p], encs[p] if encs else None, ctx, p, 'fit', img, cap, direction=direction).double()",
     "        gs = gs + g_pairs_lists(heads[p], encs[p] if encs else None, ctx, p, 'fit', img, cap, direction=direction, stu=stu_pairs(stud, ctx, p, 'fit', img, cap)).double()")

edit("def readout_r36(heads, nulls, encs, ctx, final=False):", "def readout_r36(heads, nulls, encs, ctx, final=False, stud=None):")
edit("    rec, arrs, rk = readout_lists(heads, encs, ctx, final=final)\n    t0 = time.time()\n",
     "    if stud is not None:\n        ctx.joint.clear_eval()\n    rec, arrs, rk = readout_lists(heads, encs, ctx, final=final, stud=stud)\n    t0 = time.time()\n")
edit("    g, s = fit_pair_scores(heads, encs, ctx, im, cp, direction=np.concatenate([np.zeros(len(ai) * N_SLOTS, np.int64), np.ones(len(at) * N_SLOTS, np.int64)]))",
     "    g, s = fit_pair_scores(heads, encs, ctx, im, cp, direction=np.concatenate([np.zeros(len(ai) * N_SLOTS, np.int64), np.ones(len(at) * N_SLOTS, np.int64)]), stud=stud)")
edit("""    if nulls is not None:
        (qi, caps, ai), (qc, imgs, at) = D36.locked_null_lists(ctx.d, 'lev_i', 'lev_t')
        out = {}
        for dn, di, ids, q, a in (('i2t', 0, caps, qi, ai), ('t2i', 1, imgs, qc, at)):""",
     """    if nulls is not None:
        rec['fit_null_lists_observed_J_raw'] = null_list_j(heads, nulls, encs, ctx, D36.locked_null_lists(ctx.d, 'lev_i', 'lev_t'), stud)
        if ctx.completion is not None:   # R39 COMPLETE: same locked lists, completed labels (observed-label J above stays the N label object)
            cl, cst = ctx.completion.apply(D36.locked_null_lists(ctx.d, 'lev_i', 'lev_t'))
            rec['fit_null_lists_completed_J_raw'] = dict(null_list_j(heads, nulls, encs, ctx, cl, stud), completed_rows=cst)
    if stud is not None:   # R39 JOINT: student vs frozen-H global vectors over all FIT/DEV items
        rec['r39_student_feature_change'] = ctx.joint.feature_change(stud)
        ctx.joint.clear_eval()
    rec['readout_seconds'] += time.time() - t0
    return rec, arrs, rk


def null_list_j(heads, nulls, encs, ctx, lists, stud):
    \"\"\"R36 null-list J (33 slots, Q = Uniform(33)) split into real-slot and null-slot parts for given (locked) lists/labels.\"\"\"
    (qi, caps, ai), (qc, imgs, at) = lists
    if True:
        out = {}
        for dn, di, ids, q, a in (('i2t', 0, caps, qi, ai), ('t2i', 1, imgs, qc, at)):""")
edit("            gr, _ = fit_pair_scores(heads, encs, ctx, img_, cap_, direction=di)",
     "            gr, _ = fit_pair_scores(heads, encs, ctx, img_, cap_, direction=di, stud=stud)")
edit("""        out['J_total_bidirectional'] = .5 * (out['i2t']['J_total'] + out['t2i']['J_total'])
        out['null_params'] = {p: [float(x) for x in nulls.v[p].detach().cpu()] for p in PEERS}
        rec['fit_null_lists_observed_J_raw'] = out
    rec['readout_seconds'] += time.time() - t0
    return rec, arrs, rk
""", """        out['J_total_bidirectional'] = .5 * (out['i2t']['J_total'] + out['t2i']['J_total'])
        out['null_params'] = {p: [float(x) for x in nulls.v[p].detach().cpu()] for p in PEERS}
        return out
""")

edit("""def make_opt(cfg, heads, nulls):
    o = cfg['optimizer']
    ps = list(heads.parameters()) + (list(nulls.parameters()) if nulls is not None else [])
    return torch.optim.AdamW([dict(params=ps, lr=o['lr_head'])], betas=tuple(o['betas']), eps=o['eps'], weight_decay=o['weight_decay'])


def peer_params(heads, nulls, p):
    return list(heads[p].parameters()) + ([nulls.v[p]] if nulls is not None else [])
""", """def make_opt(cfg, heads, nulls, stud=None):
    \"\"\"R39: group 0 = heads (+ nulls), lr_head (unchanged R38 group); JOINT adds group 1 = student global encoders, lr_encoder.\"\"\"
    o = cfg['optimizer']
    ps = list(heads.parameters()) + (list(nulls.parameters()) if nulls is not None else [])
    groups = [dict(params=ps, lr=o['lr_head'], base_lr=o['lr_head'], name='head')]
    if stud is not None:
        groups.append(dict(params=[q for p in PEERS for q in stud[p].trainable()], lr=o['lr_encoder'], base_lr=o['lr_encoder'], name='student_encoder'))
    return torch.optim.AdamW(groups, betas=tuple(o['betas']), eps=o['eps'], weight_decay=o['weight_decay'])


def peer_params(heads, nulls, p, stud=None):
    return list(heads[p].parameters()) + ([nulls.v[p]] if nulls is not None else []) + (stud[p].trainable() if stud is not None else [])
""")

edit("def step_lists(cfg, heads, nulls, encs, ctx, rng, rng2, pool, oracle):", "def step_lists(cfg, heads, nulls, encs, ctx, rng, rng2, pool, oracle, stud=None):")
edit("""    lists, idx = D36.sample(sup, rng, rng2, d, cfg['n_queries'], pool)
    (qi, caps, ai), (qc, imgs, at) = lists
    stats = {}
""", """    lists, idx = D36.sample(sup, rng, rng2, d, cfg['n_queries'], pool)
    stats = {}
    if sup == 'COMPLETE_XCHECK':   # R39: fixed cross-checked completion of observed-missing rows (only all-null rows change; asserted)
        lists, cst = ctx.completion.apply(lists)
        stats.update(cst)
    (qi, caps, ai), (qc, imgs, at) = lists
""")
edit("""        g = g_pairs_lists(h, encs[p] if encs else None, ctx, p, 'fit', img, cap, grad=True, enc_grad=False, proj=proj, chunk=1 << 30, direction=dirs)""",
     """        g = g_pairs_lists(h, encs[p] if encs else None, ctx, p, 'fit', img, cap, grad=True, enc_grad=False, proj=proj, chunk=1 << 30, direction=dirs,
                          stu=stu_pairs(stud, ctx, p, 'fit', img, cap, grad=True))""")

edit("""def save_ckpt(path, cfg, step, heads, nulls, opt, rng, rng2, lineage):
    obj = dict(step=step, unit_id=cfg['unit_id'], phase=cfg.get('phase', 1), heads=heads.state_dict(), nulls=nulls.state_dict() if nulls is not None else None,""",
     """def save_ckpt(path, cfg, step, heads, nulls, opt, rng, rng2, lineage, stud=None):
    obj = dict(step=step, unit_id=cfg['unit_id'], phase=cfg.get('phase', 1), heads=heads.state_dict(), nulls=nulls.state_dict() if nulls is not None else None,
               student=stud.state_dict() if stud is not None else None,""")

# ---------------------------------------------------------------- run_unit
edit("""    if ctx.need_local:
        encs = torch.nn.ModuleDict({p: _copy_enc(ctx.enc0[p]) for p in PEERS})""",
     """    joint = cfg.get('encoder') == 'joint_global'
    stud = None
    if cfg['supervision'] == 'COMPLETE_XCHECK':
        lineage['completion_table_sha256'] = ctx.completion.sha256
        lineage['completion_prep_manifest'] = dict(teachers=ctx.completion.manifest['teachers'], rule=ctx.completion.manifest['rule'])
    if joint:   # R39 JOINT: fresh student copies of the parent's global encoders (both peers), built from the parent checkpoint
        stud = R9.build_students(ctx.joint.ck, dev)
        stud_ref = {p: {n: q.detach().clone() for n, q in stud[p].named_parameters()} for p in PEERS}
        frozen0 = {p: R9.state_digest(stud[p].frozen_state()) for p in PEERS}
        cache0 = R9.state_digest({f'{s}_{k}_{p}': d[(s, k, p)] for s in ('fit', 'dev') for k in ('img', 'cap') for p in PEERS})
        lineage['student'] = dict(parent_ckpt=str(R9.C.PARENT_CKPT), parent_sha256=ctx.joint.manifest['parent']['model_best_sha256'],
                                  init_state_sha256=state_hash(stud), joint_prep_manifest=dict(tokens=ctx.joint.manifest['tokens'], equivalence=ctx.joint.manifest.get('equivalence_summary')),
                                  trainable_params=int(sum(q.numel() for p in PEERS for q in stud[p].trainable())))
    if ctx.need_local:
        encs = torch.nn.ModuleDict({p: _copy_enc(ctx.enc0[p]) for p in PEERS})""")
edit("    opt = make_opt(cfg, heads, nulls)\n", "    opt = make_opt(cfg, heads, nulls, stud)\n")
edit("""        heads.load_state_dict(ck['heads']); opt.load_state_dict(ck['optimizer'])""",
     """        heads.load_state_dict(ck['heads']); opt.load_state_dict(ck['optimizer'])
        if stud is not None:
            stud.load_state_dict(ck['student'])""")
edit("""    hist['r38'].setdefault('dir_grad_diag', {})
""", """    hist['r38'].setdefault('dir_grad_diag', {})
    hist.setdefault('r39', dict(joint=joint, supervision=cfg['supervision'], lr_encoder_at_readout={}, param_distance={}, grad_split_windows=[]))
    gs_win = []
""")
edit("                rec, arrs, rk = readout_r36(heads, nulls, encs, ctx, final=(step == steps))",
     "                rec, arrs, rk = readout_r36(heads, nulls, encs, ctx, final=(step == steps), stud=stud)\n"
     "                if stud is not None:\n"
     "                    rec['r39_param_distance'] = ctx.joint.param_distance(stud, stud_ref); hist['r39']['param_distance'][str(step)] = rec['r39_param_distance']")
edit("""                hist['windows'].append(dict(end_step=step, **rec['train_window']))
                win = []; gwin = []""", """                if gs_win:
                    a_ = np.array(gs_win)
                    rec['train_window']['r39_grad_split'] = dict(head_norm_mean=float(a_[:, 0].mean()), encoder_norm_mean=float(a_[:, 1].mean()),
                                                                 total_norm_mean=float(a_[:, 2].mean()), clip_coef_mean=float(a_[:, 3].mean()), frac_clipped=float((a_[:, 3] < 1).mean()))
                    hist['r39']['grad_split_windows'].append(dict(end_step=step, **rec['train_window']['r39_grad_split'])); gs_win = []
                hist['windows'].append(dict(end_step=step, **rec['train_window']))
                win = []; gwin = []""")
edit("                save_ckpt(out / f'ckpt_step{step}.pt', cfg, step, heads, nulls, opt, rng, rng2, lineage)",
     "                save_ckpt(out / f'ckpt_step{step}.pt', cfg, step, heads, nulls, opt, rng, rng2, lineage, stud)")
edit("            hist['r38']['lr_at_readout'][str(step)] = last_lr\n",
     "            hist['r38']['lr_at_readout'][str(step)] = last_lr\n            hist['r39']['lr_encoder_at_readout'][str(step)] = last_lr_enc\n")
edit("    last_lr = None; gc = None\n", "    last_lr = None; gc = None; last_lr_enc = None\n")
edit("""        heads.train()
        loss, jb, idx, st, gl, cx = step_lists(cfg, heads, nulls, encs, ctx, rng, rng2, pool, oracle)""",
     """        heads.train()
        if stud is not None:
            stud.train()
        loss, jb, idx, st, gl, cx = step_lists(cfg, heads, nulls, encs, ctx, rng, rng2, pool, oracle, stud)
        if step == 0 and cfg.get('phase', 1) == 1 and stud is not None:   # R39 fail-fast: student forward (grad on, training batch) == native cache
            eq = dict(train_batch_grad=ctx.joint.cache_equivalence(stud, idx[0], idx[1], 'fit', grad=True))
            ctx.joint.clear_eval()
            eq['dev_all_nograd'] = {k: v for k, v in ctx.joint.cache_equivalence(stud, np.arange(N_DEV_IMG), np.arange(N_DEV_CAP), 'dev', grad=False).items()}
            tol = cfg['joint_rule']['equivalence_tol_max_abs']
            eq['tol_max_abs'] = tol; eq['passed'] = bool(all(v['max_abs'] <= tol for part in ('train_batch_grad', 'dev_all_nograd') for v in eq[part].values()))
            hist['r39']['step0_cache_equivalence'] = eq; write_json(out / 'STEP0_CACHE_EQUIVALENCE.json', eq)
            assert eq['passed'], f'R39 student forward differs from the native cache: {eq}'""")
edit("""        gns = []
        for p in PEERS:
            ps = [q for q in peer_params(heads, nulls, p) if q.grad is not None]
            gns.append(float(torch.nn.utils.clip_grad_norm_(ps, cfg['clip_per_peer'])))""",
     """        if step == 1 and cfg.get('phase', 1) == 1 and stud is not None:   # R39 update-1 check: every student peer gets a gradient (head out is nonzero now)
            hist['r39']['update1_student_grad_norm'] = {p: float(torch.sqrt(sum((q.grad.double() ** 2).sum() for q in stud[p].trainable() if q.grad is not None))) for p in PEERS}
            assert all(v > 0 for v in hist['r39']['update1_student_grad_norm'].values()), hist['r39']['update1_student_grad_norm']
        gns = []
        for p in PEERS:
            ps = [q for q in peer_params(heads, nulls, p, stud) if q.grad is not None]
            if stud is not None:   # R39: one joint clip per peer (head + null + student); record the split before clipping
                hn = float(torch.sqrt(sum((q.grad.double() ** 2).sum() for q in peer_params(heads, nulls, p) if q.grad is not None)))
                en = float(torch.sqrt(sum((q.grad.double() ** 2).sum() for q in stud[p].trainable() if q.grad is not None)))
            gns.append(float(torch.nn.utils.clip_grad_norm_(ps, cfg['clip_per_peer'])))
            if stud is not None:
                gs_win.append((hn, en, gns[-1], min(1.0, cfg['clip_per_peer'] / (gns[-1] + 1e-6))))""")
edit("""            for pg in opt.param_groups:
                pg['lr'] = cfg['optimizer']['lr_head'] * X.lr_multiplier(step, cfg['steps'], 'cosine', cfg['optimizer']['cosine_floor'])
        last_lr = opt.param_groups[0]['lr']
        opt.step()""",
     """            for pg in opt.param_groups:   # R39: each group's own base lr x the common multiplier (group 0 = R38 head group, same value)
                pg['lr'] = pg['base_lr'] * X.lr_multiplier(step, cfg['steps'], 'cosine', cfg['optimizer']['cosine_floor'])
        last_lr = opt.param_groups[0]['lr']
        last_lr_enc = opt.param_groups[1]['lr'] if len(opt.param_groups) > 1 else None
        opt.step()
        if step == 1 and cfg.get('phase', 1) == 1 and stud is not None:   # R39 update-1 check: student changed, frozen tensors / parent cache unchanged
            ch = {p: sum(int(not torch.equal(q.detach(), stud_ref[p][n])) for n, q in stud[p].named_parameters() if q.requires_grad) for p in PEERS}
            ntr = {p: sum(1 for q in stud[p].trainable()) for p in PEERS}
            fr = {p: R9.state_digest(stud[p].frozen_state()) == frozen0[p] for p in PEERS}
            ca = R9.state_digest({f'{s}_{k}_{p}': d[(s, k, p)] for s in ('fit', 'dev') for k in ('img', 'cap') for p in PEERS}) == cache0
            oc = dict(student_tensors_changed=ch, student_trainable_tensors=ntr, frozen_embedding_and_bn_stats_unchanged=fr, frozen_H_cache_unchanged=bool(ca),
                      parent_ckpt_sha256_unchanged=sha256(R9.C.PARENT_CKPT) == lineage['student']['parent_sha256'])
            oc['passed'] = bool(all(ch[p] == ntr[p] for p in PEERS) and all(fr.values()) and ca and oc['parent_ckpt_sha256_unchanged'])
            hist['r39']['one_step_check'] = oc; write_json(out / 'ONE_STEP_CHECK.json', oc)
            assert oc['passed'], f'R39 one-step check failed: {oc}'
        if step == 0 and cfg.get('phase', 1) == 1 and cfg['supervision'] == 'COMPLETE_XCHECK':   # R39 COMPLETE one-step record (invariants asserted per batch)
            write_json(out / 'ONE_STEP_CHECK.json', dict(supervision='COMPLETE_XCHECK', first_batch=st, rule='only all-null rows change; changed rows = 0.5 agreed + 0.5 null (asserted every batch)', passed=True))""")
edit("""    return dict(heads=heads, nulls=nulls, encs=encs, init_hash=init_hash, final_hash=state_hash(heads) + (state_hash(nulls) if nulls is not None else ''),""",
     """    if stud is not None:   # R39: end-of-unit frozen-tensor invariance (parent cache, student embedding/BN stats)
        hist['r39']['end_frozen_check'] = dict(frozen_H_cache_unchanged=R9.state_digest({f'{s}_{k}_{p}': d[(s, k, p)] for s in ('fit', 'dev') for k in ('img', 'cap') for p in PEERS}) == cache0,
                                               frozen_embedding_and_bn_stats_unchanged={p: R9.state_digest(stud[p].frozen_state()) == frozen0[p] for p in PEERS})
    return dict(heads=heads, nulls=nulls, encs=encs, stud=stud, init_hash=init_hash, final_hash=state_hash(heads) + (state_hash(nulls) if nulls is not None else '') + (state_hash(stud) if stud is not None else ''),""")

# ---------------------------------------------------------------- finalize / frozen / main
edit("""                  cost=dict(params_trainable=n_params(r0['heads']) + (n_params(r0['nulls']) if r0['nulls'] is not None else 0),""",
     """                  r39=phases[-1]['train_history'].get('r39'),
                  cost=dict(params_trainable=n_params(r0['heads']) + (n_params(r0['nulls']) if r0['nulls'] is not None else 0) + (sum(q.numel() for p in PEERS for q in r0['stud'][p].trainable()) if r0.get('stud') is not None else 0),""")
edit("""            ctx = ctxs[key]
""", """            ctx = ctxs[key]
            if cfg['supervision'] == 'COMPLETE_XCHECK':
                ctx.ensure_completion()
            if cfg.get('encoder') == 'joint_global':
                ctx.ensure_joint()
""")


def main():
    for tree in ('r39f', 'r39c'):
        p = W / tree / 'train_unit.py'
        s = p.read_text()
        assert 'import r39ext as R9' not in s, f'{p} already patched'
        for old, new in E:
            n = s.count(old)
            assert n == 1, (tree, n, old[:120])
            s = s.replace(old, new)
        s = s.replace('"""R38: train_unit.py of R37', '"""R39: R38 train_unit.py + R39 edits (tools/patch_train_unit.py: COMPLETE label hook, JOINT student global encoders, two AdamW groups,\n'
                      'per-peer joint clip, in-unit cache-equivalence / one-step / frozen-tensor checks, R39 readouts). R38: train_unit.py of R37', 1)
        s = s.replace("print('R37_UNIT_DONE'", "print('R39_UNIT_DONE'").replace("print('R37C_UNIT_DONE'", "print('R39_UNIT_DONE'")
        s = s.replace("print('R37_JOB_END failed'", "print('R39_JOB_END failed'").replace("print('R37C_JOB_END failed'", "print('R39_JOB_END failed'")
        p.write_text(s)
        print('patched', p)


if __name__ == '__main__':
    main()
