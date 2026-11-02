"""One-time R38 edit of the two train_unit.py copies (r38f <- R37 r37, r38c <- R37C r37c). Every replacement must hit exactly once.
Shared/constant units execute the original statements unchanged; additions are guarded by the directional head flag, the
cosine schedule flag, or are read-only checks/diagnostics."""
import sys
from pathlib import Path
W = Path(__file__).resolve().parents[1]

R = []
def r(a, b): R.append((a, b))

r('''from data35 import load_cond, shard_mm, build_lists, load_local_encoders, load_raw, N_SLOTS
import data36 as D36''', '''from data35 import load_cond, shard_mm, build_lists, load_local_encoders, load_raw, N_SLOTS
import data36 as D36
import r38ext as X''')
# code hashing / frozen check on this tree's own directory
r('''    return {Path(f).name: sha256(f) for f in sorted(glob.glob(str(ROOT / '%TREE%' / '*.py')))}''',
  '''    return {Path(f).name: sha256(f) for f in sorted(glob.glob(str(Path(__file__).resolve().parent / '*.py')))}''')
# g_pairs_lists: per-row direction for directional heads
r('''def g_pairs_lists(h, enc, ctx, p, split, img_idx, cap_idx, grad=False, enc_grad=False, s_exact=None, chunk=8192, proj=None):''',
  '''def g_pairs_lists(h, enc, ctx, p, split, img_idx, cap_idx, grad=False, enc_grad=False, s_exact=None, chunk=8192, proj=None, direction=None):''')
r('''        hid = h.glob.hidden(x, y, s)
        if lf is not None:
            hid = torch.cat([hid, lf], -1)
        r = h.out(hid).squeeze(-1)''', '''        hid = h.glob.hidden(x, y, s)
        if lf is not None:
            hid = torch.cat([hid, lf], -1)
        if getattr(h, 'directional', False):   # R38: each row scored with its own direction's output + affine (shared trunk)
            assert direction is not None, 'directional head needs the per-row direction (0 = i2t, 1 = t2i)'
            dd = torch.as_tensor(np.broadcast_to(np.asarray(direction), img_idx.shape)[i:i + chunk].copy(), device=dev)
            outs.append(h.rows(s, hid, dd, None if s_exact is None else torch.as_tensor(s_exact[i:i + chunk], device=dev)))
            continue
        r = h.out(hid).squeeze(-1)''')
# readout_lists: DEV pairs in their own direction; GT pairs scored once per direction
r('''    img, cap, names, sizes = dev_pairs(tab)
    with torch.no_grad():''', '''    img, cap, names, sizes = dev_pairs(tab)
    directional = getattr(heads[PEERS[0]], 'directional', False)
    if directional:   # R38: GT pairs scored once per direction (g_gt: i2t, g_gt_t2i: t2i); every other DEV part in its own direction
        a_, b_ = sum(sizes[:2]), sum(sizes[:3]); assert names[2] == 'gt'
        img, cap, names, sizes = np.concatenate([img, img[a_:b_]]), np.concatenate([cap, cap[a_:b_]]), names + ['gt_t2i'], sizes + [b_ - a_]
    DIRN = dict(i_top=0, t_top=1, gt=0, rand_i=0, rand_t=1, gt_t2i=1)
    dirs = np.concatenate([np.full(n, DIRN[k], np.int64) for k, n in zip(names, sizes)])
    with torch.no_grad():''')
r('''            gsum = gsum + g_pairs_lists(h, encs[p] if encs else None, ctx, p, 'dev', img, cap, s_exact=s_ex, proj=proj)''',
  '''            gsum = gsum + g_pairs_lists(h, encs[p] if encs else None, ctx, p, 'dev', img, cap, s_exact=s_ex, proj=proj, direction=dirs)''')
r('''    g_gt = parts['gt']
''', '''    g_gt = parts['gt']
    g_gt_t2i = parts['gt_t2i'] if directional else g_gt
''')
r('''    rec['pair_win_rates'] = EV.pair_win_rates(g_gt, hard_t,''', '''    rec['pair_win_rates'] = EV.pair_win_rates(g_gt_t2i, hard_t,''')
r('''    li, lt = EV.dev_list_scores(gi_top, gt_top, g_gt, ctx.pos_i, ctx.pos_t)
''', '''    li, lt = EV.dev_list_scores(gi_top, gt_top, g_gt, ctx.pos_i, ctx.pos_t)
    if directional:
        lt = EV.dev_list_scores(gi_top, gt_top, g_gt_t2i, ctx.pos_i, ctx.pos_t)[1]
''')
r('''        im, cp, ai, at = list_pairs(lists)
        with torch.no_grad():
            gs = 0
            for p in PEERS:
                gs = gs + g_pairs_lists(heads[p], encs[p] if encs else None, ctx, p, 'fit', im, cp).double()''',
  '''        im, cp, ai, at = list_pairs(lists)
        fd = np.concatenate([np.zeros(len(ai) * N_SLOTS, np.int64), np.ones(len(at) * N_SLOTS, np.int64)])
        with torch.no_grad():
            gs = 0
            for p in PEERS:
                gs = gs + g_pairs_lists(heads[p], encs[p] if encs else None, ctx, p, 'fit', im, cp, direction=fd).double()''')
r('''    rec['heads_ab'] = {p: [float(t) for t in heads[p].affine.ab()] for p in PEERS}''',
  '''    rec['heads_ab'] = {p: [float(t) for t in heads[p].affine.ab()] for p in PEERS}
    if directional:
        rec['heads_ab_t2i'] = {p: [float(t) for t in heads[p].affine_t2i.ab()] for p in PEERS}''')
r('''    return rec, dict(gi_top=gi_top, gt_top=gt_top, g_gt=g_gt), rk''',
  '''    arrs = dict(gi_top=gi_top, gt_top=gt_top, g_gt=g_gt)
    if directional:
        arrs['g_gt_t2i'] = g_gt_t2i
    return rec, arrs, rk''')
# FIT readouts with direction
r('''def fit_pair_scores(heads, encs, ctx, img, cap):
    gs, ss = 0, 0
    for p in PEERS:
        gs = gs + g_pairs_lists(heads[p], encs[p] if encs else None, ctx, p, 'fit', img, cap).double()''',
  '''def fit_pair_scores(heads, encs, ctx, img, cap, direction=None):
    gs, ss = 0, 0
    for p in PEERS:
        gs = gs + g_pairs_lists(heads[p], encs[p] if encs else None, ctx, p, 'fit', img, cap, direction=direction).double()''')
r('''    im, cp, _, _ = list_pairs(((qi, caps, ai), (qc, imgs, at)))
    g, s = fit_pair_scores(heads, encs, ctx, im, cp)''',
  '''    im, cp, _, _ = list_pairs(((qi, caps, ai), (qc, imgs, at)))
    g, s = fit_pair_scores(heads, encs, ctx, im, cp, direction=np.concatenate([np.zeros(len(ai) * N_SLOTS, np.int64), np.ones(len(at) * N_SLOTS, np.int64)]))''')
r('''            gr, _ = fit_pair_scores(heads, encs, ctx, img_, cap_)''', '''            gr, _ = fit_pair_scores(heads, encs, ctx, img_, cap_, direction=di)''')
# training step: per-row direction, per-direction losses kept for the routing check / diagnostic
r('''    n_i = int(ri.sum())
''', '''    n_i = int(ri.sum())
    dirs = np.concatenate([np.zeros(n_i, np.int64), np.ones(len(img) - n_i, np.int64)])
''')
r('''    loss, jb, gl = 0.0, 0.0, []''', '''    loss, jb, gl = 0.0, 0.0, []
    dl_i = dl_t = 0.0''')
r('''        g = g_pairs_lists(h, encs[p] if encs else None, ctx, p, 'fit', img, cap, grad=True, enc_grad=False, proj=proj, chunk=1 << 30)''',
  '''        g = g_pairs_lists(h, encs[p] if encs else None, ctx, p, 'fit', img, cap, grad=True, enc_grad=False, proj=proj, chunk=1 << 30, direction=dirs)''')
r('''        loss = loss + .5 * (.5 * li + .5 * lt)
''', '''        loss = loss + .5 * (.5 * li + .5 * lt)
        dl_i = dl_i + .25 * li; dl_t = dl_t + .25 * lt
''')
r('''    ctxs = dict(ai=ai, at=at, ri=ri, rt=rt)''', '''    ctxs = dict(ai=ai, at=at, ri=ri, rt=rt, dir_losses=(dl_i, dl_t), dirs=dirs)''')
# run_unit: schedule validation, reference identity, first-update graph check, direction-gradient diagnostic, cosine lr
r('''    init_hash = state_hash(heads) + (state_hash(nulls) if nulls is not None else '')
''', '''    init_hash = state_hash(heads) + (state_hash(nulls) if nulls is not None else '')
    sched = cfg['optimizer'].get('schedule', 'constant'); assert sched in ('constant', 'cosine'), sched
    directional = getattr(heads[PEERS[0]], 'directional', False)
    assert directional == (cfg['structure'] == 'GD')
    ref = X.reference(cfg)
    r38 = dict(schedule=sched, directional=directional, init_state_sha256_shared_keys=X.state_hash_shared(heads, nulls),
               reference_unit=ref['unit'] if ref else None, reference_init_equal=None, reference_chain_equal_at=[], lr_at_readout={})
    if ref is not None:   # read-only reuse check: same fresh head init (shared keys) as the R36/R37C reference unit of this parent/route
        r38['reference_init_equal'] = r38['init_state_sha256_shared_keys'] == ref['init']
        assert r38['reference_init_equal'], f"init differs from reference {ref['unit']}"
''')
r('''    hist['windows'] = [w for w in hist['windows'] if w['end_step'] <= start]''',
  '''    hist['windows'] = [w for w in hist['windows'] if w['end_step'] <= start]
    hist.setdefault('r38', {}).update({k: v for k, v in r38.items() if k not in hist['r38']})
    hist['r38'].setdefault('dir_grad_diag', {})
    last_lr = None''')
r('''            hist['batch_hash'].append(chain.hex()); hist['last_step'] = step
''', '''            hist['batch_hash'].append(chain.hex()); hist['last_step'] = step
            if ref is not None:   # read-only reuse check: identical sampled lists (batch-index hash chain) as the reference unit
                kk = len(hist['batch_hash']) - 1
                assert hist['batch_hash'][-1] == ref['chain'][kk], f"batch chain differs from reference {ref['unit']} at step {step}"
                if step not in hist['r38']['reference_chain_equal_at']:
                    hist['r38']['reference_chain_equal_at'].append(step)
            hist['r38']['lr_at_readout'][str(step)] = last_lr
''')
r('''        opt.zero_grad(set_to_none=True)
        loss.backward()
''', '''        diag = None
        if step in ckpt_steps or step == 0:   # R38 diagnostic: per-direction gradients on this training batch (no effect on .grad)
            diag, flat = X.dir_grads(heads, nulls, PEERS, *cx['dir_losses'])
            hist['r38']['dir_grad_diag'][str(step)] = diag
        if step == 0 and cfg.get('phase', 1) == 1:
            gc = first_step_graph_check(cfg, heads, nulls, ctx, idx, cx, diag, a0, b0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step == 0 and cfg.get('phase', 1) == 1:   # total gradient = sum of the two direction gradients (pre-clip)
            named, gi_, gt_ = flat
            err = max(float((((q.grad if q.grad is not None else torch.zeros_like(q)) - (a + b)).abs().max())) for (_, _, q), a, b in zip(named, gi_, gt_))
            scale = max(float(q.grad.abs().max()) for (_, _, q) in named if q.grad is not None)
            gc['sum_of_direction_grads_equals_total'] = dict(max_abs_err=err, max_abs_grad=scale, passed=bool(err <= 1e-6 * scale + 1e-12))
            gc['passed'] = bool(gc['passed'] and gc['sum_of_direction_grads_equals_total']['passed'])
            hist['r38']['step1_graph_check'] = gc
            write_json(out / 'STEP1_GRAPH_CHECK.json', gc)
            assert gc['passed'], f'R38 first-update graph check failed: {gc}'
''')
r('''        gwin.append(grad_class_stats(gl, cx))
        opt.step()''', '''        gwin.append(grad_class_stats(gl, cx))
        if sched == 'cosine':   # R38 common cosine (all objectives): lr(t) = lr_head * lr_multiplier(t), t = update index; optimizer never reset
            for pg in opt.param_groups:
                pg['lr'] = cfg['optimizer']['lr_head'] * X.lr_multiplier(step, cfg['steps'], 'cosine', cfg['optimizer']['cosine_floor'])
        last_lr = opt.param_groups[0]['lr']
        opt.step()''')
# graph check function (before finalize)
r('''def finalize(cfg, ctx, res_list, t_wall, base_out):''', '''@torch.no_grad()
def first_step_graph_check(cfg, heads, nulls, ctx, idx, cx, diag, a0, b0):
    """R38 in-unit check on the first training batch, before the first update (replaces a standalone GPU gate):
    directional: (1) routed scores == all-i2t scores == all-t2i scores bitwise (both outputs zero, both affines a0/b0);
    (2) == the scores of a freshly built shared G head of the same seed (trunk/out/affine bitwise those of 'G');
    (3) routing: dL_i2t/d(out_t2i, affine_t2i) == 0 and dL_t2i/d(out_i2t, affine_i2t) == 0 exactly, the own-direction output
    and the shared trunk get nonzero gradients from each direction. shared: both directions reach the single output.
    The sum check (total grad == sum of direction grads) is added after loss.backward()."""
    img, cap = idx[0], idx[1]; dirs = cx['dirs']
    out = dict(directional=bool(getattr(heads[PEERS[0]], 'directional', False)), n_pairs=int(len(img)), n_i2t_pairs=int((dirs == 0).sum()))
    ok = True
    if out['directional']:
        ref_heads = build_heads('G', a0, b0, cfg['init_seed']).to(ctx.dev)
        eq = {}
        for p in PEERS:
            kw = dict(chunk=1 << 30)
            g_r = g_pairs_lists(heads[p], None, ctx, p, 'fit', img, cap, direction=dirs, **kw)
            g_0 = g_pairs_lists(heads[p], None, ctx, p, 'fit', img, cap, direction=0, **kw)
            g_1 = g_pairs_lists(heads[p], None, ctx, p, 'fit', img, cap, direction=1, **kw)
            g_s = g_pairs_lists(ref_heads[p], None, ctx, p, 'fit', img, cap, **kw)
            eq[p] = dict(routed_eq_i2t=bool(torch.equal(g_r, g_0)), routed_eq_t2i=bool(torch.equal(g_r, g_1)), routed_eq_fresh_shared_G=bool(torch.equal(g_r, g_s)))
            ok &= all(eq[p].values())
        out['step0_scores'] = eq
        rt = {}
        for p in PEERS:
            d = diag[p]
            z = lambda g, k: d.get(g, {}).get(k, 0.0)
            rt[p] = dict(i2t_loss_to_t2i_extras=z('out_t2i', 'norm_i2t') + z('affine_t2i', 'norm_i2t'), t2i_loss_to_i2t_output=z('out_i2t', 'norm_t2i') + z('affine_i2t', 'norm_t2i'),
                         i2t_loss_to_own_output=z('out_i2t', 'norm_i2t'), t2i_loss_to_own_output=z('out_t2i', 'norm_t2i'),
                         trunk_from_i2t=z('trunk', 'norm_i2t'), trunk_from_t2i=z('trunk', 'norm_t2i'))
            ok &= rt[p]['i2t_loss_to_t2i_extras'] == 0.0 and rt[p]['t2i_loss_to_i2t_output'] == 0.0 and rt[p]['i2t_loss_to_own_output'] > 0 and rt[p]['t2i_loss_to_own_output'] > 0 and rt[p]['trunk_from_i2t'] > 0 and rt[p]['trunk_from_t2i'] > 0
        out['routing'] = rt
    else:
        rt = {p: dict(out_from_i2t=diag[p]['out_i2t']['norm_i2t'], out_from_t2i=diag[p]['out_i2t']['norm_t2i'], trunk_from_i2t=diag[p]['trunk']['norm_i2t'], trunk_from_t2i=diag[p]['trunk']['norm_t2i']) for p in PEERS}
        ok &= all(v > 0 for p in PEERS for v in rt[p].values())
        out['routing'] = rt
    sched = cfg['optimizer'].get('schedule', 'constant')
    out['lr_update0'] = cfg['optimizer']['lr_head'] * (X.lr_multiplier(0, cfg['steps'], 'cosine', cfg['optimizer']['cosine_floor']) if sched == 'cosine' else 1.0)
    out['lr_last_update'] = cfg['optimizer']['lr_head'] * (X.lr_multiplier(cfg['steps'] - 1, cfg['steps'], 'cosine', cfg['optimizer']['cosine_floor']) if sched == 'cosine' else 1.0)
    ok &= out['lr_update0'] == cfg['optimizer']['lr_head'] and (sched == 'constant' or abs(out['lr_last_update'] - .1 * cfg['optimizer']['lr_head']) < 1e-18)
    out['passed'] = bool(ok); out['utc'] = utc()
    return out


def finalize(cfg, ctx, res_list, t_wall, base_out):''')
r('''        assert sha256(ROOT / '%TREE%' / f) == h, f'code differs from frozen protocol: {f}\'''', '''        assert sha256(Path(__file__).resolve().parent / f) == h, f'code differs from frozen protocol: {f}\'''')
r('''                  code_sha256=code_hashes(),
                  batch_stats''', '''                  code_sha256=code_hashes(), r38=phases[-1]['train_history'].get('r38'),
                  batch_stats''')

def main():
    for tree, src_tree in (('r38f', 'r37'), ('r38c', 'r37c')):
        p = W / tree / 'train_unit.py'; t = p.read_text()
        assert 'import r38ext' not in t, 'already patched'
        for a, b in R:
            a2 = a.replace('%TREE%', src_tree)
            if tree == 'r38f' and "code differs from frozen protocol" in a2:
                a2 = a2 + "  # was ROOT/'r36' (nonexistent dir; would have failed every formal unit)"
                b = b + "  # R37: was ROOT/'r36'; R38: this tree's own directory"
            n = t.count(a2); assert n == 1, (tree, n, a2[:120])
            t = t.replace(a2, b.replace('%TREE%', src_tree))
        t = t.replace('"""R37', '"""R38: train_unit.py of R37 (%s/) + R38 edits (directional G readout per row direction, common cosine schedule, read-only reference\n(R36/R37C) init/batch-chain identity, first-update graph check, direction-gradient diagnostic); shared/constant path unchanged.\nR37' % src_tree, 1)
        p.write_text(t); print('patched', tree)

if __name__ == '__main__':
    main()
