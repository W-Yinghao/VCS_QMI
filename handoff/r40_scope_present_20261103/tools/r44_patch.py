"""R44 production routing patch (applied once to the copied train_unit.py of r44f (from r42f) and r44c (from r41c); every anchor asserted unique).
ROUTE_PRESENT / ROUTE_RANDOM (cfg['routing'], supervision stays OBS_MISSING_NULL = N lists, labels unchanged):
  head + null receive grad of the full-N loss L_h (the unchanged N loss of ALL); the student encoders receive grad of
  L_e = sum_peers .5 * (.5 * loss_list_masked(i2t, m_i, p_i2t) + .5 * loss_list_masked(t2i, m_t, p_t2i)) (R37 masked row-mean / fixed p_d);
  m = observed-present rows (D36.valid_rows) for ROUTE_PRESENT, an independent Bernoulli(p_d) row mask for ROUTE_RANDOM drawn from a stateless
  generator default_rng([init_seed, 44, step]) (never consumes the sampler / init RNG; same sequence for J / BCE / NCE; resume-safe).
  Two torch.autograd.grad calls on the one shared forward graph (never L_h + L_e in one backward), the grads assigned to .grad, then the
  unchanged per-peer single clip (head + null + student) and the single optimizer.step. p_d = ctx.present (fixed from the full FIT observed
  relation + candidate tables, R40 present_recheck); p_d = 0 -> the unit refuses to run (no epsilon).
  In-unit fail-fast at updates 0 and 1 (update 0 is vacuous for the encoder: zero-init head output; update 1 must have a nonzero ALL encoder grad): (a) all-ones mask with p = 1 reproduces the ALL encoder gradient (rel 1e-5); (b) the head gradient equals an
  independent grad of the ALL loss (bitwise); (c) the encoder gradient equals an independent grad of L_e (bitwise); (d) masked rows have exactly
  zero encoder score-gradient and (if any) nonzero head score-gradient. Every update: masked rows' encoder score-gradient exactly zero (asserted);
  optimizer steps counted (one per update, asserted at each readout).
Usage: python3 tools/r44_patch.py <path/to/train_unit.py>"""
import sys
p = sys.argv[1]; s = open(p).read()


def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:80], s.count(a)); s = s.replace(a, b)


rep("def step_lists(cfg, heads, nulls, encs, ctx, rng, rng2, pool, oracle, stud=None):",
    "def step_lists(cfg, heads, nulls, encs, ctx, rng, rng2, pool, oracle, stud=None, step=None):")
rep("""    loss, jb, gl = 0.0, 0.0, []
    dl_i = dl_t = 0.0
""", """    route = cfg.get('routing')   # R44: ROUTE_PRESENT / ROUTE_RANDOM (head: full N; encoder: masked conditional rows / fixed p_d)
    if route:
        assert route in ('ROUTE_PRESENT', 'ROUTE_RANDOM') and sup == 'OBS_MISSING_NULL' and stud is not None and step is not None
        pmi, pmt = D36.valid_rows(lists); pi_, pt_ = ctx.present['p_i2t'], ctx.present['p_t2i']
        if route == 'ROUTE_RANDOM':   # independent Bernoulli(p_d) per query row; stateless stream (never the sampler / init RNG)
            r3 = np.random.default_rng([int(cfg['init_seed']), 44, int(step)])
            emi, emt = r3.random(len(pmi)) < pi_, r3.random(len(pmt)) < pt_
        else:
            emi, emt = pmi, pmt
        Ei = torch.as_tensor(emi, device=ctx.dev, dtype=torch.float32); Et = torch.as_tensor(emt, device=ctx.dev, dtype=torch.float32)
        stats.update(route_present_rows_i2t=float(pmi.sum()), route_present_rows_t2i=float(pmt.sum()), route_mask_rows_i2t=float(emi.sum()), route_mask_rows_t2i=float(emt.sum()),
                     route_mask_and_present_i2t=float((emi & pmi).sum()), route_mask_and_present_t2i=float((emt & pmt).sum()),
                     route_empty_mask_batch_i2t=float(emi.sum() == 0), route_empty_mask_batch_t2i=float(emt.sum() == 0), route_empty_mask_batch_both=float(emi.sum() + emt.sum() == 0),
                     route_mask_unique_queries_i2t=float(len(np.unique(qi[emi]))), route_mask_unique_queries_t2i=float(len(np.unique(qc[emt]))))
    le = 0.0
    loss, jb, gl = 0.0, 0.0, []
    dl_i = dl_t = 0.0
""")
rep("""        loss = loss + .5 * (.5 * li + .5 * lt)
        dl_i = dl_i + .25 * li; dl_t = dl_t + .25 * lt
""", """        loss = loss + .5 * (.5 * li + .5 * lt)
        dl_i = dl_i + .25 * li; dl_t = dl_t + .25 * lt
        if route:   # R44 encoder objective (same rows / scores; masked row sum / (n p_d), R37 loss_list_masked)
            le = le + .5 * (.5 * loss_list_masked(cfg['objective'], gi, A_i, Ei, pi_) + .5 * loss_list_masked(cfg['objective'], gt, A_t, Et, pt_))
""")
rep("""    ctxs = dict(ai=ai, at=at, ri=ri, rt=rt, dir_losses=(dl_i, dl_t), dirs=dirs, present_masks=(mi, mt) if present_only else None)""",
    """    ctxs = dict(ai=ai, at=at, ri=ri, rt=rt, dir_losses=(dl_i, dl_t), dirs=dirs, present_masks=(mi, mt) if present_only else None,
                route=route, loss_enc=le if route else None, route_masks=(emi, emt) if route else None, A=(A_i, A_t), p_d=(pi_, pt_) if route else None)""")
rep("present_probs=ctx.present if cfg['supervision'] == 'N_PRESENT_ONLY' else None,",
    "present_probs=ctx.present if (cfg['supervision'] == 'N_PRESENT_ONLY' or cfg.get('routing')) else None,")
rep("""    if cfg['supervision'] == 'N_PRESENT_ONLY':   # R40: p_d fixed from the full FIT observed relation (recomputed independently, asserted)""",
    """    if cfg.get('routing'):   # R44: p_d of the routed encoder objective = the R40 fixed presence probabilities (recomputed, asserted); p_d = 0 -> unavailable
        r40['present_check'] = R40.present_recheck(d, ctx.present, D36.null_lists, D36.valid_rows)
        write_json(out / 'PRESENT_CHECK.json', r40['present_check'])
        assert r40['present_check']['passed'], r40['present_check']
        if not (ctx.present['p_i2t'] > 0 and ctx.present['p_t2i'] > 0):
            raise RuntimeError(f"R44: p_d = 0 for {cfg['cond']} ({ctx.present}): routing technically unavailable (no epsilon)")
    if cfg['supervision'] == 'N_PRESENT_ONLY':   # R40: p_d fixed from the full FIT observed relation (recomputed independently, asserted)""")
rep("""    hist.setdefault('r39', dict(joint=joint, supervision=cfg['supervision'], lr_encoder_at_readout={}, param_distance={}, grad_split_windows=[]))""",
    """    hist.setdefault('r39', dict(joint=joint, supervision=cfg['supervision'], lr_encoder_at_readout={}, param_distance={}, grad_split_windows=[]))
    hist.setdefault('r44', dict(routing=cfg.get('routing'), updates=0, opt_steps=0, masked_rows=0, masked_rows_enc_grad_nonzero=0, masked_rows_head_grad_nonzero=0,
                                mask_rows=0, query_rows=0, empty_mask_batches=0, mask_hash=None, routing_check=None,
                                rule='head: grad of the full-N loss; encoder: grad of the masked conditional loss (fixed p_d); two autograd.grad on one graph; one clip per peer, one step'))""")
rep("""        loss, jb, idx, st, gl, cx = step_lists(cfg, heads, nulls, encs, ctx, rng, rng2, pool, oracle, stud)""",
    """        loss, jb, idx, st, gl, cx = step_lists(cfg, heads, nulls, encs, ctx, rng, rng2, pool, oracle, stud, step=step)""")
rep("""        opt.zero_grad(set_to_none=True)
        loss.backward()
""", """        opt.zero_grad(set_to_none=True)
        if cx['route']:   # R44 routed backward (two grads on the shared graph, assigned per block)
            route_backward(cfg, heads, nulls, stud, loss, cx, gl, step, hist, out)
        else:
            loss.backward()
""")
rep("""            assert all(v > 0 for v in hist['r39']['update1_student_grad_norm'].values()), hist['r39']['update1_student_grad_norm']""",
    """            assert all(v > 0 for v in hist['r39']['update1_student_grad_norm'].values()) or (cx['route'] and sum(float(m_.sum()) for m_ in cx['route_masks']) == 0), hist['r39']['update1_student_grad_norm']   # R44: an all-zero encoder mask gives a zero data gradient""")
rep("""        opt.step()
        if step == 1 and cfg.get('phase', 1) == 1:   # R40 one-step active-path rule""",
    """        opt.step()
        if cx['route']:
            hist['r44']['opt_steps'] += 1; hist['r44']['updates'] += 1
            assert hist['r44']['opt_steps'] == hist['r44']['updates'], 'R44: one optimizer step per update'
        if step == 1 and cfg.get('phase', 1) == 1:   # R40 one-step active-path rule""")
rep("""@torch.no_grad()
def first_step_graph_check(cfg, heads, nulls, ctx, idx, cx, diag, a0, b0):""",
    '''def route_backward(cfg, heads, nulls, stud, loss, cx, gl, step, hist, out):
    """R44: head/null .grad = grad of the full-N loss; student .grad = grad of the masked conditional loss; score .grad = full-loss score grads
    (grad_class_stats). Update-0 fail-fast routing checks (see tools/r44_patch.py)."""
    hp = [q for p in PEERS for q in peer_params(heads, nulls, p) if q.requires_grad]
    ep = [q for p in PEERS for q in stud[p].trainable()]
    sc = [t for pair in gl for t in pair]
    emi, emt = cx['route_masks']; r44 = hist['r44']
    if step in (0, 1) and cfg.get('phase', 1) == 1:   # update 0: head output zero-init -> encoder grads exactly 0 (vacuous); update 1: non-trivial
        A_i, A_t = cx['A']; one_i = torch.ones(len(emi), device=sc[0].device); one_t = torch.ones(len(emt), device=sc[0].device)
        l1 = sum(.5 * (.5 * loss_list_masked(cfg['objective'], gi, A_i, one_i, 1.0) + .5 * loss_list_masked(cfg['objective'], gt, A_t, one_t, 1.0)) for gi, gt in gl)
        ge_all = torch.autograd.grad(loss, ep, retain_graph=True, allow_unused=True)
        ge_one = torch.autograd.grad(l1, ep, retain_graph=True, allow_unused=True)
        gh_ref = torch.autograd.grad(loss, hp, retain_graph=True, allow_unused=True)
        ge_ref = torch.autograd.grad(cx['loss_enc'], ep, retain_graph=True, allow_unused=True)
    gh = torch.autograd.grad(loss, hp + sc, retain_graph=True, allow_unused=True)
    ge = torch.autograd.grad(cx['loss_enc'], ep + sc, allow_unused=True)
    for q, g in zip(hp, gh[:len(hp)]):
        q.grad = g
    for q, g in zip(ep, ge[:len(ep)]):
        q.grad = g
    nh, ne = gh[len(hp):], ge[len(ep):]
    for t, g in zip(sc, nh):
        t.grad = g
    zero = lambda g, t: torch.zeros_like(t) if g is None else g
    mrows = enz = hnz = 0
    for k, t in enumerate(sc):   # order: peer A (i2t, t2i), peer B (i2t, t2i)
        m = torch.as_tensor(emi if k % 2 == 0 else emt, device=t.device)
        gE, gH = zero(ne[k], t), zero(nh[k], t)
        mrows += int((~m).sum()); enz += int((gE[~m] != 0).any(1).sum()); hnz += int((gH[~m] != 0).any(1).sum())
    r44['masked_rows'] += mrows; r44['masked_rows_enc_grad_nonzero'] += enz; r44['masked_rows_head_grad_nonzero'] += hnz
    r44['mask_rows'] += int(emi.sum() + emt.sum()); r44['query_rows'] += int(len(emi) + len(emt)); r44['empty_mask_batches'] += int(emi.sum() + emt.sum() == 0)
    h_ = hashlib.sha256((r44['mask_hash'] or '').encode()); h_.update(np.packbits(np.concatenate([emi, emt])).tobytes()); r44['mask_hash'] = h_.hexdigest()
    assert enz == 0, f'R44: {enz} masked rows with a nonzero encoder score-gradient at step {step}'
    if step in (0, 1) and cfg.get('phase', 1) == 1:
        def mx(a, b, qs):
            return max((float((zero(x, q) - zero(y, q)).abs().max()) for x, y, q in zip(a, b, qs)), default=0.0)
        scale = max(float(zero(g, q).abs().max()) for g, q in zip(ge_all, ep))
        chk = dict(all_ones_p1_reproduces_ALL_encoder_grad=dict(max_abs_err=mx(ge_one, ge_all, ep), max_abs_grad=scale, tol_rel=1e-5),
                   head_grad_equals_ALL_head_grad=dict(max_abs_err=mx(gh[:len(hp)], gh_ref, hp), bitwise=all(torch.equal(zero(a, q), zero(b, q)) for a, b, q in zip(gh[:len(hp)], gh_ref, hp))),
                   encoder_grad_equals_conditional_objective=dict(bitwise=all(torch.equal(zero(a, q), zero(b, q)) for a, b, q in zip(ge[:len(ep)], ge_ref, ep))),
                   masked_rows=dict(rows=mrows, enc_score_grad_nonzero=enz, head_score_grad_nonzero=hnz),
                   mask_rows=dict(i2t=int(emi.sum()), t2i=int(emt.sum()), n=int(len(emi))), p_d=dict(i2t=cx['p_d'][0], t2i=cx['p_d'][1]),
                   routing=cfg['routing'], step=int(step))
        chk['all_ones_p1_reproduces_ALL_encoder_grad']['passed'] = bool(chk['all_ones_p1_reproduces_ALL_encoder_grad']['max_abs_err'] <= 1e-5 * scale + 1e-12)
        chk['all_ones_p1_reproduces_ALL_encoder_grad']['nontrivial'] = bool(scale > 0)
        chk['passed'] = bool(chk['all_ones_p1_reproduces_ALL_encoder_grad']['passed'] and chk['head_grad_equals_ALL_head_grad']['bitwise']
                             and chk['encoder_grad_equals_conditional_objective']['bitwise'] and enz == 0 and (mrows == 0 or hnz > 0) and (step == 0 or scale > 0))
        rc = dict(r44['routing_check'] or {}); rc[f'update{step}'] = chk; rc['passed'] = all(v['passed'] for k_, v in rc.items() if k_.startswith('update'))
        r44['routing_check'] = rc; write_json(out / 'ROUTING_CHECK.json', rc)
        assert chk['passed'], f'R44 routing check failed at update {step}: {chk}'


@torch.no_grad()
def first_step_graph_check(cfg, heads, nulls, ctx, idx, cx, diag, a0, b0):''')   # R44 fix: route_backward goes ABOVE the no_grad decorator (CPU check 1084603)
if 'import hashlib' not in s:
    rep('import argparse', 'import argparse, hashlib', 1) if 'import argparse,' not in s else None
open(p, 'w').write(s)
print('R44 patched', p)
