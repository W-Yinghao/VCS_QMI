"""R27 audit 1 (target-direction stability on fixed FIT graphs) + audit 2 (SPG history reference) -- read-only replay.

Usage: audit_graphs.py OUT_CELL_DIR CELL NB LABEL=CKPT [LABEL=CKPT ...] -- <r25_run args>
  CKPT = path to a full-state file, or MISSING:<reason> (recorded as MISSING, nothing computed).
  Incremental: a label whose OUT_CELL_DIR/<label>/RESULT.json exists is skipped.

Per checkpoint (isolated process, frozen R27 code_v11 imported read-only; no optimizer step, nothing written back):
  1. production load_full_state (R27 hook: restores the ID buffers if the state has them) into two fresh VSE peers;
  2. GMM split the next epoch would use: eval_train on the FIT train pool (seed_all(cell seed) first);
  3. search queue of the other peer: final states -> the saved ID-bearing queue; the common prefix has NO IDs -> an
     ID-bearing queue is REBUILT IN MEMORY exactly as r27_calibrate.rebuild_queue (other peer's own labeled loader,
     generator seed 20261030, train-mode forward, no grad, BN state restored). Labelled as a reconstruction.
  4. per peer (A trains/searches B's queue; B searches A's), the first NB unlabeled FIT batches of a fixed generator
     (seed 20261029, as r27_calibrate) with B >= 2; query = current peer train-mode forward_emb_ctx (fixed torch seed
     per batch, BN state restored after the forward), other peer val mode; production search + current-peer re-encoding
     for i2t and t2i -> production graph (A, Y).
  5. on each graph: production targets mean / delta (code_v11 r27_delta_control) / vcs / bce / snce; raw and normalized
     objective values (f_vcs, f_bce, f_snce at every raw vector and every normalized target), ||raw||; pairwise angles;
     cap / fallback / invalid / interior counts; collisions; tighter tolerance (1e-9) re-solves; gradients of the
     auxiliary SCE wrt the query X for every target and of the native refiner SCE (refiner train mode, fixed seed).
  6. audit 2 on the same graphs: production SPG (history max) vs audit_spg history-min variant at 1e-6 and 1e-9, vs a
     tight FISTA (1e-9); optional SLSQP (reduced coordinates, CPU) on a few rows of the first graph.
No DEV/TEST read (only the train split is loaded).
"""
import os, sys, json, copy, time, hashlib
import numpy as np
import torch

import audit_common as C
from audit_common import T, angle, qstats, sync
import audit_spg as SPG
import r25_run
import r25_co_train as R
import r27_co_train as R27
import r27_interface as I
import r27_calibrate as RC
from r27_delta_control import generate_centered_mean
from data import get_dataset
from vse import VSE, infonce
from vocab import deserialize_vocab
from co_train import eval_train, split_prob

GEN_SEED, QUEUE_SEED, QUERY_SEED = 20261029, 20261030, 20261031
TARGETS = ('mean', 'delta', 'vcs', 'bce', 'snce')
TIGHT = 1e-9
TIGHT_CAP = 40000
ANG_THR = (1e-6, 1e-3, 1e-2, 5e-2)
SLSQP_ROWS = int(os.environ.get('AUDIT_SLSQP_ROWS', '0'))
LAM = json.load(open('/home/infres/yinwang/CS_QMI/handoff/r27_qtarget_20261029/LAMBDA_flickr.json'))['lambda']


def timed(fn, *a, **k):
    t = sync(); out = fn(*a, **k); return out, sync() - t


def info_counts(info, valid):
    d = dict(n_rows=int(valid.numel()), invalid_rows=int((~valid).sum()), fallback=int(info['fallback'].sum()))
    for k in ('cap', 'interior', 'near_log_boundary'):
        if k in info:
            d[k] = int(info[k].sum())
    if 'iters' in info:
        d['iters_mean'] = float(info['iters'].double().mean()); d['iters_max'] = int(info['iters'].max())
    if 'rel' in info:
        d['rel_max'] = float(info['rel'].max())
    return d


def collisions(U, A, Y):
    """U [B,d] float64 unit targets. Pairs i<j with angle below thresholds, split by identical / different A rows;
    rows whose target is within the threshold of a single atom."""
    B = U.shape[0]
    iu = torch.triu_indices(B, B, 1, device=U.device)
    ang = 2 * torch.asin(((U[iu[0]] - U[iu[1]]).norm(dim=1) / 2).clamp(max=1))
    sameA = (A[iu[0]] == A[iu[1]]).all(1)
    Yu = C.unit(Y.double())
    atom_ang = 2 * torch.asin((torch.cdist(U, Yu) / 2).clamp(max=1)).min(1).values
    out = dict(n_pairs=int(ang.numel()), n_pairs_identical_A=int(sameA.sum()), n_distinct_A_rows=int(torch.unique(A, dim=0).shape[0]))
    for t in ANG_THR:
        m = ang < t
        out[f'pairs_lt_{t:g}'] = int(m.sum())
        out[f'pairs_lt_{t:g}_identical_A'] = int((m & sameA).sum())
        out[f'pairs_lt_{t:g}_different_A'] = int((m & ~sameA).sum())
        out[f'rows_atom_lt_{t:g}'] = int((atom_ang < t).sum())
    return out


def grad_x(X, tgt, valid):
    Xl = X.detach().clone().requires_grad_(True)
    L = infonce(Xl.mm(tgt.t())) if bool(valid.all()) else infonce(Xl[valid].mm(tgt[valid].t()))
    g, = torch.autograd.grad(L, Xl)
    return g.double(), float(L.detach())


def gcmp(g1, g2):
    a, b = g1.flatten(), g2.flatten()
    rowcos = (g1 * g2).sum(1) / (g1.norm(dim=1) * g2.norm(dim=1)).clamp(min=1e-300)
    return dict(cos=float(a @ b / (a.norm() * b.norm()).clamp(min=1e-300)), rel_norm=float(a.norm() / b.norm().clamp(min=1e-300)),
                rowcos_median=float(rowcos.median()), rowcos_min=float(rowcos.min()))


def graph_audit(gpath, X, A_, Y, neigh, refiner, seed):
    """All per-graph computations. Returns summary dict; per-row arrays saved to gpath (.npz)."""
    A64, Y64 = A_.double(), Y.double()
    c, mu, muQ, delta = T.moments(A64, Y64)
    rows = {}
    out = dict(B=int(A_.shape[0]), M=int(A_.shape[1]), d=int(Y.shape[1]))
    # ---- production targets (fp32, as training) + info
    prod, valid, infos = {}, {}, {}
    for g in TARGETS:
        if g == 'delta':
            (Tg, vg, ig), dt = timed(generate_centered_mean, A_, Y)
        else:
            (Tg, vg, ig), dt = timed(T.generate, g, A_, Y)
        prod[g], valid[g], infos[g] = Tg, vg, ig
        out[f'prod_{g}'] = dict(seconds_generate=dt, **info_counts(ig, vg))
    # ---- raw vectors in float64 (direct solver calls = what generate computes internally)
    raw = {'mean': mu, 'delta': delta}
    (raw['vcs'], iv), tv = timed(T.solve_vcs, A64, Y64)
    out['prod_vcs']['seconds_solver'] = tv
    for g in ('bce', 'snce'):
        (raw[g], ig), tg = timed(T.solve_spg, g, A64, Y64)
        out[f'prod_{g}']['seconds_solver'] = tg
        out[f'prod_{g}']['iters_solver_max'] = int(ig['iters'].max())
    nrm = {}
    for g in TARGETS:
        zn = raw[g].norm(dim=1)
        nrm[g] = raw[g] / zn.clamp(min=1e-300).unsqueeze(1)
        rows[f'rawnorm_{g}'] = zn
        if g in ('vcs', 'bce', 'snce'):
            okz = torch.isfinite(raw[g]).all(1) & (zn > T.ZMIN)
            nrm[g] = torch.where(okz.unsqueeze(1), nrm[g], C.unit(mu))      # production fallback rule
        elif g == 'delta':
            nrm[g] = torch.where((zn > 1e-6).unsqueeze(1), nrm[g], C.unit(mu))
        out[f'prod_{g}']['fp32_target_vs_fp64_maxabs'] = float((prod[g].double() - nrm[g]).abs()[valid[g]].max()) if bool(valid[g].any()) else None
        out[f'prod_{g}']['raw_norm'] = qstats(zn)
    # ---- objective matrix: f_obj at raw and at normalized for every target
    om = {}
    for o in ('vcs', 'bce', 'snce'):
        f = T.OBJ[o][0]
        om[o] = dict(zero=qstats(f(torch.zeros_like(mu), A64, c, Y64)))
        for g in TARGETS:
            fr = f(raw[g], A64, c, Y64)
            fn = f(nrm[g], A64, c, Y64)
            rows[f'obj_{o}_raw_{g}'] = fr; rows[f'obj_{o}_norm_{g}'] = fn
            om[o][f'raw_{g}'] = qstats(fr); om[o][f'norm_{g}'] = qstats(fn)
    out['objectives'] = om
    # ---- pairwise angles between normalized targets
    pa = {}
    for i, g1 in enumerate(TARGETS):
        for g2 in TARGETS[i + 1:]:
            a = angle(nrm[g1], nrm[g2]); rows[f'angle_{g1}_{g2}'] = a; pa[f'{g1}|{g2}'] = qstats(a)
    out['pair_angles_rad'] = pa
    # ---- collisions
    out['collisions'] = {g: collisions(nrm[g], A64, Y64) for g in TARGETS}
    # ---- tighter tolerance (audit 1) + SPG history reference (audit 2)
    tt, a2 = {}, {}
    (Zv9, iv9), t = timed(T.solve_vcs, A64, Y64, tol_rel=TIGHT)
    tt['vcs_exact_tol1e-9'] = dict(note='solve_vcs ignores tol_rel (Newton stops at | ||z||-1 | <= 1e-12)', seconds=t,
                                   angle_vs_prod=qstats(angle(Zv9, raw['vcs'])), rel_max=float(iv9['rel'].max()))
    f = T.f_vcs
    ref_f = f(raw['vcs'], A64, c, Y64)
    for ref in ('max', 'min'):
        for tol in (1e-6, TIGHT):
            (Z, inf), t = timed(SPG.solve_spg_ref, 'vcs', A64, Y64, tol_rel=tol, cap=TIGHT_CAP if tol == TIGHT else T.SPG_CAP, ref=ref)
            a = angle(Z, raw['vcs'])
            a2[f'vcs|spg_{ref}|{tol:g}'] = dict(seconds=t, iters_mean=float(inf['iters'].double().mean()), iters_max=int(inf['iters'].max()),
                                               cap=int(inf['cap'].sum()), rel_max=float(inf['rel'].max()),
                                               decrease_accepts=int(inf['n_decrease_accepts'].sum()),
                                               angle_vs_exact=qstats(a), obj_exact_minus=qstats(ref_f - f(Z, A64, c, Y64)))
            if ref == 'max' and tol == TIGHT:
                tt['vcs_spg_tol1e-9_vs_exact'] = a2[f'vcs|spg_{ref}|{tol:g}']
                rows['tight_angle_vcs_spg'] = a
    slsqp_out = {}
    for g in ('bce', 'snce'):
        f = T.OBJ[g][0]
        (Zf, iff), tf = timed(T.solve_fista, g, A64, Y64, tol_rel=TIGHT, cap=TIGHT_CAP)
        ff = f(Zf, A64, c, Y64)
        a2[f'{g}|fista|1e-09'] = dict(seconds=tf, iters_mean=float(iff['iters'].double().mean()), iters_max=int(iff['iters'].max()),
                                      cap=int(iff['cap'].sum()), rel_max=float(iff['rel'].max()))
        sols = {}
        for ref in ('max', 'min'):
            for tol in (1e-6, TIGHT):
                if ref == 'max' and tol == 1e-6:
                    Z = raw[g]                                          # production solution (solve_spg)
                    (Zc, inf), t = timed(SPG.solve_spg_ref, g, A64, Y64, tol_rel=tol, ref='max')
                    same = bool(torch.equal(Zc, Z))
                else:
                    (Z, inf), t = timed(SPG.solve_spg_ref, g, A64, Y64, tol_rel=tol, cap=TIGHT_CAP if tol == TIGHT else T.SPG_CAP, ref=ref)
                    same = None
                fz = f(Z, A64, c, Y64)
                key = f'{g}|spg_{ref}|{tol:g}'
                sols[key] = Z
                a = angle(Z, Zf)
                rows[f'a2_angle_vs_fista_{g}_{ref}_{tol:g}'] = a
                a2[key] = dict(seconds=t, iters_mean=float(inf['iters'].double().mean()), iters_max=int(inf['iters'].max()),
                               cap=int(inf['cap'].sum()), rel_max=float(inf['rel'].max()), decrease_accepts=int(inf['n_decrease_accepts'].sum()),
                               rejects=int(inf['n_rejects'].sum()), angle_vs_fista_tight=qstats(a), obj_fista_minus=qstats(ff - fz),
                               obj_mean=float(fz[torch.isfinite(fz)].mean()), cert_excess_max=float((ff - fz - 2 * inf['res']).max()),
                               bitwise_equal_production=same)
        for k1, k2 in ((f'{g}|spg_max|1e-06', f'{g}|spg_min|1e-06'), (f'{g}|spg_max|1e-06', f'{g}|spg_max|1e-09'),
                       (f'{g}|spg_min|1e-06', f'{g}|spg_min|1e-09'), (f'{g}|spg_max|1e-09', f'{g}|spg_min|1e-09')):
            a2[f'angle {k1} vs {k2}'] = qstats(angle(sols[k1], sols[k2]))
        tt[f'{g}_spg_tol1e-9_vs_prod'] = dict(angle=a2[f'angle {g}|spg_max|1e-06 vs {g}|spg_max|1e-09'],
                                              seconds=a2[f'{g}|spg_max|1e-09']['seconds'], cap=a2[f'{g}|spg_max|1e-09']['cap'],
                                              rel_max=a2[f'{g}|spg_max|1e-09']['rel_max'],
                                              obj_tight_minus_prod=qstats(f(sols[f'{g}|spg_max|1e-09'], A64, c, Y64) - f(raw[g], A64, c, Y64)))
        tt[f'{g}_fista_tol1e-9_vs_prod'] = dict(angle=a2[f'{g}|spg_max|1e-06']['angle_vs_fista_tight'])
        rows[f'tight_angle_{g}_spg'] = angle(sols[f'{g}|spg_max|1e-09'], raw[g])
        if SLSQP_ROWS > 0:
            sl = C.slsqp_rows(g, A64, Y64, list(range(SLSQP_ROWS)))
            slsqp_out[g] = [dict(row=o['row'], seconds=o['seconds'], nit=o['nit'], success=o['success'], feasible=o['feasible'], message=o['message'],
                                 angle_vs_fista=float(angle(o['z'].to(Zf).unsqueeze(0), Zf[o['row']:o['row'] + 1])[0]) if o['feasible'] else None,
                                 angle_vs_prod=float(angle(o['z'].to(Zf).unsqueeze(0), raw[g][o['row']:o['row'] + 1])[0]) if o['feasible'] else None,
                                 angle_vs_spg_min=float(angle(o['z'].to(Zf).unsqueeze(0), sols[f'{g}|spg_min|1e-06'][o['row']:o['row'] + 1])[0]) if o['feasible'] else None,
                                 obj_fista_minus=(float(ff[o['row']]) - o['f']) if o['feasible'] else None)
                            for o in sl]
    out['tight_tolerance'] = tt
    out['audit2_spg'] = a2
    if slsqp_out:
        out['audit2_slsqp'] = slsqp_out
    # ---- downstream gradients wrt the query X
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    with torch.no_grad():
        proto = refiner(neigh)                       # production: refiner in train mode
    gs, losses = {}, {}
    gs['native'], losses['native'] = grad_x(X, proto, torch.ones(X.shape[0], dtype=torch.bool, device=X.device))
    for g in TARGETS:
        if int(valid[g].sum()) >= 2:
            gs[g], losses[g] = grad_x(X, prod[g], valid[g])
    gd = {}
    names = [k for k in ('native',) + TARGETS if k in gs]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            gd[f'{b}|{a}'] = gcmp(gs[b], gs[a])
    out['grad_x'] = dict(pairs=gd, sce_loss=losses, norms={k: float(v.norm()) for k, v in gs.items()},
                         note=f'production loss per direction = 0.5*SCE_native + 0.5*lam*SCE_aux, lam={LAM}; values above unscaled')
    np.savez_compressed(gpath, **{k: v.detach().cpu().numpy() for k, v in rows.items()})
    return out


def main():
    sep = sys.argv.index('--')
    out_dir, cell, nb = sys.argv[1], sys.argv[2], int(sys.argv[3])
    specs = [a.split('=', 1) for a in sys.argv[4:sep]]
    opt = r25_run.parse(sys.argv[sep + 1:])
    opt.arm, opt.lam = 'a3_vcs', LAM          # not used for training; noisy_branches is not called
    R27.install()
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, "%s_vocab.json" % opt.data_name)); vocab.add_word('<mask>')
    opt.vocab_size = len(vocab)
    caps, ims = get_dataset(opt.data_path, opt.data_name, "train")
    noisy_loader, data_size, clean_labels = R.get_loader(caps, ims, vocab, "warmup", opt.batch_size, opt.workers, opt.noise_ratio, opt.noise_file)
    for label, ck in specs:
        od = os.path.join(out_dir, label)
        if os.path.exists(os.path.join(od, 'RESULT.json')):
            print('skip (done)', label, flush=True); continue
        os.makedirs(os.path.join(od, 'graphs'), exist_ok=True)
        if ck.startswith('MISSING'):
            json.dump(dict(cell=cell, label=label, status='MISSING', reason=ck.split(':', 1)[-1]), open(os.path.join(od, 'RESULT.json'), 'w'), indent=1)
            print('MISSING', label, flush=True); continue
        t_ck = time.time()
        meta = dict(cell=cell, label=label, checkpoint=ck, checkpoint_sha256=C.sha(ck), gpu=torch.cuda.get_device_name(0),
                    host=os.uname().nodename, job=os.environ.get('SLURM_JOB_ID'), nb_per_peer=nb, seeds=dict(cell=opt.seed, fit_batches=GEN_SEED,
                    queue_rebuild=QUEUE_SEED, query_dropout_base=QUERY_SEED), lam=LAM, code_v11_sha256=C.code_hashes(), audit_sha256=C.own_hashes(),
                    tol_prod=T.TOL_REL, tol_tight=TIGHT, tight_cap=TIGHT_CAP, slsqp_rows=SLSQP_ROWS)
        A = VSE(opt, vocab.word2idx); B = VSE(opt, vocab.word2idx)
        st = R.load_full_state(ck, opt, A, B)
        has_ids = st['model_A'].get('r27_queue_ids') is not None and st['model_B'].get('r27_queue_ids') is not None
        meta.update(state_epoch=int(st['epoch']), state_has_queue_ids=has_ids)
        r25_run.seed_all(opt.seed)
        epoch = st['epoch'] + 1
        t = time.time()
        prob_A, prob_B, _ = eval_train(opt, A, B, noisy_loader, data_size, copy.deepcopy(st['all_loss']), clean_labels, epoch)
        meta['eval_train_seconds'] = time.time() - t
        pred_A = split_prob(prob_A, opt.p_threshold); pred_B = split_prob(prob_B, opt.p_threshold)
        for m, prob, pred in ((A, prob_A, pred_A), (B, prob_B, pred_B)):
            cp = prob[pred]; m.avg_clean_prob = cp.mean() if len(cp) > 0 else 0.0
        meta['split'] = dict(n_clean_pred_A=int(np.sum(pred_A)), n_clean_pred_B=int(np.sum(pred_B)), n=int(len(pred_A)))
        loaders = {'A': R.get_loader(caps, ims, vocab, "train", opt.batch_size, opt.workers, opt.noise_ratio, opt.noise_file, pred=pred_B, prob=prob_B),
                   'B': R.get_loader(caps, ims, vocab, "train", opt.batch_size, opt.workers, opt.noise_ratio, opt.noise_file, pred=pred_A, prob=prob_A)}
        graphs, peers = [], {}
        for pi, (name, net, net2) in enumerate((('A', A, B), ('B', B, A))):
            pinfo = {}
            if has_ids:
                q2 = net2.memory_queue
                pinfo['queue'] = 'saved ID-bearing queue of the other peer (state)'
            else:
                g0 = torch.Generator(); g0.manual_seed(QUEUE_SEED)
                torch.manual_seed(QUEUE_SEED + pi); torch.cuda.manual_seed(QUEUE_SEED + pi)
                ltd = loaders['B' if name == 'A' else 'A'][0].dataset
                lq = torch.utils.data.DataLoader(ltd, batch_size=opt.batch_size, shuffle=True, generator=g0, pin_memory=True,
                                                 collate_fn=loaders[name][0].collate_fn, num_workers=0)
                t = time.time()
                q2, nbq = RC.rebuild_queue(opt, net2, lq)
                pinfo['queue'] = 'RECONSTRUCTION: in-memory ID-bearing queue rebuilt as r27_calibrate.rebuild_queue (prefix slots carry no IDs)'
                pinfo['queue_rebuild'] = dict(batches=nbq, seconds=time.time() - t, is_full=bool(q2.is_full))
            n, v = I.usable(q2)
            pinfo.update(queue_filled=int(n), queue_id_bearing=int(v.sum()))
            ut = loaders[name][1]
            gen = torch.Generator(); gen.manual_seed(GEN_SEED)
            utl = torch.utils.data.DataLoader(ut.dataset, batch_size=opt.batch_size, shuffle=True, generator=gen, pin_memory=True,
                                              collate_fn=ut.collate_fn, num_workers=0)
            net.train_start(); net2.val_start()
            k, bh = 0, []
            for batch in utl:
                if k >= nb:
                    break
                imgs, ilen, txt, tlen, ids = batch[0], batch[1], batch[2], batch[3], batch[4]
                if imgs.size(0) < 2:
                    continue
                bh.append(hashlib.sha256(np.asarray(ut.dataset.r27_gidx[np.asarray(ids, dtype=np.int64)]).tobytes()).hexdigest()[:16])
                keep = (copy.deepcopy(net.img_enc.state_dict()), copy.deepcopy(net.txt_enc.state_dict()))
                s = QUERY_SEED + 1000 * pi + k
                torch.manual_seed(s); torch.cuda.manual_seed(s)
                net.train_start()
                with torch.no_grad():
                    xi, xt, _, mi, mt = I.forward_emb_ctx(net, imgs, txt, tlen, imgae_lengths=ilen)
                    xi2, xt2, _ = net2.forward_emb(imgs, txt, tlen, imgae_lengths=ilen)
                net.img_enc.load_state_dict(keep[0]); net.txt_enc.load_state_dict(keep[1])
                for br, X, Xp, mean in (('i2t', xi, xi2, mt), ('t2i', xt, xt2, mi)):
                    gid = f'{name}_b{k}_{br}'
                    S, qs = I.search(q2, Xp, br, opt.k)
                    if S is None:
                        graphs.append(dict(graph=gid, status='no_proto', **qs)); continue
                    Y = I.reencode(net, br, S['cols'], mean)
                    ref = net.i2t_refiner_node if br == 'i2t' else net.t2i_refiner_node
                    t = time.time()
                    g = graph_audit(os.path.join(od, 'graphs', gid + '.npz'), X, S['A'], Y, Y[S['inv']], ref, s + 500000 + (br == 't2i'))
                    g.update(graph=gid, status='ok', seconds=time.time() - t, usable_frac=qs['nv'] / max(qs['n'], 1))
                    graphs.append(g)
                    print(json.dumps(dict(label=label, graph=gid, M=g['M'], sec=round(g['seconds'], 1),
                                          ang_vcs_bce=round(g['pair_angles_rad']['vcs|bce']['median'], 5),
                                          snce_spg_max_vs_min=round(g['audit2_spg']['angle snce|spg_max|1e-06 vs snce|spg_min|1e-06']['max'], 6))), flush=True)
                k += 1
            pinfo['batch_capid_sha16'] = bh
            peers[name] = pinfo
        meta['peers'] = peers
        meta['seconds_checkpoint'] = time.time() - t_ck
        json.dump(dict(status='OK', **meta, graphs=graphs), open(os.path.join(od, 'RESULT.json'), 'w'), indent=1, default=float)
        print('DONE', label, round(time.time() - t_ck, 1), flush=True)
        del A, B, st, q2, net, net2
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
