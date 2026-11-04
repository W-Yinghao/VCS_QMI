"""R50 (package VCS_Layout_Audit_R50_20261104, docs/SERVER_PLAN_CN.md sections 1-2; protocol FIT_FIXED_CENTER_V1): in-unit layout checks, byte-identical
in r50f / r50c (dataset constants come from the tree's common.py). Called by train_unit.run_unit after the step-0 readout and after the final readout
(phase 1). A private numpy Generator and an explicit torch CPU Generator are used and the torch RNG is forked, so the run's sampler / torch streams and the
head / student / DEV-cache tensors are untouched (state digest asserted before / after).
  P  protocol: cache manifest protocol_id == FIT_FIXED_CENTER_V1; JOINT: centring_mode 'global', and the DEV centring vector == the FIT centring vector
     (bitwise, shape [1, d]) for every peer / modality: no DEV statistic and no item-indexed (batch-layout) centre is ever applied.
  B  (JOINT) DEV items encoded (a) through the production full-split eval-table path, (b) one item at a time, (c) mixed with unrelated DEV items in a shuffled
     order with chunk 37: equal within TOL_VEC.
  C  duplicates: Flickr DEV image k = dev_ims rows 5k..5k+4 (raw rows checked identical): every raw row gives the vector of image k; FIT repeated image IDs in
     one training batch map to one student row (pairs() dedup) and a raw batch with repeated items gives equal rows.
  D  (JOINT) opaque-ID renaming: the DEV raw storage (region features / tokens) permuted with an explicit synchronous map; encoding the mapped IDs gives the
     original vectors.
  E  (amendment A3, R50_AMENDMENT_A3.json) E1: with the SAME student vectors (renamed pool, permuted pair order, chunk 333) max|delta g| <= TOL_SCORE (absolute,
     unchanged); E2: with the pool RE-ENCODED in the renamed order (chunks 97 / 301), max|delta g| <= TOL_SCORE * max(1, max|residual|) - the fp32 encoder
     chunk numerics (vectors, check B / TOL_VEC) propagate through the head in proportion to the residual magnitude; at step 0 (|residual| < 1) = TOL_SCORE.
  E  nonzero random residual head (copy of the unit head whose zero-init output layer is redrawn from a private generator): critic scores of random DEV pairs
     are equal within TOL_SCORE when the item pool is renamed / re-chunked (JOINT: the student pool re-encoded in the renamed order with chunk 97 / 301;
     frozen: the cache rows renamed) and the pairs are scored in a permuted order with chunk 333; the residual must be nonzero.
  F  (JOINT) constant raw student output: the normalised student vectors of all DEV items and of a FIT sample are bitwise one vector (no per-item code).
  P also asserts the amendment-A2 tie policy (numpy dispatch without AVX-512; NPY_DISABLE_CPU_FEATURES exported by every R50 sbatch).
  G  DEV truth: the DEV H top-100 candidate tables (evals.dev_tables) recomputed with the DEV truth vector replaced by a random relabelling are bitwise
     identical (truth enters only coverage / metrics); the oracle module is loaded only in the oracle runner process.
  Read-only diagnostics (final step, JOINT): constant-ness of the raw student output (mean pairwise cosine of raw outputs of random DEV items) and the
  variance of the raw / centred outputs. They never replace a readout."""
from __future__ import annotations
import copy
import os
import sys
from types import SimpleNamespace
import numpy as np
import torch

import common as C
from common import PEERS, N_DEV_IMG, N_DEV_CAP, N_TRAIN_IMG, N_TRAIN_CAP, write_json, utc
import evals as EV
import r39ext as R9

PROTOCOL_ID = 'FIT_FIXED_CENTER_V1'
TOL_VEC = 1e-4     # registered (amendment A1, R50_AMENDMENT_A1.json; was 1e-5): max |delta| of unit vectors when only the chunking / batch composition / item order /
                   # item naming changes (fp32 GEMM / cuDNN BiGRU over padded chunks depend on the batch); = the R39 step-0 equivalence tolerance; bitwise fraction recorded
TOL_SCORE = 1e-4   # registered (amendment A1; was 1e-5): max |delta| of critic scores g under the same changes
MIN_RESIDUAL_STD = 1e-4   # E: the redrawn residual must actually move the scores
SEED = 5050


def _cmp(a, b):
    d = (a.double() - b.double()).abs()
    return dict(n=int(d.shape[0]), max_abs=float(d.max()) if d.numel() else 0.0, bitwise_rows=float((d == 0).all(-1).double().mean()) if d.numel() else 1.0)


def _digest(ctx, heads, stud):
    t = {f'h_{k}': v for k, v in heads.state_dict().items()}
    if stud is not None:
        t.update({f's_{k}': v for k, v in stud.state_dict().items()})
    t.update({f'c_dev_{k}_{p}': ctx.d[('dev', k, p)] for k in ('img', 'cap') for p in PEERS})
    return R9.state_digest(t)


def numpy_tie_policy():
    """amendment A2: numpy dispatch without AVX-512 (exact DEV score ties are then ordered identically on every node by the evaluator's argsort)"""
    try:
        from numpy._core._multiarray_umath import __cpu_features__ as F
    except ImportError:   # numpy 1.x
        from numpy.core._multiarray_umath import __cpu_features__ as F
    avx = {k: bool(v) for k, v in F.items() if k.startswith('AVX512')}
    return dict(numpy=np.__version__, env=os.environ.get('NPY_DISABLE_CPU_FEATURES'), avx512_dispatch_enabled=[k for k, v in avx.items() if v and (k == 'AVX512F' or '_' in k)],
                passed=not avx.get('AVX512F', False) and not any(v for k, v in avx.items() if '_' in k))


def protocol_check(ctx):
    man = ctx.d['manifest']
    rec = dict(cache_protocol_id=man.get('protocol_id'), fit_center_source=(man.get('fit_center') or {}).get('source'), tie_policy=numpy_tie_policy())
    ok = rec['cache_protocol_id'] == PROTOCOL_ID and rec['tie_policy']['passed']
    J = ctx.joint
    if J is not None:
        same = {f'{k}_{p}': bool(torch.equal(J.mu[('dev', k, p)], J.mu[('fit', k, p)])) for k in ('img', 'cap') for p in PEERS}
        single = all(J.mu[(s, k, p)].ndim == 2 and J.mu[(s, k, p)].shape[0] == 1 for s in ('fit', 'dev') for k in ('img', 'cap') for p in PEERS)
        rec.update(joint_mode=J.mode, joint_protocol_id=J.manifest.get('protocol_id'), dev_center_equals_fit_center=same, centers_single_row=bool(single))
        ok = ok and J.mode == 'global' and rec['joint_protocol_id'] == PROTOCOL_ID and all(same.values()) and single
    rec['passed'] = bool(ok)
    return rec


def truth_check(ctx, gen):
    """G: DEV top-100 tables do not depend on the DEV truth vector; no oracle module in a non-oracle process."""
    old = EV.DEV_GT
    try:
        EV.DEV_GT = gen.permutation(N_DEV_CAP) % N_DEV_IMG
        tab = EV.dev_tables(ctx.d['native_dev'])
    finally:
        EV.DEV_GT = old
    ref = ctx.d['dev_tab']
    same = {k: bool(np.array_equal(tab[k], ref[k])) for k in ('i_top', 't_top')}
    oracle_loaded = [m for m in sys.modules if m.split('.')[-1] == C.ORACLE_MODULE]
    oracle_proc = C.GUARD.get('role') == 'oracle'
    rec = dict(top100_tables_identical_under_truth_relabel=same, coverage_changed_under_truth_relabel=bool(not np.array_equal(tab['cov_t'], ref['cov_t'])),
               oracle_modules_loaded=oracle_loaded, oracle_process=oracle_proc)
    rec['passed'] = bool(all(same.values()) and (oracle_proc or not oracle_loaded))
    return rec


@torch.no_grad()
def head_checks(ctx, heads, stud, gen, g_fn):
    """E (both frozen and JOINT units)."""
    dev = ctx.dev; out = {}
    n = 1500
    img = gen.integers(0, N_DEV_IMG, n); cap = gen.integers(0, N_DEV_CAP, n)
    pi_i, pi_c = gen.permutation(N_DEV_IMG), gen.permutation(N_DEV_CAP)          # new id of old item = pi[old]
    inv_i, inv_c = np.argsort(pi_i), np.argsort(pi_c)                            # old item of new id
    q = gen.permutation(n); qb = np.argsort(q)
    J = ctx.joint if stud is not None else None
    for p in PEERS:
        h = copy.deepcopy(heads[p]); h.eval()
        gt = torch.Generator().manual_seed(SEED + PEERS.index(p))
        h.out.weight.copy_((torch.randn(h.out.weight.shape, generator=gt) * .1).to(h.out.weight))
        h.out.bias.copy_((torch.randn(h.out.bias.shape, generator=gt) * .1).to(h.out.bias))
        X, Y = ctx.d[('dev', 'img', p)], ctx.d[('dev', 'cap', p)]
        it, ct = torch.as_tensor(img, device=dev), torch.as_tensor(cap, device=dev)
        if J is not None:
            U = J.encode(stud[p], p, 'dev', 'img', np.arange(N_DEV_IMG)); V = J.encode(stud[p], p, 'dev', 'cap', np.arange(N_DEV_CAP))
            s1 = g_fn(h, None, ctx, p, 'dev', img, cap, stu=(U, it, V, ct))
        else:
            s1 = g_fn(h, None, ctx, p, 'dev', img, cap)
        Xr = torch.empty_like(X); Xr[torch.as_tensor(pi_i, device=X.device)] = X
        Yr = torch.empty_like(Y); Yr[torch.as_tensor(pi_c, device=Y.device)] = Y
        d2 = dict(ctx.d); d2[('dev', 'img', p)] = Xr; d2[('dev', 'cap', p)] = Yr
        ctx2 = SimpleNamespace(d=d2, dev=dev, raw=None)
        ir, cr = pi_i[img][q], pi_c[cap][q]
        if J is not None:
            Ur = J.encode(stud[p], p, 'dev', 'img', inv_i, chunk=97); Vr = J.encode(stud[p], p, 'dev', 'cap', inv_c, chunk=301)
            s2 = g_fn(h, None, ctx2, p, 'dev', ir, cr, chunk=333, stu=(Ur, torch.as_tensor(ir, device=dev), Vr, torch.as_tensor(cr, device=dev)))
        else:
            s2 = g_fn(h, None, ctx2, p, 'dev', ir, cr, chunk=333)
        s2 = s2[torch.as_tensor(qb, device=dev)]
        if J is not None:   # A3 E1: same student vectors moved by the rename (no re-encoding) -> pure scoring layout test
            Um = torch.empty_like(U); Um[torch.as_tensor(pi_i, device=dev)] = U; Vm = torch.empty_like(V); Vm[torch.as_tensor(pi_c, device=dev)] = V
            s3 = g_fn(h, None, ctx2, p, 'dev', ir, cr, chunk=333, stu=(Um, torch.as_tensor(ir, device=dev), Vm, torch.as_tensor(cr, device=dev)))[torch.as_tensor(qb, device=dev)]
        else:
            s3 = s2
        a, b = (float(t) for t in h.affine.ab())
        resid = s1.double() - (a * (X[it] * Y[ct]).sum(-1).double() + b)
        dl = (s1.double() - s2.double()).abs(); d1 = (s1.double() - s3.double()).abs()
        mr = float(resid.abs().max()); tol2 = TOL_SCORE * max(1.0, mr)
        out[p] = dict(n_pairs=n, max_abs=float(dl.max()), bitwise_frac=float((dl == 0).double().mean()), residual_std=float(resid.std()), max_abs_residual=mr,
                      max_abs_score=float(s1.abs().max()), E1_same_vectors_max_abs=float(d1.max()), E1_tol=TOL_SCORE, E2_reencoded_tol=tol2,
                      passed=bool(float(d1.max()) <= TOL_SCORE and float(dl.max()) <= tol2 and float(resid.std()) > MIN_RESIDUAL_STD))
    out['passed'] = bool(all(out[p]['passed'] for p in PEERS))
    return out


@torch.no_grad()
def joint_checks(ctx, stud, gen, final):
    """B / C / D / F (+ diagnostics at the final readout) for JOINT units."""
    J = ctx.joint; dev = J.dev
    rec = dict(B_single_vs_table={}, B_mixed_vs_table={}, D_renamed_ids={}, F_constant_raw={}, C_duplicates={})
    for p in PEERS:
        sp = stud[p]
        for kind, n in (('img', N_DEV_IMG), ('cap', N_DEV_CAP)):
            key = f'{kind}_{p}'
            full = J.encode(sp, p, 'dev', kind, np.arange(n))
            ids = np.sort(gen.choice(n, 32 if kind == 'img' else 96, replace=False))
            single = torch.cat([J.encode(sp, p, 'dev', kind, ids[i:i + 1]) for i in range(len(ids))])
            rec['B_single_vs_table'][key] = _cmp(single, full[torch.as_tensor(ids, device=dev)])
            oth = gen.choice(n, 4 * len(ids), replace=False); mix = np.concatenate([ids, oth]); perm = gen.permutation(len(mix)); inv = np.argsort(perm)
            em = J.encode(sp, p, 'dev', kind, mix[perm], chunk=37)[torch.as_tensor(inv[:len(ids)], device=dev)]
            rec['B_mixed_vs_table'][key] = _cmp(em, full[torch.as_tensor(ids, device=dev)])
            pi = gen.permutation(n); J2 = copy.copy(J)
            if kind == 'img':
                src = J.ims['dev']; st = torch.empty_like(src); st[torch.as_tensor(pi, device=src.device)] = src; J2.ims = dict(J.ims, dev=st)
            else:
                tk, ln = J.tok['dev'], J.len['dev']; pt = torch.as_tensor(pi, device=tk.device)
                t2 = torch.empty_like(tk); t2[pt] = tk; l2 = torch.empty_like(ln); l2[pt] = ln
                J2.tok = dict(J.tok, dev=t2); J2.len = dict(J.len, dev=l2)
            rec['D_renamed_ids'][key] = _cmp(J2.encode(sp, p, 'dev', kind, pi[ids]), full[torch.as_tensor(ids, device=dev)])
            cvec = J.raw_encode(sp, 'dev', kind, torch.as_tensor([int(ids[0])])).detach()
            J3 = copy.copy(J); J3.raw_encode = lambda stud_p, split, kind_, ids_t, _c=cvec: _c.expand(len(ids_t), -1).clone()
            u_dev = J3.encode(sp, p, 'dev', kind, np.arange(n))
            nfit = N_TRAIN_IMG if kind == 'img' else N_TRAIN_CAP
            u_fit = J3.encode(sp, p, 'fit', kind, np.sort(gen.choice(nfit, 512, replace=False)))
            rec['F_constant_raw'][key] = dict(dev_rows_differing_from_row0=int((u_dev != u_dev[:1]).any(1).sum()), fit_rows_differing_from_dev_row0=int((u_fit != u_dev[:1]).any(1).sum()))
            if final:
                r = J.raw_encode(sp, 'dev', kind, torch.as_tensor(np.sort(gen.choice(n, 256, replace=False))))
                cs = r @ r.T; m = ~torch.eye(len(r), dtype=torch.bool, device=r.device)
                mu = J.mu[('dev', kind, p)][:1]; uc = J.l2norm(r - mu)
                rec.setdefault('diag_final', {})[key] = dict(raw_mean_pairwise_cos=float(cs[m].double().mean()), centred_mean_pairwise_cos=float((uc @ uc.T)[m].double().mean()),
                                                              raw_total_var=float(r.double().var(0).sum()), raw_minus_fitmean_sq_norm_mean=float(((r - mu) ** 2).sum(1).double().mean()))
        # C: duplicates
        cr = {}
        if C.DEV_IMS_ROW_STRIDE > 1:
            S = C.DEV_IMS_ROW_STRIDE; dv = np.load(C.RAW_DEV_IMS, mmap_mode='r'); ks = np.sort(gen.choice(N_DEV_IMG, 6, replace=False))
            rows = np.sort(np.concatenate([S * ks + j for j in range(S)])); raw = np.ascontiguousarray(dv[rows])
            raw_same = bool(all(np.array_equal(raw[i], raw[(i // S) * S]) for i in range(len(rows))))
            u = J.l2norm(sp.enc_img(torch.from_numpy(raw).to(dev)) - J.mu[('dev', 'img', p)][:1])
            full = J.encode(sp, p, 'dev', 'img', np.arange(N_DEV_IMG))
            cr['dev_image_rows_same_raw'] = raw_same
            cr['dev_image_row_vectors_vs_image'] = _cmp(u, full[torch.as_tensor(rows // S, device=dev)])
        else:
            cr['dev_image_rows'] = 'not applicable (one dev_ims row per image)'
        fi = gen.choice(N_TRAIN_IMG, 3, replace=False); rep = np.array([fi[0], fi[0], fi[1], fi[0], fi[2], fi[1]])
        caps = gen.choice(N_TRAIN_CAP, len(rep), replace=False)
        with torch.enable_grad():
            U, iu, V, ic = J.pairs(sp, p, 'fit', rep, caps, grad=True)
        U = U.detach()
        cr['fit_pairs_dedup_rows'] = dict(unique_rows=int(U.shape[0]), unique_ids=int(len(np.unique(rep))), same_index_for_same_id=bool(len({(int(a), int(b)) for a, b in zip(rep, iu.tolist())}) == len(np.unique(rep))))
        e = J.encode(sp, p, 'fit', 'img', rep)
        cr['fit_raw_batch_repeated_items'] = _cmp(e[[0, 0, 2, 0, 4, 2]], e)
        rec['C_duplicates'][p] = cr
    okB = all(v['max_abs'] <= TOL_VEC for part in ('B_single_vs_table', 'B_mixed_vs_table', 'D_renamed_ids') for v in rec[part].values())
    okF = all(v['dev_rows_differing_from_row0'] == 0 and v['fit_rows_differing_from_dev_row0'] == 0 for v in rec['F_constant_raw'].values())
    okC = True
    for p in PEERS:
        c = rec['C_duplicates'][p]
        okC &= c['fit_pairs_dedup_rows']['unique_rows'] == c['fit_pairs_dedup_rows']['unique_ids'] and c['fit_pairs_dedup_rows']['same_index_for_same_id']
        okC &= c['fit_raw_batch_repeated_items']['max_abs'] <= TOL_VEC
        if 'dev_image_rows_same_raw' in c and c['dev_image_rows_same_raw']:
            okC &= c['dev_image_row_vectors_vs_image']['max_abs'] <= TOL_VEC
    rec['passed_B_D'] = bool(okB); rec['passed_F'] = bool(okF); rec['passed_C'] = bool(okC)
    return rec


def layout_checks(ctx, heads, stud, cfg, out, step, g_fn):
    final = step == cfg['steps']
    gen = np.random.default_rng([SEED, int(step)])
    dig0 = _digest(ctx, heads, stud)
    devs = [torch.cuda.current_device()] if ctx.dev.type == 'cuda' else []
    with torch.random.fork_rng(devices=devs):
        rec = dict(step=int(step), unit_id=cfg['unit_id'], protocol_id=PROTOCOL_ID, tol_vec=TOL_VEC, tol_score=TOL_SCORE, utc=utc())
        rec['P'] = protocol_check(ctx)
        rec['G'] = truth_check(ctx, gen)
        rec['E'] = head_checks(ctx, heads, stud, gen, g_fn)
        if stud is not None:
            rec['joint'] = joint_checks(ctx, stud, gen, final)
    rec['state_unchanged'] = _digest(ctx, heads, stud) == dig0
    parts = dict(P=rec['P']['passed'], G=rec['G']['passed'], E=rec['E']['passed'], state=rec['state_unchanged'])
    if stud is not None:
        parts.update(B_D=rec['joint']['passed_B_D'], C=rec['joint']['passed_C'], F=rec['joint']['passed_F'])
    rec['parts'] = parts; rec['passed'] = bool(all(parts.values()))
    write_json(out / f'R50_LAYOUT_CHECK_step{step}.json', rec)
    assert rec['passed'], f'R50 layout check failed at step {step}: {parts}'
    return dict(passed=rec['passed'], parts=parts)
