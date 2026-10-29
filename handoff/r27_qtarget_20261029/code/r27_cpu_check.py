"""R27 CPU gate (DESIGN_CN.md §6.1). Writes CPU_CHECK.json. Usage: r27_cpu_check.py OUT.json [CELLS.json]

 1 vcs_vs_reference   r27 exact VCS solver vs the package NumPy/SciPy reference quadratic_ball_refiner (math_checks.py,
                      function source extracted verbatim with ast, the module itself is NOT executed): z, objective, nu,
                      KKT residual; dense Dirichlet A (reference setting) and K-sparse graph A, interior + boundary cases
 2 vcs_vs_generic     same problems vs scipy SLSQP and vs the independent FISTA solver from z0 = 0
 3 matvec_identity    H_i z = Y^T[(A_i+c)(Y z)] vs explicit sum_j (A_ij+c_j) y_j y_j^T z
 4 objective_identity sum_i r_i f_gen(x_i) == R26 frozen j_graph / -bce_graph / -snce_graph; moment form
 5 grad_at_zero       all generators: grad f(0) proportional to delta (1, 1/2, 1/tau)
 6 bce_snce_vs_generic production SPG vs scipy SLSQP (small d) and vs FISTA (moderate d): objective, angle, KKT
 7 id_buffer          native MemoryQueue.enqueue (author code, CPU tensors) + r27 ID mirror across wrap-arounds;
                      ids_state -> torch.save -> torch.load -> load_ids roundtrip; pre-ID load -> all -1; search never
                      returns a pre-ID slot and returns None when fewer than K ID-bearing slots exist
 8 stop_grad          generate() rejects grad inputs; targets carry no grad; aux-loss grad reaches only the query
 9 prefix_queues      ptr / is_full / queue size of the six Flickr common prefixes (expected pre-ID exclusion)
"""
import sys, os, ast, json, math
import numpy as np
import torch
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
REF_PY = '/home/infres/yinwang/CS_QMI/r27_review_design/VCS_R27_Review_and_Design_20261029/math_checks.py'
import r27_targets as T
import r27_interface as I
from r26_graph import coupling, j_graph, bce_graph, snce_graph, moment_j

res = {}


def ok(name, cond, **info):
    res[name] = dict(pass_=bool(cond), **info)
    print(name, 'PASS' if cond else 'FAIL', json.dumps(info, default=float)[:3000], flush=True)


def load_reference():
    src = open(REF_PY).read()
    tree = ast.parse(src)
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'quadratic_ball_refiner'][0]
    ns = {'np': np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), REF_PY, 'exec'), ns)
    return ns['quadratic_ball_refiner']


REF = load_reference()
DT = torch.float64


def unit(x):
    return x / x.norm(dim=-1, keepdim=True)


def problem(kind, B, M, d, K=5, seed=0, scale_bank=1.5):
    g = torch.Generator().manual_seed(seed)
    if kind == 'dense':
        Y = unit(torch.randn(M, d, generator=g, dtype=DT))
        A = torch.distributions.Dirichlet(torch.ones(M, dtype=DT)).sample((B,))
        return A, Y
    base = torch.randn(8, d, generator=g, dtype=DT)
    NQ = M
    bank = unit(base[torch.randint(0, 8, (NQ,), generator=g)] + scale_bank * torch.randn(NQ, d, generator=g, dtype=DT))
    q = unit(base[torch.randint(0, 8, (B,), generator=g)] + scale_bank * torch.randn(B, d, generator=g, dtype=DT))
    idx = torch.topk(q @ bank.t(), K, 1).indices
    if kind == 'dup':  # degenerate rows: every slot the same atom (t2i: five captions of one image)
        idx[: B // 3] = idx[: B // 3, :1]
    cols, inv = torch.unique(idx.flatten(), return_inverse=True)
    inv = inv.view_as(idx)
    A = torch.zeros(B, cols.numel(), dtype=DT)
    A.scatter_add_(1, inv, torch.full_like(inv, 1.0 / K, dtype=DT))
    return A, bank[cols]


def slsqp(f_np, g_np, d, start):
    r = minimize(lambda v: -f_np(v), start, jac=lambda v: -g_np(v), method='SLSQP',
                 constraints={'type': 'ineq', 'fun': lambda v: 1 - v @ v, 'jac': lambda v: -2 * v},
                 options={'ftol': 1e-14, 'maxiter': 3000})
    return r


def angle(a, b):
    return float(torch.arccos(((a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1))).clamp(-1, 1)).max())


# ----------------------------------------------------------------------------- 1-3 VCS

def check_vcs():
    rows, worst = [], dict(z=0, f=0, kkt=0, nu=0, generic_f=0, fista_z=0)
    n_int = n_bnd = 0
    for kind, B, M, d, seed in (('dense', 7, 16, 6, 1), ('dense', 9, 24, 6, 2), ('dense', 6, 30, 12, 3), ('graph', 16, 60, 6, 4),
                                ('graph', 32, 200, 64, 5), ('dup', 24, 120, 32, 6), ('graph', 32, 400, 256, 7)):
        A, Y = problem(kind, B, M, d, seed=seed)
        c, mu, muQ, delta = T.moments(A, Y)
        Z, info = T.solve_vcs(A, Y)
        Zf, inf_f = T.solve_fista('vcs', A, Y, tol_rel=1e-10, cap=20000)
        for i in range(B):
            H = (Y.t() * (A[i] + c)) @ Y
            zr, nur = REF(delta[i].numpy(), H.numpy())
            zr = torch.from_numpy(zr)
            fv = lambda v: float(delta[i] @ v - 0.5 * v @ H @ v)
            worst['z'] = max(worst['z'], float((Z[i] - zr).abs().max()))
            worst['f'] = max(worst['f'], abs(fv(Z[i]) - fv(zr)))
            worst['nu'] = max(worst['nu'], abs(float(info['nu'][i]) - nur))
            worst['kkt'] = max(worst['kkt'], float(((H + float(info['nu'][i]) * torch.eye(d, dtype=DT)) @ Z[i] - delta[i]).norm()))
            if d <= 64:
                Hn, dn = H.numpy(), delta[i].numpy()
                st = unit(mu[i:i + 1])[0].numpy()
                r = slsqp(lambda v: dn @ v - 0.5 * v @ Hn @ v, lambda v: dn - Hn @ v, d, st)
                worst['generic_f'] = max(worst['generic_f'], fv(torch.from_numpy(r.x)) - fv(Z[i]))  # >0 would mean generic beat us
            worst['fista_z'] = max(worst['fista_z'], float((Zf[i] - Z[i]).norm()))
            n_int += int(nur == 0.0); n_bnd += int(nur > 0)
        # matvec identity
        Zr = torch.randn(B, d, dtype=DT)
        Hz = T.hmatvec(Zr, A, c, Y)
        Hz_ref = torch.stack([((Y.t() * (A[i] + c)) @ Y) @ Zr[i] for i in range(B)])
        rows.append(dict(kind=kind, B=B, M=M, d=d, rank=info['rank'], max_newton=int(info['iters'].max()),
                         interior=int(info['interior'].sum()), max_rel_kkt=float(info['rel'].max()),
                         matvec_err=float((Hz - Hz_ref).abs().max()), max_range_loss=float(info['lost'].max())))
    good = (worst['z'] < 1e-7 and worst['f'] < 1e-10 and worst['kkt'] < 1e-8 and worst['nu'] < 1e-6
            and worst['generic_f'] < 1e-9 and worst['fista_z'] < 1e-4 and n_int > 0 and n_bnd > 0
            and all(r['matvec_err'] < 1e-12 and r['max_rel_kkt'] < 1e-9 for r in rows))
    ok('vcs_vs_reference', worst['z'] < 1e-7 and worst['f'] < 1e-10 and worst['kkt'] < 1e-8 and worst['nu'] < 1e-6 and n_int > 0 and n_bnd > 0,
       max_abs_z_diff=worst['z'], max_obj_diff=worst['f'], max_kkt_residual=worst['kkt'], max_nu_diff=worst['nu'],
       n_interior_rows=n_int, n_boundary_rows=n_bnd, problems=rows)
    ok('vcs_vs_generic', worst['generic_f'] < 1e-9 and worst['fista_z'] < 1e-4,
       slsqp_minus_exact_obj_max=worst['generic_f'], fista_vs_exact_max_z_dist=worst['fista_z'])
    ok('matvec_identity', all(r['matvec_err'] < 1e-12 for r in rows), max_err=max(r['matvec_err'] for r in rows))
    return good


# ----------------------------------------------------------------------------- 4-5 identities

def check_identities():
    A, Y = problem('graph', 16, 60, 12, seed=11)
    B = A.shape[0]
    X = unit(torch.randn(B, 12, dtype=DT, generator=torch.Generator().manual_seed(12))) * 0.9
    c = A.mean(0)
    W, c2, Q = coupling(A)
    Fm = X @ Y.t()
    e = dict(
        vcs=abs(float(T.f_vcs(X, A, c, Y).mean() - j_graph(W, Q, Fm))),
        vcs_moment=abs(float(T.f_vcs(X, A, c, Y).mean() - moment_j(X, A, Y))),
        bce=abs(float(T.f_bce(X, A, c, Y).mean() + bce_graph(W, Q, Fm))),
        snce=abs(float(T.f_snce(X, A, c, Y).mean() + snce_graph(W, c2, Fm))))
    ok('objective_identity', all(v < 1e-12 for v in e.values()), **e)
    Z0 = torch.zeros(B, 12, dtype=DT)
    _, _, _, delta = T.moments(A, Y)
    gz = {k: float((T.OBJ[k][1](Z0, A, c, Y) - s * delta).abs().max()) for k, s in (('vcs', 1.0), ('bce', 0.5), ('snce', 1 / T.TAU_SNCE))}
    ok('grad_at_zero', all(v < 1e-10 for v in gz.values()), **gz)
    # finite-difference check of the analytic gradients at an interior point
    fd = {}
    for k in ('vcs', 'bce', 'snce'):
        f, g = T.OBJ[k]
        Xp = X[:1] * 0.5
        gv = g(Xp, A[:1], c, Y)[0]
        num = torch.zeros(12, dtype=DT)
        for l in range(12):
            e1 = torch.zeros_like(Xp); e1[0, l] = 1e-6
            num[l] = (f(Xp + e1, A[:1], c, Y) - f(Xp - e1, A[:1], c, Y))[0] / 2e-6
        fd[k] = float((num - gv).abs().max())
    ok('gradient_finite_difference', all(v < 1e-6 for v in fd.values()), **fd)


# ----------------------------------------------------------------------------- 6 BCE / sNCE

def check_bce_snce():
    out = {}
    good = True
    for gen in ('bce', 'snce'):
        # certified bound for a concave max on the ball: f* - f(z) <= 2 * KKT residual(z); we test gap - 2 res <= 1e-10
        w = dict(generic_excess=-1e9, generic_gap=-1e9, generic_angle=0, fista_excess=-1e9, fista_gap=-1e9, fista_angle=0,
                 max_rel_kkt=0, max_iters=0, cap=0)
        for kind, B, M, d, seed in (('graph', 12, 40, 6, 21), ('dense', 8, 20, 6, 22), ('dup', 12, 60, 16, 23), ('graph', 32, 300, 128, 24)):
            A, Y = problem(kind, B, M, d, seed=seed)
            c, mu, muQ, delta = T.moments(A, Y)
            f, g = T.OBJ[gen]
            Z, info = T.solve_spg(gen, A, Y)
            w['max_rel_kkt'] = max(w['max_rel_kkt'], float(info['rel'].max()))
            w['max_iters'] = max(w['max_iters'], int(info['iters'].max())); w['cap'] += int(info['cap'].sum())
            if d <= 16:
                for i in range(B):
                    fi = lambda v: float(f(torch.from_numpy(v).unsqueeze(0), A[i:i + 1], c, Y)[0])
                    gi = lambda v: g(torch.from_numpy(v).unsqueeze(0), A[i:i + 1], c, Y)[0].numpy()
                    best = None
                    for st in (np.zeros(d), 0.5 * unit(mu[i:i + 1])[0].numpy(), 0.9 * Z[i].numpy()):
                        r = slsqp(lambda v: fi(v) if np.isfinite(fi(v)) else -1e30, gi, d, st)
                        if np.isfinite(fi(r.x)) and r.x @ r.x <= 1 + 1e-9 and (best is None or fi(r.x) > fi(best)):
                            best = r.x
                    gap = fi(best) - float(f(Z[i:i + 1], A[i:i + 1], c, Y)[0])
                    w['generic_gap'] = max(w['generic_gap'], gap)
                    w['generic_excess'] = max(w['generic_excess'], gap - 2 * float(info['res'][i]))
                    w['generic_angle'] = max(w['generic_angle'], angle(Z[i:i + 1], torch.from_numpy(best).unsqueeze(0)))
            Zf, inf_f = T.solve_fista(gen, A, Y, tol_rel=1e-9, cap=40000)
            gapf = f(Zf, A, c, Y) - f(Z, A, c, Y)
            w['fista_gap'] = max(w['fista_gap'], float(gapf.max()))
            w['fista_excess'] = max(w['fista_excess'], float((gapf - 2 * info['res']).max()))
            w['fista_angle'] = max(w['fista_angle'], angle(Z, Zf))
        out[gen] = w
        good &= w['generic_excess'] <= 1e-10 and w['fista_excess'] <= 1e-10 and w['max_rel_kkt'] <= T.TOL_REL and w['cap'] == 0
    ok('bce_snce_vs_generic', good, note='gap = generic/FISTA objective minus production SPG objective (>0: other solver better); '
       'pass requires gap <= 2*KKT residual (certified suboptimality bound). Angles reported: flat objectives leave '
       'directional slack at rel KKT 1e-6', **out)


# ----------------------------------------------------------------------------- 7 ID buffer

def check_ids():
    import co_train
    MQ = co_train.MemoryQueue
    Qs, d = 50, 3
    q = MQ.__new__(MQ)
    q.queue_size, q.dim, q.ptr, q.is_full = Qs, d, 0, False
    q.img_queue = torch.zeros(Qs, d); q.txt_queue = torch.zeros(Qs, d)
    g = np.random.default_rng(0)
    nid = 0
    bad = 0
    for step in range(40):
        n = int(g.integers(1, 23))
        cap = np.arange(nid, nid + n); nid += n
        img = cap * 7 + 3
        feats = torch.from_numpy(np.stack([cap, img, np.full(n, step)], 1)).float()
        I.enqueue_with_ids(q, feats, feats + 0.5, cap, img, 3, step, 1)
        nfill = Qs if q.is_full else q.ptr
        ids = q.r27_ids
        bad += int((ids['cap'][:nfill].float() != q.img_queue[:nfill, 0]).sum())
        bad += int((ids['img'][:nfill].float() != q.img_queue[:nfill, 1]).sum())
        bad += int((ids['witer'][:nfill].float() != q.img_queue[:nfill, 2]).sum())
    st = I.ids_state(q)
    p = os.path.join(os.environ.get('TMPDIR', '/tmp'), f'r27_ids_{os.getpid()}.pt')
    torch.save({'r27_queue_ids': st}, p)
    st2 = torch.load(p, weights_only=False)['r27_queue_ids']; os.remove(p)
    q2 = MQ.__new__(MQ); q2.queue_size, q2.ptr, q2.is_full = Qs, q.ptr, q.is_full
    q2.img_queue, q2.txt_queue = q.img_queue.clone(), q.txt_queue.clone()
    I.load_ids(q2, st2)
    rt = all(torch.equal(q.r27_ids[k], q2.r27_ids[k]) for k in I.KEYS)
    q3 = MQ.__new__(MQ); q3.queue_size, q3.ptr, q3.is_full = Qs, 10, True
    q3.img_queue, q3.txt_queue = q.img_queue.clone(), q.txt_queue.clone()
    I.load_ids(q3, None)
    pre = bool((q3.r27_ids['cap'] == -1).all())
    # search exclusion: mark half of the slots pre-ID
    q2.r27_ids['cap'][::2] = -1
    q2.img_queue.copy_(unit(torch.randn(Qs, d))); q2.txt_queue.copy_(unit(torch.randn(Qs, d)))
    S, info = I.search(q2, unit(torch.randn(6, d)), 'i2t', 5)
    excl = S is not None and bool((S['cap'] >= 0).all()) and bool((S['slots'] % 2 == 1).all())
    q2.r27_ids['cap'][1::2][4:] = -1
    S2, info2 = I.search(q2, unit(torch.randn(6, d)), 'i2t', 5)
    ok('id_buffer', bad == 0 and rt and pre and excl and S2 is None, misaligned=bad, wraps=int(nid // Qs), roundtrip=rt,
       pre_id_all_minus1=pre, search_excludes_pre_id=excl, search_none_when_lt_k=S2 is None, usable_after=info2)


# ----------------------------------------------------------------------------- 8 stop-grad

def check_stopgrad():
    A, Y = problem('graph', 10, 40, 8, seed=31)
    Yp = Y.clone().requires_grad_(True)
    raised = False
    try:
        T.generate('vcs', A, Yp)
    except AssertionError:
        raised = True
    X = unit(torch.randn(10, 8, dtype=DT)).requires_grad_(True)
    info = {}
    good = raised
    for gen in T.GENERATORS:
        Ys = unit(Yp).detach()
        tg, valid, _ = T.generate(gen, A, Ys)
        l = (X @ tg.t()).logsumexp(1).mean() - (X * tg).sum(1).mean()
        gX, gY = torch.autograd.grad(l, [X, Yp], allow_unused=True)
        info[gen] = dict(target_requires_grad=bool(tg.requires_grad), x_grad=gX is not None and float(gX.abs().sum()) > 0, y_grad_none=gY is None,
                         unit=float((tg.norm(dim=1) - 1).abs().max()), valid=int(valid.sum()))
        good &= (not tg.requires_grad) and info[gen]['x_grad'] and info[gen]['y_grad_none'] and info[gen]['unit'] < 1e-12
    ok('stop_grad', good, generate_rejects_grad_inputs=raised, **info)


# ----------------------------------------------------------------------------- 9 prefixes

def check_prefixes(cells_path):
    cells = json.load(open(cells_path))
    out = {}
    for name, cfg in cells.items():
        st = torch.load(cfg['prefix'], map_location='cpu', weights_only=False)
        out[name] = {k: dict(ptr=int(st[k]['queue']['ptr']), is_full=bool(st[k]['queue']['is_full']), size=int(st[k]['queue']['img'].shape[0]),
                             has_ids='r27_queue_ids' in st[k]) for k in ('model_A', 'model_B')}
        out[name]['epoch'] = st['epoch']
        del st
    ok('prefix_queues', all(not v[k]['has_ids'] for v in out.values() for k in ('model_A', 'model_B')),
       note='all prefix slots are pre-ID: peer A has no usable candidates in co-train epoch 1 (B queue untouched since prefix)', **out)


if __name__ == '__main__':
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 2)))
    out_path = sys.argv[1]
    check_vcs()
    check_identities()
    check_bce_snce()
    check_ids()
    check_stopgrad()
    if len(sys.argv) > 2:
        check_prefixes(sys.argv[2])
    res['all_pass'] = all(v['pass_'] for v in res.values() if isinstance(v, dict))
    json.dump(res, open(out_path, 'w'), indent=1, default=float)
    print('ALL_PASS' if res['all_pass'] else 'SOME_FAIL')
    sys.exit(0 if res['all_pass'] else 1)
