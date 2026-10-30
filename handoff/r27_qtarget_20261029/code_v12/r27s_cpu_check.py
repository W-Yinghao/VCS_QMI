"""R27-S CPU gate (DESIGN_S_CN.md §5.1). Usage: r27s_cpu_check.py OUT.json. Technical checks only.

 1 cap_solvers      generate_cap (S3 closed form; S4/S5/S6 SPG in exact span(Y,p) coordinates) vs the independent
                    full-dimensional reference r27_audit_20261029/audit_constrained.solve_constrained (FISTA + same exact
                    projection, tight tol) and vs SciPy SLSQP (small d); p deliberately outside span(Y). Pass: feasibility
                    <= 1e-9, rel KKT <= 1e-6, no cap hits, certified gap (reference - ours) <= 2*res, normalized target
                    angle to p <= 15 deg.
 2 augmented_basis  p reconstructed exactly from (Q, e, p~); objective/grad in reduced coords == full coords.
 3 a1_identity      split loss 0.5 SCE(sgX, P) + 0.5 SCE(X, sgP) vs native 0.5 SCE(X, P) with the AUTHOR GraphRefiner and the
                    AUTHOR vse.infonce (its .cuda() mask call is made a no-op on CPU only here): encoder and refiner
                    gradients, one AdamW step, optimizer state and RNG state after the step.
"""
import sys, os, json, math
import numpy as np
import torch
from scipy.optimize import minimize

sys.path.insert(1, '/home/infres/yinwang/CS_QMI/handoff/r27_audit_20261029')
import r27s_targets as S
import r27_targets as T
import audit_constrained as AC
from r27_cpu_check import problem

res = {}
DT = torch.float64


def ok(name, cond, **info):
    res[name] = dict(pass_=bool(cond), **info)
    print(name, 'PASS' if cond else 'FAIL', json.dumps(info, default=float)[:3000], flush=True)


def make_p(A, Y, seed, out=0.6):
    g = torch.Generator().manual_seed(seed)
    mu = A @ Y
    return torch.nn.functional.normalize(mu + out * torch.randn(mu.shape, generator=g, dtype=DT), dim=1)


def check_solvers():
    rows, good = [], True
    for kind, B, M, d, seed in (('graph', 10, 40, 6, 1), ('dense', 6, 12, 8, 2), ('graph', 16, 120, 32, 3), ('dup', 12, 60, 16, 4),
                                ('graph', 24, 300, 128, 5)):
        A, Y = problem(kind, B, M, d, seed=seed)
        P = make_p(A, Y, seed)
        c, mu, muQ, delta = T.moments(A, Y)
        for k in ('cap_delta', 'cap_vcs', 'cap_bce', 'cap_snce'):
            Tg, valid, info = S.generate_cap(k, A, Y, P)
            Z = Tg * info['z_norm'].unsqueeze(1)
            gen = 'vcs' if k == 'cap_delta' else S.SOLVERS[k]
            f = (lambda Z_: (delta * Z_).sum(1)) if k == 'cap_delta' else (lambda Z_, g_=gen: T.fT(g_, Z_ @ Y.t(), A, c))
            r = dict(kind=kind, d=d, M=int(A.shape[1]), solver=k, max_rel=float(info['rel'].max()), caps=int(info['cap'].sum()),
                     max_viol=float(info['violation'].max()), fallback=int(info['fallback'].sum()),
                     max_angle_deg=float(torch.rad2deg(torch.arccos(info['cos_to_p'].clamp(-1, 1))).max()),
                     cap_active=int(info['cap_active'].sum()), ball_active=int(info['ball_active'].sum()),
                     max_iters=int(info['iters'].max()))
            if k != 'cap_delta':
                Zr, ir = AC.solve_constrained(gen, A, Y, P, S.GAMMA, tol_rel=1e-10, cap=40000)
                gap = f(Zr) - f(Z)
                r['ref_gap_max'] = float(gap.max()); r['ref_excess'] = float((gap - 2 * info['res']).max())
                r['ref_angle_max'] = float(torch.arccos(((Zr * Z).sum(1) / (Zr.norm(dim=1) * Z.norm(dim=1))).clamp(-1, 1)).max())
                good &= r['ref_excess'] <= 1e-10
            if d <= 16:
                ex = -1e9
                for i in range(A.shape[0]):
                    p = P[i].numpy()
                    fi = lambda v: float(f(torch.from_numpy(v).unsqueeze(0).to(DT) if True else None)[0]) if k == 'cap_delta' else \
                        float(T.fT(gen, torch.from_numpy(v).unsqueeze(0) @ Y.t(), A[i:i + 1], c)[0])
                    if k == 'cap_delta':
                        fi = lambda v: float(delta[i].numpy() @ v)
                    best = None
                    for st in ((1 + S.GAMMA) / 2 * p, 0.99 * Z[i].numpy()):
                        rr = minimize(lambda v: -fi(v) if np.isfinite(fi(v)) else 1e30, st, method='SLSQP',
                                      constraints=[{'type': 'ineq', 'fun': lambda v: 1 - v @ v}, {'type': 'ineq', 'fun': lambda v, p=p: p @ v - S.GAMMA}],
                                      options={'ftol': 1e-14, 'maxiter': 3000})
                        x = rr.x
                        if x @ x <= 1 + 1e-9 and p @ x >= S.GAMMA - 1e-9 and np.isfinite(fi(x)) and (best is None or fi(x) > fi(best)):
                            best = x
                    zi = Z[i].numpy()
                    ex = max(ex, fi(best) - fi(zi) - 2 * float(info['res'][i]) - 1e-9)   # SLSQP itself allows ~1e-9 slack
                r['slsqp_excess'] = ex
                good &= ex <= 1e-8
            good &= r['max_rel'] <= T.TOL_REL and r['caps'] == 0 and r['max_viol'] <= 1e-9 and r['fallback'] == 0 and r['max_angle_deg'] <= 15 + 1e-6
            rows.append(r)
    ok('cap_solvers', good, rows=rows)


def check_basis():
    A, Y = problem('graph', 12, 40, 64, seed=7)                 # M < d: span(Y) is a strict subspace, p leaves it
    P = make_p(A, Y, 7)
    Qb, e, Pt = S.augmented_basis(Y, P)
    Prec = S.to_full(Pt, Qb, e)
    z = torch.nn.functional.normalize(torch.randn(12, Pt.shape[1], dtype=DT), dim=1) * 0.7
    Zf = S.to_full(z, Qb, e)
    Ya = torch.cat([Y @ Qb, torch.zeros(Y.shape[0], 1, dtype=DT)], 1)
    c = A.mean(0)
    e1 = float((Prec - P).abs().max()); e2 = float((Zf @ Y.t() - z @ Ya.t()).abs().max())
    e3 = float(((Zf * P).sum(1) - (z * Pt).sum(1)).abs().max()); e4 = float((Zf.norm(dim=1) - z.norm(dim=1)).abs().max())
    outside = float((P - (P @ Qb) @ Qb.t()).norm(dim=1).min())
    ok('augmented_basis', max(e1, e2, e3, e4) < 1e-12 and outside > 1e-3, p_recon=e1, Yz=e2, pz=e3, norm=e4, min_p_outside_spanY=outside)


def check_identity():
    torch.Tensor.cuda = lambda self, *a, **k: self      # CPU only: author infonce/MemoryQueue call .cuda()
    import co_train
    from vse import infonce
    torch.manual_seed(0)
    B, K, d, din = 16, 5, 64, 32
    enc = torch.nn.Sequential(torch.nn.Linear(din, d), torch.nn.Tanh(), torch.nn.Linear(d, d)).double()
    ref = co_train.GraphRefiner(dim=d, k=K).double(); ref.train()
    xin = torch.randn(B, din, dtype=DT)
    neigh = torch.nn.functional.normalize(torch.randn(B, K, d, dtype=DT), dim=-1)
    out = {}
    for mode in ('native', 'split'):
        torch.manual_seed(1)
        e2 = torch.nn.Sequential(torch.nn.Linear(din, d), torch.nn.Tanh(), torch.nn.Linear(d, d)).double(); e2.load_state_dict(enc.state_dict())
        r2 = co_train.GraphRefiner(dim=d, k=K).double(); r2.load_state_dict(ref.state_dict()); r2.train()
        optm = torch.optim.AdamW(list(e2.parameters()) + list(r2.parameters()), lr=5e-4, weight_decay=1e-4)
        torch.manual_seed(2)
        X = torch.nn.functional.normalize(e2(xin), dim=-1)
        P = r2(neigh)                                             # one refiner forward (dropout)
        if mode == 'native':
            loss = 0.5 * infonce(X.mm(P.t()))
        else:
            loss = 0.5 * infonce(X.detach().mm(P.t())) + 0.5 * infonce(X.mm(P.detach().t()))
        optm.zero_grad(); loss.backward()
        out[mode] = dict(ge=[p.grad.clone() for p in e2.parameters()], gr=[p.grad.clone() for p in r2.parameters() if p.grad is not None],
                         nr=sum(1 for p in r2.parameters() if p.grad is not None))
        torch.nn.utils.clip_grad_norm_(list(e2.parameters()), 2.0)
        optm.step()
        out[mode]['params'] = torch.cat([p.detach().flatten() for p in list(e2.parameters()) + list(r2.parameters())])
        out[mode]['state'] = torch.cat([v.flatten() for st in optm.state.values() for kk, v in st.items() if torch.is_tensor(v) and v.dim() > 0])
        out[mode]['rng'] = torch.get_rng_state()
    a, b = out['native'], out['split']
    de = max(float((x - y).abs().max()) for x, y in zip(a['ge'], b['ge']))
    dr = max(float((x - y).abs().max()) for x, y in zip(a['gr'], b['gr']))
    dp = float((a['params'] - b['params']).abs().max()); ds = float((a['state'] - b['state']).abs().max())
    rng = bool(torch.equal(a['rng'], b['rng']))
    ok('a1_identity', de < 1e-12 and dr < 1e-12 and a['nr'] == b['nr'] and dp < 1e-12 and ds < 1e-12 and rng,
       enc_grad_maxdiff=de, refiner_grad_maxdiff=dr, refiner_params_with_grad=[a['nr'], b['nr']], param_after_step_maxdiff=dp,
       adam_state_maxdiff=ds, rng_equal=rng, note='author GraphRefiner + author vse.infonce, float64 CPU, train-mode dropout')


if __name__ == '__main__':
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 2)))
    check_basis()
    check_identity()
    check_solvers()
    res['all_pass'] = all(v['pass_'] for v in res.values() if isinstance(v, dict))
    json.dump(res, open(sys.argv[1], 'w'), indent=1, default=float)
    print('ALL_PASS' if res['all_pass'] else 'SOME_FAIL')
    sys.exit(0 if res['all_pass'] else 1)
