"""R27 audit CPU check 2 (synthetic only, no data read). Usage: audit_check2.py OUT.json

 1 projection      proj_ball_half vs a generic solver (SLSQP, min ||y-x||^2 s.t. ||y|| <= 1, p.y >= c) on random points
                   covering all four KKT cases; feasibility; idempotence; variational inequality
                   (x - P(x)).(y - P(x)) <= 0 for random feasible y
 2 constrained     solve_constrained for vcs / bce / snce, theta in {15, 30, 45} deg, p = random unit and p = normalized mean:
                   feasible, rel <= 1e-6 (production-like tol), no cap; a 1e-9 solve has objective >= SLSQP (multi-start) - 1e-8; when the unconstrained optimum is
                   feasible the constrained solution equals it (angle)
 3 task_scores     rank / margin vs a brute-force loop on random targets, pools and multi-ID truth sets
 4 table1          transitions() on a hand-built case with known counts
"""
import sys, json, math, time
import numpy as np
import torch
from scipy.optimize import minimize

import audit_common as C
from audit_common import T
import audit_constrained as K
import table1_transitions as T1

res = {}
DT = torch.float64


def ok(name, cond, **info):
    res[name] = dict(pass_=bool(cond), **info)
    print(name, 'PASS' if cond else 'FAIL', json.dumps(info, default=float)[:2500], flush=True)


def check_projection():
    g = torch.Generator().manual_seed(1)
    d = 7
    worst = dict(feas=0.0, idem=0.0, vi=-1.0, generic_excess=-1.0)
    cases = dict(inside=0, half=0, ball=0, corner=0)
    for trial in range(400):
        p = C.unit(torch.randn(1, d, generator=g, dtype=DT))
        c = float(torch.rand(1, generator=g)) * 0.95
        x = torch.randn(1, d, generator=g, dtype=DT) * float(torch.rand(1, generator=g) * 3)
        if trial % 4 == 1:
            x = x + 2.0 * p
        if trial % 4 == 2:
            x = x - 1.5 * p
        y = K.proj_ball_half(x, p, c)
        nrm, pz = float(y.norm()), float((y * p).sum())
        worst['feas'] = max(worst['feas'], nrm - 1, c - pz)
        worst['idem'] = max(worst['idem'], float((K.proj_ball_half(y, p, c) - y).norm()))
        if float((x - y).norm()) < 1e-14:
            cases['inside'] += 1
        elif abs(nrm - 1) < 1e-9 and abs(pz - c) < 1e-9:
            cases['corner'] += 1
        elif abs(nrm - 1) < 1e-9:
            cases['ball'] += 1
        else:
            cases['half'] += 1
        Yf = torch.randn(200, d, generator=g, dtype=DT) * 1.2
        feas = (Yf.norm(dim=1) <= 1) & ((Yf * p).sum(1) >= c)
        if bool(feas.any()):
            worst['vi'] = max(worst['vi'], float(((x - y) * (Yf[feas] - y)).sum(1).max()))
        xn, pn = x[0].numpy(), p[0].numpy()
        best = None
        for st in (np.zeros(d), y[0].numpy() * 0.5, c * pn):
            r = minimize(lambda v: ((v - xn) ** 2).sum(), st, jac=lambda v: 2 * (v - xn), method='SLSQP',
                         constraints=[{'type': 'ineq', 'fun': lambda v: 1 - v @ v, 'jac': lambda v: -2 * v},
                                      {'type': 'ineq', 'fun': lambda v: v @ pn - c, 'jac': lambda v: pn}],
                         options={'ftol': 1e-15, 'maxiter': 500})
            if r.x @ r.x <= 1 + 1e-9 and r.x @ pn >= c - 1e-9:
                dd = float(((r.x - xn) ** 2).sum())
                best = dd if best is None else min(best, dd)
        if best is not None:
            worst['generic_excess'] = max(worst['generic_excess'], float(((y[0].numpy() - xn) ** 2).sum()) - best)
    good = worst['feas'] <= 1e-12 and worst['idem'] <= 1e-12 and worst['vi'] <= 1e-10 and worst['generic_excess'] <= 1e-9 and min(cases.values()) > 0
    ok('projection', good, cases=cases, **worst)


def check_constrained():
    rows, good = [], True
    for kind, B, M, d, seed in (('graph', 12, 60, 10, 21), ('dup', 12, 60, 10, 22), ('graph', 24, 120, 32, 23)):
        A, Y = C.problem(kind, B, M, d, seed=seed)
        c, mu, muQ, delta = T.moments(A, Y)
        g = torch.Generator().manual_seed(seed)
        for pk in ('random', 'mean'):
            P = C.unit(torch.randn(B, d, generator=g, dtype=DT)) if pk == 'random' else C.unit(mu)
            for gen in ('vcs', 'bce', 'snce'):
                f, gr = T.OBJ[gen]
                Zu = T.solve_vcs(A, Y)[0] if gen == 'vcs' else T.solve_fista(gen, A, Y, tol_rel=1e-10, cap=60000)[0]
                for th in (15, 30, 45):
                    cth = math.cos(math.radians(th))
                    Z6, info6 = K.solve_constrained(gen, A, Y, P, cth, tol_rel=1e-6)
                    Z, info = K.solve_constrained(gen, A, Y, P, cth, tol_rel=1e-9, cap=200000)
                    fz = f(Z, A, c, Y)
                    w = dict(excess=-1.0)
                    for i in range(min(B, 6)):
                        a_, p_ = A[i:i + 1], P[i].numpy()
                        fi = lambda v: float(f(torch.from_numpy(v).unsqueeze(0), a_, c, Y)[0])
                        gi = lambda v: gr(torch.from_numpy(v).unsqueeze(0), a_, c, Y)[0].numpy()
                        best = None
                        for st in (cth * p_, 0.5 * (cth * p_ + Z[i].numpy()), Z[i].numpy()):
                            r = minimize(lambda v: -fi(v) if math.isfinite(fi(v)) else 1e30, st, jac=lambda v: -gi(v), method='SLSQP',
                                         constraints=[{'type': 'ineq', 'fun': lambda v: 1 - v @ v, 'jac': lambda v: -2 * v},
                                                      {'type': 'ineq', 'fun': lambda v, p_=p_: v @ p_ - cth, 'jac': lambda v, p_=p_: p_}],
                                         options={'ftol': 1e-15, 'maxiter': 2000})
                            if r.x @ r.x <= 1 + 1e-9 and r.x @ p_ >= cth - 1e-9 and math.isfinite(fi(r.x)):
                                best = fi(r.x) if best is None else max(best, fi(r.x))
                        if best is not None:
                            w['excess'] = max(w['excess'], best - float(fz[i]))
                    ufeas = (Zu * P).sum(1) >= cth
                    au = float(C.angle(Z[ufeas], Zu[ufeas]).max()) if bool(ufeas.any()) else None
                    r_ = dict(kind=kind, B=B, M=int(A.shape[1]), d=d, p=pk, gen=gen, theta=th, iters_max=int(info6['iters'].max()),
                              cap=int(info6['cap'].sum()), rel_max=float(info6['rel'].max()), viol_max=float(max(info6['violation'].max(), info['violation'].max())),
                              tight_cap=int(info['cap'].sum()), tight_rel_max=float(info['rel'].max()), angle_tol6_vs_tol9_max=float(C.angle(Z6, Z).max()),
                              slsqp_minus_solver_max=w['excess'], n_unconstrained_feasible=int(ufeas.sum()), angle_vs_unconstrained_when_feasible=au,
                              n_half_active=int(info['half_active'].sum()))
                    rows.append(r_)
                    good &= (r_['cap'] == 0 and r_['rel_max'] <= 1e-6 and r_['viol_max'] <= 1e-12 and r_['slsqp_minus_solver_max'] <= 1e-8
                             and (au is None or au <= 1e-3))
    ok('constrained', good, n_half_active_total=sum(r['n_half_active'] for r in rows), cases=rows)


def check_task_scores():
    g = torch.Generator().manual_seed(5)
    good, n = True, 0
    for B, U, d in ((10, 30, 8), (40, 200, 16)):
        Tg = C.unit(torch.randn(B, d, generator=g, dtype=DT)); E = C.unit(torch.randn(U, d, generator=g, dtype=DT))
        truth = torch.zeros(B, U, dtype=torch.bool)
        for i in range(B):
            k = 1 + int(torch.randint(0, 5, (1,), generator=g))
            truth[i, torch.randperm(U, generator=g)[:k]] = True
        rk, mg, st = K.task_scores(Tg, E, truth)
        for i in range(B):
            s = (Tg[i] @ E.t()).numpy(); tr = truth[i].numpy()
            bt = s[tr].max(); comp = s[~tr]
            good &= int(rk[i]) == int((comp > bt).sum()) and abs(float(mg[i]) - (bt - comp.max())) < 1e-14
            n += 1
    ok('task_scores', good, n_rows=n)


def check_table1():
    ref = np.array([0, 0, 0, 3, 3, 12, 12, 12, 0, 5])
    new = np.array([0, 4, 15, 0, 3, 0, 6, 20, 9, 11])
    t = T1.transitions(ref, new)
    exp = {'1->2-10': 2, '1->>10': 1, '2-10->1': 1, '>10->1': 1, '>10->2-10': 1}
    good = all(t[k] == v for k, v in exp.items()) and t['matrix_ref_rows_new_cols'] == [[1, 2, 1], [1, 1, 1], [1, 1, 1]]
    ok('table1', good, got=t, expected=exp)


if __name__ == '__main__':
    torch.set_num_threads(4)
    t0 = time.time()
    check_table1()
    check_task_scores()
    check_projection()
    check_constrained()
    res['all_pass'] = all(v['pass_'] for v in res.values() if isinstance(v, dict))
    res['seconds'] = time.time() - t0
    res['audit_sha256'] = C.own_hashes()
    json.dump(res, open(sys.argv[1], 'w'), indent=1, default=float)
    sys.exit(0 if res['all_pass'] else 1)
