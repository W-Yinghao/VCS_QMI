"""R27 audit tiny-case check (CPU, synthetic only, no data read). Usage: audit_check.py OUT.json

 1 spg_max_equals_production  audit_spg.solve_spg_ref(ref='max') == production r27_targets.solve_spg bitwise
                              (Z, iters, rel) on synthetic graphs, bce and snce, several shapes incl. duplicate rows
 2 spg_min_converges          ref='min' reaches rel KKT <= tol on the same problems, no cap hits; objective within the
                              ball certificate 2*res of a tight FISTA (1e-10) solution; records #decrease-accepts
                              (> 0 somewhere shows the nonmonotone branch is exercised), iterations, angles
 3 slsqp_cost                 reduced-coordinate SLSQP on production-size rows (B=128, M~600, d=1024): seconds/row,
                              objective vs tight FISTA (decides whether SLSQP on a subset is cheap enough)
 4 ranks                      host evaluation.i2t / t2i (return_ranks) on random continuous sims (5 captions/image):
                              recalls recomputed from the returned ranks == host recalls; host ranks == independent
                              strict-count ranks; also with an injected tie to document tie behaviour
"""
import sys, json, time
import numpy as np
import torch

import audit_common as C
from audit_common import T
import audit_spg as S
from evaluation import i2t, t2i

res = {}


def ok(name, cond, **info):
    res[name] = dict(pass_=bool(cond), **info)
    print(name, 'PASS' if cond else 'FAIL', json.dumps(info, default=float)[:2500], flush=True)


PROBS = [('graph', 16, 70, 12, 1), ('graph', 32, 150, 48, 2), ('dup', 32, 150, 48, 3), ('graph', 64, 300, 128, 4),
         ('dup', 64, 300, 128, 5)]


def check_spg():
    eq_bad, rows, min_good, ndec_tot = [], [], True, 0
    for kind, B, M, d, seed in PROBS:
        A, Y = C.problem(kind, B, M, d, seed=seed)
        c = A.mean(0)
        for gen in ('bce', 'snce', 'vcs'):
            Zp, ip = T.solve_spg(gen, A, Y)
            Zm, im = S.solve_spg_ref(gen, A, Y, ref='max')
            same = torch.equal(Zp, Zm) and torch.equal(ip['iters'], im['iters']) and torch.equal(ip['rel'], im['rel'])
            if not same:
                eq_bad.append((kind, B, M, d, gen))
            Zn, inn = S.solve_spg_ref(gen, A, Y, ref='min')
            if gen == 'vcs':
                Zr, ir = T.solve_vcs(A, Y)
            else:
                Zr, ir = T.solve_fista(gen, A, Y, tol_rel=1e-10, cap=60000)
            f = T.OBJ[gen][0]
            gap_min = f(Zr, A, c, Y) - f(Zn, A, c, Y)
            gap_max = f(Zr, A, c, Y) - f(Zp, A, c, Y)
            exc = float((gap_min - 2 * inn['res']).max())
            g = bool(inn['cap'].sum() == 0 and float(inn['rel'].max()) <= T.TOL_REL and exc <= 1e-9)
            min_good &= g
            ndec_tot += int(inn['n_decrease_accepts'].sum())
            rows.append(dict(kind=kind, B=B, M=M, d=d, gen=gen, max_bitwise_equal=same,
                             it_max_mean=float(ip['iters'].double().mean()), it_min_mean=float(inn['iters'].double().mean()),
                             it_max_max=int(ip['iters'].max()), it_min_max=int(inn['iters'].max()),
                             cap_max=int(ip['cap'].sum()), cap_min=int(inn['cap'].sum()),
                             rel_min_max=float(inn['rel'].max()), dec_accepts_min=int(inn['n_decrease_accepts'].sum()),
                             dec_accepts_max=int(im['n_decrease_accepts'].sum()),
                             gap_ref_minus_min_max=float(gap_min.max()), gap_ref_minus_prod_max=float(gap_max.max()),
                             cert_excess_min=exc, angle_min_vs_ref_max=float(C.angle(Zn, Zr).max()),
                             angle_prod_vs_ref_max=float(C.angle(Zp, Zr).max()), angle_min_vs_prod_max=float(C.angle(Zn, Zp).max())))
    ok('spg_max_equals_production', not eq_bad, mismatches=eq_bad, n_cases=len(rows))
    ok('spg_min_converges', min_good and ndec_tot > 0, total_decrease_accepts_min=ndec_tot, cases=rows)


def check_slsqp():
    A, Y = C.problem('graph', 128, 1400, 1024, seed=7)    # M after merging ~ a production i2t graph
    info = dict(B=int(A.shape[0]), M=int(A.shape[1]), d=int(Y.shape[1]))
    c = A.mean(0)
    for gen in ('bce', 'snce'):
        Zf, _ = T.solve_fista(gen, A, Y, tol_rel=1e-10, cap=60000)
        out = C.slsqp_rows(gen, A, Y, [0, 1], maxiter=3000)
        f = T.OBJ[gen][0]
        info[gen] = [dict(row=o['row'], seconds=o['seconds'], nit=o['nit'], success=o['success'], feasible=o['feasible'],
                          f_fista_minus_slsqp=(float(f(Zf[o['row']:o['row'] + 1], A[o['row']:o['row'] + 1], c, Y)[0]) - o['f']) if o['feasible'] else None,
                          angle_vs_fista=float(C.angle(o['z'].unsqueeze(0), Zf[o['row']:o['row'] + 1])[0]) if o['feasible'] else None)
                     for o in out]
    ok('slsqp_cost', True, **info)


def check_ranks():
    rng = np.random.default_rng(3)
    good, det = True, {}
    for N, per in ((40, 5), (200, 5), (60, 1)):
        sims = rng.standard_normal((N, N * per)) + 2.0 * np.kron(np.eye(N), np.ones(per)) * rng.random((N, 1))
        (r1, r5, r10, _, _), (rk_i, top1_i, _, _) = i2t(N, sims, per, return_ranks=True)
        (ri1, ri5, ri10, _, _), (rk_t, top1_t, _, _) = t2i(N, sims, per, return_ranks=True)
        a = C.recalls_from_ranks(rk_i.astype(np.int64)); b = C.recalls_from_ranks(rk_t.astype(np.int64))
        ind_i, ind_t = C.ranks_independent(sims, per)
        d = dict(recall_diff=float(np.abs(np.array(a + b) - np.array([r1, r5, r10, ri1, ri5, ri10])).max()),
                 rank_i2t_equal_independent=bool((rk_i.astype(np.int64) == ind_i).all()),
                 rank_t2i_equal_independent=bool((rk_t.astype(np.int64) == ind_t).all()))
        det[f'N{N}_per{per}'] = d
        good &= d['recall_diff'] == 0 and d['rank_i2t_equal_independent'] and d['rank_t2i_equal_independent']
    # tie behaviour (documentation): duplicate one caption column of image 0 into a distractor
    N, per = 20, 5
    sims = rng.standard_normal((N, N * per))
    sims[:, 7] = sims[:, 0]
    (_, _, _, _, _), (rk_i, _, _, _) = i2t(N, sims, per, return_ranks=True)
    ind_i, _ = C.ranks_independent(sims, per)
    det['tie_case_i2t_rank_host_minus_independent'] = [int(x) for x in (rk_i.astype(np.int64) - ind_i)]
    ok('ranks', good, **det)


if __name__ == '__main__':
    torch.set_num_threads(4)
    t0 = time.time()
    check_spg()
    check_ranks()
    check_slsqp()
    res['all_pass'] = all(v['pass_'] for v in res.values() if isinstance(v, dict))
    res['seconds'] = time.time() - t0
    res['code_v11_sha256'] = C.code_hashes()
    res['audit_sha256'] = C.own_hashes()
    json.dump(res, open(sys.argv[1], 'w'), indent=1, default=float)
    sys.exit(0 if res['all_pass'] else 1)
