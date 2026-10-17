"""V10 T03 implementation gates (protocol P2 t03.target.negatives_perm1_perm8.gate and t03.smoke.prerequisites), CPU, seconds.

G1 enumeration   for small n, the mean over ALL derangements of the perm1 objective - value and every parameter gradient - equals the
                 complete U objective: MF (package objective_terms, q_mode 'u') and MLP (explicit all-pairs U); float64, package fp64
                 tolerances (atol 1e-10, rtol 1e-8).  Control: the mean over all n! ordinary permutations equals V, which differs from U.
G2 sampler law   the rejection sampler (the production DerangementStream) has no fixed point in any draw, is uniform over the D_n
                 derangements (chi-square, pre-set p >= 1e-3, fixed seed) and has P(sigma(i) = j) = 1/(n - 1) (max |z| < 5).
G3 rejection     mean attempts per derangement at n = 4096 within 5 standard errors of n!/D_n (-> e).
G4 perm8         the perm8 objective equals the mean of its eight perm1 objectives (float64, exact up to rounding).
G5 table U       the complete empirical U-J from cell counts equals the package U moments on the per-row features (MF) and the
                 explicit all-pairs U (MLP), float64.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
from scipy import stats

import t03_common as C
import t03_models as Mo
import t03_objectives as O

ATOL, RTOL = C.TOLERANCES["float64"]
GATE_SEED = 20261017


def _close(a, b) -> tuple[bool, float]:
    a, b = torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64)
    err = float((a - b).abs().max()) if a.numel() else 0.0
    return bool(torch.allclose(a, b, atol=ATOL, rtol=RTOL)), err


def _fixture(kind: str, n: int, m: int, K: int, seed: int):
    g = np.random.default_rng(seed)
    x, y = g.integers(0, m, n), g.integers(0, m, n)
    if kind == "MF":
        model, _ = Mo.build_critic("MF_VCS_M0", "two_block", m, 0.25, K, seed, dtype=torch.float64)
        with torch.no_grad():                                        # a non-zero feasible head for a non-trivial gradient check
            model.W.copy_(torch.as_tensor(g.uniform(-0.9, 0.9, (K, K))))
    else:
        model, _ = Mo.build_critic("MLP_VCS_perm1", "two_block", m, 0.25, None, seed, dtype=torch.float64)
    return model, x, y


def _objective(kind, model, X, Y, sig=None, mode="u"):
    if kind == "MF":
        A, B = model.features(X, Y)
        if sig is not None:
            return O.mf_derangement_terms(A, B, model.W, sig).j
        if mode == "u":
            return O.mf_u_moments(A, B, model.W).j
        return O.dense_v_terms_rows((A @ model.W) @ B.T).j
    if sig is not None:
        return O.mlp_derangement_terms(model, X, Y, sig).j
    n = len(X)
    Tp = torch.tanh(model(X.repeat_interleave(n, 0), Y.repeat(n, 1))).reshape(n, n)
    return (O.dense_u_terms_rows(Tp) if mode == "u" else O.dense_v_terms_rows(Tp)).j


def _value_grad(kind, model, X, Y, sig=None, mode="u"):
    model.zero_grad(set_to_none=True)
    j = _objective(kind, model, X, Y, sig, mode)
    grads = torch.autograd.grad(j, [p for p in model.parameters()])
    return float(j.detach()), torch.cat([g.reshape(-1) for g in grads])


def gate_enumeration(ns=(4, 5, 6), m: int = 3, K: int = 4) -> dict:
    rows = []
    for kind in ("MF", "MLP"):
        for n in ns:
            model, x, y = _fixture(kind, n, m, K, GATE_SEED + n)
            X, Y = Mo.onehot(x, m, dtype=torch.float64), Mo.onehot(y, m, dtype=torch.float64)
            ju, gu = _value_grad(kind, model, X, Y, mode="u")
            jv, gv = _value_grad(kind, model, X, Y, mode="v")
            ders, perms = O.all_derangements(n), O.all_permutations(n)
            vals, grads = zip(*(_value_grad(kind, model, X, Y, torch.as_tensor(d[None])) for d in ders))
            pvals, pgrads = zip(*(_value_grad(kind, model, X, Y, torch.as_tensor(p[None])) for p in perms))
            okv, ev = _close(np.mean(vals), ju)
            okg, eg = _close(torch.stack(grads).mean(0), gu)
            okpv, epv = _close(np.mean(pvals), jv)
            okpg, epg = _close(torch.stack(pgrads).mean(0), gv)
            uv_gap = abs(ju - jv)
            rows.append({"critic": kind, "n": n, "m": m, "K": K if kind == "MF" else None, "n_derangements": len(ders),
                         "D_n_formula": O.derangement_count(n), "n_permutations": len(perms), "U": ju, "V": jv, "U_minus_V": ju - jv,
                         "derangement_mean_value_err": ev, "derangement_mean_grad_err": eg, "n_grad_elements": int(gu.numel()),
                         "permutation_mean_minus_V_value_err": epv, "permutation_mean_minus_V_grad_err": epg,
                         "permutation_mean_minus_U_value": float(np.mean(pvals)) - ju,
                         "passed": bool(okv and okg and okpv and okpg and len(ders) == O.derangement_count(n) and uv_gap > 1e3 * ATOL)})
    return {"gate": "G1 enumeration: mean over all derangements = complete U (value and gradients); all permutations = V (control)",
            "tolerance": {"atol": ATOL, "rtol": RTOL, "dtype": "float64"}, "rows": rows,
            "status": "PASS" if all(r["passed"] for r in rows) else "FAIL"}


def gate_sampler_law(cases=((4, 36000), (5, 44000)), seed: int = GATE_SEED) -> dict:
    rows = []
    for n, draws in cases:
        stream = O.DerangementStream(np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, n]))), n)
        sig = stream.draw(draws)
        ders = O.all_derangements(n)
        index = {tuple(d): i for i, d in enumerate(ders)}
        fixed = int(np.sum(sig == np.arange(n)))
        valid = O.is_derangement_matrix(sig, n)
        freq = np.bincount([index[tuple(s)] for s in sig], minlength=len(ders))
        chi = stats.chisquare(freq)
        marg = np.zeros((n, n))
        for i in range(n):
            marg[i] = np.bincount(sig[:, i], minlength=n)
        p_off = 1.0 / (n - 1)
        se = math.sqrt(p_off * (1 - p_off) / draws)
        z = np.abs(marg / draws - p_off)[~np.eye(n, dtype=bool)] / se
        rows.append({"n": n, "draws": draws, "classes": len(ders), "fixed_points": fixed, "all_rows_valid_derangements": valid,
                     "chi2": float(chi.statistic), "dof": len(ders) - 1, "p_value": float(chi.pvalue),
                     "diagonal_marginal_count": int(np.trace(marg)), "max_abs_z_offdiag_marginal": float(z.max()),
                     "mean_attempts": stream.record()["mean_attempts"], "expected_attempts": O.expected_attempts(n),
                     "passed": bool(fixed == 0 and valid and chi.pvalue >= 1e-3 and z.max() < 5 and np.trace(marg) == 0)})
    return {"gate": "G2 sampler law: uniform over derangements, no fixed points, P(sigma(i)=j) = 1/(n-1)", "pre_set": {"chi2_p_min": 1e-3,
            "max_abs_z": 5, "seed": seed}, "rows": rows, "status": "PASS" if all(r["passed"] for r in rows) else "FAIL"}


def gate_rejection_rate(n: int = 4096, draws: int = 2000, seed: int = GATE_SEED) -> dict:
    stream = O.DerangementStream(np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, n, 1]))), n)
    t0 = time.perf_counter()
    stream.draw(draws)
    sec = time.perf_counter() - t0
    p = 1.0 / O.expected_attempts(n)
    sd = math.sqrt((1 - p) / p ** 2)
    mean = stream.attempts / stream.draws
    z = (mean - 1 / p) / (sd / math.sqrt(draws))
    return {"gate": "G3 rejection rate at the FIT size", "n": n, "draws": draws, "mean_attempts": mean, "expected": 1 / p, "z": z,
            "seconds": sec, "seconds_per_derangement": sec / draws, "status": "PASS" if abs(z) < 5 else "FAIL"}


def gate_perm8(n: int = 64, m: int = 4, K: int = 4) -> dict:
    out = []
    for kind in ("MF", "MLP"):
        model, x, y = _fixture(kind, n, m, K, GATE_SEED + 99)
        X, Y = Mo.onehot(x, m, dtype=torch.float64), Mo.onehot(y, m, dtype=torch.float64)
        sig = torch.as_tensor(O.DerangementStream(np.random.default_rng(GATE_SEED), n).draw(C.PERM8_R))
        with torch.no_grad():
            j8 = float(_objective(kind, model, X, Y, sig))
            js = [float(_objective(kind, model, X, Y, sig[r:r + 1])) for r in range(C.PERM8_R)]
        ok, err = _close(j8, np.mean(js))
        out.append({"critic": kind, "J_perm8": j8, "mean_J_perm1": float(np.mean(js)), "abs_err": err, "passed": ok})
    return {"gate": "G4 perm8 = mean of its 8 perm1 objectives", "rows": out, "status": "PASS" if all(r["passed"] for r in out) else "FAIL"}


def gate_table_u(n: int = 300, m: int = 5, K: int = 4) -> dict:
    out = []
    for kind in ("MF", "MLP"):
        model, x, y = _fixture(kind, n, m, K, GATE_SEED + 7)
        X, Y = Mo.onehot(x, m, dtype=torch.float64), Mo.onehot(y, m, dtype=torch.float64)
        cnt = O.Counts.from_xy(x, y, m)
        with torch.no_grad():
            T = Mo.critic_table(model, m).numpy()
            ref_u = float(_objective(kind, model, X, Y, mode="u"))
            ref_v = float(_objective(kind, model, X, Y, mode="v"))
        oku, eu = _close(O.empirical_j(T, cnt, "u"), ref_u)
        okv, ev = _close(O.empirical_j(T, cnt, "v"), ref_v)
        out.append({"critic": kind, "n": n, "m": m, "table_U": O.empirical_j(T, cnt, "u"), "reference_U": ref_u, "err_U": eu,
                    "table_V": O.empirical_j(T, cnt, "v"), "reference_V": ref_v, "err_V": ev, "passed": bool(oku and okv)})
    return {"gate": "G5 count-table U/V J = backend U moments (MF) / explicit all-pairs (MLP)", "rows": out,
            "status": "PASS" if all(r["passed"] for r in out) else "FAIL"}


def run_gates() -> dict:
    t0 = time.perf_counter()
    with torch.random.fork_rng(devices=[]):
        gates = {"G1_enumeration": gate_enumeration(), "G2_sampler_law": gate_sampler_law(), "G3_rejection_rate": gate_rejection_rate(),
                 "G4_perm8_mean": gate_perm8(), "G5_table_U": gate_table_u()}
    return {"schema": "v10_t03.gates.v1", "gates": gates, "status": "PASS" if all(g["status"] == "PASS" for g in gates.values()) else "FAIL",
            "seconds": time.perf_counter() - t0}
