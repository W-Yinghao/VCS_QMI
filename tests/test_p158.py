"""P158 tests: the joint-vs-product Gaussian-pair oracle (brute force via scipy), irrelevant coordinates cancel (S_B = S_M), the oracle identities
(L2 / Bregman increments; r_BA and R_orth at the oracle), the quad class contains the oracle logit exactly, nested step 0 = coarse model, the
constant-zero control fails the rule, and nested.py unchanged vs HEAD (P109 / P118 dependency)."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import multivariate_normal, norm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
import p158_common as C  # noqa: E402
import p158_aggregate as AG  # noqa: E402
from vcs_estim import increments as INC  # noqa: E402
from vcs_measure import nested_joint as NJ  # noqa: E402


def test_oracle_matches_brute_force_density_ratio():
    rng = np.random.default_rng(0); rho = 0.37; x, y = C.sample_p(200, rho, rng)
    for k in (10, 5, 2):
        S = np.array([[1.0, rho], [rho, 1.0]]); lr = np.zeros(200)
        for j in range(k):
            lr += multivariate_normal(np.zeros(2), S).logpdf(np.stack([x[:, j], y[:, j]], 1)) - norm.logpdf(x[:, j]) - norm.logpdf(y[:, j])
        assert np.allclose(C.log_ratio(x, y, rho, k), lr, atol=1e-10)
    assert np.all(C.log_ratio(x, y, 0.0, 10) == 0.0)


def test_irrelevant_coordinates_cancel_and_chain_order():
    rng = np.random.default_rng(1); x, y = C.sample_p(500, 0.3, rng)
    assert np.array_equal(C.eta(x, y, 0.3, C.KSET["B"]), C.eta(x, y, 0.3, C.KSET["M"]))
    o = C.oracle(0.3, n=20_000)
    assert o["sets"]["B"]["S"] == o["sets"]["M"]["S"] and o["sets"]["A"]["S"] < o["sets"]["M"]["S"]
    assert o["sets"]["B"]["G"] == o["sets"]["M"]["G"] and o["sets"]["A"]["G"] < o["sets"]["M"]["G"]
    z = C.oracle(0.0, n=2_000); assert all(z["sets"][k]["S"] == 0.0 and abs(z["sets"][k]["G"]) < 1e-12 for k in C.CHAIN)


def test_oracle_identities_l2_and_bregman():
    o = C.oracle(0.22, n=200_000)
    sets = o["sets"]
    for s, st in o["steps"].items():                          # population identities; on the finite truth sample the gaps are O(1 / sqrt n)
        b, a = s.split("->")
        assert abs(st["identity_gap_L2"]) < 5 * (st["L2_se"] + sets[b]["S_se"] + sets[a]["S_se"]) + 1e-12, s
        assert abs(st["identity_gap_Bregman"]) < 5 * (st["Bregman_se"] + sets[b]["G_se"] + sets[a]["G_se"]) + 1e-12, s
    assert o["steps"]["B->M"]["identity_gap_L2"] == 0.0 and o["steps"]["B->M"]["identity_gap_Bregman"] == 0.0   # eta_B = eta_M exactly


def test_r_ba_and_r_orth_vanish_at_the_oracle():
    rng = np.random.default_rng(2); rho = 0.22; n = 100_000; xp, yp = C.sample_p(n, rho, rng); pi = C.shift(n, rng)
    f = {k: (C.oracle_logit(xp, yp, rho, C.KSET[k]), C.oracle_logit(xp, yp[pi], rho, C.KSET[k])) for k in C.CHAIN}
    com = INC.chain_summary([np.tanh(f[k][0]) for k in C.CHAIN], [np.tanh(f[k][1]) for k in C.CHAIN], C.CHAIN)
    js = C.js_chain_summary([f[k][0] for k in C.CHAIN], [f[k][1] for k in C.CHAIN], C.CHAIN)
    assert com["R_orth"] == 0.0 and js["R_orth"] == 0.0                       # eta_B = eta_M exactly -> D(B, M) = 0 and D(M, A) = D(B, A)
    assert abs(com["steps"]["M->A"]["r_BA"]) < 0.01 and abs(js["steps"]["M->A"]["r_BA"]) < 0.01   # population zero; O(1/sqrt n) here
    assert com["steps"]["B->M"]["r_BA"] == 0.0 and js["steps"]["B->M"]["r_BA"] == 0.0


def test_quad_class_contains_the_oracle_logit():
    rho, k = 0.22, 5; rng = np.random.default_rng(3); x, y = C.sample_p(300, rho, rng)
    m = NJ.QuadCritic(k).double(); a = 0.5 * (0.5 - 1.0 / (2 * (1 - rho ** 2))); b = 0.5 * rho / (1 - rho ** 2)
    w = torch.zeros(5 * k, dtype=torch.float64); w[2 * k:3 * k] = a; w[3 * k:4 * k] = a; w[4 * k:5 * k] = b
    with torch.no_grad():
        m.lin.weight.copy_(w[None]); m.lin.bias.fill_(-0.25 * k * np.log(1 - rho ** 2))
        f = m(torch.as_tensor(x[:, :k], dtype=torch.float64), torch.as_tensor(y[:, :k], dtype=torch.float64)).numpy()
    assert np.allclose(f, C.oracle_logit(x, y, rho, k), atol=1e-9)


def test_nested_step0_equals_coarse_and_mlp_shapes():
    torch.manual_seed(0); coarse = NJ.QuadCritic(5); x = torch.randn(50, 10); y = torch.randn(50, 10)
    m = NJ.NestedJointCritic(coarse, lambda a, b: (a[:, :5], b[:, :5]), NJ.MLPCritic(10))
    assert torch.allclose(m(x, y), coarse(x[:, :5], y[:, :5]))
    assert NJ.MLPCritic(10)(x, y).shape == (50,)


def test_fit_runs_and_nested_offers_step0():
    rng = np.random.default_rng(4); x, y = C.sample_p(300, 0.5, rng); pi = C.shift(300, rng)
    xt, yt, nt = torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32), torch.as_tensor(pi)
    a = NJ.fit(xt[:, :5], yt[:, :5], nt, objective="vcs", kind="quad", steps=20, seed=0)
    m = NJ.fit(xt[:, :10], yt[:, :10], nt, objective="js", kind="quad", steps=20, seed=1, coarse=a, cmap=lambda u, v: (u[:, :5], v[:, :5]))
    assert hasattr(m, "best_step") and m.best_step >= 0
    fp, fq = NJ.readouts(m, xt[:, :10], yt[:, :10], nt); assert fp.shape == (300,) and fq.shape == (300,) and fp.dtype == np.float64


def test_zero_control_perfect_residuals_wrong_increment_and_fails_rule():
    n = 200; f = {k: (np.zeros(n), np.zeros(n)) for k in C.CHAIN}
    import p158_truth as T
    ro = T.chain_readouts(f)
    for rd in ("common", "js_native"):
        assert ro[rd]["R_orth"] == 0.0 and all(v == 0.0 for v in ro[rd]["increment"].values()) and ro[rd]["ordering_violations"] == 0
    o = C.oracle(0.22, n=20_000); ti = o["sets"]["M"]["S"] - o["sets"]["A"]["S"]; assert ti > 0.02
    # synthetic summaries: zero control vs a baseline with half the oracle increment recovered
    mk = lambda inc, rmse, bm, pb, pa, ro_: {"inc_rmse_M->A": rmse, "inc_err_B->M": {"mean": bm}, "inc_M->A": {"mean": inc}, "post_B": {"mean": pb}, "post_A": {"mean": pa}, "R_orth": {"mean": ro_}}
    cells = {c: {"oracle": o, "summary": {"vcs/indep/common": mk(0.6 * ti, 0.02, 0.003, 0.01, 0.01, 0.002), "vcs/zero/common": mk(0.0, ti, 0.0, 0.05, 0.03, 0.0)}} for c in ("weak", "moderate")}
    r = AG.rule(cells, "vcs", "zero", "common")
    assert r["verdict"] != "error reduced" and not r["weak"]["c1_rmse"] and not r["weak"]["c4_no_shrinkage"]


def test_nested_py_unchanged_vs_head():
    r = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "src/vcs_measure/nested.py"], cwd=REPO)
    assert r.returncode == 0, "nested.py modified: P109 I2 / P118 reproducibility must be re-verified"
