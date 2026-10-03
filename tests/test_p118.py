"""P118 (v5 NEXT-I-NEST) tests: the Gaussian conditional oracle, S_B = S_M by construction, nested.py backward compatibility (unchanged vs HEAD),
nested step 0 = coarse model, and the constant-zero control (perfect consistency residuals, wrong increments)."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
import p118_common as C  # noqa: E402
from vcs_measure import nested as NS  # noqa: E402
from vcs_estim.increments import YPairLinear  # noqa: E402


def test_oracle_matches_brute_force_density_ratio():
    mu = C.class_means(); rng = np.random.default_rng(0); z, y, n = C.sample(500, 1.5, rng, mu)
    for k in (8, 4):
        kk = min(k, C.R); v = np.zeros(C.D); v[:C.R] = 1 / np.sqrt(C.R); a = 0.75 * v[:kk]
        def logN(s):
            return -0.5 * np.sum((z[:, :kk] - mu[y][:, :kk] - s * a) ** 2, 1)
        p1 = C.p_n1(y); pz_n = np.exp(np.where(n == 1, logN(1), logN(-1))); pz = p1 * np.exp(logN(1)) + (1 - p1) * np.exp(logN(-1))
        r = pz_n / pz; eta_bf = (r - 1) / (r + 1)
        assert np.allclose(C.eta_set(z, n, y, 1.5, mu, k), eta_bf, atol=1e-10)


def test_irrelevant_coordinates_cancel():
    mu = C.class_means(); rng = np.random.default_rng(1); z, y, n = C.sample(1000, 2.0, rng, mu)
    assert np.array_equal(C.eta_set(z, n, y, 2.0, mu, 64), C.eta_set(z, n, y, 2.0, mu, 8))
    o = C.oracle_S(2.0, n=20000); assert o["B"]["S"] == o["M"]["S"] and o["A"]["S"] < o["M"]["S"]
    assert C.oracle_S(0.0, n=1000)["B"]["S"] == 0.0


def test_nested_py_unchanged_vs_head():
    r = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "src/vcs_measure/nested.py"], cwd=REPO)
    assert r.returncode == 0, "nested.py modified: P109 I2 reproducibility must be re-verified"


def test_nested_step0_equals_coarse():
    torch.manual_seed(0); coarse = YPairLinear(4, 10); z = torch.randn(50, 8); n = torch.randint(0, 2, (50,)); y = torch.randint(0, 10, (50,))
    m = NS.NestedCritic(coarse, lambda x: x[:, :4], YPairLinear(8, 10))
    assert torch.allclose(m(z, n, y), coarse(z[:, :4], n, y))


def test_zero_control_perfect_residuals_wrong_increment():
    n = 200; p1 = np.full(n, 0.5); ro = C.zero_readouts(n); w = np.stack([1 - p1, p1], 1)
    chain = [(ro[0], ro[2][0])] * 3; ch = NS.chain_w(chain, C.CHAIN, w)
    assert ch["R_orth"] == 0.0 and ch["nesting_violations"] == 0 and all(s["delta_J"] == 0.0 for s in ch["steps"].values())
    o = C.oracle_S(2.0, n=20000); true_inc = o["M"]["S"] - o["A"]["S"]
    assert true_inc > 0.02 and (ch["steps"]["M->A"]["delta_J"] - true_inc) < -0.02        # its increment error is the whole increment


def test_posterior_mse_exact_weights():
    tp = np.array([0.5, -0.2]); tq = np.array([[0.1, 0.3], [0.0, -0.4]]); w = np.array([[0.2, 0.8], [0.8, 0.2]])
    ep = np.array([0.4, 0.0]); eq = np.array([[0.0, 0.5], [0.1, -0.5]])
    hand = 0.5 * np.mean((tp - ep) ** 2) + 0.5 * np.mean([0.2 * 0.01 + 0.8 * 0.04, 0.8 * 0.01 + 0.2 * 0.01])
    assert abs(C.posterior_mse(tp, tq, w, ep, eq) - hand) < 1e-15
