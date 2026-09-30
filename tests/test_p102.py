"""P102 (T line) — increment identities on the exact discrete oracle toy, the finite-critic residual identity, R_orth, paired-bootstrap integrity."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vcs_estim import increments as I  # noqa: E402

TOL = 1e-12


def _chain():
    p, q = I.toy_joint(n_y=3, n_x=12, seed=1, strength=1.3)
    groups = [np.arange(12), np.arange(12) // 2, np.arange(12) // 4, np.arange(12) // 12]
    return p, q, [I.oracle_t(p, q, g) for g in groups]


def test_oracle_increment_identity_float64():
    """Delta = S_B - S_A = E_M (T_B* - T_A*)^2 for every nested step; J(T*) = S at the oracle."""
    p, q, ts = _chain()
    S = [I.exact_em(p, q, t ** 2) for t in ts]
    for l in range(len(ts) - 1):
        assert abs((S[l] - S[l + 1]) - I.exact_em(p, q, (ts[l] - ts[l + 1]) ** 2)) < TOL
        assert S[l] >= S[l + 1] - TOL                                       # coarsening never adds dependence
    for t, s in zip(ts, S):
        assert abs(I.exact_j(p, q, t) - s) < TOL


def test_oracle_r_orth_is_zero():
    p, q, ts = _chain()
    d = [ts[l] - ts[l + 1] for l in range(len(ts) - 1)]
    rorth = sum(I.exact_em(p, q, x ** 2) for x in d) - I.exact_em(p, q, (ts[0] - ts[-1]) ** 2)
    cross = -2 * sum(I.exact_em(p, q, d[l] * d[k]) for l in range(len(d)) for k in range(l + 1, len(d)))
    assert abs(rorth) < TOL and abs(rorth - cross) < TOL


def test_finite_residual_identity_on_samples():
    """r_BA = Delta_J - D_T = 2 E_M[(C - T_B)(T_B - T_A)] for arbitrary bounded critics on arbitrary samples (algebra, float64)."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = int(rng.integers(5, 400))
        tbp, tbq, tap, taq = (np.tanh(rng.normal(0, 2, n)) for _ in range(4))
        inc = I.increment(tbp, tbq, tap, taq)
        assert inc["r_BA_identity_gap"] < TOL
        assert inc["D_T"] >= 0


def test_residual_zero_at_oracle_and_nonzero_off_oracle():
    p, q, ts = _chain()
    tb, ta = ts[0], ts[2]
    r_or = 2 * I.exact_em(p, q, (1 - tb) * (tb - ta), (-1 - tb) * (tb - ta))
    assert abs(r_or) < TOL
    tb_bad = np.clip(tb + 0.2, -1, 1)
    r_bad = 2 * I.exact_em(p, q, (1 - tb_bad) * (tb_bad - ta), (-1 - tb_bad) * (tb_bad - ta))
    assert abs(r_bad) > 1e-4


def test_paired_bootstrap_keeps_images_together():
    """The same resampled image index is applied to every critic's P and Q outputs: with outputs equal to the image id, every replicate of
    (mean over h P) - (mean over logit Q) is exactly zero."""
    n = 50; ids = np.arange(n, dtype=np.float64)
    tp = {"h": ids.copy(), "logit_h": ids.copy()}; tq = {"h": ids.copy(), "logit_h": ids.copy()}
    B = I.paired_bootstrap(tp, tq, lambda a, b: [a["h"].mean() - b["logit_h"].mean(), a["h"].mean()], reps=200, seed=3)
    assert np.all(np.abs(B[:, 0]) < TOL) and B[:, 1].std() > 0


def test_critic_fitters_run_and_select_by_common_score():
    torch.manual_seed(0); n, d, k = 400, 8, 3
    z = torch.randn(n, d); y = torch.randint(0, k, (n,)); nn_ = (torch.rand(n) < 0.5).long()
    z[:, 0] += 1.5 * (2 * nn_ - 1)                                          # dependence between z and N given Y
    n_neg = nn_[torch.randperm(n)]
    for obj in ("vcs", "js"):
        m, pick, vals = I.fit_picked(z, nn_, n_neg, y, objective=obj, n_classes=k, steps=60, seed=0)
        assert pick in ("linear", "mlp") and set(vals) == {"linear", "mlp"}
        with torch.no_grad():
            tp, tq = torch.tanh(m(z, nn_, y)).double().numpy(), torch.tanh(m(z, n_neg, y)).double().numpy()
        assert I.j_from_t(tp, tq) > 0.05
