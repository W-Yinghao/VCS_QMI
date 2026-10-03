"""P125 (v6 V6-STRUCT) tests: the shared trainer reproduces the P85 train_neural exactly for the JointMLP reference; parameter matching of the
Interaction-MLP; quadratic logit forms (full and low-rank) against explicit formulas; non-zero low-rank factor init (non-zero gradient); a tiny cell runs."""
import math
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vcs_estim import p125  # noqa: E402
from vcs_estim.benchmark import build_roles, train_neural  # noqa: E402
from vcs_estim.data import setting_from_mi  # noqa: E402


def test_trainer_reproduces_train_neural_for_jointmlp():
    st = setting_from_mi("gaussian", 4.0, 20); roles, _ = build_roles(st, 20, 20, 512, 0, smoke=True); dev = torch.device("cpu")
    for kind in ("vcs", "js"):
        c1, i1 = train_neural(kind, "product", 5e-4, roles, 20, 64, 200, 0, dev)
        c2, i2 = p125.train_critic(p125.factory("joint", 20), kind, 5e-4, roles, 64, 200, 0, dev)
        assert i1["selected_update"] == i2["selected_update"] and abs(i1["select_risk"] - i2["select_risk"]) == 0.0
        for (k1, v1), (k2, v2) in zip(c1.state_dict().items(), c2.state_dict().items()):
            assert k1 == k2 and torch.equal(v1, v2)


@pytest.mark.parametrize("d", [10, 20, 100])
def test_interaction_param_match(d):
    target = p125.joint_params(d); m = p125.factory("inter", d)(); n = sum(p.numel() for p in m.parameters())
    assert abs(n - target) / target < 0.02
    ref = sum(p.numel() for p in p125.factory("joint", d)().parameters()); assert ref == target


def test_quadratic_full_formula():
    torch.manual_seed(0); q = p125.QuadLogit(5, 5, None); x, y = torch.randn(7, 5), torch.randn(7, 5)
    ref = torch.stack([q.b[0] + q.u @ x[i] + q.v @ y[i] + x[i] @ q.A @ x[i] + y[i] @ q.B @ y[i] + x[i] @ q.C @ y[i] for i in range(7)])
    assert torch.allclose(q.pairs(x, y), ref, atol=1e-5)


def test_quadratic_lowrank_formula_and_nonzero_grad():
    torch.manual_seed(1); q = p125.QuadLogit(6, 6, 3); x, y = torch.randn(9, 6), torch.randn(9, 6)
    A, B, C = q.Pa @ q.Qa.T, q.Pb @ q.Qb.T, q.Pc @ q.Qc.T
    ref = torch.stack([q.b[0] + q.u @ x[i] + q.v @ y[i] + x[i] @ A @ x[i] + y[i] @ B @ y[i] + x[i] @ C @ y[i] for i in range(9)])
    assert torch.allclose(q.pairs(x, y), ref, atol=1e-5)
    q.pairs(x, y).sum().backward(); assert all(p.grad is not None and p.grad.abs().sum() > 0 for n, p in q.named_parameters() if n != "b") and q.b.grad.abs().sum() > 0


def test_candidates_by_dimension():
    assert p125.candidates_for(20) == ("joint", "inter", "quad_full")
    assert p125.candidates_for(100) == ("joint", "inter", "quad_full", "quad_r4", "quad_r16")


def test_tiny_cell_runs():
    res = p125.struct_cell("C1_gauss_mid", 512, 0, False, torch.device("cpu"), budget=50, lrs=(5e-4,), smoke=True)
    sel = [r for r in res["rows"] if r["selected"]]; assert len(sel) == 6
    assert all(math.isfinite(r["posterior_mse"]) and r["n_params"] > 0 for r in sel)
    rot = p125.struct_cell("C1_gauss_mid", 512, 0, True, torch.device("cpu"), budget=50, lrs=(5e-4,), smoke=True)
    assert {r["candidate"] for r in rot["rows"]} == {"joint", "inter"} and rot["rotation"]["orthogonality_error"] < 1e-10
