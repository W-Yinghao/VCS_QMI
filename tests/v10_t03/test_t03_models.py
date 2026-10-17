"""Critics: architecture, shared initial states (MF perm1 / M0 / M1; MLP perm1 / perm8), RNG isolation, the m x m table."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

import t03_common as C
import t03_models as Mo
from v10.mf_vcs import SimplexBilinearCritic


def test_mf_methods_share_initial_state_and_start_at_w0():
    hashes = set()
    for meth in C.MF_METHODS:
        model, info = Mo.build_critic(meth, "diagonal_mixture", 32, 0.75, 16, 1001)
        assert isinstance(model, SimplexBilinearCritic) and model.temperature == 1.0
        assert model.left.in_features == 32 and model.left.out_features == 16 and tuple(model.W.shape) == (16, 16)
        assert not bool(model.W.detach().any())
        hashes.add(info["init_state_sha256"])
    assert len(hashes) == 1


def test_mlp_methods_share_initial_state_and_architecture():
    a, ia = Mo.build_critic("MLP_VCS_perm1", "two_block", 8, 0.25, None, 1001)
    b, ib = Mo.build_critic("MLP_VCS_perm8", "two_block", 8, 0.25, None, 1001)
    assert ia["init_state_sha256"] == ib["init_state_sha256"]
    layers = list(a.net)
    assert [type(l) for l in layers] == [nn.Linear, nn.GELU, nn.Linear, nn.GELU, nn.Linear]
    assert (layers[0].in_features, layers[0].out_features, layers[2].out_features, layers[4].out_features) == (16, 64, 64, 1)
    T = Mo.critic_table(a, 8)
    assert T.shape == (8, 8) and bool((T.abs() < 1).all())                         # T = tanh f


def test_init_depends_on_algo_seed_and_condition_not_on_global_rng():
    torch.manual_seed(123)
    st = torch.random.get_rng_state()
    _, i1 = Mo.build_critic("MF_VCS_M0", "two_block", 8, 0.25, 16, 1001)
    assert torch.equal(st, torch.random.get_rng_state())                        # the global generator is untouched
    torch.manual_seed(999)
    _, i2 = Mo.build_critic("MF_VCS_M0", "two_block", 8, 0.25, 16, 1001)
    _, i3 = Mo.build_critic("MF_VCS_M0", "two_block", 8, 0.25, 16, 1002)
    _, i4 = Mo.build_critic("MF_VCS_M0", "two_block", 8, 0.75, 16, 1001)
    _, i5 = Mo.build_critic("MF_VCS_M0", "two_block", 8, 0.25, 8, 1001)
    assert i1["init_state_sha256"] == i2["init_state_sha256"]
    assert len({i1["init_state_sha256"], i3["init_state_sha256"], i4["init_state_sha256"], i5["init_state_sha256"]}) == 4


def test_init_is_pytorch_default_linear_init():
    seed = Mo.build_critic("MF_VCS_M0", "two_block", 8, 0.25, 16, 1001)[1]["init_seed64"]
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        ref = SimplexBilinearCritic(8, 8, 16, 16)
    model, _ = Mo.build_critic("MF_VCS_M1", "two_block", 8, 0.25, 16, 1001)
    assert all(torch.equal(p, q) for p, q in zip(ref.state_dict().values(), model.state_dict().values()))


def test_table_equals_rowwise_evaluation():
    g = np.random.default_rng(0)
    x, y = g.integers(0, 6, 50), g.integers(0, 6, 50)
    for meth, K in (("MF_VCS_M0", 4), ("MLP_VCS_perm1", None)):
        model, _ = Mo.build_critic(meth, "two_block", 6, 0.25, K, 3, dtype=torch.float64)
        if K:
            with torch.no_grad():
                model.W.uniform_(-1, 1)
        X, Y = Mo.onehot(x, 6, dtype=torch.float64), Mo.onehot(y, 6, dtype=torch.float64)
        with torch.no_grad():
            rows = model(X, Y) if K else torch.tanh(model(X, Y))
        T = Mo.critic_table(model, 6)
        assert torch.allclose(T[x, y], rows, atol=1e-14, rtol=0)


def test_unknown_method_and_missing_k():
    with pytest.raises(ValueError):
        Mo.build_critic("MF_dense", "two_block", 8, 0.25, 16, 1)
    with pytest.raises(ValueError):
        Mo.build_critic("MF_VCS_M0", "two_block", 8, 0.25, None, 1)
