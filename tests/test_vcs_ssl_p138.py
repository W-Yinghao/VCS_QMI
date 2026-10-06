"""P138 (v7 V7-PAIR): sampled image shifts x all view pairs.  K = B-1 equals all_view_tokens (loss and gradients, VCS and JS);
exact-enumeration unbiasedness of loss and gradient; the finite-population variance formula; shift-RNG determinism / resume;
config acceptance and refusals; a CPU trainer smoke through the dispatcher."""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import pytest
import torch
import yaml
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))
from vcs_ssl.config import ConfigError  # noqa: E402
from vcs_ssl.models.critic import FixedCosineCritic  # noqa: E402
from vcs_ssl.objectives import (P138_SCOPE, all_view_tokens_loss, compute_objective_views_p104, sample_image_shifts,  # noqa: E402
                                sampled_shift_all_view_loss)
from test_vcs_ssl_v2 import _load, _small, synthetic  # noqa: E402

D = 16  # all synthetic tensors are explicitly float64; no global default-dtype change (other test modules share the process)


def _views(V=4, B=8, seed=0, grad=False):
    g = torch.Generator().manual_seed(seed)
    vs = [F.normalize(torch.randn(B, D, generator=g, dtype=torch.float64), dim=1) for _ in range(V)]
    if grad:
        vs = [v.requires_grad_(True) for v in vs]
    return vs


def _critic():
    return FixedCosineCritic(D, 2.0, -1.0).double()


def _grads(loss, vs):
    return torch.cat([g.reshape(-1) for g in torch.autograd.grad(loss, vs)])


@pytest.mark.parametrize("objective", ["vcs", "js"])
def test_full_shift_set_equals_all_view_tokens(objective):
    V, B = 4, 8
    vs = _views(V, B, grad=True)
    c = _critic()
    ref = all_view_tokens_loss(vs, c, negative_detach=False, chunk_size=7, objective=objective)
    new = sampled_shift_all_view_loss(vs, c, shifts=list(range(1, B)), objective=objective)
    assert new["n_pos"] == ref["n_pos"] == B * V * (V - 1) and new["n_neg"] == ref["n_neg"] == B * V * V * (B - 1)
    assert torch.allclose(new["loss"], ref["loss"], atol=1e-12, rtol=0)
    for k in ("J_raw", "R_binary", "t_pos_mean", "t_neg_mean", "t_pos_second", "t_neg_second"):
        assert torch.allclose(new[k], ref[k], atol=1e-12, rtol=0), k
    g_ref, g_new = _grads(ref["loss"], vs), _grads(new["loss"], vs)
    assert torch.allclose(g_new, g_ref, atol=1e-12, rtol=0)
    # shift order does not matter
    perm = sampled_shift_all_view_loss(vs, c, shifts=list(range(B - 1, 0, -1)), objective=objective)
    assert torch.allclose(perm["loss"], ref["loss"], atol=1e-12, rtol=0)


@pytest.mark.parametrize("objective", ["vcs", "js"])
def test_unbiased_loss_and_gradient_by_enumeration(objective):
    V, B, K = 3, 6, 2
    vs = _views(V, B, seed=1, grad=True)
    c = _critic()
    full = sampled_shift_all_view_loss(vs, c, shifts=list(range(1, B)), objective=objective)
    g_full = _grads(full["loss"], vs)
    subsets = list(itertools.combinations(range(1, B), K))
    losses, grads = [], []
    for sub in subsets:
        s = sampled_shift_all_view_loss(vs, c, shifts=list(sub), objective=objective)
        losses.append(s["loss"].detach()); grads.append(_grads(s["loss"], vs))
    assert torch.allclose(torch.stack(losses).mean(), full["loss"].detach(), atol=1e-12, rtol=0)
    assert torch.allclose(torch.stack(grads).mean(0), g_full, atol=1e-12, rtol=0)


@pytest.mark.parametrize("objective", ["vcs", "js"])
def test_finite_population_variance(objective):
    V, B, K = 3, 7, 3
    vs = _views(V, B, seed=2)
    c = _critic()
    with torch.no_grad():
        lp = sampled_shift_all_view_loss(vs, c, shifts=[1], objective=objective)
        # H_d = L_{single shift d} − L_P;  L_P is common to every single-shift loss
        single = torch.stack([sampled_shift_all_view_loss(vs, c, shifts=[d], objective=objective)["loss"] for d in range(1, B)])
        Dn = B - 1
        s2 = single.var(unbiased=True)  # differences of L_P cancel: Var over d of (L_P + H_d) = Var of H_d
        vals = torch.stack([sampled_shift_all_view_loss(vs, c, shifts=list(sub), objective=objective)["loss"]
                            for sub in itertools.combinations(range(1, B), K)])
    var_exact = vals.var(unbiased=False)  # population variance over all equally likely K-subsets
    var_formula = (1.0 / K) * (1.0 - K / Dn) * s2
    assert math.isclose(float(var_exact), float(var_formula), rel_tol=1e-9, abs_tol=1e-15)
    assert lp["n_neg"] == B * V * V


def test_shift_sampler_distinct_deterministic_and_resumable():
    g = torch.Generator().manual_seed(123)
    seq = [sample_image_shifts(256, 16, g) for _ in range(5)]
    for s in seq:
        assert len(s) == 16 and len(set(s)) == 16 and all(1 <= d <= 255 for d in s)
    g2 = torch.Generator().manual_seed(123)
    assert [sample_image_shifts(256, 16, g2) for _ in range(5)] == seq
    g3 = torch.Generator().manual_seed(7)
    sample_image_shifts(256, 16, g3); state = g3.get_state()
    a = [sample_image_shifts(256, 16, g3) for _ in range(3)]
    g4 = torch.Generator(); g4.set_state(state)
    assert [sample_image_shifts(256, 16, g4) for _ in range(3)] == a
    assert sorted(sample_image_shifts(9, 8, torch.Generator().manual_seed(0))) == list(range(1, 9))
    for bad in (0, 256):
        with pytest.raises(ValueError):
            sample_image_shifts(256, bad, g)


def test_loss_input_checks():
    vs = _views(4, 8)
    c = _critic()
    for bad in ([], [1, 1], [0], [8]):
        with pytest.raises(ValueError):
            sampled_shift_all_view_loss(vs, c, shifts=bad)


def test_only_selected_pairs_scored():
    V, B, K = 4, 32, 5
    s = sampled_shift_all_view_loss(_views(V, B), _critic(), shifts=[1, 3, 5, 7, 9])
    assert s["score_elements_computed"] == B * V * (V - 1) + K * B * V * V


# ----------------------------------------------------------------------------------------------------------------------------- configs
PARENTS = {"vcs": "cifar10_hpY_AP3_views4_800ep_seed0.yaml", "js": "cifar10_hpJS_AP3_views4_800ep_seed0.yaml"}


def p138(method: str, k: int = 16) -> dict:
    d = yaml.safe_load((REPO / "configs" / PARENTS[method]).read_text())
    d["pairing"]["pair_scope"] = P138_SCOPE
    d["pairing"]["k"] = k
    d["pairing"].pop("all_view_chunk", None)
    return d


@pytest.mark.parametrize("method", ["vcs", "js"])
def test_config_accepted(tmp_path, method):
    cfg = _load(tmp_path, p138(method), f"{method}.yaml")
    assert cfg["pairing"]["pair_scope"] == P138_SCOPE and cfg["pairing"]["k"] == 16


def test_config_refusals(tmp_path):
    muts = [lambda d: d["pairing"].update({"negative_detach": True}),
            lambda d: d["pairing"].update({"all_view_chunk": 256}),
            lambda d: d["model"]["critic"].update({"affine_mode": "learned"}),
            lambda d: d["pairing"].update({"k": 256}),
            lambda d: d["views"].update({"count": 2})]
    for i, mutate in enumerate(muts):
        d = p138("vcs"); mutate(d)
        with pytest.raises(ConfigError):
            _load(tmp_path, d, f"bad{i}.yaml")
    d = p138("js"); d["objective"].pop("js_all_view_tokens")
    with pytest.raises(ConfigError):
        _load(tmp_path, d, "badjs.yaml")


@pytest.mark.parametrize("method", ["vcs", "js"])
def test_dispatcher_uses_k_shifts(tmp_path, method):
    cfg = _small(_load(tmp_path, p138(method), f"d{method}.yaml"), batch=32)
    V, B = 4, 32
    vs = _views(V, B, grad=True)
    feats = {"views_z": vs, "views_p": vs, "views_h": vs}
    c = _critic()
    out = compute_objective_views_p104(feats, cfg=cfg, critic=c, pair_generator=torch.Generator().manual_seed(5))
    assert len(out["shift"]) == 16 and len(set(out["shift"])) == 16
    assert out["n_neg"] == 16 * B * V * V and out["critic_pair_evals"] == B * V * (V - 1) + 16 * B * V * V
    out["loss"].backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in vs)


@pytest.mark.parametrize("method", ["vcs", "js"])
def test_trainer_smoke_cpu(tmp_path, method):
    if True:
        from vcs_ssl.train import Trainer
        data, m = synthetic()
        cfg = _small(_load(tmp_path, p138(method), f"t{method}.yaml"), batch=32)
        run_dir = Path(cfg["run"]["output_root"]) / f"p138{method}"; run_dir.mkdir(parents=True, exist_ok=True)
        tr = Trainer(cfg, run_dir=run_dir, data=data, manifest=m, device=torch.device("cpu"), stage="TEST", smoke_steps=2, smoke_epoch_steps=2,
                     epoch_eval=False)
        tr.setup()
        tr.run()
        assert tr.cost["critic_pair_evaluations"] > 0 and tr.cost["negative_pairs"] > 0
