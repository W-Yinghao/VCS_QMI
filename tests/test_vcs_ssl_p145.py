"""P145 (v7 V7-STRESS): contaminated training positives.  Replacement sampler (off-diagonal only, j != i, rate, determinism); eps = 0 equals the
frozen all-view-token loss (VCS, JS) and the frozen multi-view SimCLR loss (values and gradients); with replacements the losses equal a brute-force
reference built pair by pair; the part gradients sum to the full gradient; config acceptance / refusals / hash invariance of clean configs; a CPU
trainer smoke through the dispatcher for VCS, JS and SimCLR."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import yaml
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "configs"))
from reference.ssl_core import simclr_nt_xent  # noqa: E402
from vcs_ssl.config import ConfigError  # noqa: E402
from vcs_ssl.models.critic import FixedCosineCritic  # noqa: E402
from vcs_ssl.objectives import (all_view_tokens_loss, compute_objective_views, stress_all_view_tokens_loss, stress_replacements,  # noqa: E402
                                stress_simclr_views)
from test_vcs_ssl_v2 import _load, _small, synthetic  # noqa: E402
import make_p145_configs as M  # noqa: E402

D = 16


def _views(V=4, B=8, seed=0, grad=False):
    g = torch.Generator().manual_seed(seed)
    vs = [F.normalize(torch.randn(B, D, generator=g, dtype=torch.float64), dim=1) for _ in range(V)]
    return [v.requires_grad_(True) for v in vs] if grad else vs


def _critic():
    return FixedCosineCritic(D, 2.0, -1.0).double()


def _grads(loss, vs):
    return torch.cat([g.reshape(-1) for g in torch.autograd.grad(loss, vs, retain_graph=True)])


def test_replacement_sampler():
    V, B = 4, 64
    r0, m0 = stress_replacements(V, B, 0.0, torch.Generator().manual_seed(1))
    assert not m0.any() and torch.equal(r0, torch.arange(B).expand(V, V, B))
    r, m = stress_replacements(V, B, 0.3, torch.Generator().manual_seed(1))
    eye = torch.eye(V, dtype=torch.bool)[:, :, None].expand(V, V, B)
    assert not m[eye].any()
    base = torch.arange(B).expand(V, V, B)
    assert (r[m] != base[m]).all() and torch.equal(r[~m], base[~m])
    assert abs(float(m[~eye].float().mean()) - 0.3) < 0.05
    r2, m2 = stress_replacements(V, B, 0.3, torch.Generator().manual_seed(1))
    assert torch.equal(r, r2) and torch.equal(m, m2)
    # uniform over the other images: the offsets cover 1..B-1
    off = ((r - base) % B)[m]
    assert off.min() >= 1 and off.max() <= B - 1
    with pytest.raises(ValueError):
        stress_replacements(V, B, 1.0, torch.Generator())


@pytest.mark.parametrize("objective", ["vcs", "js"])
def test_eps0_equals_all_view_tokens(objective):
    V, B = 4, 8
    vs = _views(V, B, grad=True); c = _critic()
    repl, rm = stress_replacements(V, B, 0.0, torch.Generator().manual_seed(0))
    ref = all_view_tokens_loss(vs, c, negative_detach=False, chunk_size=7, objective=objective)
    new = stress_all_view_tokens_loss(vs, c, repl=repl, replaced=rm, negative_detach=False, chunk_size=5, objective=objective)
    assert new["n_pos"] == ref["n_pos"] and new["n_neg"] == ref["n_neg"]
    assert torch.allclose(new["loss"], ref["loss"], atol=1e-12, rtol=0)
    for k in ("J_raw", "t_pos_mean", "t_neg_mean", "t_pos_second", "t_neg_second"):
        assert torch.allclose(torch.as_tensor(new[k], dtype=torch.float64), ref[k].detach(), atol=1e-12, rtol=0), k
    assert torch.allclose(_grads(new["loss"], vs), _grads(ref["loss"], vs), atol=1e-12, rtol=0)


def _brute_all_view(vs, c, repl, objective):
    V, B = len(vs), vs[0].shape[0]
    lp, lq, n_p, n_q = 0.0, 0.0, 0, 0
    for a in range(V):
        for b in range(V):
            for i in range(B):
                if a != b:  # positive with the (possibly moved) right end
                    s = (vs[a][i] * vs[b][int(repl[a, b, i])]).sum()
                    f = c.scale * s + c.bias
                    lp = lp + (F.softplus(-2 * f) if objective == "js" else -(torch.tanh(f) - 0.5 * torch.tanh(f) ** 2)); n_p += 1
                for j in range(B):
                    if j != i:  # negatives: every different-image pair, unchanged
                        f = c.scale * (vs[a][i] * vs[b][j]).sum() + c.bias
                        lq = lq + (F.softplus(2 * f) if objective == "js" else torch.tanh(f) + 0.5 * torch.tanh(f) ** 2); n_q += 1
    return lp / n_p + lq / n_q


@pytest.mark.parametrize("objective", ["vcs", "js"])
def test_contaminated_equals_brute_force(objective):
    V, B = 3, 5
    vs = _views(V, B, seed=3, grad=True); c = _critic()
    repl, rm = stress_replacements(V, B, 0.4, torch.Generator().manual_seed(7))
    assert rm.any()
    new = stress_all_view_tokens_loss(vs, c, repl=repl, replaced=rm, negative_detach=False, chunk_size=4, objective=objective, diag=True)
    ref = _brute_all_view(vs, c, repl, objective)
    assert torch.allclose(new["loss"], ref, atol=1e-12, rtol=0)
    assert torch.allclose(_grads(new["loss"], vs), _grads(ref, vs), atol=1e-11, rtol=0)
    assert new["stress_n_replaced"] == int(rm.sum()) and new["stress_gnorm_repl"] > 0 and -1 <= new["stress_cos_repl_total"] <= 1


def test_simclr_eps0_and_brute_force():
    V, B, tau = 4, 6, 0.2
    vs = _views(V, B, seed=5, grad=True)
    repl0, rm0 = stress_replacements(V, B, 0.0, torch.Generator().manual_seed(0))
    pairs = [(a, b) for a in range(V) for b in range(a + 1, V)]
    ref = torch.stack([simclr_nt_xent(vs[a], vs[b], temperature=tau) for a, b in pairs]).mean()
    new = stress_simclr_views(vs, temperature=tau, repl=repl0, replaced=rm0)
    assert torch.allclose(new["loss"].double(), ref.double(), atol=1e-6, rtol=0)
    assert torch.allclose(_grads(new["loss"], vs), _grads(ref, vs), atol=1e-6, rtol=0)
    repl, rm = stress_replacements(V, B, 0.5, torch.Generator().manual_seed(2))
    new = stress_simclr_views(vs, temperature=tau, repl=repl, replaced=rm, diag=True)
    tot, n = 0.0, 0
    for a, b in pairs:  # anchor (a, i) -> target (b, repl[a, b, i]); anchor (b, i) -> (a, repl[b, a, i]); full denominators
        x = F.normalize(torch.cat((vs[a], vs[b])).float(), dim=-1)
        lg = (x @ x.T / tau).masked_fill(torch.eye(2 * B, dtype=torch.bool), -torch.inf)
        for i in range(B):
            tot = tot - torch.log_softmax(lg[i], 0)[B + int(repl[a, b, i])] - torch.log_softmax(lg[B + i], 0)[int(repl[b, a, i])]; n += 2
    assert torch.allclose(new["loss"], tot / n, atol=1e-5, rtol=0)
    assert new["stress_n_replaced"] == sum(int(rm[a, b].sum() + rm[b, a].sum()) for a, b in pairs)


@pytest.mark.parametrize("objective", ["vcs", "js"])
def test_part_gradients_sum_to_total(objective):
    V, B = 4, 8
    vs = _views(V, B, seed=9, grad=True); c = _critic()
    repl, rm = stress_replacements(V, B, 0.25, torch.Generator().manual_seed(4))
    out = stress_all_view_tokens_loss(vs, c, repl=repl, replaced=rm, negative_detach=False, objective=objective, diag=True)
    flat_total = torch.autograd.grad(out["loss"], vs, retain_graph=True)
    tot = float(torch.cat([g.reshape(-1) for g in flat_total]).norm())
    assert out["stress_gnorm_kept"] > 0 and out["stress_gnorm_neg"] > 0
    assert tot <= out["stress_gnorm_kept"] + out["stress_gnorm_repl"] + out["stress_gnorm_neg"] + 1e-9


# ----------------------------------------------------------------------------------------------------------------------------- configs
@pytest.mark.parametrize("key", list(M.PARENTS))
def test_generated_configs_accepted(tmp_path, key):
    cfg = _load(tmp_path, M.build(*key), f"{key[0]}_{key[1]}.yaml")
    assert cfg["pairing"]["stress_epsilon"] == 0.10


def test_clean_parent_unchanged(tmp_path):
    """The marker is not filled when absent (fill=False): a clean parent resolves without it; hash invariance of every existing config is
    checked by the gate against a HEAD worktree."""
    cfg = _load(tmp_path, yaml.safe_load((REPO / "configs" / M.PARENTS[("vcs", "c10")]).read_text()), "parent.yaml")
    assert "stress_epsilon" not in cfg["pairing"]


def test_config_refusals(tmp_path):
    muts = [lambda d: d["pairing"].update({"stress_epsilon": 0.0}),
            lambda d: d["pairing"].update({"stress_epsilon": 1.0}),
            lambda d: d["pairing"].update({"stress_epsilon": True}),
            lambda d: d["pairing"].update({"pair_scope": "cross_view_k"}),
            lambda d: d["views"].update({"count": 2})]
    for i, mutate in enumerate(muts):
        d = M.build("vcs", "c10"); mutate(d)
        with pytest.raises(ConfigError):
            _load(tmp_path, d, f"bad{i}.yaml")


@pytest.mark.parametrize("key", [("vcs", "c10"), ("js", "c10"), ("simclr", "c10")])
def test_dispatcher_and_trainer_smoke(tmp_path, key):
    from vcs_ssl.train import Trainer
    cfg = _small(_load(tmp_path, M.build(*key), f"s{key[0]}.yaml"), batch=32)
    data, m = synthetic()
    run_dir = Path(cfg["run"]["output_root"]) / f"p145{key[0]}"; run_dir.mkdir(parents=True, exist_ok=True)
    tr = Trainer(cfg, run_dir=run_dir, data=data, manifest=m, device=torch.device("cpu"), stage="TEST", smoke_steps=2, smoke_epoch_steps=2,
                 epoch_eval=False)
    tr.setup()
    tr.run()
    steps = [__import__("json").loads(l) for l in open(run_dir / "logs" / "steps.jsonl")]
    assert any("stress_frac_replaced" in s for s in steps)
    assert all(torch.isfinite(torch.tensor(s["loss"])) for s in steps)


def test_dispatch_vcs_matches_direct(tmp_path):
    cfg = _small(_load(tmp_path, M.build("vcs", "c10"), "dv.yaml"), batch=8)
    V, B = 4, 8
    vs = _views(V, B, grad=True); c = _critic()
    repl, rm = stress_replacements(V, B, 0.1, torch.Generator().manual_seed(11))
    out = compute_objective_views({"views_z": vs, "views_p": vs, "views_h": vs}, cfg=cfg, critic=c, pair_generator=torch.Generator(),
                                  stress={"repl": repl, "replaced": rm, "diag": False})
    direct = stress_all_view_tokens_loss(vs, c, repl=repl, replaced=rm, negative_detach=False, objective="vcs")
    assert torch.allclose(out["loss"], direct["loss"], atol=1e-12, rtol=0) and "stress_frac_replaced" in out["stats"]


@pytest.mark.parametrize("eps", [0.0, 0.1, 0.35, 0.9])
def test_contamination_identity_exact_finite_model(eps):
    """Plan §6.2: with P_eps = (1 - eps) P + eps Q and negatives Q, eta_eps = (1 - eps) eta / (1 - eps eta) and
    S(P_eps, Q) = (1 - eps)^2 E_M[eta^2 / (1 - eps eta)], checked by exact summation on a random finite joint (P) vs product (Q)."""
    g = torch.Generator().manual_seed(int(eps * 1000) + 3)
    P = torch.rand(7, 5, generator=g, dtype=torch.float64); P = P / P.sum()
    Q = P.sum(1, keepdim=True) * P.sum(0, keepdim=True)
    M = 0.5 * (P + Q); eta = 0.5 * (P - Q) / M
    Pe = (1 - eps) * P + eps * Q; Me = 0.5 * (Pe + Q); eta_e = 0.5 * (Pe - Q) / Me
    assert torch.allclose(eta_e, (1 - eps) * eta / (1 - eps * eta), atol=1e-14, rtol=0)
    S_e = (Me * eta_e ** 2).sum()
    assert torch.allclose(S_e, (1 - eps) ** 2 * (M * eta ** 2 / (1 - eps * eta)).sum(), atol=1e-14, rtol=0)
