"""P95 (package v1 estimator improvements inside full SSL; owner 2026-09-29): residual cosine+MLP critic, simplex dictionary critic,
observation-noise cosine critic, online cosine-critic refresh, and the matched balanced-logistic (JS) control.  Synthetic data only."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO, REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from reference.ssl_core import vcs_from_scores  # noqa: E402
from vcs_ssl.checkpoint import load_checkpoint  # noqa: E402
from vcs_ssl.config import SCHEMA, ConfigError, load_config, policy_checks  # noqa: E402
from vcs_ssl.models.critic import (CosineCritic, DictionarySimplexCritic, NoisyCosineCritic, ResidualCosineMLPCritic,  # noqa: E402
                                   build_critic, critic_impl_name)
from vcs_ssl.objectives import compute_objective_views, js_matched_pair_loss_negdetach, vcs_pair_loss_negdetach  # noqa: E402
from vcs_ssl.utils import sha256_json  # noqa: E402
from test_vcs_ssl_v2 import _base, _env, _load, _small, run_trainer, synthetic  # noqa: E402

D = 128


def cosine(d: dict) -> dict:
    c = d["model"]["critic"]
    c["input"] = "cosine"; c["cosine_scale_init"] = 5.0; c["hidden_dims"] = [512, 512]
    d["pairing"]["negative_detach"] = True
    return d


def variant(d: dict, name: str) -> dict:
    d = cosine(d); c = d["model"]["critic"]
    if name == "residual":
        c["input"] = "residual_cosine_mlp"; c["residual_lambda"] = 0.25
    elif name == "dictionary":
        c["input"] = "dictionary_simplex"
    elif name == "noise":
        c["observation_noise_tau"] = 0.3
    elif name == "refresh":
        c["refresh_every_epochs"] = 1; c["refresh_batches"] = 2
    elif name == "js":
        d["objective"]["loss"] = "js_matched_logistic"
    return d


def _crit_cfg(**kw) -> dict:
    c = copy.deepcopy(_base()["model"]["critic"]); c.update({"input": "cosine", "cosine_scale_init": 5.0, "hidden_dims": [64, 64]}); c.update(kw)
    return c


def _z(n=48, seed=0, grad=False):
    g = torch.Generator().manual_seed(seed)
    a, b = F.normalize(torch.randn(n, D, generator=g), dim=1), F.normalize(torch.randn(n, D, generator=g), dim=1)
    return a.requires_grad_(grad), b.requires_grad_(grad)


# ----------------------------------------------------------------------------------------------------------------- critics
def test_residual_and_dictionary_bounded_grads_and_weights():
    torch.manual_seed(0)
    r = ResidualCosineMLPCritic(D, [64, 64], lam=0.5, scale_init=50.0)
    q = DictionarySimplexCritic(D, [64, 64], scale_init=50.0)
    for crit in (r, q):
        l, rt = _z(grad=True)
        t = crit(l * 3, rt * 3)  # large inputs push every member towards saturation
        assert t.abs().max() <= 1.0 + 1e-6
        t.sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in crit.parameters() if p.requires_grad)
        assert l.grad is not None and l.grad.abs().sum() > 0
    w = q.weights()
    assert torch.allclose(w.sum(), torch.tensor(1.0)) and torch.allclose(w, torch.full((3,), 1 / 3))
    assert q.theta.grad is not None and q.theta.grad.abs().sum() > 0
    st = q.pop_stats()
    assert st["dict_disagreement_D"] >= -1e-6 and abs(sum(st[f"dict_w_{m}"] for m in q.MEMBERS) - 1) < 1e-6
    sr = r.pop_stats()
    assert sr["res_lambda"] == 0.5 and "res_mlp_part_mean" in sr
    # exact convex-combination identity on one call
    q2 = DictionarySimplexCritic(D, [64, 64], scale_init=5.0)
    with torch.no_grad():
        q2.theta.copy_(torch.tensor([0.3, -1.0, 0.7]))
    l, rt = _z()
    ts = torch.stack((q2.cos(l, rt), q2.mlp(l, rt), q2.bil(l, rt)), -1)
    t = q2(l, rt); w = q2.weights()
    assert torch.allclose(t, ts @ w, atol=1e-6)
    s = q2.pop_stats()
    D_manual = float((ts.square().mean(0) * w).sum() - t.square().mean())
    assert s["dict_disagreement_D"] == pytest.approx(D_manual, abs=1e-6) and D_manual >= 0
    # eval mode: no statistics accumulated
    q2.eval(); q2(l, rt); assert q2._n_calls == 0


def test_noise_scale_counter_and_eval_clean():
    torch.manual_seed(1)
    crit = NoisyCosineCritic(D, scale_init=5.0, tau=0.3)
    l, r = _z(n=4096)
    e = crit._noise(l)
    assert e.std().item() == pytest.approx(0.3 / D ** 0.5, rel=0.02)
    assert e.square().sum(-1).mean().sqrt().item() == pytest.approx(0.3, rel=0.02)
    c0 = int(crit.noise_calls)
    crit.train(); t1 = crit(l, r); assert int(crit.noise_calls) == c0 + 2
    crit.eval(); t_eval = crit(l, r)
    ref = torch.tanh(crit.scale * (l * r).sum(-1) + crit.bias)
    assert torch.equal(t_eval, ref) and not torch.allclose(t1, ref)
    # dot-product perturbation variance (2 tau^2 + tau^4)/d for unit vectors (v2 §10.1)
    crit.train()
    u = l + crit._noise(l); v = r + crit._noise(r)
    var = ((u * v).sum(-1) - (l * r).sum(-1)).var().item()
    assert var == pytest.approx((2 * 0.09 + 0.3 ** 4) / D, rel=0.1)
    # reproducible from state: same seed + counter -> same noise
    st = copy.deepcopy(crit.state_dict()); a = crit._noise(l)
    crit2 = NoisyCosineCritic(D, scale_init=5.0, tau=0.3); crit2.load_state_dict(st)
    assert torch.equal(crit2._noise(l), a)
    s = crit.pop_stats(); assert s["noise_total_rms"] == 0.3 and s["noise_coordinate_sd"] == pytest.approx(0.3 / D ** 0.5)


def test_state_dict_round_trips_and_builder():
    for kw, cls in (({"input": "residual_cosine_mlp", "residual_lambda": 0.5}, ResidualCosineMLPCritic),
                    ({"input": "dictionary_simplex"}, DictionarySimplexCritic),
                    ({"observation_noise_tau": 0.1}, NoisyCosineCritic), ({}, CosineCritic)):
        c = _crit_cfg(**kw)
        torch.manual_seed(3); a = build_critic(c, feature_dim=D)
        assert type(a) is cls and critic_impl_name(c).endswith(cls.__name__)
        torch.manual_seed(4); b = build_critic(c, feature_dim=D)
        b.load_state_dict(a.state_dict())
        l, r = _z(); a.eval(); b.eval()
        assert torch.equal(a(l, r), b(l, r))
    # recipe cosine critic: identical init / keys as before (noise variant shares scale/bias keys)
    torch.manual_seed(5); cc = build_critic(_crit_cfg(), feature_dim=D)
    assert set(cc.state_dict()) == {"scale", "bias"} and float(cc.scale) == 5.0
    nk = set(build_critic(_crit_cfg(observation_noise_tau=0.1), feature_dim=D).state_dict())
    assert nk == {"scale", "bias", "noise_seed", "noise_calls"}


# ----------------------------------------------------------------------------------------------------------------- JS control
def test_js_matched_gradient_scale_at_zero_and_shared_pairs():
    crit = CosineCritic(D, scale_init=0.0)  # f = 0 everywhere at init (a = 0, b = 0)
    z1, z2 = _z(grad=True)
    g1 = torch.Generator().manual_seed(9); g2 = torch.Generator().manual_seed(9)
    sj, shj = js_matched_pair_loss_negdetach(z1, z2, crit, k=8, generator=g1)
    sv, shv = vcs_pair_loss_negdetach(z1, z2, crit, k=8, generator=g2)
    assert torch.equal(shj, shv)  # identical shift draw from the same generator state
    ga = torch.autograd.grad(sj["loss"], [crit.scale, crit.bias], retain_graph=True)
    gb = torch.autograd.grad(sv["loss"], [crit.scale, crit.bias], retain_graph=True)
    for x, y in zip(ga, gb):
        assert torch.allclose(x, y, atol=1e-6)  # dL_JS/df = dL_V/df = -1 (P), +1 (Q) at f = 0
    gz = torch.autograd.grad(sj["loss"], z1, retain_graph=True)[0]; gzv = torch.autograd.grad(sv["loss"], z1)[0]
    assert torch.allclose(gz, gzv, atol=1e-6)
    assert float(sj["J_raw"]) == pytest.approx(0.0, abs=1e-7) and float(sj["loss"]) == pytest.approx(2 * np.log(2), abs=1e-6)


def test_views_objective_dispatch_and_stats(tmp_path):
    for name in ("residual", "dictionary", "noise", "js"):
        d = variant(_base(), name); d["views"]["count"] = 4
        cfg = _small(_load(tmp_path, d, f"{name}.yaml"))
        torch.manual_seed(0); crit = build_critic(cfg["model"]["critic"], feature_dim=D); crit.train()
        vs = [F.normalize(torch.randn(16, D), dim=1).requires_grad_(True) for _ in range(4)]
        out = compute_objective_views({"views_z": vs}, cfg=cfg, critic=crit, pair_generator=torch.Generator().manual_seed(1))
        out["loss"].backward()
        assert torch.isfinite(out["loss"]) and vs[0].grad.abs().sum() > 0 and -3 <= out["stats"]["J_raw"] <= 1
        if name == "js":
            assert out["stats"]["js_loss"] == pytest.approx(float(out["loss"]))
        if name == "dictionary":
            assert out["stats"]["dict_disagreement_D"] >= -1e-6 and "dict_w_cosine" in out["stats"]
        if name == "noise":
            assert out["stats"]["noise_coord_sd_emp"] == pytest.approx(0.3 / D ** 0.5, rel=0.1)


# ----------------------------------------------------------------------------------------------------------------- config policy / hash
def test_config_policy_and_old_config_hash_unchanged(tmp_path):
    for name in ("residual", "dictionary", "noise", "refresh", "js"):
        policy_checks(_load(tmp_path, variant(_base(), name), f"p{name}.yaml"))
    bad = []
    d = variant(_base(), "noise"); d["model"]["critic"]["input"] = "ordered_concat"; bad.append(d)
    d = variant(_base(), "refresh"); d["model"]["critic"]["input"] = "residual_cosine_mlp"; d["model"]["critic"]["residual_lambda"] = 0.5; bad.append(d)
    d = variant(_base(), "js"); d["model"]["critic"]["observation_noise_tau"] = 0.1; bad.append(d)
    d = variant(_base(), "residual"); d["model"]["critic"]["residual_lambda"] = 1.0; bad.append(d)
    d = cosine(_base()); d["model"]["critic"]["residual_lambda"] = 0.5; bad.append(d)
    d = variant(_base(), "js"); d["run"]["method"] = "simclr_matched"; bad.append(d)
    for i, b in enumerate(bad):
        with pytest.raises(ConfigError):
            _load(tmp_path, b, f"bad{i}.yaml")
    # P95 fields are not inserted into configs that do not carry them: the resolved dict and config_hash of every frozen config are unchanged
    for f in ("residual_lambda", "observation_noise_tau", "refresh_every_epochs", "refresh_batches"):
        assert SCHEMA["model"]["critic"][f].fill is False
    cfg = _load(tmp_path, _base(), "old.yaml")
    assert not any(k in cfg["model"]["critic"] for k in ("residual_lambda", "observation_noise_tau", "refresh_every_epochs", "refresh_batches"))
    recipe = REPO / "configs" / "cifar10_hpK_a5_views4_800ep_vcs_seed0.yaml"
    if recipe.is_file():
        rc = load_config(recipe, env=_env(tmp_path))
        assert "residual_lambda" not in rc["model"]["critic"]


# ----------------------------------------------------------------------------------------------------------------- trainer
@pytest.mark.parametrize("name", ["residual", "dictionary", "noise", "refresh", "js"])
def test_trainer_smoke_variants(tmp_path, name):
    data, m = synthetic()
    d = variant(_base(), name); d["views"]["count"] = 4
    cfg = _small(_load(tmp_path, d, f"t{name}.yaml"), epochs=2)
    tr, st = run_trainer(cfg, data, m, run_id=f"t{name}", smoke_steps=4, smoke_epoch_steps=2, epoch_eval=False)
    assert st == "COMPLETED", tr.failure_reason
    steps = [json.loads(l) for l in (tr.run_dir / "logs" / "steps.jsonl").read_text().splitlines() if l.strip()]
    assert all(r["J_raw"] is not None for r in steps)
    rm = json.loads((tr.run_dir / "run_manifest.json").read_text())
    assert rm["hparams"]["objective_loss"] == cfg["objective"]["loss"]
    if name == "refresh":
        rec = [json.loads(l) for l in (tr.run_dir / "logs" / "critic_refresh.jsonl").read_text().splitlines() if l.strip()]
        assert len(rec) == 1 and rec[0]["epoch"] == 1 and rec[0]["J_after"] >= rec[0]["J_before"] - 1e-12
    if name == "dictionary":
        assert all("dict_w_cosine" in r for r in steps)


def test_refresh_changes_only_critic_and_keeps_rng(tmp_path):
    data, m = synthetic()
    d = variant(_base(), "refresh"); d["views"]["count"] = 4
    cfg = _small(_load(tmp_path, d, "rr.yaml"), epochs=2)
    run_dir = Path(cfg["run"]["output_root"]) / "rr"; run_dir.mkdir(parents=True, exist_ok=True)
    from vcs_ssl.train import Trainer
    from vcs_ssl.checkpoint import capture_rng, rng_fingerprint
    tr = Trainer(cfg, run_dir=run_dir, data=data, manifest=m, device=torch.device("cpu"), stage="TEST", smoke_steps=4, smoke_epoch_steps=2, epoch_eval=False)
    tr.setup()
    with torch.no_grad():
        tr.critic.scale.fill_(0.5); tr.critic.bias.fill_(0.3)  # a deliberately poor (a, b)
    enc = {k: v.clone() for k, v in tr.encoder.state_dict().items()}; proj = {k: v.clone() for k, v in tr.projector.state_dict().items()}
    before = rng_fingerprint(capture_rng(tr.device, tr.loader_gen, tr.pair_gen))
    rec = tr.refresh_cosine_critic(1)
    after = rng_fingerprint(capture_rng(tr.device, tr.loader_gen, tr.pair_gen))
    assert before == after
    assert all(torch.equal(enc[k], v) for k, v in tr.encoder.state_dict().items())  # parameters AND BN buffers bit-identical
    assert all(torch.equal(proj[k], v) for k, v in tr.projector.state_dict().items())
    assert rec["applied"] and rec["J_after"] > rec["J_before"] and (rec["a_after"], rec["b_after"]) != (0.5, 0.3)
    assert float(tr.critic.scale) == pytest.approx(rec["a_after"])


@pytest.mark.parametrize("name", ["dictionary", "noise", "refresh"])
def test_stop_resume_variants(tmp_path, name):
    data, m = synthetic()
    d = variant(_base(), name); d["views"]["count"] = 4
    cfg = _small(_load(tmp_path, d, f"s{name}.yaml"), epochs=2)
    tr, st = run_trainer(cfg, data, m, run_id=f"s{name}", smoke_steps=4, smoke_epoch_steps=2, epoch_eval=False, stop_after_steps=2)
    assert st == "STOPPED_BUDGET"
    ck = load_checkpoint(tr.run_dir / "checkpoints" / "last.pt")
    tr2, st2 = run_trainer(cfg, data, m, run_id=f"s{name}", smoke_steps=4, smoke_epoch_steps=2, epoch_eval=False, resume_from=tr.run_dir / "checkpoints" / "last.pt")
    assert st2 == "COMPLETED" and tr2.step == 4
    if name == "dictionary":
        assert "theta" in ck["critic_state"]
    if name == "noise":
        assert int(load_checkpoint(tr.run_dir / "checkpoints" / "last.pt")["critic_state"]["noise_calls"]) > int(ck["critic_state"]["noise_calls"]) > 0
    # uninterrupted run with the same config reaches the same critic state (determinism incl. the new state)
    tr3, st3 = run_trainer(cfg, data, m, run_id=f"u{name}", smoke_steps=4, smoke_epoch_steps=2, epoch_eval=False)
    a = load_checkpoint(tr.run_dir / "checkpoints" / "last.pt")["critic_state"]; b = load_checkpoint(tr3.run_dir / "checkpoints" / "last.pt")["critic_state"]
    for k in a:
        assert torch.allclose(a[k].float(), b[k].float(), atol=1e-5), k
