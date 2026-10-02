"""P107 (package v4 module A next batch; owner 2026-10-01): A-L1 (matched JS on the fixed scorer f = 2s − 1), A-L2 (matched JS, a, b learned
from (2, −1)), A-P1 (fixed scorer, full negative gradient), A-P2 (fixed scorer, all view tokens), A-P3 (all view tokens + full gradient).
Synthetic data only."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO, REPO / "tests", REPO / "configs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from reference.ssl_core import cyclic_negative_indices  # noqa: E402
from vcs_ssl.checkpoint import load_checkpoint  # noqa: E402
from vcs_ssl.config import ConfigError, load_config  # noqa: E402
from vcs_ssl.models.critic import CosineCritic, FixedCosineCritic, build_critic, critic_impl_name  # noqa: E402
from vcs_ssl.objectives import all_view_tokens_loss, compute_objective_views, js_matched_pair_loss_negdetach  # noqa: E402
from test_vcs_ssl_p104 import _views, dense_all_view_J, variant  # noqa: E402
from test_vcs_ssl_v2 import _base, _env, _load, _small, run_trainer, synthetic  # noqa: E402

D = 32
NAMES = ("A-L1", "A-L2", "A-P1", "A-P2", "A-P3")


def p107_variant(name: str) -> dict:
    """The five v4 cells on the small pilot base (same option set as configs/make_p107_configs.py)."""
    d = variant(_base(), "G2")
    c, p, o = d["model"]["critic"], d["pairing"], d["objective"]
    if name == "A-L2":
        c["affine_mode"] = "learned"
    if name in ("A-L1", "A-L2"):
        o["loss"] = "js_matched_logistic"
    if name == "A-L1":
        o["js_fixed_scorer"] = True
    if name in ("A-P1", "A-P3"):
        p["negative_detach"] = False
    if name in ("A-P2", "A-P3"):
        p["pair_scope"] = "all_view_tokens"; p["all_view_chunk"] = 16
    return d


def _cfg(tmp_path, name, **kw) -> dict:
    return _small(_load(tmp_path, p107_variant(name), f"{name.replace('-', '')}.yaml"), **kw)


# ----------------------------------------------------------------------------------------------------------------- JS on the fixed scorer
def test_js_fixed_scorer_hand_formula_and_no_critic_grad():
    l, r = _views(V=2, B=7, grad=True, dtype=torch.float64)
    crit = FixedCosineCritic(D, scale=2.0, bias=-1.0).double()
    g = torch.Generator().manual_seed(3)
    s, sh = js_matched_pair_loss_negdetach(l, r, crit, k=3, generator=g, negative_detach=True)
    f_pos = 2.0 * (l * r).sum(-1) - 1.0
    k = 3; idx, sh2 = cyclic_negative_indices(7, k, generator=torch.Generator().manual_seed(3), device=l.device)
    assert [int(x) for x in sh2] == [int(x) for x in sh]
    left = l.unsqueeze(0).expand(k, -1, -1).reshape(-1, D); right = r.detach()[idx].reshape(-1, D)
    f_neg = 2.0 * (left * right).sum(-1) - 1.0
    ref = F.softplus(-2 * f_pos).mean() + F.softplus(2 * f_neg).mean()
    assert torch.allclose(s["loss"], ref, atol=1e-12)
    s["loss"].backward()
    assert l.grad.abs().sum() > 0 and r.grad.abs().sum() > 0 and list(crit.parameters()) == []


def test_js_fixed_matches_learned_critic_at_same_init_on_first_step(tmp_path):
    """A-L1 and A-L2 start from the identical scorer; their first-step losses and encoder-input gradients coincide."""
    vs1 = _views(grad=True, dtype=torch.float64); vs2 = [v.detach().clone().requires_grad_(True) for v in vs1]
    c1, c2 = _cfg(tmp_path, "A-L1"), _cfg(tmp_path, "A-L2")
    k1, k2 = build_critic(c1["model"]["critic"], feature_dim=D).double(), build_critic(c2["model"]["critic"], feature_dim=D).double()
    assert isinstance(k1, FixedCosineCritic) and isinstance(k2, CosineCritic) and float(k2.scale) == 2.0 and float(k2.bias) == -1.0
    o1 = compute_objective_views({"views_z": vs1}, cfg=c1, critic=k1, pair_generator=torch.Generator().manual_seed(5))
    o2 = compute_objective_views({"views_z": vs2}, cfg=c2, critic=k2, pair_generator=torch.Generator().manual_seed(5))
    assert torch.allclose(o1["loss"], o2["loss"], atol=1e-12) and "js_loss" in o1["stats"]
    g1 = torch.autograd.grad(o1["loss"], vs1); g2 = torch.autograd.grad(o2["loss"], vs2, retain_graph=True)
    assert all(torch.allclose(a, b, atol=1e-12) for a, b in zip(g1, g2))
    assert all(p.grad is None for p in k2.parameters())
    gp = torch.autograd.grad(o2["loss"], list(k2.parameters()))
    assert all(x.abs() > 0 for x in gp)  # A-L2's a, b are trained


# ----------------------------------------------------------------------------------------------------------------- A-P1 / A-P2 / A-P3
def test_AP1_full_gradient_reaches_negatives_vs_G2_detach(tmp_path):
    cfg_full, cfg_det = _cfg(tmp_path, "A-P1"), _load(tmp_path, variant(_base(), "G2"), "G2.yaml")
    crit = FixedCosineCritic(D, scale=2.0, bias=-1.0).double()
    vs = _views(grad=True, dtype=torch.float64)
    full = compute_objective_views({"views_z": vs}, cfg=cfg_full, critic=crit, pair_generator=torch.Generator().manual_seed(2))["loss"]
    det = compute_objective_views({"views_z": vs}, cfg=cfg_det, critic=crit, pair_generator=torch.Generator().manual_seed(2))["loss"]
    assert torch.allclose(full, det, atol=1e-12)  # same value, different routing
    gf, gd = torch.autograd.grad(full, vs), torch.autograd.grad(det, vs)
    assert any(not torch.allclose(a, b, atol=1e-9) for a, b in zip(gf, gd))


@pytest.mark.parametrize("V,B", [(4, 6), (3, 5)])
def test_AP2_AP3_counts_value_and_key_gradients(V, B):
    vs = _views(V=V, B=B, grad=True, dtype=torch.float64)
    crit = FixedCosineCritic(D, scale=2.0, bias=-1.0).double()
    det = all_view_tokens_loss(vs, crit, negative_detach=True, chunk_size=5)    # A-P2
    full = all_view_tokens_loss(vs, crit, negative_detach=False, chunk_size=5)  # A-P3
    assert (det["n_pos"], det["n_neg"]) == (V * B * (V - 1), V * B * V * (B - 1)) == (full["n_pos"], full["n_neg"])
    for nd, out in ((True, det), (False, full)):
        ref, n_p, n_q = dense_all_view_J(vs, crit, negative_detach=nd)
        assert torch.allclose(out["loss"], ref["loss"], atol=1e-12)
        g, gr = torch.autograd.grad(out["loss"], vs, retain_graph=True), torch.autograd.grad(ref["loss"], vs)
        assert all(torch.allclose(a, b, atol=1e-12) for a, b in zip(g, gr))
    # key-side gradient: A-P3 (full) routes gradient through the Q keys, A-P2 (detach) does not -> input gradients differ, while the
    # P part is shared; the detached Q gradient is exactly half of the full one (each unordered negative appears twice in Q)
    g_det = torch.autograd.grad(det["loss"], vs, retain_graph=True); g_full = torch.autograd.grad(full["loss"], vs, retain_graph=True)
    p_only = -(det["t_pos_mean"] - 0.5 * det["t_pos_second"]); g_p = torch.autograd.grad(p_only, vs)
    assert any((a - b).abs().sum() > 1e-9 for a, b in zip(g_full, g_det))
    for i in range(V):
        assert torch.allclose(g_det[i] - g_p[i], 0.5 * (g_full[i] - g_p[i]), atol=1e-12)


# ----------------------------------------------------------------------------------------------------------------- policy / configs
def test_policy_marker_required_and_refusals(tmp_path):
    for n in NAMES:
        _cfg(tmp_path, n)
    bad = []
    d = p107_variant("A-L1"); d["objective"].pop("js_fixed_scorer"); bad.append(d)                     # fixed + JS without the marker
    d = p107_variant("A-L2"); d["objective"]["js_fixed_scorer"] = True; bad.append(d)                   # marker on a learned scorer
    d = p107_variant("A-P1"); d["objective"]["js_fixed_scorer"] = True; bad.append(d)                   # marker without the JS loss
    d = p107_variant("A-L1"); d["pairing"]["pair_scope"] = "all_view_tokens"; d["pairing"]["all_view_chunk"] = 16; bad.append(d)  # JS + all-view
    d = p107_variant("A-L1"); d["model"]["critic"]["observation_noise_tau"] = 0.3; bad.append(d)        # noise on the fixed scorer
    d = variant(_base(), "G1"); d["objective"]["loss"] = "js_matched_logistic"; bad.append(d)          # P104 refusal kept
    for i, b in enumerate(bad):
        with pytest.raises(ConfigError):
            _load(tmp_path, b, f"bad{i}.yaml")


def test_generated_configs_hashes_and_routing(tmp_path):
    man = REPO / "configs" / "P107_SHA256.json"
    if not man.is_file():
        pytest.skip("configs not generated")
    want = {"A-L1": ("FixedCosineCritic", "js_matched_logistic", "cross_view_k", True),
            "A-L2": ("CosineCritic", "js_matched_logistic", "cross_view_k", True),
            "A-P1": ("FixedCosineCritic", "negative_J", "cross_view_k", False),
            "A-P2": ("FixedCosineCritic", "negative_J", "all_view_tokens", True),
            "A-P3": ("FixedCosineCritic", "negative_J", "all_view_tokens", False),
            # P107 addendum 1: same-initialisation learned controls of the selected A-P2 / A-P3
            "A-P2F": ("CosineCritic", "negative_J", "all_view_tokens", True),
            "A-P3F": ("CosineCritic", "negative_J", "all_view_tokens", False)}
    for f, rec in json.loads(man.read_text())["configs"].items():
        p = REPO / "configs" / f
        assert hashlib.sha256(p.read_bytes()).hexdigest() == rec["sha256"]
        cfg = load_config(p, env=_env(tmp_path))
        impl, loss, scope, nd = want[rec["logical_variant"]]
        assert critic_impl_name(cfg["model"]["critic"]).endswith(impl) and cfg["objective"]["loss"] == loss
        assert cfg["pairing"].get("pair_scope", "cross_view_k") == scope and cfg["pairing"]["negative_detach"] is nd
        c = cfg["model"]["critic"]
        assert float(c["cosine_scale_init"]) == 2.0 and float(c["cosine_bias_init"]) == -1.0
        parent = yaml.safe_load((REPO / "configs" / f"cifar10_hpX_{rec['parent_p104_cell']}_views4_800ep_seed{rec['seed']}.yaml").read_text()) \
            if (REPO / "configs" / f"cifar10_hpX_{rec['parent_p104_cell']}_views4_800ep_seed{rec['seed']}.yaml").is_file() else None
        if parent is not None:
            mine = yaml.safe_load(p.read_text())
            assert mine["model"] == parent["model"] and mine["train"] == parent["train"] and mine["optimizer"] == parent["optimizer"]


# ----------------------------------------------------------------------------------------------------------------- trainer
@pytest.mark.parametrize("name", NAMES)
def test_trainer_smoke_variants(tmp_path, name):
    data, m = synthetic()
    cfg = _cfg(tmp_path, name, epochs=2)
    tr, st = run_trainer(cfg, data, m, run_id=f"t{name}", smoke_steps=4, smoke_epoch_steps=2, epoch_eval=False)
    assert st == "COMPLETED", tr.failure_reason
    steps = [json.loads(l) for l in (tr.run_dir / "logs" / "steps.jsonl").read_text().splitlines() if l.strip()]
    assert all(r["J_raw"] is not None and abs(r["J_raw"]) < 3 for r in steps)
    rm = json.loads((tr.run_dir / "run_manifest.json").read_text())
    fixed = name != "A-L2"
    assert rm["hparams"]["trainable_affine"] is (not fixed) and rm["hparams"]["critic_trainable_params"] == (0 if fixed else 2)
    ck = load_checkpoint(tr.run_dir / "checkpoints" / "last.pt")
    if fixed:
        assert float(ck["critic_state"]["scale"]) == 2.0 and float(ck["critic_state"]["bias"]) == -1.0


@pytest.mark.parametrize("name", NAMES)
def test_stop_resume_identical(tmp_path, name):
    data, m = synthetic()
    cfg = _cfg(tmp_path, name, epochs=2)
    tr, st = run_trainer(cfg, data, m, run_id=f"s{name}", smoke_steps=4, smoke_epoch_steps=2, epoch_eval=False, stop_after_steps=2)
    assert st == "STOPPED_BUDGET"
    tr2, st2 = run_trainer(cfg, data, m, run_id=f"s{name}", smoke_steps=4, smoke_epoch_steps=2, epoch_eval=False,
                           resume_from=tr.run_dir / "checkpoints" / "last.pt")
    assert st2 == "COMPLETED" and tr2.step == 4
    tr3, st3 = run_trainer(cfg, data, m, run_id=f"u{name}", smoke_steps=4, smoke_epoch_steps=2, epoch_eval=False)
    a = load_checkpoint(tr2.run_dir / "checkpoints" / "last.pt"); b = load_checkpoint(tr3.run_dir / "checkpoints" / "last.pt")
    for part in ("critic_state", "encoder_state", "projector_state"):
        for kk in a[part]:
            assert torch.allclose(a[part][kk].float(), b[part][kk].float(), atol=1e-5), (part, kk)
