"""P104 (package v3 first batch; owner 2026-09-30): G line (fixed angular affine, full negative routing), U line (all view tokens),
N line (noise-draw average of losses, noisy matched-JS control).  Covers the package v3 §8 test table.  Synthetic data only."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO, REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from reference.ssl_core import vcs_from_scores  # noqa: E402
from vcs_ssl.checkpoint import capture_rng, load_checkpoint, rng_fingerprint  # noqa: E402
from vcs_ssl.config import SCHEMA, ConfigError, load_config  # noqa: E402
from vcs_ssl.models.critic import CosineCritic, FixedCosineCritic, NoisyCosineCritic, build_critic, critic_impl_name  # noqa: E402
from vcs_ssl.objectives import (all_view_tokens_loss, compute_objective_views, compute_objective_views_p104,  # noqa: E402
                                js_matched_pair_loss_negdetach, noisy_pair_loss_repeats, vcs_pair_loss, vcs_pair_loss_negdetach)
from vcs_ssl.optim import build_optimizer  # noqa: E402
from test_vcs_ssl_v2 import _base, _env, _load, _small, run_trainer, synthetic  # noqa: E402

D = 32
NAMES = ("G1", "G2", "G3", "G4", "U1", "U2", "N1", "N2")


def cosine(d: dict) -> dict:
    c = d["model"]["critic"]
    c["input"] = "cosine"; c["cosine_scale_init"] = 5.0; c["hidden_dims"] = [512, 512]
    d["pairing"]["negative_detach"] = True; d["views"]["count"] = 4
    return d


def variant(d: dict, name: str) -> dict:
    """The eight v3 cells on the small pilot base (same option set as configs/make_p104_configs.py)."""
    d = cosine(d); c, p = d["model"]["critic"], d["pairing"]
    c["cosine_bias_init"] = 0.0
    if name in ("G1", "G3", "U2"):
        c["affine_mode"] = "fixed"; c["cosine_scale_init"] = 1.0
    if name == "G2":
        c["affine_mode"] = "fixed"; c["cosine_scale_init"] = 2.0; c["cosine_bias_init"] = -1.0
    if name in ("G3", "G4"):
        p["negative_detach"] = False
    if name in ("U1", "U2"):
        p["pair_scope"] = "all_view_tokens"; p["all_view_chunk"] = 16
    if name in ("N1", "N2"):
        c["observation_noise_tau"] = 0.3; c["noise_repeats"] = 4 if name == "N1" else 1; c["noise_eval_repeats"] = 3
    if name == "N2":
        d["objective"]["loss"] = "js_matched_logistic"
    return d


def _cfg(tmp_path, name, **kw) -> dict:
    return _small(_load(tmp_path, variant(_base(), name), f"{name}.yaml"), **kw)


def _views(V=4, B=6, seed=0, grad=False, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    return [F.normalize(torch.randn(B, D, generator=g, dtype=dtype), dim=1).requires_grad_(grad) for _ in range(V)]


def dense_all_view_J(views, crit, *, negative_detach):
    """Brute-force reference for the U line: explicit ordered pair lists, P and Q averaged over their own counts."""
    V, B = len(views), views[0].shape[0]
    tok = [(v, i) for v in range(V) for i in range(B)]
    pl, pr, ql, qr = [], [], [], []
    for (v, i) in tok:
        for (w, j) in tok:
            if (v, i) == (w, j):
                continue
            if i == j:
                pl.append(views[v][i]); pr.append(views[w][j])
            else:
                ql.append(views[v][i]); qr.append(views[w][j].detach() if negative_detach else views[w][j])
    tp = crit(torch.stack(pl), torch.stack(pr)); tq = crit(torch.stack(ql), torch.stack(qr))
    return vcs_from_scores(tp, tq), len(pl), len(ql)


# ----------------------------------------------------------------------------------------------------------------- identities
def test_risk_identity_with_unequal_counts():
    g = torch.Generator().manual_seed(0)
    tp, tq = torch.tanh(torch.randn(7, generator=g)), torch.tanh(torch.randn(53, generator=g))
    s = vcs_from_scores(tp, tq)
    assert float(s["J_raw"]) == pytest.approx(1.0 - float(s["R_binary"]), abs=1e-6)
    out = all_view_tokens_loss(_views(V=3, B=5), CosineCritic(D, scale_init=3.0), negative_detach=True, chunk_size=4)
    assert out["n_pos"] != out["n_neg"]
    assert float(out["J_raw"]) == pytest.approx(1.0 - float(out["R_binary"]), abs=1e-6)


def test_noise_average_identity():
    """J(mean_r T_r) − mean_r J(T_r) = ½ (E_P Var_r T + E_Q Var_r T) >= 0, and the N1 loss is the MEAN OF PER-DRAW losses."""
    torch.manual_seed(0); crit = NoisyCosineCritic(D, scale_init=5.0, tau=0.3); crit.train()
    z1, z2 = _views(V=2, B=12)
    R, k = 5, 4
    st0 = copy.deepcopy(crit.state_dict())
    out, shifts = noisy_pair_loss_repeats(z1, z2, crit, k=k, generator=torch.Generator().manual_seed(3), negative_detach=True, repeats=R)
    # replay the same counter stream by hand
    crit.load_state_dict(st0)
    from vcs_ssl.objectives import cyclic_negative_indices
    idx, sh = cyclic_negative_indices(len(z1), k, generator=torch.Generator().manual_seed(3), device=z1.device)
    assert torch.equal(sh, shifts)
    left = z1.unsqueeze(0).expand(k, -1, -1).reshape(-1, D); right = z2[idx].reshape(-1, D)
    TP, TQ = [], []
    for _ in range(R):
        TP.append(torch.tanh(crit.logits(z1, z2))); TQ.append(torch.tanh(crit.logits(left, right)))
    TP, TQ = torch.stack(TP), torch.stack(TQ)
    per = torch.stack([vcs_from_scores(TP[r], TQ[r])["J_raw"] for r in range(R)])
    assert float(out["loss"]) == pytest.approx(-float(per.mean()), abs=1e-6)
    gap = float(vcs_from_scores(TP.mean(0), TQ.mean(0))["J_raw"] - per.mean())
    var = 0.5 * (TP.var(0, unbiased=False).mean() + TQ.var(0, unbiased=False).mean())
    assert gap == pytest.approx(float(var), abs=1e-6) and gap > 0


# ----------------------------------------------------------------------------------------------------------------- G line
def test_fixed_critic_no_params_encoder_grads_and_round_trip():
    c = FixedCosineCritic(D, scale=2.0, bias=-1.0)
    assert list(c.parameters()) == [] and set(c.state_dict()) == {"scale", "bias"} and c.trainable_affine is False
    c2 = FixedCosineCritic(D, scale=7.0, bias=0.0); c2.load_state_dict(c.state_dict())
    l, r = _views(V=2, B=9, grad=True)
    assert torch.equal(c(l, r), c2(l, r)) and float(c2.scale) == 2.0 and float(c2.bias) == -1.0
    assert torch.allclose(c(l, r), torch.tanh(2.0 * (l * r).sum(-1) - 1.0))
    assert torch.allclose(c.score_matrix(l @ r.T).diagonal(), c(l, r))
    s = vcs_pair_loss(l, r, c, k=3, generator=torch.Generator().manual_seed(0))[0]
    s["loss"].backward()
    assert l.grad.abs().sum() > 0 and r.grad.abs().sum() > 0
    enc, proj = torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)
    ocfg = _base()["optimizer"]
    opt = build_optimizer(enc, proj, c, ocfg) if "lr" in ocfg else None
    if opt is not None:
        assert all(g.get("name") != "critic" for g in opt.param_groups)
    with pytest.raises(ValueError):
        FixedCosineCritic(D, scale=float("nan"), bias=0.0)


def test_builder_routes_and_learned_bias_zero_is_recipe_identical():
    base = copy.deepcopy(_base()["model"]["critic"]); base.update({"input": "cosine", "cosine_scale_init": 5.0, "hidden_dims": [64, 64]})
    torch.manual_seed(5); a = build_critic(base, feature_dim=D)
    torch.manual_seed(5); b = build_critic({**base, "cosine_bias_init": 0.0, "affine_mode": "learned"}, feature_dim=D)
    assert type(a) is type(b) is CosineCritic and set(a.state_dict()) == set(b.state_dict()) == {"scale", "bias"}
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())
    f = build_critic({**base, "affine_mode": "fixed", "cosine_scale_init": 2.0, "cosine_bias_init": -1.0}, feature_dim=D)
    assert type(f) is FixedCosineCritic and critic_impl_name({**base, "affine_mode": "fixed"}).endswith("FixedCosineCritic")
    assert (float(f.scale), float(f.bias)) == (2.0, -1.0)
    n = build_critic({**base, "observation_noise_tau": 0.3, "cosine_bias_init": 0.0}, feature_dim=D)
    assert type(n) is NoisyCosineCritic
    with pytest.raises(TypeError):
        n.score_matrix(torch.zeros(2, 2))
    with pytest.raises(TypeError):
        n.embed(torch.zeros(2, D))


def test_original_paths_unchanged_when_new_options_off(tmp_path):
    """Frozen recipe path: same routing (dispatcher returns None), same loss and gradients as before the P104 edits."""
    d = cosine(_base())
    cfg = _small(_load(tmp_path, d, "frozen.yaml"))
    assert not any(k in cfg["model"]["critic"] for k in ("affine_mode", "cosine_bias_init", "noise_repeats", "noise_eval_repeats"))
    assert not any(k in cfg["pairing"] for k in ("pair_scope", "all_view_chunk"))
    for f in ("affine_mode", "cosine_bias_init", "noise_repeats", "noise_eval_repeats"):
        assert SCHEMA["model"]["critic"][f].fill is False
    for f in ("pair_scope", "all_view_chunk"):
        assert SCHEMA["pairing"][f].fill is False
    torch.manual_seed(0); crit = build_critic(cfg["model"]["critic"], feature_dim=D)
    vs = _views(grad=True)
    assert compute_objective_views_p104({"views_z": vs}, cfg=cfg, critic=crit, pair_generator=torch.Generator().manual_seed(1)) is None
    out = compute_objective_views({"views_z": vs}, cfg=cfg, critic=crit, pair_generator=torch.Generator().manual_seed(1))
    # hand-built frozen definition: mean over the 6 view pairs of the negative-detached pair loss with one shift draw per pair
    g = torch.Generator().manual_seed(1); ls = []
    for a in range(4):
        for b in range(a + 1, 4):
            ls.append(vcs_pair_loss_negdetach(vs[a], vs[b], crit, k=cfg["pairing"]["k"], generator=g)[0]["loss"])
    ref = torch.stack(ls).mean()
    assert torch.allclose(out["loss"], ref, atol=1e-7)
    ga = torch.autograd.grad(out["loss"], vs + [crit.scale], retain_graph=True); gb = torch.autograd.grad(ref, vs + [crit.scale])
    assert all(torch.allclose(x, y, atol=1e-7) for x, y in zip(ga, gb))
    # learned cell with an explicit bias_init 0 (G4 / U1 / N*) is identical to the recipe critic when routing is unchanged
    d2 = cosine(_base()); d2["model"]["critic"]["cosine_bias_init"] = 0.0
    cfg2 = _small(_load(tmp_path, d2, "b0.yaml"))
    torch.manual_seed(0); crit2 = build_critic(cfg2["model"]["critic"], feature_dim=D)
    out2 = compute_objective_views({"views_z": vs}, cfg=cfg2, critic=crit2, pair_generator=torch.Generator().manual_seed(1))
    assert torch.equal(out2["loss"], out["loss"])


def test_full_routing_uses_reference_loss(tmp_path):
    cfg = _cfg(tmp_path, "G3")
    crit = build_critic(cfg["model"]["critic"], feature_dim=D)
    vs = _views(grad=True)
    out = compute_objective_views({"views_z": vs}, cfg=cfg, critic=crit, pair_generator=torch.Generator().manual_seed(1))
    g = torch.Generator().manual_seed(1)
    ls = [vcs_pair_loss(vs[a], vs[b], crit, k=cfg["pairing"]["k"], generator=g)[0]["loss"] for a in range(4) for b in range(a + 1, 4)]
    assert torch.allclose(out["loss"], torch.stack(ls).mean(), atol=1e-7)


# ----------------------------------------------------------------------------------------------------------------- U line
@pytest.mark.parametrize("V,B", [(3, 5), (4, 6), (2, 7)])
def test_all_view_counts_and_chunked_equals_dense(V, B):
    vs = _views(V=V, B=B, grad=True, dtype=torch.float64)
    for crit in (CosineCritic(D, scale_init=4.0, bias_init=0.3).double(), FixedCosineCritic(D, scale=2.0, bias=-1.0).double()):
        for nd in (True, False):
            ref, n_p, n_q = dense_all_view_J(vs, crit, negative_detach=nd)
            assert (n_p, n_q) == (V * B * (V - 1), V * B * V * (B - 1))
            params = [p for p in crit.parameters()]
            g_ref = torch.autograd.grad(ref["loss"], vs + params)
            for chunk in (1, 3, V * B, 1000):
                out = all_view_tokens_loss(vs, crit, negative_detach=nd, chunk_size=chunk)
                assert (out["n_pos"], out["n_neg"]) == (n_p, n_q)
                assert torch.allclose(out["loss"], ref["loss"], atol=1e-12)
                g = torch.autograd.grad(out["loss"], vs + params)
                assert all(torch.allclose(x, y, atol=1e-12) for x, y in zip(g, g_ref))


def test_all_view_detach_is_half_of_full_Q_input_gradient():
    """Q is the full ordered set, so each unordered negative appears twice; detaching the right side keeps exactly half of the Q input
    gradient (P part unchanged) while the critic-parameter gradient is identical."""
    vs = _views(V=3, B=5, grad=True, dtype=torch.float64)
    crit = CosineCritic(D, scale_init=4.0, bias_init=0.3).double()
    full = all_view_tokens_loss(vs, crit, negative_detach=False, chunk_size=4)
    det = all_view_tokens_loss(vs, crit, negative_detach=True, chunk_size=4)
    ps = list(crit.parameters())
    g_full = torch.autograd.grad(full["loss"], vs + ps, retain_graph=True)
    g_det = torch.autograd.grad(det["loss"], vs + ps, retain_graph=True)
    p_only = -(det["t_pos_mean"] - 0.5 * det["t_pos_second"])
    g_p = torch.autograd.grad(p_only, vs)
    for i in range(len(vs)):
        q_full, q_det = g_full[i] - g_p[i], g_det[i] - g_p[i]
        assert torch.allclose(q_det, 0.5 * q_full, atol=1e-12)
    for a, b in zip(g_full[len(vs):], g_det[len(vs):]):
        assert torch.allclose(a, b, atol=1e-12)


def test_all_view_permutation_invariance_and_refusals():
    vs = _views(V=4, B=6, dtype=torch.float64)
    crit = FixedCosineCritic(D, scale=1.0, bias=0.0).double()
    a = all_view_tokens_loss(vs, crit, negative_detach=True, chunk_size=5)["loss"]
    perm = torch.randperm(6, generator=torch.Generator().manual_seed(0))
    b = all_view_tokens_loss([vs[i][perm] for i in (2, 0, 3, 1)], crit, negative_detach=True, chunk_size=7)["loss"]
    assert torch.allclose(a, b, atol=1e-12)
    torch.manual_seed(0)
    with pytest.raises(TypeError):
        all_view_tokens_loss(_views(), NoisyCosineCritic(D, scale_init=5.0, tau=0.3), negative_detach=True)


def test_finite_difference_matches_autograd():
    for nd in (False,):
        for crit in (FixedCosineCritic(D, scale=2.0, bias=-0.5).double(), CosineCritic(D, scale_init=3.0, bias_init=0.2).double()):
            base = _views(V=3, B=4, dtype=torch.float64)
            xs = [v.detach().clone().requires_grad_(True) for v in base]
            fn = lambda *z: all_view_tokens_loss(list(z), crit, negative_detach=nd, chunk_size=5)["loss"]  # noqa: E731
            assert torch.autograd.gradcheck(fn, tuple(xs), eps=1e-6, atol=1e-6)
            z1, z2 = [v.detach().clone().requires_grad_(True) for v in _views(V=2, B=6, dtype=torch.float64)]
            fn2 = lambda a, b: vcs_pair_loss(a, b, crit, k=3, generator=torch.Generator().manual_seed(4))[0]["loss"]  # noqa: E731
            assert torch.autograd.gradcheck(fn2, (z1, z2), eps=1e-6, atol=1e-6)


# ----------------------------------------------------------------------------------------------------------------- N line
def test_R1_equals_P95_noise_path_and_js_control(tmp_path):
    for loss in ("negative_J", "js_matched_logistic"):
        d = cosine(_base()); d["model"]["critic"]["observation_noise_tau"] = 0.3; d["objective"]["loss"] = loss
        if loss != "negative_J":
            d["model"]["critic"]["noise_repeats"] = 1  # the N2 marker (JS + noise is refused without it)
        cfg_p95 = _small(_load(tmp_path, d, f"p95_{loss}.yaml"))
        torch.manual_seed(0); crit = build_critic(cfg_p95["model"]["critic"], feature_dim=D); crit.train()
        st = copy.deepcopy(crit.state_dict()); crit_b = copy.deepcopy(crit)
        vs = _views(grad=True)
        g = torch.Generator().manual_seed(1)
        k = cfg_p95["pairing"]["k"]
        if loss == "negative_J":
            ref = compute_objective_views({"views_z": vs}, cfg=cfg_p95, critic=crit, pair_generator=g)["loss"]
        else:  # the frozen P95 JS pair function (now reading the noisy logit)
            ref = torch.stack([js_matched_pair_loss_negdetach(vs[a], vs[b], crit, k=k, generator=g)[0]["loss"]
                               for a in range(4) for b in range(a + 1, 4)]).mean()
        calls_ref = int(crit.noise_calls)
        g = torch.Generator().manual_seed(1)
        new = torch.stack([noisy_pair_loss_repeats(vs[a], vs[b], crit_b, k=k, generator=g, negative_detach=True, repeats=1,
                                                   objective="vcs" if loss == "negative_J" else "js")[0]["loss"]
                           for a in range(4) for b in range(a + 1, 4)]).mean()
        assert int(crit_b.noise_calls) == calls_ref == int(st["noise_calls"]) + 6 * 4
        assert torch.allclose(new, ref, atol=1e-7), loss
        ga = torch.autograd.grad(new, vs + [crit_b.scale, crit_b.bias], retain_graph=True)
        gb = torch.autograd.grad(ref, vs + [crit.scale, crit.bias])
        assert all(torch.allclose(x, y, atol=1e-6) for x, y in zip(ga, gb))


def test_tau_zero_and_eval_mode_match_clean_path():
    torch.manual_seed(0); crit = NoisyCosineCritic(D, scale_init=5.0, tau=0.3)
    z1, z2 = _views(V=2, B=10)
    clean = CosineCritic(D, scale_init=5.0)
    ref = vcs_pair_loss_negdetach(z1, z2, clean, k=4, generator=torch.Generator().manual_seed(2))[0]["loss"]
    crit.eval()
    out = noisy_pair_loss_repeats(z1, z2, crit, k=4, generator=torch.Generator().manual_seed(2), negative_detach=True, repeats=4)[0]["loss"]
    assert torch.allclose(out, ref, atol=1e-7)
    crit.train(); crit.tau = 0.0
    out0 = noisy_pair_loss_repeats(z1, z2, crit, k=4, generator=torch.Generator().manual_seed(2), negative_detach=True, repeats=3)[0]["loss"]
    assert torch.allclose(out0, ref, atol=1e-7)


def test_noise_rng_replay_and_counter(tmp_path):
    cfg = _cfg(tmp_path, "N1")
    torch.manual_seed(0); crit = build_critic(cfg["model"]["critic"], feature_dim=D); crit.train()
    st = copy.deepcopy(crit.state_dict()); vs = _views()
    a = compute_objective_views({"views_z": vs}, cfg=cfg, critic=crit, pair_generator=torch.Generator().manual_seed(1))
    assert int(crit.noise_calls) == int(st["noise_calls"]) + 6 * 4 * 4  # 6 view pairs x R=4 x (pos, neg) x (left, right)
    crit.load_state_dict(st)
    b = compute_objective_views({"views_z": vs}, cfg=cfg, critic=crit, pair_generator=torch.Generator().manual_seed(1))
    assert torch.equal(a["loss"], b["loss"]) and a["shift"] == b["shift"]
    B, k = vs[0].shape[0], cfg["pairing"]["k"]
    assert a["critic_pair_evals"] == 4 * 6 * (B + B * k) and a["stats"]["noise_repeats"] == 4.0
    c = compute_objective_views({"views_z": vs}, cfg=cfg, critic=crit, pair_generator=torch.Generator().manual_seed(1))
    assert not torch.equal(c["loss"], a["loss"])  # fresh draws on the next call


# ----------------------------------------------------------------------------------------------------------------- config policy / generated configs
def test_config_policy_rejections(tmp_path):
    for n in NAMES:
        _cfg(tmp_path, n)
    bad = []
    d = variant(_base(), "G1"); d["model"]["critic"]["observation_noise_tau"] = 0.3; bad.append(d)
    d = variant(_base(), "G1"); d["objective"]["loss"] = "js_matched_logistic"; bad.append(d)
    d = variant(_base(), "G1"); d["model"]["critic"]["affine_mode"] = "frozen"; bad.append(d)
    d = variant(_base(), "U1"); d["views"]["count"] = 2; bad.append(d)
    d = variant(_base(), "U1"); d["model"]["critic"]["observation_noise_tau"] = 0.3; bad.append(d)
    d = variant(_base(), "G4"); d["pairing"]["all_view_chunk"] = 64; bad.append(d)
    d = variant(_base(), "N1"); d["model"]["critic"]["observation_noise_tau"] = 0.0; bad.append(d)
    d = variant(_base(), "N1"); d["model"]["critic"]["noise_repeats"] = 0; bad.append(d)
    d = variant(_base(), "N1"); d["views"]["count"] = 2; bad.append(d)
    d = variant(_base(), "G4"); d["model"]["critic"]["noise_eval_repeats"] = 4; bad.append(d)
    d = variant(_base(), "G4"); d["run"]["method"] = "simclr_matched"; bad.append(d)
    d = variant(_base(), "G4"); d["model"]["critic"]["input"] = "ordered_concat"; bad.append(d)
    d = variant(_base(), "N2"); [d["model"]["critic"].pop(x) for x in ("noise_repeats", "noise_eval_repeats", "cosine_bias_init")]; bad.append(d)  # P95 rule kept
    for i, b in enumerate(bad):
        with pytest.raises(ConfigError):
            _load(tmp_path, b, f"bad{i}.yaml")


def test_generated_configs_hashes_and_routing(tmp_path):
    man = REPO / "configs" / "P104_SHA256.json"
    if not man.is_file():
        pytest.skip("configs not generated")
    m = json.loads(man.read_text())
    want = {"G1": "FixedCosineCritic", "G2": "FixedCosineCritic", "G3": "FixedCosineCritic", "U2": "FixedCosineCritic",
            "G4": "CosineCritic", "U1": "CosineCritic", "N1": "NoisyCosineCritic", "N2": "NoisyCosineCritic",
            "G2F": "CosineCritic", "U2F": "CosineCritic"}  # P104 addenda 1 / 3: same-initialisation learned controls
    for f, rec in m["configs"].items():
        p = REPO / "configs" / f
        assert hashlib.sha256(p.read_bytes()).hexdigest() == rec["sha256"]
        cfg = load_config(p, env=_env(tmp_path))
        assert critic_impl_name(cfg["model"]["critic"]).endswith(want[rec["logical_variant"]])
        base = yaml.safe_load((REPO / "configs" / rec["base"]).read_text())
        assert base["train"] == yaml.safe_load(p.read_text())["train"]
    recipe = REPO / "configs" / "cifar10_hpK_a5_views4_800ep_vcs_seed0.yaml"
    rc = load_config(recipe, env=_env(tmp_path))
    assert not any(k in rc["model"]["critic"] for k in ("affine_mode", "cosine_bias_init", "noise_repeats", "noise_eval_repeats"))


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
    fixed = name in ("G1", "G2", "G3", "U2")
    assert rm["hparams"]["trainable_affine"] is (not fixed) and rm["hparams"]["critic_trainable_params"] == (0 if fixed else 2)
    summ = json.loads((tr.run_dir / "summary.json").read_text())
    cost = summ["cost"]
    B, V, k = cfg["train"]["batch_size_images"], 4, cfg["pairing"]["k"]
    assert cost["encoder_updates"] == 4 and cost["critic_updates"] == (0 if fixed else 4)
    if name.startswith("U"):
        assert cost["positive_pairs"] == 4 * V * B * (V - 1) and cost["negative_pairs"] == 4 * V * B * V * (B - 1)
    else:
        assert cost["positive_pairs"] == 4 * 6 * B and cost["negative_pairs"] == 4 * 6 * B * k
    R = 4 if name == "N1" else 1
    assert cost["critic_pair_evaluations"] == R * (cost["positive_pairs"] + cost["negative_pairs"])
    assert "steady_state_views_per_s_actual" in summ
    if fixed:
        ck = load_checkpoint(tr.run_dir / "checkpoints" / "last.pt")
        assert float(ck["critic_state"]["scale"]) == cfg["model"]["critic"]["cosine_scale_init"]


@pytest.mark.parametrize("name", ["G1", "U1", "N1", "N2"])
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
    assert a["p104_cost"] == {**b["p104_cost"], "refresh_seconds": a["p104_cost"]["refresh_seconds"]}


def test_noisy_evaluation_does_not_touch_training_state(tmp_path):
    data, m = synthetic()
    cfg = _cfg(tmp_path, "N1", epochs=1)
    run_dir = Path(cfg["run"]["output_root"]) / "ev"; run_dir.mkdir(parents=True, exist_ok=True)
    from vcs_ssl.train import Trainer
    tr = Trainer(cfg, run_dir=run_dir, data=data, manifest=m, device=torch.device("cpu"), stage="TEST", smoke_steps=2, smoke_epoch_steps=2, epoch_eval=False)
    tr.setup(); assert tr.run() == "COMPLETED"
    enc = {k: v.clone() for k, v in tr.encoder.state_dict().items()}; crit = {k: v.clone() for k, v in tr.critic.state_dict().items()}
    before = rng_fingerprint(capture_rng(tr.device, tr.loader_gen, tr.pair_gen))
    res = tr.epoch_evaluation(tr.ckpt_dir / "last.pt", epoch=1)
    assert rng_fingerprint(capture_rng(tr.device, tr.loader_gen, tr.pair_gen)) == before and res["training_rng_untouched"]
    assert all(torch.equal(enc[k], v) for k, v in tr.encoder.state_dict().items())  # BN buffers included
    assert all(torch.equal(crit[k], v) for k, v in tr.critic.state_dict().items())  # noise counter untouched
    ch = res["critic_holdout"]
    assert len(ch["per_repeat"]) == cfg["evaluation"]["critic_validation"]["repeats"]
    assert 0 <= res["heldout_gate_noisy"] <= 1 and 0 <= res["heldout_gate_clean"] <= 1
    res2 = tr.epoch_evaluation(tr.ckpt_dir / "last.pt", epoch=1)
    assert res2["heldout_J_noisy"] == res["heldout_J_noisy"]  # separate, seeded evaluation noise stream
