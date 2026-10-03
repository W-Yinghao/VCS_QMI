"""P114 (v5 NEXT-A-JS-AP3): the matched-JS control of P107 A-P3 — the balanced logistic loss mean_P softplus(−2f) + mean_Q softplus(2f) on the
all-view-token path with the fixed scorer f = 2s − 1, full (non-detached) gradients.  P115 (v5 NEXT-A-AUG) configs are checked too.
Synthetic data only."""
from __future__ import annotations

import hashlib
import json
import math
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

from vcs_ssl.checkpoint import load_checkpoint  # noqa: E402
from vcs_ssl.config import ConfigError, load_config  # noqa: E402
from vcs_ssl.models.critic import CosineCritic, FixedCosineCritic, build_critic, critic_impl_name  # noqa: E402
from vcs_ssl.objectives import all_view_tokens_loss, compute_objective_views  # noqa: E402
from make_p104_configs import flat  # noqa: E402
from test_vcs_ssl_p104 import _views, variant  # noqa: E402
from test_vcs_ssl_p107 import p107_variant  # noqa: E402
from test_vcs_ssl_v2 import _base, _env, _load, _small, run_trainer, synthetic  # noqa: E402

D = 32


def p114_variant() -> dict:
    d = p107_variant("A-P3")
    d["objective"]["loss"] = "js_matched_logistic"; d["objective"]["js_fixed_scorer"] = True; d["objective"]["js_all_view_tokens"] = True
    return d


def _cfg(tmp_path, d=None, name="P114", **kw) -> dict:
    return _small(_load(tmp_path, d if d is not None else p114_variant(), f"{name}.yaml"), **kw)


def dense_all_view_js(views, scale, bias, *, negative_detach):
    """Brute-force reference: explicit ordered token-pair lists (self pairs excluded; same image -> P, different image -> Q), logit
    f = a<l, r> + b, JS loss with P and Q averaged over their own counts."""
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
    fp = scale * (torch.stack(pl) * torch.stack(pr)).sum(-1) + bias
    fq = scale * (torch.stack(ql) * torch.stack(qr)).sum(-1) + bias
    return F.softplus(-2 * fp).mean() + F.softplus(2 * fq).mean(), len(pl), len(ql)


# ----------------------------------------------------------------------------------------------------------------- loss mechanics
@pytest.mark.parametrize("V,B,chunk", [(4, 6, 5), (4, 6, 24), (4, 6, 1), (3, 5, 7)])
def test_dense_vs_chunked_loss_and_gradients(V, B, chunk):
    vs = _views(V=V, B=B, grad=True, dtype=torch.float64)
    crit = FixedCosineCritic(D, scale=2.0, bias=-1.0).double()
    out = all_view_tokens_loss(vs, crit, negative_detach=False, chunk_size=chunk, objective="js")
    ref, n_p, n_q = dense_all_view_js(vs, 2.0, -1.0, negative_detach=False)
    assert (out["n_pos"], out["n_neg"]) == (n_p, n_q) == (V * B * (V - 1), V * B * V * (B - 1))
    assert torch.allclose(out["loss"], ref, atol=1e-12) and torch.allclose(out["js_loss"], ref.detach(), atol=1e-12)
    g, gr = torch.autograd.grad(out["loss"], vs), torch.autograd.grad(ref, vs)
    assert all(torch.allclose(a, b, atol=1e-12) for a, b in zip(g, gr))


def test_chunk_size_invariance_and_stats_from_tanh():
    vs = _views(V=4, B=6, grad=False, dtype=torch.float64)
    crit = FixedCosineCritic(D, scale=2.0, bias=-1.0).double()
    a = all_view_tokens_loss(vs, crit, negative_detach=False, chunk_size=3, objective="js")
    b = all_view_tokens_loss(vs, crit, negative_detach=False, chunk_size=24, objective="js")
    v = all_view_tokens_loss(vs, crit, negative_detach=False, chunk_size=7, objective="vcs")
    assert torch.allclose(a["loss"], b["loss"], atol=1e-12)
    for k in ("J_raw", "t_pos_mean", "t_neg_mean", "t_pos_second", "t_neg_second"):  # VCS statistics of T = tanh(f), identical across objectives
        assert torch.allclose(a[k], v[k], atol=1e-12), k


def test_self_pair_and_same_image_exclusion():
    """Including self pairs in P, or same-image pairs in Q, changes the value: the masks are what the reference defines."""
    vs = _views(V=3, B=4, grad=False, dtype=torch.float64)
    crit = FixedCosineCritic(D, scale=2.0, bias=-1.0).double()
    out = all_view_tokens_loss(vs, crit, negative_detach=False, chunk_size=5, objective="js")
    flat_ = torch.cat(vs); n = flat_.shape[0]; B = 4
    f = 2.0 * flat_ @ flat_.T - 1.0
    ids = torch.arange(n) % B; same = ids[:, None] == ids[None, :]; diag = torch.eye(n, dtype=torch.bool)
    good = F.softplus(-2 * f[same & ~diag]).mean() + F.softplus(2 * f[~same]).mean()
    with_self = F.softplus(-2 * f[same]).mean() + F.softplus(2 * f[~same]).mean()
    q_with_same = F.softplus(-2 * f[same & ~diag]).mean() + F.softplus(2 * f[~diag]).mean()
    assert torch.allclose(out["loss"], good, atol=1e-12)
    assert not torch.allclose(out["loss"], with_self, atol=1e-6) and not torch.allclose(out["loss"], q_with_same, atol=1e-6)


def test_equal_total_P_and_Q_weights_and_f0_gradient_matches_minus_J():
    """At f = 0 (a = b = 0, learned critic): JS loss = log 4; d loss / d b = −1 + 1 = 0 (equal total P and Q weight), and the (a, b)
    gradients of JS and of −J coincide (both have per-pair f-gradient ∓1 / n at f = 0)."""
    vs = _views(V=4, B=5, grad=True, dtype=torch.float64)
    c_js = CosineCritic(D, scale_init=0.0, bias_init=0.0).double(); c_j = CosineCritic(D, scale_init=0.0, bias_init=0.0).double()
    js = all_view_tokens_loss(vs, c_js, negative_detach=False, chunk_size=6, objective="js")
    mj = all_view_tokens_loss(vs, c_j, negative_detach=False, chunk_size=6, objective="vcs")
    assert abs(float(js["loss"].detach()) - math.log(4.0)) < 1e-12
    g_js = torch.autograd.grad(js["loss"], [c_js.scale, c_js.bias]); g_j = torch.autograd.grad(mj["loss"], [c_j.scale, c_j.bias])
    assert abs(float(g_js[1])) < 1e-12 and abs(float(g_j[1])) < 1e-12
    assert all(torch.allclose(a, b, atol=1e-12) for a, b in zip(g_js, g_j))


def test_full_gradient_reaches_keys():
    vs = _views(V=4, B=6, grad=True, dtype=torch.float64)
    crit = FixedCosineCritic(D, scale=2.0, bias=-1.0).double()
    full = all_view_tokens_loss(vs, crit, negative_detach=False, chunk_size=5, objective="js")["loss"]
    det = all_view_tokens_loss(vs, crit, negative_detach=True, chunk_size=5, objective="js")["loss"]
    assert torch.allclose(full, det, atol=1e-12)
    gf, gd = torch.autograd.grad(full, vs), torch.autograd.grad(det, vs)
    assert any((a - b).abs().sum() > 1e-9 for a, b in zip(gf, gd))


def test_dispatch_and_no_trainable_critic(tmp_path):
    cfg = _cfg(tmp_path)
    crit = build_critic(cfg["model"]["critic"], feature_dim=D).double()
    assert isinstance(crit, FixedCosineCritic) and list(crit.parameters()) == []
    vs = _views(grad=True, dtype=torch.float64)
    out = compute_objective_views({"views_z": vs}, cfg=cfg, critic=crit, pair_generator=torch.Generator().manual_seed(1))
    ref, _, _ = dense_all_view_js(vs, 2.0, -1.0, negative_detach=False)
    assert torch.allclose(out["loss"], ref, atol=1e-12) and out["stats"]["js_loss"] == pytest.approx(float(ref), abs=1e-10)


# ----------------------------------------------------------------------------------------------------------------- policy / configs
def test_policy_markers(tmp_path):
    _cfg(tmp_path)
    for n in ("A-L1", "A-P2", "A-P3"):  # P107 cells unaffected
        _small(_load(tmp_path, p107_variant(n), f"{n.replace('-', '')}.yaml"))
    bad = []
    d = p114_variant(); d["objective"].pop("js_all_view_tokens"); bad.append(d)                    # JS + all-view without the P114 marker (P107 refusal kept)
    d = p114_variant(); d["objective"].pop("js_fixed_scorer"); bad.append(d)                       # P114 marker without the fixed-scorer marker
    d = p114_variant(); d["model"]["critic"]["affine_mode"] = "learned"; d["objective"].pop("js_fixed_scorer"); bad.append(d)  # learned scorer
    d = p107_variant("A-L1"); d["objective"]["js_all_view_tokens"] = True; bad.append(d)           # P114 marker on the K8 path
    d = p107_variant("A-P3"); d["objective"]["js_all_view_tokens"] = True; bad.append(d)           # P114 marker without the JS loss
    for i, b in enumerate(bad):
        with pytest.raises(ConfigError):
            _load(tmp_path, b, f"bad{i}.yaml")


def test_generated_p114_configs(tmp_path):
    man = REPO / "configs" / "P114_SHA256.json"
    if not man.is_file():
        pytest.skip("configs not generated")
    for f, rec in json.loads(man.read_text())["configs"].items():
        p = REPO / "configs" / f
        assert hashlib.sha256(p.read_bytes()).hexdigest() == rec["sha256"]
        cfg = load_config(p, env=_env(tmp_path))
        assert critic_impl_name(cfg["model"]["critic"]).endswith("FixedCosineCritic") and cfg["objective"]["loss"] == "js_matched_logistic"
        assert cfg["pairing"]["pair_scope"] == "all_view_tokens" and cfg["pairing"]["negative_detach"] is False
        mine, parent = yaml.safe_load(p.read_text()), yaml.safe_load((REPO / "configs" / rec["parent"]).read_text())
        fm, fp = flat(mine), flat(parent)
        changed = {k for k in set(fm) | set(fp) if fm.get(k) != fp.get(k)}
        assert changed == {"objective.loss", "objective.js_fixed_scorer", "objective.js_all_view_tokens", "run.stage"}, changed
        # RNG roles identical to the A-P3 reference: seed, pairing RNG, data / views / model / optimizer / train blocks
        assert mine["run"]["seed"] == parent["run"]["seed"]
        for blk in ("data", "views", "model", "optimizer", "train", "pairing"):
            assert mine[blk] == parent[blk], blk


def test_generated_p115_configs(tmp_path):
    man = REPO / "configs" / "P115_SHA256.json"
    if not man.is_file():
        pytest.skip("configs not generated")
    for f, rec in json.loads(man.read_text())["configs"].items():
        p = REPO / "configs" / f
        assert hashlib.sha256(p.read_bytes()).hexdigest() == rec["sha256"]
        load_config(p, env=_env(tmp_path))
        mine, parent, strong = (yaml.safe_load((REPO / "configs" / x).read_text()) for x in (f, rec["parent"], rec["strong_reference"]))
        fm, fp = flat(mine), flat(parent)
        changed = {k for k in set(fm) | set(fp) if fm.get(k) != fp.get(k)} - {"run.stage", "logging.checkpoint_epochs"}
        if rec["factor"] == "croponly":
            assert changed == {"views.random_resized_crop.scale"} and mine["views"]["random_resized_crop"]["scale"] == strong["views"]["random_resized_crop"]["scale"]
        else:
            assert changed == {f"views.color_jitter.{k}" for k in ("brightness", "contrast", "saturation", "hue")}
            assert all(mine["views"]["color_jitter"][k] == strong["views"]["color_jitter"][k] for k in ("brightness", "contrast", "saturation", "hue"))
        assert {20, 100, 400, 800} <= set(mine["logging"]["checkpoint_epochs"]) and set(parent["logging"]["checkpoint_epochs"]) <= set(mine["logging"]["checkpoint_epochs"])
        assert mine["views"]["color_jitter"]["p"] == parent["views"]["color_jitter"]["p"] and mine["views"]["grayscale_p"] == parent["views"]["grayscale_p"]


# ----------------------------------------------------------------------------------------------------------------- trainer
def test_trainer_smoke(tmp_path):
    data, m = synthetic()
    cfg = _cfg(tmp_path, epochs=2)
    tr, st = run_trainer(cfg, data, m, run_id="tP114", smoke_steps=4, smoke_epoch_steps=2, epoch_eval=False)
    assert st == "COMPLETED", tr.failure_reason
    steps = [json.loads(l) for l in (tr.run_dir / "logs" / "steps.jsonl").read_text().splitlines() if l.strip()]
    assert all(r["J_raw"] is not None and abs(r["J_raw"]) < 3 for r in steps)
    rm = json.loads((tr.run_dir / "run_manifest.json").read_text())
    assert rm["hparams"]["trainable_affine"] is False and rm["hparams"]["critic_trainable_params"] == 0


def test_stop_resume_identical(tmp_path):
    data, m = synthetic()
    cfg = _cfg(tmp_path, epochs=2)
    tr, st = run_trainer(cfg, data, m, run_id="sP114", smoke_steps=4, smoke_epoch_steps=2, epoch_eval=False, stop_after_steps=2)
    assert st == "STOPPED_BUDGET"
    tr2, st2 = run_trainer(cfg, data, m, run_id="sP114", smoke_steps=4, smoke_epoch_steps=2, epoch_eval=False,
                           resume_from=tr.run_dir / "checkpoints" / "last.pt")
    assert st2 == "COMPLETED" and tr2.step == 4
    tr3, st3 = run_trainer(cfg, data, m, run_id="uP114", smoke_steps=4, smoke_epoch_steps=2, epoch_eval=False)
    a = load_checkpoint(tr2.run_dir / "checkpoints" / "last.pt"); b = load_checkpoint(tr3.run_dir / "checkpoints" / "last.pt")
    for part in ("encoder_state", "projector_state"):
        for kk in a[part]:
            assert torch.allclose(a[part][kk].float(), b[part][kk].float(), atol=1e-5), (part, kk)


def test_same_init_as_AP3(tmp_path):
    """P114 and A-P3 differ only in the loss: identical encoder / projector initialisation (same seed, same build path)."""
    from vcs_ssl.models import build_models
    c_js, c_ap3 = _cfg(tmp_path), _small(_load(tmp_path, p107_variant("A-P3"), "AP3.yaml"))
    a, b = build_models(c_js, seed=0, device=torch.device("cpu")), build_models(c_ap3, seed=0, device=torch.device("cpu"))
    for part in ("encoder", "projector"):
        sa, sb = a[part].state_dict(), b[part].state_dict()
        assert all(torch.equal(sa[k], sb[k]) for k in sa)
