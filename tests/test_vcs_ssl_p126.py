"""P126 (v6 V6-CURVE): the fixed anchored-quadratic angular scorer f(s) = a[(s − κ) + λ s(1 − s)], T = tanh f, a, κ, λ fixed buffers.
Checks the endpoints, the slope formula and bounds, the actual zero, λ = 0 ≡ A-P3 (FixedCosineCritic) bit for bit on logits / losses /
gradients (dense and chunked all-view, K8, matched JS), the matrix path, the JS path reading the same logits, the config policy, the
generated configs and the trainer.  Synthetic data only."""
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
from vcs_ssl.models.critic import (FixedCosineCritic, FixedCurvedCosineCritic, build_critic, critic_impl_name,  # noqa: E402
                                   curve_actual_zero)
from vcs_ssl.objectives import all_view_tokens_loss, compute_objective_views, js_matched_pair_loss_negdetach  # noqa: E402
from vcs_ssl.optim import build_optimizer  # noqa: E402
from make_p104_configs import flat  # noqa: E402
from test_vcs_ssl_p104 import _views  # noqa: E402
from test_vcs_ssl_p107 import p107_variant  # noqa: E402
from test_vcs_ssl_p114 import p114_variant  # noqa: E402
from test_vcs_ssl_v2 import _env, _load, _small, run_trainer, synthetic  # noqa: E402

D = 32
LAMS = (-0.25, 0.0, 0.25)


def p126_variant(lam: float) -> dict:
    d = p107_variant("A-P3")
    d["model"]["critic"]["affine_mode"] = "fixed_curved"; d["model"]["critic"]["curvature_lambda"] = float(lam)
    return d


def _cfg(tmp_path, lam=0.25, name=None, **kw) -> dict:
    return _small(_load(tmp_path, p126_variant(lam), f"{name or 'P126_' + str(lam)}.yaml"), **kw)


def ref_logit(s, a, kappa, lam):
    return a * ((s - kappa) + lam * s * (1 - s))


def dense_all_view(views, logit_fn, *, negative_detach, objective):
    """Brute force: explicit ordered token pairs (self excluded; same image -> P, different image -> Q), P and Q averaged on their own counts."""
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
    fp = logit_fn((torch.stack(pl) * torch.stack(pr)).sum(-1)); fq = logit_fn((torch.stack(ql) * torch.stack(qr)).sum(-1))
    if objective == "js":
        return F.softplus(-2 * fp).mean() + F.softplus(2 * fq).mean()
    tp, tq = torch.tanh(fp), torch.tanh(fq)
    return -((tp - 0.5 * tp ** 2).mean() + (-tq - 0.5 * tq ** 2).mean())


# ----------------------------------------------------------------------------------------------------------------- the scorer itself
@pytest.mark.parametrize("a,kappa", [(2.0, 0.5), (1.5, 0.25), (3.0, 0.75)])
@pytest.mark.parametrize("lam", LAMS)
def test_endpoints_and_formula(a, kappa, lam):
    c = FixedCurvedCosineCritic(D, a, -a * kappa, lam).double()
    s = torch.linspace(-1, 1, 401, dtype=torch.float64)
    assert torch.allclose(c.logits_from_similarity(s), ref_logit(s, a, kappa, lam), atol=1e-13)
    f01 = c.logits_from_similarity(torch.tensor([0.0, 1.0], dtype=torch.float64))
    assert float(f01[0]) == pytest.approx(-a * kappa, abs=1e-14) and float(f01[1]) == pytest.approx(a * (1 - kappa), abs=1e-14)
    rec = c.curve_record()
    assert rec["f_at_0"] == pytest.approx(-a * kappa) and rec["f_at_1"] == pytest.approx(a * (1 - kappa)) and rec["kappa_anchor"] == pytest.approx(kappa)


@pytest.mark.parametrize("lam", LAMS)
def test_slope_formula_vs_autograd_and_monotone_bounds(lam):
    a = 2.0
    c = FixedCurvedCosineCritic(D, a, -1.0, lam).double()
    s = torch.linspace(-1, 1, 2001, dtype=torch.float64, requires_grad=True)
    f = c.logits_from_similarity(s)
    (g,) = torch.autograd.grad(f.sum(), s)
    assert torch.allclose(g, c.slope_from_similarity(s.detach()), atol=1e-12)
    assert torch.allclose(g, a * (1 + lam * (1 - 2 * s.detach())), atol=1e-12)
    assert float(g.min()) >= a / 4 - 1e-12 and float(g.max()) <= 7 * a / 4 + 1e-12
    assert bool((f[1:] > f[:-1]).all())  # strictly increasing on [-1, 1]
    rec = c.curve_record()
    assert rec["slope_min_on_cos"] == pytest.approx(float(g.min()), abs=1e-9) and rec["slope_max_on_cos"] == pytest.approx(float(g.max()), abs=1e-9)


def test_actual_zero_solver():
    # a = 2, kappa = 0.5: lambda = +0.25 -> s^2 - 5s + 2 = 0; lambda = -0.25 -> s^2 + 3s - 2 = 0; lambda = 0 -> s0 = kappa
    assert curve_actual_zero(2.0, -1.0, 0.25) == pytest.approx((5 - math.sqrt(17)) / 2, abs=1e-12)
    assert curve_actual_zero(2.0, -1.0, -0.25) == pytest.approx((-3 + math.sqrt(17)) / 2, abs=1e-12)
    assert curve_actual_zero(2.0, -1.0, 0.0) == pytest.approx(0.5, abs=1e-12)
    for a, kappa, lam in ((1.5, 0.25, 0.25), (3.0, 0.75, -0.25), (2.0, -0.4, 0.1)):
        s0 = curve_actual_zero(a, -a * kappa, lam)
        assert abs(ref_logit(s0, a, kappa, lam)) < 1e-12
    assert curve_actual_zero(2.0, -5.0, 0.25) is None  # kappa = 2.5: f < 0 on all of [-1, 1]
    c = FixedCurvedCosineCritic(D, 2.0, -1.0, 0.25)
    assert c.curve_record()["actual_zero_s0"] == pytest.approx((5 - math.sqrt(17)) / 2, abs=1e-12)


def test_invalid_parameters_refused():
    for a, b, lam in ((0.0, -1.0, 0.1), (-1.0, 0.0, 0.0), (2.0, -1.0, 0.3), (2.0, float("nan"), 0.0), (float("inf"), 0.0, 0.0)):
        with pytest.raises(ValueError):
            FixedCurvedCosineCritic(D, a, b, lam)


def test_no_trainable_params_and_no_optimizer_group(tmp_path):
    cfg = _cfg(tmp_path, 0.25)
    crit = build_critic(cfg["model"]["critic"], feature_dim=D)
    assert isinstance(crit, FixedCurvedCosineCritic) and not isinstance(crit, FixedCosineCritic)
    assert list(crit.parameters()) == [] and crit.trainable_affine is False
    assert {n for n, _ in crit.named_buffers()} == {"scale", "bias", "curvature"}
    from vcs_ssl.models import build_models
    built = build_models(cfg, seed=0, device=torch.device("cpu"))
    opt = build_optimizer(built["encoder"], built["projector"], built["critic"], cfg["optimizer"])
    assert "critic" not in [g.get("name") for g in opt.param_groups]


@pytest.mark.parametrize("lam", (-0.25, 0.25))
def test_score_matrix_equals_pairwise(lam):
    c = FixedCurvedCosineCritic(D, 2.0, -1.0, lam).double()
    z1, z2 = (F.normalize(torch.randn(9, D, dtype=torch.float64, generator=torch.Generator().manual_seed(k)), dim=1) for k in (1, 2))
    M = c.score_matrix(z1 @ z2.T)
    pair = torch.stack([c(z1, z2[torch.full((9,), j)]) for j in range(9)], dim=1)
    assert torch.allclose(M, pair, atol=1e-13)
    assert torch.allclose(torch.diagonal(M), c(z1, z2), atol=1e-13)
    assert torch.allclose(c.logits(z1, z2), ref_logit((z1 * z2).sum(-1), 2.0, 0.5, lam), atol=1e-13)


# ----------------------------------------------------------------------------------------------------------------- lambda = 0 == A-P3
def test_lambda0_bit_identical_scorer():
    cu, fx = FixedCurvedCosineCritic(D, 2.0, -1.0, 0.0), FixedCosineCritic(D, 2.0, -1.0)
    z1, z2 = (F.normalize(torch.randn(11, D, generator=torch.Generator().manual_seed(k)), dim=1) for k in (3, 4))
    assert torch.equal(cu.logits(z1, z2), fx.logits(z1, z2)) and torch.equal(cu(z1, z2), fx(z1, z2))
    C = z1 @ z2.T
    assert torch.equal(cu.score_matrix(C), fx.score_matrix(C))


@pytest.mark.parametrize("objective", ("vcs", "js"))
@pytest.mark.parametrize("chunk", (5, 24, 1))
def test_lambda0_bit_identical_all_view(objective, chunk):
    cu, fx = FixedCurvedCosineCritic(D, 2.0, -1.0, 0.0), FixedCosineCritic(D, 2.0, -1.0)
    va, vb = _views(V=4, B=6, grad=True), _views(V=4, B=6, grad=True)
    oa = all_view_tokens_loss(va, cu, negative_detach=False, chunk_size=chunk, objective=objective)
    ob = all_view_tokens_loss(vb, fx, negative_detach=False, chunk_size=chunk, objective=objective)
    assert torch.equal(oa["loss"], ob["loss"]) and torch.equal(oa["J_raw"], ob["J_raw"])
    ga, gb = torch.autograd.grad(oa["loss"], va), torch.autograd.grad(ob["loss"], vb)
    assert all(torch.equal(x, y) for x, y in zip(ga, gb))


def test_lambda0_bit_identical_k8_paths(tmp_path):
    """Cross-view K pairing (VCS via forward) and the matched-JS K8 path (via logits) under the same generator draws."""
    cu, fx = FixedCurvedCosineCritic(D, 2.0, -1.0, 0.0), FixedCosineCritic(D, 2.0, -1.0)
    za, zb = _views(V=2, B=10, grad=True), _views(V=2, B=10, grad=True)
    sa, _ = js_matched_pair_loss_negdetach(za[0], za[1], cu, k=4, generator=torch.Generator().manual_seed(7), negative_detach=False)
    sb, _ = js_matched_pair_loss_negdetach(zb[0], zb[1], fx, k=4, generator=torch.Generator().manual_seed(7), negative_detach=False)
    assert torch.equal(sa["loss"], sb["loss"])
    assert all(torch.equal(x, y) for x, y in zip(torch.autograd.grad(sa["loss"], za), torch.autograd.grad(sb["loss"], zb)))
    cfg_c, cfg_f = (_small(_load(tmp_path, p, f"k8_{i}.yaml")) for i, p in enumerate((p126_variant(0.0), p107_variant("A-P3"))))
    for cfg in (cfg_c, cfg_f):  # the K8 cross-view VCS path: same configs but pair_scope cross_view_k
        cfg["pairing"]["pair_scope"] = "cross_view_k"; cfg["pairing"].pop("all_view_chunk", None)
    vc, vf = _views(V=4, B=8, grad=True), _views(V=4, B=8, grad=True)
    oc = compute_objective_views({"views_z": vc}, cfg=cfg_c, critic=cu, pair_generator=torch.Generator().manual_seed(3))
    of = compute_objective_views({"views_z": vf}, cfg=cfg_f, critic=fx, pair_generator=torch.Generator().manual_seed(3))
    assert torch.equal(oc["loss"], of["loss"])
    assert all(torch.equal(x, y) for x, y in zip(torch.autograd.grad(oc["loss"], vc), torch.autograd.grad(of["loss"], vf)))


# ----------------------------------------------------------------------------------------------------------------- lambda != 0 paths
@pytest.mark.parametrize("lam", (-0.25, 0.25))
@pytest.mark.parametrize("objective", ("vcs", "js"))
def test_all_view_curved_matches_dense_reference(lam, objective):
    """The all-view matrix path (VCS: score_matrix; JS: logits_from_similarity) uses the CURVED logit — equals a brute-force reference built from
    f(s) = a[(s − κ) + λ s(1 − s)], and differs from the affine one."""
    vs = _views(V=4, B=6, grad=True, dtype=torch.float64)
    c = FixedCurvedCosineCritic(D, 2.0, -1.0, lam).double()
    out = all_view_tokens_loss(vs, c, negative_detach=False, chunk_size=7, objective=objective)
    ref = dense_all_view(vs, lambda s: ref_logit(s, 2.0, 0.5, lam), negative_detach=False, objective=objective)
    aff = dense_all_view(vs, lambda s: 2.0 * s - 1.0, negative_detach=False, objective=objective)
    assert torch.allclose(out["loss"], ref, atol=1e-12) and not torch.allclose(out["loss"], aff, atol=1e-6)
    g, gr = torch.autograd.grad(out["loss"], vs), torch.autograd.grad(ref, vs)
    assert all(torch.allclose(x, y, atol=1e-12) for x, y in zip(g, gr))


@pytest.mark.parametrize("lam", (-0.25, 0.25))
def test_js_k8_reads_curved_logits(lam):
    z = _views(V=2, B=10, grad=False, dtype=torch.float64)
    c = FixedCurvedCosineCritic(D, 2.0, -1.0, lam).double()
    st, sh = js_matched_pair_loss_negdetach(z[0], z[1], c, k=3, generator=torch.Generator().manual_seed(11), negative_detach=True)
    from reference.ssl_core import cyclic_negative_indices
    idx, _ = cyclic_negative_indices(10, 3, generator=torch.Generator().manual_seed(11), device=z[0].device)
    fp = ref_logit((z[0] * z[1]).sum(-1), 2.0, 0.5, lam)
    left = z[0].unsqueeze(0).expand(3, -1, -1).reshape(-1, D); right = z[1][idx].reshape(-1, D)
    fq = ref_logit((left * right).sum(-1), 2.0, 0.5, lam)
    assert float(st["loss"]) == pytest.approx(float(F.softplus(-2 * fp).mean() + F.softplus(2 * fq).mean()), abs=1e-12)


# ----------------------------------------------------------------------------------------------------------------- policy / configs
def test_policy(tmp_path):
    for lam in LAMS:
        cfg = _cfg(tmp_path, lam, name=f"ok{lam}")
        assert critic_impl_name(cfg["model"]["critic"]).endswith("FixedCurvedCosineCritic")
    for n in ("A-L1", "A-P3"):  # earlier cells unaffected
        _small(_load(tmp_path, p107_variant(n), f"{n.replace('-', '')}.yaml"))
    _small(_load(tmp_path, p114_variant(), "P114.yaml"))
    d = p126_variant(0.25); d["objective"]["loss"] = "js_matched_logistic"; d["objective"]["js_fixed_scorer"] = True; d["objective"]["js_all_view_tokens"] = True
    _small(_load(tmp_path, d, "js_curved.yaml"))  # the later matched-JS of the same shape (v6 §7.1) is expressible
    bad = []
    d = p126_variant(0.3); bad.append(d)                                                             # |lambda| > 1/4
    d = p126_variant(0.25); d["model"]["critic"].pop("curvature_lambda"); bad.append(d)              # fixed_curved without lambda
    d = p107_variant("A-P3"); d["model"]["critic"]["curvature_lambda"] = 0.1; bad.append(d)          # lambda with affine_mode 'fixed'
    d = p126_variant(0.25); d["model"]["critic"]["cosine_scale_init"] = -2.0; bad.append(d)          # a <= 0
    d = p126_variant(0.25); d["model"]["critic"].pop("cosine_bias_init"); bad.append(d)              # kappa implicit
    d = p126_variant(0.25); d["model"]["critic"]["curvature_lambda"] = True; bad.append(d)           # bool is not a number here
    d = p126_variant(0.25); d["objective"]["loss"] = "js_matched_logistic"; d["objective"]["js_all_view_tokens"] = True; bad.append(d)  # JS w/o marker
    d = p126_variant(0.25); d["model"]["critic"]["observation_noise_tau"] = 0.3; bad.append(d)       # noise on top
    d = p126_variant(0.25); d["model"]["critic"]["affine_mode"] = "curved"; bad.append(d)            # unknown mode
    for i, b in enumerate(bad):
        with pytest.raises(ConfigError):
            _load(tmp_path, b, f"bad{i}.yaml")


def test_generated_p126_configs(tmp_path):
    man = REPO / "configs" / "P126_SHA256.json"
    if not man.is_file():
        pytest.skip("configs not generated")
    recs = json.loads(man.read_text())["configs"]
    assert len(recs) == 4 and sorted(r["lambda"] for r in recs.values()) == [-0.25, -0.25, 0.25, 0.25]
    for f, rec in recs.items():
        p = REPO / "configs" / f
        assert hashlib.sha256(p.read_bytes()).hexdigest() == rec["sha256"]
        cfg = load_config(p, env=_env(tmp_path))
        assert critic_impl_name(cfg["model"]["critic"]).endswith("FixedCurvedCosineCritic")
        mine, parent = yaml.safe_load(p.read_text()), yaml.safe_load((REPO / "configs" / rec["parent"]).read_text())
        fm, fp = flat(mine), flat(parent)
        assert {k for k in set(fm) | set(fp) if fm.get(k) != fp.get(k)} == {"model.critic.affine_mode", "model.critic.curvature_lambda", "run.stage"}
        assert mine["run"]["seed"] == parent["run"]["seed"] == 0
        for blk in ("data", "views", "optimizer", "train", "pairing", "objective"):
            assert mine[blk] == parent[blk], blk


# ----------------------------------------------------------------------------------------------------------------- trainer
def test_trainer_smoke_and_manifest(tmp_path):
    data, m = synthetic()
    cfg = _cfg(tmp_path, 0.25, epochs=2)
    tr, st = run_trainer(cfg, data, m, run_id="tP126", smoke_steps=4, smoke_epoch_steps=2, epoch_eval=False)
    assert st == "COMPLETED", tr.failure_reason
    steps = [json.loads(l) for l in (tr.run_dir / "logs" / "steps.jsonl").read_text().splitlines() if l.strip()]
    assert all(r["J_raw"] is not None and abs(r["J_raw"]) < 3 for r in steps)
    rm = json.loads((tr.run_dir / "run_manifest.json").read_text())
    hp = rm["hparams"]
    assert hp["trainable_affine"] is False and hp["critic_trainable_params"] == 0
    assert hp["p126_curve"]["lambda"] == 0.25 and hp["p126_curve"]["actual_zero_s0"] == pytest.approx((5 - math.sqrt(17)) / 2, abs=1e-12)
    assert hp["p126_curve"]["kappa_anchor"] == pytest.approx(0.5)


def test_stop_resume_identical(tmp_path):
    data, m = synthetic()
    cfg = _cfg(tmp_path, -0.25, epochs=2)
    tr, st = run_trainer(cfg, data, m, run_id="sP126", smoke_steps=4, smoke_epoch_steps=2, epoch_eval=False, stop_after_steps=2)
    assert st == "STOPPED_BUDGET"
    tr2, st2 = run_trainer(cfg, data, m, run_id="sP126", smoke_steps=4, smoke_epoch_steps=2, epoch_eval=False,
                           resume_from=tr.run_dir / "checkpoints" / "last.pt")
    assert st2 == "COMPLETED" and tr2.step == 4
    tr3, st3 = run_trainer(cfg, data, m, run_id="uP126", smoke_steps=4, smoke_epoch_steps=2, epoch_eval=False)
    a = load_checkpoint(tr2.run_dir / "checkpoints" / "last.pt"); b = load_checkpoint(tr3.run_dir / "checkpoints" / "last.pt")
    for part in ("encoder_state", "projector_state"):
        for kk in a[part]:
            assert torch.allclose(a[part][kk].float(), b[part][kk].float(), atol=1e-5), (part, kk)


def test_lambda0_trainer_equals_AP3(tmp_path):
    """λ = 0 curved scorer and A-P3 (FixedCosineCritic) train identically: same init, same steps -> bit-identical encoder / projector states."""
    data, m = synthetic()
    c0 = _cfg(tmp_path, 0.0, name="lam0", epochs=1)
    ca = _small(_load(tmp_path, p107_variant("A-P3"), "AP3ref.yaml"), epochs=1)
    t0, s0 = run_trainer(c0, data, m, run_id="l0", smoke_steps=3, smoke_epoch_steps=3, epoch_eval=False)
    ta, sa = run_trainer(ca, data, m, run_id="ap3", smoke_steps=3, smoke_epoch_steps=3, epoch_eval=False)
    assert s0 == sa == "COMPLETED"
    a = load_checkpoint(t0.run_dir / "checkpoints" / "last.pt"); b = load_checkpoint(ta.run_dir / "checkpoints" / "last.pt")
    for part in ("encoder_state", "projector_state"):
        for kk in a[part]:
            assert torch.equal(a[part][kk], b[part][kk]), (part, kk)
