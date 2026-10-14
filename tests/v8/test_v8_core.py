"""V8 core tests (CPU): agreement with the package reference, the pi = 1/2 identity with RPS, tail consistency, batch-plan invariants."""
from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "v8"))
REF = Path("/home/infres/yinwang/CS_QMI/outputs/V8/package_ref_v1/VCS_V8_Utility_Aligned_CS_Server_v1_20261014/reference")
import v8_core as C  # noqa: E402
import v8_labels as LB  # noqa: E402


@pytest.fixture(scope="module")
def ref():
    if str(REF) not in sys.path:
        sys.path.insert(0, str(REF))
    import ordinal_torch as R
    return R


def _rand(B=64, seed=0, scale=3.0):
    g = torch.Generator().manual_seed(seed); logits = torch.randn(B, 5, generator=g, dtype=torch.float64) * scale
    votes = torch.randint(1, 6, (B, 6), generator=g); n = torch.where(torch.rand(B, generator=g) < 0.5, 3, 6)
    hist = torch.stack([torch.tensor(LB.histogram(votes[b, :n[b]].tolist())) for b in range(B)]).double()
    prior = torch.tensor([0.888, 0.563, 0.288, 0.118], dtype=torch.float64)
    return logits, hist, prior


def test_outputs_match_reference(ref):
    logits, _, prior = _rand()
    o, r = C.outputs(logits, prior), ref.distribution_outputs(logits.float(), prior.float())
    for k in ("q", "u", "ell", "T", "mu"):
        assert torch.allclose(o[k].float(), r[k].float(), atol=1e-5), k
    assert torch.all(o["u"][:, :-1] >= o["u"][:, 1:] - 1e-12)                                      # natural tails non-increasing
    assert torch.allclose(o["mu"], (o["q"] * C.LEVELS.to(o["q"])).sum(-1), atol=1e-10)            # mu = 1 + sum tails = sum h q_h


@pytest.mark.parametrize("kind", ["cs", "bce", "ce", "mse", "l1"])
def test_point_losses_match_reference(ref, kind):
    logits, hist, prior = _rand(seed=1)
    a = C.point_loss(C.outputs(logits, prior), hist, prior, kind); b = ref.loss_per_image(logits.float(), hist.float(), prior.float(), kind)
    assert torch.allclose(a.float(), b, atol=1e-5)


def test_rps_random_rater_form_differs_by_model_free_constant(ref):
    logits, hist, prior = _rand(seed=2); y = torch.flip(torch.cumsum(torch.flip(hist, [-1]), -1), [-1])[:, 1:]
    a = C.point_loss(C.outputs(logits, prior), hist, prior, "rps"); b = ref.loss_per_image(logits.float(), hist.float(), prior.float(), "rps")
    assert torch.allclose(a.float() - b, (y * (1 - y)).mean(-1).float(), atol=1e-5)


def test_cs_equals_rps_when_all_priors_are_half():
    logits, hist, _ = _rand(seed=3); half = torch.full((4,), 0.5, dtype=torch.float64); o = C.outputs(logits, half)
    assert torch.allclose(C.point_loss(o, hist, half, "cs"), C.point_loss(o, hist, half, "rps"), atol=1e-12)
    assert torch.allclose(o["T"], 2 * o["u"] - 1, atol=1e-12)


def test_cs_minimiser_is_the_natural_tail():
    """Pointwise: for a fixed target tail y and prior pi, the cs term is minimised at u = y (proper on each threshold)."""
    pi = torch.tensor([0.888, 0.563, 0.288, 0.118], dtype=torch.float64); us = torch.linspace(1e-4, 1 - 1e-4, 20001, dtype=torch.float64)
    for y in (0.05, 0.3, 0.7, 0.95):
        for k in range(4):
            t = (us - pi[k]) / (pi[k] + (1 - 2 * pi[k]) * us); val = y * (1 - t) ** 2 / (8 * pi[k]) + (1 - y) * (1 + t) ** 2 / (8 * (1 - pi[k]))
            assert abs(float(us[val.argmin()]) - y) < 1e-3


@pytest.mark.parametrize("kind", ["rank", "list"])
def test_ranking_matches_reference(ref, kind):
    g = torch.Generator().manual_seed(4); mu = torch.randn(7, generator=g, dtype=torch.float64) + 3; hb = torch.randint(1, 6, (7,), generator=g).double()
    a = C.ranking_loss(mu, hb, kind); b = ref.task_ranking_loss(mu, hb, {"rank": "pair", "list": "list"}[kind])
    assert torch.allclose(a, b, atol=1e-10)


@pytest.mark.parametrize("kind", C.LOSSES)
def test_gradients_finite_at_extreme_logits(kind):
    logits = torch.tensor([[50., -50, 0, 0, 0], [-50, -50, -50, -50, 50], [0., 0, 0, 0, 0], [10, 9, -30, 4, 1]], dtype=torch.float32, requires_grad=True)
    hist = torch.tensor([[1, 0, 0, 0, 0], [0, 0, 0, 0, 1], [.2, .2, .2, .2, .2], [0, 1 / 3, 1 / 3, 1 / 3, 0]], dtype=torch.float32)
    prior = torch.tensor([0.888, 0.563, 0.288, 0.118]); w = torch.ones(4)
    loss, _, _ = C.batch_loss(logits, hist, w, prior, kind, [0, 0, 1, 1], [2, 2, 2, 2]); loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


def test_batch_loss_weights_and_ranking_normalisation():
    logits, hist, prior = _rand(B=6, seed=5); w = torch.tensor([1., 1, 2, 2, 3, 3], dtype=torch.float64)
    l_mse, _, o = C.batch_loss(logits, hist, w, prior, "mse", [0, 0, 1, 1, 2, 3], [2, 2, 2, 2, 2, 2])
    lp = C.point_loss(o, hist, prior, "mse"); wf = w.to(lp); assert torch.allclose(l_mse, (wf * lp).sum() / wf.sum())
    l_rank, parts, o = C.batch_loss(logits, hist, w, prior, "rank", [0, 0, 1, 1, 2, 3], [2, 2, 1, 1, 1, 1])
    hb = (hist * C.LEVELS.to(hist)).sum(-1).to(o["mu"]); lp = C.point_loss(o, hist, prior, "rank"); wf = w.to(o["mu"])
    r0 = C.ranking_loss(o["mu"][:2], hb[:2], "rank"); r1 = C.ranking_loss(o["mu"][2:4], hb[2:4], "rank")
    expect = (wf * lp).sum() / wf.sum() + (0.5 * r0 + 1.0 * r1) / 1.5
    assert torch.allclose(l_rank, expect) and parts["ranking_subgroups"] == 2 and parts["ranking_pairs"] == 2     # singletons give no ranking term


def test_split_sizes():
    assert C.split_sizes(15) == [5, 5, 5] and C.split_sizes(13) == [7, 6] and C.split_sizes(7) == [7] and C.split_sizes(30) == [6, 6, 6, 6, 6] and C.split_sizes(1) == [1]


def test_batch_manifest_invariants():
    rng = np.random.default_rng(0); pi = {}; k = 0
    for p in range(300):
        n = int(rng.choice([6, 7, 13, 14, 15, 30])); pi[f"p{p}"] = list(range(k, k + n)); k += n
    owner = {i: p for p, imgs in pi.items() for i in imgs}
    B = C.batch_manifest(pi, seed=0, epoch=1)
    flat = [i for b in B for i, _, _ in b]; assert sorted(flat) == list(range(k))                    # every image exactly once
    assert all(len(b) <= C.BATCH_MAX for b in B)
    sub = {}
    for b in B:
        for i, g, ns in b:
            sub.setdefault(g, []).append(i)
    assert all(len(v) <= C.GROUP_MAX and len({owner[i] for i in v}) == 1 for v in sub.values())         # subgroups: one prompt, <= 7
    assert all(len({g for i, g, _ in b if True}) >= 1 for b in B)
    assert B == C.batch_manifest(pi, seed=0, epoch=1) and B != C.batch_manifest(pi, seed=0, epoch=2)    # deterministic per (seed, epoch)
    n_sub = Counter(owner[v[0]] for v in sub.values())
    assert all(ns == n_sub[owner[i]] for b in B for i, g, ns in b)


def test_warmup_cosine():
    assert C.warmup_cosine(0, 1000, 2e-5) == pytest.approx(1e-8) and C.warmup_cosine(100, 1000, 2e-5) == pytest.approx(2e-5)
    assert C.warmup_cosine(1000, 1000, 2e-5) == pytest.approx(0.0, abs=1e-15) and C.warmup_cosine(550, 1000, 2e-5) == pytest.approx(1e-5)


def test_label_histograms_and_priors():
    h = LB.histogram([4, 3, 3]); assert np.allclose(h, [0, 0, 2 / 3, 1 / 3, 0]); t = LB.tails(h)
    assert np.allclose(t, [1, 1, 1 / 3, 0]) and abs(1 + t.sum() - np.mean([4, 3, 3])) < 1e-12
    with pytest.raises(ValueError):
        LB.histogram([]);
    with pytest.raises(ValueError):
        LB.histogram([0, 3])
    recs = [{"tail": [1, .5, .2, 0], "w": 1.0}, {"tail": [1, 1, 1, 1], "w": 3.0}]
    assert np.allclose(LB.fit_priors(recs), [1, .875, .8, .75])
