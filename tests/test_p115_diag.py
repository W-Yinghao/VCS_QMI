"""P115 diagnostics: crop overlap, loss decomposition = training loss, analytic pulls = autograd, determinism, checkpoint mapping, smoke."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as T

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from reference.ssl_core import cyclic_negative_indices, simclr_nt_xent, vcs_from_scores  # noqa: E402
from vcs_diag.augdiag import (batch_diagnostics, crop_overlap, known_box_views, objective_terms, positive_pulls,  # noqa: E402
                              resolve_checkpoints)
from vcs_ssl.models.critic import CosineCritic, FixedCosineCritic  # noqa: E402
from vcs_ssl.objectives import all_view_tokens_loss  # noqa: E402

torch.manual_seed(0)


def _cfg(method="vcs_qmi", scope="cross_view_k", nd=True, k=3):
    return {"run": {"method": method}, "objective": {"loss": "negative_J", "simclr_temperature": 0.2},
            "pairing": {"k": k, "negative_detach": nd, "pair_scope": scope}, "model": {"normalization": {"eps": 1e-8, "vcs_and_simclr": "l2"}}}


def _vz(V=3, B=6, D=8, grad=True):
    z = F.normalize(torch.randn(V * B, D, dtype=torch.float64), dim=1)
    if grad:
        z.requires_grad_(True)
    return z, list(z.chunk(V))


# ---------------------------------------------------------------------------------------------- crop overlap
def test_crop_overlap_known_boxes():
    assert crop_overlap((0, 0, 32, 32), (0, 0, 32, 32)) == (1.0, 1.0)
    assert crop_overlap((0, 0, 16, 16), (16, 16, 16, 16)) == (0.0, 0.0)
    iou, area = crop_overlap((0, 0, 16, 32), (8, 0, 16, 32))       # 8x32 overlap of two 16x32 boxes
    assert iou == pytest.approx(256 / (512 + 512 - 256)) and area == pytest.approx(256 / 1024)
    assert crop_overlap((4, 4, 8, 8), (0, 0, 32, 32)) == (pytest.approx(64 / 1024), pytest.approx(64 / 1024))   # nested box


# ---------------------------------------------------------------------------------------------- decomposition = training loss
@pytest.mark.parametrize("nd", [True, False])
def test_vcs_cross_view_terms_equal_training_loss(nd):
    crit = FixedCosineCritic(8, 2.0, -1.0).double()
    z, vz = _vz()
    t = objective_terms(_cfg(nd=nd), crit, vz, pair_seed=5)
    g = torch.Generator().manual_seed(5); losses = []
    for a in range(3):
        for b in range(a + 1, 3):
            idx, _ = cyclic_negative_indices(6, 3, generator=g)
            left = vz[a].unsqueeze(0).expand(3, -1, -1).reshape(-1, 8); right = (vz[b].detach() if nd else vz[b])[idx].reshape(-1, 8)
            losses.append(vcs_from_scores(crit(vz[a], vz[b]), crit(left, right))["loss"])
    ref = torch.stack(losses).mean()
    assert float(t["Lp"] + t["Lq"]) == pytest.approx(float(ref), abs=1e-12)
    g1 = torch.autograd.grad(t["Lp"] + t["Lq"], z, retain_graph=True)[0]; g2 = torch.autograd.grad(ref, z)[0]
    assert torch.allclose(g1, g2, atol=1e-12)
    gp = torch.autograd.grad(t["Lp"], z, retain_graph=True)[0]; gq = torch.autograd.grad(t["Lq"], z, retain_graph=True)[0]
    assert torch.allclose(gp + gq, g1, atol=1e-12)


@pytest.mark.parametrize("nd", [True, False])
def test_vcs_all_view_terms_equal_training_loss(nd):
    crit = FixedCosineCritic(8, 2.0, -1.0).double()
    z, vz = _vz(V=4, B=5)
    t = objective_terms(_cfg(scope="all_view_tokens", nd=nd), crit, vz, pair_seed=0)
    ref = all_view_tokens_loss(vz, crit, negative_detach=nd, chunk_size=7)["loss"]
    assert float(t["Lp"] + t["Lq"]) == pytest.approx(float(ref), abs=1e-12)
    assert torch.allclose(torch.autograd.grad(t["Lp"] + t["Lq"], z, retain_graph=True)[0], torch.autograd.grad(ref, z)[0], atol=1e-12)
    assert len(t["c_pos"]) == 4 * 5 * 3 and int(t["s_neg"].numel()) == 4 * 5 * 4 * 4


def test_simclr_terms_equal_training_loss():
    z, vz = _vz(V=4, B=5)
    t = objective_terms(_cfg(method="simclr_matched"), None, vz, pair_seed=0)
    ref = torch.stack([simclr_nt_xent(vz[a], vz[b], temperature=0.2) for a in range(4) for b in range(a + 1, 4)]).mean()
    assert float(t["Lp"] + t["Lq"]) == pytest.approx(float(ref), abs=1e-10)
    assert torch.allclose(torch.autograd.grad(t["Lp"] + t["Lq"], z, retain_graph=True)[0], torch.autograd.grad(ref, z)[0], atol=1e-10)


# ---------------------------------------------------------------------------------------------- analytic pulls = autograd
@pytest.mark.parametrize("kind", ["vcs_k", "vcs_k_learned", "vcs_all", "simclr"])
def test_positive_pulls_match_autograd(kind):
    if kind == "simclr":
        cfg, crit = _cfg(method="simclr_matched"), None
    elif kind == "vcs_all":
        cfg, crit = _cfg(scope="all_view_tokens", nd=False), FixedCosineCritic(8, 2.0, -1.0).double()
    elif kind == "vcs_k_learned":
        cfg, crit = _cfg(nd=True), CosineCritic(8, scale_init=7.0, bias_init=-3.0).double()
    else:
        cfg, crit = _cfg(nd=False), FixedCosineCritic(8, 2.0, -1.0).double()
    z, vz = _vz(V=4, B=5)
    t = objective_terms(cfg, crit, vz, pair_seed=1)
    pa, pb, _ = positive_pulls(t, crit, vz)
    B = 5; ta, tb = t["va"] * B + t["img"], t["vb"] * B + t["img"]
    tot = torch.zeros_like(z).index_add_(0, ta, pa).index_add_(0, tb, pb)
    assert torch.allclose(tot, torch.autograd.grad(t["Lp"], z)[0], atol=1e-10)


# ---------------------------------------------------------------------------------------------- determinism
def _tf():
    return T.Compose([T.RandomResizedCrop(32, scale=(0.08, 1.0)), T.RandomHorizontalFlip(), T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                      T.RandomGrayscale(0.2), T.ToTensor()])


def test_known_box_views_deterministic_and_rng_isolated():
    imgs = np.random.default_rng(0).integers(0, 255, size=(10, 32, 32, 3), dtype=np.uint8); uids = np.array([3, 7, 1])
    torch.manual_seed(123); before = torch.rand(3)
    torch.manual_seed(123); v1, b1 = known_box_views(imgs, uids, _tf(), 4, seed=9); after = torch.rand(3)
    v2, b2 = known_box_views(imgs, uids, _tf(), 4, seed=9)
    assert torch.equal(before, after), "diagnostic RNG must not disturb the global RNG"
    assert all(torch.equal(x, y) for x, y in zip(v1, v2)) and np.array_equal(b1, b2)
    v3, b3 = known_box_views(imgs, uids, _tf(), 4, seed=10)
    assert not np.array_equal(b1, b3)
    assert (b1[..., 2] > 0).all() and (b1[..., 0] + b1[..., 2] <= 32).all() and (b1[..., 1] + b1[..., 3] <= 32).all()


def test_resolve_checkpoints():
    m = resolve_checkpoints(["initial.pt", "epoch_100.pt", "epoch_200.pt", "epoch_400.pt", "epoch_800.pt", "last.pt"])
    assert [x["epoch"] for x in m] == [100, 100, 400, 800] and m[0]["note"].startswith("epoch 20 not saved") and m[1]["duplicate"]
    m2 = resolve_checkpoints(["epoch_020.pt", "epoch_100.pt", "epoch_400.pt", "epoch_800.pt"])
    assert [x["epoch"] for x in m2] == [20, 100, 400, 800] and not any(x["duplicate"] for x in m2)


# ---------------------------------------------------------------------------------------------- CPU smoke of one batch
@pytest.mark.parametrize("method,scope,nd", [("vcs_qmi", "cross_view_k", True), ("vcs_qmi", "all_view_tokens", False), ("simclr_matched", "cross_view_k", True)])
def test_batch_diagnostics_smoke(method, scope, nd):
    enc = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
    proj = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8))
    crit = FixedCosineCritic(8, 2.0, -1.0) if method == "vcs_qmi" else None
    imgs = np.random.default_rng(1).integers(0, 255, size=(12, 32, 32, 3), dtype=np.uint8)
    views, boxes = known_box_views(imgs, np.arange(8), _tf(), 3, seed=4)
    out = batch_diagnostics(_cfg(method=method, scope=scope, nd=nd), enc, proj, crit, views, boxes, torch.device("cpu"), pair_seed=2)
    shares = [r["share_of_P_grad"] for r in out["crop_groups"]["iou"] if r["n_pos"]]
    assert sum(shares) == pytest.approx(1.0, abs=1e-4)
    assert out["token_pull"]["autograd_check_rel_err"] < 1e-4
    assert out["grad"]["encoder"]["P"] > 0 and out["view_pairs"]["cancellation_ratio"] <= 1.0 + 1e-6
