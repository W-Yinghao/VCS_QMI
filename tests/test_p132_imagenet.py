"""P132 ImageNet preflight: the distributed all-view-token VCS / matched-JS and multi-view SimCLR losses.
(1) world size 1 equals the repo's single-process implementations (all_view_tokens_loss with the fixed (2, -1) scorer; reference NT-Xent);
(2) two gloo processes on CPU, each holding half of the global batch: sum_r loss_r / W equals the single-process loss on the global batch, and the
    DDP-averaged parameter gradients equal the single-process gradients (vcs, js, simclr; full and negative-detached Q for vcs)."""
import os
import socket
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F

from vcs_ssl.imagenet import dist_allview_tokens_loss, dist_simclr_multiview_loss, multiview_transform
from vcs_ssl.models.critic import FixedCosineCritic
from vcs_ssl.objectives import all_view_tokens_loss
from reference.ssl_core import simclr_nt_xent


def _views(B=6, V=3, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [F.normalize(torch.randn(B, d, generator=g, dtype=torch.float64), dim=-1) for _ in range(V)]


@pytest.mark.parametrize("objective", ["vcs", "js"])
@pytest.mark.parametrize("negdet", [False, True])
def test_world1_matches_repo_allview(objective, negdet):
    vz = _views()
    crit = FixedCosineCritic(8, 2.0, -1.0).double()
    ref = all_view_tokens_loss(vz, crit, negative_detach=negdet, chunk_size=5, objective=objective)
    got = dist_allview_tokens_loss(vz, 2.0, -1.0, objective=objective, negative_detach=negdet, chunk_size=7)
    assert torch.allclose(got["loss"], ref["loss"], atol=1e-12, rtol=0)
    assert torch.allclose(got["J_global"], ref["J_raw"].detach(), atol=1e-12, rtol=0)


def test_world1_matches_reference_simclr():
    vz = [v.float() for v in _views(B=5, V=4)]
    pairs = [(a, b) for a in range(4) for b in range(a + 1, 4)]
    ref = torch.stack([simclr_nt_xent(vz[a], vz[b], temperature=0.2) for a, b in pairs]).mean()
    got = dist_simclr_multiview_loss(vz, temperature=0.2)["loss"]
    assert torch.allclose(got, ref, atol=1e-6, rtol=0)


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__(); torch.manual_seed(123); self.lin = torch.nn.Linear(16, 8).double()

    def forward(self, x):  # x [n, 16] -> unit-norm [n, 8]
        return F.normalize(self.lin(x), dim=-1)


def _inputs(B=6, V=3):
    g = torch.Generator().manual_seed(7)
    return torch.randn(V, B, 16, generator=g, dtype=torch.float64)


def _loss(kind, views):
    if kind == "simclr":
        return dist_simclr_multiview_loss([v.double() for v in views], temperature=0.2)["loss"].double()
    obj, negdet = {"vcs": ("vcs", False), "vcs_negdet": ("vcs", True), "js": ("js", False)}[kind]
    return dist_allview_tokens_loss(views, 2.0, -1.0, objective=obj, negative_detach=negdet, chunk_size=4)["loss"]


def _single(kind):
    net = _Tiny(); X = _inputs()
    loss = _loss(kind, [net(X[v]) for v in range(X.shape[0])])
    loss.backward()
    return float(loss.detach()), [p.grad.clone() for p in net.parameters()]


def _worker(rank, world, init, kind, out):
    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=world)
    try:
        net = _Tiny(); X = _inputs(); b = X.shape[1] // world
        loss = _loss(kind, [net(X[v, rank * b:(rank + 1) * b]) for v in range(X.shape[0])])
        loss.backward()
        grads = [p.grad.clone() for p in net.parameters()]
        for gr in grads:
            dist.all_reduce(gr); gr /= world                      # DDP semantics: average over ranks
        lt = torch.tensor([float(loss.detach())], dtype=torch.float64); dist.all_reduce(lt)
        if rank == 0:
            torch.save({"loss_sum_over_W": float(lt) / world, "grads": grads}, out)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("kind", ["vcs", "vcs_negdet", "js", "simclr"])
def test_two_process_gloo_equals_single(kind):
    with tempfile.TemporaryDirectory() as td:
        init = "file://" + os.path.join(td, "pg"); out = os.path.join(td, "res.pt")
        mp.spawn(_worker, args=(2, init, kind, out), nprocs=2, join=True)
        res = torch.load(out)
    l1, g1 = _single(kind)
    assert abs(res["loss_sum_over_W"] - l1) < 1e-10
    for a, b in zip(res["grads"], g1):
        assert torch.allclose(a, b, atol=1e-10, rtol=0), (kind, float((a - b).abs().max()))


def test_transform_shapes():
    from PIL import Image
    im = Image.new("RGB", (500, 375), (120, 30, 200))
    x = multiview_transform(blur_p=0.5)(im)
    assert tuple(x.shape) == (3, 224, 224) and torch.isfinite(x).all()
