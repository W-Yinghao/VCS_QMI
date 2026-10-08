"""P139 (V7-IMAGENET-PIPELINE): GPU-stage augmentation and distributed bookkeeping.
(1) vectorised per-sample ops == torchvision tensor functional ops applied one image at a time with the same parameters;
(2) parameters are independent per image (no shared sigma / factors), probabilities and ranges as P132;
(3) close to the P132 PIL path for fixed parameters (bounded uint8 differences, reported in detail by the GPU check job);
(4) DistributedSampler(drop_last=True): no duplicated image within an epoch across ranks, all global batches full;
(5) gloo, 2 processes, real DistributedDataParallel: world-scaled loss + DDP averaging == single-process gradients for vcs / js / simclr,
    and the non-detached gather carries the remote-token gradient (differs from negative_detach)."""
import os
import socket
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F

from vcs_ssl.imagenet import dist_allview_tokens_loss, dist_simclr_multiview_loss
from vcs_ssl.imagenet_gpu_aug import (JITTER, KERNEL, SIGMA, apply_views, gaussian_blur_per_sample, pil_reference_view, sample_params,
                                      tensor_reference_view)


def _imgs(n=12, h=40, w=36, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(n, 3, 1, 1, generator=g) * 255
    noise = torch.rand(n, 3, h, w, generator=g) * 120 - 60
    ramp = torch.linspace(0, 80, w).view(1, 1, 1, w)
    return (base + noise + ramp).clamp(0, 255).round().to(torch.uint8)


def test_blur_matches_torchvision_per_sample():
    from torchvision.transforms import functional as TF
    x = _imgs().float() / 255
    sig = torch.tensor([0.1, 0.35, 0.9, 1.4, 2.0, 0.6, 1.1, 0.2, 1.9, 0.5, 1.0, 1.7])
    y = gaussian_blur_per_sample(x, sig)
    for i in range(len(sig)):
        ref = TF.gaussian_blur(x[i], [KERNEL, KERNEL], [float(sig[i]), float(sig[i])])
        assert torch.allclose(y[i], ref, atol=2e-6), (i, (y[i] - ref).abs().max())


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_apply_views_matches_tensor_reference(seed):
    x = _imgs(n=40, seed=seed)
    p = sample_params(40, torch.Generator().manual_seed(seed))
    p["jitter"][:] = True; p["blur"][::2] = True                       # exercise every op on many images
    y = apply_views(x, p, normalize=False)
    for i in range(40):
        ref = tensor_reference_view(x[i], p, i)
        assert torch.allclose(y[i], ref, atol=3e-6), (i, (y[i] - ref).abs().max())


def test_params_independent_and_in_range():
    p = sample_params(20000, torch.Generator().manual_seed(3))
    assert p["sigma"].unique().numel() > 19000                         # per-image sigma (v2 GaussianBlur would share one per call)
    assert SIGMA[0] <= float(p["sigma"].min()) and float(p["sigma"].max()) <= SIGMA[1]
    for key, lim in zip(("brightness", "contrast", "saturation"), JITTER[:3]):
        assert 1 - lim <= float(p[key].min()) and float(p[key].max()) <= 1 + lim
    assert -JITTER[3] <= float(p["hue"].min()) and float(p["hue"].max()) <= JITTER[3]
    for key, pr in (("flip", 0.5), ("jitter", 0.8), ("gray", 0.2), ("blur", 0.5)):
        assert abs(float(p[key].float().mean()) - pr) < 0.015, key
    perms = {tuple(r) for r in p["order"].tolist()}
    assert len(perms) == 24                                             # all op orders occur
    # no correlation between images' sigmas (lag-1)
    s = p["sigma"]; c = torch.corrcoef(torch.stack((s[:-1], s[1:])))[0, 1]
    assert abs(float(c)) < 0.03


def test_close_to_pil_path_fixed_params():
    x = _imgs(n=24, h=48, w=48, seed=5)
    p = sample_params(24, torch.Generator().manual_seed(5))
    y = apply_views(x, p, normalize=False)
    diffs = torch.stack([(y[i] - pil_reference_view(x[i], p, i)).abs().mean() * 255 for i in range(24)])
    assert float(diffs.mean()) < 2.0 and float(diffs.max()) < 6.0, diffs   # mean abs diff in uint8 levels (PIL rounds per op)


def test_distributed_sampler_drop_last_no_duplicates():
    from torch.utils.data import DistributedSampler
    n, W, b = 1003, 4, 8
    seen, per_rank = [], []
    for r in range(W):
        s = DistributedSampler(range(n), num_replicas=W, rank=r, shuffle=True, seed=0, drop_last=True); s.set_epoch(1)
        idx = list(s); per_rank.append(len(idx)); seen += idx
    assert len(set(seen)) == len(seen)                                  # no padding duplicates across ranks
    assert len(set(per_rank)) == 1                                      # equal length on every rank -> same step count
    steps = per_rank[0] // b                                            # DataLoader(drop_last=True): every global batch has W*b images
    assert steps * b <= per_rank[0]


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


class _Enc(torch.nn.Module):
    def __init__(self):
        super().__init__(); torch.manual_seed(0); self.lin = torch.nn.Linear(6, 5, dtype=torch.float64)

    def forward(self, x):
        return F.normalize(self.lin(x), dim=-1)


def _loss(kind, vz):
    if kind == "simclr":
        return dist_simclr_multiview_loss(vz, 0.2)["loss"]
    obj, det = {"vcs": ("vcs", False), "vcs_det": ("vcs", True), "js": ("js", False)}[kind]
    return dist_allview_tokens_loss(vz, 2.0, -1.0, objective=obj, negative_detach=det)["loss"]


def _worker(rank, world, port, kind, xs, out_path):
    os.environ["MASTER_ADDR"] = "127.0.0.1"; os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    model = torch.nn.parallel.DistributedDataParallel(_Enc())
    b = xs[0].shape[0] // world
    vz = [model(x[rank * b:(rank + 1) * b]) for x in xs]
    _loss(kind, vz).backward()                                          # DDP all-reduces (averages) the gradients
    if rank == 0:
        torch.save({k: v.grad.clone() for k, v in model.module.named_parameters()}, out_path)
    dist.destroy_process_group()


@pytest.mark.parametrize("kind", ["vcs", "vcs_det", "js", "simclr"])
def test_real_ddp_matches_single_process(kind):
    g = torch.Generator().manual_seed(1)
    xs = [torch.randn(8, 6, generator=g, dtype=torch.float64) for _ in range(3)]
    enc = _Enc()
    _loss(kind, [enc(x) for x in xs]).backward()
    ref = {k: v.grad.clone() for k, v in enc.named_parameters()}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "g.pt")
        mp.spawn(_worker, args=(2, _free_port(), kind, xs, path), nprocs=2, join=True)
        got = torch.load(path)
    for k in ref:
        assert torch.allclose(got[k], ref[k], atol=1e-10, rtol=0), (kind, k, (got[k] - ref[k]).abs().max())


def test_full_gather_differs_from_detached():
    g = torch.Generator().manual_seed(2)
    xs = [torch.randn(8, 6, generator=g, dtype=torch.float64) for _ in range(2)]
    grads = {}
    for kind in ("vcs", "vcs_det"):
        enc = _Enc(); _loss(kind, [enc(x) for x in xs]).backward(); grads[kind] = enc.lin.weight.grad.clone()
    assert (grads["vcs"] - grads["vcs_det"]).abs().max() > 1e-6        # the Q-side (key) gradient is present in the full version
