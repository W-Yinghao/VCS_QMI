"""P132 — ImageNet-1k preflight (v6 §8.3: data / distributed pairing / throughput only; no full training).

New, self-contained module: nothing in the CIFAR training path imports it.
  * ImageNet manifest + class index (FMCA-AV/imagenet/manifests/imagenet1k_{train,val}.tsv; images under /projects/common/imagenet).
  * Multi-view ImageNet dataset (224 px SimCLR-style views).
  * Distributed losses with cross-GPU negatives: the A-P3 all-view-token VCS objective (and its matched JS), and multi-view SimCLR NT-Xent.
    Both gather the unit-norm tokens of every rank with a differentiable all_gather; each rank sums the per-pair terms of its LOCAL anchor
    rows against ALL global tokens and divides by the GLOBAL pair counts, times world_size, so that
        sum_r loss_r / W  ==  single-process loss on the concatenated global batch, and
        DDP-averaged gradients  ==  gradients of that single-process loss
    (tested on CPU with gloo in tests/test_p132_imagenet.py).  With world_size 1 (no process group) they reduce to the single-process loss.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import functional as F

IMAGENET_ROOT = "/projects/common/imagenet/ILSVRC/Data/CLS-LOC"
MANIFEST_DIR = "/projects/EEG-foundation-model/yinghao/FMCA-AV/imagenet/manifests"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------------------------------------------- data
def read_manifest(split: str, manifest_dir: str = MANIFEST_DIR) -> list[tuple[str, str]]:
    """[(relative path, wnid)] in file order; split in {"train", "val"}."""
    with open(os.path.join(manifest_dir, f"imagenet1k_{split}.tsv"), newline="") as fh:
        rd = csv.reader(fh, delimiter="\t")
        header = next(rd)
        if header[:2] != ["path", "wnid"]:
            raise ValueError(f"unexpected manifest header {header}")
        return [(r[0], r[1]) for r in rd]


def class_index(rows: list[tuple[str, str]]) -> dict[str, int]:
    """Sorted-wnid class index (the torchvision ImageFolder convention)."""
    return {w: i for i, w in enumerate(sorted({w for _, w in rows}))}


def multiview_transform(size: int = 224, blur_p: float = 0.5, min_scale: float = 0.08):
    """SimCLR ImageNet view (crop 0.08–1, flip, jitter 0.4/0.4/0.4/0.1 p 0.8, grayscale 0.2, blur p `blur_p`), ImageNet normalisation."""
    from torchvision import transforms as T
    ops = [T.RandomResizedCrop(size, scale=(min_scale, 1.0), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
           T.RandomHorizontalFlip(),
           T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
           T.RandomGrayscale(p=0.2)]
    if blur_p > 0:
        ops.append(T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=blur_p))
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)


class ImageNetMultiView(torch.utils.data.Dataset):
    """Returns (list of `views` tensors [3, size, size], label).  The JPEG is decoded once per item; every view is an independent draw."""

    def __init__(self, split: str = "train", views: int = 2, size: int = 224, blur_p: float = 0.5, root: str = IMAGENET_ROOT,
                 rows: list[tuple[str, str]] | None = None, classes: dict[str, int] | None = None) -> None:
        self.rows = rows if rows is not None else read_manifest(split)
        self.cls = classes if classes is not None else class_index(read_manifest("train"))   # labels always from the train class set
        self.root, self.views = root, int(views)
        self.tf = multiview_transform(size=size, blur_p=blur_p)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        from PIL import Image
        rel, wnid = self.rows[i]
        with Image.open(os.path.join(self.root, rel)) as im:
            im = im.convert("RGB")
            return [self.tf(im) for _ in range(self.views)], self.cls[wnid]


# ------------------------------------------------------------------------------------------------ distributed pairing
@dataclass
class DistInfo:
    world: int
    rank: int


def dist_info() -> DistInfo:
    if dist.is_available() and dist.is_initialized():
        return DistInfo(dist.get_world_size(), dist.get_rank())
    return DistInfo(1, 0)


def gather_with_grad(x: Tensor) -> Tensor:
    """Concatenate x [b, d] of every rank along dim 0 in rank order, differentiably (gradients return to the owning rank, summed)."""
    di = dist_info()
    if di.world == 1:
        return x
    from torch.distributed.nn.functional import all_gather
    return torch.cat(all_gather(x), dim=0)


def dist_allview_tokens_loss(views_z: list[Tensor], scale: float = 2.0, bias: float = -1.0, *, objective: str = "vcs",
                             negative_detach: bool = False, chunk_size: int = 1024) -> dict[str, Any]:
    """All-view-token objective of A-P3 over the GLOBAL batch (P: ordered distinct tokens of the same image, N_P = V·B(V−1); Q: ordered
    tokens of different images, N_Q = V·B·V(B−1); separate means), fixed scorer T = tanh(a·s + b).  objective "vcs": loss = −J; "js": matched
    balanced logistic mean_P softplus(−2f) + mean_Q softplus(2f).  Each rank's rows are its LOCAL tokens; columns are ALL tokens.
    Returns loss_r (scaled by world size, see the module docstring) and the global statistics (J etc. summed over ranks, detached)."""
    if objective not in ("vcs", "js"):
        raise ValueError("objective must be 'vcs' or 'js'")
    di = dist_info(); V = len(views_z); b = views_z[0].shape[0]
    glob = [gather_with_grad(z) for z in views_z]                       # V × [B, d]
    B = glob[0].shape[0]
    cols = torch.cat(glob, dim=0)                                         # view-major global tokens [V·B, d]
    keys = cols.detach() if negative_detach else cols
    rows = torch.cat(views_z, dim=0)                                      # view-major local tokens [V·b, d]
    dev = rows.device
    col_img = torch.arange(V * B, device=dev) % B
    col_tok = torch.arange(V * B, device=dev)
    loc = torch.arange(V * b, device=dev)
    row_img = di.rank * b + (loc % b)                                     # global image id of each local row
    row_tok = (loc // b) * B + row_img                                    # global flat token index of each local row
    zero = rows.new_zeros(())
    s_p = s_q = s_p2 = s_q2 = l_p = l_q = zero
    n_p = n_q = 0
    for lo in range(0, V * b, chunk_size):
        hi = min(lo + chunk_size, V * b)
        same = row_img[lo:hi, None] == col_img[None, :]
        diag = row_tok[lo:hi, None] == col_tok[None, :]
        mp, mq = same & ~diag, ~same
        fp = (scale * (rows[lo:hi] @ cols.T) + bias)[mp]
        fq = (scale * (rows[lo:hi] @ keys.T) + bias)[mq]
        if objective == "js":
            l_p = l_p + F.softplus(-2.0 * fp).sum(); l_q = l_q + F.softplus(2.0 * fq).sum()
            tp, tq = torch.tanh(fp).detach(), torch.tanh(fq).detach()
        else:
            tp, tq = torch.tanh(fp), torch.tanh(fq)
        s_p = s_p + tp.sum(); s_q = s_q + tq.sum(); s_p2 = s_p2 + tp.square().sum(); s_q2 = s_q2 + tq.square().sum()
        n_p += int(mp.sum()); n_q += int(mq.sum())
    N_P, N_Q = V * B * (V - 1), V * B * V * (B - 1)
    if n_p != V * b * (V - 1) or n_q != V * b * V * (B - 1):
        raise RuntimeError("distributed all-view pair counting failed")
    j_loc = s_p / N_P - s_q / N_Q - 0.5 * s_p2 / N_P - 0.5 * s_q2 / N_Q
    loss_local = (l_p / N_P + l_q / N_Q) if objective == "js" else -j_loc
    stats = torch.stack([s_p, s_q, s_p2, s_q2]).detach()
    if di.world > 1:
        stats = stats.clone(); dist.all_reduce(stats)
    mp_, mq_, sp_, sq_ = stats[0] / N_P, stats[1] / N_Q, stats[2] / N_P, stats[3] / N_Q
    return {"loss": loss_local * di.world, "J_global": (mp_ - mq_ - 0.5 * sp_ - 0.5 * sq_), "t_pos_mean": mp_, "t_neg_mean": mq_,
            "n_pos_global": N_P, "n_neg_global": N_Q, "n_views": V, "global_batch_images": B}


def dist_simclr_multiview_loss(views_z: list[Tensor], temperature: float = 0.2) -> dict[str, Any]:
    """Multi-view SimCLR as in the CIFAR controls: NT-Xent over every view pair (a, b), averaged over pairs; per pair 2B anchors, self
    masked, the positive kept in the denominator.  Each rank's anchors are its local rows of views a and b; keys are the global views a, b."""
    di = dist_info(); V = len(views_z); b = views_z[0].shape[0]
    zs = [F.normalize(z.float() if z.dtype in (torch.float16, torch.bfloat16) else z, dim=-1, eps=1e-8) for z in views_z]
    glob = [gather_with_grad(z) for z in zs]
    B = glob[0].shape[0]
    dev = zs[0].device
    loc = torch.arange(b, device=dev); g = di.rank * b + loc
    pairs = [(a, c) for a in range(V) for c in range(a + 1, V)]
    total = zs[0].new_zeros(())
    for a, c in pairs:
        anchors = torch.cat((zs[a], zs[c]), dim=0)                       # [2b]: rows (a, g) then (c, g)
        keys = torch.cat((glob[a], glob[c]), dim=0)                      # [2B]: (a, 0..B-1) then (c, 0..B-1)
        logits = anchors @ keys.T / temperature
        self_idx = torch.cat((g, B + g)); tgt = torch.cat((B + g, g))
        logits = logits.masked_fill(F.one_hot(self_idx, 2 * B).bool(), -torch.inf)
        total = total + F.cross_entropy(logits, tgt, reduction="sum") / (2 * B)
    loss_local = total / len(pairs)
    return {"loss": loss_local * di.world, "n_pairs": len(pairs), "global_batch_images": B}
