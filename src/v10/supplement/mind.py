"""Disclosed 2D MIND adaptation, not MIND-SSC or an unmodified 3D reproduction.

Based on Heinrich et al., Medical Image Analysis 2012, doi:10.1016/j.media.2012.05.008.
Four axial neighbours, Gaussian weighted patch SSD, local four-neighbour variance,
maximum-one normalisation. Constant images have an all-one descriptor.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from v10 import adapter as A


@dataclass(frozen=True)
class MINDConfig:
    radius: int = 1
    sigma: float = 0.8
    offset: int = 1
    variance_floor: float = 1e-6

    def __post_init__(self):
        if self.radius < 1 or self.offset < 1:
            raise ValueError("MIND radius and offset must be positive integers")
        if any(not math.isfinite(v) or v <= 0 for v in (self.sigma, self.variance_floor)):
            raise ValueError("MIND sigma and variance floor must be finite and positive")


def mind2d(image, config=MINDConfig()):
    """Single robust-normalised scalar image [1,H,W] -> descriptor [4,H,W]."""
    if image.ndim != 3 or image.shape[0] != 1 or not image.is_floating_point():
        raise ValueError("MIND requires a floating scalar image [1,H,W]")
    radius, offset = config.radius, config.offset
    h, w = image.shape[-2:]
    if min(h, w) <= max(radius, offset):
        raise ValueError("Image is too small for reflected descriptor padding")
    padded = F.pad(image[None], (offset,) * 4, mode="reflect")
    distances = []
    axis = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel = torch.exp(-0.5 * axis.square() / config.sigma**2)
    kernel = kernel[:, None] * kernel[None, :]
    kernel = (kernel / kernel.sum())[None, None]
    for dy, dx in ((-offset, 0), (offset, 0), (0, -offset), (0, offset)):
        shifted = padded[:, :, offset + dy:offset + dy + h, offset + dx:offset + dx + w]
        ssd = (image[None] - shifted).square()
        distances.append(F.conv2d(F.pad(ssd, (radius,) * 4, mode="reflect"), kernel))
    distance = torch.cat(distances, dim=1)
    variance = distance.mean(1, keepdim=True).clamp_min(config.variance_floor)
    # Subtracting min distance is exactly division by the largest exp descriptor,
    # while avoiding an all-underflowed descriptor on a highly contrasted patch.
    descriptor = torch.exp(-(distance - distance.amin(1, keepdim=True)) / variance)
    return descriptor[0]


class DescriptorRigidEnergy(A.Q.RigidEnergy):
    """Encode once, then sample every descriptor channel on the common support.

    Channels retain their image-axis orientation; no hidden rotation of the MIND
    channel basis. This is an explicit limitation of the declared 2D adaptation.
    """

    def __init__(self, reference, moving, rng):
        if reference.ndim != 3 or moving.shape != reference.shape:
            raise ValueError("Descriptor maps must have equal [D,H,W] shapes")
        super().__init__(torch, F, "comir", reference, moving, rng)
        self.u = A.Q.sample_pts(torch, F, reference, self.x[None])[0]

    def batch(self, theta, translation):
        moving = A.Q.sample_pts(torch, F, self.mov,
                               A.Q.apply_T(torch, theta, translation, self.x))
        return -(self.u[None] - moving).square().mean((-2, -1))
