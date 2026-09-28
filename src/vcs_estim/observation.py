"""Estimator package v1 (§9, stage 3 — defined now for the stage-0 check only): independent Gaussian observation noise of fixed
per-coordinate sd sigma on both sides, drawn from separate streams, with no normalisation after the noise (sigma = 0 is explicit)."""
from __future__ import annotations

import torch


def observe(x: torch.Tensor, y: torch.Tensor, sigma: float, gen_x: torch.Generator, gen_y: torch.Generator):
    if sigma == 0:
        return x.clone(), y.clone()
    nx = torch.randn(x.shape, generator=gen_x, dtype=x.dtype); ny = torch.randn(y.shape, generator=gen_y, dtype=y.dtype)
    return x + sigma * nx, y + sigma * ny
