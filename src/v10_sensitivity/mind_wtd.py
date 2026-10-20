"""D2 named post-DEV sensitivity variant MIND_2D_warp_then_describe.

The frozen MIND_2D_adapted row (v10.supplement.mind.DescriptorRigidEnergy) describes the
perturbed moving image once, in its own axis frame, and samples those channels at T(x).
This variant resamples the moving image into the reference frame first, then describes it:

  grid stage   per search angle one rotation-only warp over the host grid box BOX2, described
               once; that angle's translations reuse it at x + R(-theta) t (the host critic's
               grid machinery, Q.RigidEnergy.grid_values kind 'critic')
  exact stage  the full (theta, t) warp over the host exact box BOX1, described, sampled at x
               (every refinement step, F_hat and F_true; differentiable)

Everything else is the frozen row's: MINDConfig (radius, sigma, offset, variance floor),
reference descriptor, energy -mean (u - v)^2 over positions and channels, support, positions,
permutation stream, perturbations, searcher, bilinear sampling convention and data roles.
MIND's footprint is offset + radius = 2 px, far inside the 16-px critic margin the host boxes
were built for, so the box padding never reaches a sampled descriptor value.

Import only after the frozen tree has been bound (v10_sensitivity.frozen.import_runner).
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from v10 import adapter as A
from v10.supplement.mind import MINDConfig, mind2d
from v10.supplement.registration import FiniteEnergy, RegistrationSession

VARIANT = "MIND_2D_warp_then_describe"
FAMILY = "MIND_2D_adapted"


class WarpThenDescribeEnergy(A.Q.RigidEnergy):
    """F(theta, t) = -mean_x,c (D_U(x) - MIND(V o T_{theta,t})(x))^2 on the frozen MIND support."""

    def __init__(self, reference, moving, rng, *, config):
        if reference.ndim != 3 or moving.ndim != 3 or moving.shape[0] != 1:
            raise ValueError("Expected a reference descriptor [D,H,W] and a moving scalar image [1,H,W]")
        if reference.shape[-2:] != moving.shape[-2:] or not moving.is_floating_point():
            raise ValueError("Reference descriptor and moving image must share one floating image grid")
        # Same rng consumption as the frozen DescriptorRigidEnergy: positions, then the permutation.
        super().__init__(torch, F, "comir", reference, moving, rng)
        self.config = config
        self.u = A.Q.sample_pts(torch, F, reference, self.x[None])[0]
        device = moving.device
        self.box1 = A.Q.grid_coords(torch, *A.Q.BOX1, device)
        self.box2 = A.Q.grid_coords(torch, *A.Q.BOX2, device)
        self.n1 = A.Q.BOX1[1] - A.Q.BOX1[0] + 1
        self.n2 = A.Q.BOX2[1] - A.Q.BOX2[0] + 1
        self.describe_calls = 0

    def describe(self, coordinates, size):
        """Moving image resampled at the given reference-frame box coordinates, then MIND-described."""
        warped = A.Q.sample_pts(torch, F, self.mov, coordinates)[0, :, 0].reshape(1, size, size)
        self.describe_calls += 1
        return mind2d(warped, self.config)

    def batch(self, theta, translation):
        rows = []
        for k in range(len(theta)):
            coordinates = A.Q.apply_T(torch, theta[k:k + 1], translation[k:k + 1], self.box1)
            descriptor = self.describe(coordinates, self.n1)
            moving = A.Q.sample_pts(torch, F, descriptor, (self.x - A.Q.BOX1[0])[None])[0]
            rows.append(-(self.u - moving).square().mean())
        return torch.stack(rows)

    def grid_values(self, theta_degrees, translations):
        device, dtype = translations.device, translations.dtype
        theta = torch.full((1,), math.radians(theta_degrees), device=device, dtype=dtype)
        zero = torch.zeros(1, 2, device=device, dtype=dtype)
        descriptor = self.describe(A.Q.apply_T(torch, theta, zero, self.box2), self.n2)
        # W_{theta,t}(z) = W_theta(z + R(-theta) t): the host's shift for one rotation-only warp per angle.
        c, s = math.cos(math.radians(theta_degrees)), math.sin(math.radians(theta_degrees))
        shifts = torch.stack([c * translations[:, 0] + s * translations[:, 1],
                              -s * translations[:, 0] + c * translations[:, 1]], -1)
        origin = A.Q.BOX2[0]
        values = []
        for block in shifts.split(A.Q.CHUNK["comir"]):
            moving = A.Q.sample_pts(torch, F, descriptor, self.x[None] + block[:, None] - origin)
            values.append(-(self.u[None] - moving).square().mean((-2, -1)))
        return torch.cat(values)


class WarpThenDescribeSession(RegistrationSession):
    """The frozen RegistrationSession; only the MIND moving-side energy is the variant.

    Inputs, reference descriptor, per-core caching, the positions rng, the finite-energy
    guard, the no-cross-perturbation cache rule and the timing fields are the frozen ones.
    """

    def energy(self, core, perturbation):
        if self.family != FAMILY:
            raise ValueError(VARIANT + " applies only to the MIND_2D_adapted family")
        config = MINDConfig(**self.candidate["descriptor"])
        A.synchronize(self.device)
        started = time.perf_counter()
        ref_seconds = 0.0
        if self.core != core:
            self.inputs.ref.clear(); self.inputs.mov.clear()
            intensity, learned = self.inputs.reference(core)
            with torch.no_grad():
                self.reference = mind2d(intensity, config)
            self.core = core
            A.synchronize(self.device); ref_seconds = time.perf_counter() - started
        intensity, learned = self.inputs.moving(core, perturbation)
        rng = np.random.default_rng(A.Q.seed_of(f"R6-REG-P2|pos|{core}|{perturbation}"))
        energy = WarpThenDescribeEnergy(self.reference, intensity, rng, config=config)
        A.synchronize(self.device)
        total = time.perf_counter() - started
        self.inputs.mov.pop((core, perturbation), None)
        return FiniteEnergy(energy), {"preparation_seconds": total,
                                      "reference_preparation_seconds": ref_seconds,
                                      "moving_and_energy_preparation_seconds": total - ref_seconds,
                                      "variant": VARIANT}
