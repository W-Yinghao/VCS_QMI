"""Common host geometry and role-gated inputs for all supplemental methods."""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from v10 import adapter as A
from .learned import MLPRigidEnergy
from .mind import DescriptorRigidEnergy, MINDConfig, mind2d
from .rulsif import gaussian_features


def intensity_range(image):
    # Exact host 0.5/99.5 percentile convention. A degenerate range is a recorded
    # failure, not silently rescaled by a data-dependent new histogram protocol.
    low, high = np.percentile(image.detach().cpu().numpy(), (0.5, 99.5))
    if not (np.isfinite(low) and np.isfinite(high) and high > low):
        raise ValueError("Degenerate or non-finite native MI intensity range")
    return float(low), float(high)


def patches(image, centres, radius=4):
    axis = torch.arange(-radius, radius + 1, device=image.device, dtype=centres.dtype)
    dy, dx = torch.meshgrid(axis, axis, indexing="ij")
    offsets = torch.stack((dy.reshape(-1), dx.reshape(-1)), -1)
    positions = (centres[:, None] + offsets[None]).reshape(-1, 2)
    return A.Q.sample_pts(torch, F, image, positions[None])[0, :, 0].reshape(len(centres), -1)


class RuLSIFRigidEnergy(A.Q.RigidEnergy):
    """Warp the intensity patches at common positions; exact V moments of g-1."""
    def __init__(self, reference, moving, rng, *, ratio):
        A.assert_frozen(ratio)
        if not bool(ratio.fitted):
            raise ValueError("Unfitted RuLSIF model")
        super().__init__(torch, F, "comir", reference, moving, rng)
        self.ratio = ratio
        self.left = gaussian_features(patches(reference, self.x), ratio.centres_left, ratio.bandwidth)
        w = ratio.coefficients
        self.left_weighted = self.left * w
        self.q_quadratic = (self.left.T @ self.left / len(self.x)) * (w[:, None] * w[None, :])
        axis = torch.arange(-4, 5, device=self.x.device, dtype=self.x.dtype)
        dy, dx = torch.meshgrid(axis, axis, indexing="ij")
        offsets = torch.stack((dy.flatten(), dx.flatten()), -1)
        self.patch_positions = (self.x[:, None] + offsets[None]).reshape(-1, 2)

    def batch(self, theta, translation):
        rows = []
        for k in range(len(theta)):
            coords = A.Q.apply_T(torch, theta[k:k + 1], translation[k:k + 1], self.patch_positions)
            values = A.Q.sample_pts(torch, F, self.mov, coords)[0, :, 0].reshape(len(self.x), 81)
            right = gaussian_features(values, self.ratio.centres_right, self.ratio.bandwidth)
            gp = (self.left_weighted * right).sum(-1)
            gq2 = ((right @ self.q_quadratic) * right).sum(-1).mean()
            # alpha=1/2: J(g-1) = 2 E_P g - (E_P g² + E_Q g²)/2 - 1.
            rows.append(2 * gp.mean() - 0.5 * (gp.square().mean() + gq2) - 1)
        return torch.stack(rows)

    def grid_values(self, theta_degrees, translations):
        theta = torch.full((1,), math.radians(theta_degrees), device=translations.device,
                           dtype=translations.dtype)
        return torch.cat([self.batch(theta.expand(len(block)), block) for block in translations.split(4)])


class FiniteEnergy:
    """Reject bad energies before the host's nan_to_num could conceal them."""
    def __init__(self, energy):
        self.energy = energy

    @staticmethod
    def checked(value):
        if not torch.isfinite(value).all():
            raise FloatingPointError("Non-finite registration energy")
        return value

    def grid_values(self, angle, translations):
        return self.checked(self.energy.grid_values(angle, translations))

    def __call__(self, parameters):
        return self.checked(self.energy(parameters))

    def value(self, angle, translation):
        value = self.energy.value(angle, translation)
        if not math.isfinite(value):
            raise FloatingPointError("Non-finite registration truth energy")
        return value


class RegistrationSession:
    """One frozen checkpoint, one data role. Cache reference only for the current core."""
    def __init__(self, family, trainer, guard, role, cores, device, candidate):
        if role not in ("CAL", "DEV"):
            raise PermissionError("Only currently activated CAL/DEV are supported")
        self.family, self.trainer, self.device = family, trainer, torch.device(device)
        self.candidate = candidate
        self.perturbations = {}
        for core in cores:
            guard.log_rule(core, {role}, "V10 supplemental frozen perturbation rule")
            self.perturbations[core] = A.Q.perturbations_p2(core)
        self.inputs = A.GuardedInputs(guard, {role}, "V10 supplemental " + role,
                                      self.perturbations, self.device)
        self.core, self.reference = None, None

    def energy(self, core, perturbation):
        A.synchronize(self.device)
        started = time.perf_counter()
        ref_seconds = 0.0
        if self.core != core:
            self.inputs.ref.clear(); self.inputs.mov.clear()
            intensity, learned = self.inputs.reference(core)
            with torch.no_grad():
                if self.family in ("MLP_VCS_perm8", "InfoNCE_host[J_perm1]"):
                    self.reference = self.trainer.eU(learned[None])[0]
                elif self.family == "CoMIR_native":
                    self.reference = self.trainer.reference_encoder(learned[None])[0]
                elif self.family == "MIND_2D_adapted":
                    self.reference = mind2d(intensity, MINDConfig(**self.candidate["descriptor"]))
                else:
                    self.reference = intensity
            self.core = core
            A.synchronize(self.device); ref_seconds = time.perf_counter() - started
        intensity, learned = self.inputs.moving(core, perturbation)
        rng = np.random.default_rng(A.Q.seed_of(f"R6-REG-P2|pos|{core}|{perturbation}"))
        if self.family in ("MLP_VCS_perm8", "InfoNCE_host[J_perm1]"):
            draws = 8 if self.family == "MLP_VCS_perm8" else 1
            energy = MLPRigidEnergy(self.reference, learned, rng, encoder=self.trainer.eV,
                                    critic=self.trainer.critic, draws=draws,
                                    extra_seed=A.Q.seed_of(f"V10|MLP-extra-registration-perms|{core}|{perturbation}"))
        elif self.family in ("MI_native", "NMI_native"):
            kind = "mi" if self.family == "MI_native" else "nmi"
            energy = A.Q.RigidEnergy(torch, F, kind, self.reference, intensity, rng,
                                     ranges=(intensity_range(self.reference), intensity_range(intensity)))
        elif self.family == "CoMIR_native":
            with torch.no_grad():
                latent = self.trainer.moving_encoder(learned[None])[0]
            energy = A.Q.RigidEnergy(torch, F, "comir", self.reference, latent, rng)
        elif self.family == "MIND_2D_adapted":
            with torch.no_grad():
                descriptor = mind2d(intensity, MINDConfig(**self.candidate["descriptor"]))
            energy = DescriptorRigidEnergy(self.reference, descriptor, rng)
        elif self.family == "RuLSIF_exact":
            energy = RuLSIFRigidEnergy(self.reference, intensity, rng, ratio=self.trainer.ratio)
        else:
            raise ValueError("Unknown supplemental family: " + self.family)
        A.synchronize(self.device)
        total = time.perf_counter() - started
        # No cross-perturbation moving-latent cache; this cost is paid and recorded.
        self.inputs.mov.pop((core, perturbation), None)
        return FiniteEnergy(energy), {"preparation_seconds": total,
                                      "reference_preparation_seconds": ref_seconds,
                                      "moving_and_energy_preparation_seconds": total - ref_seconds}
