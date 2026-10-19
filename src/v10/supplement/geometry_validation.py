"""Validation-only derivative checks for native bilinearly sampled energies.

No training/registration precision is changed. fp64 copies avoid cancellation,
and the finite-difference step stays inside the local interpolation cells.
"""
import copy

import torch

from v10 import adapter as A


def native_gradient_check(energy, parameters, gradient):
    if energy.kind not in ('mi', 'nmi', 'comir'):
        raise ValueError('Native geometry check only supports MI/NMI/latent MSE')
    reference = copy.copy(energy)
    for name in ('x', 'u', 'mov'):
        setattr(reference, name, getattr(energy, name).detach().double())
    p = parameters.detach().double().requires_grad_()
    value = reference(p)
    expected = torch.autograd.grad(value, p)[0]
    coords = A.Q.apply_T(torch, p[0:1]/A.Q.RHO, p[1:3][None], reference.x)
    margin = float((coords-coords.round()).abs().min().detach())
    # Translation speed is one; rotation uses u=rho*theta. An L1 bound on the
    # radius bounds each coordinate's rotational speed, including small steps.
    speed = max(1., float((reference.x-A.Q.CEN).abs().sum(-1).max())/A.Q.RHO)
    h = min(1e-5, margin/(8*speed))
    if h < 1e-7:
        raise RuntimeError('Validation point is too close to a bilinear boundary')
    differences = []
    for step in (h, h/2):
        eye = torch.eye(3, dtype=p.dtype, device=p.device)*step
        with torch.no_grad():
            fd = torch.stack([(reference(p.detach()+e)-reference(p.detach()-e))/(2*step) for e in eye])
        torch.testing.assert_close(expected, fd, atol=1e-6, rtol=1e-3)
        differences.append(fd)
    torch.testing.assert_close(differences[0], differences[1], atol=1e-6, rtol=1e-3)
    # Retain the original fp32 tolerance against an accurate derivative; do not
    # compare to a coarse finite difference across interpolation discontinuities.
    torch.testing.assert_close(gradient.double(), expected, atol=5e-4, rtol=.05)
    with torch.no_grad():
        torch.testing.assert_close(energy(parameters.detach()).double(), value.detach(), atol=1e-5, rtol=1e-4)
    return {'method': 'fp64_native_energy_copy_with_two_local_finite_differences',
            'execution_precision_unchanged': 'fp32', 'boundary_margin': margin, 'step': h,
            'fd_fp64': differences[0].cpu().tolist(), 'fd_half_fp64': differences[1].cpu().tolist(),
            'gradient_fp64': expected.cpu().tolist(), 'reference_atol': 1e-6, 'reference_rtol': 1e-3,
            'fp32_atol': 5e-4, 'fp32_rtol': .05}
