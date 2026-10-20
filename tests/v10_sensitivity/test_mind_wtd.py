"""D2 variant MIND_2D_warp_then_describe on synthetic images (CPU; no scientific data).

Checks that the variant keeps the frozen row's positions, permutation and reference
descriptor; equals the frozen energy exactly where both definitions coincide (zero
rotation, integer shift); fixes the axis-frame mismatch under rotation; has a correct
geometry gradient; uses the host's grid shift R(-theta) t; recovers a known rigid
transform with the unchanged common searcher; and keeps the frozen session bookkeeping.
"""
import copy
import math

import numpy as np
import pytest
import torch

from v10 import adapter as A
from v10.supplement.contract import candidates
from v10.supplement.mind import DescriptorRigidEnergy, MINDConfig, mind2d
from v10.supplement.registration import FiniteEnergy, RegistrationSession
from v10_sensitivity import mind_wtd as W

CONFIG = MINDConfig(**candidates("MIND_2D_adapted")[0]["descriptor"])
torch.set_num_threads(2)


def smooth_image(seed, size=512, blobs=80):
    g = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    image = np.zeros((size, size))
    for _ in range(blobs):
        cy, cx = g.uniform(0, size, 2)
        s = g.uniform(5, 24)
        image += g.normal() * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * s * s))
    image = (image - image.mean()) / image.std()
    return torch.tensor(image, dtype=torch.float32)[None]


@pytest.fixture(scope="module")
def images():
    return smooth_image(11), smooth_image(12)


def pair(reference_image, moving_image, seed=5):
    reference = mind2d(reference_image, CONFIG)
    frozen = DescriptorRigidEnergy(reference, mind2d(moving_image, CONFIG), np.random.default_rng(seed))
    variant = W.WarpThenDescribeEnergy(reference, moving_image, np.random.default_rng(seed), config=CONFIG)
    return frozen, variant


def as_double(energy, reference_image=None, moving_image=None):
    """fp64 copy (the native_gradient_check pattern): every geometry/value tensor cast, descriptors recomputed."""
    out = copy.copy(energy)
    out.x = energy.x.double()
    if reference_image is not None:
        out.u = A.Q.sample_pts(torch, torch.nn.functional, mind2d(reference_image.double(), CONFIG), out.x[None])[0]
    if isinstance(energy, W.WarpThenDescribeEnergy):
        out.mov = moving_image.double()
        out.box1, out.box2 = energy.box1.double(), energy.box2.double()
    else:
        out.mov = mind2d(moving_image.double(), CONFIG)
    return out


def test_positions_permutation_and_reference_equal_the_frozen_row(images):
    frozen, variant = pair(*images)
    assert torch.equal(frozen.x, variant.x) and torch.equal(frozen.perm, variant.perm)
    assert torch.equal(frozen.u, variant.u) and variant.u.shape == (A.P.N_POS, 4)
    assert variant.kind == "comir" and variant.n1 == 292 and variant.n2 == 394


def test_variant_equals_frozen_energy_at_zero_rotation_integer_shift(images):
    frozen, variant = pair(*images)
    f64, v64 = as_double(frozen, images[0], images[1]), as_double(variant, images[0], images[1])
    shifts = torch.tensor([[0.0, 0.0], [3.0, -5.0], [-17.0, 22.0], [36.0, 32.0]], dtype=torch.float64)
    zero = torch.zeros(len(shifts), dtype=torch.float64)
    torch.testing.assert_close(v64.batch(zero, shifts), f64.batch(zero, shifts), atol=1e-12, rtol=1e-9)
    # production fp32 agrees to fp32 resolution
    torch.testing.assert_close(variant.batch(zero.float(), shifts.float()), frozen.batch(zero.float(), shifts.float()),
                               atol=2e-5, rtol=1e-4)
    # grid stage == exact stage when the shift R(-0) t is an integer
    torch.testing.assert_close(v64.grid_values(0.0, shifts), v64.batch(zero, shifts), atol=1e-12, rtol=1e-9)


def test_warp_then_describe_removes_the_axis_frame_mismatch_under_rotation(images):
    reference_image = images[0]
    for theta, shift, expect_gain in ((24.0, (6.0, -9.0), True), (0.0, (6.0, -9.0), False)):
        moving_image = A.Q.perturb_map(torch, torch.nn.functional, reference_image, theta, shift)
        frozen, variant = pair(reference_image, moving_image)
        e_frozen, e_variant = frozen.value(theta, shift), variant.value(theta, shift)
        assert e_frozen <= 0 and e_variant <= 0
        if expect_gain:
            assert abs(e_variant) < 0.25 * abs(e_frozen), (e_variant, e_frozen)
        else:
            assert abs(e_variant - e_frozen) <= 1e-4 + 1e-3 * abs(e_frozen), (e_variant, e_frozen)


def test_geometry_gradient_against_fp64_finite_differences(images):
    moving_image = A.Q.perturb_map(torch, torch.nn.functional, images[0], 11.0, (4.0, -7.0))
    _, variant = pair(images[0], moving_image)
    v64 = as_double(variant, images[0], moving_image)
    p = torch.tensor([math.radians(7.3) * A.Q.RHO, 4.37, -2.71], dtype=torch.float64, requires_grad=True)
    gradient = torch.autograd.grad(v64(p), p)[0]
    h = 1e-6
    with torch.no_grad():
        fd = torch.stack([(v64(p.detach() + e) - v64(p.detach() - e)) / (2 * h) for e in torch.eye(3, dtype=torch.float64) * h])
    torch.testing.assert_close(gradient, fd, atol=1e-7, rtol=1e-3)
    p32 = p.detach().float().requires_grad_()
    gradient32 = torch.autograd.grad(variant(p32), p32)[0]
    # the supplement gate's fp32 geometry tolerance against the accurate fp64 derivative
    torch.testing.assert_close(gradient32.double(), gradient, atol=5e-4, rtol=0.05)


def test_grid_shift_is_the_host_inverse_rotation(images):
    theta, truth = 9.0, (12.0, -8.0)
    moving_image = A.Q.perturb_map(torch, torch.nn.functional, images[0], theta, truth)
    _, variant = pair(images[0], moving_image)
    _, grid = A.Q.grid_points()
    translations = torch.tensor(grid, dtype=torch.float32)
    with torch.no_grad():
        values = variant.grid_values(theta, translations)
    best = grid[int(values.argmax())]
    assert np.hypot(*(best - np.array(truth))) <= 4.0 + 1e-9, best
    assert variant.describe_calls == 1


def test_unchanged_common_searcher_recovers_a_known_transform(images):
    theta, truth = 17.0, (11.0, -23.0)
    moving_image = A.Q.perturb_map(torch, torch.nn.functional, images[0], theta, truth)
    _, variant = pair(images[0], moving_image)
    estimate, translation, n_grid, f_hat = A.Q.optimise_rigid(torch, FiniteEnergy(variant), torch.device("cpu"))
    assert n_grid == 21 * 508 and math.isfinite(f_hat)
    assert A.Q.corner_error_full(estimate, translation, theta, truth) < 9.0


class FakeInputs:
    def __init__(self, reference_image, moving_image):
        self.reference_image, self.moving_image = reference_image, moving_image
        self.ref, self.mov, self.calls = {}, {}, []

    def reference(self, core):
        self.calls.append(("reference", core))
        self.ref[core] = (self.reference_image, None)
        return self.ref[core]

    def moving(self, core, i):
        self.calls.append(("moving", core, i))
        self.mov[(core, i)] = (self.moving_image, None)
        return self.mov[(core, i)]


def session(cls, images, family="MIND_2D_adapted"):
    s = object.__new__(cls)
    s.family, s.trainer, s.device = family, None, torch.device("cpu")
    s.candidate, s.perturbations = candidates("MIND_2D_adapted")[0], {}
    s.inputs, s.core, s.reference = FakeInputs(*images), None, None
    return s


def test_session_keeps_the_frozen_bookkeeping_and_only_swaps_the_energy(images):
    assert issubclass(W.WarpThenDescribeSession, RegistrationSession)
    variant_session, frozen_session = session(W.WarpThenDescribeSession, images), session(RegistrationSession, images)
    energy, prep = variant_session.energy("1B_Z1", 0)
    frozen_energy, frozen_prep = frozen_session.energy("1B_Z1", 0)
    assert isinstance(energy, FiniteEnergy) and isinstance(energy.energy, W.WarpThenDescribeEnergy)
    assert isinstance(frozen_energy.energy, DescriptorRigidEnergy)
    assert set(prep) == set(frozen_prep) | {"variant"} and prep["variant"] == W.VARIANT
    assert torch.equal(energy.energy.x, frozen_energy.energy.x) and torch.equal(energy.energy.u, frozen_energy.energy.u)
    assert torch.equal(variant_session.reference, frozen_session.reference)
    assert variant_session.inputs.mov == {}
    variant_session.energy("1B_Z1", 1)
    assert [c for c in variant_session.inputs.calls if c[0] == "reference"] == [("reference", "1B_Z1")]
    with pytest.raises(ValueError):
        session(W.WarpThenDescribeSession, images, family="CoMIR_native").energy("1B_Z1", 0)
