"""Numerical identities and rejection paths, executed only in Slurm validation.

Tiny tensors establish algebra/gradient correctness, never scientific performance.
The separate real-data validation runs full FIT32 and complete CAL8 searches.
"""
import copy
import itertools
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from v10 import adapter as A
from v10.supplement import checkpoints as CP
from v10.supplement.contract import candidates, schedule
from v10.supplement.learned import mlp_energy_rows
from v10.supplement.mind import mind2d
from v10.supplement.registration import FiniteEnergy, intensity_range
from v10.supplement.rulsif import ExactRuLSIF, gaussian_features, kernel_moments
from v10.supplement.training import FixedTrainer

DEVICE = os.environ.get('V10_TEST_DEVICE', 'cpu')


def rand(*shape, grad=False):
    return torch.randn(*shape, device=DEVICE, dtype=torch.float64, requires_grad=grad)


@pytest.mark.parametrize('mode', ['v', 'u'])
def test_exact_kernel_moments_enumerated_values_and_gradients(mode):
    torch.manual_seed(47)
    a, b = rand(7, 4, grad=True).sigmoid(), rand(7, 4, grad=True).sigmoid()
    z = a * b
    all_pairs = torch.stack([a[i] * b[j] for i in range(7) for j in range(7)
                             if mode == 'v' or i != j])
    expected = z.mean(0), z.T @ z / 7, all_pairs.mean(0), all_pairs.T @ all_pairs / len(all_pairs)
    actual = kernel_moments(a, b, mode)
    for x, y in zip(actual, expected):
        torch.testing.assert_close(x, y, atol=1e-12, rtol=1e-10)
    weights = [rand(*x.shape) for x in actual]
    ga = torch.autograd.grad(sum((x * w).sum() for x, w in zip(actual, weights)), (a, b), retain_graph=True)
    gb = torch.autograd.grad(sum((x * w).sum() for x, w in zip(expected, weights)), (a, b))
    for x, y in zip(ga, gb): torch.testing.assert_close(x, y, atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize('mode', ['v', 'u'])
def test_rulsif_fit_and_common_j_match_explicit_product(mode):
    torch.manual_seed(18)
    cx, cy = rand(5, 3), rand(5, 2)
    model = ExactRuLSIF(cx, cy, 1.7, 1e-3)
    scenes = [(rand(n, 3), rand(n, 2)) for n in (6, 9)]
    hs, ps = [], []
    for x, y in scenes:
        a, b = model.features(x, y)
        p = a * b
        q = torch.stack([a[i] * b[j] for i in range(len(a)) for j in range(len(b))
                         if mode == 'v' or i != j])
        ps.append(p.mean(0)); hs.append((p.T @ p / len(p) + q.T @ q / len(q)) / 2)
    h, hp = torch.stack(hs).mean(0), torch.stack(ps).mean(0)
    expected = torch.linalg.solve(h + 1e-3 * torch.eye(5, dtype=h.dtype, device=DEVICE), hp)
    info = model.fit(scenes, q_mode=mode)
    torch.testing.assert_close(model.coefficients, expected, atol=1e-11, rtol=1e-9)
    assert info['scene_count'] == 2 and info['linear_solve_residual'] < 1e-12
    x, y = rand(6, 3, grad=True), rand(6, 2, grad=True)
    a, b = model.features(x, y)
    tp = (a * b) @ model.coefficients - 1
    tq = torch.stack([(a[i] * b[j]) @ model.coefficients - 1 for i in range(6) for j in range(6)
                      if mode == 'v' or i != j])
    reference = (tp - tp.square()/2).mean() - (tq + tq.square()/2).mean()
    actual = model.common_j(x, y, q_mode=mode)
    torch.testing.assert_close(actual, reference, atol=1e-11, rtol=1e-9)
    ga = torch.autograd.grad(actual, (x, y), retain_graph=True)
    gb = torch.autograd.grad(reference, (x, y))
    for aa, bb in zip(ga, gb): torch.testing.assert_close(aa, bb, atol=1e-11, rtol=1e-9)


def test_rulsif_does_not_clip_negative_coefficients_or_ratio():
    centres = torch.zeros(2, 1, dtype=torch.float64, device=DEVICE)
    model = ExactRuLSIF(centres, centres, 1., 1e-3)
    model.coefficients.copy_(torch.tensor([-3., 1.], device=DEVICE)); model.fitted.fill_(True)
    torch.testing.assert_close(model(centres, centres), centres.new_full((2,), -2.))
    with pytest.raises(ValueError): kernel_moments(centres[:1], centres[:1], 'u')
    for width in (0., -1., float('nan')):
        with pytest.raises(ValueError): gaussian_features(centres, centres, width)


def test_mlp_permutation_average_value_and_gradient():
    torch.manual_seed(81)
    critic = nn.Sequential(nn.Linear(6, 9), nn.Tanh(), nn.Linear(9, 1)).to(DEVICE).double()
    u, v = rand(4, 3, grad=True), rand(2, 4, 3, grad=True)
    perms = torch.tensor(list(itertools.permutations(range(4))), device=DEVICE)
    actual = mlp_energy_rows(critic, u, v, perms)
    tp = critic(torch.cat((u[None].expand(2, -1, -1), v), -1)).squeeze(-1).tanh()
    reference = []
    for row in v:
        pairs = torch.stack([torch.cat((ui, vj)) for ui in u for vj in row])
        tq = critic(pairs).squeeze(-1).tanh()
        reference.append(-(tq + tq.square()/2).mean())
    reference = (tp - tp.square()/2).mean(-1) + torch.stack(reference)
    torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-10)
    ga = torch.autograd.grad(actual.sum(), (u, v), retain_graph=True)
    gb = torch.autograd.grad(reference.sum(), (u, v))
    for aa, bb in zip(ga, gb): torch.testing.assert_close(aa, bb, atol=1e-12, rtol=1e-10)
    one = mlp_energy_rows(critic, u, v, perms[:1])
    repeated = mlp_energy_rows(critic, u, v, perms[:1].expand(8, -1))
    torch.testing.assert_close(one, repeated, atol=1e-12, rtol=1e-10)


def test_infonce_raw_logits_match_host_and_explicit_cross_entropy():
    torch.manual_seed(91)
    critic = nn.Sequential(nn.Linear(6, 7), nn.Tanh(), nn.Linear(7, 1)).to(DEVICE).double()
    u, v = rand(2, 5, 3, grad=True), rand(2, 5, 3, grad=True)
    loss, j = A.mlp_objective('InfoNCE_host', critic, u, v)
    matrix = torch.stack([torch.stack([torch.stack([critic(torch.cat((u[s, i], v[s, k])))[0]
                         for k in range(5)]) for i in range(5)]) for s in range(2)])
    reference = (matrix.logsumexp(-1) - matrix.diagonal(dim1=-2, dim2=-1)).mean()
    host = A.P.loss_fn(torch, F, 'infonce', None, None, matrix)
    assert j is None
    torch.testing.assert_close(loss, host, atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(loss, reference, atol=1e-12, rtol=1e-10)
    ga = torch.autograd.grad(loss, (u, v), retain_graph=True)
    gb = torch.autograd.grad(reference, (u, v))
    for aa, bb in zip(ga, gb): torch.testing.assert_close(aa, bb, atol=1e-12, rtol=1e-10)


def test_mind_constant_affine_intensity_translation_and_gradient():
    torch.manual_seed(17)
    constant = torch.ones(1, 17, 19, dtype=torch.float64, device=DEVICE)
    assert torch.equal(mind2d(constant), constant.expand(4, -1, -1))
    x = rand(1, 17, 19, grad=True)
    d = mind2d(x)
    torch.testing.assert_close(d, mind2d(2.3 * x + 1.7), atol=1e-11, rtol=1e-9)
    shifted = mind2d(torch.roll(x, 1, -1))
    torch.testing.assert_close(shifted[:, 3:-3, 3:-3], torch.roll(d, 1, -1)[:, 3:-3, 3:-3],
                               atol=1e-12, rtol=1e-10)
    assert torch.isfinite(torch.autograd.grad(d.square().mean(), x)[0]).all()
    small = rand(1, 7, 8, grad=True)
    assert torch.autograd.gradcheck(lambda y: mind2d(y)[:, 2:-2, 2:-2], (small,), fast_mode=True)


def test_intensity_and_finite_energy_reject_degeneracy():
    x = torch.arange(100, dtype=torch.float32, device=DEVICE).reshape(1, 10, 10)
    assert intensity_range(x) == tuple(np.percentile(np.arange(100), (0.5, 99.5)))
    with pytest.raises(ValueError): intensity_range(torch.ones_like(x))
    with pytest.raises(FloatingPointError): FiniteEnergy.checked(x * float('nan'))


def test_fixed_checkpoint_configuration_identity_and_content_guards(tmp_path):
    candidate = candidates('MIND_2D_adapted')[0]
    trainer = FixedTrainer('MIND_2D_adapted', candidate, 1117, DEVICE)
    identity = {'family': 'MIND_2D_adapted', 'phase': 'VALIDATION', 'seed': 1117}
    path = tmp_path / 'fixed.pt'
    CP.save(trainer, path, identity); CP.load(trainer, path, identity)
    assert schedule('MIND_2D_adapted')['steps'] == 0
    with pytest.raises(RuntimeError): CP.load(trainer, path, {**identity, 'seed': 1})
    bad = copy.deepcopy(candidate); bad['descriptor']['sigma'] = 1.2
    with pytest.raises(RuntimeError): CP.load(FixedTrainer('MIND_2D_adapted', bad, 1117, DEVICE), path, identity)
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(RuntimeError): CP.load(trainer, path, identity)
