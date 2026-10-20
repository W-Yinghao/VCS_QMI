"""RuLSIF_exact_v2: constant-basis algebra, unchanged v1 moments, support law, functional gate,
pre-registered grid and v1 immutability. Tiny or synthetic tensors only; no scientific data."""
import copy
import hashlib
import inspect
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from v10 import adapter as A
from v10.supplement import contract as C
from v10.supplement import rulsif_gate as G
from v10.supplement import rulsif_v2 as V2
from v10.supplement.registration import RuLSIFRigidEnergy, patches
from v10.supplement.rulsif import ExactRuLSIF, gaussian_features, kernel_moments
from v10.supplement.training import KernelTrainer

DEVICE = os.environ.get('V10_TEST_DEVICE', 'cpu')
REPO = Path(__file__).resolve().parents[2]


def rand(*shape, grad=False):
    return torch.randn(*shape, device=DEVICE, dtype=torch.float64, requires_grad=grad)


def explicit_design(model, x, y, mode):
    """Brute-force pair enumeration with an explicit constant column."""
    kx = gaussian_features(x, model.centres_left, model.bandwidth_left)
    ky = gaussian_features(y, model.centres_right, model.bandwidth_right)
    one = torch.ones(1, dtype=x.dtype, device=x.device)
    p = torch.stack([torch.cat((one, kx[i] * ky[i])) for i in range(len(x))])
    q = torch.stack([torch.cat((one, kx[i] * ky[j])) for i in range(len(x)) for j in range(len(y))
                     if mode == 'v' or i != j])
    return p, q


@pytest.mark.parametrize('mode', ['v', 'u'])
@pytest.mark.parametrize('penalize_constant', [False, True])
def test_constant_basis_solve_and_readout_match_bruteforce(mode, penalize_constant):
    torch.manual_seed(23)
    cx, cy = rand(5, 3), rand(5, 2)
    model = V2.ExactRuLSIFV2(cx, cy, 1.3, 0.7, 1e-3, penalize_constant=penalize_constant)
    assert model.coefficients.shape == (6,)
    scenes = [(rand(n, 3), rand(n, 2)) for n in (7, 9, 6)]
    hs, ps = [], []
    for x, y in scenes:
        p, q = explicit_design(model, x, y, mode)
        ps.append(p.mean(0)); hs.append((p.T @ p / len(p) + q.T @ q / len(q)) / 2)
    h, hp = torch.stack(hs).mean(0), torch.stack(ps).mean(0)
    penalty = torch.ones(6, dtype=h.dtype, device=DEVICE)
    if not penalize_constant:
        penalty[0] = 0
    expected = torch.linalg.solve(h + 1e-3 * torch.diag(penalty), hp)
    info = model.fit(scenes, q_mode=mode)
    torch.testing.assert_close(model.coefficients, expected, atol=1e-11, rtol=1e-9)
    assert info['coefficient_count'] == 6 and info['ridge_applies_to_constant'] is penalize_constant
    assert info['constant_coefficient'] == pytest.approx(float(expected[0]))
    assert info['J_fit_moments'] == pytest.approx(float(2 * hp @ expected - expected @ h @ expected - 1), abs=1e-12)
    if not penalize_constant:   # normal equation row 0: E_{(P+Q)/2}[g] = 1 exactly
        assert info['mixture_mean_of_g'] == pytest.approx(1.0, abs=1e-12)
        assert info['J_fit_moments'] >= 0                  # theta = (1, 0) gives J = 0 and no penalty
    # Stationary point of J(g - 1) - lambda sum_l penalty_l theta_l^2 on the FIT moments.
    theta = model.coefficients.detach().clone().requires_grad_(True)
    objective = 2 * hp @ theta - theta @ h @ theta - 1 - 1e-3 * (penalty * theta.square()).sum()
    assert torch.autograd.grad(objective, theta)[0].abs().max() < 1e-10
    x, y = rand(6, 3, grad=True), rand(6, 2, grad=True)
    p, q = explicit_design(model, x, y, mode)
    tp, tq = p @ model.coefficients - 1, q @ model.coefficients - 1
    reference = (tp - tp.square() / 2).mean() - (tq + tq.square() / 2).mean()
    actual = model.common_j(x, y, q_mode=mode)
    torch.testing.assert_close(actual, reference, atol=1e-11, rtol=1e-9)
    for a, b in zip(torch.autograd.grad(actual, (x, y), retain_graph=True), torch.autograd.grad(reference, (x, y))):
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-9)
    torch.testing.assert_close(model(x, y), p @ model.coefficients, atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize('mode', ['v', 'u'])
def test_v2_class_without_constant_and_equal_widths_is_v1(mode):
    torch.manual_seed(31)
    cx, cy = rand(4, 3), rand(4, 2)
    scenes = [(rand(n, 3), rand(n, 2)) for n in (6, 8)]
    v1 = ExactRuLSIF(cx, cy, 1.1, 1e-3); v1.fit(scenes, q_mode=mode)
    v2 = V2.ExactRuLSIFV2(cx, cy, 1.1, 1.1, 1e-3, constant=False); v2.fit(scenes, q_mode=mode)
    torch.testing.assert_close(v2.coefficients, v1.coefficients, atol=1e-13, rtol=1e-11)
    x, y = rand(5, 3), rand(5, 2)
    torch.testing.assert_close(v2.common_j(x, y, q_mode=mode), v1.common_j(x, y, q_mode=mode), atol=1e-13, rtol=1e-11)
    # Separate widths act on their own side only.
    v3 = V2.ExactRuLSIFV2(cx, cy, 0.4, 2.5, 1e-3)
    torch.testing.assert_close(v3.left_features(x)[:, 1:], gaussian_features(x, cx, 0.4), atol=0, rtol=0)
    torch.testing.assert_close(v3.right_features(y)[:, 1:], gaussian_features(y, cy, 2.5), atol=0, rtol=0)
    assert torch.equal(v3.left_features(x)[:, 0], torch.ones(5, dtype=x.dtype, device=DEVICE))


@pytest.mark.parametrize('mode', ['v', 'u'])
def test_exact_v_u_moments_unchanged_and_constant_block_is_marginal_means(mode):
    torch.manual_seed(5)
    a, b = rand(8, 4).sigmoid(), rand(8, 4).sigmoid()
    hp, hp2, hq, hq2 = kernel_moments(a, b, mode)
    ap, bp = V2.with_constant(a), V2.with_constant(b)
    xp, xp2, xq, xq2 = kernel_moments(ap, bp, mode)
    # v1 kernel block identical up to fp64 summation order (the BLAS blocking changes with the
    # extra column); constant entries are 1 and the P / Q means of the products.
    for v1, v2 in ((hp, xp[1:]), (hp2, xp2[1:, 1:]), (hq, xq[1:]), (hq2, xq2[1:, 1:])):
        torch.testing.assert_close(v2, v1, atol=1e-15, rtol=1e-13)
    assert float(xp[0]) == 1 and float(xq[0]) == pytest.approx(1, abs=1e-15)
    assert float(xp2[0, 0]) == 1 and float(xq2[0, 0]) == pytest.approx(1, abs=1e-15)
    torch.testing.assert_close(xp2[0, 1:], hp, atol=1e-15, rtol=1e-13)
    torch.testing.assert_close(xq2[0, 1:], hq, atol=1e-15, rtol=1e-13)
    pairs = [(i, j) for i in range(8) for j in range(8) if mode == 'v' or i != j]
    torch.testing.assert_close(hq, torch.stack([a[i] * b[j] for i, j in pairs]).mean(0), atol=1e-14, rtol=1e-12)


def test_constant_makes_zero_critic_attainable_and_v1_class_cannot():
    torch.manual_seed(3)
    x, y = rand(9, 3), rand(9, 2)
    v2 = V2.ExactRuLSIFV2(rand(4, 3), rand(4, 2), 1.0, 1.0, 1e-3)
    v2.coefficients.zero_(); v2.coefficients[0] = 1; v2.fitted.fill_(True)
    assert float(v2.common_j(x, y)) == pytest.approx(0.0, abs=1e-14)       # T = g - 1 = 0
    v1 = ExactRuLSIF(rand(4, 3), rand(4, 2), 1.0, 1e-3)
    v1.coefficients.zero_(); v1.fitted.fill_(True)
    assert float(v1.common_j(x, y)) == pytest.approx(-1.0, abs=1e-14)      # T = -1: J = -1


def test_v2_checkpoint_state_and_v1_incompatibility():
    cx, cy = rand(3, 2), rand(3, 2)
    v2 = V2.ExactRuLSIFV2(cx, cy, 1.0, 1.0, 1e-2)
    v2.fit([(rand(5, 2), rand(5, 2))])
    state = copy.deepcopy(v2.state_dict())
    other = V2.ExactRuLSIFV2(cx, cy, 1.0, 1.0, 1e-2)
    other.load_state_dict(state, strict=True)
    assert torch.equal(other.coefficients, v2.coefficients) and bool(other.fitted)
    with pytest.raises(RuntimeError):
        ExactRuLSIF(cx, cy, 1.0, 1e-2).load_state_dict(state, strict=True)


def stationary_field(g, sigma):
    """Stationary, isotropic smooth field: full-resolution white noise, separable Gaussian blur."""
    z = torch.randn(1, 1, 576, 576, generator=g)
    r = int(3 * sigma) + 1
    axis = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-axis ** 2 / (2 * sigma ** 2)); k = k / k.sum()
    z = torch.nn.functional.conv2d(torch.nn.functional.conv2d(z, k.view(1, 1, 1, -1)), k.view(1, 1, -1, 1))
    c = (z.shape[-1] - 512) // 2
    return z[0, 0, c:c + 512, c:c + 512]


def synthetic_images(seed, dependent=True, sigma=2.0):
    """512 x 512 robust-normalised pair: V = f(U) + 0.3 noise (aligned) or an independent V."""
    g = torch.Generator().manual_seed(seed)
    u = stationary_field(g, sigma)
    if dependent:
        z = u / u.std()
        noise = stationary_field(g, sigma)
        v = torch.tanh(2 * z) + 0.3 * z.square() + 0.3 * noise / noise.std()
    else:
        v = stationary_field(g, sigma)
    norm = lambda m: ((m - m.median()) / (m.quantile(.75) - m.quantile(.25))).numpy().astype(np.float32)
    return norm(u), norm(v)


class FakeGuard:
    def __init__(self):
        self.reads = []

    def check(self, core, roles, purpose):
        assert roles == {'FIT'}
        self.reads.append({'core': core, 'role': 'FIT', 'purpose': purpose})

    def log_rule(self, core, roles, purpose):
        assert roles == {'FIT'}
        self.reads.append({'core': core, 'role': 'FIT', 'purpose': purpose, 'kind': 'rule'})


class FakeInputs:
    def __init__(self, images):
        self.images = images

    def reference(self, core):
        return (torch.as_tensor(self.images[core][0], device=DEVICE)[None],)

    def moving(self, core, index):
        assert index == 0
        return (torch.as_tensor(self.images[core][1], device=DEVICE)[None],)


def gate_fixture(dependent, constant, multiplier, sigma=2.0, rule='per_modality_median'):
    images = {f'SYN_{k}': synthetic_images(100 + k, dependent, sigma) for k in range(5)}
    guard = FakeGuard()
    train, held = ['SYN_0', 'SYN_1', 'SYN_2'], ['SYN_3', 'SYN_4']
    model, scenes, meta = V2.fit_kernel_model(
        guard, train, constant=constant, support='disk_r129', centres_per_core=8, positions_per_core=512,
        bandwidth_rule=rule, bandwidth_multiplier=multiplier, ridge=1e-3, alpha=0.5, stream_tag='TEST|gate',
        device=DEVICE, purpose='synthetic', loader=lambda c, key: images[c])
    model.fit(scenes)
    energy = V2.RuLSIFV2RigidEnergy if isinstance(model, V2.ExactRuLSIFV2) else RuLSIFRigidEnergy
    rows = G.evaluate(model, energy, FakeInputs(images), guard, held)
    assert len(rows) == 20 and {r['core'] for r in rows} == set(held)
    assert {r['kind'] for r in guard.reads if 'kind' in r} == {'rule'}
    return G.summarize(rows)


THRESHOLDS = {'mean_F_true_gt': 0.0, 'F_true_positive_fraction_min': 0.75, 'true_beats_imposed_fraction_min': 0.75}


def test_functional_gate_passes_separable_fixture_and_fails_degenerate_ones():
    separable = gate_fixture(True, True, 1.0)
    assert G.verdict(separable, THRESHOLDS)['status'] == 'PASS', separable
    assert separable['mean_F_true'] > 0.01 and separable['frac_true_beats_local6'] == 1
    # v1's failure mode: localized kernels and no constant basis pin T near -1 (J < 0 at the truth).
    degenerate = gate_fixture(True, False, 0.2, rule='joint_median')
    verdict = G.verdict(degenerate, THRESHOLDS)
    assert verdict['status'] == 'FAIL' and degenerate['max_F_true'] < 0, degenerate
    assert not verdict['conditions']['mean_F_true_positive'] and not verdict['conditions']['F_true_positive_fraction']
    # The constant basis alone restores J >= 0 on the same fixture and bandwidth.
    repaired = gate_fixture(True, True, 0.2, rule='joint_median')
    assert G.verdict(repaired, THRESHOLDS)['status'] == 'PASS' and repaired['min_F_true'] > 0, repaired
    # No dependence at all: the truth is not special, the gate fails.
    independent = gate_fixture(False, True, 1.0, sigma=1.0)
    verdict = G.verdict(independent, THRESHOLDS)
    assert verdict['status'] == 'FAIL' and not verdict['conditions']['F_true_positive_fraction'], independent


def test_disk_support_is_the_registration_law_and_square_is_v1():
    seed = A.Q.seed_of('TEST|support')
    positions = V2.support_positions(np.random.default_rng(seed), A.P.N_POS, 'disk_r129')
    image = torch.zeros(1, A.Q.S, A.Q.S, device=DEVICE)
    host = A.Q.RigidEnergy(torch, torch.nn.functional, 'comir', image, image, np.random.default_rng(seed))
    assert torch.equal(torch.as_tensor(positions, dtype=torch.float32, device=DEVICE), host.x)
    radius = np.hypot(positions[:, 0] - A.Q.CEN, positions[:, 1] - A.Q.CEN)
    assert radius.max() <= A.Q.R_OMEGA and positions.min() - 4 >= 0 and positions.max() + 4 <= A.Q.S - 1
    square = V2.support_positions(np.random.default_rng(1), 1000, 'square_v1')
    assert square.min() >= 5 and square.max() < A.P.SIZE - 6
    with pytest.raises(ValueError):
        V2.support_positions(np.random.default_rng(1), 3, 'box')


def test_shared_fitting_reproduces_v1_rule_draw_order():
    """fit_scenes with v1's settings consumes the RNG exactly as v1's KernelTrainer source does."""
    images = {f'SYN_{k}': synthetic_images(7 + k) for k in range(4)}
    cores = sorted(images)
    scenes, cx, cy, medians, meta = V2.fit_scenes(
        FakeGuard(), cores, support='square_v1', centres_per_core=2, positions_per_core=64,
        stream_tag='V10|RuLSIF|FIT|1017', device=DEVICE, purpose='synthetic', loader=lambda c, key: images[c])
    g = np.random.default_rng(A.Q.seed_of('V10|RuLSIF|FIT|1017'))      # transcription of v1 L48-64
    reference = []
    for core in cores:
        left, right = images[core]
        pos = torch.as_tensor(g.uniform(5, A.P.SIZE - 6, (64, 2)), device=DEVICE, dtype=torch.float32)
        reference.append((patches(torch.as_tensor(left, device=DEVICE)[None], pos),
                          patches(torch.as_tensor(right, device=DEVICE)[None], pos)))
    rl, rr = [], []
    for left, right in reference:
        ids = torch.as_tensor(g.choice(len(left), 2, replace=False), device=DEVICE)
        rl.append(left[ids]); rr.append(right[ids])
    assert all(torch.equal(a, c) and torch.equal(b, d) for (a, b), (c, d) in zip(scenes, reference))
    assert torch.equal(cx, torch.cat(rl)) and torch.equal(cy, torch.cat(rr))
    d = torch.pdist(torch.cat((cx, cy), -1).double())
    assert medians['joint'] == float(d[d > 0].median())
    assert medians['left'] == float(torch.pdist(cx.double()).median())
    assert 'g.uniform(5, A.P.SIZE - 6' in inspect.getsource(KernelTrainer)   # v1 law named above


def test_registration_energy_v2_equals_exact_common_j_on_the_same_pairs():
    torch.manual_seed(11)
    u, v = (torch.as_tensor(m, device=DEVICE)[None] for m in synthetic_images(3))
    cx, cy = torch.randn(6, 81, device=DEVICE), torch.randn(6, 81, device=DEVICE)
    model = V2.ExactRuLSIFV2(cx, cy, 3.0, 9.0, 1e-3)
    model.coefficients.copy_(torch.linspace(-.5, 1.5, 7, device=DEVICE)); model.fitted.fill_(True)
    A.freeze_for_test(model)
    energy = V2.RuLSIFV2RigidEnergy(u, v, np.random.default_rng(4), ratio=model)
    theta, shift = torch.tensor([0.21], device=DEVICE), torch.tensor([[3.5, -7.25]], device=DEVICE)
    left = patches(u, energy.x)
    coords = A.Q.apply_T(torch, theta, shift, energy.patch_positions)
    right = A.Q.sample_pts(torch, torch.nn.functional, v, coords)[0, :, 0].reshape(len(energy.x), 81)
    torch.testing.assert_close(energy.batch(theta, shift)[0], model.common_j(left, right), atol=2e-5, rtol=1e-4)
    with pytest.raises(TypeError):
        V2.RuLSIFV2RigidEnergy(u, v, np.random.default_rng(4), ratio=ExactRuLSIF(cx, cy, 9.0, 1e-3))


def frozen_record(grid):
    return {'family': 'RuLSIF_exact_v2', 'status': 'FROZEN_BEFORE_CAL', 'candidate_grid': grid,
            'candidate_grid_sha256': C.digest(grid)}


def test_candidate_grid_is_read_from_the_frozen_record_only():
    row = {'order': 0, 'bandwidth_multiplier': 1.0, 'ridge': 1e-3, 'alpha': 0.5, 'centres_per_core': 4,
           'fit_positions_per_core': 2048, 'patch_radius': 4, 'constant_basis': True, 'fit_support': 'disk_r129'}
    grid = [dict(row, order=i, ridge=r) for i, r in enumerate((1e-4, 1e-3, 1e-2))]
    assert C.rulsif_v2_grid(frozen_record(grid)) == grid
    for bad in ({**frozen_record(grid), 'status': 'GATE_DEFINED_GRID_PENDING'},
                {**frozen_record(grid), 'candidate_grid_sha256': '0' * 64},
                frozen_record([dict(row, order=i) for i in range(10)]),
                frozen_record([dict(row, order=1)]),
                frozen_record([dict(row, alpha=1.0)]),
                frozen_record([dict(row, constant_basis=False)]),
                frozen_record([dict(row, fit_support='square_v1')])):
        with pytest.raises(RuntimeError):
            C.rulsif_v2_grid(bad)
    record = C.read(C.RULSIF_V2_RECORD)
    if record['status'] in C.RULSIF_V2_FROZEN:
        assert C.rulsif_v2_grid() == record['candidate_grid']
        expected = (C.read(C.RULSIF_V2_AMENDMENT)['eligible_candidates'] if C.RULSIF_V2_AMENDMENT.exists()
                    else record['candidate_grid'])
        assert C.candidates('RuLSIF_exact_v2') == expected
        assert C.schedule('RuLSIF_exact_v2') == C.schedule('RuLSIF_exact')
    else:
        with pytest.raises(RuntimeError):
            C.candidates('RuLSIF_exact_v2')


def amendment_for(grid, orders, record_sha='r' * 64):
    eligible = [row for row in grid if row['order'] in orders]
    return {'family': 'RuLSIF_exact_v2', 'status': C.RULSIF_V2_AMENDMENT_STATUS,
            'amends': {'preregistration_record': {'sha256': record_sha},
                       'original_gate_reports': {'RuLSIF_exact_v2': {'sha256': 'o' * 64, 'rows_sha256': 'w' * 64}}},
            'amended_rule': {'required_levels': ['deployed_position_heldout'], 'eligible_orders': list(orders),
                             'candidate_rule': 'level A'},
            'eligible_candidates': eligible, 'eligible_candidates_sha256': C.digest(eligible)}


def test_post_gate_amendment_selects_the_declared_eligible_subset():
    row = {'order': 0, 'bandwidth_multiplier': 1.0, 'ridge': 1e-3, 'alpha': 0.5, 'constant_basis': True,
           'fit_support': 'disk_r129', 'patch_radius': 4}
    grid = [dict(row, order=i, ridge=r) for i, r in enumerate((1e-4, 1e-3, 1e-2))]
    good = amendment_for(grid, [0, 1])
    assert C.rulsif_v2_eligible(grid, good, 'r' * 64) == grid[:2]
    assert C.rulsif_v2_eligible(grid, amendment_for(grid, [0, 2]), 'r' * 64) == [grid[0], grid[2]]
    tampered = copy.deepcopy(good); tampered['eligible_candidates'][1]['ridge'] = 0.5
    for bad in ({**good, 'status': 'DRAFT'}, amendment_for(grid, [0, 1], 'x' * 64), amendment_for(grid, [1, 0]),
                amendment_for(grid, []), amendment_for(grid, [0, 3]), tampered,
                {**good, 'eligible_candidates_sha256': '0' * 64},
                {**good, 'amended_rule': {**good['amended_rule'], 'required_levels': ['level_c']}}):
        with pytest.raises((RuntimeError, KeyError)):
            C.rulsif_v2_eligible(grid, bad, 'r' * 64)


V1_SOURCE_SHA256 = {
    # sha256 of inspect.getsource at ddd3bcb; v1 code paths must stay byte-identical.
    'rulsif.py': 'cfc90545a7317a0a77164bc77278f832557e94c0d2cc480db14469126844bdcf',
    'KernelTrainer': 'c7e5a00ac67d0440ac51359702aee62e44e0745f3ea033bd6523148096fcc5cf',
    'RuLSIFRigidEnergy': '67fd328d067ff549fe8f5abdb67c34eb36d873565489f9cfabb15fa13605a1ca',
}


def test_v1_family_definition_and_code_unchanged():
    from v10.supplement import rulsif
    digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
    assert digest(inspect.getsource(rulsif)) == V1_SOURCE_SHA256['rulsif.py']
    assert digest(inspect.getsource(KernelTrainer)) == V1_SOURCE_SHA256['KernelTrainer']
    assert digest(inspect.getsource(RuLSIFRigidEnergy)) == V1_SOURCE_SHA256['RuLSIFRigidEnergy']
    assert C.candidates('RuLSIF_exact') == [
        {'order': i, 'bandwidth_multiplier': w, 'ridge': r, 'alpha': 0.5, 'centres': 128,
         'fit_positions_per_core': 2048, 'patch_radius': 4}
        for i, (w, r) in enumerate((w, r) for w in (0.5, 1.0, 2.0) for r in (1e-4, 1e-3, 1e-2))]
    assert C.FAMILIES[:7] == ("MLP_VCS_perm8", "InfoNCE_host[J_perm1]", "MI_native", "NMI_native",
                              "CoMIR_native", "RuLSIF_exact", "MIND_2D_adapted")


def test_v2_release_verifier_accepts_complete_release_and_rejects_changes(tmp_path, monkeypatch):
    import json
    row = {'order': 0, 'bandwidth_rule': 'per_modality_median', 'bandwidth_multiplier': 0.5, 'ridge': 1e-3,
           'penalize_constant': False, 'alpha': 0.5, 'centres_per_core': 8, 'fit_positions_per_core': 2048,
           'patch_radius': 4, 'constant_basis': True, 'fit_support': 'disk_r129'}
    grid = [row]
    record = tmp_path / 'RECORD.json'
    record.write_text(json.dumps(frozen_record(grid)))
    monkeypatch.setattr(C, 'RULSIF_V2_RECORD', record)
    monkeypatch.setattr(C, 'RULSIF_V2_AMENDMENT', tmp_path / 'NO_AMENDMENT.json')
    project = tmp_path / 'project'
    (project / 'handoff').mkdir(parents=True)
    policy = project / 'handoff/POLICY.json'
    policy.write_text(json.dumps({'policy_id': 'test_policy', 'validated_supplement_overlap': True, 'normal_gpu_limit': 8}))
    (project / 'handoff/ACTIVE_SCHEDULING_POLICY.json').write_text(json.dumps({'policy_id': 'test_policy', 'policy': str(policy)}))
    monkeypatch.setattr(C, 'PROJECT', project)
    inventory = C.source_inventory(); code = C.digest(inventory)
    def write(name, value):
        (tmp_path / name).write_text(json.dumps(value))
        return {'path': str(tmp_path / name), 'sha256': C.sha(tmp_path / name)}
    fit_gate = {'stage': 'FIT_FUNCTIONAL_GATE', 'status': 'PASS', 'family': 'RuLSIF_exact_v2', 'code_sha256': code,
                'held_test_read': False, 'dev_read': False, 'preregistration_sha256': C.sha(record),
                'rule_mode': 'preregistered',
                'families': {'RuLSIF_exact_v2': {'status': 'PASS', 'candidates': [{'candidate': row, 'status': 'PASS'}]}}}
    gpu_gate = {'stage': 'T01_AND_EXECUTION_GATE', 'status': 'PASS', 'family': 'RuLSIF_exact_v2', 'code_sha256': code,
                'held_test_read': False, 'dev_read': False}
    lock = tmp_path / 'lock.json'; lock.write_text('{}')
    def release(**changes):
        value = {'status': C.RULSIF_V2_RELEASE_STATUS, 'family': 'RuLSIF_exact_v2', 'source_files': inventory,
                 'code_sha256': code, 'parent_protocol_sha256': C.PARENT_PROTOCOL_SHA256,
                 'families': {'RuLSIF_exact_v2': {'candidates': grid, 'schedule': C.schedule('RuLSIF_exact_v2')}},
                 'preregistration': {'path': str(record), 'sha256': C.sha(record)},
                 'acceptance': {'FIT_GATE': write('fit.json', changes.pop('fit', fit_gate)),
                                'GPU_GATE': write('gpu.json', changes.pop('gpu', gpu_gate))},
                 'scheduling_authorization': {'path': str(policy), 'sha256': C.sha(policy)},
                 'data_lock': {'path': str(lock), 'sha256': C.sha(lock)}}
        value.update(changes)
        path = tmp_path / 'RELEASE.json'; path.write_text(json.dumps(value))
        return path
    path = release()
    assert C.verify_release_v2(path, C.sha(path))['family'] == 'RuLSIF_exact_v2'
    with pytest.raises(RuntimeError):
        C.verify_release_v2(path, '0' * 64)
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    with pytest.raises(RuntimeError):
        C.require_release_v2(path)
    failing = copy.deepcopy(fit_gate); failing['families']['RuLSIF_exact_v2']['candidates'][0]['status'] = 'FAIL'
    for bad in (dict(status='FROZEN_VALIDATED'), dict(code_sha256='0' * 64), dict(fit=failing),
                dict(gpu={**gpu_gate, 'code_sha256': '1' * 64}), dict(gpu={**gpu_gate, 'dev_read': True}),
                dict(families={'RuLSIF_exact_v2': {'candidates': grid * 2, 'schedule': C.schedule('RuLSIF_exact_v2')}})):
        bad_path = release(**bad)
        with pytest.raises(RuntimeError):
            C.verify_release_v2(bad_path, C.sha(bad_path))
    # A preregistered-mode report is not accepted once an amendment exists, and vice versa.
    second = dict(row, order=1, ridge=1e-2)
    grid2 = [row, second]
    record.write_text(json.dumps(frozen_record(grid2)))
    amendment = tmp_path / 'AMENDMENT.json'
    amendment.write_text(json.dumps(amendment_for(grid2, [0], C.sha(record))))
    monkeypatch.setattr(C, 'RULSIF_V2_AMENDMENT', amendment)
    cited = C.read(amendment)['amends']['original_gate_reports']['RuLSIF_exact_v2']['sha256']
    amended_gate = {**fit_gate, 'preregistration_sha256': C.sha(record), 'rule_mode': 'amended',
                    'amendment_sha256': C.sha(amendment), 'required_levels': ['deployed_position_heldout'],
                    'reproduction': {'verdict_differences': [], 'original_report_sha256': cited},
                    'families': {'RuLSIF_exact_v2': {'status': 'PASS', 'eligible_orders': [0],
                                 'candidates': [{'candidate': row, 'status': 'PASS'}, {'candidate': second, 'status': 'FAIL'}]}}}
    families = {'RuLSIF_exact_v2': {'candidates': [row], 'schedule': C.schedule('RuLSIF_exact_v2')}}
    pinned = {'path': str(amendment), 'sha256': C.sha(amendment)}
    path = release(fit=amended_gate, families=families, amendment=pinned,
                   preregistration={'path': str(record), 'sha256': C.sha(record)})
    assert C.verify_release_v2(path, C.sha(path))['family'] == 'RuLSIF_exact_v2'
    differing = {**amended_gate, 'reproduction': {'verdict_differences': [{'order': 0}], 'original_report_sha256': cited}}
    wrong_set = copy.deepcopy(amended_gate); wrong_set['families']['RuLSIF_exact_v2']['eligible_orders'] = [0, 1]
    for bad in (dict(fit=fit_gate | {'preregistration_sha256': C.sha(record)}), dict(fit=differing), dict(fit=wrong_set)):
        bad_path = release(families=families, amendment=pinned, preregistration={'path': str(record), 'sha256': C.sha(record)}, **bad)
        with pytest.raises(RuntimeError):
            C.verify_release_v2(bad_path, C.sha(bad_path))
    bad_path = release(fit=amended_gate, families=families, preregistration={'path': str(record), 'sha256': C.sha(record)})
    with pytest.raises(RuntimeError):      # release without the amendment pinned
        C.verify_release_v2(bad_path, C.sha(bad_path))
    path = release(fit=amended_gate, families=families, amendment=pinned,
                   preregistration={'path': str(record), 'sha256': C.sha(record)})
    (project / 'handoff/ACTIVE_SCHEDULING_POLICY.json').write_text(json.dumps({'policy_id': 'other', 'policy': str(policy)}))
    with pytest.raises(RuntimeError):
        C.verify_release_v2(path, C.sha(path))


def test_truth_terms_add_up_to_the_energy_at_the_identity():
    for constant in (True, False):
        images = {f'SYN_{k}': synthetic_images(200 + k) for k in range(3)}
        model, scenes, _ = V2.fit_kernel_model(
            FakeGuard(), ['SYN_0', 'SYN_1'], constant=constant, support='disk_r129', centres_per_core=4,
            positions_per_core=256, bandwidth_rule='joint_median', bandwidth_multiplier=1.0, ridge=1e-3,
            alpha=0.5, stream_tag='TEST|terms', device=DEVICE, purpose='synthetic', loader=lambda c, key: images[c])
        model.fit(scenes); A.freeze_for_test(model)
        energy_class = V2.RuLSIFV2RigidEnergy if constant else RuLSIFRigidEnergy
        u, v = (torch.as_tensor(m, device=DEVICE)[None] for m in images['SYN_2'])
        energy = energy_class(u, v, np.random.default_rng(9), ratio=model)
        dependence, quadratic = G.truth_terms(energy)
        with torch.no_grad():
            value = float(energy.batch(torch.zeros(1, device=DEVICE), torch.zeros(1, 2, device=DEVICE))[0])
        assert dependence + quadratic == pytest.approx(value, abs=2e-5)


def test_copula_map_is_the_mid_rank_empirical_cdf_and_monotone_invariant():
    torch.manual_seed(4)
    x = torch.randn(1, 6, 7, device=DEVICE)
    x[0, 0, :3] = 0.25                                  # ties share their mid-rank
    u = V2.copula_map(x)
    flat, n = x.flatten(), x.numel()
    expected = torch.stack([((flat < v).sum() + 0.5 * (flat == v).sum()) / n for v in flat]).reshape(x.shape)
    torch.testing.assert_close(u, expected.to(u.dtype), atol=1e-7, rtol=0)
    torch.testing.assert_close(V2.copula_map(torch.exp(3 * x) + 2), u, atol=0, rtol=0)
    assert float(u.min()) > 0 and float(u.max()) < 1
