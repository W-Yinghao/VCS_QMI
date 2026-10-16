"""Unit tests of src/v10/adapter.py and the T01 / T02 script helpers on SYNTHETIC tensors with tiny encoders (no image read).

Covered here: host pin + V9-writer tripwires, the data gate (refusals happen before any read), the host batch RNG (equal to the
independent T00 replay), MF / MLP objectives, the trainer (one step per method, W projection, W weight decay 0, M1 gap logging),
the registration energy on the host warp chain (frozen contract, batched vs package backend, fixed-W state hash, fp64 finite
differences at the lattice transforms used by T01), checkpoint round trip, and the merge / budget fallbacks.
NOT covered (T01 / T02 jobs): the real encoders on real FIT / DEV images, CUDA, timing.
"""
from __future__ import annotations

import copy
import importlib.util
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from v10 import adapter as A
from v10 import mf_vcs as MF

REPO = Path(__file__).resolve().parents[2]


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"v10_script_{name}", REPO / "scripts" / "v10" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T01 = load_script("t01_validate")
T02 = load_script("t02_timing")


def tiny_models(nn_):
    def enc(cin):
        return nn_.Sequential(nn_.Conv2d(cin, 8, 3, padding=1), nn_.GELU(), nn_.Conv2d(8, 16, 3, padding=1))
    return enc(3), enc(1), nn_.Sequential(nn_.Linear(32, 16), nn_.GELU(), nn_.Linear(16, 1))


def smooth_image(c, seed, size=512):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, c, size, size, generator=g)
    for _ in range(3):
        x = F.avg_pool2d(x, 5, stride=1, padding=2)
    return (x / x.std())[0]


@pytest.fixture(scope="module")
def synthetic_imgs():
    return {f"S{i}": (smooth_image(3, 10 + i), smooth_image(1, 20 + i)) for i in range(4)}


# ------------------------------------------------------------------------------------------------------------ host, data gate, RNG
def test_host_pinned_and_writers_tripwired():
    assert A.host_code_hash() == A.HOST_CODE_HASH == "6c6f094684fbc50a"
    for mod, _, names in A._V9_WRITERS:
        for n in names:
            with pytest.raises(RuntimeError, match="must not be called from V10"):
                getattr(mod, n)()
    assert A.Q.P is A.P and A.G.P is A.P


def test_data_guard_refuses_before_any_read():
    guard = A.DataGuard()
    held, dev_cores = guard.held(), guard.included("DEV")
    assert len(held) == 134 and len(guard.included("FIT")) == 32 and len(guard.included("CAL")) == 8 and len(dev_cores) == 32
    with pytest.raises(PermissionError):
        guard.check(held[0], {"FIT", "CAL", "DEV"}, "probe")
    with pytest.raises(PermissionError):
        guard.check(dev_cores[0], {"FIT"}, "probe")
    with pytest.raises(PermissionError):
        guard.check("XX_nope", {"FIT"}, "probe")
    with pytest.raises(PermissionError):
        guard.check(dev_cores[0], {"TEST"}, "probe")
    with pytest.raises(PermissionError):
        A.load_learned_inputs(guard, [held[0]], {"FIT", "DEV"}, "probe", "cpu")
    with pytest.raises(PermissionError):
        A.GuardedInputs(guard, {"FIT"}, "probe", {}, "cpu").reference(dev_cores[0])
    assert guard.reads == []
    path = guard.check(guard.included("FIT")[0], {"FIT"}, "probe")      # granted: path only, nothing loaded here
    assert path.name.endswith(".npz") and "R6_REG/cache" in str(path) and len(guard.reads) == 1


def test_host_batch_rng_equals_independent_t00_replay():
    guard = A.DataGuard()
    fit = A.fit_cores(guard, 1.0, 901)
    stats = A.replay_permutation_fixed_points(fit, 901, 200)
    # numbers of /projects/.../outputs_v10/T00/host_audit_evidence/perm_fixed_points.json (independent numpy replay of G.cmd_train)
    assert stats == {"n_sc": 16, "n_perms": 3200, "mean_fixed_points": 1.053125, "share_with_>=1_fixed_point": 0.6534375,
                     "distinct_cores_per_step": True}
    hb = A.HostBatchSampler(fit, 901).draw()
    assert hb.yx.shape == (16, 128, 2) and hb.perm.shape == (16, 128) and hb.yx.min() >= 16 and hb.yx.max() <= 143


# ------------------------------------------------------------------------------------------------------------ objectives
def test_mf_head_sides_equal_features_and_energy_rows_equal_backend():
    torch.manual_seed(0)
    head = A.MFHead(16, 16, 8, 8).double()
    with torch.no_grad():
        head.W.uniform_(-1, 1)
    x, y = torch.randn(64, 16, dtype=torch.float64), torch.randn(3, 64, 16, dtype=torch.float64)
    a, b0 = head.features(x, y[0])
    assert torch.equal(a, head.features_a(x)) and torch.equal(b0, head.features_b(y[0]))
    B = head.features_b(y)
    perm = torch.randperm(64)
    perms = torch.stack([perm] + [torch.randperm(64) for _ in range(7)])
    for est in ("moments", "perm1", "perms", "dense"):
        j, t = A.mf_energy_rows(a, B, head.W, est, perm=perm, perms=perms, dense_block=16, return_terms=True)
        for c in range(3):
            sc = MF.SceneBatch(a, B[c], "v")
            if est in ("moments", "dense"):
                ref = MF.objective_terms([sc], head.W, weights=[1.0])
            else:
                ref = MF.sampled_permutation_terms([sc], head.W, weights=[1.0], permutations=[perm[None] if est == "perm1" else perms])
            for k in ("p1", "p2", "q1", "q2"):
                assert abs(float(t[k][c].detach()) - float(getattr(ref, k).detach())) <= 1e-13
            assert abs(float(j[c].detach()) - float(ref.j.detach())) <= 1e-13


def test_mlp_objectives_dense_perm_average_blocks_and_host_formula():
    torch.manual_seed(1)
    cr = tiny_models(nn)[2].double()
    du, dv = torch.randn(2, 4, 16, dtype=torch.float64, requires_grad=True), torch.randn(2, 4, 16, dtype=torch.float64, requires_grad=True)
    perms = torch.tensor(list(itertools.permutations(range(4))))
    _, jd = A.mlp_objective("MLP_dense", cr, du, dv)
    # all-permutation average, scene by scene (independent permutations per scene average to the same V)
    jp = torch.stack([A.mlp_objective("MLP_perm1", cr, du, dv, perm=torch.stack([p, p]))[1] for p in perms]).mean()
    assert abs(float((jd - jp).detach())) <= 1e-13
    _, jb = A.mlp_objective("MLP_dense", cr, du, dv, block=3, checkpoint=True)
    gd, gb = torch.autograd.grad(jd, (du, dv)), torch.autograd.grad(jb, (du, dv))
    assert abs(float(jd - jb)) <= 1e-13 and all(float((x - y).abs().max()) <= 1e-13 for x, y in zip(gd, gb))
    perm = torch.stack([torch.randperm(4), torch.randperm(4)])
    loss1, j1 = A.mlp_objective("MLP_perm1", cr, du, dv, perm=perm)
    dvq = torch.stack([dv[s, perm[s]] for s in range(2)])
    host = A.P.loss_fn(torch, F, "vcs", cr(torch.cat([du, dv], -1)).squeeze(-1), cr(torch.cat([du, dvq], -1)).squeeze(-1))
    assert torch.equal(loss1, host)
    _, j8 = A.mlp_objective("MLP_perm8", cr, du, dv, perms=perm[:, None].expand(-1, 8, -1).contiguous())
    assert abs(float(j8 - j1)) <= 1e-13


# ------------------------------------------------------------------------------------------------------------ trainer
def make_trainer(method, imgs, **kw):
    cfg = A.TrainConfig(method=method, seed=7, n_sc=2, K=8, **kw)
    return A.Trainer(cfg, imgs, sorted(imgs), "cpu", build_models=tiny_models)


@pytest.mark.parametrize("method", ["MF_perm1", "MF_M0", "MF_M1", "MLP_perm1", "MLP_perm8", "MLP_dense"])
def test_trainer_one_step_each_method(method, synthetic_imgs):
    t = make_trainer(method, synthetic_imgs)
    r = t.step(t.draw())
    assert math.isfinite(r["loss"]) and math.isfinite(r["grad_norm_preclip"])
    if method.startswith("MF_"):
        assert r["w_absmax"] <= 1.0
    if method == "MF_M1":
        assert {"m1_gap", "m1_iterations", "m1_converged", "m1_status"} <= set(r) and r["m1_gap_scope"] == "inner_pre_outer_batch"
        assert all(p is not t.head.W for g in t.opt.param_groups for p in g["params"]) and not t.head.W.requires_grad


def test_trainer_same_batch_and_positive_terms_across_methods(synthetic_imgs):
    ta, tb = make_trainer("MF_perm1", synthetic_imgs, w_init="uniform"), make_trainer("MF_M0", synthetic_imgs, w_init="uniform")
    ba, bb = ta.draw(), tb.draw()
    assert ba[0].cores == bb[0].cores and np.array_equal(ba[0].perm, bb[0].perm) and torch.equal(ba[1], bb[1])
    _, oa = ta.forward(*ba)
    _, ob = tb.forward(*bb)
    assert torch.equal(oa["terms"].p1, ob["terms"].p1) and torch.equal(oa["terms"].p2, ob["terms"].p2)
    assert [s.scene_id for s in oa["scenes"]] == ["0", "1"]


def test_trainer_projection_weight_decay_and_encoder_updates(synthetic_imgs):
    t = make_trainer("MF_M0", synthetic_imgs)
    batch = t.draw()
    with torch.no_grad():
        _, out = t.forward(*batch)
        sol = MF.profiled_objective(out["scenes"], weights=A.host_weights(2), tolerance=1e-6, max_iterations=3000, require_converged=False)
        t.head.W.copy_(sol.solution.w)
    assert int((t.head.W.abs() == 1).sum()) > 0
    enc_before = [p.detach().clone() for p in t.eU.parameters()]
    r = t.step(batch)
    assert r["w_absmax_preprojection"] > 1.0 and r["w_absmax"] <= 1.0
    assert max(float((p - b).abs().max()) for p, b in zip(t.eU.parameters(), enc_before)) > 0
    t = make_trainer("MF_M0", synthetic_imgs, w_init="uniform")
    W0, f0 = t.head.W.detach().clone(), t.head.left.weight.detach().clone()
    for p in t.clip_params:
        p.grad = torch.zeros_like(p)
    t.opt.step()
    assert torch.equal(t.head.W, W0)
    assert torch.allclose(t.head.left.weight, f0 * (1 - 1e-3 * 1e-4), rtol=0, atol=1e-9)
    t = make_trainer("MF_perm1", synthetic_imgs)
    W0 = t.head.W.detach().clone()
    t.step(t.draw())
    assert float((t.head.W - W0).abs().max()) > 0 and all(float(p.grad.abs().max()) == 0 for p in t.eU.parameters() if p.grad is not None)


# ------------------------------------------------------------------------------------------------------------ registration energy
@pytest.fixture(scope="module")
def reg_setup():
    torch.manual_seed(3)
    eU, eV, cr = tiny_models(nn)
    head = A.MFHead(16, 16, 8, 8)
    with torch.no_grad():
        head.W.uniform_(-1, 1)
        for lin in (head.left, head.right):
            lin.weight.mul_(10)
    A.freeze_for_test(eU, eV, cr, head)
    ref_img, mov = smooth_image(3, 31), smooth_image(1, 32)
    with torch.no_grad():
        DU = eU(ref_img[None])[0]
    return eU, eV, cr, head, ref_img, DU, mov


def rng0():
    return np.random.default_rng(A.Q.seed_of("R6-REG-P2|pos|SYN|0"))


def test_energy_frozen_contract_host_chain_and_package_equality(reg_setup):
    eU, eV, cr, head, ref_img, DU, mov = reg_setup
    trainable = A.MFHead(16, 16, 8, 8)
    with pytest.raises(ValueError):
        A.MFRigidEnergy(DU, mov, rng0(), eV=eV, head=trainable)
    host = A.Q.RigidEnergy(torch, F, "critic", DU, mov, rng0(), eV=eV, critic=cr)
    e = A.MFRigidEnergy(DU, mov, rng0(), eV=eV, head=head, estimator="moments", record=True)
    assert torch.equal(e.perm, host.perm) and torch.equal(e.x, host.x)
    p = torch.tensor([3.0, 4.5, -2.25], requires_grad=True)
    E = e(p)
    assert torch.equal(e.last["dv"][0], A.host_moving_descriptors(e, p))
    sc = MF.SceneBatch(e.A, e.last["B"][0], "v")
    pm, pdn = MF.objective_terms([sc], head.W, weights=[1.0]), MF.dense_objective_terms([sc], head.W, weights=[1.0], chunk_size=1024)
    assert abs(float(E - pm.j)) <= 1e-5 and abs(float(pm.j - pdn.j)) <= 1e-5
    g1, g2 = torch.autograd.grad(E, p, retain_graph=True)[0], torch.autograd.grad(pdn.j, p)[0]
    assert float((g1 - g2).abs().max()) <= 1e-5 + 1e-4 * float(g2.abs().max()) and float(g1.abs().max()) > 0


def test_fixed_w_state_hash_unchanged_across_candidates(reg_setup):
    eU, eV, cr, head, ref_img, DU, mov = reg_setup
    h0 = A.state_hash(eU, eV, head)
    e = A.MFRigidEnergy(DU, mov, rng0(), eV=eV, head=head)
    tt = torch.tensor(A.Q.grid_points()[1][:64], dtype=torch.float32)
    with torch.no_grad():
        v = e.grid_values(-6.0, tt)
    assert v.shape == (64,) and bool(torch.isfinite(v).all())
    for k in range(3):
        p = torch.tensor([1.0 * k, 2.0, -1.0 * k], requires_grad=True)
        (-e(p)).backward()
        assert p.grad is not None and float(p.grad.abs().max()) > 0
    assert A.state_hash(eU, eV, head) == h0
    assert all(q.grad is None and not q.requires_grad for m in (eV, head) for q in m.parameters())


def test_geometry_fd_machinery_on_lattice_transforms(reg_setup):
    """The T01 fp64 finite-difference machinery (lattice transforms, kink counter, pre-registered tolerance) on the host chain
    with tiny modules: zero kink crossings, FD within FD_ATOL + FD_RTOL * scale for every term and both steps."""
    eU, eV, cr, head, ref_img, DU, mov = reg_setup
    specs = T01.geometry_specs()
    eV64, head64 = copy.deepcopy(eV).double(), copy.deepcopy(head).double()
    e = A.MFRigidEnergy(DU.double(), mov, rng0(), eV=eV64, head=head64, dtype=torch.float64, record=True)
    for s in specs:
        assert s["margin"] >= s["expected_margin"] - 1e-9 and abs(s["theta_deg"]) <= 30 and math.hypot(*s["p"][1:]) <= 51
        for d in range(3):
            for h in T01.FD_STEPS:
                assert T01.kink_crossings(s["p"], d, h) == 0
        p = torch.tensor(s["p"], dtype=torch.float64, requires_grad=True)
        tv = T01.term_values(e, p)
        g = {k: torch.autograd.grad(v, p, retain_graph=True)[0].numpy() for k, v in tv.items() if k in ("J_moments", "q1_moments", "q2_moments", "p1", "J_dense")}
        with torch.no_grad():
            for d in range(3):
                for h in T01.FD_STEPS:
                    ev = np.zeros(3)
                    ev[d] = h
                    plus = T01.term_values(e, torch.tensor(s["p"] + ev, dtype=torch.float64))
                    minus = T01.term_values(e, torch.tensor(s["p"] - ev, dtype=torch.float64))
                    for k, gk in g.items():
                        fd = (float(plus[k]) - float(minus[k])) / (2 * h)
                        assert abs(fd - gk[d]) <= T01.FD_ATOL + T01.FD_RTOL * float(np.abs(gk).max()), (s["name"], k, d, h, fd, gk[d])


def test_checkpoint_pair_roundtrip(reg_setup, tmp_path):
    eU, eV, cr, head, ref_img, DU, mov = reg_setup
    path = tmp_path / "pair.pt"
    A.save_pair(path, eU, eV, head, {"purpose": "unit test", "K": 8})
    eU2, eV2, head2, meta = A.load_pair(path, "cpu", build_models=tiny_models)
    A.freeze_for_test(eU2, eV2, head2)
    assert meta == {"purpose": "unit test", "K": 8} and A.state_hash(eU2, eV2, head2) == A.state_hash(eU, eV, head)
    with torch.no_grad():
        DU2 = eU2(ref_img[None])[0]
    p = torch.tensor([2.0, -1.5, 3.25])
    with torch.no_grad():
        a = A.MFRigidEnergy(DU, mov, rng0(), eV=eV, head=head)(p)
        b = A.MFRigidEnergy(DU2, mov, rng0(), eV=eV2, head=head2)(p)
    assert torch.equal(a, b)


class AnalyticEnergy:
    """F(theta, t) = -((theta - theta*) rho)^2 - |t - t*|^2 with the RigidEnergy interface used by Q.optimise_rigid."""

    def __init__(self, truth):
        self.th, self.t = math.radians(truth[0]), torch.tensor(truth[1:], dtype=torch.float32)

    def grid_values(self, th_deg, ts):
        return -((math.radians(th_deg) - self.th) * A.Q.RHO) ** 2 - ((ts - self.t) ** 2).sum(-1)

    def __call__(self, p):
        return -(p[0] - self.th * A.Q.RHO) ** 2 - ((p[1:3] - self.t) ** 2).sum()

    def value(self, th_deg, t):
        return float(-((math.radians(th_deg) - self.th) * A.Q.RHO) ** 2 - sum((a - b) ** 2 for a, b in zip(t, self.t.tolist())))


def test_registration_record_through_the_host_searcher():
    truth = (7.3, 12.25, -20.5)
    rec = A.registration_record(AnalyticEnergy(truth), "cpu", truth)
    assert rec["n_grid"] == 21 * 508 and rec["err_full"] < 0.5 and rec["rot_err_deg"] < 0.05 and rec["F_true"] == 0.0
    assert rec["F_hat"] <= 1e-6 and set(rec) == {"theta", "t", "theta_hat", "t_hat", "err_full", "rot_err_deg", "trans_err_full", "F_hat",
                                                 "F_true", "n_grid", "seconds"}
    assert A.Q.corner_error_full(5.0, (1.0, 2.0), 5.0, (1.0, 2.0)) == 0.0


# ------------------------------------------------------------------------------------------------------------ scripts
def test_t01_merge_without_parts_is_not_run(tmp_path):
    class Args:
        out_root = tmp_path
        repo_out = tmp_path / "repo"
    T01.merge(Args)
    tv = json.loads((tmp_path / "repo" / "TORCH_VALIDATION.json").read_text())
    gg = json.loads((tmp_path / "repo" / "GEOMETRY_GRADIENT_CHECK.json").read_text())
    assert tv["overall_status"] == "NOT_RUN" and gg["overall_status"] == "NOT_RUN" and tv["host_adapter_verified"] is False
    assert set(tv["parts"]) == {"verify_torch_cpu", "verify_torch_cuda", "core_tol_cpu", "core_tol_cuda", "adapter_cpu", "adapter_cuda"}


def test_t02_budget_without_timing_is_not_run_and_ledger_map_complete(tmp_path):
    class Args:
        repo_out = tmp_path
    assert T02.budget(Args) == 1
    assert json.loads((tmp_path / "BUDGET.json").read_text())["status"] == "NOT_RUN"
    units = T02.ledger_units()
    assert len(units) == 60 and all((T02.METHOD_MAP[m], n, k) in units for (n, k) in T02.GRID for m in T02.METHOD_MAP)


def test_python_code_hash_equals_job_header_pipeline():
    import subprocess
    sh = subprocess.run("git ls-files src/v10 scripts/v10 | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16", shell=True,
                        cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    assert sh == A.code_hash()
