"""EPFL host and geometry checks on CPU (src/v10/epfl/host.py, geometry.py): pixel units, support constants and sentinel poisoning,
warp conventions, the EPFL moving / training input paths (= REG-P3 Inputs3, not the BF log path), perturbations, FIT336 trainers, the
EPFL registration session, MF geometry gradients (fp64 finite differences) and the checkpoint round trip.  Real FIT / CAL windows through
the EPFL gate at stage GATE_CHECKS; no DEV window, no registration error.  V10_TEST_DEVICE=cuda runs the same code on a GPU."""
import hashlib
import json
import math
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from v10 import adapter as A
from v10.epfl import geometry as G, guard as GD, host as H

DEVICE = os.environ.get("V10_TEST_DEVICE", "cpu")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


@pytest.fixture(scope="module")
def guard():
    return GD.EPFLDataGuard(record=GD.ACTIVATION_RECORD, expect_record_sha256=sha(GD.ACTIVATION_RECORD))


def failed(rows):
    return [r["name"] for r in rows if r.get("required", True) and r["status"] != "PASS"]


def test_pixel_units_and_support_constants(guard):
    rows = G.pixel_unit_rows() + G.support_constant_rows() + G.manifest_unit_rows(guard)
    assert not failed(rows), failed(rows)
    assert H.native_px(42.0) == 10.5 and H.native_px(9.0) == 2.25
    assert abs(G.R_V - 255.04) < 0.01 and G.R_V <= A.Q.CEN


def test_tripwires_on_reg_p3_writers():
    for name in ("prepare_epfl", "cmd_perturb", "register_one", "write_cache", "main"):
        with pytest.raises(RuntimeError):
            getattr(H.P3, name)()


def test_candidates_and_schedules():
    assert [len(H.candidates(f)) for f in H.BF1_METHODS] == [3, 3, 9, 3]
    assert [c["cap"] for c in H.candidates("MF_VCS_M1")] == [10, 30, 100] * 3
    assert len(H.FAMILIES) == 11 and H.schedule("MF_VCS_M1")["cal_steps"] == list(range(250, 3001, 250))
    assert all(c["status"] == "DRAFT_REQUIRES_OWNER_DECISION" and c["centres_per_fit_window"] == 1 for c in H.candidates("RuLSIF_exact"))


def test_warp_conventions_on_a_fit_and_a_cal_window(guard):
    rows = G.warp_rows(guard, guard.included("FIT")[3], "FIT", DEVICE) + G.warp_rows(guard, guard.included("CAL")[2], "CAL", DEVICE)
    assert not failed(rows), [r for r in rows if r["status"] != "PASS"]


def test_epfl_inputs_equal_reg_p3_inputs3_and_perturbations(guard):
    cal = guard.included("CAL")
    pert = H.perturbations_for(guard, cal, "CAL", "test: perturbation rule")
    ref = json.loads((guard.r6_root / "perturbations.json").read_text())
    assert all(pert[w] == ref[w] for w in cal)
    w = cal[4]; dev = torch.device(DEVICE)
    mine = H.EPFLGuardedInputs(guard, {"CAL"}, "test", {w: pert[w]}, dev)
    theirs = H.P3.Inputs3(torch, F, guard.r6_root, {w: pert[w]}, dev)
    for i in (0, 5, 9):
        a, b = mine.moving(w, i), theirs.moving(w, i)
        assert torch.equal(a[0], b[0]) and a[0] is a[1]
    ra, rb = mine.reference(w), theirs.reference(w)
    assert torch.equal(ra[0], rb[0]) and torch.equal(ra[1], rb[1])
    assert {r["role"] for r in guard.reads} <= {"FIT", "CAL"}
    with pytest.raises(PermissionError):
        H.EPFLGuardedInputs(guard, {"CAL"}, "test", {"water_0021": [[0, 0, 0]]}, dev).moving("water_0021", 0)   # DEV window


def test_fit_set(guard):
    fit = H.fit_windows(guard)
    assert len(fit) == 336 and len({guard.unit_of(w) for w in fit}) == 330


def mlp_energy_factory(guard, window):
    import sys
    sys.path.insert(0, str(A.REPO / "scripts" / "v10"))
    import t01_validate as T
    dev, dt = torch.device(DEVICE), torch.float32     # the host energies run in float32 (Q.RigidEnergy positions are float32)
    pert = H.perturbations_for(guard, [window], "CAL", "test support")
    inputs = H.EPFLGuardedInputs(guard, {"CAL"}, "test support", pert, dev)
    ref_l = inputs.reference(window)[0]; mov = inputs.moving(window, 2)[0].to(dt)
    eU, eV, cr, head = T.build_models_seeded(T.SEED, dev, dt); A.freeze_for_test(eU, eV, cr, head)
    DU = A.reference_descriptors(eU, inputs, window, dt)
    rng = lambda: np.random.default_rng(A.Q.seed_of(f"R6-REG-P2|pos|{window}|2"))   # noqa: E731
    rg = lambda M: tuple(float(x) for x in np.percentile(M.cpu().numpy(), [0.5, 99.5]))   # noqa: E731
    ranges = (rg(ref_l), rg(mov))
    mk = lambda fn: (lambda m: fn(mov if m is None else m))   # noqa: E731
    return {"MF_moments": (mk(lambda m: A.MFRigidEnergy(DU, m, rng(), eV=eV, head=head, estimator="moments", dtype=dt)), G.R_VP_CRITIC),
            "MLP_critic": (mk(lambda m: A.Q.RigidEnergy(torch, F, "critic", DU, m, rng(), eV=eV, critic=cr)), G.R_VP_CRITIC),
            "MI": (mk(lambda m: A.Q.RigidEnergy(torch, F, "mi", ref_l.to(dt), m, rng(), ranges=ranges)), G.R_VP_POINT)}


def test_effective_support_by_sentinel_poisoning(guard):
    w = guard.included("CAL")[0]
    rows = G.support_rows(guard, w, "CAL", DEVICE, mlp_energy_factory(guard, w))
    assert not failed(rows), [r for r in rows if r["status"] != "PASS"]
    frac = [r["reflected_fraction_of_Vp"] for r in rows if r["name"].startswith("support.V_level")]
    assert all(0 <= f < 0.5 for f in frac)


def test_mf_geometry_gradients_fp64_and_frozen_contract(guard):
    rows = G.gradient_rows(guard, guard.included("FIT")[10], DEVICE, torch.float64, states=("sharpened_x10",))
    assert not failed(rows), [r for r in rows if r["status"] != "PASS"]


def test_checkpoint_round_trip(guard, tmp_path):
    rows = G.checkpoint_rows(guard, guard.included("FIT")[20], DEVICE, str(tmp_path))
    assert not failed(rows), [r for r in rows if r["status"] != "PASS"]


@pytest.fixture(scope="module")
def mlp_trainer(guard):
    return H.new_trainer("MLP_VCS_perm1", H.candidates("MLP_VCS_perm1")[0], 1117, guard, DEVICE)


def test_epfl_trainer_full_fit336_one_step(mlp_trainer):
    t = mlp_trainer
    assert len(t.imgs) == 336 and t.sampler.n_sc == 16 and t.cfg.crop == 160
    u, v = next(iter(t.imgs.values()))
    assert tuple(u.shape) == (3, 512, 512) and tuple(v.shape) == (1, 512, 512)
    rec = t.step()
    assert math.isfinite(rec["loss"]) and math.isfinite(rec["grad_norm_preclip"]) and t.n_steps == 1


def test_epfl_registration_session_energy(guard, mlp_trainer):
    from v10.supplement import checkpoints as CP
    t = mlp_trainer; cal = guard.included("CAL")[1]
    with CP.evaluation_state(t):
        s = H.EPFLRegistrationSession("MLP_VCS_perm1", t, guard, "CAL", [cal], DEVICE, H.candidates("MLP_VCS_perm1")[0])
        e, prep = s.energy(cal, 3)
        p = torch.tensor([0.2, 1.5, -2.5], device=DEVICE, requires_grad=True)
        val = e(p); g = torch.autograd.grad(val, p)[0]
        assert math.isfinite(float(val)) and bool(torch.isfinite(g).all()) and float(g.abs().max()) > 0
        assert prep["preparation_seconds"] >= 0 and e.energy.kind == "critic"
        assert e.energy.x.shape == (A.P.N_POS, 2)
    with pytest.raises(PermissionError):
        H.EPFLRegistrationSession("MLP_VCS_perm1", t, guard, "DEV", [guard.included("DEV")[0]], DEVICE, {})
