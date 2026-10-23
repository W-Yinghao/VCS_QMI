"""R22-A2 CPU checks (synthetic; no data, no GPU): the sharp-critic objectives against brute force, alpha fixed and b alone in its own Adam
group (lr 1e-3, x0.1 with the student's schedule), the per-epoch record, checkpoint/resume, the seed-expansion rule, and the R22-A arms
bitwise unchanged against the train.py of the A launch commit (df8a0707).  Run with test_r22.py in r22_cpu_test.sbatch."""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(HERE))
from losses import OBJECTIVES  # noqa: E402
from losses_sharp import ALPHA, B0, B_LR, HARD, OBJECTIVES_SHARP, BCESharpObjective, VCSSharpObjective, _offdiag  # noqa: E402
from test_r22 import _brute_vcs, _fixture, _sims, _state  # noqa: E402
from v9.n2.train import WeightedStudent  # noqa: E402

A_COMMIT = "df8a0707"          # the R22-A launch commit: its train.py / losses.py ran the A arms


def _git_show(rel):
    return subprocess.run(["git", "-C", str(REPO), "show", f"{A_COMMIT}:{rel}"], capture_output=True, check=True).stdout


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("b", [-10.0, -9.3, -11.0])
def test_sharp_j_matches_bruteforce(weighted, b):
    s = _sims(B=7, seed=int(-b * 10)); obj = VCSSharpObjective(weighted=weighted).double()
    with torch.no_grad():
        obj.critic.b.fill_(b)
    w = torch.tensor(np.random.default_rng(3).uniform(0.1, 1, 7)) if weighted else torch.ones(7, dtype=torch.float64)
    loss, ell = obj(s, w, 0.55 if weighted else 1.0, "mean")
    T = torch.tanh(20.0 * s.detach() + b)
    ref = 1 - _brute_vcs(T, w if weighted else None, 0.55 if weighted else 1.0)
    assert abs(float(loss) - ref) < 1e-12 and ell.shape == (7,)
    if not weighted:                                                   # 1 - J = E_P (1-T)^2 / 2 + E_Q (1+T)^2 / 2
        d, off = T.diagonal(), T[~torch.eye(7, dtype=torch.bool)]
        assert abs(float(loss) - float(0.5 * ((1 - d) ** 2).mean() + 0.5 * ((1 + off) ** 2).mean())) < 1e-12
    ls, _ = obj(s, w, 0.55 if weighted else 1.0, "sum"); assert abs(float(ls) - 7 * float(loss)) < 1e-10


def test_bce_sharp_matches_reference():
    s = _sims(B=8, seed=11); obj = BCESharpObjective().double()
    with torch.no_grad():
        obj.critic.b.fill_(-9.5)
    loss, _ = obj(s, torch.ones(8, dtype=torch.float64), 1.0, "mean")
    f = 20.0 * s.detach() - 9.5; ref = 0.0
    for i in range(8):
        ref += float(F.softplus(-2 * f[i, i])) / 8
        for j in range(8):
            if i != j:
                ref += float(F.softplus(2 * f[i, j])) / 56
    assert abs(float(loss) - ref) < 1e-12


@pytest.mark.parametrize("arm", ["VCS_SHARP", "VCSW_SHARP", "BCE_SHARP"])
def test_alpha_fixed_b_own_group_and_gradients(arm):
    obj = OBJECTIVES_SHARP[arm]()
    assert ALPHA == 20.0 and B0 == -10.0 and B_LR == 1e-3 and obj.critic.alpha == 20.0 and not isinstance(obj.critic.alpha, torch.Tensor)
    params = list(obj.parameters()); assert len(params) == 1 and params[0] is obj.critic.b and float(obj.critic.b) == -10.0
    assert set(obj.state_dict()) == {"critic.b"}
    (g,) = obj.param_groups(); assert g["params"] == [obj.critic.b] and g["lr"] == 1e-3 and g["base_lr"] == 1e-3
    s = torch.rand(9, 9, requires_grad=True); loss, ell = obj(s, torch.rand(9), 0.5, "sum"); loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(s.grad).all() and s.grad.abs().sum() > 0 and float(obj.critic.b.grad.abs()) > 0
    cfg = obj.config(); assert cfg["alpha_learnable"] is False and cfg["b0"] == -10.0 and cfg["b_lr"] == 1e-3


def test_epoch_record_matches_bruteforce_and_resets():
    obj = VCSSharpObjective(); g = torch.Generator().manual_seed(2)
    mats = [torch.rand(6, 6, generator=g) * 0.4 + 0.3 for _ in range(3)]
    for m in mats:
        obj(m, torch.ones(6), 1.0, "sum")
    rec = obj.epoch_stats()
    Ts = [torch.tanh(20.0 * m - 10.0) for m in mats]
    eye = torch.eye(6, dtype=torch.bool)
    for T in Ts:
        assert torch.equal(_offdiag(T).reshape(-1), T[~eye])         # same entries, row-major order
    assert rec["updates"] == 3 and rec["b"] == -10.0 and rec["alpha"] == 20.0
    assert abs(rec["hard_neg_frac_Q"] - np.mean([float((T[~eye] > HARD).float().mean()) for T in Ts])) < 1e-6
    assert abs(rec["EQ_T_mean"] - np.mean([float(T[~eye].mean()) for T in Ts])) < 1e-6
    assert abs(rec["J_hat"] - np.mean([_brute_vcs(T.double()) for T in Ts])) < 1e-5 and abs(rec["J_train"] - rec["J_hat"]) < 1e-7
    assert rec["pool_n_Q"] == 30 and abs(rec["T_Q_quantiles"]["0.5"] - float(torch.quantile(Ts[0][~eye], 0.5))) < 1e-6   # pool = update 0
    assert obj.epoch_stats() == {}                                     # reset after each epoch


@pytest.mark.parametrize("arm", ["VCS_SHARP", "VCSW_SHARP", "BCE_SHARP"])
def test_a2_lr_groups_epoch_log_and_bitwise_resume(tmp_path, arm):
    cfg, rows, w, col, di, dc = _fixture(seed=6)
    cfg = dataclasses.replace(cfg, lr_update=1)                        # the x0.1 drop happens at epoch 1 of 2
    ww = w if arm == "VCSW_SHARP" else {p: 1.0 for p in w}
    mk = lambda out: WeightedStudent(cfg, rows, ww, col, di, dc, tmp_path / out, 4, loss_scale="sum", objective=OBJECTIVES_SHARP[arm]())
    full = mk("full").fit(log_every=1)
    part = mk("part"); part.fit(epochs=1, log_every=1)
    s0 = _state(tmp_path / "part" / "last.pt")
    g0 = s0["opt"]["param_groups"]
    assert len(g0) == 2 and g0[0]["lr"] == cfg.learning_rate and g0[1]["lr"] == 1e-3 and g0[1]["base_lr"] == 1e-3 and len(g0[1]["params"]) == 1
    (tmp_path / "part" / "done.json").unlink()
    res = mk("part").fit(epochs=2, log_every=1)
    sf, sp = _state(tmp_path / "full" / "last.pt"), _state(tmp_path / "part" / "last.pt")
    gf = sf["opt"]["param_groups"]
    assert gf[0]["lr"] == cfg.learning_rate * 0.1 and gf[1]["lr"] == 1e-3 * 0.1 and gf[1]["name"] == "r22a2_critic_b"
    assert set(sf["objective"]) == {"critic.b"} and torch.equal(sf["objective"]["critic.b"], sp["objective"]["critic.b"])
    assert float(sf["objective"]["critic.b"]) != -10.0
    assert all(torch.equal(sf["model"][k], sp["model"][k]) for k in sf["model"]) and res["best"] == full["best"]
    lf, lp = (json.loads((tmp_path / x / "train_log.json").read_text()) for x in ("full", "part"))
    assert lp["resumed_at"] == [1] and [r["loss"] for r in lp["steps"]] == [r["loss"] for r in lf["steps"]]
    for e_f, e_p in zip(lf["epochs"], lp["epochs"]):
        assert e_f["objective_epoch"] == e_p["objective_epoch"] and e_f["objective_epoch"]["updates"] == e_f["steps_in_epoch"]
        assert all(np.isfinite(e_f["objective_epoch"][k]) for k in ("b", "hard_neg_frac_Q", "J_hat", "EP_T_mean", "EQ_T_mean"))
    assert "hard_neg_frac_Q" in lf["steps"][-1] and "b" in lf["steps"][-1]
    man = json.loads((tmp_path / "full" / "run_manifest.json").read_text()); assert man["objective"]["name"] == OBJECTIVES_SHARP[arm]().name


# ---------------------------------------------------------------------------------------------- R22-A arms unchanged
def test_losses_py_unchanged_since_a_launch():
    assert hashlib.sha256((HERE / "losses.py").read_bytes()).hexdigest() == hashlib.sha256(_git_show("v9_r22/vcsloss/losses.py")).hexdigest()


@pytest.mark.parametrize("arm", ["VCS", "VCSW", "BCE", "NCE", "HINGE"])
def test_a_arms_bitwise_against_a_launch_train_py(tmp_path, arm):
    old_py = tmp_path / "train_a_launch.py"; old_py.write_bytes(_git_show("src/v9/n2/train.py"))
    spec = importlib.util.spec_from_file_location("v9.n2.train_a_launch", old_py)
    mod = importlib.util.module_from_spec(spec); mod.__package__ = "v9.n2"; spec.loader.exec_module(mod)
    cfg, rows, w, col, di, dc = _fixture(seed=6)
    cfg = dataclasses.replace(cfg, lr_update=1)                        # exercise the lr loop across a drop
    ww = w if arm == "VCSW" else {p: 1.0 for p in w}
    runs = {}
    for tag, cls in (("new", WeightedStudent), ("old", mod.WeightedStudent)):
        obj = OBJECTIVES[arm]()
        runs[tag] = cls(cfg, rows, ww, col, di, dc, tmp_path / tag, 4, loss_scale="sum", objective=obj).fit(log_every=1)
    sa, sb = _state(tmp_path / "new" / "last.pt"), _state(tmp_path / "old" / "last.pt")
    assert set(sa) == set(sb) and all(torch.equal(sa["model"][k], sb["model"][k]) for k in sa["model"])
    assert sa["opt"]["param_groups"] == sb["opt"]["param_groups"]
    for k, st in sa["opt"]["state"].items():
        assert all(torch.equal(st[x], sb["opt"]["state"][k][x]) for x in st)
    if arm not in ("NCE", "HINGE"):
        assert all(torch.equal(sa["objective"][k], sb["objective"][k]) for k in sa["objective"])
    la, lb = (json.loads((tmp_path / x / "train_log.json").read_text()) for x in ("new", "old"))
    assert json.dumps(la["steps"]) == json.dumps(lb["steps"]) and runs["new"]["best"] == runs["old"]["best"]     # text: NaN-safe
    strip = lambda e: {k: v for k, v in e.items() if k != "seconds"}
    assert json.dumps([strip(e) for e in la["epochs"]]) == json.dumps([strip(e) for e in lb["epochs"]])
    assert all("objective_epoch" not in e for e in la["epochs"]) and len(la["epochs"]) == 2
    assert (tmp_path / "new" / "run_manifest.json").read_text() == (tmp_path / "old" / "run_manifest.json").read_text()


def test_seed_expansion_rule():
    import r22_student as R
    u = lambda x: {"best_rsum": x}
    base = {"s901__HINGE_ConstantU_N2M": u(443.6), "s901__HINGE_VCSW_N2M": u(467.0)}
    assert R.a2_rule({**base, "s901__VCS_SHARP": u(434.0)})["decision"] == "ADD_SEEDS_902_903"          # 9.6 below: within
    assert R.a2_rule({**base, "s901__VCS_SHARP": u(420.0)})["decision"] == "PENDING"                    # VCSW_SHARP not done yet
    assert R.a2_rule({**base, "s901__VCS_SHARP": u(420.0), "s901__VCSW_SHARP": u(457.0)})["decision"] == "STOP"   # exactly 10 below: not within
    assert R.a2_rule({**base, "s901__VCS_SHARP": u(420.0), "s901__VCSW_SHARP": u(480.0)})["decision"] == "ADD_SEEDS_902_903"
    assert set(R.A2_ARMS) == set(OBJECTIVES_SHARP) and all(a in R.WEIGHT_FILE for a in R.A2_ARMS) and R.WEIGHT_FILE["VCSW_SHARP"] == "VCS_Weight"
    assert R.MAIN_ARMS == ("VCS", "VCSW", "BCE", "NCE")
