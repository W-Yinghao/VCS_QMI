"""R22-A CPU checks (synthetic; no data, no GPU): the objectives against brute-force loops, gradients, weighted-P normalisation,
loss scale, the unchanged hinge path, and objective checkpoint/resume through the host trainer.  Run as a Slurm cpu-high job."""
from __future__ import annotations

import importlib.util
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(HERE))
from losses import OBJECTIVES, BCEObjective, InfoNCEObjective, VCSObjective, vcs_terms  # noqa: E402
from v9.n2 import noise as NZ, roles as RL  # noqa: E402
from v9.n2.fixtures import collate_pairs, make_fixture  # noqa: E402
from v9.n2.teacher import HostConfig  # noqa: E402
from v9.n2.train import WeightedStudent  # noqa: E402

FROZEN_TRAIN = Path("/home/infres/yinwang/CS_QMI/handoff/v9_full_20261017/source/src/v9/n2/train.py")   # sha256 cafac01f... (N2F/N2M base)


def _sims(B=7, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(B, B, generator=g, dtype=dtype).requires_grad_(True)


def _brute_vcs(T, w=None, mean_v=1.0):
    B = T.shape[0]; ep = ep2 = eq = eq2 = 0.0
    for i in range(B):
        wi = 1.0 if w is None else float(w[i]) / mean_v
        ep += wi * float(T[i, i]) / B; ep2 += wi * float(T[i, i]) ** 2 / B
        for j in range(B):
            if i != j:
                eq += float(T[i, j]) / (B * (B - 1)); eq2 += float(T[i, j]) ** 2 / (B * (B - 1))
    return ep - eq - 0.5 * ep2 - 0.5 * eq2


def _critic_T(obj, sims):
    return torch.tanh(obj.critic.f(sims.to(torch.float32)).to(torch.float64))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_j_matches_bruteforce_and_squared_form(seed):
    s = _sims(seed=seed); obj = VCSObjective().double()
    with torch.no_grad():
        obj.critic.a.fill_(0.3 * seed - 0.2); obj.critic.b.fill_(0.1 * seed - 0.4)
    T = torch.tanh(obj.critic.f(s))
    J = float(vcs_terms(T)["J"]); assert abs(J - _brute_vcs(T.detach())) < 1e-12
    loss, ell = obj(s, torch.ones(7, dtype=torch.float64), 1.0, "mean")
    assert abs(float(loss) - (1 - J)) < 1e-12 and ell.shape == (7,)
    d, off = T.diagonal(), T[~torch.eye(7, dtype=torch.bool)]                 # 1 - J = E_P (1-T)^2 / 2 + E_Q (1+T)^2 / 2
    assert abs(float(loss) - float(0.5 * ((1 - d) ** 2).mean() + 0.5 * ((1 + off) ** 2).mean())) < 1e-12


def test_weighted_p_bruteforce_normalisation_and_invariance():
    s = _sims(B=6, seed=4); g = np.random.default_rng(0); w = torch.tensor(g.uniform(0, 1, 6)); obj = VCSObjective(weighted=True).double()
    loss, _ = obj(s, w, 0.6, "mean")
    T = torch.tanh(obj.critic.f(s)).detach()
    assert abs(float(loss) - (1 - _brute_vcs(T, w, 0.6))) < 1e-12
    l2, _ = obj(s, 3.0 * w, 1.8, "mean"); assert abs(float(l2) - float(loss)) < 1e-12        # v and the normaliser scale together
    l1, _ = obj(s, torch.ones(6, dtype=torch.float64), 1.0, "mean"); l0, _ = VCSObjective().double()(s, w, 0.6, "mean")
    assert abs(float(l1) - float(l0)) < 1e-12                                                  # v = 1, mean 1: weighted == unweighted
    # over a pool covered by whole batches, the weighted P masses mean_i(v_i / mean_v_full) average to exactly 1 (P_w is a probability)
    v_pool = g.uniform(0, 1, 6 * 8); mean_v = v_pool.mean()
    masses = [(v_pool[k * 6:(k + 1) * 6] / mean_v).mean() for k in range(8)]
    assert abs(np.mean(masses) - 1.0) < 1e-12


@pytest.mark.parametrize("arm", ["VCS", "VCSW", "BCE", "NCE"])
def test_gradients_finite_and_critic_trained(arm):
    obj = OBJECTIVES[arm](); s = torch.rand(9, 9, requires_grad=True); v = torch.rand(9)
    loss, ell = obj(s, v, 0.5, "sum"); loss.backward()
    assert loss.dim() == 0 and ell.shape == (9,) and torch.isfinite(loss) and torch.isfinite(s.grad).all() and s.grad.abs().sum() > 0
    for p in obj.parameters():
        assert p.grad is not None and torch.isfinite(p.grad) and float(p.grad.abs()) > 0
    assert (len(list(obj.parameters())) == 0) == (arm == "NCE")
    st = obj.stats(); assert all(np.isfinite(x) for x in st.values())


def test_bce_and_infonce_match_reference_and_scale():
    s = _sims(B=8, seed=7)
    bce = BCEObjective().double(); lb, _ = bce(s, torch.ones(8, dtype=torch.float64), 1.0, "mean")
    f = bce.critic.f(s).detach(); B = 8; ref = 0.0
    for i in range(B):
        ref += float(F.softplus(-2 * f[i, i])) / B
        for j in range(B):
            if i != j:
                ref += float(F.softplus(2 * f[i, j])) / (B * (B - 1))
    assert abs(float(lb) - ref) < 1e-12
    nce = InfoNCEObjective(0.05); ln, _ = nce(s, None, 1.0, "mean")
    z = s.detach() / 0.05; t = torch.arange(B)
    assert abs(float(ln) - 0.5 * float(F.cross_entropy(z, t) + F.cross_entropy(z.t(), t))) < 1e-10
    for obj in (VCSObjective().double(), bce, nce):
        a, _ = obj(s, torch.ones(B, dtype=torch.float64), 1.0, "mean"); b, _ = obj(s, torch.ones(B, dtype=torch.float64), 1.0, "sum")
        assert abs(float(b) - B * float(a)) < 1e-9


# ---------------------------------------------------------------------------------------------- host trainer
def _fixture(seed=5, rho=0.3):
    fx = make_fixture(n_images=24, seed=seed)
    roles = RL.assign_roles(sorted(set(str(i) for i in fx["caption_image"].values())), seed, 0.25, min_sizes={k: 1 for k in RL.ANCHOR_ROLES})
    pairs = NZ.make_pair_table(fx["caption_image"], RL.role_of_image(roles)); observed, _, _ = NZ.inject_mismatch(pairs, rho, seed)
    cfg = HostConfig(vocab_size=fx["vocab"], img_dim=fx["img_dim"], word_dim=8, embed_size=16, sim_dim=8, num_region=fx["n_region"],
                     batch_size=8, num_epochs=2, teacher_epochs=1, warmup_epochs=1)
    dev = make_fixture(n_images=6, seed=9)
    u = [r["pair_id"] for r in observed if r["role"] == "U"]; rng = np.random.default_rng(1)
    return cfg, observed, {p: float(rng.uniform(0.2, 1)) for p in u}, partial(collate_pairs, fx["images"], fx["captions"]), dev["images"], \
        [dev["captions"][i] for i in range(30)]


def _state(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def test_hinge_path_unchanged_against_frozen_train_py(tmp_path):
    spec = importlib.util.spec_from_file_location("v9.n2.train_frozen", FROZEN_TRAIN)
    mod = importlib.util.module_from_spec(spec); mod.__package__ = "v9.n2"; spec.loader.exec_module(mod)
    cfg, rows, w, col, di, dc = _fixture()
    for scale in ("sum", "mean"):
        a = WeightedStudent(cfg, rows, w, col, di, dc, tmp_path / f"new_{scale}", 3, loss_scale=scale).fit(log_every=1)
        b = mod.WeightedStudent(cfg, rows, w, col, di, dc, tmp_path / f"old_{scale}", 3, loss_scale=scale).fit(log_every=1)
        sa, sb = _state(tmp_path / f"new_{scale}" / "last.pt"), _state(tmp_path / f"old_{scale}" / "last.pt")
        assert "objective" not in sa and set(sa["model"]) == set(sb["model"])
        assert all(torch.equal(sa["model"][k], sb["model"][k]) for k in sa["model"])
        la, lb = (json.loads((tmp_path / f"{x}_{scale}" / "train_log.json").read_text()) for x in ("new", "old"))
        assert [r["loss"] for r in la["steps"]] == [r["loss"] for r in lb["steps"]] and a["best"] == b["best"]
        assert (tmp_path / f"new_{scale}" / "run_manifest.json").read_text() == (tmp_path / f"old_{scale}" / "run_manifest.json").read_text()


@pytest.mark.parametrize("arm", ["VCS", "VCSW", "BCE", "NCE"])
def test_objective_trains_checkpoints_and_resumes_bitwise(tmp_path, arm):
    cfg, rows, w, col, di, dc = _fixture(seed=6)
    ww = w if arm == "VCSW" else {p: 1.0 for p in w}
    full = WeightedStudent(cfg, rows, ww, col, di, dc, tmp_path / "full", 4, loss_scale="sum", objective=OBJECTIVES[arm]()).fit(log_every=1)
    part = WeightedStudent(cfg, rows, ww, col, di, dc, tmp_path / "part", 4, loss_scale="sum", objective=OBJECTIVES[arm]()); part.fit(epochs=1, log_every=1)
    (tmp_path / "part" / "done.json").unlink()
    res = WeightedStudent(cfg, rows, ww, col, di, dc, tmp_path / "part", 4, loss_scale="sum", objective=OBJECTIVES[arm]()).fit(epochs=2, log_every=1)
    sf, sp = _state(tmp_path / "full" / "last.pt"), _state(tmp_path / "part" / "last.pt")
    assert all(torch.equal(sf["model"][k], sp["model"][k]) for k in sf["model"]) and res["best"] == full["best"]
    lf, lp = (json.loads((tmp_path / x / "train_log.json").read_text()) for x in ("full", "part"))
    assert lp["resumed_at"] == [1] and [r["step"] for r in lp["steps"]] == [r["step"] for r in lf["steps"]]
    assert [r["loss"] for r in lp["steps"]] == [r["loss"] for r in lf["steps"]]
    assert all(np.isfinite(r["loss"]) for r in lf["steps"]) and lf["epochs"][0]["phase"] == f"objective:{OBJECTIVES[arm]().name}"
    man = json.loads((tmp_path / "full" / "run_manifest.json").read_text()); assert man["objective"]["name"] == OBJECTIVES[arm]().name
    if arm != "NCE":
        assert set(sf["objective"]) == {"critic.a", "critic.b"} and torch.equal(sf["objective"]["critic.a"], sp["objective"]["critic.a"])
        init = OBJECTIVES[arm]().state_dict()
        assert not torch.equal(sf["objective"]["critic.b"], init["critic.b"])               # the critic scalars were trained
        assert len(sf["opt"]["param_groups"]) == 2 and "alpha" in lf["steps"][-1] and "grad_norm_pre_clip" in lf["steps"][-1]
    else:
        assert sf["objective"] == {} and len(sf["opt"]["param_groups"]) == 1
