"""R22-A3 CPU checks (synthetic batches and the CPU host fixture; the only real files read are the locked N2F weight files and their N2F
run manifests): weighted InfoNCE against a brute-force loop (loss and gradient), v = 1 recovering A's NCE bitwise (objective and trainer),
the full-pool normaliser of the weighted hinge, low-w pairs kept as negatives, checkpoint/resume, the weight files through the original
reader, the A3 rule, and the A / A2 sources and constants unchanged since the A2 launch (ccb5f331).  Run with test_r22.py, test_r22_a2.py."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(HERE))
from losses import OBJECTIVES, InfoNCEObjective  # noqa: E402
from losses_nce_w import OBJECTIVES_A3, WeightedInfoNCEObjective  # noqa: E402
from test_r22 import _fixture, _sims, _state  # noqa: E402
from v9.n2.train import WeightedStudent, hinge_per_anchor  # noqa: E402

A2_COMMIT = "ccb5f331"


def _git_show(commit, rel):
    return subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{rel}"], capture_output=True, check=True).stdout


def _brute_nce_w(s, v, mean_v, tau=0.05):
    """sum_i (v_i / mean_v) * (row_i + col_i) / 2 / B with explicit exp/log loops (torch scalars, so autograd gives the reference gradient)."""
    B = s.shape[0]; z = s / tau; total = 0.0
    for i in range(B):
        row = -(z[i, i] - torch.log(sum(torch.exp(z[i, j]) for j in range(B))))
        col = -(z[i, i] - torch.log(sum(torch.exp(z[j, i]) for j in range(B))))
        total = total + (float(v[i]) / mean_v) * 0.5 * (row + col) / B
    return total


def _unit(best):
    """Synthetic unit record with every metric _pairs() differences (best rSum, the six Recalls, last and last-5 rSum)."""
    import r22_student as R
    return {"best_rsum": best, **{m: best / 6 for m in R.RECALLS}, "last_rsum": best - 1.0, "mean_last5_rsum": best - 2.0}

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_weighted_nce_matches_bruteforce_loss_and_gradient(seed):
    s = _sims(B=7, seed=seed); v = torch.tensor(np.random.default_rng(seed).uniform(0, 1, 7)); mean_v = 0.37 + 0.1 * seed
    obj = WeightedInfoNCEObjective("VCS_Weight")
    loss, ell = obj(s, v, mean_v, "mean"); (g,) = torch.autograd.grad(loss, s)
    s2 = s.detach().clone().requires_grad_(True); ref = _brute_nce_w(s2, v, mean_v); (g2,) = torch.autograd.grad(ref, s2)
    assert abs(float(loss) - float(ref)) < 1e-10 and torch.allclose(g, g2, rtol=0, atol=1e-10)
    ls, _ = obj(s, v, mean_v, "sum"); assert abs(float(ls) - 7 * float(loss)) < 1e-9
    _, ell_a = InfoNCEObjective(0.05)(s, None, 1.0, "mean"); assert torch.equal(ell, ell_a)       # per-anchor terms are A's
    st = obj.stats(); assert set(st) == {"nce_w", "nce_unweighted", "batch_mass_P"} and all(math.isfinite(x) for x in st.values())


def test_v1_recovers_a_nce_bitwise():
    for arm in ("NCE_VCSW", "NCE_BCEW"):
        for B in (5, 128):
            s = torch.rand(B, B, generator=torch.Generator().manual_seed(B)).requires_grad_(True)
            la, ea = InfoNCEObjective(0.05)(s, torch.ones(B), 1.0, "sum"); (ga,) = torch.autograd.grad(la, s)
            lw, ew = OBJECTIVES_A3[arm]()(s, torch.ones(B), 1.0, "sum"); (gw,) = torch.autograd.grad(lw, s)
            assert torch.equal(la, lw) and torch.equal(ea, ew) and torch.equal(ga, gw)


def test_normaliser_is_the_weighted_hinge_one_and_negatives_unweighted(tmp_path):
    cfg, rows, w, col, di, dc = _fixture(seed=6)
    st = WeightedStudent(cfg, rows, w, col, di, dc, tmp_path / "st", 4, loss_scale="sum", objective=OBJECTIVES_A3["NCE_VCSW"]())
    u = [r["pair_id"] for r in rows if r["role"] == "U"]; n_a = len(rows) - len(u)
    assert math.isclose(st.mean_v_full, (n_a + math.fsum(w[p] for p in u)) / len(rows), rel_tol=1e-15)
    s = _sims(B=8, seed=3); v = torch.tensor(np.random.default_rng(5).uniform(0, 1, 8)); v[2] = 0.0
    loss, ell = WeightedInfoNCEObjective("BCE_Weight")(s, v, st.mean_v_full, "sum")
    assert abs(float(loss) - float((v * ell).sum() / st.mean_v_full)) < 1e-10           # the hinge path's sum(v ell) / mean_v_full form
    er, ec = hinge_per_anchor(s.detach(), cfg.margin, hard_negative=True)                  # same normaliser expression with the hinge terms
    assert abs(float(((v * (er + ec)).sum()) / st.mean_v_full) - float((v / st.mean_v_full * (er + ec)).mean() * 8)) < 1e-10
    (g,) = torch.autograd.grad(loss, s)
    assert float(g[0, 2].abs()) > 0 and float(g[2, 0].abs()) > 0          # pair 2 (w = 0) is still a negative in row 0 / column 0
    assert float(g[2, 2]) == 0.0           # ... but its own positive terms (row 2, column 2, the only terms holding z_22) carry w = 0
    loss0, _ = WeightedInfoNCEObjective("BCE_Weight")(s, torch.zeros(8, dtype=torch.float64), 1.0, "sum"); assert float(loss0) == 0.0


@pytest.mark.parametrize("arm", ["NCE_VCSW", "NCE_BCEW"])
def test_a3_trains_checkpoints_and_resumes_bitwise(tmp_path, arm):
    cfg, rows, w, col, di, dc = _fixture(seed=6)
    mk = lambda out: WeightedStudent(cfg, rows, w, col, di, dc, tmp_path / out, 4, loss_scale="sum", objective=OBJECTIVES_A3[arm]())
    full = mk("full").fit(log_every=1)
    mk("part").fit(epochs=1, log_every=1); (tmp_path / "part" / "done.json").unlink()
    res = mk("part").fit(epochs=2, log_every=1)
    sf, sp = _state(tmp_path / "full" / "last.pt"), _state(tmp_path / "part" / "last.pt")
    assert sf["objective"] == {} and len(sf["opt"]["param_groups"]) == 1
    assert all(torch.equal(sf["model"][k], sp["model"][k]) for k in sf["model"]) and res["best"] == full["best"]
    lf, lp = (json.loads((tmp_path / x / "train_log.json").read_text()) for x in ("full", "part"))
    assert lp["resumed_at"] == [1] and [r["loss"] for r in lp["steps"]] == [r["loss"] for r in lf["steps"]]
    assert all(np.isfinite(r["loss"]) and np.isfinite(r["nce_w"]) for r in lf["steps"]) and "objective_epoch" not in lf["epochs"][0]
    man = json.loads((tmp_path / "full" / "run_manifest.json").read_text())
    assert man["objective"]["name"] == OBJECTIVES_A3[arm]().name and man["objective"]["tau"] == 0.05 and man["mean_v_full_pool"] < 1.0


def test_a3_with_unit_weights_is_a_nce_bitwise_through_the_trainer(tmp_path):
    cfg, rows, w, col, di, dc = _fixture(seed=6)
    ones = {p: 1.0 for p in w}
    a = WeightedStudent(cfg, rows, ones, col, di, dc, tmp_path / "nce", 4, loss_scale="sum", objective=OBJECTIVES["NCE"]()).fit(log_every=1)
    b = WeightedStudent(cfg, rows, ones, col, di, dc, tmp_path / "w1", 4, loss_scale="sum", objective=OBJECTIVES_A3["NCE_VCSW"]()).fit(log_every=1)
    sa, sb = _state(tmp_path / "nce" / "last.pt"), _state(tmp_path / "w1" / "last.pt")
    assert all(torch.equal(sa["model"][k], sb["model"][k]) for k in sa["model"]) and a["best"] == b["best"]
    assert all(torch.equal(sa["opt"]["state"][k][x], sb["opt"]["state"][k][x]) for k in sa["opt"]["state"] for x in sa["opt"]["state"][k])
    la, lb = (json.loads((tmp_path / x / "train_log.json").read_text()) for x in ("nce", "w1"))
    assert [r["loss"] for r in la["steps"]] == [r["loss"] for r in lb["steps"]] and [r["nce"] for r in la["steps"]] == [r["nce_w"] for r in lb["steps"]]
    assert [e["dev"] for e in la["epochs"]] == [e["dev"] for e in lb["epochs"]]


# ---------------------------------------------------------------------------------------------- real weight files, rule, unchanged A / A2
@pytest.mark.parametrize("seed", [901, 902, 903])
def test_weight_files_through_the_original_reader_and_n2f_hinge_identity(seed):
    import r22_student as R
    for arm, method in (("NCE_VCSW", "VCS_Weight"), ("NCE_BCEW", "BCE_Weight")):
        assert R.WEIGHT_FILE[arm] == method and R.A3_HINGE[arm] == method
        inp = R.weight_inputs(seed, method)                         # N2F frozen sha256 + v9.n2.weights.read_weight_lock (sha and row count)
        w = np.array(list(inp["weights"].values()))
        assert len(w) == 130500 and w.min() >= 0.0 and w.max() <= 1.0 and inp["lock"]["record"]["method"] == method
        man = json.loads((R._n2f_unit(seed, method) / "run_manifest.json").read_text())
        assert man["weight_lock_sha256"] == inp["sha256"]["weights"] == hashlib.sha256(Path(inp["paths"]["weights"]).read_bytes()).hexdigest()
        assert math.isclose(man["sum_w_U"], math.fsum(w.tolist()), rel_tol=1e-12)
    assert R.WEIGHT_FILE["NCE_VCSW"] == R.WEIGHT_FILE["VCSW"] == R.WEIGHT_FILE["VCSW_SHARP"]


def test_a3_rule():
    import r22_student as R
    u = _unit

    def rule(units):
        return R.a3_rule(units, R._pairs(units, R.PAIRS_A3))

    assert rule({})["decision"] == "PENDING"
    assert rule({"s901__NCE_VCSW": u(464.0)})["decision"] == "ADD_SEEDS_902_903"          # >= 464.0
    assert rule({"s901__NCE_VCSW": u(463.9)})["decision"] == "STOP"
    base = {f"s{s}__HINGE_VCSW_N2M": u(467.0) for s in R.SEEDS} | {f"s{s}__HINGE_VCSW_N2F": u(468.0) for s in R.SEEDS}
    met = rule({**base, "s901__NCE_VCSW": u(475.0), "s902__NCE_VCSW": u(466.0), "s903__NCE_VCSW": u(471.0),
                **{f"s{s}__NCE_BCEW": u(466.0) for s in R.SEEDS}})
    v = {(f["a"], f["b"]): f["verdict"] for f in met["final"]}
    assert v[("NCE_VCSW", "HINGE_VCSW_N2M")] == "MET"                    # mean +3.67, 2/3 positive
    assert v[("NCE_VCSW", "NCE_BCEW")] == "MET"                          # mean +4.0, 2/3 positive (902: 0.0 is not positive)
    assert v[("NCE_VCSW", "HINGE_VCSW_N2F")] == "NOT_MET"                # mean +2.67
    two = rule({**base, "s901__NCE_VCSW": u(480.0)})
    assert all(f["verdict"] == "PENDING_3_SEEDS" for f in two["final"]) and [f["pre_registered"] for f in two["final"]] == [True, True, False]
    one_pos = rule({**base, **{f"s{s}__NCE_BCEW": u(400.0) for s in R.SEEDS}, "s901__NCE_VCSW": u(500.0), "s902__NCE_VCSW": u(460.0),
                    "s903__NCE_VCSW": u(460.0)})
    assert {(f["a"], f["b"]): f["verdict"] for f in one_pos["final"]}[("NCE_VCSW", "HINGE_VCSW_N2M")] == "NOT_MET"   # mean +4.0 but 1/3


def test_a_and_a2_unchanged_since_a2_launch(tmp_path):
    for rel in ("v9_r22/vcsloss/losses.py", "v9_r22/vcsloss/losses_sharp.py", "src/v9/n2/train.py"):
        assert hashlib.sha256((REPO / rel).read_bytes()).hexdigest() == hashlib.sha256(_git_show(A2_COMMIT, rel)).hexdigest(), rel
    import r22_student as R                                            # first: c1_student / trace_open come from sys.modules for the old copy
    old = tmp_path / "r22_student_a2_launch.py"; old.write_bytes(_git_show(A2_COMMIT, "v9_r22/vcsloss/r22_student.py"))
    spec = importlib.util.spec_from_file_location("r22_student_a2_launch", old)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    assert (R.SEEDS, R.MAIN_ARMS, R.A2_ARMS, R.PAIRS, R.PAIRS_A2, R.A2_RULE, R.N2F, R.N2M) == \
        (mod.SEEDS, mod.MAIN_ARMS, mod.A2_ARMS, mod.PAIRS, mod.PAIRS_A2, mod.A2_RULE, mod.N2F, mod.N2M)
    assert {k: R.WEIGHT_FILE[k] for k in mod.WEIGHT_FILE} == mod.WEIGHT_FILE and set(R.WEIGHT_FILE) - set(mod.WEIGHT_FILE) == set(R.A3_ARMS)
    assert not set(R.A3_ARMS) & (set(OBJECTIVES) | set(mod.A2_ARMS)) and set(R.A3_ARMS) == set(OBJECTIVES_A3)
    for f in ("weight_inputs", "make_student", "smoke", "run", "endpoints", "baseline_dirs", "identity", "_pairs", "a2_rule", "_critic_track"):
        assert callable(getattr(R, f))
    units = {"s901__HINGE_ConstantU_N2M": _unit(443.6), "s901__HINGE_VCSW_N2M": _unit(467.0),
             "s901__VCS_SHARP": _unit(434.0), "s901__VCSW_SHARP": _unit(457.0)}
    assert R.a2_rule(units) == mod.a2_rule(units) and R._pairs(units, R.PAIRS_A2) == mod._pairs(units, mod.PAIRS_A2)
