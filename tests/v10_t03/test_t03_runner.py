"""The unit runner on a miniature unit table (tiny suite, 4 steps): prerequisites -> BLOCKED, smoke -> CAL -> formal resolution of
the CAL placeholders through the tuning-suite selection, COMPLETE kept, FAIL kept and retried into attempts/, the package checker
wrapper writes nothing into the package tree."""
from __future__ import annotations

import json

import pytest

import t03_common as C
import t03_run as R
import t03_train as TR

COND = {"family": "two_block", "m": "8", "rho": "0.25"}


def row(uid, block, method, lr, cap="NA", deps=(), seed="7", algo="1017", K="4", rule="SMOKE_NONE"):
    return {"unit_id": uid, "block": block, **COND, "method": method, "K": K, "lr": lr, "cap": cap, "data_root_seed": seed,
            "algo_seed": algo, "steps": "4", "ckpt_every": "2", "selection_rule": rule, "depends_on": ";".join(deps)}


@pytest.fixture()
def env(out_root, tiny_sizes):
    return {"ident": C.identity(), "env": {"note": "test"}}


def _pass_gates(ident, status="PASS"):
    C.write_json_atomic(R.gates_path(), {"status": status, "identity": ident})


def test_prerequisites_block_then_complete_and_keep(env, out_root):
    ident = env["ident"]
    s = row("SMK_A", "smoke", "MF_VCS_M0", "3e-4")
    assert R.run_unit(s, ident=ident, env=env["env"]) == "BLOCKED"                 # no gates yet
    assert json.loads((R.unit_dir("SMK_A") / "result.json").read_text())["reason"].startswith("gates not run")
    _pass_gates(ident, "FAIL")
    assert R.run_unit(s, ident=ident, env=env["env"]) == "BLOCKED"
    _pass_gates(ident)
    assert R.run_unit(s, ident=ident, env=env["env"]) == "COMPLETE"
    res = R.read_result("SMK_A")
    assert res["prerequisites"]["suite_check"]["status"] == "PASS" and res["prerequisites"]["gates"]["status"] == "PASS"
    mtime = (R.unit_dir("SMK_A") / "result.json").stat().st_mtime_ns
    assert R.run_unit(s, ident=ident, env=env["env"]) == "COMPLETE"
    assert (R.unit_dir("SMK_A") / "result.json").stat().st_mtime_ns == mtime     # kept, not rerun
    chk = json.loads(R.check_path(7).read_text())
    assert chk["status"] == "PASS" and chk["n_checks"] > 100 and chk["n_checks_passed"] == chk["n_checks"]
    assert "result" not in chk and "checks" not in chk                                    # truth-free summary: no checker rows
    full = R.full_check_path(R.check_path(7))
    assert C.TRUTH_DIR_NAME in full.parts and json.loads(full.read_text())["result"]["n_checks"] == chk["n_checks"]
    assert chk["full_record"]["sha256"] == C.sha256_file(full)


def test_cal_and_formal_resolution(env, out_root):
    ident = env["ident"]
    _pass_gates(ident)
    smk = row("SMK_B", "smoke", "MF_VCS_M1", "3e-4", cap="10")
    cal = [row(f"CAL_{i}", "CAL", "MF_VCS_M1", lr, cap=str(cap), deps=["SMK_B"], rule="CAL_UJ_SELECT")
           for i, (lr, cap) in enumerate((lr, cap) for lr in ("1e-4", "3e-4", "1e-3") for cap in (10, 30, 100))]
    main = row("MAIN_B", "main", "MF_VCS_M1", "CAL", cap="CAL", deps=[r["unit_id"] for r in cal], seed="8", algo="1001", rule="OWN_CAL_UJ_CKPT")
    assert R.run_unit(cal[0], ident=ident, env=env["env"]) == "BLOCKED"           # smoke not complete
    assert R.run_unit(smk, ident=ident, env=env["env"]) == "COMPLETE"
    assert R.run_unit(main, ident=ident, env=env["env"]) == "BLOCKED"             # CAL candidates missing
    for r in cal:
        assert R.run_unit(r, ident=ident, env=env["env"]) == "COMPLETE"
    assert R.run_unit(main, ident=ident, env=env["env"]) == "COMPLETE"
    res = R.read_result("MAIN_B")
    sel = res["hyperparameter_resolution"]["group_selection"]
    traces = {r["unit_id"]: R.read_result(r["unit_id"])["checkpoints"] for r in cal}
    best = max(c["CAL_UJ"] for t in traces.values() for c in t)
    cands = sorted((c["step"], i) for i, r in enumerate(cal) for c in traces[r["unit_id"]] if c["CAL_UJ"] == best)
    exp_step, exp_ord = cands[0]
    assert (sel["selected"]["ordinal"], sel["selected"]["step"]) == (exp_ord, exp_step)
    assert res["config"]["lr"] == sel["selected"]["lr"] and res["config"]["cap"] == sel["selected"]["cap"]
    assert res["config"]["data_root_seed"] == 8 and res["selection"]["own_cal"]["rule"].startswith("max CAL U-J")
    assert set(res["hyperparameter_resolution"]["dependency_result_sha256"]) == {r["unit_id"] for r in cal}


def test_dependency_identity_mismatch_blocks(env, out_root):
    ident = env["ident"]
    _pass_gates(ident)
    s = row("SMK_C", "smoke", "MF_VCS_M0", "3e-4")
    assert R.run_unit(s, ident=ident, env=env["env"]) == "COMPLETE"
    other = dict(ident, t03_code_hash="0000000000000000")
    _pass_gates(other)
    c = row("CAL_C", "CAL", "MF_VCS_M0", "1e-4", deps=["SMK_C"], rule="CAL_UJ_SELECT")
    assert R.run_unit(c, ident=other, env=env["env"]) == "BLOCKED"
    assert "code hash" in R.read_result("CAL_C")["reason"]


def test_failure_kept_and_retry_archives(env, out_root, monkeypatch):
    ident = env["ident"]
    _pass_gates(ident)
    s = row("SMK_D", "smoke", "MLP_VCS_perm1", "3e-4", K="NA")
    orig = TR.train_unit

    def boom(*a, **k):
        raise FloatingPointError("injected failure")
    monkeypatch.setattr(TR, "train_unit", boom)
    assert R.run_unit(s, ident=ident, env=env["env"]) == "FAIL"
    f = R.read_result("SMK_D")
    assert f["failure"]["stage"] == "train" and "injected" in f["failure"]["message"]
    monkeypatch.setattr(TR, "train_unit", orig)
    assert R.run_unit(s, ident=ident, env=env["env"]) == "FAIL"                    # kept without --retry-failed
    assert R.run_unit(s, ident=ident, env=env["env"], retry_failed=True) == "COMPLETE"
    att = R.unit_dir("SMK_D") / "attempts" / "1" / "result.json"
    assert json.loads(att.read_text())["status"] == "FAIL"


def test_package_checker_wrapper_leaves_the_package_untouched(tiny_suite, out_root):
    suite, _ = tiny_suite
    before = sorted(p.name for p in (C.PACKAGE / "synthetic" / "validation").iterdir())
    rec = R.run_package_checker(suite, out_root / "checks" / "x.json")
    assert rec["status"] == "PASS" and rec["checker_sha256"] == C.CHECKER_SHA256
    assert sorted(p.name for p in (C.PACKAGE / "synthetic" / "validation").iterdir()) == before
    assert sorted(p.name for p in (out_root / "checks").iterdir()) == ["evaluator_only", "x.json"]
    assert [p.name for p in (out_root / "checks" / "evaluator_only").iterdir()] == ["x.json"]         # scratch directory removed


def test_run_threads():
    assert R.run_threads(3) == 3
    for bad in (0, -1, True):
        with pytest.raises(SystemExit):
            R.run_threads(bad)


def test_resolution_denial_fails_the_unit(env, out_root, monkeypatch):
    """A refused access during the hyperparameter resolution (even when swallowed) makes the unit FAIL before any training."""
    ident = env["ident"]
    _pass_gates(ident)
    s = row("SMK_E", "smoke", "MF_VCS_M0", "3e-4")
    orig = R.resolve

    def sneaky(*a, **k):
        try:
            open(out_root / "runs" / "OTHER_UNIT" / "evaluator_only" / "evaluation.json")
        except PermissionError:
            pass
        return orig(*a, **k)
    monkeypatch.setattr(R, "resolve", sneaky)
    assert R.run_unit(s, ident=ident, env=env["env"]) == "FAIL"
    f = R.read_result("SMK_E")
    assert f["failure"]["stage"] == "prerequisites" and "hyperparameter resolution" in f["failure"]["message"]
    assert not (R.unit_dir("SMK_E") / "checkpoints.pt").exists()


def test_select_command_reports_every_group(env, out_root):
    """t03_run.py select over the real unit table: 60 (condition, method) groups, every one BLOCKED while no CAL unit has run; with
    the 3 candidates of one group COMPLETE under the current T03 code hash that group is SELECTED (max CAL U-J over candidates x
    checkpoints) and the other 59 stay BLOCKED."""
    import t03_units as U
    ident = env["ident"]
    path = out_root / "selection" / "T03_CAL_SELECTION.json"
    assert R.main(["select"]) == 0
    sel = json.loads(path.read_text())
    assert sel["n_groups"] == 60 and sel["n_selected"] == 0
    assert all(g["status"] == "BLOCKED" and "NOT_RUN" in g["reason"] for g in sel["groups"])
    rows = [U.parse(r) for r in U.load_rows() if r["block"] == "CAL"][:3]           # condition 1, MF_VCS_perm1, lr 1e-4 / 3e-4 / 1e-3
    assert [r["method"] for r in rows] == ["MF_VCS_perm1"] * 3
    for i, r in enumerate(rows):
        C.write_json_atomic(R.unit_dir(r["unit_id"]) / "result.json", {
            "status": "COMPLETE", "provenance": {"identity": ident}, "config": {"method": r["method"], "block": "CAL", "lr": r["lr"], "cap": r["cap"]},
            "checkpoints": [{"step": 250 * (k + 1), "CAL_UJ": (0.1 if k == 3 else 0.0) + 0.01 * i} for k in range(8)]})
    assert R.main(["select"]) == 0
    sel = json.loads(path.read_text())
    g = sel["groups"][0]
    assert sel["n_selected"] == 1 and g["status"] == "SELECTED" and g["candidates"] == [r["unit_id"] for r in rows]
    assert (g["selected"]["unit_id"], g["selected"]["step"], g["selected"]["lr"]) == (rows[2]["unit_id"], 1000, 1e-3)
    assert all(x["status"] == "BLOCKED" for x in sel["groups"][1:])


def test_units_check_command(out_root, tmp_path):
    assert R.main(["units", "--check", "--output", str(tmp_path / "u.json")]) == 0
    assert json.loads((tmp_path / "u.json").read_text())["status"] == "PASS"
