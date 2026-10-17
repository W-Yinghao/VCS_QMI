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
    assert chk["status"] == "PASS" and chk["result"]["n_checks"] > 100


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
    assert [p.name for p in (out_root / "checks").iterdir()] == ["x.json"]


def test_units_check_command(out_root, tmp_path):
    assert R.main(["units", "--check", "--output", str(tmp_path / "u.json")]) == 0
    assert json.loads((tmp_path / "u.json").read_text())["status"] == "PASS"
