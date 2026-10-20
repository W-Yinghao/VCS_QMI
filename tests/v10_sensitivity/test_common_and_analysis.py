"""Provenance guards, pre-registration constants and the descriptive read-outs (no Torch, no data)."""
import json
import math
import subprocess

import pytest

from v10_sensitivity import analysis as S
from v10_sensitivity import common as C


def test_preregistration_constants_are_the_written_design():
    pre = C.prereg()
    assert pre["status"] == "PREREGISTERED_BEFORE_ANY_D1_D2_D3_JOB"
    assert pre["D1"]["epochs"] == [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 23]
    assert pre["D1"]["steps"] == [32 * e for e in pre["D1"]["epochs"]] and pre["D1"]["steps"][-1] == 736
    assert pre["common"]["dev24"]["excluded"] == list(C.DEV24_EXCLUDED)
    assert pre["common"]["rotation_strata_deg"] == [list(s) for s in S.STRATA]
    assert pre["common"]["selection_rule"] == ["success42 descending", "success9 descending",
                                               "median_err ascending", "checkpoint step ascending (earlier)"]
    assert pre["D2"]["variant"] == "MIND_2D_warp_then_describe"
    assert {(j["family"], j["partition"]) for j in pre["D3"]["jobs"]} == {
        (f, p) for f in ("MLP_VCS_perm8", "InfoNCE_host[J_perm1]") for p in ("RTX6000PRO", "H100")}
    assert pre["hard_rules"]["held"]["TEST134"].startswith("HOLD")


def test_write_once_refuses_a_different_record(tmp_path):
    path = tmp_path / "x.json"
    C.write_once(path, {"a": 1})
    C.write_once(path, {"a": 1})
    with pytest.raises(RuntimeError):
        C.write_once(path, {"a": 2})
    with pytest.raises(ValueError):
        C.write(tmp_path / "nan.json", {"a": math.nan})


def make_tree(root):
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "src").mkdir()
    (root / "src/a.py").write_text("x = 1\n")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "t"], check=True)
    return C.git_head(root)


def test_verify_release_refuses_changed_sources_head_and_pins(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    head = make_tree(root)
    release = {"status": "FROZEN_VALIDATED", "source_root": str(root), "source_commit": head, "code_sha256": "c",
               "source_files": {"src/a.py": C.sha(root / "src/a.py")},
               "data_lock": {"sha256": "e65c60a33f095933e673613fc58f2754485933524259e548c6a0f5dc380f72c5"}}
    path = tmp_path / "RELEASE.json"
    C.write(path, release)
    ref = {"path": str(path), "sha256": C.sha(path), "source_root": str(root), "source_commit": head, "code_sha256": "c"}
    assert C.verify_release(ref)["source_commit"] == head
    with pytest.raises(RuntimeError):
        C.verify_release({**ref, "sha256": "0" * 64})
    with pytest.raises(RuntimeError):
        C.verify_release({**ref, "code_sha256": "other"})
    (root / "src/a.py").write_text("x = 2\n")
    with pytest.raises(RuntimeError, match="Frozen source file changed"):
        C.verify_release(ref)


def test_frozen_inputs_of_the_preregistration_still_verify():
    """Read-only: the pinned frozen releases, their trees and the pipeline validation manifest are unchanged."""
    inputs = C.prereg()["frozen_inputs"]
    assert C.verify_release(inputs["comir_release"])["family"] == "CoMIR_native"
    assert C.verify_release(inputs["mind_release"])["family"] == "MIND_2D_adapted"
    assert C.verify_manifest(inputs["pipeline_validation_manifest"])["code_sha256"] == \
        "3f1405ed4272d54935f7d9ccc42cdb9e41e55469f154c5a2961fe6b9f36072ac"


def test_verify_freeze_requires_pin_and_unchanged_files(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "PROTOCOL.json").write_bytes(C.PREREGISTRATION.read_bytes())
    files = {"src/v10_sensitivity/common.py": C.sha(C.REPO / "src/v10_sensitivity/common.py")}
    C.write(bundle / "FREEZE.json", {"status": "FROZEN_FOR_SUBMISSION", "diagnostic_sources": {"files": files},
                                     "preregistration": {"sha256": C.sha(C.PREREGISTRATION)}})
    monkeypatch.setattr(C, "BUNDLE", bundle)
    pin = C.sha(bundle / "FREEZE.json")
    assert C.verify_freeze(pin)["status"] == "FROZEN_FOR_SUBMISSION"
    with pytest.raises(RuntimeError):
        C.verify_freeze("0" * 64)
    monkeypatch.delenv("V10D_FREEZE_SHA256", raising=False)
    with pytest.raises(RuntimeError):
        C.verify_freeze(None)
    C.write(bundle / "FREEZE.json", {"status": "FROZEN_FOR_SUBMISSION",
                                     "diagnostic_sources": {"files": {"src/v10_sensitivity/common.py": "0" * 64}},
                                     "preregistration": {"sha256": C.sha(C.PREREGISTRATION)}})
    with pytest.raises(RuntimeError, match="changed after freeze"):
        C.verify_freeze(C.sha(bundle / "FREEZE.json"))


def test_partition_rule():
    assert C.require_partition("D1", "L40S") == "L40S"
    assert C.require_partition("D3", "H100") == "H100"
    for diagnostic, partition in (("D1", "H100"), ("D2", "RTX6000PRO"), ("D3", "L40S"), ("D3", None)):
        with pytest.raises(RuntimeError):
            C.require_partition(diagnostic, partition or "")


def row(core, i, err, theta=0.0, f_hat=0.0, f_true=0.0):
    return {"core": core, "perturbation": i, "err_full": err, "theta": theta, "theta_hat": theta, "t_hat": [0.0, 0.0],
            "F_hat": f_hat, "F_true": f_true}


def test_dev24_rule_matches_decision_section_6_1():
    # FIT/CAL cores on the other slide at the eight decision positions (F4 H4 H9 D5 G3 G5 H2 I5)
    fit = ["2B_F4", "2B_H4", "2B_H9", "1B_D5", "1B_G3", "2B_Q1"]
    cal = ["1B_G5", "1B_H2", "1B_I5", "2B_Q2"]
    dev_cores = list(C.DEV24_EXCLUDED) + [f"1B_Z{i}" for i in range(24)]
    records = [row(c, i, 1.0) for c in dev_cores for i in range(10)]
    kept = C.dev24_rows(records, fit, cal)
    assert len(kept) == 240 and not {r["core"] for r in kept} & set(C.DEV24_EXCLUDED)
    with pytest.raises(RuntimeError):
        C.dev24_rows(records, fit[1:], cal)
    with pytest.raises(RuntimeError):
        C.dev24_rows(records, fit + ["2B_Z0"], cal)


def test_metrics_equal_the_frozen_records_metrics():
    from v10.supplement.records import metrics as frozen_metrics
    rows = [row("c", i, e) for i, e in enumerate((0.5, 8.9, 9.0, 41.9, 42.0, 300.0, 3.3))]
    assert S.metrics(rows) == frozen_metrics(rows)


def test_rotation_strata_boundaries():
    assert [S.stratum_of(t) for t in (0, -9.999, 10, 19.99, -20, 30, -30)] == \
        ["0-10", "0-10", "10-20", "10-20", "20-30", "20-30", "20-30"]
    with pytest.raises(RuntimeError):
        S.stratum_of(30.0001)
    rows = [row("a", 0, 1.0, 5), row("a", 1, 50.0, 25), row("a", 2, 1.0, 26)]
    strata = S.rotation_strata(rows)
    assert strata["0-10"]["success42"] == 1.0 and strata["10-20"] == {"n": 0}
    assert strata["20-30"]["n"] == 2 and strata["20-30"]["success42"] == 0.5 and strata["spread_success42"] == 0.5


def test_paired_differences_bootstrap_failure_split_and_replay():
    old = [row(c, i, 100.0 if (c == "b" and i < 5) else 1.0) for c in ("a", "b") for i in range(10)]
    new = [row(c, i, 1.0) for c in ("a", "b") for i in range(10)]
    diff = S.paired_core_differences(new, old, 42.0)
    assert diff == {"a": 0.0, "b": 0.5}
    with pytest.raises(RuntimeError):
        S.paired_core_differences(new[:-1], old, 42.0)
    ci = S.bootstrap_mean(diff.values())
    assert ci["mean"] == 0.25 and ci["ci_low"] <= 0.25 <= ci["ci_high"] and ci == S.bootstrap_mean(diff.values())
    assert S.averaged_core_differences([diff, {"a": 1.0, "b": 0.0}]) == {"a": 0.5, "b": 0.25}
    failures = [row("a", 0, 50.0, f_hat=2.0, f_true=1.0), row("a", 1, 60.0, f_hat=0.0, f_true=1.0), row("a", 2, 1.0)]
    assert S.failure_split(failures) == {"failures": 2, "energy_optimum_wrong": 1, "search_missed_higher_F": 1}
    same = S.replay_compare(json.loads(json.dumps(old)), old)
    assert same["status"] == "IDENTICAL" and same["identical_rows"] == 20
    moved = json.loads(json.dumps(old))
    moved[3]["t_hat"] = [0.0, 1e-3]
    other = S.replay_compare(moved, old)
    assert other["status"] == "MISMATCH_DISCLOSED" and other["identical_rows"] == 19
    assert other["max_abs_difference"]["t_hat"] == pytest.approx(1e-3)
