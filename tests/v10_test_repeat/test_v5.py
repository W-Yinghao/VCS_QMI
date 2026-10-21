"""CPU tests of TEST_REPEAT134 v5 (the inputs after v4's INPUT_LOCK failure).  No TEST image or label: the rule and the
builder run on synthetic DIAG.json files in a copied bundle (records and receipts are metadata); the A / B1 input stages run
on DEV cores (REG-P1 validation1 files) against REG-P1's own DEV cache.

    cd ssl_pilot_v10_test_repeat && OMP_NUM_THREADS=1 /home/infres/yinwang/CS_QMI/env/bin/python -m pytest -q tests/v10_test_repeat
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("V10_TR_BUNDLE", tempfile.mkdtemp(prefix="tr134_test_bundle_"))   # never the real bundle
os.environ.pop("SLURM_JOB_ID", None)
sys.path.insert(0, str(REPO / "scripts/v10_test_repeat"))
import tr_common as C  # noqa: E402
import build_freeze as B  # noqa: E402
import prepare_inputs as PI  # noqa: E402

PY = "/home/infres/yinwang/CS_QMI/env/bin/python"
BF1_SRC = C.HAND / "v10_bf1_20261017" / "source"
HOST_HASH = "6c6f094684fbc50a"
REAL_BUNDLE = C.HAND / "v10_test_repeat134_20261020"
R6 = Path("/home/infres/yinwang/CS_QMI/outputs/R6_REG")
DEV_DATA = Path("/projects/EEG-foundation-model/yinghao/data/bf_shg_4550300/validation1")
RAW_OK = {k: True for k in ("v10_test_repeat_manifest", "v9_g1_frozen_member_csv", "v10_data_manifest",
                            "v9_g1_prepared_manifest", "v10_prepared_manifest")}


# ------------------------------------------------------------------------------------------------- synthetic diagnostic
def summary(**kw) -> dict:
    s = {"raw_members_match": dict(RAW_OK), "v10_vs_v9_identical": False,
         "v10_vs_v9_differing_arrays": {"shg_log": {"n_differ": 2, "max_ulp": 1, "max_abs": 1.2e-7, "mean_abs": 1e-12}},
         "rebuild_deterministic": True, "rebuilds_equal_to_v10": ["default_1", "default_2"],
         "rebuilds_equal_to_v9_g1": ["no_X86_V4", "baseline_X86_V2"], "node_has_X86_V4": True,
         "v9_cache_written_in_original_sequence": True, "v9_cache_inside_v9_g1_prepare_window": True,
         "v9_manifest_pins_cache_sha256": False}
    s.update(kw)
    return s


def synthetic_diag(s: dict, v10_sha: str, node_v4: bool = True) -> dict:
    def run(env, log1p, add):
        return {"NPY_DISABLE_CPU_FEATURES": env,
                "dispatch": {"host": "nodecpu01", "cpu_model": "CPU X", "numpy": "2.5.2", "X86_V3": True, "X86_V4": node_v4,
                             "log1p_float64": log1p, "add_float64": add, "NPY_DISABLE_CPU_FEATURES": env}}
    k = "X86_V4" if node_v4 else "baseline(X86_V2)"
    return {"schema": "v10.test_repeat134.input_diagnostic.v1", "core": "1B_D8", "job": B.DIAG_JOB, "evaluated": False,
            "labels_or_results_read": False, "summary": s, "provenance": {"v10_rebuild": {"sha256": v10_sha}},
            "rebuilds": {"default_1": run("", k, "X86_V3"), "default_2": run("", k, "X86_V3"),
                         "no_X86_V4": run("X86_V4", "baseline(X86_V2)", "X86_V3"),
                         "baseline_X86_V2": run("X86_V3 X86_V4", "baseline(X86_V2)", "baseline(X86_V2)")},
            "authorization_sha256": B.DIAG_AUTH_SHA, "activation_sha256": C.sha(REPO / C.RECORDS / C.V4_ACTIVATION),
            "code_commit": B.DIAG_COMMIT, "input_lock_sha256": B.V4_LOCK_SHA, "v10_rebuild_matches_lock": True}


def variant_summary(variant: str) -> dict:
    return {"A": summary(), "B1": summary(rebuilds_equal_to_v9_g1=[]), "C": summary(rebuild_deterministic=False)}[variant]


def synthetic_ev(variant: str) -> dict:
    lock = C.read(REAL_BUNDLE / C.V4_LOCK)                                       # metadata only (hashes, core ids)
    v4file = next(f["sha256"] for f in lock["files"] if f["core"] == "1B_D8")
    paths = {"v4_lock": C.V4_LOCK, "v4_prepare_receipt": "exits/1051404.json",
             "diagnostic": "diagnostics/1B_D8/1051581/DIAG.json", "diagnostic_receipt": "diagnostics/exits/1051581.json"}
    return {"core": "1B_D8", "lock": lock, "diag": synthetic_diag(variant_summary(variant), v4file),
            "receipt": {"job": B.DIAG_JOB, "partition": "cpu-high", "nodelist": "nodecpu01"}, "v4_file_sha256": v4file,
            "paths": paths, "shas": {v: "0" * 64 for v in paths.values()}}


def protocol_for(p4: dict, man: dict, variant: str) -> dict:
    ev = synthetic_ev(variant)
    return B.protocol_v5(p4, man, B.select_v5(ev["diag"]), ev)


def p4_and_manifest():
    return C.read(REPO / C.RECORDS / C.V4_PROTOCOL), C.read(REPO / C.RECORDS / C.MANIFEST)


# ------------------------------------------------------------------------------------------------------------- the rule
def test_decision_v5_verbatim():
    d = B.decision_v5()
    assert d["question_verbatim"].startswith("TEST_REPEAT 的输入锁卡在 1B_D8") and d["question_verbatim"].endswith("怎么处理？")
    assert d["chosen_option_verbatim"] == "先诊断，按结果处理 (Recommended)"
    assert d["chosen_option_text_verbatim"].endswith("另加一个排除它的 TEST133 敏感性集。")
    assert [x["option"] for x in d["options_not_chosen"]] == ["直接排除 1B_D8", "先停在这里"]
    assert set(d["decision_rule_as_relayed"]) == {"fixed", "A", "B1", "C"}
    rec = REPO / C.RECORDS / C.DECISION_V5
    if rec.exists():                                                                # recorded before the builder opened DIAG.json
        assert rec.read_text() == C.dumps(d)


def test_select_v5_rule():
    sel = lambda **kw: B.select_v5({"summary": summary(**kw)})
    assert sel()["variant"] == "A" and sel()["reproducing_rebuild"] == "no_X86_V4"
    assert sel(rebuilds_equal_to_v9_g1=["baseline_X86_V2", "default_2"])["reproducing_rebuild"] == "default_2"   # fixed order
    assert sel(rebuilds_equal_to_v9_g1=[])["variant"] == "B1"                      # nothing reproduces V9-G1's file
    assert sel(v9_cache_inside_v9_g1_prepare_window=False)["variant"] == "B1"      # V9-G1's file outside its window
    assert sel(v9_cache_inside_v9_g1_prepare_window=None)["variant"] == "B1"
    assert sel(rebuild_deterministic=False)["variant"] == "C"
    assert sel(raw_members_match={**RAW_OK, "v10_data_manifest": False})["variant"] == "C"
    assert sel(raw_members_match={})["variant"] == "C"
    assert sel(rebuild_deterministic=False, rebuilds_equal_to_v9_g1=[])["variant"] == "C"   # C is checked first


def test_dispatch_pin():
    for node_v4 in (True, False):
        d = synthetic_diag(summary(), "0" * 64, node_v4)
        pin = B.dispatch_pin(d, "no_X86_V4")
        assert (pin["log1p_float64"], pin["add_float64"], pin["NPY_DISABLE_CPU_FEATURES"]) == ("baseline(X86_V2)", "X86_V3", "X86_V4")
        pin = B.dispatch_pin(d, "baseline_X86_V2")
        assert pin["NPY_DISABLE_CPU_FEATURES"] == "X86_V3 X86_V4"
        pin = B.dispatch_pin(d, "default_1")                                         # the node's own default kernels
        assert pin["NPY_DISABLE_CPU_FEATURES"] == ("" if node_v4 else "X86_V4") and pin["reproduced_by"] == "default_1"


def test_protocol_v5_variants():
    p4, man = p4_and_manifest()
    a = protocol_for(p4, man, "A")
    sp = a["inputs"]
    assert sp["v5"]["variant"] == "A" and sp["root"] == str(B.INPUTS_V5) != sp["v4_lock"]["root"]
    assert sp["numpy_dispatch"]["NPY_DISABLE_CPU_FEATURES"] == "X86_V4" and sp["job"]["nodelist"] == "nodecpu01"
    assert "sensitivity" not in a["sets"] and a["jobs"]["test_units"] == p4["jobs"]["test_units"]
    assert {k: v for k, v in a.items() if k not in ("inputs", "jobs", "version", "schema", "id", "status", "reporting",
                                                    "cost_estimate")} == {k: v for k, v in p4.items() if k not in (
        "inputs", "jobs", "version", "schema", "id", "status", "reporting", "cost_estimate")}       # nothing else changes
    b = protocol_for(p4, man, "B1")
    sp = b["inputs"]
    assert sp["v5"]["variant"] == "B1" and sp["root"] == sp["v4_lock"]["root"] and list(sp["accepted_exceptions"]) == ["1B_D8"]
    sens = b["sets"]["sensitivity"]
    assert list(sens) == ["TEST133", "TEST111"] and [len(x["cores"]) for x in sens.values()] == [133, 111]
    assert all("1B_D8" not in x["cores"] and x["excluded"] == ["1B_D8"] for x in sens.values())
    assert set(sens["TEST133"]["cores"]) == set(man["test"]["cores"]) - {"1B_D8"}
    assert set(sens["TEST111"]["cores"]) == set(man["test"]["sensitivity112"]["cores"]) - {"1B_D8"}
    assert b["sets"]["position_groups"] == {"TEST134": 97, "TEST112": 75, "TEST133": 96, "TEST111": 74}
    assert sens["TEST133"]["cores_sha256"] == C.sha_bytes(C.canonical(sorted(sens["TEST133"]["cores"])))
    assert b["jobs"]["test_units"] == p4["jobs"]["test_units"]
    with pytest.raises(RuntimeError, match="case C"):
        protocol_for(p4, man, "C")


# ------------------------------------------------------------------------------------------------- the v5 freeze, kept
def test_v5_records_unchanged_since_the_freeze():
    """v6 leaves v5 as frozen: the committed v5 records and the bundle copies equal 5fdcb942 byte for byte."""
    for name in (C.DECISION_V5, C.PROTOCOL, C.AMENDMENT, C.DELTA, C.V5_ACTIVATION):
        b = B.git_bytes(B.V5_COMMIT, f"{C.RECORDS}/{name}")
        assert (REPO / C.RECORDS / name).read_bytes() == b == (REAL_BUNDLE / name).read_bytes()
    assert C.sha(REAL_BUNDLE / C.V5_ACTIVATION) == B.V5_ACTIVATION_SHA


# ------------------------------------------------------------------------------------------------------- input_lock_v5
def test_input_lock_v5(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "BUNDLE", tmp_path)
    prot = {"inputs": {"root": "/r", "v5": {"variant": "B1"}}}
    with pytest.raises(RuntimeError, match="missing"):
        C.input_lock_v5("a" * 64, prot)
    base = {"status": "PASS", "job": "5", "activation_sha256": "a" * 64, "variant": "B1", "root": "/r"}
    C.write(tmp_path / C.LOCK, base)
    with pytest.raises(RuntimeError):                                                # no exit receipt
        C.input_lock_v5("a" * 64, prot)
    C.write(tmp_path / "exits" / "5.json", {"job": "5", "exit_code": 0})
    assert C.input_lock_v5("a" * 64, prot)["job"] == "5"
    for bad in ({"variant": "A"}, {"root": "/x"}, {"status": "FAIL_V5_B1"}, {"activation_sha256": "b" * 64}):
        C.write(tmp_path / C.LOCK, {**base, **bad})
        with pytest.raises(RuntimeError):
            C.input_lock_v5("a" * 64, prot)
    (tmp_path / C.LOCK).unlink()
    C.write(tmp_path / C.V4_LOCK, base)                                              # v4's lock never licenses v5
    with pytest.raises(RuntimeError, match="missing"):
        C.input_lock_v5("a" * 64, prot)


# ------------------------------------------------------------------------------------------------- input stages on DEV
def dev_cores(n=2) -> list:
    man = C.read(REPO / C.RECORDS / C.MANIFEST)
    dev = sorted(C.read(Path(man["units"][0]["unit_dir"]) / "DEV.identity.json")["cores"])
    cores = [c for c in dev if (DEV_DATA / f"{c}_WB.tif").exists() and (R6 / "cache" / f"{c}.npz").exists()][:n]
    assert len(cores) == n
    return cores


def raw_dir(tmp_path, cores) -> tuple:
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = []
    for c in cores:
        (raw / f"{c}_BF.tif").symlink_to(DEV_DATA / f"{c}_WB.tif")
        (raw / f"{c}_SHG.tif").symlink_to(DEV_DATA / f"{c}_MI.tif")
        rows.append({"core": c, "bf_sha256": C.sha(DEV_DATA / f"{c}_WB.tif"), "shg_sha256": C.sha(DEV_DATA / f"{c}_MI.tif")})
    return raw, rows


def stage_env(monkeypatch, tmp_path, cores, rows, inputs) -> None:
    rec = {"manifest": {"test": {"cores": cores, "raw": rows}}, "protocol": {"inputs": inputs}, "activation": {},
           "activation_sha256": "a" * 64, "shas": {C.MANIFEST: "m" * 64}}
    monkeypatch.setattr(C, "BUNDLE", tmp_path / "bundle")
    monkeypatch.setattr(C, "records", lambda pin=None: rec)
    monkeypatch.setattr(C, "verify_code", lambda a, c: {"commit": c})
    monkeypatch.setenv("SLURM_JOB_ID", "4300")
    monkeypatch.setenv("V10_TR_EXPECT_ACTIVATION", "a" * 64)
    monkeypatch.setenv("V10_TR_EXPECT_COMMIT", "c" * 40)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")


def inputs_common(raw, v9_root) -> dict:
    return {"raw_root": str(raw), "v9_prepared_root": str(v9_root), "host_source_root": str(BF1_SRC), "host_code_hash": HOST_HASH}


def login_pin() -> dict:
    d = PI.dispatch_info()
    return {"log1p_float64": "baseline(X86_V2)", "add_float64": d["add_float64"], "NPY_DISABLE_CPU_FEATURES": "X86_V4"}


def test_stage_a_on_dev(tmp_path, monkeypatch):
    """Variant A on two DEV cores against REG-P1's own cache: the pinned-dispatch rebuild into a new root locks PASS."""
    cores = dev_cores()
    raw, rows = raw_dir(tmp_path, cores)
    inputs = {**inputs_common(raw, R6), "root": str(tmp_path / "v5root"), "numpy_dispatch": login_pin(),
              "v4_lock": {"root": str(tmp_path / "v4root"), "sha256": "x"}, "v5": {"variant": "A"}}
    stage_env(monkeypatch, tmp_path, cores, rows, inputs)
    assert PI.main([]) == 0
    lk = C.read(C.BUNDLE / C.LOCK)
    assert (lk["status"], lk["variant"], lk["root"], lk["raw_members_verified"]) == ("PASS", "A", inputs["root"], 4)
    assert lk["v9_cache_comparison"]["identical"] is True and [f["core"] for f in lk["files"]] == sorted(cores)
    assert lk["build_dispatch"]["log1p_float64"] == "baseline(X86_V2)" == lk["numpy_dispatch_pin"]["log1p_float64"]
    assert lk["build_dispatch"]["NPY_DISABLE_CPU_FEATURES"] == "X86_V4"
    with pytest.raises(RuntimeError, match="locked once"):
        PI.main([])


def test_stage_a_refuses_wrong_dispatch_before_reading(tmp_path, monkeypatch):
    cores = dev_cores(1)
    raw, rows = raw_dir(tmp_path, cores)
    if PI.dispatch_info()["X86_V4"]:
        pytest.skip("this node has X86_V4")
    pin = {"log1p_float64": "X86_V4", "add_float64": "X86_V3", "NPY_DISABLE_CPU_FEATURES": ""}
    inputs = {**inputs_common(raw, R6), "root": str(tmp_path / "v5root"), "numpy_dispatch": pin,
              "v4_lock": {"root": str(tmp_path / "v4root"), "sha256": "x"}, "v5": {"variant": "A"}}
    stage_env(monkeypatch, tmp_path, cores, rows, inputs)
    with pytest.raises(RuntimeError, match="no lock written"):
        PI.main([])
    assert not (C.BUNDLE / C.LOCK).exists() and not (tmp_path / "v5root" / "manifest.json").exists()


def build_root(raw, rows, root) -> None:
    table = {r["core"]: {"bf_sha256": r["bf_sha256"], "shg_sha256": r["shg_sha256"]} for r in rows}
    code = ("import json, sys; sys.path.insert(0, %r); import prepare_inputs as PI; from pathlib import Path; "
            "P, h = PI.host_modules(Path(%r), %r); PI.prepare(P, json.loads(%r), Path(%r), Path(%r))"
            % (str(REPO / "scripts/v10_test_repeat"), str(BF1_SRC), HOST_HASH, json.dumps(table), str(raw), str(root)))
    subprocess.run([PY, "-c", code], env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True, capture_output=True)


def nudge(npz: Path) -> None:
    with np.load(npz) as z:
        a = {k: z[k].copy() for k in z.files}
    a["shg_log"][10, 11] = np.nextafter(a["shg_log"][10, 11], np.float32(np.inf))
    np.savez_compressed(npz, **a)


@pytest.fixture(scope="module")
def b1_roots(tmp_path_factory):
    t = tmp_path_factory.mktemp("b1")
    cores = dev_cores()
    raw, rows = raw_dir(t, cores)
    build_root(raw, rows, t / "v4root")
    return t, cores, raw, rows


def b1_case(tmp_path, monkeypatch, b1_roots, nudge_cores, exceptions, change_file=False, lock_sha=None):
    t, cores, raw, rows = b1_roots
    v4root, v9root = tmp_path / "v4root", tmp_path / "v9root"
    shutil.copytree(t / "v4root", v4root)
    shutil.copytree(t / "v4root", v9root)
    for c in nudge_cores:
        nudge(v9root / "cache" / f"{c}.npz")
    stage_env(monkeypatch, tmp_path, cores, rows, {})
    v4lock = {"status": "FAIL_V9_CACHE_MISMATCH", "root": str(v4root), "files": PI.lock_files(v4root, cores),
              "prepared_manifest_sha256": C.sha(v4root / "manifest.json"), "omega": [117, 394], "size": 512, "down": 4}
    C.write(C.BUNDLE / C.V4_LOCK, v4lock)
    if change_file:
        nudge(v4root / "cache" / f"{cores[0]}.npz")                                  # changed after the v4 lock
    inputs = {**inputs_common(raw, v9root), "root": str(v4root),
              "v4_lock": {"root": str(v4root), "sha256": lock_sha or C.sha(C.BUNDLE / C.V4_LOCK)},
              "accepted_exceptions": {c: {"file_sha256": C.sha(t / "v4root" / "cache" / f"{c}.npz")} for c in exceptions},
              "v5": {"variant": "B1"}}
    C.records()["protocol"]["inputs"] = inputs                                     # the monkeypatched records
    return cores


def test_stage_b1_on_dev(tmp_path, monkeypatch, b1_roots):
    cores = b1_case(tmp_path, monkeypatch, b1_roots, nudge_cores=[dev_cores()[1]], exceptions=[dev_cores()[1]])
    assert PI.main([]) == 0
    lk = C.read(C.BUNDLE / C.LOCK)
    assert (lk["status"], lk["variant"], lk["raw_members_verified"]) == ("PASS", "B1", 4)
    assert lk["v9_cache_comparison"]["differing"] == [cores[1]] and lk["exceptions_are_exactly_the_differing_cores"] is True
    assert lk["changed_since_v4_lock"] == [] and lk["prepared_manifest_unchanged"] is True


@pytest.mark.parametrize("case", ["extra_difference", "missing_difference", "file_changed"])
def test_stage_b1_fails(tmp_path, monkeypatch, b1_roots, case):
    c0, c1 = dev_cores()
    kw = {"extra_difference": dict(nudge_cores=[c0, c1], exceptions=[c1]),
          "missing_difference": dict(nudge_cores=[], exceptions=[c1]),
          "file_changed": dict(nudge_cores=[c1], exceptions=[c1], change_file=True)}[case]
    b1_case(tmp_path, monkeypatch, b1_roots, **kw)
    assert PI.main([]) == 3
    lk = C.read(C.BUNDLE / C.LOCK)
    assert lk["status"] == "FAIL_V5_B1"
    if case == "file_changed":
        assert lk["changed_since_v4_lock"] == [c0]


def test_stage_b1_refuses_another_v4_lock(tmp_path, monkeypatch, b1_roots):
    c1 = dev_cores()[1]
    b1_case(tmp_path, monkeypatch, b1_roots, nudge_cores=[c1], exceptions=[c1], lock_sha="0" * 64)
    with pytest.raises(RuntimeError, match="v5 pin"):
        PI.main([])
    assert not (C.BUNDLE / C.LOCK).exists()


def test_check_pins_mode(monkeypatch, tmp_path):
    stage_env(monkeypatch, tmp_path, [], [], {"v5": {"variant": "B1"}})
    monkeypatch.delenv("SLURM_JOB_ID")
    assert PI.main(["--check-pins"]) == 0
    with pytest.raises(RuntimeError):                                               # the stage itself needs Slurm
        PI.main([])


def test_build_child_refuses_outside_its_job(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.delenv("V10_TR_INPUTS_PARENT", raising=False)
    with pytest.raises(RuntimeError, match="child"):
        PI.build_child({"kernels": {}, "cores": {}})


def test_cpu_sbatch():
    p = REPO / "slurm/v10_test_repeat/cpu.sbatch"
    subprocess.run(["bash", "-n", str(p)], check=True)
    text = p.read_text()
    for needle in ("--gres=gpu:0", "V10_TR_EXPECT_ACTIVATION:?", "V10_TR_EXPECT_COMMIT:?", "unset NPY_DISABLE_CPU_FEATURES",
                   "ssl_pilot_v10_test_repeat_exec_v5_20261020", '"stage": "inputs_v5"', 'p = Path(sys.argv[1]) / "exits"'):
        assert needle in text
    assert "gpu:1" not in text and "#SBATCH --partition" not in text                 # partition / node on the sbatch line
