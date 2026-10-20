"""CPU tests of the V10 TEST_REPEAT134 stage.  No GPU, no TEST image or label: TEST data appear only as manifest metadata
(core ids, member sha256); every data-touching test uses DEV / FIT records or synthetic files.

    cd ssl_pilot_v10_test_repeat && OMP_NUM_THREADS=1 /home/infres/yinwang/CS_QMI/env/bin/python -m pytest -q tests/v10_test_repeat
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
os.environ["V10_TR_BUNDLE"] = tempfile.mkdtemp(prefix="tr134_test_bundle_")   # never the real bundle
os.environ.pop("SLURM_JOB_ID", None)
sys.path.insert(0, str(REPO / "scripts/v10_test_repeat"))
import tr_common as C  # noqa: E402
import evaluate as E  # noqa: E402
import build_freeze as B  # noqa: E402
import analyze as AN  # noqa: E402

PY = "/home/infres/yinwang/CS_QMI/env/bin/python"
BF1_SRC = C.HAND / "v10_bf1_20261017" / "source"
SUPP_SOURCES = {"pipeline": C.CS / "ssl_pilot_v10_pipeline_20261019", "native": C.CS / "ssl_pilot_v10_native_recovery_20261019",
                "rv2_xhw": C.CS / "ssl_pilot_v10_rulsif2_xhw"}
SIX = ("MF_VCS_M1", "MF_VCS_M0", "MF_VCS_perm1", "MLP_VCS_perm1", "MLP_VCS_perm8", "InfoNCE_host[J_perm1]")


@pytest.fixture(scope="module")
def pr():
    return B.load_phase2()


@pytest.fixture(scope="module")
def recs(pr):
    return B.build(pr)


REAL_BUNDLE = C.HAND / "v10_test_repeat134_20261020"


@pytest.fixture(scope="module")
def recs4(pr):
    """The v4 records, built read-only against the real bundle (it holds the observed DEV re-evaluations)."""
    saved = C.BUNDLE
    C.BUNDLE = REAL_BUNDLE
    try:
        b4 = B.build_v4(pr)
        texts = B.render_v4(pr)
    finally:
        C.BUNDLE = saved
    return {**b4, "texts": texts}


def run(code: str, cwd=None, env=None) -> str:
    e = {**os.environ, "OMP_NUM_THREADS": "1", **(env or {})}
    return subprocess.run([PY, "-c", code], cwd=cwd, env=e, capture_output=True, text=True, check=True).stdout


# ------------------------------------------------------------------------------------------------------------ records
def test_manifest_units(recs):
    m = recs["manifest"]
    assert m["unit_counts"] == {"total": 42, "primary": 30, "secondary_replication": 12}
    prim = Counter(u["family"] for u in m["units"] if u["tier"] == "primary")
    assert prim == {"MF_VCS_perm1": 3, "MF_VCS_M0": 3, "MF_VCS_M1": 3, "MLP_VCS_perm1": 3, "MLP_VCS_perm8": 3,
                    "InfoNCE_host[J_perm1]": 3, "MI_native": 1, "NMI_native": 1, "CoMIR_native": 3, "RuLSIF_exact": 3,
                    "MIND_2D_adapted": 1, "RuLSIF_exact_v2": 3}
    sec = sorted((u["family"], u["seed"]) for u in m["units"] if u["tier"] == "secondary_replication")
    assert sec == sorted((f, s) for f in SIX for s in (1004, 1005))
    labels = {u["family"]: u["row_label"] for u in m["units"]}
    assert labels["RuLSIF_exact"] == "failed_adaptation" and labels["RuLSIF_exact_v2"] == "post_gate_amendment_A1"
    excluded = " ".join(x["what"] for x in m["excluded_from_test_repeat"])
    assert "D1" in excluded and "D2" in excluded and "K8/K32" in excluded and "hw_replicates" in excluded
    for u in m["units"]:
        d = Path(u["unit_dir"])
        assert C.sha(d / u["checkpoint"]["path"]) == u["checkpoint"]["sha256"]
        assert C.read(d / "COMPLETE.json")["selected"]["checkpoint_sha256"] == u["checkpoint"]["sha256"]
        assert C.sha(d / "COMPLETE.json") == u["complete_sha256"]
        assert u["test_gpu_type"] == u["hardware"]["gpu_type"] == C.GPU_TYPES[u["hardware"]["gpu_name"]]
        assert u["code"]["kind"] in ("bf1", "supp")


def test_test_metadata(recs, pr):
    t = recs["manifest"]["test"]
    assert len(t["cores"]) == 134 == len(set(t["cores"])) and t["cores"] == sorted(t["cores"])
    assert t["cores_sha256"] == C.sha_bytes(C.canonical(t["cores"]))
    s = t["sensitivity112"]
    assert len(s["cores"]) == 112 and len(s["excluded"]) == 22 and sorted(s["cores"] + s["excluded"]) == t["cores"]
    assert len(t["position_groups"]) == 97 and len(s["position_groups"]) == 75
    p2 = C.read(pr.REPO / "v10/protocol/V10_PROTOCOL_P2_20261017.json")["tma_colocation_sensitivity"]["TEST_REPEAT"]
    assert sorted(p2["full_ids"]) == t["cores"] and sorted(p2["retained_ids"]) == s["cores"]
    assert len(t["raw"]) == 134 and all(len(r["bf_sha256"]) == 64 and len(r["shg_sha256"]) == 64 for r in t["raw"])
    dev = set(C.read(Path(recs["manifest"]["units"][0]["unit_dir"]) / "DEV.identity.json")["cores"])
    assert not dev & set(t["cores"])                 # DEV32 and TEST134 are disjoint


def test_protocol(recs, pr):
    p, m = recs["protocol"], recs["manifest"]
    pairs = {(u["family"], u["test_gpu_type"]) for u in m["units"]}
    plan = {(g["family"], g["gpu_type"]): g for g in p["dev_gate"]["plan"]}
    assert set(plan) == pairs and len(plan) == len(p["dev_gate"]["plan"])
    keys = {u["key"]: u for u in m["units"]}
    for (fam, gpu), g in plan.items():
        u = keys[g["unit"]]
        assert u["family"] == fam and u["hardware"]["gpu_type"] == gpu
    assert {f for f, g in pairs if g == "L40S"} == set(m["families"])
    assert [tuple(c) for c in p["statistics"]["paired_contrasts"]] == list(pr.CONTRASTS)
    assert p["statistics"]["bootstrap"]["B"] == 10000
    assert any("never pooled" in r for r in p["reporting"]["rules"])
    for x in p["jobs"]["test_units"]:
        h, mi, _ = map(int, x["time_limit"].split(":"))
        assert (h * 60 + mi) * 60 >= 1.5 * x["estimated_seconds"]
    assert recs["decision"]["chosen_option_verbatim"] == "开始 (Recommended)"


# --------------------------------------------------------------------------------------------------------------- guard
def fake_lock(tmp_path, cores):
    root = tmp_path / "inputs"
    (root / "cache").mkdir(parents=True)
    files = []
    for c in cores:
        p = root / "cache" / f"{c}.npz"
        p.write_bytes(("synthetic " + c).encode())
        files.append({"core": c, "path": str(p), "sha256": C.sha(p)})
    return {"root": str(root), "files": files, "manifest_sha256": "0" * 64}


def test_guard(tmp_path, monkeypatch):
    cores = [f"9Z_X{i}" for i in range(134)]
    lock = fake_lock(tmp_path, cores)
    man = {"test": {"cores": cores}}
    with pytest.raises(RuntimeError):
        C.TestRepeatGuard(man, lock, "a" * 64)                     # outside Slurm
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("V10_TR_EXPECT_ACTIVATION", "b" * 64)
    with pytest.raises(RuntimeError):
        C.TestRepeatGuard(man, lock, "a" * 64)                     # activation pin differs
    monkeypatch.setenv("V10_TR_EXPECT_ACTIVATION", "a" * 64)
    g = C.TestRepeatGuard(man, lock, "a" * 64)
    assert g.included("TEST_REPEAT") == sorted(cores)
    with pytest.raises(RuntimeError):
        g.included("DEV")
    for roles in ({"DEV"}, {"TEST_REPEAT", "DEV"}, {"FIT"}, set()):
        with pytest.raises(PermissionError):
            g.check(cores[0], roles, "t")
    with pytest.raises(PermissionError):
        g.check("1B_B5", {"TEST_REPEAT"}, "t")                     # a DEV core is not a TEST_REPEAT core
    assert g.check(cores[0], {"TEST_REPEAT"}, "t") == Path(lock["root"]) / "cache" / f"{cores[0]}.npz"
    g.log_rule(cores[1], {"TEST_REPEAT"}, "rule")
    assert g.reads[-1]["kind"] == "rule" and g.reads[-1]["path"] is None and len(g.reads) == 2
    (Path(lock["root"]) / "cache" / f"{cores[2]}.npz").write_bytes(b"tampered")
    with pytest.raises(PermissionError):
        g.check(cores[2], {"TEST_REPEAT"}, "t")
    with pytest.raises(RuntimeError):
        C.TestRepeatGuard({"test": {"cores": cores[:133]}}, lock, "a" * 64)


# ------------------------------------------------------------------------------------------------------ gate compare
def test_compare_rows():
    base = [{"core": c, "perturbation": i, "err_full": 1.0 + i, "F_hat": 0.5, "t_hat": [1.0, 2.0], "seconds": 1.0,
             "job": "1", "role": "DEV"} for c in ("A", "B") for i in range(10)]
    same = [dict(r) for r in base]
    assert C.compare_rows(base, same)["status"] == "PASS"
    timing = [dict(r, seconds=9.9, job="2", full_registration_seconds=3.0) for r in base]
    assert C.compare_rows(base, [dict(r, seconds=9.9, job="2") for r in base])["status"] == "PASS"
    assert C.compare_rows(base, timing)["status"] == "PASS"
    ulp = [dict(r) for r in base]
    ulp[3]["err_full"] = np.nextafter(ulp[3]["err_full"], 10.0)
    res = C.compare_rows(base, ulp)
    assert res["status"] == "FAIL" and res["first_mismatches"][0]["field"] == "err_full"
    lst = [dict(r) for r in base]
    lst[0]["t_hat"] = [1.0, 2.0000001]
    assert C.compare_rows(base, lst)["status"] == "FAIL"
    assert C.compare_rows(base, base[:-1])["status"] == "FAIL"
    extra = [dict(r) for r in base]
    extra[0]["new_field"] = 1
    assert C.compare_rows(base, extra)["status"] == "FAIL"


# ------------------------------------------------------------------------------------------------- frozen sources
@pytest.mark.parametrize("name", sorted(SUPP_SOURCES))
def test_session_replica(name, tmp_path):
    reg = SUPP_SOURCES[name] / "src/v10/supplement/registration.py"
    assert E.check_session_replica(reg)["body_statements"] == 6
    bad = tmp_path / "registration.py"
    bad.write_text(reg.read_text().replace('"V10 supplemental frozen perturbation rule"', '"changed"'))
    with pytest.raises(RuntimeError):
        E.check_session_replica(bad)


def test_bf1_source_frozen(recs):
    pin = next(u for u in recs["manifest"]["units"] if u["code"]["kind"] == "bf1")["code"]["freeze"]["sha256"]
    out = run("import sys; sys.path.insert(0, 'scripts/v10_bf1'); import common; f = common.freeze(); print(f['source_commit'])",
              cwd=BF1_SRC, env={"V10_BF1_EXPECT_FREEZE": pin})
    assert out.strip() == "b8f59ea825dc7a2064457d47a628bef09ca4e185"


def test_supplement_sources_identical(recs):
    releases = {}
    for u in recs["manifest"]["units"]:
        if u["code"]["kind"] == "supp":
            releases[u["code"]["release"]["path"]] = u["code"]
    assert len(releases) == 8                       # 7 supplement families + the RuLSIF_exact_v2 cross-hardware release
    for path, code in releases.items():
        rel = C.read(path)
        out = run("import json, sys; sys.path.insert(0, 'src'); from v10.supplement import contract; "
                  "inv = contract.source_inventory(); print(json.dumps([inv, contract.digest(inv)]))", cwd=code["source_root"])
        inv, digest = json.loads(out)
        assert inv == rel["source_files"] and digest == rel["code_sha256"] == code["code_sha256"]


def test_perturbation_rule_matches_dev_records(recs):
    u = next(x for x in recs["manifest"]["units"] if x["key"] == "MF_VCS_M1__1001")
    rows = C.lines(Path(u["unit_dir"]) / "DEV.jsonl")
    cores = sorted({r["core"] for r in rows})[:4]
    out = run("import json, sys; sys.path.insert(0, 'src'); from v9.reg import g1 as G; "
              f"print(json.dumps({{c: G.Q.perturbations_p2(c) for c in {cores!r}}}))", cwd=BF1_SRC)
    rule = json.loads(out)
    for r in rows:
        if r["core"] in rule:
            assert rule[r["core"]][r["perturbation"]] == [r["theta"], r["t"][0], r["t"][1]]


def test_prepare_fixture_on_dev(tmp_path, recs):
    """The prepare path (raw sha check -> REG-P1 cmd_prepare verbatim -> array comparison) on two DEV cores, against the R6
    cache that REG-P1 built from the same files.  The fixture never names a TEST file."""
    data = Path("/projects/EEG-foundation-model/yinghao/data/bf_shg_4550300/validation1")
    dev = sorted(C.read(Path(recs["manifest"]["units"][0]["unit_dir"]) / "DEV.identity.json")["cores"])
    cores = [c for c in dev if (data / f"{c}_WB.tif").exists()][:2]
    assert len(cores) == 2
    raw = tmp_path / "raw"
    raw.mkdir()
    table = {}
    for c in cores:
        (raw / f"{c}_BF.tif").symlink_to(data / f"{c}_WB.tif")
        (raw / f"{c}_SHG.tif").symlink_to(data / f"{c}_MI.tif")
        table[c] = {"bf_sha256": C.sha(data / f"{c}_WB.tif"), "shg_sha256": C.sha(data / f"{c}_MI.tif")}
    code = ("import json, sys; sys.path.insert(0, %r); import prepare_inputs as PI; from pathlib import Path; "
            "P, h = PI.host_modules(Path(%r), '6c6f094684fbc50a'); "
            "PI.prepare(P, json.loads(%r), Path(%r), Path(%r)); "
            "print(json.dumps(PI.compare_cache(Path(%r), Path('/home/infres/yinwang/CS_QMI/outputs/R6_REG'), json.loads(%r))))"
            % (str(REPO / "scripts/v10_test_repeat"), str(BF1_SRC), json.dumps(table), str(raw), str(tmp_path / "root"),
               str(tmp_path / "root"), json.dumps(cores)))
    res = json.loads(run(code).strip().splitlines()[-1])
    assert res["identical"] is True and res["cores"] == 2


def test_stages_refuse_outside_slurm():
    with pytest.raises(RuntimeError):
        E.main(["test", "MF_VCS_M1__1001"])
    with pytest.raises(RuntimeError):
        E.main(["gate", "L40S__MF_VCS_M1"])
    import prepare_inputs as PI
    with pytest.raises(RuntimeError):
        PI.main()


def test_sbatch():
    text = (REPO / "slurm/v10_test_repeat/gpu.sbatch").read_text()
    subprocess.run(["bash", "-n", str(REPO / "slurm/v10_test_repeat/gpu.sbatch")], check=True)
    for needle in ("--exclude=node51,node53,node60", "V10_TR_EXPECT_ACTIVATION:?", "V10_TR_EXPECT_COMMIT:?",
                   "CUBLAS_WORKSPACE_CONFIG=:4096:8", "unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE NVIDIA_TF32_OVERRIDE",
                   "OMP_NUM_THREADS=4", '"exit_code"', "ssl_pilot_v10_test_repeat_exec_v4_20261020"):
        assert needle in text


# ------------------------------------------------------------------------------------------------------------ analysis
def test_analysis_dev_beside_reproduces_phase2(recs, recs4, pr):
    """With synthetic TEST rows for two families, the v4 analysis computes TEST and same-job DEV contrasts, and its
    recorded-DEV32 contrasts equal the committed phase-2 sidecar (same function, same keys)."""
    AN.PR = pr
    man = recs["manifest"]
    rng = np.random.default_rng(0)
    plan = {t["unit"]: t for t in recs4["protocol"]["jobs"]["test_units"]}
    units = []
    for u in man["units"]:
        dev = C.lines(Path(u["unit_dir"]) / "DEV.jsonl")
        e = {"u": u, "plan": plan[u["key"]], "dev_rows": dev, "test": None, "blocked": None, "reeval_rows": None}
        if u["family"] in ("MF_VCS_M1", "MF_VCS_M0") and u["tier"] == "primary":
            rows = [{"core": c, "perturbation": i, "err_full": float(rng.gamma(2.0, 4.0))} for c in man["test"]["cores"]
                    for i in range(10)]
            kept = set(man["test"]["sensitivity112"]["cores"])
            e["reeval_rows"] = dev
            e["test"] = {"comp": {"TEST134": C.metrics(rows), "TEST112": C.metrics([r for r in rows if r["core"] in kept]),
                                  "dev_reevaluation": {"DEV32": C.metrics(dev)}, "job": "0",
                                  "hardware": {"gpu_type": u["hardware"]["gpu_type"]}}, "rows": rows, "alloc_s": 0}
        units.append(e)
    shas = {n: "0" * 64 for n in C.FROZEN_RECORDS + (C.ACTIVATION,)}
    D = {"rec": {"manifest": man, "protocol": recs4["protocol"], "decision": recs["decision"], "decision_v4": recs4["decision_v4"],
                 "shas": shas}, "lock": None, "receipts": {}, "units": units, "truth_keys": 0, "noise": []}
    A = AN.analyse(D, pr)
    side = json.loads((pr.REPO / "v10/V10_PHASE2_RESULTS_20261019.json").read_text())
    ref = {(c["A"], c["B"]): c for c in side["contrasts"]}
    for c in A["contrasts"]:
        d, r = c["DEV32_recorded"], ref[(c["A"], c["B"])]["DEV32"]
        assert d["point"]["s42"] == r["point"]["s42"] and list(d["core_ci"]["s42"]) == list(r["core_ci"]["s42"])
        if {c["A"], c["B"]} == {"MF_VCS_M1", "MF_VCS_M0"}:
            assert c["TEST134"] is not None and c["DEV32_same_job"]["point"]["s42"] == d["point"]["s42"]
        else:
            assert c["TEST134"] is None and c["DEV32_same_job"] is None
    md = AN.render(D, A)
    assert "pending" in md and "TEST_REPEAT134" in md and "不是独立确认" in md and "不可解释" in md


# ------------------------------------------------------------------------------------------- stage plumbing (no GPU)
def synthetic_rows(cores, role, ck, seed=1):
    rng = np.random.default_rng(seed)
    return [{"theta": 1.0, "t": [0.5, -0.5], "theta_hat": 1.0, "t_hat": [0.5, -0.5], "err_full": float(rng.gamma(2.0, 4.0)),
             "F_hat": 0.1, "F_true": 0.1, "n_grid": 10668, "seconds": 0.01, "core": c, "perturbation": i, "role": role,
             "checkpoint_sha256": ck} for c in cores for i in range(10)]


def plumbing(monkeypatch, recs, key, protocol=None):
    man, prot = recs["manifest"], protocol or recs["protocol"]
    u = next(x for x in man["units"] if x["key"] == key)
    rec = {"manifest": man, "protocol": prot, "decision": recs["decision"], "activation_sha256": "a" * 64,
           "shas": {C.MANIFEST: "m" * 64}, "code": {"commit": "c" * 40}, "job": "777"}
    monkeypatch.setattr(E, "pins", lambda: rec)
    monkeypatch.setattr(E, "load_source", lambda unit: (type("R", (), {"A": None})(), {"kind": unit["code"]["kind"]}))
    gpu = {"L40S": "NVIDIA L40S", "RTX6000PRO": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "H100": "NVIDIA H100 NVL"}
    monkeypatch.setattr(E, "environment", lambda A, p, t: {"gpu_name": gpu[t], "gpu_type": t, "host": "nodeX",
                                                            "slurm_partition": t})
    return u, rec


def test_stage_gate_plumbing(monkeypatch, recs, recs4):
    u, rec = plumbing(monkeypatch, recs, "MF_VCS_M1__1001", recs4["protocol"])
    g = next(x for x in recs4["protocol"]["dev_gate"]["plan"] if x["unit"] == u["key"])
    recorded = C.lines(Path(u["unit_dir"]) / "DEV.jsonl")

    def fake_eval(run, unit, role, test_guard, out_path, budget, rows=None):
        assert role == "DEV" and test_guard is None
        C.write(out_path.with_suffix(".identity.json"), C.read(Path(unit["unit_dir"]) / "DEV.identity.json"))
        for r in (rows or recorded):
            C.append(out_path, dict(r, seconds=0.5))
        return {"metrics": C.metrics(rows or recorded)}, {"data_guard": [{"role": "FIT"}, {"role": "DEV"}], "test_guard": []}
    monkeypatch.setattr(E, "evaluate_unit", fake_eval)
    assert E.stage_gate(g["gate_id"]) == 0
    gate = C.read(C.BUNDLE / "gate" / g["gate_id"] / "777" / "GATE.json")
    assert gate["status"] == "PASS" and gate["rows"]["rows_new"] == 320 and gate["test_read"] is False
    changed = [dict(r) for r in recorded]
    changed[5]["err_full"] += 1e-12
    monkeypatch.setattr(E, "evaluate_unit", lambda *a, **k: fake_eval(*a, rows=changed))
    monkeypatch.setattr(E, "pins", lambda: {**rec, "job": "778"})
    assert E.stage_gate(g["gate_id"]) == 3
    assert C.read(C.BUNDLE / "gate" / g["gate_id"] / "778" / "GATE.json")["status"] == "FAIL"


def v4_setup(monkeypatch, recs, recs4, tmp_path, key):
    monkeypatch.setattr(C, "BUNDLE", tmp_path / "bundle")                         # one isolated bundle per test
    u, rec = plumbing(monkeypatch, recs, key, recs4["protocol"])
    man = recs["manifest"]
    lock = {**fake_lock(tmp_path, man["test"]["cores"]), "status": "PASS", "job": "700", "activation_sha256": "a" * 64}
    C.write(C.BUNDLE / "INPUT_LOCK.json", lock)
    C.write(C.BUNDLE / "exits" / "700.json", {"job": "700", "exit_code": 0})
    monkeypatch.setenv("SLURM_JOB_ID", "777")
    monkeypatch.setenv("V10_TR_EXPECT_ACTIVATION", "a" * 64)
    return u, man


def v4_evaluator(seen, dev_rows=None, dev_error=None):
    def fake_eval(run, unit, role, guard, out_path, budget):
        seen.append(role)
        if role == "DEV":
            assert guard is None
            if dev_error is not None:
                raise dev_error
            rows = dev_rows if dev_rows is not None else C.lines(Path(unit["unit_dir"]) / "DEV.jsonl")
            C.write(out_path.with_suffix(".identity.json"), C.read(Path(unit["unit_dir"]) / "DEV.identity.json"))
            for r in rows:
                C.append(out_path, r)
            return {"metrics": C.metrics(rows)}, {"data_guard": [{"role": "FIT"}, {"role": "DEV"}], "test_guard": []}
        assert role == "TEST_REPEAT" and isinstance(guard, C.TestRepeatGuard)
        cores = guard.included(role)
        for c in cores:
            guard.log_rule(c, {role}, "rule")
        rows = synthetic_rows(cores, role, unit["checkpoint"]["sha256"])
        for r in rows:
            C.append(out_path, r)
        return {"metrics": C.metrics(rows)}, {"data_guard": [{"role": "FIT"}], "test_guard": guard.reads}
    return fake_eval


@pytest.mark.parametrize("key", ["MIND_2D_adapted__1001", "MF_VCS_M1__1004"])
def test_unit_job_pairs_dev_and_test(monkeypatch, recs, recs4, tmp_path, key):
    u, man = v4_setup(monkeypatch, recs, recs4, tmp_path, key)
    recorded = C.lines(Path(u["unit_dir"]) / "DEV.jsonl")
    moved = [dict(r, err_full=r["err_full"] + 0.01, theta_hat=r["theta_hat"] + 1e-4) for r in recorded]
    seen = []
    monkeypatch.setattr(E, "evaluate_unit", v4_evaluator(seen, moved))
    assert E.stage_test(key) == 0 and seen == ["DEV", "TEST_REPEAT"]               # DEV first, then TEST, one job
    out = C.BUNDLE / "results" / "TEST_REPEAT" / key
    comp = C.read(out / "COMPLETE.json")
    rows = C.lines(out / "TEST_REPEAT.jsonl")
    ref = comp["dev_reevaluation"]
    assert comp["TEST134"] == C.metrics(rows) and ref["check"]["status"] == "PASS" and ref["DEV32"] == C.metrics(moved)
    assert ref["recorded"]["DEV32"] == u["dev"]["DEV32"] and comp["hardware"]["gpu_type"] == u["hardware"]["gpu_type"]
    assert comp["single_job"] is True and comp["jobs"] == ["777"] and comp["scope"] == "evaluator_only"
    assert all(C.sha(out / n) == v for n, v in comp["artifacts"].items())
    assert E.stage_test(key) == 0 and C.read(out / "COMPLETE.json") == comp            # never rerun or overwritten


@pytest.mark.parametrize("what", ["F_true", "band", "truth", "config"])
def test_unit_job_blocks_before_test(monkeypatch, recs, recs4, tmp_path, what):
    key = "MF_VCS_perm1__1001"
    u, man = v4_setup(monkeypatch, recs, recs4, tmp_path, key)
    recorded = C.lines(Path(u["unit_dir"]) / "DEV.jsonl")
    rows, err = [dict(r) for r in recorded], None
    if what == "F_true":
        rows[3]["F_true"] = rows[3]["F_true"] + 1e-7
    elif what == "band":
        for r in rows[:40]:
            r["err_full"] = r["err_full"] + 50.0                                     # a gross error: many <42 / <9 flips
    elif what == "truth":
        rows[0]["t"] = [rows[0]["t"][0] + 1e-9, rows[0]["t"][1]]
    else:
        err = RuntimeError("resolved BF1 configuration differs")
    seen = []
    monkeypatch.setattr(E, "evaluate_unit", v4_evaluator(seen, rows, err))
    assert E.stage_test(key) == 3 and seen == ["DEV"]                                  # TEST is never read
    out = C.BUNDLE / "results" / "TEST_REPEAT" / key
    blk = C.read(out / "BLOCKED.json")
    assert blk["status"] == "BLOCKED" and blk["test_read"] is False and not (out / "TEST_REPEAT.jsonl").exists()
    with pytest.raises(RuntimeError, match="BLOCKED"):
        E.stage_test(key)


def test_unit_job_transient_error_is_not_blocked(monkeypatch, recs, recs4, tmp_path):
    key = "MF_VCS_M0__1001"
    u, man = v4_setup(monkeypatch, recs, recs4, tmp_path, key)
    monkeypatch.setattr(E, "evaluate_unit", v4_evaluator([], None, RuntimeError("CUDA error: an illegal memory access")))
    with pytest.raises(RuntimeError, match="CUDA error"):
        E.stage_test(key)
    assert not (C.BUNDLE / "results" / "TEST_REPEAT" / key / "BLOCKED.json").exists()   # resumable, not blocked


def test_prepare_main_has_no_gate_precondition(monkeypatch, recs, recs4, tmp_path):
    import prepare_inputs as PI
    monkeypatch.setattr(C, "BUNDLE", tmp_path / "bundle")
    monkeypatch.setenv("SLURM_JOB_ID", "999")
    monkeypatch.setenv("V10_TR_EXPECT_ACTIVATION", "a" * 64)
    monkeypatch.setenv("V10_TR_EXPECT_COMMIT", "c" * 40)
    rec = {"manifest": recs["manifest"], "protocol": recs4["protocol"], "activation": {}, "activation_sha256": "a" * 64,
           "shas": {C.MANIFEST: "m" * 64}}
    monkeypatch.setattr(C, "records", lambda pin=None: rec)
    monkeypatch.setattr(C, "verify_code", lambda a, c: {"commit": c})

    class Reached(Exception):
        pass
    monkeypatch.setattr(PI, "host_modules", lambda *a: (_ for _ in ()).throw(Reached()))
    with pytest.raises(Reached):                                                      # no gate is consulted first
        PI.main()


# ----------------------------------------------------------------------------------------------- v2 gate amendment
def gate_rows(n_cores=32, seed=3):
    rng = np.random.default_rng(seed)
    return [{"core": f"C{c:02d}", "perturbation": i, "role": "DEV", "checkpoint_sha256": "k", "theta": 1.0, "t": [0.5, 0.5],
             "n_grid": 10668, "err_full": float(rng.gamma(2.0, 4.0)), "F_hat": 0.3, "F_true": 0.31, "theta_hat": 1.01,
             "t_hat": [0.4, 0.6], "seconds": 1.0} for c in range(n_cores) for i in range(10)]


def verdict(gpu, rec_rows, new_rows, ident_new=None):
    cmp = C.compare_rows(rec_rows, new_rows)
    diag = C.gate_diagnostics(rec_rows, new_rows, {"unit": 1}, ident_new or {"unit": 1})
    return C.gate_verdict(gpu, cmp["status"] == "PASS", C.metrics(rec_rows) == C.metrics(new_rows), True, diag), diag


def test_gate_verdict_tolerance_rules():
    base = gate_rows()
    for gpu in ("L40S", "RTX6000PRO", "H100"):
        assert verdict(gpu, base, [dict(r) for r in base])[0]["status"] == "PASS"
    small = [dict(r, err_full=r["err_full"] + 1e-6, F_hat=r["F_hat"] * (1 + 1e-7)) for r in base]
    v, d = verdict("RTX6000PRO", base, small)
    assert v["status"] == "PASS_TOLERANCE" and all(v["criteria"].values()) and d["bitwise_rows"] == 0
    assert set(d["abs_delta_err_full"]) >= {"q50", "q90", "q95", "q99", "max"} and d["flips"] == {"success42": 0, "success9": 0}
    assert verdict("L40S", base, small)[0]["status"] == "FAIL"           # L40S stays bitwise
    assert verdict("H100", base, small)[0]["status"] == "PASS_TOLERANCE"
    # large single-row moves are benign under (d) when the typical row is unchanged (the calibration pattern)
    rows = [dict(r) for r in base]
    for r in rows[:40]:
        r["err_full"] += 2.0 if r["err_full"] < 30 else 0.0
    v, d = verdict("RTX6000PRO", base, rows)
    assert d["abs_delta_err_full"]["q50"] == 0.0 and abs(d["abs_delta_err_full"]["max"] - 2.0) < 1e-9
    assert v["criteria"]["d"] and v["criteria"]["a"]
    # (b): net success counts may move by at most 3 of 320
    for n_cross, ok in ((3, True), (4, False)):
        rows = [dict(r) for r in base]
        idx = [k for k, r in enumerate(rows) if 8.0 < r["err_full"] < 9.0][:n_cross]
        if len(idx) < n_cross:
            idx = [k for k, r in enumerate(rows) if r["err_full"] < 9.0][:n_cross]
        for k in idx:
            rows[k]["err_full"] = 9.0 + 1e-9
        v, d = verdict("RTX6000PRO", base, rows)
        assert d["flips"]["success9"] == n_cross and v["criteria"]["b"] is ok
    # (c): DEV32 median moved by more than 0.25 -> FAIL; (d): typical row moved by more than 0.25 -> FAIL
    v, _ = verdict("RTX6000PRO", base, [dict(r, err_full=r["err_full"] + 0.3) for r in base])
    assert v["status"] == "FAIL" and v["criteria"]["c"] is False and v["criteria"]["d"] is False
    # (a): a perturbation truth differs, the evaluation identity differs, or a value is not finite -> FAIL
    rows = [dict(r) for r in small]
    rows[0]["theta"] = 1.0000001
    assert verdict("RTX6000PRO", base, rows)[0]["criteria"]["a"] is False
    rows = [dict(r) for r in small]
    rows[0]["t"] = [0.5, 0.5000001]
    assert verdict("RTX6000PRO", base, rows)[0]["criteria"]["a"] is False
    assert verdict("RTX6000PRO", base, small, ident_new={"unit": 2})[0]["status"] == "FAIL"
    rows = [dict(r) for r in small]
    rows[0]["F_hat"] = float("nan")
    assert verdict("RTX6000PRO", base, rows)[0]["criteria"]["a"] is False
    assert C.gate_verdict("RTX6000PRO", False, False, False, {}, rebuild_error="resolved configuration differs")["status"] == "FAIL"


def test_in_job_check_rules():
    base = gate_rows()
    ok = C.in_job_check(C.gate_diagnostics(base, [dict(r) for r in base], {"u": 1}, {"u": 1}), True)
    assert ok["status"] == "PASS"
    noisy = [dict(r, err_full=r["err_full"] + (2.0 if k % 3 == 0 else 0.0), F_hat=r["F_hat"] + 1e-3) for k, r in enumerate(base)]
    assert C.in_job_check(C.gate_diagnostics(base, noisy, {"u": 1}, {"u": 1}), True)["status"] in ("PASS", "BLOCKED")
    small = [dict(r, err_full=r["err_full"] + 0.01) for r in base]
    assert C.in_job_check(C.gate_diagnostics(base, small, {"u": 1}, {"u": 1}), True)["status"] == "PASS"
    ftrue = [dict(r) for r in small]
    ftrue[0]["F_true"] = 0.3100001
    v = C.in_job_check(C.gate_diagnostics(base, ftrue, {"u": 1}, {"u": 1}), True)
    assert v["status"] == "BLOCKED" and v["criteria"]["e"] is False
    big = [dict(r, err_full=r["err_full"] + 0.6) for r in base]                       # median moved by 0.6 > 0.5
    v = C.in_job_check(C.gate_diagnostics(base, big, {"u": 1}, {"u": 1}), True)
    assert v["status"] == "BLOCKED" and v["criteria"]["f"] is False
    rows = [dict(r) for r in base]
    for r in [x for x in rows if x["err_full"] < 9.0][:13]:
        r["err_full"] = 9.5                                                            # 13 crossings of 9 > 12
    v = C.in_job_check(C.gate_diagnostics(base, rows, {"u": 1}, {"u": 1}), True)
    assert v["criteria"]["f"] is False
    rows = [dict(r) for r in base]
    for r in [x for x in rows if x["err_full"] < 9.0][:12]:
        r["err_full"] = 9.5                                                            # 12 crossings are inside the band
    v = C.in_job_check(C.gate_diagnostics(base, rows, {"u": 1}, {"u": 1}), True)
    assert v["criteria"]["f"] is (abs(C.metrics(rows)["median_err"] - C.metrics(base)["median_err"]) <= 0.5)
    assert C.in_job_check(C.gate_diagnostics(base, small, {"u": 1}, {"u": 2}), True)["criteria"]["a"] is False
    assert C.in_job_check({}, False, rebuild_error="configuration differs")["status"] == "BLOCKED"


def test_v4_records_and_delta(recs4, recs):
    texts = recs4["texts"]
    delta = json.loads(texts[C.DELTA])
    for rel, names in B.EVALUATOR_DEFS.items():
        assert set(names) <= set(delta["code"][rel]["identical"])
    p4 = json.loads(texts[C.PROTOCOL])
    plan = {t["unit"]: t for t in p4["jobs"]["test_units"]}
    assert len(plan) == 42
    assert all(plan[u["key"]]["gpu_type"] == u["hardware"]["gpu_type"] for u in recs["manifest"]["units"])
    assert Counter(t["gpu_type"] for t in plan.values()) == {"L40S": 28, "RTX6000PRO": 12, "H100": 2}
    assert p4["in_job_check"]["thresholds"] == {"count_abs": 12, "median_abs": 0.5}
    assert p4["perturbations"] == recs4["v2_protocol"]["perturbations"] and p4["statistics"]["paired_contrasts"] == \
        recs4["v2_protocol"]["statistics"]["paired_contrasts"]
    dec = json.loads(texts[C.DECISION_V4])
    assert dec["chosen_option_verbatim"] == "同作业配对，评一次 (Recommended)" and len(dec["options_not_chosen"]) == 2
    am = json.loads(texts[C.AMENDMENT])
    assert len(am["observed_dev_reevaluations"]) == 9 and all(x["F_true_bitwise"] for x in am["observed_dev_reevaluations"])
    assert max(abs(x["d_lt9"]) for x in am["observed_dev_reevaluations"]) <= 12
    act = json.loads(texts[C.ACTIVATION])
    assert act["status"] == "ACTIVE_FOR_FROZEN_CHECKPOINT_EVALUATION_ONLY" and "v4" in act["owner_decisions"]
