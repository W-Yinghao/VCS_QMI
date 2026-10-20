"""EPFL finals hardware release (owner 2026-10-19: 「final 放开，CAL 留 L40S」): GPU acceptance by phase, science identity vs 7ceb2f1e, the
cross-hardware gate precondition, the release checks, the final identity / reuse rule, selection pins, unit states and per-unit job lines.
CPU; no image is read."""
import copy
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from v10.epfl import hardware as HW, production as PR

REPO = PR.REPO
D = REPO / "v10" / "epfl"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"epfl_{name}", REPO / "scripts" / "v10_epfl" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def test_gpu_acceptance_by_phase():
    for g in HW.FINAL_GPUS:
        assert HW.gpu_accepted("final", g)
    assert HW.gpu_accepted("cal", "NVIDIA L40S")
    for g in ("NVIDIA RTX PRO 6000 Blackwell Server Edition", "NVIDIA H100 NVL", "NVIDIA A100", "NVIDIA RTX PRO 6000", "NVIDIA H100"):
        assert not HW.gpu_accepted("cal", g)
    for g in ("NVIDIA A100", "NVIDIA RTX PRO 6000", "NVIDIA H100", "L40S"):
        assert not HW.gpu_accepted("final", g)                                   # exact CUDA names only
    assert HW.FINAL_GPUS == ("NVIDIA L40S", "NVIDIA RTX PRO 6000 Blackwell Server Edition", "NVIDIA H100 NVL")
    from v10.epfl import runtime as RT
    assert set(HW.FINAL_GPUS) <= set(RT.ACCEPTED_GPUS)                           # the unchanged environment check accepts them


def test_science_code_byte_identical_to_the_parent_source():
    sci = HW.science_identity()
    assert len(sci) > 50 and all(here == parent for here, parent in sci.values())
    assert "src/v10/epfl/host.py" in sci and "src/v10/epfl/production.py" in sci and "src/v10/adapter.py" in sci
    assert "src/v10/epfl/hardware.py" not in sci and "scripts/v10_epfl/run.py" not in sci
    review = json.loads(HW.DELTA_REVIEW_HW.read_text())
    assert review["science_identity"] == {f: v[0] for f, v in sci.items()} and review["numerics_identical_to_item5_gate_commit"] is True
    assert set(review["changed_files"]) <= set(HW.NON_SCIENCE_CHANGES)


def test_amendment_record_verbatim():
    a = json.loads(HW.AMENDMENT.read_text())
    assert a["decided_by"]["choice_verbatim"] == "final 放开，CAL 留 L40S"
    assert a["decided_by"]["offer_verbatim"].startswith("EPFL final、BF 和 EPFL 的 RuLSIF v2 final")
    assert "--partition=RTX6000PRO,L40S,H100" in a["reasons"]["standing_instruction"]
    assert a["rule"]["finals"]["gpus"] == list(HW.FINAL_GPUS) and a["rule"]["cal"]["gpus"] == ["NVIDIA L40S"]
    assert "no claim of cross-hardware equivalence" in a["reporting"]


def _rows(ok=True, criterion=True):
    def cmp(allclose=True):
        return {"allclose": allclose, "finite": True, "atol": 1e-5, "rtol": 1e-4, "worst_err_over_bound": 0.01 if allclose else 3.0,
                "max_abs_err": 1e-12, "max_rel_err": 1e-7, "bitwise": False}
    crit = HW.XHW_CRITERION_ID + ": T01 round-trip tolerance" if criterion else ""
    return {"rows": [{"name": HW.CHECKPOINT_ROWS[0], "status": "PASS"}] +
                    [{"name": n, "status": "PASS" if ok else "FAIL", "criterion": crit, "energy": cmp(), "gradient": cmp(ok)} for n in HW.CHECKPOINT_ROWS[1:]]}


def _gate(tmp_path, label, rows=None, criterion=True, **over):
    g = {"schema": "v10.epfl.item5_gate.v1", "status": "PASS", "pins": {"commit": "c0ffee", "record_sha256": PR.GATE_RECORD_SHA256, "job": "77"},
         "environment": {"gpu_name": {"RTX6000PRO": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "H100": "NVIDIA H100 NVL"}[label]},
         "parts": {k: {"status": "PASS"} for k in range(8)}, "data_reads_roles": ["CAL", "FIT"]}
    if criterion:
        g["xhw_criterion"] = {"id": HW.XHW_CRITERION_ID}
    for k, v in over.items():
        g[k] = v
    d = tmp_path / "xhw" / label / "gpu_gate" / g["pins"]["job"]; d.mkdir(parents=True, exist_ok=True)
    (d / "GATE.json").write_text(json.dumps(g)); (d / "checkpoint.json").write_text(json.dumps(rows if rows is not None else _rows()))
    return d / "GATE.json"


def test_cross_hardware_gate_precondition(tmp_path):
    rel = {"xhw_gates": {"dir": str(tmp_path / "xhw")}}
    assert HW.xhw_gate(rel, "NVIDIA L40S", "c0ffee") is None                      # L40S: the unchanged item-5 gate 1048825
    with pytest.raises(RuntimeError):
        HW.xhw_gate(rel, "NVIDIA H100 NVL", "c0ffee")                             # no gate yet: refused
    _gate(tmp_path, "H100", status="FAIL")
    _gate(tmp_path, "H100", pins={"commit": "other", "record_sha256": PR.GATE_RECORD_SHA256, "job": "78"})
    _gate(tmp_path, "H100", criterion=False, pins={"commit": "c0ffee", "record_sha256": PR.GATE_RECORD_SHA256, "job": "80"})   # 3ced3da0-style
    _gate(tmp_path, "H100", rows=_rows(criterion=False), pins={"commit": "c0ffee", "record_sha256": PR.GATE_RECORD_SHA256, "job": "81"})
    _gate(tmp_path, "H100", rows=_rows(ok=False), pins={"commit": "c0ffee", "record_sha256": PR.GATE_RECORD_SHA256, "job": "82"})
    bad = _rows(); bad["rows"][0]["status"] = "FAIL"
    _gate(tmp_path, "H100", rows=bad, pins={"commit": "c0ffee", "record_sha256": PR.GATE_RECORD_SHA256, "job": "83"})   # state hash must stay bitwise
    with pytest.raises(RuntimeError):
        HW.xhw_gate(rel, "NVIDIA H100 NVL", "c0ffee")
    p = _gate(tmp_path, "H100", pins={"commit": "c0ffee", "record_sha256": PR.GATE_RECORD_SHA256, "job": "79"})
    assert HW.xhw_gate(rel, "NVIDIA H100 NVL", "c0ffee")["path"] == str(p)
    with pytest.raises(RuntimeError):
        HW.xhw_gate(rel, "NVIDIA RTX PRO 6000 Blackwell Server Edition", "c0ffee")   # per GPU type


def test_amended_checkpoint_rows_on_cpu():
    """The amended checkpoint part runs the unchanged procedure; on CPU the round trip is exact, so the T01-tolerance rows pass and record
    their measured differences (zero) and bitwise flags (True)."""
    import tempfile
    from v10.epfl import guard as GD
    xg = load_script("gpu_gate_xhw")
    g = GD.EPFLDataGuard(record=GD.ACTIVATION_RECORD, expect_record_sha256=sha(GD.ACTIVATION_RECORD))
    with tempfile.TemporaryDirectory() as tmp:
        rows = xg.checkpoint_rows_xhw(g, g.included("FIT")[0], "cpu", tmp)
    assert [r["name"] for r in rows] == list(HW.CHECKPOINT_ROWS) and all(r["status"] == "PASS" for r in rows)
    for r in rows[1:]:
        assert r["criterion"].startswith(HW.XHW_CRITERION_ID) and r["energy"]["atol"] == 1e-5 and r["gradient"]["rtol"] == 1e-4
        assert r["energy"]["bitwise"] is True and r["gradient"]["max_abs_err"] == 0.0
    assert rows[0]["criterion"] == "bitwise (unchanged)"
    d = Path(tempfile.mkdtemp()); (d / "checkpoint.json").write_text(json.dumps({"rows": rows}))
    assert HW.checkpoint_rows_ok(d / "checkpoint.json")


def fixture_release(tmp_path):
    par = json.loads(Path(PR.PROJECT / "handoff" / "v10_epfl_production_20261019" / "RELEASE.json").read_text())
    inv = PR.source_inventory()
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    rel = {"schema": HW.RELEASE_SCHEMA_HW, "status": HW.RELEASE_STATUS_HW, "source_commit": head, "source_files": inv, "code_sha256": PR.digest(inv),
           "numerics_identity": PR.numerics_identity(), "science_identity": HW.science_identity(),
           "parent_release": {"path": str(PR.PROJECT / "handoff" / "v10_epfl_production_20261019" / "RELEASE.json"), "sha256": HW.PARENT_RELEASE_SHA256},
           "cal_release": {"sha256": HW.PARENT_RELEASE_SHA256}, "results_root": par["results_dir"],
           **{k: par[k] for k in ("frozen_protocol", "activation_record", "group_record", "data_lock", "item5_gate", "item5_timing", "delta_review")},
           "hardware_amendment": {"path": str(HW.AMENDMENT.relative_to(REPO)), "sha256": sha(HW.AMENDMENT)},
           "delta_review_hw": {"path": str(HW.DELTA_REVIEW_HW.relative_to(REPO)), "sha256": sha(HW.DELTA_REVIEW_HW)},
           "xhw_gate_amendment": {"path": str(HW.XHW_AMENDMENT.relative_to(REPO)), "sha256": sha(HW.XHW_AMENDMENT)},
           "phases": ["FINAL_DEV"], "final_gpus": list(HW.FINAL_GPUS), "cal_gpu": HW.CAL_GPU,
           "xhw_gates": {"dir": str(tmp_path / "xhw"), "criterion": HW.XHW_CRITERION_ID},
           "families": copy.deepcopy(par["families"]), "selection_pins": {}}
    p = tmp_path / "RELEASE.json"; p.write_text(json.dumps(rel)); return p, rel


def test_hardware_release_accepts_and_refuses(tmp_path):
    from v10.epfl import host as H
    p, rel = fixture_release(tmp_path)
    kw = {"candidates_fn": H.production_candidates, "schedule_fn": H.production_schedule}
    assert HW.verify_release_hw(p, sha(p), "MF_VCS_perm1", **kw)["schema"] == HW.RELEASE_SCHEMA_HW
    with pytest.raises(RuntimeError, match="finals only"):
        HW.verify_release_hw(p, sha(p), "MF_VCS_perm1", phase="cal", **kw)      # CAL never runs under the hardware release
    with pytest.raises(RuntimeError):
        HW.verify_release_hw(p, sha(p), "RuLSIF_exact_v2", **kw)                 # BLOCKED in the parent, not READY here
    for mutate in (lambda r: r.update(final_gpus=["NVIDIA L40S", "NVIDIA A100"]), lambda r: r.update(cal_gpu="NVIDIA H100 NVL"),
                   lambda r: r.update(results_root="/tmp/elsewhere"), lambda r: r["parent_release"].update(sha256="0" * 64),
                   lambda r: r.update(phases=["CAL", "FINAL_DEV"]), lambda r: r["families"]["MF_VCS_perm1"].update(candidates=[]),
                   lambda r: r["science_identity"].update({"src/v10/adapter.py": ["x", "x"]}), lambda r: r.update(source_commit="deadbeef"),
                   lambda r: r["xhw_gates"].update(criterion="bitwise"), lambda r: r["xhw_gate_amendment"].update(sha256="0" * 64)):
        r = copy.deepcopy(rel); mutate(r); p.write_text(json.dumps(r))
        with pytest.raises(RuntimeError):
            HW.verify_release_hw(p, sha(p), "MF_VCS_perm1", **kw)


class _Ctx:
    def __init__(self, hardware):
        self.hardware, self.schedule_fn = hardware, (lambda f: {"steps": 1})


def test_final_identity_carries_the_gpu_and_reuse_ignores_it(tmp_path, monkeypatch):
    run = load_script("run")
    calls = {}
    monkeypatch.setattr(run, "checked_selection", lambda root, fam, rsha, d, g, ctx, require_exit=True: (calls.setdefault("cal_sha", rsha) and
                                                                                                          {"selected": {"candidate": {"order": 0}}}, "SELSHA"))
    monkeypatch.setattr(run, "run_unit", lambda ctx, unit, ident, alloc, env: ident)
    l40s = {"gpu_name": "NVIDIA L40S", "node": "node50"}; h100 = {"gpu_name": "NVIDIA H100 NVL", "node": "nodesumo01"}
    ident = run.run_final(_Ctx(h100), tmp_path, "MF_VCS_perm1", 1001, "NEWREL", "d", "g", None, {}, cal_release_sha="PARENT")
    assert calls["cal_sha"] == "PARENT" and ident["release_sha256"] == "NEWREL" and ident["hardware"] == {"gpu_name": "NVIDIA H100 NVL"}
    assert ident["selection_dependency_sha256"] == "SELSHA"
    unit = tmp_path / "FINAL_DEV" / "MF_VCS_perm1" / "seed_1001"; unit.mkdir(parents=True)
    (unit / "COMPLETE.json").write_text(json.dumps({"status": "COMPLETE", "identity": ident}))
    again = run.run_final(_Ctx(l40s), tmp_path, "MF_VCS_perm1", 1001, "NEWREL", "d", "g", None, {}, cal_release_sha="PARENT")
    assert again == ident                                                           # COMPLETE on H100 is reused from an L40S job
    with pytest.raises(RuntimeError, match="pinned"):
        run.run_final(_Ctx(l40s), tmp_path, "MF_VCS_perm1", 1002, "NEWREL", "d", "g", None, {}, cal_release_sha="PARENT", selection_pin="OTHER")
    plain = run.run_final(_Ctx(None), tmp_path, "MF_VCS_perm1", 1003, "OLDREL", "d", "g", None, {})
    assert "hardware" not in plain and plain["release_sha256"] == "OLDREL"          # the parent release path is unchanged


def test_resume_on_another_gpu_type_is_refused(tmp_path):
    from v10.supplement.records import require_identity
    a = PR.identity("NEWREL", "MF_VCS_perm1", {"order": 0}, 1001, "FINAL_DEV", "d", "g", {"steps": 1}, "SEL")
    require_identity(tmp_path, {**a, "hardware": {"gpu_name": "NVIDIA L40S"}})
    with pytest.raises(RuntimeError):
        require_identity(tmp_path, {**a, "hardware": {"gpu_name": "NVIDIA H100 NVL"}})


def test_unit_states_and_per_unit_job_lines(tmp_path, monkeypatch):
    rh = load_script("release_hw")
    root = tmp_path / "results"
    fam = {"MF_VCS_perm1": {"status": "READY", "schedule": {"final_seeds": [1001, 1002, 1003]}, "time_limits": {"final": "02:13:00"}},
           "MF_VCS_M1": {"status": "READY", "schedule": {"final_seeds": [1001]}, "time_limits": {"final": "04:56:00"}},
           "RuLSIF_exact_v2": {"status": "BLOCKED"}}
    (root / "CAL" / "MF_VCS_perm1").mkdir(parents=True); (root / "CAL" / "MF_VCS_perm1" / "SELECTION.json").write_text("{}")
    (root / "FINAL_DEV" / "MF_VCS_perm1" / "seed_1001").mkdir(parents=True)
    (root / "FINAL_DEV" / "MF_VCS_perm1" / "seed_1001" / "COMPLETE.json").write_text("{}")
    (root / "FINAL_DEV" / "MF_VCS_perm1" / "seed_1002").mkdir(parents=True)                      # started (running / interrupted)
    st = {(u["family"], u["seed"]): (u["state"], u["selection"]) for u in HW.unit_states(root, fam)}
    assert st == {("MF_VCS_perm1", 1001): ("COMPLETE", True), ("MF_VCS_perm1", 1002): ("STARTED", True), ("MF_VCS_perm1", 1003): ("NOT_STARTED", True),
                  ("MF_VCS_M1", 1001): ("NOT_STARTED", False)}
    rel = {"source_root": "/src", "source_commit": "c", "results_root": str(root), "families": fam,
           "parent_release": {"path": str(tmp_path / "parent" / "RELEASE.json")}}
    (tmp_path / "b").mkdir(); rp = tmp_path / "b" / "RELEASE.json"; rp.write_text(json.dumps(rel))
    monkeypatch.setattr(rh, "pending_old_jobs", lambda: {"v10e_fin_mfp1_1003": [("1049999", "PENDING")]})
    rh.write_jobs(rp)
    plan = json.loads((tmp_path / "b" / "JOBS_PLAN.json").read_text())
    assert [(r["family"], r["seed"]) for r in plan["submit_now"]] == [("MF_VCS_perm1", 1003)]
    assert [(r["family"], r["seed"]) for r in plan["after_cal"]] == [("MF_VCS_M1", 1001)]
    assert {(r["family"], r["seed"]) for r in plan["not_listed"]} == {("MF_VCS_perm1", 1001), ("MF_VCS_perm1", 1002)}
    c = plan["submit_now"][0]["command"]
    assert "--partition=RTX6000PRO,L40S,H100" in c and "--exclude=node51,node53,node60" in c and "--time=02:13:00" in c and "prod_final_hw.sbatch" in c
    assert plan["submit_now"][0]["old_jobs_same_unit"] == [{"job": "1049999", "state": "PENDING"}]


def test_xhw_amendment_record_verbatim():
    a = json.loads(HW.XHW_AMENDMENT.read_text())
    assert a["owner"]["words_verbatim"] == "什么快什么来吧" and len(a["owner"]["options_verbatim"]) == 2
    assert a["cites"]["release"]["sha256"].startswith("7cbd1c21") and a["cites"]["hardware_amendment"]["sha256"] == sha(HW.AMENDMENT)
    assert a["rule"]["t01_definition"]["constants"]["float32"] == {"atol": 1e-5, "rtol": 1e-4}
    assert "POST-FAILURE CRITERION CHANGE" in a["label"] and a["failed_gate"]["job"] == "1050341"
