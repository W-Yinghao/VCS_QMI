"""One-family EPFL RuLSIF_exact_v2 releases: GPU acceptance by phase, the cross-hardware v2 gate licence, science identity vs the gate commit,
release checks (real acceptance records of T1024), final identity with the GPU, time limits.  CPU; no image is read."""
import copy
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from v10.epfl import production as PR, rulsif_v2_release as R2

REPO = PR.REPO


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"epfl_{name}", REPO / "scripts" / "v10_epfl" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def test_gpu_acceptance_by_phase():
    assert R2.gpu_accepted("cal", "NVIDIA L40S") and not R2.gpu_accepted("cal", "NVIDIA H100 NVL")
    assert all(R2.gpu_accepted("final", g) for g in R2.FINAL_GPUS) and not R2.gpu_accepted("final", "NVIDIA H100")


def _xhw(tmp, label, **over):
    g = {"schema": "v10.epfl.rulsif_v2_gpu_gate.v1", "status": "PASS", "cross_hardware": True,
         "pins": {"commit": "c0ffee", "record_sha256": R2.GATE_CHECKS_SHA256, "group_record_sha256": R2.GROUP_SHA256, "job": "9"},
         "environment": {"gpu_name": {"RTX6000PRO": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "H100": "NVIDIA H100 NVL"}[label]},
         "parts": {k: {"status": "PASS"} for k in R2.GATE_PARTS}, "data_reads_roles": ["CAL", "FIT"], "dev_read": False}
    g.update(over)
    d = Path(tmp) / "rulsif_v2_gpu_gate_xhw" / label / g["pins"]["job"]; d.mkdir(parents=True, exist_ok=True)
    (d / "GATE.json").write_text(json.dumps(g)); return d / "GATE.json"


def test_cross_hardware_v2_gate_licence(tmp_path):
    rel = {"source_commit": "c0ffee", "hardware": {"xhw_gates_dir": str(tmp_path)}}
    assert R2.xhw_gate(rel, "NVIDIA L40S") is None
    with pytest.raises(RuntimeError):
        R2.xhw_gate(rel, "NVIDIA H100 NVL")
    _xhw(tmp_path, "H100", cross_hardware=False, pins={"commit": "c0ffee", "record_sha256": R2.GATE_CHECKS_SHA256, "group_record_sha256": R2.GROUP_SHA256, "job": "1"})
    _xhw(tmp_path, "H100", status="FAIL", pins={"commit": "c0ffee", "record_sha256": R2.GATE_CHECKS_SHA256, "group_record_sha256": R2.GROUP_SHA256, "job": "2"})
    _xhw(tmp_path, "H100", pins={"commit": "other", "record_sha256": R2.GATE_CHECKS_SHA256, "group_record_sha256": R2.GROUP_SHA256, "job": "3"})
    with pytest.raises(RuntimeError):
        R2.xhw_gate(rel, "NVIDIA H100 NVL")
    p = _xhw(tmp_path, "H100")
    assert R2.xhw_gate(rel, "NVIDIA H100 NVL")["path"] == str(p)
    with pytest.raises(RuntimeError):
        R2.xhw_gate(rel, "NVIDIA RTX PRO 6000 Blackwell Server Edition")


def test_science_identical_to_the_gate_commit():
    sci = R2.science_identity()
    assert "src/v10/epfl/host.py" in sci and "src/v10/supplement/rulsif_v2.py" in sci and "src/v10/epfl/rulsif_v2_release.py" not in sci


def fixture_release(tmp_path, T=1024):
    from v10.epfl import host as H
    br = load_script("rulsif_v2_build_release")
    acc = br.ACCEPTANCE[T]
    f, t = PR.read(acc["fit_gate"]), PR.read(acc["timing"])
    inv = PR.source_inventory()
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    rel = {"schema": R2.RELEASE_SCHEMA, "status": R2.RELEASE_STATUS, "family": R2.FAMILY, "kernel_dictionary": {"total_centres": T, "role": R2.CENTRES[T]},
           "source_commit": head, "gate_commit": R2.GATE_COMMIT, "source_files": inv, "code_sha256": PR.digest(inv), "science_identity": R2.science_identity(),
           "records": {k: {"path": p, "sha256": sha(REPO / p)} for k, p in br.RECORDS.items()},
           "acceptance": {k: {"path": str(p), "sha256": sha(p)} for k, p in acc.items()},
           "candidates": [c for c in H.production_candidates(R2.FAMILY) if c["order"] in f["eligible_orders"]], "schedule": H.production_schedule(R2.FAMILY),
           "budget": {"cap_seconds": PR.CAP_SECONDS, "margin_seconds": PR.CAP_MARGIN, "max_candidates": PR.MAX_CANDIDATES},
           "hardware": {"cal_gpu": R2.CAL_GPU, "final_gpus": list(R2.FINAL_GPUS), "xhw_gates_dir": str(tmp_path / "xhw")},
           "results_dir": str(tmp_path / "results")}
    p = tmp_path / "RELEASE.json"; p.write_text(json.dumps(rel)); return p, rel, t


def test_release_accepts_and_refuses(tmp_path):
    from v10.epfl import host as H
    p, rel, t = fixture_release(tmp_path)
    kw = {"candidates_fn": H.production_candidates, "schedule_fn": H.production_schedule}
    assert R2.verify_release(p, sha(p), "cal", **kw)["kernel_dictionary"] == {"total_centres": 1024, "role": "primary"}
    assert [c["order"] for c in rel["candidates"]] == list(range(8)) and t["assembled"]["complete_candidates_within_cap"] == 8
    for mutate in (lambda r: r["kernel_dictionary"].update(total_centres=1536), lambda r: r.update(candidates=r["candidates"][:3]),
                   lambda r: r["hardware"].update(final_gpus=["NVIDIA L40S"]), lambda r: r.update(results_dir="/tmp/x"),
                   lambda r: r["acceptance"]["fit_gate"].update(sha256="0" * 64), lambda r: r.update(source_commit="deadbeef"),
                   lambda r: r["records"]["activation_activated"].update(sha256="0" * 64), lambda r: r["kernel_dictionary"].update(role="primary", total_centres=2048)):
        r = copy.deepcopy(rel); mutate(r); p.write_text(json.dumps(r))
        with pytest.raises(RuntimeError):
            R2.verify_release(p, sha(p), "final", **kw)


def test_time_limits_from_the_measured_cost():
    br = load_script("rulsif_v2_build_release")
    assert br.time_limit(1.8776, 8 * 60 + 20) == "04:16:00" and br.time_limit(1.9384, 23 * 60 + 59) == "04:23:00"
    assert br.time_limit(7.9, 8 * 60 + 20) == "08:20:00" and br.time_limit(0.1, 8 * 60 + 20) == "01:00:00"


def test_final_identity_carries_gpu_and_kernel_dictionary(tmp_path, monkeypatch):
    run = load_script("run")
    monkeypatch.setattr(run, "checked_selection", lambda *a, **k: ({"selected": {"candidate": {"order": 0}}}, "SEL"))
    monkeypatch.setattr(run, "run_unit", lambda ctx, unit, ident, alloc, env: ident)
    ctx = run.Context(None, "cuda", {}, schedule_fn=lambda f: {"steps": 1}, centres_total=1024)
    ctx.hardware = {"gpu_name": "NVIDIA H100 NVL"}
    ident = run.run_final(ctx, tmp_path, R2.FAMILY, 1001, "REL", "d", "g", None, {})
    assert ident["hardware"] == {"gpu_name": "NVIDIA H100 NVL"} and ident["kernel_dictionary"] == {"total_centres": 1024, "role": "primary"}
    unit = tmp_path / "FINAL_DEV" / R2.FAMILY / "seed_1001"; unit.mkdir(parents=True)
    (unit / "COMPLETE.json").write_text(json.dumps({"identity": ident}))
    ctx.hardware = {"gpu_name": "NVIDIA L40S"}
    assert run.run_final(ctx, tmp_path, R2.FAMILY, 1001, "REL", "d", "g", None, {}) == ident    # COMPLETE reused on another GPU type


def test_release_rules_record():
    r = json.loads((PR.EPFL_DIR / "EPFL_RULSIF_V2_RELEASE_RULES_20261020.json").read_text())
    assert r["owner_rules_verbatim"] == {"2026-10-19": "final 放开，CAL 留 L40S", "2026-10-20": "什么快什么来吧"}
    assert "no identical row" in r["amended_acceptance"]["applicability"] and r["cal"]["gpu"] == "NVIDIA L40S"
