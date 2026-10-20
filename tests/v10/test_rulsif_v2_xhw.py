"""Cross-hardware RuLSIF_exact_v2 finals: GPU models per phase, delta allowlist, CAL-release seeds refused,
one GPU model per final, gate discovery and release verification. CPU only; no data, no Slurm."""
import copy
import json
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/v10_supplement"))
import rulsif_v2_xhw as X  # noqa: E402
import run as R  # noqa: E402
from v10.supplement import contract as C  # noqa: E402

L40S, RTX, H100 = X.FINAL_GPU_NAMES


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {"path": str(path), "sha256": C.sha(path)}


class FakeAdapter:
    def __init__(self, gpu, expected, **changes):
        self.value = {"gpu_name": gpu, **expected["environment"],
                      "precision": {"settings": expected["settings"], "env": expected["env"]}, **changes}

    def environment(self, device):
        assert device == "cuda"
        return copy.deepcopy(self.value)

    @staticmethod
    def precision_settings(precision):
        return precision["settings"]


def test_runtime_environment_cal_stays_l40s_finals_accept_three_models(tmp_path, monkeypatch):
    expected = {"environment": {"torch": "2.9.0+cu128", "cuda_runtime": "12.8", "cudnn": 91002, "torch_threads": 4},
                "settings": {"float32_matmul_precision": "highest", "cudnn_conv": "tf32"},
                "env": {"TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": None, "NVIDIA_TF32_OVERRIDE": None}}
    dump(tmp_path / "PROTOCOL.json", {"precision": expected})
    monkeypatch.setattr(R, "PARENT", tmp_path)
    for gpu in X.FINAL_GPU_NAMES:
        monkeypatch.setattr(R, "A", FakeAdapter(gpu, expected), raising=False)
        assert R.runtime_environment(X.FINAL_GPU_NAMES)["gpu_name"] == gpu
        if gpu == L40S:
            assert R.runtime_environment()["gpu_name"] == R.runtime_environment(X.CAL_GPU_NAMES)["gpu_name"] == L40S
        else:
            with pytest.raises(RuntimeError, match="defined on L40S"):
                R.runtime_environment(X.CAL_GPU_NAMES)
    bad = [FakeAdapter("NVIDIA A100-SXM4-80GB", expected), FakeAdapter(RTX, expected, torch="2.8.0+cu128"),
           FakeAdapter(H100, expected, cudnn=90100), FakeAdapter(H100, expected, torch_threads=8),
           FakeAdapter(RTX, expected, precision={"settings": {"float32_matmul_precision": "high"}, "env": expected["env"]}),
           FakeAdapter(RTX, expected, precision={"settings": expected["settings"], "env": {"NVIDIA_TF32_OVERRIDE": "1"}})]
    for adapter in bad:                    # every other environment and precision check still applies
        monkeypatch.setattr(R, "A", adapter, raising=False)
        with pytest.raises(RuntimeError):
            R.runtime_environment(X.FINAL_GPU_NAMES)


def test_delta_allows_only_the_hardware_plumbing():
    base = {"src/v10/supplement/rulsif_v2.py": "1", "scripts/v10_supplement/run.py": "2",
            "scripts/v10_supplement/validate.py": "3", "tests/v10/test_rulsif_v2.py": "4",
            "v10/supplement/RULSIF_V2_PREREGISTRATION.json": "5", "slurm/v10_supplement/run.sbatch": "6"}
    assert X.delta(base, dict(base))["allowlisted_only"] and X.delta(base, dict(base))["science_identical"]
    amended = {**base, "scripts/v10_supplement/run.py": "9", "scripts/v10_supplement/validate.py": "8",
               "scripts/v10_supplement/rulsif_v2_xhw.py": "7", "scripts/v10_supplement/rulsif_v2_xhw_release.py": "6",
               "tests/v10/test_rulsif_v2_xhw.py": "5"}
    change = X.delta(base, amended)
    assert change["allowlisted_only"] and change["science_identical"] and change["changed"] == list(X.CHANGED)
    assert change["added"] == sorted(X.ADDED) and change["unchanged"] == 4
    for bad in ({**amended, "src/v10/supplement/rulsif_v2.py": "x"}, {**amended, "v10/supplement/RULSIF_V2_PREREGISTRATION.json": "x"},
                {**amended, "tests/v10/test_rulsif_v2.py": "x"}, {**amended, "slurm/v10_supplement/run.sbatch": "x"},
                {**amended, "scripts/v10_supplement/other.py": "x"},
                {k: v for k, v in amended.items() if k != "src/v10/supplement/rulsif_v2.py"}):
        change = X.delta(base, bad)
        assert not (change["allowlisted_only"] and change["science_identical"]), bad


def test_final_dir_refuses_cal_release_seed_and_keeps_one_gpu_model(tmp_path):
    env = {"gpu_name": RTX, "gpu_capability": "12.0", "host": "node58", "slurm_nodelist": "node58",
           "slurm_partition": "RTX6000PRO", "slurm_job_id": "101"}
    gate, ident = {"gate_job": "77"}, {"release_sha256": "xhw", "seed": 1001}
    old = tmp_path / "old"
    dump(old / "IDENTITY.json", {"release_sha256": X.BASE_RELEASE_SHA256, "seed": 1001})
    with pytest.raises(RuntimeError, match="stays valid"):
        X.prepare_final_dir(old, ident, env, gate)
    assert not (old / "HARDWARE.json").exists()
    foreign = tmp_path / "foreign"
    dump(foreign / "IDENTITY.json", {"release_sha256": "other", "seed": 1001})
    with pytest.raises(RuntimeError, match="another unit"):
        X.prepare_final_dir(foreign, ident, env, gate)
    fresh = tmp_path / "fresh"
    hardware = X.prepare_final_dir(fresh, ident, env, gate)
    assert hardware["gpu_name"] == RTX and hardware["node"] == "node58" and hardware["partition"] == "RTX6000PRO"
    assert C.read(fresh / "HARDWARE.json")["first_job"] == "101"
    dump(fresh / "IDENTITY.json", ident)
    X.prepare_final_dir(fresh, ident, {**env, "slurm_job_id": "102", "host": "node59"}, gate)   # same model resumes
    with pytest.raises(RuntimeError, match="one GPU model"):
        X.prepare_final_dir(fresh, ident, {**env, "gpu_name": H100}, gate)


def make_gate(root, job, gpu, code, manifest_sha, exit_code=0, status="PASS"):
    dump(root / "exits" / f"{job}.json", {"job": job, "exit_code": exit_code, "validation_manifest_sha256": manifest_sha})
    report = dump(root / "runs" / job / X.FAMILY / "GATE.json",
                  {"status": status, "stage": "T01_AND_EXECUTION_GATE", "family": X.FAMILY, "code_sha256": code,
                   "environment": {"gpu_name": gpu}, "job": job})
    dump(root / "runs" / job / "GATE.json", {"status": status, "requested_families": [X.FAMILY], "code_sha256": code,
                                              "job": job, "families": [{"family": X.FAMILY, **report}]})


def test_gate_is_required_per_gpu_model(tmp_path):
    root = tmp_path / "validation"
    manifest = dump(root / "MANIFEST.json", {"accepted_gpu_names": list(X.FINAL_GPU_NAMES)})
    l40s = dump(tmp_path / "l40s.json", {"status": "PASS", "environment": {"gpu_name": L40S},
                                          "code_sha256": X.BASE_CODE_SHA256, "job": "1049287"})
    release = {"final_gpu_names": list(X.FINAL_GPU_NAMES), "code_sha256": "code", "xhw_validation_manifest": manifest,
               "l40s_gate": l40s}
    assert X.gate_for(L40S, release)["gate_job"] == "1049287"
    for gpu in (RTX, H100):
        with pytest.raises(RuntimeError, match="No PASS"):
            X.gate_for(gpu, release)
    make_gate(root, "11", RTX, "code", manifest["sha256"], exit_code=1)
    make_gate(root, "12", RTX, "other code", manifest["sha256"])
    make_gate(root, "13", H100, "code", "0" * 64)
    make_gate(root, "15", H100, "code", manifest["sha256"], status="FAIL")
    for gpu in (RTX, H100):
        with pytest.raises(RuntimeError, match="No PASS"):
            X.gate_for(gpu, release)
    make_gate(root, "14", RTX, "code", manifest["sha256"])
    found = X.gate_for(RTX, release)
    assert found["gate_job"] == "14" and found["gate_receipt"]["path"].endswith("14.json")
    with pytest.raises(RuntimeError, match="No PASS"):
        X.gate_for(H100, release)
    with pytest.raises(RuntimeError, match="outside the release"):
        X.gate_for("NVIDIA A100-SXM4-80GB", release)


def test_cross_hardware_release_verification(tmp_path, monkeypatch):
    inventory = C.source_inventory(); code = C.digest(inventory)
    base_files = dict(inventory)
    for path in X.ADDED:
        base_files.pop(path, None)
    base_files["scripts/v10_supplement/run.py"] = "0" * 64
    bundle, project = tmp_path / "bundle", tmp_path / "project"
    monkeypatch.setattr(X, "BUNDLE", bundle); monkeypatch.setattr(X, "PROJECT", project)
    monkeypatch.setattr(X, "BASE_CODE_SHA256", "c" * 64); monkeypatch.setattr(X, "BASE_COMMIT", "b" * 40)
    parent = dump(tmp_path / "parent/PROTOCOL.json", {"precision": {}})
    monkeypatch.setattr(X, "PARENT", tmp_path / "parent"); monkeypatch.setattr(X, "PARENT_PROTOCOL_SHA256", parent["sha256"])
    attachment = {"candidates": C.candidates(X.FAMILY), "schedule": C.schedule(X.FAMILY)}
    lock = dump(tmp_path / "DATA_LOCK.json", {"files": []})
    l40s = dump(tmp_path / "l40s.json", {"status": "PASS", "environment": {"gpu_name": L40S}, "code_sha256": "c" * 64, "job": "1"})
    base = dump(bundle / "RELEASE.json", {"code_sha256": "c" * 64, "source_commit": "b" * 40, "source_files": base_files,
                                          "parent_protocol_sha256": parent["sha256"], "families": {X.FAMILY: attachment},
                                          "preregistration": {"sha256": C.sha(X.RULSIF_V2_RECORD)},
                                          "amendment": {"sha256": C.sha(X.RULSIF_V2_AMENDMENT)},
                                          "acceptance": {"GPU_GATE": l40s}, "data_lock": lock})
    monkeypatch.setattr(X, "BASE_RELEASE_SHA256", base["sha256"])
    selection = dump(bundle / "results/CAL" / X.FAMILY / "SELECTION.json",
                     {"status": "SELECTED", "release_sha256": base["sha256"], "job": X.CAL_JOB})
    monkeypatch.setattr(X, "SELECTION_SHA256", selection["sha256"])
    receipt = dump(bundle / "exits" / f"{X.CAL_JOB}.json", {"job": X.CAL_JOB, "exit_code": 0, "release_sha256": base["sha256"]})
    review = dump(tmp_path / "DELTA_REVIEW.json", {"status": "PASS", "new_code_sha256": code, "base_code_sha256": "c" * 64})
    manifest = dump(tmp_path / "MANIFEST.json", {"accepted_gpu_names": list(X.FINAL_GPU_NAMES), "code_sha256": code,
                                                  "source_root": str(X.REPO.resolve())})
    policy = dump(project / "handoff/POLICY.json", {"policy_id": "p", "validated_supplement_overlap": True, "normal_gpu_limit": 8})
    dump(project / "handoff/ACTIVE_SCHEDULING_POLICY.json", {"policy_id": "p", "policy": policy["path"]})

    def release(**changes):
        value = {"status": X.XHW_RELEASE_STATUS, "family": X.FAMILY, "phases": ["final"], "source_files": inventory,
                 "code_sha256": code, "base_release": {"path": str(bundle / "RELEASE.json"), "sha256": base["sha256"]},
                 "delta_review": review, "parent_protocol_sha256": parent["sha256"], "families": {X.FAMILY: attachment},
                 "preregistration": {"sha256": C.sha(X.RULSIF_V2_RECORD)}, "amendment": {"sha256": C.sha(X.RULSIF_V2_AMENDMENT)},
                 "cal_selection": selection, "cal_exit": receipt, "cal_gpu_names": list(X.CAL_GPU_NAMES),
                 "final_gpu_names": list(X.FINAL_GPU_NAMES), "xhw_validation_manifest": manifest, "l40s_gate": l40s,
                 "data_lock": lock, "scheduling_authorization": policy, "owner_decision": X.OWNER_DECISION,
                 "reporting_rule": X.REPORTING_RULE, **changes}
        return dump(tmp_path / "RELEASE_XHW.json", value)

    good = release()
    assert X.verify_xhw_release(good["path"], good["sha256"])["final_gpu_names"] == list(X.FINAL_GPU_NAMES)
    with pytest.raises(RuntimeError):
        X.verify_xhw_release(good["path"], "0" * 64)
    for changes in (dict(status="FROZEN_PREREGISTERED_GATED"), dict(phases=["cal", "final"]), dict(code_sha256="0" * 64),
                    dict(final_gpu_names=list(X.FINAL_GPU_NAMES) + ["NVIDIA A100-SXM4-80GB"]), dict(cal_gpu_names=[L40S, RTX]),
                    dict(cal_selection={**selection, "sha256": "0" * 64}), dict(owner_decision={"chosen_option_verbatim": "x"}),
                    dict(reporting_rule="pooled only"), dict(l40s_gate={**l40s, "sha256": "0" * 64}),
                    dict(families={X.FAMILY: {**attachment, "candidates": attachment["candidates"][:3]}})):
        bad = release(**changes)
        with pytest.raises(RuntimeError):
            X.verify_xhw_release(bad["path"], bad["sha256"])
    # Science delta beyond the allowlist, failed CAL receipt, other active policy: all refused.
    good = release()
    science = dict(base_files); science["src/v10/supplement/rulsif_v2.py"] = "0" * 64
    for target, value in ((bundle / "RELEASE.json", {**C.read(bundle / "RELEASE.json"), "source_files": science}),
                          (bundle / "exits" / f"{X.CAL_JOB}.json", {**C.read(receipt["path"]), "exit_code": 1}),
                          (project / "handoff/ACTIVE_SCHEDULING_POLICY.json", {"policy_id": "q", "policy": policy["path"]})):
        saved = Path(target).read_text()
        dump(target, value)
        if Path(target) == bundle / "RELEASE.json":
            monkeypatch.setattr(X, "BASE_RELEASE_SHA256", C.sha(target))
        with pytest.raises(RuntimeError):
            X.verify_xhw_release(good["path"], good["sha256"])
        Path(target).write_text(saved)
        monkeypatch.setattr(X, "BASE_RELEASE_SHA256", base["sha256"])
    assert X.verify_xhw_release(good["path"], good["sha256"])["family"] == X.FAMILY
