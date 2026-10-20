"""D3: the amendment differs from the frozen runtime check only in the GPU-name condition.

The frozen pipeline runner is bound in a clean subprocess exactly as d3_gate.py binds it,
so this test never mixes the frozen and the working-tree v10 packages in one interpreter.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

from v10_sensitivity import common as C
from v10_sensitivity import hardware as H

HERE = Path(__file__).resolve().parent
PIPELINE = Path(C.prereg()["frozen_inputs"]["pipeline_validation_manifest"]["source_root"])


def probe():
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "SLURM_JOB_PARTITION")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run([sys.executable, "-I", "-B", str(HERE / "helpers/frozen_runtime_probe.py"), str(PIPELINE),
                             str(C.REPO / "src")], capture_output=True, text=True, env=env, timeout=300)
    assert result.returncode == 0, result.stderr[-3000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_amendment_changes_only_the_gpu_name_condition():
    out = probe()
    assert Path(out["run_module_file"]).resolve().is_relative_to(PIPELINE)
    original, amended = out["original"], out["amended"]
    # the frozen check: L40S passes, every other GPU is refused by name
    assert original["l40s"] == "PASS"
    for name in ("h100", "rtx", "h100_on_rtx_partition"):
        assert original[name] == "REFUSED: Supplemental BF budget is defined on L40S"
    # the amendment: D3 GPUs on their own partition pass, L40S and a partition/GPU mismatch are refused
    assert amended["h100"] == amended["rtx"] == "PASS+amendment"
    assert amended["l40s"].startswith("REFUSED") and amended["h100_on_rtx_partition"].startswith("REFUSED")
    # every non-name check refuses in both, with the frozen messages
    expected = {"h100_torch": "Common runtime environment mismatch: torch",
                "h100_cudnn": "Common runtime environment mismatch: cudnn",
                "h100_threads": "Common runtime environment mismatch: torch_threads",
                "h100_tf32_matmul": "Common precision settings differ",
                "h100_override": "Common precision environment overrides differ"}
    for name, message in expected.items():
        assert amended[name] == "REFUSED: " + message
        assert original[name] == "REFUSED: Supplemental BF budget is defined on L40S"  # name check comes first
        # on L40S the frozen check reaches the same condition and refuses with the same message
        assert original["l40s_" + name.split("_", 1)[1]] == "REFUSED: " + message


def test_pool_matching_rules():
    base = {"torch": "t", "cuda_runtime": "c", "cudnn": 1, "torch_threads": 4, "precision": {"env": {}}}
    expected = {"environment": {"torch": "t", "cuda_runtime": "c", "cudnn": 1, "torch_threads": 4}, "settings": {}, "env": {}}
    H.validate_environment({**base, "slurm_partition": "H100", "gpu_name": "NVIDIA H100 NVL"}, {}, expected)
    H.validate_environment({**base, "slurm_partition": "RTX6000PRO",
                            "gpu_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition"}, {}, expected)
    for partition, gpu in (("L40S", "NVIDIA L40S"), ("H100", "NVIDIA RTX PRO 6000 Blackwell Server Edition"),
                           ("A100", "NVIDIA A100"), ("H100", None)):
        try:
            H.validate_environment({**base, "slurm_partition": partition, "gpu_name": gpu}, {}, expected)
        except RuntimeError:
            continue
        raise AssertionError(f"{partition}/{gpu} was accepted")


def test_d3_gate_binds_the_patched_frozen_runner_and_the_byte_copied_manifest(tmp_path):
    """Dry binding in a subprocess: validate.verify_manifest() on a byte copy, patched run module identity."""
    manifest = Path(C.prereg()["frozen_inputs"]["pipeline_validation_manifest"]["path"])
    copy = tmp_path / "D3/validation/MANIFEST.json"
    copy.parent.mkdir(parents=True)
    copy.write_bytes(manifest.read_bytes())
    code = f"""
import sys; sys.path.append({str(C.REPO / 'src')!r})
from pathlib import Path
from v10_sensitivity import frozen as Z, hardware as H
root = Path({str(PIPELINE)!r})
R = Z.import_runner(root, 'run'); R.runtime_environment = H.amended_runtime_environment(R, scope='probe')
V = Z.import_script(root, 'scripts/v10_supplement/validate.py', 'v10_frozen_validate')
pin, manifest = V.verify_manifest()
import run
assert run is R and run.runtime_environment.v10d_amended
assert Path(V.__file__).resolve().is_relative_to(root)
print(pin, (pin.parent / 'runs' / 'JOB').as_posix())
"""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(SLURM_JOB_ID="JOB", V10_VALIDATION_MANIFEST=str(copy), PYTHONDONTWRITEBYTECODE="1",
               V10_VALIDATION_SHA256=C.sha(manifest))
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True, env=env, timeout=300)
    assert result.returncode == 0, result.stderr[-3000:]
    pin, target = result.stdout.split()
    assert Path(pin) == copy.resolve() and Path(target).is_relative_to(tmp_path.resolve())
