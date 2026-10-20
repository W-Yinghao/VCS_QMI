"""Subprocess probe (clean interpreter): bind the frozen pipeline runner exactly as d3_gate.py does,
then call the ORIGINAL and the AMENDED runtime_environment on fake environments.

argv: <frozen source root> <diag src>; prints one JSON object. No Torch, no CUDA, no data.
"""
import copy
import json
import os
from pathlib import Path
import sys

frozen_root, diag_src = Path(sys.argv[1]), sys.argv[2]
sys.path.append(diag_src)
from v10_sensitivity import frozen as Z  # noqa: E402
from v10_sensitivity import hardware as H  # noqa: E402

R = Z.import_runner(frozen_root, "run")
expected = R.read(R.PARENT / "PROTOCOL.json")["precision"]
settings = dict(expected["settings"])


def environment(partition, gpu, **changes):
    env = {"slurm_partition": partition, "gpu_name": gpu, "torch": expected["environment"]["torch"],
           "cuda_runtime": expected["environment"]["cuda_runtime"], "cudnn": expected["environment"]["cudnn"],
           "torch_threads": expected["environment"]["torch_threads"],
           "precision": {"settings": dict(settings), "env": copy.deepcopy(expected["env"])}}
    for key, value in changes.items():
        if key == "settings":
            env["precision"]["settings"].update(value)
        elif key == "precision_env":
            env["precision"]["env"].update(value)
        else:
            env[key] = value
    return env


class FakeAdapter:
    def __init__(self, env):
        self.env = env

    def environment(self, device):
        assert device == "cuda"
        return copy.deepcopy(self.env)

    @staticmethod
    def precision_settings(record):
        return record["settings"]


cases = {
    "l40s": environment("L40S", "NVIDIA L40S"),
    "h100": environment("H100", "NVIDIA H100 NVL"),
    "rtx": environment("RTX6000PRO", "NVIDIA RTX PRO 6000 Blackwell Server Edition"),
    "h100_on_rtx_partition": environment("RTX6000PRO", "NVIDIA H100 NVL"),
    "h100_torch": environment("H100", "NVIDIA H100 NVL", torch="2.8.0+cu128"),
    "h100_cudnn": environment("H100", "NVIDIA H100 NVL", cudnn=91100),
    "h100_threads": environment("H100", "NVIDIA H100 NVL", torch_threads=8),
    "h100_tf32_matmul": environment("H100", "NVIDIA H100 NVL", settings={"float32_matmul_precision": "high"}),
    "h100_override": environment("H100", "NVIDIA H100 NVL",
                                 precision_env={"NVIDIA_TF32_OVERRIDE": {"set": True, "value": "0"}}),
}
# the same non-name mismatches on L40S, where the frozen check reaches them
for name in ("torch", "cudnn", "threads", "tf32_matmul", "override"):
    twin = copy.deepcopy(cases["h100_" + name])
    twin["slurm_partition"], twin["gpu_name"] = "L40S", "NVIDIA L40S"
    cases["l40s_" + name] = twin
original = R.runtime_environment
amended = H.amended_runtime_environment(R, scope="probe")
out = {"run_module_file": R.__file__, "original": {}, "amended": {}}
for name, env in cases.items():
    os.environ["SLURM_JOB_PARTITION"] = env["slurm_partition"]
    R.A = FakeAdapter(env)
    for label, function in (("original", original), ("amended", amended)):
        try:
            result = function()
            out[label][name] = "PASS" + ("+amendment" if "hardware_amendment" in result else "")
        except RuntimeError as error:
            out[label][name] = "REFUSED: " + str(error)
print(json.dumps(out))
