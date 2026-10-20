"""D3 hardware amendment: the frozen gate's L40S-name condition only, in this process only.

scripts/v10_supplement/run.py runtime_environment() requires gpu_name == 'NVIDIA L40S' and
then checks torch, cuda_runtime, cudnn, torch_threads, the precision settings and the
precision override variables against the parent protocol. The amended function replaces
only the first condition (as the seed replication's hardware.py did), restricted to the two
pre-registered D3 GPU types; all other checks are performed exactly as before.
"""
from __future__ import annotations

from .common import require, sha

D3_POOL = {"RTX6000PRO": "NVIDIA RTX PRO 6000", "H100": "NVIDIA H100"}
RUNTIME_KEYS = ("torch", "cuda_runtime", "cudnn", "torch_threads")
PARENT_PROTOCOL_SHA256 = "de4c0c3943d5a54b3a7de73a84785f240f7ee5b1396d173fd2109999a5555e9d"


def hardware_allowed(environment, pool=D3_POOL):
    """environment['slurm_partition'] is the job's SLURM_JOB_PARTITION as recorded by the frozen adapter."""
    partition = environment.get("slurm_partition")
    gpu = environment.get("gpu_name")
    return partition in pool and isinstance(gpu, str) and gpu.startswith(pool[partition])


def validate_environment(environment, settings, expected, pool=D3_POOL):
    """Every frozen runtime check except the L40S name; the GPU must match its Slurm partition."""
    require(hardware_allowed(environment, pool),
            f"GPU {environment.get('gpu_name')!r} on partition {environment.get('slurm_partition')!r} "
            f"is outside the pre-registered D3 pool {sorted(pool)}")
    for key in RUNTIME_KEYS:
        require(environment[key] == expected["environment"][key], "Common runtime environment mismatch: " + key)
    require(settings == expected["settings"], "Common precision settings differ")
    require(environment["precision"]["env"] == expected["env"], "Common precision environment overrides differ")


def amended_runtime_environment(runner, *, pool=D3_POOL, scope):
    """Replacement for runner.runtime_environment, bound to the frozen runner module."""
    original = runner.runtime_environment

    def runtime_environment():
        parent = runner.PARENT / "PROTOCOL.json"
        require(sha(parent) == PARENT_PROTOCOL_SHA256, "Parent common protocol changed")
        environment = runner.A.environment("cuda")
        expected = runner.read(parent)["precision"]
        validate_environment(environment, runner.A.precision_settings(environment["precision"]), expected, pool)
        environment["hardware_amendment"] = {
            "scope": scope,
            "replaced_condition": "gpu_name == 'NVIDIA L40S' (scripts/v10_supplement/run.py runtime_environment)",
            "replacement": {partition: f"slurm_partition == {partition!r} and gpu_name startswith {prefix!r}"
                            for partition, prefix in pool.items()},
            "unchanged_checks": list(RUNTIME_KEYS) + ["precision settings", "precision override variables"],
            "original_function": f"{original.__module__}.{original.__qualname__}",
            "claim_scope": "a gate on another GPU type; no cross-device equivalence claim",
        }
        return environment

    runtime_environment.v10d_amended = True
    return runtime_environment
