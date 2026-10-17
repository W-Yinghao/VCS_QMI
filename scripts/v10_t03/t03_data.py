"""V10 T03 data: the package generator (sha256-pinned, unchanged), one complete suite per data root seed, sample-only loading.

Decision 9.2: "使用原包生成器及其样本/真值分离方式，记录每套数据的生成器与样本 hash".  A suite = generate_suite(root_seed=...)
with all 12 conditions and FIT / CAL / EVAL = 4096 / 2048 / 8192 independent base observations, written once per root seed under
<out>/data/seed<root>/ (atomic rename; a concurrent writer of the same seed loses the race and verifies the winner).  Every load
re-hashes samples.npz against the generator's metadata sample_sha256.  This module only reads FIT / CAL / EVAL samples through the
package load_samples (which refuses any oracle field); it has no truth reader (only the evaluator module has one).
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import t03_common as C


def load_generator():
    """Import the package generator from its package path after checking its pinned sha256 (one module object per process).  Its
    truth APIs are wrapped in memory by t03_common.guard_truth_api (the file is unchanged) so that they refuse to run inside the truth
    barrier: the generator's analytic tables are as much evaluator-only as the truth files."""
    mod = sys.modules.get("v10_synthetic")
    if mod is not None and Path(getattr(mod, "__file__", "")).resolve() == C.GENERATOR_PATH.resolve():
        return C.guard_truth_api(mod)
    C.verify_pinned_file(C.GENERATOR_PATH, C.GENERATOR_SHA256, "package generator")
    spec = importlib.util.spec_from_file_location("v10_synthetic", C.GENERATOR_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v10_synthetic"] = mod
    spec.loader.exec_module(mod)
    return C.guard_truth_api(mod)


GEN = load_generator()
SUITE_RECORD = "T03_SUITE_RECORD.json"


def suite_dir(root_seed: int, root: Path | None = None) -> Path:
    return (root or C.out_root()) / "data" / f"seed{int(root_seed)}"


def _suite_record(directory: Path, root_seed: int, role_sizes: dict, families, sizes, rhos, generation_seconds: float) -> dict:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = {}
    for entry in manifest["cases"]:
        cid = entry["case_id"]
        cases[cid] = {"family": entry["family"], "m": entry["m"], "rho": entry["rho"], "roles": {}}
        for role in ("fit", "cal", "eval"):
            meta = json.loads((directory / cid / role / "metadata.json").read_text(encoding="utf-8"))
            digest = C.sha256_file(directory / cid / role / "samples.npz")
            if digest != meta["sample_sha256"]:
                raise RuntimeError(f"{cid}/{role}: samples.npz sha256 {digest} != generator metadata {meta['sample_sha256']}")
            cases[cid]["roles"][role] = {"n": meta["n_independent_base_observations"], "sample_sha256": digest,
                                         "metadata_sha256": C.sha256_file(directory / cid / role / "metadata.json"),
                                         "spawn_key": meta["spawn_key"], "state_fingerprint_uint32": meta["state_fingerprint_uint32"],
                                         "base_id_namespace": meta["base_id_namespace"]}
    return {"schema": "v10_t03.suite_record.v1", "root_seed": int(root_seed), "generator_path": str(C.GENERATOR_PATH.relative_to(C.REPO)),
            "generator_sha256": C.sha256_file(C.GENERATOR_PATH), "generator_call": {"function": "generate_suite", "root_seed": int(root_seed),
            "families": list(families), "sizes": list(sizes), "rhos": list(rhos), "role_sizes": dict(role_sizes)},
            "numpy_version": np.__version__, "manifest_sha256": C.sha256_file(manifest_path), "n_cases": manifest["n_cases"],
            "n_independent_base_observations_total": manifest["n_independent_base_observations_total"], "cases": cases,
            "created_utc": C.utc_now(), "generation_seconds": generation_seconds,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "truth_policy": "truth files stay in the package-layout evaluator-only case subdirectory; "
            "the T03 truth barrier denies them to training and CAL selection; only the evaluator module reads them"}


def verify_suite(directory: Path, root_seed: int, role_sizes: dict | None = None) -> dict:
    """Re-hash every sample file against the generator metadata and the suite record; returns the record."""
    role_sizes = dict(role_sizes or C.ROLE_SIZES)
    record = json.loads((directory / SUITE_RECORD).read_text(encoding="utf-8"))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    problems = []
    if record["root_seed"] != int(root_seed) or manifest["root_seed"] != int(root_seed):
        problems.append(f"root seed {record['root_seed']} / {manifest['root_seed']} != {root_seed}")
    if record["generator_sha256"] != C.GENERATOR_SHA256:
        problems.append(f"suite generated by generator sha256 {record['generator_sha256']}, pinned {C.GENERATOR_SHA256}")
    if record["manifest_sha256"] != C.sha256_file(directory / "manifest.json"):
        problems.append("manifest.json changed after generation")
    for cid, case in record["cases"].items():
        for role, info in case["roles"].items():
            if info["n"] != role_sizes[role]:
                problems.append(f"{cid}/{role}: n {info['n']} != {role_sizes[role]}")
            if C.sha256_file(directory / cid / role / "samples.npz") != info["sample_sha256"]:
                problems.append(f"{cid}/{role}: samples.npz changed after generation")
    if problems:
        raise RuntimeError(f"suite {directory} failed verification: " + "; ".join(problems))
    return record


def ensure_suite(root_seed: int, root: Path | None = None, *, role_sizes: dict | None = None, families=C.FAMILIES, sizes=C.MS,
                 rhos=C.RHOS) -> dict:
    """The complete suite of a data root seed: generated once (temporary directory + atomic rename), then verified on every use."""
    role_sizes = dict(role_sizes or C.ROLE_SIZES)
    final = suite_dir(root_seed, root)
    if not (final / SUITE_RECORD).exists():
        final.parent.mkdir(parents=True, exist_ok=True)
        tmp = final.parent / f".tmp_{final.name}_{os.getpid()}_{time.time_ns()}"
        t0 = time.perf_counter()
        GEN.generate_suite(tmp, root_seed=int(root_seed), families=tuple(families), sizes=tuple(sizes), rhos=tuple(rhos),
                           role_sizes=role_sizes)
        record = _suite_record(tmp, root_seed, role_sizes, families, sizes, rhos, time.perf_counter() - t0)
        C.write_json_atomic(tmp / SUITE_RECORD, record)
        try:
            os.rename(tmp, final)
        except OSError:
            if not (final / SUITE_RECORD).exists():
                raise
            shutil.rmtree(tmp)                                       # another writer won the race: verify its suite below
    return verify_suite(final, root_seed, role_sizes)


@dataclass(frozen=True)
class SampleSet:
    """Sample-only view of one role of one case: integer observations, never any oracle value."""
    case_id: str
    role: str
    m: int
    x: np.ndarray
    y: np.ndarray
    sample_sha256: str

    @property
    def n(self) -> int:
        return int(len(self.x))


def load_role(case_directory: Path, role: str, m: int, expect_sha256: str | None = None) -> SampleSet:
    """Package load_samples (refuses oracle fields) + range / id checks + the sample sha256."""
    case_directory = Path(case_directory)
    path = case_directory / role / "samples.npz"
    digest = C.sha256_file(path)
    meta = json.loads((case_directory / role / "metadata.json").read_text(encoding="utf-8"))
    if digest != meta["sample_sha256"] or (expect_sha256 is not None and digest != expect_sha256):
        raise RuntimeError(f"{path}: sha256 {digest} does not match the generator metadata / suite record")
    if meta.get("contains_oracle_values") is not False:
        raise RuntimeError(f"{path}: metadata does not certify an oracle-free payload")
    s = GEN.load_samples(case_directory, role)
    x, y = s["x"].astype(np.int64), s["y"].astype(np.int64)
    if not (np.array_equal(s["base_id"], np.arange(len(x))) and x.min() >= 0 and y.min() >= 0 and x.max() < m and y.max() < m):
        raise RuntimeError(f"{path}: observations outside [0, {m}) or base ids not 0..n-1")
    return SampleSet(case_directory.name, role, int(m), x, y, digest)


def load_case_samples(record: dict, suite: Path, cid: str, roles=("fit", "cal")) -> dict[str, SampleSet]:
    case = record["cases"][cid]
    return {r: load_role(suite / cid, r, case["m"], case["roles"][r]["sample_sha256"]) for r in roles}
