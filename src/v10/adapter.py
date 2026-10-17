"""V10 MF-VCS real-host adapter on the V9-G1 BF-SHG registration host (phase 1: T01 validation, T02 timing).

Host (imported read-only, never modified): src/v9/reg/g1.py (G) and the top-level modules it imports, r6_reg_pilot (P) and
r6_reg_p2 (Q).  The three files are pinned by the hash the formal V9-G1 jobs printed
(cat g1.py r6_reg_pilot.py r6_reg_p2.py | sha256 = 6c6f094684fbc50a...); import refuses any other bytes.

Reused unchanged: P.build_models (encoders, host init order), P.learned_inputs / P.load_img (R6 cache, read-only), P.sample_map,
P.j_value, P.loss_fn, G.sample_fit, Q.RigidEnergy (warp-then-encode chain; only j_critic is overridden), Q.apply_T,
Q.sample_pts, Q.optimise_rigid, Q.Inputs, Q.seed_of, Q.perturbations_p2.
Transcribed (the host has no callable for it): the inline training-batch block of G.cmd_train L147-166 at eps = 0, with the
identical numpy RNG call order (HostBatchSampler).
Every V9 writer (G.cmd_train / cmd_comir / cmd_register / ensure_layout, P.cmd_*, Q.cmd_*) is replaced IN THIS PROCESS by a
tripwire that raises, so nothing imported here can write under outputs_v9 (the files themselves are untouched).

MF objectives use the unchanged package backend v10.mf_vcs: per-scene V mode (the host's ordinary within-scene permutation
reference; fixed points allowed), locked scene weights 1/n_sc (the host's implicit weighting with equal n per scene),
W in [-1, 1]^{K x K} projected after every optimiser step, AdamW weight decay 0 for W, no output tanh, ridge 0 by default.
  MF_perm1  sampled_permutation_terms with the host's own per-scene permutation of the step (perm1)
  MF_M0     objective_terms: exact V moments
  MF_M1     the profiled head on the SAME batch tensors; training uses require_converged=False and logs the gap per batch.
            Owner decision 2026-10-17 (v10/decisions/V10_PHASE1_DECISIONS_20261017_CN.md section 2; contract key "m1"), rule
            'relative_zero_head' (M1Config default): per batch ONE QuadraticHead snapshot of the current fixed features;
            g_ref = G_b(0) = ||Delta_b||_1 read from that snapshot's delta (no HVP, no extra moment pass); stop when
            G_b(W) <= tau_b = 0.01 g_ref, else at the iteration cap (CAL candidates 10 / 30 / 100; approximate head, its true
            FW gap logged); warm start = the previous batch's W (first batch W = 0); an exactly zero Delta_b returns W = 0 with
            status zero_reference_gap (no division, the solver is not called with a zero tolerance); a same-input fp64
            diagnostic when g_ref is near the working-dtype resolution (its HVPs / moment pass counted in the m1_*_total fields).
            Rule 'absolute' (the unchanged package profiled_objective with an absolute FW-gap target) defaults to
            require_converged=True: the strict numeric-audit mode (package design 7, decision 2.1).  With require_converged=False
            passed explicitly it is the pre-decision approximate mode, kept only for reference rows (T01 cap-binding / degenerate
            case, T02 cap-binding cost references): the Trainer refuses it unless TrainConfig.absolute_reference_only=True; it is
            never the training rule.  History: the package start W = 0 with the absolute target 1e-4 was degenerate at the host
            init (||Delta||_1 = 3.2e-5 < 1e-4 on the first T01 / T02 batch: iteration 0, W = 0, zero encoder gradient); T01 keeps
            that case as a diagnostic.  mf_vcs.py is unchanged (package sha256).
  MF_dense  dense_objective_terms: exact dense reference (validation only)
MLP (the host critic, T = tanh f), same P / Q, scene order and normalisation:
  MLP_perm1 host-identical (P.loss_fn 'vcs' on the host's permutation)
  MLP_perm8 the host's permutation + 7 more per scene from a separate stream (the host data stream is not touched)
  MLP_dense all within-scene pairs (V), optionally in row blocks with non-reentrant checkpointing
Test-time energy (MFRigidEnergy): encoders + feature maps + W fixed (eval mode, requires_grad False, enforced), nothing
refitted per candidate, autograd kept for the geometry parameters (no torch.no_grad around the energy), warp-then-encode,
P and Q both from the same warped-then-encoded moving descriptors.
Provenance: environment() / precision_record() record the effective numeric environment after all imports (owner decision
2026-10-17 section 3, option 2A: versions, GPU, matmul / cuDNN fp32 precisions as returned, determinism switches,
CUBLAS_WORKSPACE_CONFIG and both TF32 override variables, unset recorded as null); no precision setting is changed here.
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from v10 import mf_vcs as MF
from v10.mf_vcs import (SimplexBilinearCritic, dense_objective_terms, objective_terms, partition_scenes, profiled_objective,
                        sampled_permutation_terms)

# ------------------------------------------------------------------------------------------------------------ constants
REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "v10" / "package_v10_server_v1_20261016"
PACKAGE_MF_VCS_SHA256 = "cdf5df2e43bacb74b963f7b6b519a962f91fc8e41f28b4f6a4e2696816024973"
V10_MANIFEST = REPO / "v10" / "T00" / "V10_DATA_MANIFEST.csv"
V10_MANIFEST_SHA256 = "cc309cece34b574b2a518b5775a1e30987ebd9015693cca1543d5610f13564f6"
HOST_CODE_HASH = "6c6f094684fbc50a"          # cat g1.py r6_reg_pilot.py r6_reg_p2.py | sha256sum | cut -c1-16 (V9-G1 job headers)
OUT_ROOT = Path(os.environ.get("V10_OUT_ROOT", "/projects/EEG-foundation-model/yinghao/outputs_v10"))
FORBIDDEN_PATH_PARTS = ("/closed/", "outputs_v9/reg_test", "test.zip", "/test/")
DESC = 16                                     # host descriptor width (P.build_models desc=16)
TOLERANCES = {"float64": (1e-10, 1e-8), "float32": (1e-5, 1e-4)}   # package V10_DESIGN 6.1: (atol, rtol)
CODE_PATHS = ("src/v10", "scripts/v10")
JOB_PIN_PATHS = ("src/v10", "scripts/v10", "slurm/v10")              # pinned at submission (slurm/v10/submit.sh: V10_EXPECT_CODE)
DIRTY_PATHS = ("src/v10", "scripts/v10", "slurm/v10", "tests/v10", "src/v9/reg")   # = the dirty check of slurm/v10/v10_header.sh
M1_PROTOCOL_TOLERANCE = 1e-4          # package V10_PROTOCOL head.M1_gap_absolute_target (absolute FW gap): rule 'absolute' only
M1_CAP_BINDING_TOLERANCE = 1e-300     # rule 'absolute': far below any reachable gap, the solver always runs its whole iteration budget
M1_RULES = ("relative_zero_head", "absolute")
M1_RELATIVE_TOLERANCE = 0.01          # decision 2026-10-17 2.1: tau_b = 0.01 * G_b(0) = 0.01 * ||Delta_b||_1
M1_CAL_CAPS = (10, 30, 100)           # decision 2026-10-17 2.1 / 8.1: M1 iteration caps (CAL candidates)
M1_FP64_DIAGNOSTIC_BELOW = 1e4        # fp64 same-input diagnostic when g_ref <= 1e4 * M1 resolution unit (m1_resolution_unit)
TF32_ENV_VARS = ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "NVIDIA_TF32_OVERRIDE")   # decision 2026-10-17 section 3: always recorded


# ------------------------------------------------------------------------------------------------------------ host import
def _import_host():
    from v9.reg import g1 as G                # inserts src/v9/reg into sys.path; imports r6_reg_pilot / r6_reg_p2 top-level (G L35-38)
    P, Q = G.P, G.Q
    if sys.modules.get("r6_reg_pilot") is not P or sys.modules.get("r6_reg_p2") is not Q or Q.P is not P:
        raise ImportError("host module identity: expected the top-level r6_reg_pilot / r6_reg_p2 objects used by g1")
    files = (Path(G.__file__).resolve(), Path(P.__file__).resolve(), Path(Q.__file__).resolve())
    digest = hashlib.sha256(b"".join(f.read_bytes() for f in files)).hexdigest()[:16]
    if digest != HOST_CODE_HASH:
        raise ImportError(f"host code hash {digest} != pinned {HOST_CODE_HASH}: the audited V9-G1 host changed; re-audit first")
    return G, P, Q, files


G, P, Q, HOST_FILES = _import_host()

_V9_WRITERS = ((G, "G", ("cmd_train", "cmd_comir", "cmd_register", "ensure_layout")),
               (P, "P", ("cmd_prepare", "cmd_perturb", "cmd_train", "cmd_comir", "cmd_register", "main")),
               (Q, "Q", ("cmd_perturb", "cmd_register", "main")))


def _tripwire(label):
    def _raise(*args, **kwargs):
        raise RuntimeError(f"V10 adapter: {label} writes V9 outputs (outputs_v9 / R6 roots) and must not be called from V10")
    _raise.__name__ = "v10_tripwire_" + label.replace(".", "_")
    _raise.v10_tripwire = True
    return _raise


for _mod, _label, _names in _V9_WRITERS:
    for _n in _names:
        setattr(_mod, _n, _tripwire(f"{_label}.{_n}"))


def host_code_hash() -> str:
    return hashlib.sha256(b"".join(f.read_bytes() for f in HOST_FILES)).hexdigest()[:16]


# ------------------------------------------------------------------------------------------------------------ provenance
def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args) -> str:
    r = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else f"<git error {r.returncode}: {r.stderr.strip()[:200]}>"


def code_hash(paths: Sequence[str] = CODE_PATHS) -> str:
    """sha256 over '<sha256>  <path>' lines of the tracked files under paths (sorted bytewise) -> first 16 hex.
    Shell equivalent (slurm/v10 headers): git ls-files src/v10 scripts/v10 | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16"""
    files = sorted(f for f in git("ls-files", *paths).split("\n") if f and not f.startswith("<git error"))
    lines = "".join(f"{sha256_file(REPO / f)}  {f}\n" for f in files)
    return hashlib.sha256(lines.encode()).hexdigest()[:16]


def check_job_pin() -> dict:
    """Jobs run the code pinned at submission: slurm/v10/submit.sh exports V10_EXPECT_CODE = code_hash(JOB_PIN_PATHS) and every V10
    entry point calls this.  Refuses (SystemExit) when the pin is set and the tree's hash differs (code committed after submission);
    an unset pin (interactive use, unit tests) is recorded as 'unset'."""
    expected, actual = os.environ.get("V10_EXPECT_CODE"), code_hash(JOB_PIN_PATHS)
    if expected and expected != actual:
        raise SystemExit(f"[v10] REFUSED: code hash {actual} of {' '.join(JOB_PIN_PATHS)} != pinned {expected} (changed after submission)")
    return {"expected": expected, "actual": actual, "status": "match" if expected else "unset"}


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:  # noqa: BLE001 - provenance fields are best effort
        return default


def env_explicit(name: str) -> dict:
    """An environment variable recorded explicitly: {'set': False, 'value': None} when unset (never assumed empty)."""
    value = os.environ.get(name)
    return {"set": value is not None, "value": value}


PRECISION_SETTING_KEYS = ("float32_matmul_precision", "cuda_matmul_fp32_precision_raw", "backends_fp32_precision_raw",
                          "cudnn_fp32_precision_raw", "cudnn_conv_fp32_precision", "cudnn_rnn_fp32_precision_raw",
                          "deterministic_algorithms", "deterministic_algorithms_warn_only", "cudnn_benchmark", "cudnn_deterministic")


def precision_record(device=None) -> dict:
    """Decision 2026-10-17 section 3 (option 2A, contract key "precision"): the EFFECTIVE numeric environment, read after all imports
    (the callers run it inside their entry points) and never set here.  Raw fp32_precision strings are stored as returned
    ('none' = inherited from the parent backend; not rewritten to 'ieee').  torch.get_float32_matmul_precision() is the legacy getter
    (torch 2.9 warns on it; reading changes no setting).  CUBLAS_WORKSPACE_CONFIG and both TF32 override variables are recorded
    explicitly, unset included."""
    device = torch.device(device) if device is not None else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        matmul_precision = torch.get_float32_matmul_precision()
    rec = {"decision": "V10-phase1-decisions-20261017 section 3 (2A: common recorded environment, settings not changed)",
           "torch_version": torch.__version__, "cuda_version_torch_build": torch.version.cuda,
           "cudnn_version": _safe(torch.backends.cudnn.version), "cudnn_available": _safe(torch.backends.cudnn.is_available),
           "float32_matmul_precision": matmul_precision,
           "cuda_matmul_fp32_precision_raw": _safe(lambda: torch.backends.cuda.matmul.fp32_precision),
           "backends_fp32_precision_raw": _safe(lambda: torch.backends.fp32_precision),
           "cudnn_fp32_precision_raw": _safe(lambda: torch.backends.cudnn.fp32_precision),
           "cudnn_conv_fp32_precision": _safe(lambda: torch.backends.cudnn.conv.fp32_precision),
           "cudnn_rnn_fp32_precision_raw": _safe(lambda: torch.backends.cudnn.rnn.fp32_precision),
           "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
           "deterministic_algorithms_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
           "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
           "env": {name: env_explicit(name) for name in ("CUBLAS_WORKSPACE_CONFIG",) + TF32_ENV_VARS},
           "recorded_after_imports_of": {m: m in sys.modules for m in ("v10.mf_vcs", "v10.adapter", "v9.reg.g1", "r6_reg_pilot", "r6_reg_p2")},
           "device": str(device) if device is not None else None}
    if device is not None and device.type == "cuda":
        prop = torch.cuda.get_device_properties(device)
        smi = _safe(lambda: subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                           capture_output=True, text=True, timeout=30).stdout.strip())
        rec.update({"gpu_name": torch.cuda.get_device_name(device), "gpu_capability": f"{prop.major}.{prop.minor}",
                    "gpu_total_bytes": prop.total_memory, "nvidia_smi_name_driver": smi})
    return rec


def precision_settings(rec: dict) -> dict:
    """The settings part of a precision_record (to verify that nothing changed them during a run)."""
    return {k: rec.get(k) for k in PRECISION_SETTING_KEYS}


def environment(device=None) -> dict:
    device = torch.device(device) if device is not None else None
    prec = precision_record(device)
    env = {"host": socket.gethostname(), "utc": utc_now(), "python": sys.version.split()[0], "executable": sys.executable,
           "torch": torch.__version__, "numpy": np.__version__, "cuda_runtime": torch.version.cuda,
           "cudnn": prec["cudnn_version"], "device": str(device) if device else None,
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
           "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST"), "git_commit": git("rev-parse", "HEAD"),
           "git_dirty_code": git("status", "--porcelain", "--", *DIRTY_PATHS),
           "code_hash_src_scripts_v10": code_hash(), "host_code_hash": host_code_hash(), "mf_vcs_sha256": sha256_file(MF.__file__),
           "job_code_hash": code_hash(JOB_PIN_PATHS), "job_pin_expected": os.environ.get("V10_EXPECT_CODE"),
           "submit_commit": os.environ.get("V10_SUBMIT_COMMIT"),
           "torch_threads": torch.get_num_threads(),
           # phase-1 flat keys, kept for continuity; the complete decision-3 record is "precision" (same reads)
           "float32_matmul_precision": prec["float32_matmul_precision"], "cudnn_conv_fp32_precision": prec["cudnn_conv_fp32_precision"],
           "cuda_matmul_fp32_precision": prec["cuda_matmul_fp32_precision_raw"],
           "deterministic_algorithms": prec["deterministic_algorithms"], "cudnn_benchmark": prec["cudnn_benchmark"],
           "cudnn_deterministic": prec["cudnn_deterministic"], "cublas_workspace_config": prec["env"]["CUBLAS_WORKSPACE_CONFIG"]["value"],
           # protocol P2 precision_environment.record_contract: both override variables as keys, raw value or null when unset
           **{name: prec["env"][name]["value"] for name in TF32_ENV_VARS},
           "precision": prec}
    if device is not None and device.type == "cuda":
        prop = torch.cuda.get_device_properties(device)
        env.update({"gpu_name": torch.cuda.get_device_name(device), "gpu_total_bytes": prop.total_memory,
                    "gpu_capability": f"{prop.major}.{prop.minor}", "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")})
    return env


def write_json_atomic(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=_json_default) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, torch.dtype):
        return str(o)
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def synchronize(device) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextlib.contextmanager
def cudnn_conv_precision(value: str | None):
    """Temporarily set torch.backends.cudnn.conv.fp32_precision ('tf32' is the installed default; 'ieee' disables TF32)."""
    if value is None:
        yield None
        return
    old = _safe(lambda: torch.backends.cudnn.conv.fp32_precision)
    torch.backends.cudnn.conv.fp32_precision = value
    try:
        yield value
    finally:
        if old is not None:
            torch.backends.cudnn.conv.fp32_precision = old


# ------------------------------------------------------------------------------------------------------------ data gate
class DataGuard:
    """Phase-1 data gate.  Only BF_SHG rows of the V10 manifest (v10/T00, sha256 pinned) with inclusion=include and an allowed
    role are readable; the R6 role must agree; no path may point into closed/ or outputs_v9/reg_test.  Every granted read is
    logged in self.reads (core, role, purpose, path).  The BF-SHG TEST cores are HELD (inclusion=exclude) and are refused."""

    def __init__(self, manifest=V10_MANIFEST, expect_sha256: str | None = V10_MANIFEST_SHA256, r6_root=None):
        raw = Path(manifest).read_bytes()
        self.manifest_path, self.manifest_sha256 = str(manifest), hashlib.sha256(raw).hexdigest()
        if expect_sha256 and self.manifest_sha256 != expect_sha256:
            raise RuntimeError(f"V10 manifest sha256 {self.manifest_sha256} != pinned {expect_sha256}")
        rows = list(csv.DictReader(raw.decode("utf-8").splitlines()))
        self.rows = {r["record_id"]: r for r in rows if r["dataset"] == "BF_SHG"}
        self.r6_root = Path(r6_root) if r6_root is not None else Path(G.P1_ROOT)
        self.r6_roles = json.loads((self.r6_root / "manifest.json").read_text())["roles"]
        self.reads: list[dict] = []

    def included(self, role: str) -> list[str]:
        return sorted(c for c, r in self.rows.items() if r["inclusion"] == "include" and r["role"] == role)

    def held(self) -> list[str]:
        return sorted(c for c, r in self.rows.items() if r["inclusion"] != "include")

    def check(self, core: str, allowed_roles: Iterable[str], purpose: str) -> Path:
        allowed = set(allowed_roles)
        if not allowed or not allowed <= {"FIT", "CAL", "DEV"}:
            raise PermissionError(f"allowed roles must be a non-empty subset of FIT/CAL/DEV, got {sorted(allowed)}")
        r = self.rows.get(core)
        if r is None:
            raise PermissionError(f"{core}: not a BF_SHG record of the V10 manifest")
        if r["inclusion"] != "include":
            raise PermissionError(f"{core}: inclusion={r['inclusion']} ({r['exclusion_reason'][:100]})")
        if r["role"] not in allowed:
            raise PermissionError(f"{core}: role {r['role']} not in {sorted(allowed)} for {purpose}")
        if self.r6_roles.get(core) != r["role"]:
            raise PermissionError(f"{core}: R6 role {self.r6_roles.get(core)} != V10 role {r['role']}")
        path = self.r6_root / "cache" / f"{core}.npz"
        for text in (str(path), r["fixed_path"], r["moving_path"]):
            if any(bad in text for bad in FORBIDDEN_PATH_PARTS):
                raise PermissionError(f"{core}: path {text} is in a forbidden location")
        self.reads.append({"core": core, "role": r["role"], "purpose": purpose, "path": str(path), "utc": utc_now()})
        return path

    def log_rule(self, core: str, allowed_roles: Iterable[str], purpose: str) -> None:
        """A truth-related quantity derived from a frozen rule (no file read), e.g. Q.perturbations_p2(core): the same role gate as
        check(), logged with kind 'rule' and no path."""
        self.check(core, allowed_roles, purpose)
        self.reads[-1].update({"kind": "rule", "path": None})


def load_learned_inputs(guard: DataGuard, cores: Iterable[str], allowed_roles, purpose: str, device) -> dict:
    """Host learned-method inputs (P.learned_inputs: BF RGB robust-norm [3,512,512], log1p SHG robust-norm [1,512,512])."""
    out = {}
    for c in sorted(set(cores)):
        guard.check(c, allowed_roles, purpose)
        out[c] = P.learned_inputs(torch, guard.r6_root, c, torch.device(device))
    return out


class GuardedInputs(Q.Inputs):
    """Q.Inputs (host registration inputs: reference (BF lum, BF RGB), moving V o T*^-1) behind the data gate."""

    def __init__(self, guard: DataGuard, allowed_roles, purpose: str, pert: dict, device):
        super().__init__(torch, F, guard.r6_root, pert, torch.device(device))
        self.guard, self.allowed, self.purpose = guard, set(allowed_roles), purpose

    def reference(self, c):
        if c not in self.ref:
            self.guard.check(c, self.allowed, self.purpose + ":reference")
        return super().reference(c)

    def moving(self, c, i):
        if (c, i) not in self.mov:
            self.guard.check(c, self.allowed, self.purpose + f":moving#{i}")
        return super().moving(c, i)


def fit_cores(guard: DataGuard, frac: float = 1.0, seed: int = 1001) -> list[str]:
    """FIT cores (V10 manifest include FIT == R6 FIT, checked) sampled with the host rule G.sample_fit (all cores at frac 1)."""
    fit_full = guard.included("FIT")
    r6_fit = sorted(c for c, r in guard.r6_roles.items() if r == "FIT")
    if fit_full != r6_fit:
        raise RuntimeError("V10 include-FIT set differs from the R6 FIT set")
    fit, _ = G.sample_fit(fit_full, frac, seed)
    return fit


# ------------------------------------------------------------------------------------------------------------ host training batch
@dataclass
class HostBatch:
    step: int
    cores: list
    y0: np.ndarray
    x0: np.ndarray
    k: np.ndarray
    flip: np.ndarray
    yx: np.ndarray        # [n_sc, n_pos, 2] float64 as drawn; cast to float32 on use (host G L165)
    perm: np.ndarray      # [n_sc, n_pos] int64, one ordinary permutation per scene (fixed points allowed)


class HostBatchSampler:
    """Transcription of G.cmd_train L147-166 at eps = 0 with the identical numpy call order on g = default_rng(1000 + seed):
    choice(fit, n_sc, replace=len(fit) < n_sc); per crop integers(0, SIZE - crop, 2), integers(4), integers(2);
    uniform(inner, crop - inner - 1, (n_sc, n_pos, 2)); n_sc x permutation(n_pos).  n_sc defaults to the host's min(16, len(fit))."""

    def __init__(self, fit: Sequence[str], seed: int, *, n_sc: int | None = None, n_pos: int = 128, crop: int = 160):
        self.fit, self.seed = list(fit), int(seed)
        self.g = np.random.default_rng(1000 + self.seed)
        self.n_sc = min(16, len(self.fit)) if n_sc is None else int(n_sc)
        self.n_pos, self.crop, self.inner = int(n_pos), int(crop), P.RF_HALF + 1
        self.step = 0

    def draw(self) -> HostBatch:
        g, n_sc, crop = self.g, self.n_sc, self.crop
        self.step += 1
        cores = g.choice(self.fit, n_sc, replace=len(self.fit) < n_sc)
        y0, x0 = np.empty(n_sc, np.int64), np.empty(n_sc, np.int64)
        k, fl = np.empty(n_sc, np.int64), np.empty(n_sc, bool)
        for i in range(n_sc):
            yx0 = g.integers(0, P.SIZE - crop, 2)
            y0[i], x0[i] = yx0[0], yx0[1]
            k[i] = int(g.integers(4))
            fl[i] = bool(g.integers(2))
        yx = g.uniform(self.inner, crop - self.inner - 1, (n_sc, self.n_pos, 2))
        perm = np.stack([g.permutation(self.n_pos) for _ in range(n_sc)])
        return HostBatch(self.step, [str(c) for c in cores], y0, x0, k, fl, yx, perm)


def replay_permutation_fixed_points(fit: Sequence[str], seed: int, steps: int, n_sc: int | None = None) -> dict:
    """Fixed-point statistics of the training permutations of the first `steps` host steps (numpy only, no data)."""
    s = HostBatchSampler(fit, seed, n_sc=n_sc)
    fp, distinct = [], True
    for _ in range(steps):
        hb = s.draw()
        distinct &= len(set(hb.cores)) == len(hb.cores)
        fp += [int((p == np.arange(len(p))).sum()) for p in hb.perm]
    fp = np.array(fp)
    return {"n_sc": s.n_sc, "n_perms": int(fp.size), "mean_fixed_points": float(fp.mean()),
            "share_with_>=1_fixed_point": float((fp > 0).mean()), "distinct_cores_per_step": bool(distinct)}


def assemble_crops(imgs: dict, hb: HostBatch, crop: int = 160):
    """Host G L150-155: aligned crop, identical rot90 k and horizontal flip on both modalities."""
    bu, bv = [], []
    for i, c in enumerate(hb.cores):
        U, V = imgs[c]
        y0, x0, k = int(hb.y0[i]), int(hb.x0[i]), int(hb.k[i])
        u, v = U[:, y0:y0 + crop, x0:x0 + crop], V[:, y0:y0 + crop, x0:x0 + crop]
        u, v = torch.rot90(u, k, (1, 2)), torch.rot90(v, k, (1, 2))
        if bool(hb.flip[i]):
            u, v = u.flip(2), v.flip(2)
        bu.append(u)
        bv.append(v)
    return torch.stack(bu), torch.stack(bv)


def batch_descriptors(eU, eV, bu, bv, yx_np):
    """Host G L156, L165-166 (eps 0): DU, DV = eU(crops), eV(crops); du[s], dv[s] = sample_map at the float32-rounded yx[s]."""
    DU, DV = eU(bu), eV(bv)
    yx = torch.tensor(yx_np, dtype=torch.float32, device=DU.device).to(DU.dtype)
    du = torch.stack([P.sample_map(torch, F, DU[s], yx[s]) for s in range(len(yx))])
    dv = torch.stack([P.sample_map(torch, F, DV[s], yx[s]) for s in range(len(yx))])
    return du, dv


# ------------------------------------------------------------------------------------------------------------ MF head and objectives
class MFHead(SimplexBilinearCritic):
    """The package SimplexBilinearCritic (two-sided learnable projection + softmax, box head W, no tanh, no ridge),
    with the two sides available separately for registration (features_a / features_b == features())."""

    def features_a(self, x):
        return torch.softmax(self.left(x) / self.temperature, -1)

    def features_b(self, y):
        return torch.softmax(self.right(y) / self.temperature, -1)


def scene_ids(n_sc: int, n_pos: int, device) -> torch.Tensor:
    return torch.arange(n_sc, device=device).repeat_interleave(n_pos)


def host_weights(n_sc: int) -> list[float]:
    """Locked scene weights: the host's implicit 1/n_sc (equal n_pos per scene, G L172 .mean())."""
    return [1.0 / n_sc] * n_sc


def mf_scenes(A, B, n_sc: int, n_pos: int):
    """Local V scenes in crop order s = 0..n_sc-1 (partition_scenes sorts the ids, which are 0..n_sc-1)."""
    return partition_scenes(A, B, scene_ids(n_sc, n_pos, A.device), q_mode="v")


@dataclass
class M1Config:
    """M1 head rule (module doc).  rule='relative_zero_head' (decision 2026-10-17 2.1, the protocol rule): threshold
    relative_tolerance * G_b(0) of the current batch, iteration cap max_iterations, no absolute floor; `tolerance` must stay None;
    require_converged defaults to False (training: approximate head at the cap, true gap logged).
    rule='absolute': the unchanged package profiled_objective with the absolute FW-gap target `tolerance` (None -> the package
    value 1e-4).  require_converged defaults to True there = the strict numeric-audit mode (package design 7: strict acceptance uses
    the default require_converged=True; decision 2.1: the strict absolute mode is kept for the numerical audit); an unconverged head
    raises.  require_converged=False must be passed explicitly: the pre-decision approximate mode, kept only for reference rows (T01
    cap-binding / degenerate-case rows, T02 cap-binding cost references); the Trainer accepts rule 'absolute' only with
    TrainConfig.absolute_reference_only=True (protocol P2 m1_rule.strict_absolute_mode: never the training rule).
    fp64_diagnostic_below: the same-input fp64 diagnostic runs when g_ref <= fp64_diagnostic_below * m1_resolution_unit(dtype) (None
    disables; never runs in float64)."""
    max_iterations: int = 30                    # the cap; CAL candidates 10 / 30 / 100 (M1_CAL_CAPS)
    rule: str = "relative_zero_head"
    relative_tolerance: float = M1_RELATIVE_TOLERANCE
    tolerance: float | None = None              # rule 'absolute' only
    ridge: float = 0.0                          # V10_PROTOCOL head.M1_ridge_primary
    require_converged: bool | None = None       # None -> the rule's default: False (relative_zero_head), True (absolute: strict audit)
    fp64_diagnostic_below: float | None = M1_FP64_DIAGNOSTIC_BELOW

    def __post_init__(self):
        if self.rule not in M1_RULES:
            raise ValueError(f"M1 rule must be one of {M1_RULES}, got {self.rule!r}")
        if isinstance(self.max_iterations, bool) or not isinstance(self.max_iterations, int) or self.max_iterations < 0:
            raise ValueError("max_iterations must be a nonnegative integer")
        if self.rule == "relative_zero_head":
            if self.tolerance is not None:
                raise ValueError("an absolute tolerance belongs to rule='absolute'; the relative rule's threshold is relative_tolerance * "
                                 "G_b(0) (decision 2026-10-17 2.1: no absolute floor)")
            rt = self.relative_tolerance
            if isinstance(rt, bool) or not isinstance(rt, (int, float)) or not math.isfinite(rt) or not 0 < rt < 1:
                raise ValueError("relative_tolerance must be a finite number in (0, 1)")
        else:
            if self.tolerance is None:
                self.tolerance = M1_PROTOCOL_TOLERANCE
            if isinstance(self.tolerance, bool) or not math.isfinite(self.tolerance) or self.tolerance <= 0:
                raise ValueError("the absolute tolerance must be finite and positive")
        if self.require_converged is None:
            self.require_converged = self.rule == "absolute"
        if not isinstance(self.require_converged, bool):
            raise ValueError("require_converged must be None (the rule's default) or a bool")
        fb = self.fp64_diagnostic_below
        if fb is not None and (isinstance(fb, bool) or not math.isfinite(fb) or fb < 0):
            raise ValueError("fp64_diagnostic_below must be None or a finite nonnegative number")


def m1_resolution_unit(dtype) -> float:
    """2 eps(dtype): the L1 mass of the two moment matrices that Delta_b differences (E_P[a b^T] and E_Q[a b^T], each of L1 mass 1
    for simplex features with weights summing to 1) times the unit roundoff, i.e. the scale of the rounding error of Delta_b and of
    the FW gap.  Analytic; no extra pass over the data."""
    return 2.0 * torch.finfo(dtype).eps


def same_input_scene(s: MF.SceneBatch, dtype) -> MF.SceneBatch:
    """fp64 diagnostic only: a copy of an already validated SceneBatch whose feature VALUES are cast exactly to `dtype`, built without
    re-running SceneBatch validation (fp32 simplex rows sum to 1 only to fp32 rounding, which the float64 check (32 eps) rejects).
    Nothing is renormalised; the validated original is the input of the computation."""
    c = object.__new__(MF.SceneBatch)
    cast = (lambda t: None if t is None else t.detach().to(dtype))
    for name, value in (("a", cast(s.a)), ("b", cast(s.b)), ("q_mode", s.q_mode), ("a_marginal", cast(s.a_marginal)),
                        ("b_marginal", cast(s.b_marginal)), ("scene_id", s.scene_id), ("forbidden_edges", None)):
        object.__setattr__(c, name, value)
    return c


@contextlib.contextmanager
def count_head_hvp():
    """Counts QuadraticHead._hvp calls (the package solver's Hessian-vector products) inside the block.  The class attribute is
    wrapped for the duration of the block and restored afterwards; values are unchanged and src/v10/mf_vcs.py is not modified.
    solve_box_head calls _hvp 1 (initial F) + (iterations + 1) (gaps) + 2 per non-final iteration + 1 per momentum restart,
    so restarts = calls - 2 - 3 * iterations."""
    orig = MF.QuadraticHead._hvp
    counter = {"hvp": 0}

    def counted(self, direction):
        counter["hvp"] += 1
        return orig(self, direction)
    MF.QuadraticHead._hvp = counted
    try:
        yield counter
    finally:
        MF.QuadraticHead._hvp = orig


MF_METHODS = ("MF_perm1", "MF_M0", "MF_M1", "MF_dense")
MLP_METHODS = ("MLP_perm1", "MLP_perm8", "MLP_dense")
M1_STOP_REASONS = ("gap_tolerance", "iteration_budget_exhausted", "zero_reference_gap")   # protocol P2 m1_rule.log_field_rules
_SOLVER_STOP = {"gap_tolerance": "gap_tolerance", "iteration_budget_exhausted": "iteration_budget_exhausted"}   # the backend status strings


def _m1_info(m1: M1Config, *, dtype, W0, w, w0_zero: bool, g_ref, gap_start, gap_final, tolerance_effective, iterations: int,
             stop_reason: str, converged: bool, solver_status, hvp_calls: int, solver_hvp_calls: int, moments_passes: int,
             fp64_diagnostic: dict) -> dict:
    """Per-batch M1 log (contract key m1.log_fields, prefixed m1_): gap_ref_zero, gap_start, gap_final, gap_relative_to_zero,
    tolerance_effective, iterations, stop_reason, converged, hvp_calls, moments_passes, W update; m1_W is the solved head (not logged).
    Counters (decision 2.3: every added moment pass or HVP is counted): m1_hvp_calls / m1_moments_passes = the working-dtype M1
    computation (solver HVPs, plus 1 HVP when the gap of a non-zero warm start is logged at a zero reference; the QuadraticHead
    snapshot + the outer objective_terms); m1_fp64_diag_hvp_calls / m1_fp64_diag_moments_passes = the same-input fp64 diagnostic of
    this batch (0 when it did not run); m1_hvp_calls_total / m1_moments_passes_total = all M1 work of the batch."""
    dw = (w.detach() - W0)
    rel = (gap_final / g_ref) if (g_ref is not None and g_ref > 0) else None
    ran = bool(fp64_diagnostic.get("triggered"))
    diag_hvp = int(fp64_diagnostic["hvp_calls"]) if ran else 0
    diag_mom = int(fp64_diagnostic["moments_passes"]) if ran else 0
    return {"m1_rule": m1.rule, "m1_relative_tolerance": m1.relative_tolerance if m1.rule == "relative_zero_head" else None,
            "m1_absolute_tolerance": m1.tolerance if m1.rule == "absolute" else None, "m1_max_iterations": m1.max_iterations,
            "m1_ridge": m1.ridge, "m1_require_converged": m1.require_converged, "m1_gap_scope": "inner_pre_outer_batch",
            "m1_dtype": str(dtype).replace("torch.", ""),
            "m1_gap_ref_zero": g_ref, "m1_gap_start": gap_start, "m1_gap_final": gap_final, "m1_gap_relative_to_zero": rel,
            "m1_tolerance_effective": tolerance_effective, "m1_iterations": int(iterations), "m1_stop_reason": stop_reason,
            "m1_converged": bool(converged), "m1_solver_status": solver_status, "m1_hvp_calls": int(hvp_calls),
            "m1_momentum_restarts": (solver_hvp_calls - 2 - 3 * int(iterations)) if solver_status is not None else 0,
            "m1_moments_passes": int(moments_passes), "m1_fp64_diag_hvp_calls": diag_hvp, "m1_fp64_diag_moments_passes": diag_mom,
            "m1_hvp_calls_total": int(hvp_calls) + diag_hvp, "m1_moments_passes_total": int(moments_passes) + diag_mom,
            "m1_W_start_is_zero": bool(w0_zero),
            "m1_W_start_absmax": float(W0.abs().max()), "m1_W_final_absmax": float(w.detach().abs().max()),
            "m1_W_update_fro": float(dw.norm()), "m1_W_update_absmax": float(dw.abs().max()),
            "m1_fp64_diagnostic": fp64_diagnostic, "m1_W": w}


def _m1_absolute(scenes, W, weights, m1: M1Config):
    """Rule 'absolute', numerically unchanged: the package profiled_objective (its own QuadraticHead + solve_box_head +
    objective_terms: 2 moment passes) with require_converged = m1.require_converged (True by default: the strict numeric-audit mode,
    an unconverged head raises; False only when passed explicitly, for the reference rows).  G_b(0) is only known when the start W is
    exactly 0 (then it is the solver's iteration-0 gap); no extra pass is made to compute it otherwise."""
    W0 = W.detach()
    with count_head_hvp() as cnt:
        prof = profiled_objective(scenes, weights=weights, ridge=m1.ridge, initial=W0, tolerance=m1.tolerance,
                                  max_iterations=m1.max_iterations, require_converged=m1.require_converged)
    sol = prof.solution
    w0_zero = not bool(W0.any())
    gap_start = sol.history[0]["fw_gap"]
    info = _m1_info(m1, dtype=W0.dtype, W0=W0, w=sol.w, w0_zero=w0_zero, g_ref=gap_start if w0_zero else None, gap_start=gap_start,
                    gap_final=sol.fw_gap, tolerance_effective=m1.tolerance, iterations=sol.iterations,
                    stop_reason=_SOLVER_STOP[sol.status], converged=sol.converged, solver_status=sol.status, hvp_calls=cnt["hvp"],
                    solver_hvp_calls=cnt["hvp"], moments_passes=2,
                    fp64_diagnostic={"triggered": False, "reason": "rule 'absolute' (strict audit / pre-decision reference mode)"})
    return prof.terms, info


def _m1_fp64_diagnostic(scenes, weights, m1: M1Config, *, delta, W0, w, g_ref: float, gap_start: float, gap_final: float,
                        tau: float, zero: bool) -> dict:
    """Same-input fp64 diagnostic (decision 2026-10-17 2.3), diagnostic only: when g_ref is near the working-dtype resolution, the
    same feature values (cast exactly) through the same backend QuadraticHead in float64: g_ref, gap at the start and the returned W,
    and whether the stop decision of the working dtype agrees.  Costs one fp64 snapshot + up to 2 fp64 HVPs, counted here; the
    training head, W and gradients are not touched."""
    dtype = delta.dtype
    unit = m1_resolution_unit(dtype)
    below = m1.fp64_diagnostic_below
    trigger_at = None if below is None else below * unit
    base = {"rule": "g_ref <= fp64_diagnostic_below * 2 eps(dtype)", "resolution_unit": unit, "trigger_at": trigger_at}
    if dtype == torch.float64:
        return {"triggered": False, "reason": "working dtype float64", **base}
    if below is None or g_ref > trigger_at:
        return {"triggered": False, "reason": "g_ref above the trigger" if below is not None else "disabled", **base}
    with count_head_hvp() as cnt:
        head = MF.QuadraticHead([same_input_scene(s, torch.float64) for s in scenes], weights=weights, ridge=m1.ridge)
        g64 = float(head.delta.abs().sum())
        zero64 = not bool(head.delta.any())
        gs64 = g64 if not bool(W0.any()) else float(head.fw_gap(W0.to(torch.float64)))
        wf = w.detach().to(torch.float64)
        gf64 = g64 if not bool(wf.any()) else float(head.fw_gap(wf))
        delta_err = float((delta.detach().to(torch.float64) - head.delta).abs().max())
    tau64 = m1.relative_tolerance * g64
    stop_w = True if zero else bool(gap_final <= tau)
    stop_64 = True if zero64 else bool(gf64 <= tau64)
    return {"triggered": True, **base, "gap_ref_zero_fp64": g64, "zero_reference_fp64": zero64,
            "gap_ref_zero_rel_err": abs(g_ref - g64) / g64 if g64 > 0 else None, "delta_max_abs_err": delta_err,
            "gap_start_fp64": gs64, "gap_final_fp64": gf64, "gap_final_rel_err": abs(gap_final - gf64) / gf64 if gf64 > 0 else None,
            "tolerance_effective_fp64": tau64, "stop_rule_met_working_dtype": stop_w, "stop_rule_met_fp64": stop_64,
            "decisions_agree": bool(stop_w == stop_64 and zero == zero64), "hvp_calls": cnt["hvp"], "moments_passes": 1}


def _m1_relative(scenes, W, weights, m1: M1Config):
    """Decision 2026-10-17 2.1 / 2.3 (the protocol rule).  One QuadraticHead snapshot of the batch (moment pass 1) gives Delta_b and
    g_ref = G_b(0) = ||Delta_b||_1 (H 0 = 0: no HVP); tau_b = relative_tolerance * g_ref; the package solve_box_head from the warm
    start W with tolerance tau_b and the cap; the outer objective_terms at the solved W on the SAME tensors (moment pass 2).  Exactly
    zero Delta_b: W = 0 is the maximiser of the negative-semidefinite empirical head objective -> W = 0, status zero_reference_gap,
    the solver is not called (it requires a positive tolerance) and nothing is divided by g_ref."""
    W0 = W.detach()
    MF._matrix(W0, scenes[0].a, scenes[0].b, box=True)            # the warm start must be a feasible head (as solve_box_head checks)
    moments = 0
    with count_head_hvp() as cnt:
        head = MF.QuadraticHead(scenes, weights=weights, ridge=m1.ridge)
        moments += 1
        g_ref = float(head.delta.abs().sum())
        if not math.isfinite(g_ref):
            raise FloatingPointError(f"M1 zero-head reference gap is not finite: {g_ref}")
        zero = not bool(head.delta.any())
        if zero != (g_ref == 0.0):
            raise FloatingPointError("inconsistent zero-reference test")   # cannot happen: a sum of |entries| is 0 iff every entry is 0
        w0_zero = not bool(W0.any())
        if zero:
            w = torch.zeros_like(head.delta)
            gap_start = 0.0 if w0_zero else float(head.fw_gap(W0))      # the start gap is logged (1 HVP when the warm start is non-zero)
            sol, tau, iterations, gap_final, stop, converged, solver_hvp = None, None, 0, 0.0, "zero_reference_gap", True, 0
        else:
            tau = m1.relative_tolerance * g_ref
            n0 = cnt["hvp"]
            sol = MF.solve_box_head(head, initial=W0, tolerance=tau, max_iterations=m1.max_iterations)
            solver_hvp = cnt["hvp"] - n0
            w, iterations, gap_final, converged = sol.w, sol.iterations, sol.fw_gap, sol.converged
            gap_start, stop = sol.history[0]["fw_gap"], _SOLVER_STOP[sol.status]
    if m1.require_converged and not converged:
        raise RuntimeError(f"M1 head did not meet its relative gap threshold: gap={gap_final:.6g}, threshold={tau:.6g} (= "
                           f"{m1.relative_tolerance:g} x G_b(0) {g_ref:.6g}); cap {m1.max_iterations}")
    terms = objective_terms(scenes, w.detach(), weights=weights, ridge=m1.ridge)
    moments += 1
    diag = _m1_fp64_diagnostic(scenes, weights, m1, delta=head.delta, W0=W0, w=w, g_ref=g_ref, gap_start=gap_start,
                               gap_final=gap_final, tau=tau, zero=zero)
    info = _m1_info(m1, dtype=W0.dtype, W0=W0, w=w, w0_zero=w0_zero, g_ref=g_ref, gap_start=gap_start, gap_final=gap_final,
                    tolerance_effective=tau, iterations=iterations, stop_reason=stop, converged=converged,
                    solver_status=None if sol is None else sol.status, hvp_calls=cnt["hvp"], solver_hvp_calls=solver_hvp,
                    moments_passes=moments, fp64_diagnostic=diag)
    return terms, info


def mf_objective(method: str, scenes, W, weights, *, perm=None, m1: M1Config | None = None, ridge: float = 0.0,
                 dense_chunk: int | None = None, dense_checkpoint: bool = False, check_w: bool = True, check_permutations: bool = True):
    """(ObjectiveTerms, info).  perm: [n_sc, n_pos] int64 on the feature device, row s = the host permutation of scene s.
    check_w / check_permutations=False only inside an already-validated training loop (package README section 3: no repair);
    SceneBatch construction (finite, nonnegative, simplex rows) always validates; M1 always checks its start W.
    M1 info (_m1_info): the zero-head reference gap, start / final gap, relative gap, effective threshold, iterations, stop reason,
    convergence, HVP calls and moment passes of this batch (counted at the call sites: 2 = the QuadraticHead snapshot + the outer
    objective_terms; the fp64 diagnostic, when it runs, counts its own: m1_fp64_diag_*; all of it: m1_*_total), momentum restarts,
    W update and the solved W (m1_W)."""
    info: dict = {}
    if method == "MF_perm1":
        if perm is None:
            raise ValueError("MF_perm1 needs the host permutations")
        terms = sampled_permutation_terms(scenes, W, weights=weights, permutations=[perm[s][None] for s in range(len(scenes))], ridge=ridge,
                                          check_w=check_w, check_permutations=check_permutations)
    elif method == "MF_M0":
        terms = objective_terms(scenes, W, weights=weights, ridge=ridge, check_w=check_w)
    elif method == "MF_dense":
        terms = dense_objective_terms(scenes, W, weights=weights, ridge=ridge, chunk_size=dense_chunk, checkpoint_blocks=dense_checkpoint,
                                      check_w=check_w)
    elif method == "MF_M1":
        m1 = m1 or M1Config()
        terms, info = (_m1_absolute if m1.rule == "absolute" else _m1_relative)(scenes, W, weights, m1)
    else:
        raise ValueError(f"unknown MF method {method}")
    return terms, info


def mlp_objective(method: str, cr, du, dv, *, perm=None, perms=None, block: int | None = None, checkpoint: bool = False):
    """(loss, J) of the host MLP critic, T = tanh f; P = (du_i, dv_i); Q per scene; equal scene weights, as the host.
    perm [n_sc, n] (host), perms [n_sc, R, n] (row 0 = host permutation)."""
    n_sc, n, _ = du.shape
    fp = cr(torch.cat([du, dv], -1)).squeeze(-1)                                           # host G L172
    if method == "MLP_perm1":
        dvq = torch.stack([dv[s, perm[s]] for s in range(n_sc)])                            # host G L171
        fq = cr(torch.cat([du, dvq], -1)).squeeze(-1)
        loss = P.loss_fn(torch, F, "vcs", fp, fq)
        return loss, -loss
    if method == "MLP_perm8":
        R = perms.shape[1]
        dvq = torch.stack([dv[s][perms[s]] for s in range(n_sc)])                          # [n_sc, R, n, 16]
        fq = cr(torch.cat([du[:, None].expand(-1, R, -1, -1), dvq], -1)).squeeze(-1)        # [n_sc, R, n]
        loss = P.loss_fn(torch, F, "vcs", fp, fq)
        return loss, -loss
    if method == "MLP_dense":
        if block is None:                                                                   # the host's InfoNCE pair layout (G L168)
            fq = cr(torch.cat([du[:, :, None].expand(-1, -1, n, -1), dv[:, None].expand(-1, n, -1, -1)], -1)).squeeze(-1)
            loss = P.loss_fn(torch, F, "vcs", fp, fq)
            return loss, -loss

        def q_sum(du_blk, dv_all):
            f = cr(torch.cat([du_blk[:, :, None].expand(-1, -1, dv_all.shape[1], -1),
                              dv_all[:, None].expand(-1, du_blk.shape[1], -1, -1)], -1)).squeeze(-1)
            t = torch.tanh(f)
            return (-t - 0.5 * t * t).sum()
        total = fp.new_zeros(())
        for i0 in range(0, n, block):
            blk = du[:, i0:i0 + block]
            if checkpoint and torch.is_grad_enabled():
                total = total + torch.utils.checkpoint.checkpoint(q_sum, blk, dv, use_reentrant=False)
            else:
                total = total + q_sum(blk, dv)
        tp = torch.tanh(fp)
        J = (tp - 0.5 * tp * tp).mean() + total / (n_sc * n * n)
        return -J, J
    raise ValueError(f"unknown MLP method {method}")


# ------------------------------------------------------------------------------------------------------------ trainer
def grad_l2(params) -> float:
    """L2 norm (in float64) of the existing .grad tensors of params; a parameter without .grad contributes 0."""
    gs = [p.grad for p in params if p.grad is not None]
    return float(torch.sqrt(sum((g.detach().double() ** 2).sum() for g in gs))) if gs else 0.0


@dataclass
class TrainConfig:
    method: str
    seed: int
    K: int = 16                          # V10_PROTOCOL head.default_width_for_interface_smoke
    temperature: float = 1.0             # head.softmax_temperature_primary
    encoder_lr: float = 1e-3             # host G L145 (inherited)
    head_lr: float = 1e-3                # MF feature maps + W (CAL candidates 1e-4 / 3e-4 / 1e-3)
    weight_decay: float = 1e-4           # host AdamW decay on encoders (+ MF feature maps / MLP critic); W: 0 (M0_W_weight_decay)
    clip_norm: float = 1.0               # host clip over all optimiser-trained parameters (MF: incl. W for perm1 / M0)
    n_sc: int | None = None              # host min(16, len(fit))
    n_pos: int = 128
    crop: int = 160
    m1: M1Config = field(default_factory=M1Config)
    mlp_perms: int = 8
    dense_block: int | None = None
    dense_checkpoint: bool = False
    dtype: str = "float32"
    w_init: str = "zeros"                # package default (SimplexBilinearCritic); "uniform" only for perm1 / M0 checks / timing
    checks: str = "first_step"           # W-box / permutation re-checks: first step only (validated loop) or "every_step"
    grad_spot_every: int | None = None   # encoder / feature-map gradient-norm spot checks (pre-clip) on steps 0, k, 2k, ...;
                                         # None = method default (MF_M1: 50, decision 2026-10-17 2.3; other methods: 0 = off)
    absolute_reference_only: bool = False   # True only for MF_M1 reference rows under rule 'absolute' (T01 cap-binding / degenerate
                                            # case, T02 cap-binding cost references); rule 'absolute' is never the training rule


class Trainer:
    """One V10 trainer for MF_* / MLP_* on the host batch (identical crops, positions and permutations for every method)."""

    def __init__(self, cfg: TrainConfig, imgs: dict, fit: Sequence[str], device, *, build_models: Callable | None = None):
        if cfg.method not in ("MF_perm1", "MF_M0", "MF_M1") + MLP_METHODS:
            raise ValueError(f"unknown training method {cfg.method}")
        if cfg.checks not in ("first_step", "every_step"):
            raise ValueError(f"checks must be 'first_step' or 'every_step', got {cfg.checks}")
        if not isinstance(cfg.absolute_reference_only, bool):
            raise ValueError("absolute_reference_only must be a bool")
        m1_absolute = cfg.method == "MF_M1" and cfg.m1.rule == "absolute"
        if m1_absolute and not cfg.absolute_reference_only:
            raise ValueError("M1 rule 'absolute' is never the training rule (decision 2026-10-17 2.1; protocol P2 m1_rule.strict_absolute_mode): "
                             "it trains only reference rows, with TrainConfig.absolute_reference_only=True")
        if cfg.absolute_reference_only and not m1_absolute:
            raise ValueError("absolute_reference_only applies only to MF_M1 with M1 rule 'absolute'")
        if cfg.method == "MF_M1" and cfg.w_init != "zeros":
            raise ValueError("MF_M1 starts from W = 0 (decision 2026-10-17 2.1: no random non-zero initialisation); later batches "
                             "warm-start from the previous batch's W")
        spot = cfg.grad_spot_every if cfg.grad_spot_every is not None else (50 if cfg.method == "MF_M1" else 0)
        if isinstance(spot, bool) or not isinstance(spot, int) or spot < 0:
            raise ValueError(f"grad_spot_every must be None or a nonnegative integer, got {cfg.grad_spot_every!r}")
        self.grad_spot_every = spot
        self.n_steps = 0
        self.cfg, self.device, self.dtype = cfg, torch.device(device), getattr(torch, cfg.dtype)
        torch.manual_seed(cfg.seed)                                      # host G L141
        eU, eV, cr = (build_models or P.build_models)(nn)                # host init order enc(3), enc(1), critic (P L153)
        self.eU, self.eV = eU.to(self.device, self.dtype), eV.to(self.device, self.dtype)
        self.mf = cfg.method.startswith("MF_")
        enc = list(self.eU.parameters()) + list(self.eV.parameters())
        if self.mf:
            desc = self.eU[-1].out_channels if isinstance(self.eU, nn.Sequential) else DESC
            self.head = MFHead(desc, desc, cfg.K, cfg.K, temperature=cfg.temperature).to(self.device, self.dtype)   # built after the (discarded) host critic
            self.critic = None
            if cfg.w_init == "uniform":
                gen = torch.Generator(device="cpu").manual_seed(Q.seed_of(f"V10|W-init|{cfg.seed}|{cfg.K}"))
                with torch.no_grad():
                    self.head.W.copy_(torch.empty(cfg.K, cfg.K, dtype=torch.float64).uniform_(-0.9, 0.9, generator=gen))
            elif cfg.w_init != "zeros":
                raise ValueError(cfg.w_init)
            fmap = list(self.head.left.parameters()) + list(self.head.right.parameters())
            groups = [{"params": enc, "lr": cfg.encoder_lr, "weight_decay": cfg.weight_decay, "name": "encoders"},
                      {"params": fmap, "lr": cfg.head_lr, "weight_decay": cfg.weight_decay, "name": "feature_maps"}]
            if cfg.method == "MF_M1":
                self.head.W.requires_grad_(False)                        # W is solved per batch, never an optimiser parameter
            else:
                groups.append({"params": [self.head.W], "lr": cfg.head_lr, "weight_decay": 0.0, "name": "W"})
        else:
            self.head, self.critic = None, cr.to(self.device, self.dtype)
            groups = [{"params": enc + list(self.critic.parameters()), "lr": cfg.encoder_lr, "weight_decay": cfg.weight_decay, "name": "host_all"}]
        self.opt = torch.optim.AdamW(groups)
        self.clip_params = [p for g in groups for p in g["params"]]
        self.spot_groups = {"encoders": enc}                             # gradient-norm spot checks (read only, pre-clip)
        if self.mf:
            self.spot_groups["feature_maps"] = list(self.head.left.parameters()) + list(self.head.right.parameters())
        else:
            self.spot_groups["critic"] = list(self.critic.parameters())
        self.sampler = HostBatchSampler(fit, cfg.seed, n_sc=cfg.n_sc, n_pos=cfg.n_pos, crop=cfg.crop)
        self.g_extra = np.random.default_rng(Q.seed_of(f"V10|MLP-extra-perms|{cfg.seed}"))   # separate stream: host stream untouched
        self.imgs = imgs

    # -- batch
    def draw(self):
        hb = self.sampler.draw()
        bu, bv = assemble_crops(self.imgs, hb, self.cfg.crop)
        return hb, bu.to(self.dtype), bv.to(self.dtype)

    def extra_perms(self, hb: HostBatch) -> np.ndarray:
        R = self.cfg.mlp_perms
        return np.stack([np.concatenate([hb.perm[s][None], np.stack([self.g_extra.permutation(hb.perm.shape[1]) for _ in range(R - 1)])])
                         if R > 1 else hb.perm[s][None] for s in range(hb.perm.shape[0])])

    # -- objective
    def forward(self, hb: HostBatch, bu, bv, mark: Callable[[str], None] | None = None) -> tuple[torch.Tensor, dict]:
        """mark(name), if given, is called after 'descriptors' (encoders + sampling), 'scenes' (MF: feature maps + partition /
        SceneBatch validation) and 'objective'.  MF W-box / permutation checks run on the first step only unless
        cfg.checks == 'every_step' (forward() outside step() always checks: n_steps stays 0)."""
        du, dv = batch_descriptors(self.eU, self.eV, bu, bv, hb.yx)
        if mark:
            mark("descriptors")
        perm = torch.as_tensor(hb.perm, device=self.device)
        n_sc, n_pos = hb.perm.shape
        if self.mf:
            chk = self.cfg.checks == "every_step" or self.n_steps == 0
            A, B = self.head.features(du.reshape(-1, du.shape[-1]), dv.reshape(-1, dv.shape[-1]))
            scenes = mf_scenes(A, B, n_sc, n_pos)
            if mark:
                mark("scenes")
            terms, info = mf_objective(self.cfg.method, scenes, self.head.W, host_weights(n_sc), perm=perm, m1=self.cfg.m1,
                                       check_w=chk, check_permutations=chk)
            if mark:
                mark("objective")
            return -terms.f, {"terms": terms, "info": {**info, "checks_on": chk}, "A": A, "B": B, "du": du, "dv": dv, "scenes": scenes,
                              "perm": perm}
        perms = None
        if self.cfg.method == "MLP_perm8":
            perms = torch.as_tensor(self.extra_perms(hb), device=self.device)
        loss, J = mlp_objective(self.cfg.method, self.critic, du, dv, perm=perm, perms=perms, block=self.cfg.dense_block,
                                checkpoint=self.cfg.dense_checkpoint)
        if mark:
            mark("objective")
        return loss, {"J": J, "du": du, "dv": dv, "perm": perm, "perms": perms, "info": {}}

    def step(self, batch=None, *, keep=False, mark: Callable[[str], None] | None = None) -> dict:
        """One optimiser step on the next host batch (or on `batch`).  mark(name), if given, is called at the phase
        boundaries 'batch', 'descriptors', ('scenes'), 'objective', 'backward', 'update' (T02 synchronized phase timing)."""
        hb, bu, bv = batch if batch is not None else self.draw()
        if mark:
            mark("batch")
        loss, out = self.forward(hb, bu, bv, mark=mark)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if mark:
            mark("backward")
        spot = None
        if self.grad_spot_every and self.n_steps % self.grad_spot_every == 0:
            spot = {"trainer_step": self.n_steps, "every": self.grad_spot_every, "scope": "pre-clip gradients of this step",
                    **{f"{name}_grad_norm": grad_l2(params) for name, params in self.spot_groups.items()},
                    **{f"{name}_params_without_grad": sum(p.grad is None for p in params) for name, params in self.spot_groups.items()}}
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.clip_params, self.cfg.clip_norm))
        self.opt.step()
        rec = {"step": hb.step, "loss": float(loss.detach()), "grad_norm_preclip": grad_norm}
        if spot is not None:
            rec["encoder_gradient_diagnostic"] = spot
        if self.mf:
            with torch.no_grad():
                if self.cfg.method == "MF_M1":
                    self.head.W.copy_(out["info"]["m1_W"])               # warm start of the next batch; saved with the encoder
                else:
                    rec["w_absmax_preprojection"] = float(self.head.W.abs().max())
                    self.head.project_head_()
                rec["w_absmax"] = float(self.head.W.abs().max())
            rec.update({k: v for k, v in out["info"].items() if k != "m1_W"})
            rec["J"] = float(out["terms"].j.detach())
        else:
            rec["J"] = float(out["J"].detach())
        self.n_steps += 1
        if mark:
            mark("update")
        if keep:
            rec["_out"] = out
            rec["_batch"] = (hb, bu, bv)
        return rec

    def modules(self) -> tuple:
        return (self.eU, self.eV, self.head) if self.mf else (self.eU, self.eV, self.critic)


# ------------------------------------------------------------------------------------------------------------ registration (test time)
def freeze_for_test(*modules) -> None:
    for m in modules:
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)


def assert_frozen(*modules) -> None:
    for m in modules:
        if m.training or any(p.requires_grad for p in m.parameters()):
            raise ValueError("test-time energy needs fixed models: eval() and requires_grad_(False) on every parameter")


def mf_energy_rows(A, B, W, estimator: str = "moments", *, perm=None, perms=None, dense_block: int = 512, return_terms: bool = False):
    """Batched fixed-W MF energies of C candidates of one registration.  A [N, r] reference side (fixed), B [C, N, s] moving
    side per candidate, W [r, s].  One scene, weight 1, V mode: J = mean(t - t^2/2) - q1 - q2/2 with
    moments  q1 = u^T W v / N^2, q2 = <W, G_A W G_B> / N^2;  perm1: the fixed permutation;  perms: R fixed permutations;
    dense: all N^2 pairs (row blocks; validation only).  Same object and normalisation as mf_vcs (validated in T01)."""
    N = A.shape[0]
    AW = A @ W
    t = (AW[None] * B).sum(-1)                                            # [C, N]
    p1, p2 = t.mean(-1), t.square().mean(-1)
    if estimator == "moments":
        q1 = (B.sum(1) @ (W.T @ A.sum(0))) / (N * N)
        GAW = (A.T @ A) @ W
        q2 = ((GAW[None] @ (B.transpose(1, 2) @ B)) * W[None]).sum((-2, -1)) / (N * N)
    elif estimator == "perm1":
        neg = (AW[None] * B[:, perm]).sum(-1)
        q1, q2 = neg.mean(-1), neg.square().mean(-1)
    elif estimator == "perms":
        neg = (AW[None, None] * B[:, perms]).sum(-1)                      # [C, R, N]
        q1, q2 = neg.mean((-2, -1)), neg.square().mean((-2, -1))
    elif estimator == "dense":
        q1s, q2s = [], []
        for c in range(B.shape[0]):
            s1 = s2 = B.new_zeros(())
            for i0 in range(0, N, dense_block):
                sc = AW[i0:i0 + dense_block] @ B[c].T
                s1, s2 = s1 + sc.sum(), s2 + sc.square().sum()
            q1s.append(s1 / (N * N))
            q2s.append(s2 / (N * N))
        q1, q2 = torch.stack(q1s), torch.stack(q2s)
    else:
        raise ValueError(f"unknown estimator {estimator}")
    j = p1 - q1 - 0.5 * (p2 + q2)
    return (j, {"p1": p1, "p2": p2, "q1": q1, "q2": q2}) if return_terms else j


class MFRigidEnergy(Q.RigidEnergy):
    """The host's rigid registration energy (Q.RigidEnergy, kind 'critic': same RNG consumption -> same x and permutation for
    every method, BOX1 / BOX2, exact batch() and grid_values()) with the fixed-W MF energy in j_critic.
    dtype float64 casts x, boxes, reference descriptors and the moving image (validation only; the host hard-codes float32)."""

    def __init__(self, ref, mov, rng, *, eV, head: MFHead, estimator: str = "moments", dtype=torch.float32, perms=None,
                 dense_block: int = 512, record: bool = False):
        assert_frozen(eV, head)
        super().__init__(torch, F, "critic", ref.to(torch.float32), mov, rng, eV=eV, critic=None)
        if dtype != torch.float32:
            self.x, self.box1, self.box2 = self.x.to(dtype), self.box1.to(dtype), self.box2.to(dtype)
            self.u = P.sample_map(torch, F, ref.to(dtype), self.x)
            self.mov = mov.to(dtype)
        self.head, self.estimator, self.perms, self.dense_block, self.record = head, estimator, perms, dense_block, record
        self.A = head.features_a(self.u)                                  # [N, K]: reference side, constant per registration
        self.W = head.W
        self.last: dict | None = None

    def j_critic(self, dv):                                              # dv [C, N, 16] -> [C]
        B = self.head.features_b(dv)
        out = mf_energy_rows(self.A, B, self.W, self.estimator, perm=self.perm, perms=self.perms, dense_block=self.dense_block,
                             return_terms=self.record)
        if self.record:
            self.last = {"dv": dv, "B": B, "terms": out[1], "J": out[0]}
            return out[0]
        return out


def host_moving_descriptors(energy: Q.RigidEnergy, p):
    """The exact path of Q.RigidEnergy.batch for one candidate p = (u, ty, tx) (Q L194-197, L213-214) -> dv [N, 16]."""
    th, t = p[0:1] / Q.RHO, p[1:3][None]
    n = Q.BOX1[1] - Q.BOX1[0] + 1
    Wk = Q.sample_pts(torch, F, energy.mov, Q.apply_T(torch, th, t, energy.box1))[0, :, 0].reshape(1, 1, n, n)
    return P.sample_map(torch, F, energy.kw["eV"](Wk)[0], energy.x - Q.BOX1[0])


def reference_descriptors(eU, inputs: Q.Inputs, core: str, dtype=torch.float32):
    """Host Q L338-342: BF RGB reference encoded once per core per checkpoint (no geometry dependence)."""
    with torch.no_grad():
        return eU(inputs.reference(core)[1][None].to(dtype))[0]


class TimedEnergy:
    """Proxy for Q.optimise_rigid: accumulates synchronized wall time of the grid stage; counts exact-energy calls."""

    def __init__(self, energy, device):
        self.energy, self.device = energy, torch.device(device)
        self.grid_seconds, self.grid_calls, self.exact_calls = 0.0, 0, 0

    def grid_values(self, th_deg, ts):
        synchronize(self.device)
        t0 = time.perf_counter()
        out = self.energy.grid_values(th_deg, ts)
        synchronize(self.device)
        self.grid_seconds += time.perf_counter() - t0
        self.grid_calls += 1
        return out

    def __call__(self, p):
        self.exact_calls += 1
        return self.energy(p)


def register_timed(energy, device) -> dict:
    """One registration with the unchanged host searcher (Q.optimise_rigid: 21 x 508 grid, top 3, 100 Adam steps lr 0.5).
    Timing record only: no truth-dependent quantity (error, F_true) is computed."""
    te = TimedEnergy(energy, device)
    synchronize(device)
    t0 = time.perf_counter()
    th, t, n_grid, f_hat = Q.optimise_rigid(torch, te, torch.device(device))
    synchronize(device)
    total = time.perf_counter() - t0
    return {"seconds": total, "grid_seconds": te.grid_seconds, "refine_and_final_seconds": total - te.grid_seconds,
            "grid_calls": te.grid_calls, "exact_energy_calls": te.exact_calls, "n_grid": int(n_grid), "F_hat": float(f_hat),
            "theta_hat_deg": float(th), "t_hat": [float(x) for x in t]}


def registration_record(energy, device, truth) -> dict:
    """One registration with the unchanged host searcher and the host's per-registration record (Q.register_images L256-259):
    theta, t, theta_hat, t_hat, err_full (Q.corner_error_full: mean corner error of the central 834-px patch, full-resolution
    px), rot_err_deg, trans_err_full, F_hat, F_true (exact energy at the truth), n_grid, seconds.  truth = (theta_deg, ty, tx).
    For T05; phase 1 only unit-tests it (an analytic energy through the real searcher) and never calls it on data."""
    thd, ty, tx = truth
    t0 = time.time()
    th, t, n_grid, f_hat = Q.optimise_rigid(torch, energy, torch.device(device))
    sec = time.time() - t0
    return {"theta": thd, "t": [ty, tx], "theta_hat": th, "t_hat": t.tolist(), "err_full": Q.corner_error_full(th, t, thd, (ty, tx)),
            "rot_err_deg": abs(th - thd), "trans_err_full": float(np.hypot(*(t - np.array([ty, tx])))) * P.DOWN, "F_hat": f_hat,
            "F_true": energy.value(thd, (ty, tx)), "n_grid": n_grid, "seconds": sec}


# ------------------------------------------------------------------------------------------------------------ state, checkpoints
def state_hash(*modules) -> str:
    """sha256 over (module index, name, dtype, shape, bytes) of every parameter and buffer."""
    h = hashlib.sha256()
    for i, m in enumerate(modules):
        for name, t in m.state_dict().items():
            a = t.detach().to("cpu").contiguous()
            h.update(f"{i}|{name}|{a.dtype}|{tuple(a.shape)}|".encode())
            h.update(a.numpy().tobytes())
    return h.hexdigest()


PAIR_FORMAT = "V10-MF-pair-v1"


def save_pair(path, eU, eV, head: MFHead, meta: dict) -> str:
    """Indivisible (encoders, feature maps, W) checkpoint; meta must be plain JSON types (weights_only loading)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": PAIR_FORMAT, "eU": eU.state_dict(), "eV": eV.state_dict(), "mf_head": head.state_dict(),
                "mf_head_cfg": {"x_dim": head.left.in_features, "y_dim": head.right.in_features, "r": head.left.out_features,
                                "s": head.right.out_features, "temperature": float(head.temperature)}, "meta": meta}, path)
    return sha256_file(path)


def load_pair(path, device, dtype=torch.float32, build_models: Callable | None = None):
    st = torch.load(path, map_location=torch.device(device), weights_only=True)
    if st.get("format") != PAIR_FORMAT:
        raise ValueError(f"{path}: not a {PAIR_FORMAT} checkpoint")
    eU, eV, _ = (build_models or P.build_models)(nn)
    c = st["mf_head_cfg"]
    head = MFHead(c["x_dim"], c["y_dim"], c["r"], c["s"], temperature=c["temperature"])
    for m, key in ((eU, "eU"), (eV, "eV"), (head, "mf_head")):
        m.to(torch.device(device), dtype)
        m.load_state_dict(st[key])
    return eU, eV, head, st["meta"]


# ------------------------------------------------------------------------------------------------------------ comparisons
def compare(x, y, dtype_name: str) -> dict:
    """allclose rule of the package (V10_DESIGN 6.1): |x - y| <= atol + rtol |y| elementwise, y = reference."""
    atol, rtol = TOLERANCES[dtype_name]
    x = torch.as_tensor(x).detach().to("cpu", torch.float64)
    y = torch.as_tensor(y).detach().to("cpu", torch.float64)
    diff = (x - y).abs()
    bound = atol + rtol * y.abs()
    finite = bool(torch.isfinite(x).all() and torch.isfinite(y).all())
    ref_max = float(y.abs().max()) if y.numel() else 0.0
    return {"allclose": bool(finite and (diff <= bound).all()), "max_abs_err": float(diff.max()) if diff.numel() else 0.0,
            "max_rel_err": float((diff / y.abs().clamp_min(1e-300)).max()) if diff.numel() else 0.0,
            "worst_err_over_bound": float((diff / bound).max()) if diff.numel() else 0.0,
            "ref_max_abs": ref_max, "numel": int(y.numel()), "atol": atol, "rtol": rtol, "finite": finite,
            "ref_over_atol": ref_max / atol}       # power: a reference at or below atol cannot reveal a wrong term (low power)


def compare_many(xs: Sequence, ys: Sequence, dtype_name: str) -> dict:
    parts = [compare(x, y, dtype_name) for x, y in zip(xs, ys)]
    return {"allclose": all(p["allclose"] for p in parts), "max_abs_err": max(p["max_abs_err"] for p in parts),
            "max_rel_err": max(p["max_rel_err"] for p in parts), "worst_err_over_bound": max(p["worst_err_over_bound"] for p in parts),
            "ref_max_abs": max(p["ref_max_abs"] for p in parts), "numel": sum(p["numel"] for p in parts),
            "atol": parts[0]["atol"], "rtol": parts[0]["rtol"], "finite": all(p["finite"] for p in parts), "n_tensors": len(parts),
            "ref_over_atol_min": min(p["ref_over_atol"] for p in parts)}
