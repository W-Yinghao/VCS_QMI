"""V10 T03 (owner decision 2026-10-17 section 9): shared constants, RNG stream keys, provenance and the truth barrier.

T03 code lives OUTSIDE src/v10 and scripts/v10 so the BF measurement identity (code hash of src/v10 + scripts/v10) is untouched.
T03 identities (same formula as src/v10/adapter.py code_hash: sha256 over '<sha256>  <path>' lines of the tracked files, paths sorted
bytewise, first 16 hex):
  T03 code hash = src/v10 (imported backend + adapter) + scripts/v10_t03 (this measurement code)
  T03 job pin   = the code hash paths + slurm/v10_t03 + the T03 inputs read at run time (v10/protocol/T03_UNITS.csv, the package
                  generator synthetic/src/v10_synthetic.py and its checker synthetic/tests/check_synthetic.py)
Analysis tools (v10/tools/t03_aggregate.py), docs and results are not hashed (the BF convention for v10/tools).

Truth barrier (decision 9.2 "truth 仅交 evaluator；训练和 CAL 选择只读样本"): a process-wide audit hook that, while a
truth_barrier() block is active, refuses (TruthAccessError) every open / link / rename / copy / symlink of a path with an
'evaluator_only' component (the package truth directory and the T03 evaluation outputs) and of the EVAL sample directory of a T03
case, plus every process creation.  Training and CAL selection run inside the barrier; the evaluator refuses to run inside it.
"""
from __future__ import annotations

import contextlib
import gzip
import hashlib
import json
import math
import os
import re
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ------------------------------------------------------------------------------------------------------------ paths and pins
PACKAGE = REPO / "v10" / "package_v10_server_v1_20261016"
GENERATOR_PATH = PACKAGE / "synthetic" / "src" / "v10_synthetic.py"
GENERATOR_SHA256 = "3c5d7d22e129eea703bf0075f7ee660e65c77ca6c55ef6173b96199efd37e73c"     # protocol P2 t03.generator.sha256 = package manifest
CHECKER_PATH = PACKAGE / "synthetic" / "tests" / "check_synthetic.py"
CHECKER_SHA256 = "15b82a14a23f927cb851ae455584c4ffc72b5aea0daea13342ab3e5ce884bd3c"       # package MANIFEST_SHA256.json
PROTOCOL_PATH = REPO / "v10" / "protocol" / "V10_PROTOCOL_P2_20261017.json"
UNITS_CSV = REPO / "v10" / "protocol" / "T03_UNITS.csv"
UNITS_CSV_SHA256 = "fa05019200f6f7ecb40c3467e4504ade0a5583c5fcfaa9f327d0f3c6ea0d86a2"     # protocol P2 t03.units_file.sha256
OUT_ROOT_DEFAULT = "/projects/EEG-foundation-model/yinghao/outputs_v10/T03"
T03_CODE_PATHS = ("src/v10", "scripts/v10_t03")
T03_PIN_PATHS = T03_CODE_PATHS + ("slurm/v10_t03", "v10/protocol/T03_UNITS.csv",
                                  "v10/package_v10_server_v1_20261016/synthetic/src/v10_synthetic.py",
                                  "v10/package_v10_server_v1_20261016/synthetic/tests/check_synthetic.py")
BF_CODE_PATHS = ("src/v10", "scripts/v10")                       # the BF measurement identity (recorded, never changed here)
DIRTY_PATHS = T03_PIN_PATHS + ("tests/v10_t03", "src/v9/reg")    # = the dirty check of slurm/v10_t03/t03_header.sh
PIN_ENV, COMMIT_ENV, OUT_ENV = "V10_T03_EXPECT_CODE", "V10_T03_SUBMIT_COMMIT", "V10_T03_OUT_ROOT"


def out_root() -> Path:
    return Path(os.environ.get(OUT_ENV, OUT_ROOT_DEFAULT))


# ------------------------------------------------------------------------------------------------------------ protocol constants
# Decision 2026-10-17 section 9 (contract key "T03"); tests/v10_t03 checks every value against V10_PROTOCOL_P2_20261017.json.
FAMILIES = ("two_block", "diagonal_mixture")
MS = (8, 32)
RHOS = (0.0, 0.25, 0.75)
ROLE_SIZES = {"fit": 4096, "cal": 2048, "eval": 8192}
METHODS = ("MF_VCS_perm1", "MF_VCS_M0", "MF_VCS_M1", "MLP_VCS_perm1", "MLP_VCS_perm8")
MF_METHODS = METHODS[:3]
MLP_METHODS = METHODS[3:]
K_MAIN, K_CAPACITY = 16, (8, 32)
TEMPERATURE = 1.0
MLP_HIDDEN = 64
PERM8_R = 8
STEPS, CKPT_EVERY = 2000, 250
LEARNING_RATES = (1e-4, 3e-4, 1e-3)
M1_CAPS = (10, 30, 100)
NON_W_WEIGHT_DECAY, W_WEIGHT_DECAY, RIDGE, CLIP_NORM = 1e-4, 0.0, 0.0, 1.0
TUNING_SEED = 1017
MAIN_DATA_SEEDS = tuple(range(2026101700, 2026101710))
MAIN_ALGO_SEED, EXTRA_ALGO_SEEDS = 1001, (1002, 1003)
SMOKE_DATA_SEED = 20261016                                       # landing choice (protocol landing_choices "T03 smoke"), not an owner decision
CAPACITY_CONDITIONS = (("two_block", 32, 0.25), ("two_block", 32, 0.75), ("diagonal_mixture", 32, 0.25), ("diagonal_mixture", 32, 0.75))
IDENTITY_ATOL = 2e-13                                            # S - J_pop(T) = E_M(T-eta)^2 in float64: the package check's tolerance
TOLERANCES = {"float64": (1e-10, 1e-8), "float32": (1e-5, 1e-4)}  # package V10_DESIGN 6.1 (atol, rtol); never loosened

# ------------------------------------------------------------------------------------------------------------ RNG streams
# Named numpy SeedSequence streams (as the package generator): no Python hash(), no order-dependent spawn().  Algorithm streams are
# keyed by (algo seed, condition) and never by the data root seed: with the algorithm seed fixed, every data replicate sees the same
# initial state and the same permutation index sequence (decision 9.3: data variance with the algorithm seed held at 1001).
STREAM_SCHEME_VERSION = 1
STREAMS = {"init": 1, "perm": 2, "perm8_extra": 3, "diag_perm": 4}
FAMILY_CODE = {"two_block": 11, "diagonal_mixture": 23}          # = v10_synthetic.FAMILIES (checked in tests)
ROLE_CODE = {"fit": 101, "cal": 211, "eval": 307}                # = v10_synthetic.ROLES (checked in tests)
CRITIC_CODE = {"MF": 1, "MLP": 2}


def rho_words(rho: float) -> tuple[int, int]:
    rho = 0.0 if float(rho) == 0.0 else float(rho)
    return struct.unpack(">II", struct.pack(">d", rho))


def stream(root: int, name: str, family: str, m: int, rho: float, *extra: int) -> np.random.SeedSequence:
    if isinstance(root, bool) or int(root) != root or root < 0:
        raise ValueError("stream root must be a nonnegative integer")
    hi, lo = rho_words(rho)
    key = (STREAM_SCHEME_VERSION, STREAMS[name], FAMILY_CODE[family], int(m), hi, lo, *(int(e) for e in extra))
    return np.random.SeedSequence(int(root), spawn_key=key)


def stream_rng(ss: np.random.SeedSequence) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(ss))


def stream_seed64(ss: np.random.SeedSequence) -> int:
    return int(ss.generate_state(1, dtype=np.uint64)[0])


def stream_record(ss: np.random.SeedSequence) -> dict:
    return {"entropy": int(ss.entropy), "spawn_key": [int(k) for k in ss.spawn_key],
            "state_fingerprint_uint32": [int(v) for v in ss.generate_state(4)]}


def case_id(family: str, m: int, rho: float) -> str:
    """= v10_synthetic.case_id (checked in tests)."""
    rho = 0.0 if float(rho) == 0.0 else float(rho)
    return f"{family}__m{int(m):03d}__rho{format(rho, '.17g').replace('.', 'p')}"


# ------------------------------------------------------------------------------------------------------------ truth barrier
CASE_DIR_RE = re.compile(r"^(two_block|diagonal_mixture)__m\d{3}__rho[0-9a-z+-]+$")
TRUTH_DIR_NAME = "evaluator_only"
_PROCESS_EVENTS = ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty", "pty.spawn")
_PATH_EVENTS = {"open": (0,), "os.link": (0, 1), "os.rename": (0, 1), "os.symlink": (0, 1), "shutil.copyfile": (0, 1),
                "shutil.copytree": (0, 1), "shutil.move": (0, 1)}
# one barrier state per process, shared by every copy of this module (the hook installed first consults the same dict)
_BARRIER: dict[str, Any] = getattr(sys, "_v10_t03_barrier", None) or {"depth": 0, "allow_eval": True, "phase": None, "denied": []}
sys._v10_t03_barrier = _BARRIER


class TruthAccessError(PermissionError):
    """Raised when training / CAL code touches evaluator-only material while the truth barrier is active."""


def forbidden_reason(path, *, allow_eval: bool) -> str | None:
    """'evaluator_only' for any path with that component (raw or after resolving symlinks); 'EVAL samples' for <case>/eval/...
    of a T03 case directory unless allow_eval; else None."""
    try:
        raw = os.fsdecode(path)
    except TypeError:
        return None
    for candidate in {os.path.abspath(raw), os.path.realpath(raw)}:
        parts = Path(candidate).parts
        if TRUTH_DIR_NAME in parts:
            return TRUTH_DIR_NAME
        if not allow_eval:
            for i in range(len(parts) - 1):
                if CASE_DIR_RE.match(parts[i]) and parts[i + 1] == "eval":
                    return "EVAL samples"
    return None


def _audit(event: str, args: tuple) -> None:
    if not _BARRIER["depth"]:
        return
    if event in _PATH_EVENTS:
        for i in _PATH_EVENTS[event]:
            if i < len(args) and isinstance(args[i], (str, bytes, os.PathLike)):
                reason = forbidden_reason(args[i], allow_eval=_BARRIER["allow_eval"])
                if reason:
                    _BARRIER["denied"].append({"event": event, "path": os.fsdecode(args[i]), "reason": reason, "phase": _BARRIER["phase"]})
                    raise TruthAccessError(f"T03 truth barrier ({_BARRIER['phase']}): {event} of {os.fsdecode(args[i])} refused ({reason} "
                                           "is evaluator-only; training and CAL selection read FIT / CAL samples only)")
    elif event.startswith(_PROCESS_EVENTS):
        _BARRIER["denied"].append({"event": event, "path": None, "reason": "process creation", "phase": _BARRIER["phase"]})
        raise TruthAccessError(f"T03 truth barrier ({_BARRIER['phase']}): {event} refused (no process creation inside training / CAL)")


if not getattr(sys, "_v10_t03_audit_installed", False):         # one permanent hook per process (audit hooks cannot be removed)
    sys.addaudithook(_audit)
    sys._v10_t03_audit_installed = True


@contextlib.contextmanager
def truth_barrier(phase: str, *, allow_eval: bool = False):
    """Training / CAL phase: evaluator-only paths (and EVAL samples unless allow_eval) cannot be opened.  Nesting keeps the strictest
    setting of the outer block."""
    prev = (_BARRIER["allow_eval"], _BARRIER["phase"])
    _BARRIER["allow_eval"] = (prev[0] and allow_eval) if _BARRIER["depth"] else allow_eval
    _BARRIER["phase"] = phase
    _BARRIER["depth"] += 1
    try:
        yield _BARRIER
    finally:
        _BARRIER["depth"] -= 1
        _BARRIER["allow_eval"], _BARRIER["phase"] = prev if _BARRIER["depth"] else (True, None)


def barrier_active() -> bool:
    return bool(_BARRIER["depth"])


GUARDED_GENERATOR_API = ("exact_truth", "load_evaluation_truth")     # the package generator's analytic / file truth APIs


def guard_truth_api(mod):
    """Wrap the generator's truth APIs in memory (idempotent) so that they raise TruthAccessError inside the truth barrier."""
    for name in GUARDED_GENERATOR_API:
        fn = getattr(mod, name)
        if getattr(fn, "v10_t03_guarded", False):
            continue

        def guarded(*args, _fn=fn, **kwargs):
            if barrier_active():
                raise TruthAccessError(f"v10_synthetic.{_fn.__name__} is evaluator-only; refused inside the truth barrier")
            return _fn(*args, **kwargs)
        guarded.__name__, guarded.__doc__, guarded.v10_t03_guarded = fn.__name__, fn.__doc__, True
        setattr(mod, name, guarded)
    return mod


def barrier_denials() -> list[dict]:
    return list(_BARRIER["denied"])


# ------------------------------------------------------------------------------------------------------------ provenance
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args, repo: Path = REPO) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else f"<git error {r.returncode}: {r.stderr.strip()[:200]}>"


def tracked(paths: Sequence[str], repo: Path = REPO) -> list[str]:
    out = git("ls-files", *paths, repo=repo)
    if out.startswith("<git error"):
        raise RuntimeError(out)
    return sorted(f for f in out.split("\n") if f)


def tree_hash(paths: Sequence[str], repo: Path = REPO) -> str:
    """= adapter.code_hash for the given paths; shell: git ls-files <paths> | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16."""
    files = sorted(tracked(paths, repo), key=lambda s: s.encode())
    lines = "".join(f"{sha256_file(repo / f)}  {f}\n" for f in files)
    return hashlib.sha256(lines.encode()).hexdigest()[:16]


def check_job_pin() -> dict:
    """Jobs run the code pinned at submission (slurm/v10_t03/submit.sh exports V10_T03_EXPECT_CODE = tree_hash(T03_PIN_PATHS)).
    Refuses (SystemExit) when the pin is set and differs; an unset pin (interactive use, unit tests) is recorded as 'unset'."""
    expected, actual = os.environ.get(PIN_ENV), tree_hash(T03_PIN_PATHS)
    if expected and expected != actual:
        raise SystemExit(f"[v10_t03] REFUSED: job pin {actual} of {' '.join(T03_PIN_PATHS)} != pinned {expected} (changed after submission)")
    return {"expected": expected, "actual": actual, "status": "match" if expected else "unset"}


def verify_pinned_file(path: Path, expect_sha256: str, label: str) -> str:
    digest = sha256_file(path)
    if digest != expect_sha256:
        raise RuntimeError(f"{label} {path} has sha256 {digest}, expected the pinned {expect_sha256}; refusing to run")
    return digest


def identity() -> dict:
    """Code identity of this process (read outside the truth barrier: it runs git)."""
    return {"t03_code_hash": tree_hash(T03_CODE_PATHS), "t03_code_paths": list(T03_CODE_PATHS),
            "t03_job_pin": tree_hash(T03_PIN_PATHS), "t03_pin_paths": list(T03_PIN_PATHS),
            "bf_code_hash_src_scripts_v10": tree_hash(BF_CODE_PATHS), "git_commit": git("rev-parse", "HEAD"),
            "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"), "git_dirty": git("status", "--porcelain", "--", *DIRTY_PATHS),
            "submit_commit": os.environ.get(COMMIT_ENV), "job_pin_expected": os.environ.get(PIN_ENV),
            "generator_sha256": sha256_file(GENERATOR_PATH), "checker_sha256": sha256_file(CHECKER_PATH),
            "units_csv_sha256": sha256_file(UNITS_CSV)}


def cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def environment(device: str = "cpu") -> dict:
    """Identity + the decision-3 precision record (src/v10/adapter.precision_record, read only) + host / SLURM / versions."""
    import torch
    from v10 import adapter as A
    rec = {"utc": utc_now(), "host": socket.gethostname(), "python": sys.version.split()[0], "executable": sys.executable,
           "numpy": np.__version__, "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
           "torch_interop_threads": torch.get_num_interop_threads(), "cpu_model": cpu_model(), "os_cpu_count": os.cpu_count(),
           "sched_affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
           "env_threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "SLURM_CPUS_PER_TASK")},
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
           "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST"), "device": device,
           "host_code_hash_v9_reg": A.host_code_hash(), "mf_vcs_sha256": A.sha256_file(A.MF.__file__),
           "precision": A.precision_record(device), **identity()}
    return rec


# ------------------------------------------------------------------------------------------------------------ JSON
def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def jsonable(o):
    """Recursive conversion for json.dumps(allow_nan=False): numpy scalars / arrays, Paths, tuples; non-finite floats -> None."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "detach") and hasattr(o, "cpu"):                 # torch tensor
        return jsonable(o.detach().cpu().tolist())
    if o is None or isinstance(o, (str, int, bool)):
        return o
    if type(o).__module__ == "torch" and type(o).__name__ == "dtype":
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def dumps(obj) -> str:
    return json.dumps(jsonable(obj), indent=1, allow_nan=False, ensure_ascii=False) + "\n"


def write_json_atomic(path, obj) -> str:
    """Write, fsync-free atomic replace; returns the sha256 of the written bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dumps(obj).encode("utf-8")
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return sha256_bytes(data)


def write_json_gz_atomic(path, obj) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = gzip.compress(dumps(obj).encode("utf-8"), mtime=0)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return sha256_bytes(data)


def read_json(path):
    path = Path(path)
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


def fmt_lr(lr: float) -> str:
    return {1e-4: "1e-4", 3e-4: "3e-4", 1e-3: "1e-3"}.get(float(lr), repr(float(lr)))


def chunks(seq: Iterable, n: int):
    buf = []
    for x in seq:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf
