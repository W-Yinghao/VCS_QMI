"""V10 TEST_REPEAT134: frozen records, pins, the TEST_REPEAT data guard and the DEV-gate comparator.

Standard library only (no Torch, no image access), so the CPU tests import it directly.  Nothing here reads a TEST image or
label: TestRepeatGuard can only be constructed inside a Slurm job whose submission pinned the activation record, and it only
grants reads of the locked, prepared TEST_REPEAT inputs for the 134 cores of the frozen manifest.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CS = Path("/home/infres/yinwang/CS_QMI")
HAND = CS / "handoff"
BUNDLE = Path(os.environ.get("V10_TR_BUNDLE", str(HAND / "v10_test_repeat134_20261020")))
OUT = Path(os.environ.get("V10_TR_OUT", "/projects/EEG-foundation-model/yinghao/outputs_v10/test_repeat134_20261020"))
RECORDS = "v10/test_repeat134"                       # committed copies of the frozen records (repo-relative)
DECISION, MANIFEST = "OWNER_DECISION.json", "TEST_REPEAT134_MANIFEST.json"
V1_PROTOCOL, V1_ACTIVATION = "TEST_REPEAT134_PROTOCOL.json", "TEST_REPEAT134_ACTIVATION.json"     # v1 (a85cc21b), unchanged
V2_PROTOCOL, V2_ACTIVATION = "TEST_REPEAT134_PROTOCOL_V2.json", "TEST_REPEAT134_ACTIVATION_V2.json"  # v2 (ca9f517c), unchanged
V2_AMENDMENT, V2_DELTA = "TEST_REPEAT134_GATE_AMENDMENT_V2.json", "TEST_REPEAT134_DELTA_REVIEW_V2.json"
DECISION_V4 = "OWNER_DECISION_V4.json"
PROTOCOL, ACTIVATION = "TEST_REPEAT134_PROTOCOL_V4.json", "TEST_REPEAT134_ACTIVATION_V4.json"     # current version (v4)
AMENDMENT, DELTA = "TEST_REPEAT134_AMENDMENT_V4.json", "TEST_REPEAT134_DELTA_REVIEW_V4.json"
FROZEN_RECORDS = (DECISION, DECISION_V4, MANIFEST, PROTOCOL, AMENDMENT, DELTA, V2_PROTOCOL, V2_ACTIVATION, V2_AMENDMENT,
                  V2_DELTA, V1_PROTOCOL, V1_ACTIVATION)                                         # hashed into ACTIVATION
IN_JOB = {"count_abs": 12, "median_abs": 0.5}        # v4 (f): coarse gross-error band of the in-job DEV check
TOLERANCE_GPU_TYPES = ("RTX6000PRO", "H100")         # v2: bitwise PASS, else PASS_TOLERANCE (a)-(d), else FAIL
TOLERANCE = {"count_abs": 3, "median_abs": 0.25, "median_of_row_abs": 0.25}   # v2 (b), (c), (d)
PASSING = {"L40S": ("PASS",), "RTX6000PRO": ("PASS", "PASS_TOLERANCE"), "H100": ("PASS", "PASS_TOLERANCE")}
ROLE = "TEST_REPEAT"
CODE_FILES = ("scripts/v10_test_repeat/tr_common.py", "scripts/v10_test_repeat/evaluate.py",
              "scripts/v10_test_repeat/prepare_inputs.py", "slurm/v10_test_repeat/gpu.sbatch")
# Fields that measure time or name the job; every other field of a registration row must agree bitwise in the DEV gate.
TIMING_FIELDS = ("seconds", "preparation_seconds", "reference_preparation_seconds",
                 "moving_and_energy_preparation_seconds", "full_registration_seconds", "job")
GPU_TYPES = {"NVIDIA L40S": "L40S", "NVIDIA RTX PRO 6000 Blackwell Server Edition": "RTX6000PRO", "NVIDIA H100 NVL": "H100"}
EXCLUDE_NODES = "node51,node53,node60"


# ------------------------------------------------------------------------------------------------------------- records
def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canonical(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read(path):
    return json.loads(Path(path).read_text())


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1, allow_nan=False) + "\n"


def write(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(dumps(obj))
    os.replace(tmp, path)


def write_once(path, obj) -> None:
    """Results are never overwritten: an existing file must already hold exactly this content."""
    path = Path(path)
    if path.exists():
        require(path.read_text() == dumps(obj), f"{path} exists with different content; results are never overwritten")
        return
    write(path, obj)


def append(path, row) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def lines(path) -> list:
    p = Path(path)
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()] if p.exists() else []


@contextlib.contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"{path} is held by another process")
        yield


def require(cond, msg: str) -> None:
    if not cond:
        raise RuntimeError("TEST_REPEAT134 refused: " + msg)


def metrics(rows) -> dict:
    """The frozen DEV metric (BF1 common.metrics == supplement records.metrics): err_full in full-resolution px."""
    errs = [float(r["err_full"]) for r in rows]
    require(errs and all(math.isfinite(x) and x >= 0 for x in errs), "empty or non-finite registration errors")
    return {"n": len(errs), "success42": sum(x < 42 for x in errs) / len(errs),
            "success9": sum(x < 9 for x in errs) / len(errs), "median_err": statistics.median(errs)}


def git(*args, cwd=REPO) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def committed_bytes(rel: str) -> bytes:
    return subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=REPO, capture_output=True, check=True).stdout


# ---------------------------------------------------------------------------------------------------------- frozen pins
def records(expect_activation: str | None = None) -> dict:
    """The four frozen records, read from the bundle; each equals its committed repo copy at HEAD, the activation lists the
    sha256 of the other three, and (in a job) the activation sha256 equals the value pinned at submission."""
    out = {}
    for name in FROZEN_RECORDS + (ACTIVATION,):
        b = (BUNDLE / name).read_bytes()
        require(b == (REPO / RECORDS / name).read_bytes() == committed_bytes(f"{RECORDS}/{name}"),
                f"{name}: bundle copy, working tree and HEAD differ")
        out[name] = (b, json.loads(b))
    act_b, act = out[ACTIVATION]
    if expect_activation is not None:
        require(sha_bytes(act_b) == expect_activation, "activation record differs from the submission pin")
    for name in FROZEN_RECORDS:
        require(act["records"][name] == sha_bytes(out[name][0]), f"activation does not pin the current {name}")
    return {"decision": out[DECISION][1], "decision_v4": out[DECISION_V4][1], "manifest": out[MANIFEST][1],
            "protocol": out[PROTOCOL][1], "amendment": out[AMENDMENT][1], "delta": out[DELTA][1], "activation": act,
            "activation_sha256": sha_bytes(act_b),
            "shas": {name: sha_bytes(v[0]) for name, v in out.items()}}


def verify_code(activation: dict, expect_commit: str | None) -> dict:
    """The TEST-stage code is exactly the code the activation pins, committed and clean."""
    for rel, value in activation["code"]["files"].items():
        require(sha(REPO / rel) == value and sha_bytes(committed_bytes(rel)) == value, f"TEST-stage code changed: {rel}")
    require(git("status", "--porcelain", "--", *activation["code"]["files"]) == "", "TEST-stage code has local changes")
    head = git("rev-parse", "HEAD")
    if expect_commit is not None:
        require(head == expect_commit, f"repository HEAD {head[:10]} != submission pin {expect_commit[:10]}")
    return {"commit": head, "files": activation["code"]["files"]}


def require_slurm() -> str:
    job = os.environ.get("SLURM_JOB_ID")
    require(job, "TEST_REPEAT stages run only inside Slurm")
    return job


def unit(manifest: dict, key: str) -> dict:
    hits = [u for u in manifest["units"] if u["key"] == key]
    require(len(hits) == 1, f"unknown unit {key}")
    return hits[0]


def gate_spec(protocol: dict, gate_id: str) -> dict:
    hits = [g for g in protocol["dev_gate"]["plan"] if g["gate_id"] == gate_id]
    require(len(hits) == 1, f"unknown gate {gate_id}")
    return hits[0]


def receipt_ok(job: str) -> bool:
    p = BUNDLE / "exits" / f"{job}.json"
    if not p.exists():
        return False
    r = read(p)
    return r.get("job") == job and r.get("exit_code") == 0


def gate_accepted(rec: dict, gate: dict, gpu_type: str, records: dict) -> bool:
    """v2 acceptance of one GATE.json: a passing status for its GPU type, its own job exited 0, and either the current
    activation or, for the 12 L40S gates only, the v1 activation and commit that the v2 activation names (the L40S evaluator
    and its bitwise criterion are unchanged in v2; see the delta review)."""
    if not (rec.get("gate_id") == gate["gate_id"] and rec.get("gpu_type") == gpu_type and rec.get("status") in PASSING[gpu_type]
            and receipt_ok(rec.get("job", ""))):
        return False
    if rec.get("activation_sha256") == records["activation_sha256"]:
        return True
    v1 = records["activation"].get("accepted_v1_l40s_gate_records")
    return bool(v1 and gpu_type == "L40S" and rec.get("status") == "PASS" and gate["gate_id"] in v1["gate_ids"]
                and rec.get("activation_sha256") == v1["activation_sha256"] and rec.get("code_commit") == v1["code_commit"])


def gate_passed(records: dict, family: str, gpu_type: str) -> dict | None:
    """The accepted record of the planned DEV gate for (family, GPU type), or None."""
    plan = [g for g in records["protocol"]["dev_gate"]["plan"] if g["family"] == family and g["gpu_type"] == gpu_type]
    if len(plan) != 1:
        return None
    for p in sorted((BUNDLE / "gate" / plan[0]["gate_id"]).glob("*/GATE.json")):
        rec = read(p)
        if gate_accepted(rec, plan[0], gpu_type, records):
            return rec
    return None


def test_plan(records: dict, key: str) -> dict:
    hits = [t for t in records["protocol"]["jobs"]["test_units"] if t["unit"] == key]
    require(len(hits) == 1, f"no TEST plan entry for {key}")
    return hits[0]


def input_lock(activation_sha: str) -> dict:
    p = BUNDLE / "INPUT_LOCK.json"
    require(p.exists(), "TEST_REPEAT inputs are not prepared (INPUT_LOCK.json missing)")
    lk = read(p)
    require(lk.get("status") == "PASS" and lk.get("activation_sha256") == activation_sha and receipt_ok(lk["job"]),
            "INPUT_LOCK.json is not a PASS of this activation from a job that exited 0")
    return lk


# ------------------------------------------------------------------------------------------------------------ the guard
class TestRepeatGuard:
    """DataGuard interface for the TEST_REPEAT role only (the frozen V10 DataGuard refuses TEST by design).

    included / check / log_rule / reads / r6_root as in v10.adapter.DataGuard.  Only role TEST_REPEAT, only the 134 manifest
    cores, only the prepared inputs whose sha256 the INPUT_LOCK recorded; every grant is logged like DataGuard.reads."""

    def __init__(self, manifest: dict, lock_record: dict, activation_sha: str):
        require_slurm()
        require(os.environ.get("V10_TR_EXPECT_ACTIVATION") == activation_sha, "activation pin missing in this job")
        self.cores = list(manifest["test"]["cores"])
        require(len(self.cores) == 134 == len(set(self.cores)), "TEST_REPEAT core list is not 134 unique cores")
        files = {f["core"]: f for f in lock_record["files"]}
        require(sorted(files) == sorted(self.cores), "input lock does not cover exactly the 134 cores")
        self.r6_root = Path(lock_record["root"])
        self.files = files
        self.manifest_sha256 = lock_record["manifest_sha256"]
        self.verified: set = set()
        self.reads: list = []

    def included(self, role: str) -> list:
        require(role == ROLE, f"TEST_REPEAT guard serves role {ROLE} only, not {role}")
        return sorted(self.cores)

    def held(self) -> list:
        return []

    def check(self, core: str, allowed_roles, purpose: str) -> Path:
        allowed = set(allowed_roles)
        if allowed != {ROLE}:
            raise PermissionError(f"TEST_REPEAT guard: allowed roles must be exactly {{{ROLE}}}, got {sorted(allowed)}")
        if core not in self.files:
            raise PermissionError(f"{core}: not a TEST_REPEAT134 core")
        path = self.r6_root / "cache" / f"{core}.npz"
        if core not in self.verified:
            if Path(self.files[core]["path"]) != path or sha(path) != self.files[core]["sha256"]:
                raise PermissionError(f"{core}: prepared TEST input differs from INPUT_LOCK.json")
            self.verified.add(core)
        self.reads.append({"core": core, "role": ROLE, "purpose": purpose, "path": str(path), "utc": utc()})
        return path

    def log_rule(self, core: str, allowed_roles, purpose: str) -> None:
        self.check(core, allowed_roles, purpose)
        self.reads[-1].update({"kind": "rule", "path": None})


class Budget:
    """Allocation stand-in for the frozen evaluators: they only call check(); no family ledger of a frozen bundle is touched."""

    def __init__(self, ledger: Path, label: str):
        self.ledger, self.label, self.last = Path(ledger), label, 0.0
        self.start = float(os.environ.get("V10_TR_START", time.time()))

    def check(self) -> None:
        if time.monotonic() - self.last > 60:
            append(self.ledger, {"utc": utc(), "event": "HEARTBEAT", "label": self.label,
                                 "job": os.environ.get("SLURM_JOB_ID"), "elapsed_s": time.time() - self.start})
            self.last = time.monotonic()


# ------------------------------------------------------------------------------------------------------- gate comparison
def compare_rows(recorded: list, new: list, *, timing=TIMING_FIELDS) -> dict:
    """Bitwise comparison of two DEV registration sets keyed by (core, perturbation): same keys, same field names, and every
    non-timing field equal (floats compared with ==, lists elementwise).  Returns the verdict with the first mismatches."""
    def keyed(rows, what):
        out = {}
        for r in rows:
            k = (r["core"], r["perturbation"])
            require(k not in out, f"duplicate {what} row {k}")
            out[k] = r
        return out
    a, b = keyed(recorded, "recorded"), keyed(new, "re-evaluated")
    mismatches = []
    if set(a) != set(b):
        mismatches.append({"kind": "keys", "missing": sorted(set(a) - set(b))[:5], "extra": sorted(set(b) - set(a))[:5]})
    max_abs = {}
    for k in sorted(set(a) & set(b)):
        ra, rb = a[k], b[k]
        fa, fb = set(ra) - set(timing), set(rb) - set(timing)
        if fa != fb:
            mismatches.append({"kind": "fields", "row": list(k), "only_recorded": sorted(fa - fb), "only_new": sorted(fb - fa)})
            continue
        for f in sorted(fa):
            if ra[f] != rb[f]:
                mismatches.append({"kind": "value", "row": list(k), "field": f, "recorded": ra[f], "new": rb[f]})
                if isinstance(ra[f], (int, float)) and isinstance(rb[f], (int, float)):
                    max_abs[f] = max(max_abs.get(f, 0.0), abs(float(ra[f]) - float(rb[f])))
    return {"status": "PASS" if not mismatches else "FAIL", "rows_recorded": len(a), "rows_new": len(b),
            "n_mismatches": len(mismatches), "first_mismatches": mismatches[:20], "max_abs_difference": max_abs,
            "compared_fields": sorted(set().union(*[set(r) for r in recorded]) - set(timing)) if recorded else [],
            "excluded_timing_fields": list(timing)}


IDENTITY_ROW_FIELDS = ("core", "perturbation", "role", "checkpoint_sha256", "theta", "t", "n_grid")


def _quantile(xs: list, q: float) -> float:
    ys = sorted(xs)
    if not ys:
        return 0.0
    k = (len(ys) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return ys[lo] + (ys[hi] - ys[lo]) * (k - lo)


def gate_diagnostics(recorded: list, new: list, recorded_identity: dict, new_identity: dict) -> dict:
    """Everything a v2 gate records besides the bitwise comparison: per-field bitwise flags, bitwise-equal row count,
    |delta err_full| max and quantiles, energy (F_hat, F_true) and estimate deltas, identity / perturbation identity,
    finiteness, DEV32 success counts and median.  (No gradient field exists in a registration row.)"""
    a = {(r["core"], r["perturbation"]): r for r in recorded}
    b = {(r["core"], r["perturbation"]): r for r in new}
    keys = sorted(set(a) & set(b))
    fields = sorted((set().union(*[set(r) for r in recorded]) | set().union(*[set(r) for r in new])) - set(TIMING_FIELDS))
    bitwise_field = {f: all(a[k].get(f) == b[k].get(f) for k in keys) for f in fields}
    bitwise_rows = sum(all(a[k].get(f) == b[k].get(f) for f in fields) for k in keys)

    def numeric(v):
        return [float(x) for x in (v if isinstance(v, list) else [v])]
    finite = all(math.isfinite(x) for r in new for f, v in r.items() if f not in ("core", "role", "checkpoint_sha256", "job")
                 and isinstance(v, (int, float, list)) and not isinstance(v, bool) for x in numeric(v))
    derr = [abs(float(b[k]["err_full"]) - float(a[k]["err_full"])) for k in keys]
    flips = {t: sum((float(a[k]["err_full"]) < t) != (float(b[k]["err_full"]) < t) for k in keys) for t in (42, 9)}
    deltas = {}
    for f in ("F_hat", "F_true", "theta_hat", "t_hat", "rot_err_deg", "trans_err_full"):
        if all(f in a[k] and f in b[k] for k in keys):
            d = [abs(x - y) for k in keys for x, y in zip(numeric(b[k][f]), numeric(a[k][f]))]
            rel = [abs(x - y) / abs(y) for k in keys for x, y in zip(numeric(b[k][f]), numeric(a[k][f])) if y != 0]
            deltas[f] = {"max_abs": max(d) if d else 0.0, "max_rel": max(rel) if rel else 0.0}
    ma, mb = metrics(recorded), metrics(new)
    k_rec = {t: sum(float(r["err_full"]) < t for r in recorded) for t in (42, 9)}
    k_new = {t: sum(float(r["err_full"]) < t for r in new) for t in (42, 9)}
    return {
        "rows_recorded": len(a), "rows_new": len(b), "same_keys": set(a) == set(b), "bitwise_rows": bitwise_rows,
        "bitwise_fields": bitwise_field, "identity_equal": recorded_identity == new_identity,
        "perturbation_fields_equal": all(a[k].get(f) == b[k].get(f) for k in keys for f in IDENTITY_ROW_FIELDS),
        "all_values_finite": finite,
        "abs_delta_err_full": {"q50": _quantile(derr, 0.5), "q90": _quantile(derr, 0.9), "q95": _quantile(derr, 0.95),
                               "q99": _quantile(derr, 0.99), "max": max(derr) if derr else 0.0,
                               "within_0.05": sum(d <= 0.05 for d in derr), "identical": sum(d == 0 for d in derr)},
        "flips": {"success42": flips[42], "success9": flips[9]},
        "deltas": deltas, "gradient_deltas": "not available: registration rows record no gradient",
        "counts": {"recorded": {"lt42": k_rec[42], "lt9": k_rec[9]}, "new": {"lt42": k_new[42], "lt9": k_new[9]}},
        "median": {"recorded": ma["median_err"], "new": mb["median_err"], "abs_difference": abs(mb["median_err"] - ma["median_err"])},
        "metrics_equal": ma == mb}


def gate_verdict(gpu_type: str, bitwise_pass: bool, metrics_equal: bool, roles_ok: bool, diag: dict,
                 rebuild_error: str | None = None) -> dict:
    """v2 decision.  L40S: bitwise only (the v1 criterion).  RTX6000PRO / H100: PASS if bitwise; else PASS_TOLERANCE iff
    (a) every identity, configuration (trainer rebuild, checkpoint load), checkpoint-hash, input-hash (DATA_LOCK) and
    perturbation-truth field (theta, t per registration) is bitwise identical and every value is finite; (b) |delta DEV32
    success<42| <= 3/320 and |delta success<9| <= 3/320; (c) |delta DEV32 median err_full| <= 0.25; (d) the median over
    registrations of |delta err_full| <= 0.25; else FAIL."""
    if rebuild_error is not None:
        return {"status": "FAIL", "reason": "(a) " + rebuild_error, "criteria": {"a": False}}
    exact = bitwise_pass and metrics_equal and roles_ok          # the v1 PASS condition, unchanged
    if exact:
        return {"status": "PASS", "reason": "bitwise", "criteria": {}}
    if gpu_type not in TOLERANCE_GPU_TYPES:
        return {"status": "FAIL", "reason": "L40S requires bitwise agreement", "criteria": {}}
    n = diag["rows_recorded"]
    cr, cn = diag["counts"]["recorded"], diag["counts"]["new"]
    crit = {"a": bool(roles_ok and diag["same_keys"] and diag["rows_new"] == n and diag["identity_equal"]
                      and diag["perturbation_fields_equal"] and diag["all_values_finite"]),
            "b": abs(cn["lt42"] - cr["lt42"]) <= TOLERANCE["count_abs"] and abs(cn["lt9"] - cr["lt9"]) <= TOLERANCE["count_abs"],
            "c": diag["median"]["abs_difference"] <= TOLERANCE["median_abs"],
            "d": diag["abs_delta_err_full"]["q50"] <= TOLERANCE["median_of_row_abs"]}
    ok = all(crit.values())
    return {"status": "PASS_TOLERANCE" if ok else "FAIL",
            "reason": "not bitwise; tolerance criteria (a)-(d) " + ("all hold" if ok else "not all hold"), "criteria": crit}


def in_job_check(diag: dict, roles_ok: bool, rebuild_error: str | None = None) -> dict:
    """v4 in-job check of the same-job DEV32 re-evaluation against the recorded DEV32 rows, before any TEST read:
    (a) identity, configuration (trainer rebuild), checkpoint sha, input hashes (DATA_LOCK), perturbation truths theta / t and
    n_grid bitwise equal, every value finite; (e) F_true bitwise equal in all 320 rows (the frozen energy is reproduced);
    (f) coarse gross-error band: |delta count<42| <= 12, |delta count<9| <= 12 (of 320), |delta median err_full| <= 0.5."""
    if rebuild_error is not None:
        return {"status": "BLOCKED", "reason": "(a) " + rebuild_error, "criteria": {"a": False}}
    n = diag["rows_recorded"]
    cr, cn = diag["counts"]["recorded"], diag["counts"]["new"]
    crit = {"a": bool(roles_ok and n == 320 and diag["same_keys"] and diag["rows_new"] == n and diag["identity_equal"]
                      and diag["perturbation_fields_equal"] and diag["all_values_finite"]),
            "e": diag["bitwise_fields"].get("F_true") is True,
            "f": (abs(cn["lt42"] - cr["lt42"]) <= IN_JOB["count_abs"] and abs(cn["lt9"] - cr["lt9"]) <= IN_JOB["count_abs"]
                  and diag["median"]["abs_difference"] <= IN_JOB["median_abs"])}
    ok = all(crit.values())
    return {"status": "PASS" if ok else "BLOCKED", "criteria": crit,
            "reason": "(a), (e), (f) hold" if ok else "failed: " + ", ".join(k for k, v in crit.items() if not v)}
