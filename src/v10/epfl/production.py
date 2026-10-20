"""EPFL production contract (main-session decisions 2026-10-19; EPFL_PROTOCOL_DECIDED.json -> EPFL_PROTOCOL_FROZEN.json).  Standard library only.

Release form of the BF supplement runner (src/v10/supplement/contract.py, records.py), with the EPFL differences:
  release        one EPFL release (status FROZEN_VALIDATED) for the production families; each family READY or BLOCKED (RuLSIF_exact_v2 until v2
                 is frozen and gated on EPFL); pins the frozen protocol, the stage-ACTIVATED activation record, the frozen group record v2, the
                 data lock, the item-5 gate (PASS) and timing (MEASURED) reports, the delta review, the source inventory and the numerics identity
  numerics       the files that determine the numbers (adapter, mf_vcs, supplement modules, the V9/R6 host) must be byte-identical to the item-5
                 gate commit c8ef14d, so the gate's acceptance carries over; the EPFL-specific deltas (gate, host, runner) are listed in the
                 delta review and covered by the CPU tests, the CPU smoke and the first formal CAL candidate (VALIDATION_REVIEW policy)
  budget         per family CAL: at most 9 candidates and 8 allocated L40S GPU-h (first limit reached; 30 s accounting margin); allocated time
                 runs from each job's Slurm StartTime (V10_EPFL_JOB_START) and earlier jobs count through their exit receipts
  hardware       L40S only (decision 5)
  statistics     CAL pooled by registration (as V10 BF); DEV by EPFL scene unit (72 units) with the frozen sensitivity sets
Nothing here reads an image.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path

from v10.supplement.records import BudgetExceeded, append, lock, metrics, rows, selection_key, utc, write   # noqa: F401  (re-exported)

REPO = Path(__file__).resolve().parents[3]
PROJECT = REPO.parent
EPFL_DIR = REPO / "v10" / "epfl"
BUNDLE_DEFAULT = PROJECT / "handoff" / "v10_epfl_production_20261019"
RELEASE_SCHEMA, RELEASE_STATUS = "v10.epfl.release.v1", "FROZEN_VALIDATED"
GATE_COMMIT = "c8ef14d9cf345a48889f15a9c4ad248621dfa0e1"
GATE_RECORD_SHA256 = "cc20086d8c7fba4c0a08cd00dbe064b9138da17d97074e10a358f5519da06043"
CAP_SECONDS, CAP_MARGIN, MAX_CANDIDATES = 8 * 3600, 30, 9
BUDGET_GPU = "NVIDIA L40S"
N_PERT = 10
NUMERICS_FILES = ("src/v10/adapter.py", "src/v10/mf_vcs.py", "src/v10/__init__.py",
                  "src/v10/supplement/__init__.py", "src/v10/supplement/checkpoints.py", "src/v10/supplement/comir.py",
                  "src/v10/supplement/contract.py", "src/v10/supplement/geometry_validation.py", "src/v10/supplement/learned.py",
                  "src/v10/supplement/mind.py", "src/v10/supplement/records.py", "src/v10/supplement/registration.py",
                  "src/v10/supplement/rulsif.py", "src/v10/supplement/training.py", "v10/supplement/COMIR_SOURCE.json",
                  "src/v9/__init__.py", "src/v9/reg/__init__.py", "src/v9/reg/g1.py", "src/v9/reg/g2.py",
                  "src/v9/reg/r6_reg_pilot.py", "src/v9/reg/r6_reg_p2.py", "src/v9/reg/r6_reg_p3.py")
DEV_SETS = ("DEV72_primary", "DEV_minus_weaker_relations", "DEV_minus_boundary_and_saturated", "DEV_minus_all")


def sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def require(ok, why):
    if not ok:
        raise RuntimeError(why)


# ------------------------------------------------------------------------------------------------------------ identity of the code
def source_inventory(repo: Path = REPO) -> dict:
    """sha256 of every file the production run depends on (code, job scripts, tests, the EPFL / supplement records in the tree)."""
    paths = set()
    for rel in ("src/v10", "src/v9", "scripts/v10_epfl", "scripts/v10"):
        paths.update(p for p in (repo / rel).rglob("*.py") if "__pycache__" not in p.parts)
    paths.update((repo / "slurm/v10_epfl").glob("*.sbatch"))
    paths.update((repo / "tests/v10").glob("test_epfl*.py"))
    paths.update((repo / "v10/epfl").glob("*.json"))
    paths.update((repo / "v10/supplement").glob("*.json"))
    return {str(p.relative_to(repo)): sha(p) for p in sorted(paths)}


def numerics_identity(repo: Path = REPO) -> dict:
    return {f: sha(repo / f) for f in NUMERICS_FILES if (repo / f).exists()}


def git_blob_sha256(commit: str, path: str, repo: Path = REPO):
    r = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{path}"], capture_output=True)
    return hashlib.sha256(r.stdout).hexdigest() if r.returncode == 0 else None


def gate_numerics_identity(repo: Path = REPO) -> dict:
    """The numerics files as they were at the item-5 gate commit (from git)."""
    return {f: git_blob_sha256(GATE_COMMIT, f, repo) for f in NUMERICS_FILES if git_blob_sha256(GATE_COMMIT, f, repo) is not None}


# ------------------------------------------------------------------------------------------------------------ release
def verify_release(path, expected_sha, family=None, *, candidates_fn=None, schedule_fn=None, repo: Path = REPO) -> dict:
    """Every release check except the Slurm requirement (the runner adds it).  family=None checks the release only."""
    path = Path(path).resolve()
    require(bool(expected_sha) and sha(path) == expected_sha, "Missing or changed submission-pinned EPFL release")
    rel = read(path)
    require(rel.get("schema") == RELEASE_SCHEMA and rel.get("status") == RELEASE_STATUS, "Not a frozen, validated EPFL release")
    inv = source_inventory(repo)
    require(rel.get("source_files") == inv and rel.get("code_sha256") == digest(inv), "EPFL source identity changed since the release")
    num = numerics_identity(repo)
    require(rel.get("numerics_identity") == num == rel.get("gate_numerics_identity"),
            "Numerics files differ from the item-5 gate commit: the gate acceptance does not carry over")
    for key in ("frozen_protocol", "activation_record", "group_record", "data_lock", "item5_gate", "item5_timing", "delta_review"):
        ref = rel.get(key) or {}
        p = Path(ref.get("path", ""))
        p = p if p.is_absolute() else repo / p
        require(bool(ref.get("path")) and p.exists() and sha(p) == ref.get("sha256"), "Pinned input changed: " + key)
    proto = read(_abs(rel["frozen_protocol"]["path"], repo))
    require(proto.get("status") == "FROZEN" and proto.get("schema") == "v10.epfl.protocol.v2", "Protocol is not the frozen EPFL protocol")
    act = read(_abs(rel["activation_record"]["path"], repo))
    require(act.get("stage") == "ACTIVATED" and act.get("frozen_epfl_protocol_sha256") == rel["frozen_protocol"]["sha256"]
            and act.get("frozen_group_record_sha256") == rel["group_record"]["sha256"], "Activation record is not the stage-ACTIVATED record of this protocol")
    gate = read(_abs(rel["item5_gate"]["path"], repo))
    require(gate.get("status") == "PASS" and gate.get("pins", {}).get("commit") == GATE_COMMIT
            and gate.get("pins", {}).get("record_sha256") == GATE_RECORD_SHA256 and gate.get("environment", {}).get("gpu_name") == BUDGET_GPU,
            "Item-5 gate is not the PASS record of the pinned gate commit on L40S")
    timing = read(_abs(rel["item5_timing"]["path"], repo))
    require(timing.get("status") == "MEASURED" and timing.get("pins", {}).get("commit") == GATE_COMMIT and timing.get("gpu") == BUDGET_GPU,
            "Item-5 timing is not the MEASURED record of the pinned gate commit on L40S")
    if (Path(path).parent / "PAUSE").exists():
        raise RuntimeError("EPFL production is paused (PAUSE marker in the release directory)")
    if family is not None:
        entry = rel.get("families", {}).get(family)
        require(entry is not None, "Family not in the EPFL release: " + family)
        if entry.get("status") != "READY":
            raise RuntimeError(f"{family} is {entry.get('status')}: {entry.get('reason', '')}")
        if candidates_fn is not None:
            require(entry.get("candidates") == candidates_fn(family), "Candidate grid differs from the frozen definition: " + family)
        if schedule_fn is not None:
            require(entry.get("schedule") == schedule_fn(family), "Schedule differs from the frozen definition: " + family)
        require(entry.get("budget") == {"cap_seconds": CAP_SECONDS, "margin_seconds": CAP_MARGIN, "max_candidates": MAX_CANDIDATES},
                "Budget rule differs from decision 4")
        require(len(entry.get("candidates", [])) <= MAX_CANDIDATES, "More than 9 candidates")
    return rel


def require_release(path, family, **kw):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("EPFL production runs only through Slurm")
    return verify_release(path, os.environ.get("V10_EPFL_EXPECT_RELEASE"), family, **kw)


def _abs(p, repo: Path = REPO) -> Path:
    p = Path(p)
    return p if p.is_absolute() else repo / p


def identity(release_sha, family, candidate, seed, phase, data_sha, group_sha, schedule, dependency=None) -> dict:
    return {"schema": "v10.epfl.unit.v1", "release_sha256": release_sha, "family": family, "candidate": candidate, "seed": seed,
            "phase": phase, "schedule": schedule, "data_lock_sha256": data_sha, "group_record_sha256": group_sha,
            "selection_dependency_sha256": dependency}


# ------------------------------------------------------------------------------------------------------------ budget ledger
class Allocation:
    """Allocated wall time from the job's Slurm StartTime (V10_EPFL_JOB_START, epoch seconds) into <root>/COST_LEDGER.jsonl.  For CAL, the
    family's earlier jobs count with their exit receipts (<root>/exits/<job>.json, written by the sbatch wrapper); check() raises
    BudgetExceeded at cap - margin (decision 4: 8 L40S GPU-h per family, first limit reached, no extension)."""

    def __init__(self, root, family, phase, *, job=None, started=None, release_sha=None, cap_seconds=CAP_SECONDS, margin=CAP_MARGIN):
        self.path = Path(root) / "COST_LEDGER.jsonl"; self.root = Path(root)
        self.family, self.phase = family, phase
        self.job = job or os.environ["SLURM_JOB_ID"]
        self.started = float(started if started is not None else os.environ["V10_EPFL_JOB_START"])
        self.release_sha = release_sha if release_sha is not None else os.environ.get("V10_EPFL_EXPECT_RELEASE")
        self.cap, self.margin = cap_seconds, margin
        previous = {}
        for r in rows(self.path):
            if r["family"] == family and r["phase"] == "CAL" and r["job"] != self.job:
                previous[r["job"]] = max(previous.get(r["job"], 0), r["allocated_seconds"])
        if phase == "CAL":
            for job in previous:
                rp = self.root / "exits" / f"{job}.json"
                require(rp.exists(), "Prior CAL allocation lacks an exit receipt; reconcile the Slurm exit before a retry")
                rec = read(rp)
                require(rec.get("job") == job and rec.get("release_sha256") == self.release_sha, "Prior CAL exit receipt identity differs")
                sec = rec.get("allocated_seconds")
                require(isinstance(sec, (int, float)) and math.isfinite(sec) and sec >= 0, "Invalid prior allocation")
                previous[job] = max(previous[job], sec)
        self.previous = sum(previous.values()) if phase == "CAL" else 0.0
        self.heartbeat = time.monotonic()
        self.record("START")

    def elapsed(self):
        return max(0.0, time.time() - self.started)

    def remaining(self):
        return self.cap - self.margin - self.previous - self.elapsed()

    def record(self, event, **extra):
        append(self.path, {"utc": utc(), "job": self.job, "family": self.family, "phase": self.phase, "event": event,
                           "allocated_seconds": self.elapsed(), "previous_cal_seconds": self.previous, "gpu_count": 1,
                           "accounting": "Slurm StartTime to now; queue wait separate", **extra})
        self.heartbeat = time.monotonic()

    def check(self):
        if time.monotonic() - self.heartbeat > 30:
            self.record("HEARTBEAT")
        if self.phase == "CAL" and self.remaining() <= 0:
            raise BudgetExceeded("The 8 L40S allocated GPU-h CAL family cap is exhausted")


# ------------------------------------------------------------------------------------------------------------ resume and completion
def truncate_steps(path, last_step: int) -> int:
    """On resume from the checkpoint at last_step: keep only the training-log rows with step <= last_step (rows written after the checkpoint
    by an interrupted job would otherwise be duplicated).  Atomic rewrite; returns the number of rows removed."""
    path = Path(path)
    if not path.exists():
        return 0
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    keep = [ln for ln in lines if json.loads(ln)["step"] <= last_step]
    steps = [json.loads(ln)["step"] for ln in keep]
    require(steps == sorted(steps) and len(steps) == len(set(steps)), "Training log has duplicate or unordered steps before the checkpoint")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text("".join(ln + "\n" for ln in keep)); os.replace(tmp, path)
    return len(lines) - len(keep)


def verified_complete(directory, ident, schedule):
    """As the supplement's verified_complete, for EPFL units (required artifacts include DEV for FINAL_DEV)."""
    directory = Path(directory); p = directory / "COMPLETE.json"
    if not p.exists():
        return None
    result = read(p)
    require(result.get("status") == "COMPLETE" and result.get("identity") == ident, "Stale COMPLETE identity: " + str(directory))
    require(result.get("steps") == schedule["steps"], "Incomplete training marked COMPLETE")
    require([r["step"] for r in result.get("cal_checkpoints", [])] == schedule["cal_steps"], "Missing prescribed CAL checkpoints")
    require(result.get("selected") == min(result["cal_checkpoints"], key=selection_key), "Checkpoint selection differs from the CAL readouts")
    required = {"IDENTITY.json", "CONFIG.json", "DATA_READS.json"}
    for step in schedule["save_steps"]:
        required.update({f"checkpoints/step_{step:04d}.pt", f"checkpoints/step_{step:04d}.json"})
    for step in schedule["cal_steps"]:
        required.update({f"CAL/step_{step:04d}.jsonl", f"CAL/step_{step:04d}.summary.json"})
    if ident["phase"] == "FINAL_DEV":
        required.update({"DEV.jsonl", "DEV.summary.json"})
    arts = result.get("artifacts", {})
    require(required <= arts.keys(), "COMPLETE lacks mandatory artifacts")
    for name, expected in arts.items():
        a = (directory / name).resolve()
        require(a.is_relative_to(directory.resolve()) and sha(a) == expected, "COMPLETE artifact mismatch: " + name)
    return result


# ------------------------------------------------------------------------------------------------------------ DEV statistics (decision 3, 7)
def unit_metrics(records):
    errs = [float(r["err_full"]) for r in records]
    return {"n": len(errs), "success42": sum(e < 42 for e in errs) / len(errs), "success9": sum(e < 9 for e in errs) / len(errs),
            "median_err": statistics.median(errs), "median_err_native_px": statistics.median(errs) / 4}


def dev_statistics(records, group: dict) -> dict:
    """Registration-level and unit-macro DEV metrics on the primary set and the frozen sensitivity sets.  records carry window, perturbation,
    err_full; units from the frozen group record (scene units).  Unit-macro = mean over units of the unit's success fraction over all its
    registrations (windows x perturbations); the perturbations of a unit are not independent observations."""
    unit_of = group["window_unit"]; out = {}
    for name in DEV_SETS:
        s = group["sensitivity_sets"][name]; windows = set(s["windows"])
        sel = [r for r in records if r["window"] in windows]
        require({r["window"] for r in sel} == windows, f"DEV records do not cover the set {name}")
        per_unit = {}
        for r in sel:
            per_unit.setdefault(unit_of[r["window"]], []).append(r)
        um = {u: unit_metrics(v) for u, v in sorted(per_unit.items())}
        require(sorted(um) == sorted(s["units"]), f"unit set differs for {name}")
        out[name] = {"registrations": unit_metrics(sel), "n_units": len(um),
                     "unit_macro": {"success42": statistics.fmean(m["success42"] for m in um.values()),
                                    "success9": statistics.fmean(m["success9"] for m in um.values()),
                                    "median_of_unit_medians_native_px": statistics.median(m["median_err_native_px"] for m in um.values())},
                     "per_unit": um if name == "DEV72_primary" else None}
    return out
