"""Pinned scheduling and acceptance checks. Standard library; no image reads."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time


POLICY_ID = "v10_ready_dispatch_20261019"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def utc():
    return time.strftime("%FT%TZ", time.gmtime())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def ref(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def checked_ref(record):
    p = Path(record["path"])
    require(sha(p) == record["sha256"], "Changed pinned file: " + str(p))
    return read(p)


def command(args):
    p = subprocess.run(args, capture_output=True, text=True, timeout=45)
    require(p.returncode == 0, "Command failed: " + repr(args) + ": " + p.stderr)
    return p.stdout


def queue():
    result = []
    for line in command(["squeue", "-u", "yinwang", "-r", "-h", "-o",
                         "%i|%j|%T|%q|%b|%R"]).splitlines():
        fields = line.split("|", 5)
        require(len(fields) == 6, "Malformed queue row")
        result.append(dict(zip(("id", "name", "state", "qos", "gres", "reason"), fields)))
    return result


def gpu_count(gres):
    return sum(int(p.rsplit(":", 1)[1]) for p in (gres or "").split(",")
               if p.startswith("gres/gpu:") and p.rsplit(":", 1)[1].isdigit())


def capacity(rows):
    normal = [r for r in rows if r["qos"] == "normal"]
    project = [r for r in normal if gpu_count(r["gres"]) > 0
               and r["name"].startswith(("v9", "v10"))]
    return (len(normal) < 29 and len(project) < 16
            and sum(r["state"] == "PENDING" for r in project) < 8)


def science_equivalence(old_root, new_root, old_files, new_files):
    """No numerical code change is eligible for a scheduling-only migration."""
    require(old_files.keys() == new_files.keys(), "Source file set changed")
    changed = [p for p in old_files if old_files[p] != new_files[p]]
    allowed = "src/v10/supplement/contract.py"
    require(changed == [allowed], "Migration changes more than the release guard: " + repr(changed))
    for root, files in ((old_root, old_files), (new_root, new_files)):
        for p, h in files.items():
            require(sha(Path(root)/p) == h, "Changed source: " + str(Path(root)/p))
    trees = []
    for root in (old_root, new_root):
        tree = ast.parse((Path(root)/allowed).read_text())
        matches = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "require_release"]
        require(len(matches) == 1, "Release guard missing/duplicated")
        tree.body.remove(matches[0])
        trees.append(ast.dump(tree, include_attributes=False))
    require(trees[0] == trees[1], "Method recipes or other contract functions changed")
    return {"status": "PASS", "changed_files": changed,
            "all_other_files_byte_identical": True,
            "contract_ast_outside_require_release_identical": True,
            "validation_tested_code_sha256": digest(old_files),
            "execution_target_code_sha256": digest(new_files),
            "scope": "Scheduling authorization only; no training, data, energy, search, recipe, precision or validation change"}


def exit_receipt(path, job, expected_hash, hash_key):
    receipt = read(path)
    require(receipt.get("job") == job and receipt.get("exit_code") == 0
            and receipt.get(hash_key) == expected_hash, "Missing matching successful exit receipt")
    seconds = receipt.get("allocated_seconds", receipt.get("batch_seconds"))
    require(isinstance(seconds, (int, float)) and math.isfinite(seconds) and seconds >= 0,
            "Invalid actual allocation duration")
    return receipt


def report_ok(path, family, code_sha, manifest_sha, job):
    report = read(path)
    require(report.get("status") == "PASS" and report.get("family") == family
            and report.get("job") == job and report.get("code_sha256") == code_sha
            and report.get("validation_manifest_sha256") == manifest_sha,
            "Validation report not PASS for the exact family, code, manifest and job")
    require(report.get("held_test_read") is False and report.get("dev_read") is False,
            "Validation used non-validation data roles")
    return report


def parsed_job(result):
    require(result.returncode == 0 and re.fullmatch(r"\d+(;[^\s]+)?", result.stdout.strip()),
            "Ambiguous/failed submission; do not retry without reconciliation")
    return result.stdout.strip().split(";")[0]

