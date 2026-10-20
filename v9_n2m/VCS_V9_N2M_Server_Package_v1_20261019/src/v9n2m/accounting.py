"""Read Slurm accounting once per distinct allocation, including failed attempts."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .common import read_json, sha256_file, utc_now, write_csv, write_json


def parse_sacct(text):
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 7:
            raise ValueError(f"Unexpected sacct field count: {line!r}")
        job, state, elapsed, tres, exit_code, started, ended = parts[:7]
        # .batch and .extern repeat parent allocation duration and are excluded.
        if not re.fullmatch(r"\d+(?:_\d+)?", job):
            continue
        resources = dict(x.split("=", 1) for x in tres.split(",") if "=" in x)
        gpu_count = int(resources["gres/gpu"]) if "gres/gpu" in resources else None
        seconds = int(elapsed) if elapsed.isdigit() else None
        terminal = state.split()[0].rstrip("+") in {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}
        rows.append({"job_id": job, "state": state, "elapsed_s": seconds, "gpu_count": gpu_count,
                     "gpu_allocation_hours": seconds*gpu_count/3600 if seconds is not None and gpu_count is not None else None,
                     "exit_code": exit_code, "start": started, "end": ended, "terminal": terminal,
                     "alloc_tres": tres})
    if len({r["job_id"] for r in rows}) != len(rows):
        raise ValueError("sacct returned multiple parent records for a job; inspect restarts before summing")
    return rows


def collect_accounting(experiment_path, out_dir, sacct_text_path=None):
    experiment_path, out = Path(experiment_path), Path(out_dir)
    exp = read_json(experiment_path)
    jobs, receipts = set(), []
    for unit in exp["units"]:
        if unit["action"] != "RUN":
            continue
        for p in (Path(unit["output_dir"]) / "attempts").glob("*.json"):
            rec = read_json(p)
            if rec.get("job_id"):
                jobs.add(str(rec["job_id"]))
            receipts.append({"path": str(p), "sha256": sha256_file(p), "job_id": rec.get("job_id")})
    verification_job = exp.get("target_environment", {}).get("job_id")
    if verification_job:
        jobs.add(str(verification_job))
    for p in (experiment_path.parent / "diagnostics").rglob("*.meta.json"):
        rec = read_json(p)
        if rec.get("hardware", {}).get("job_id"):
            jobs.add(str(rec["hardware"]["job_id"]))
    if any(not re.fullmatch(r"\d+(?:_\d+)?", job) for job in jobs):
        raise ValueError("Invalid job ID in an attempt receipt")
    if sacct_text_path:
        raw = Path(sacct_text_path).read_text()
        command = None
    elif jobs:
        command = ["sacct", "-n", "-P", "-j", ",".join(sorted(jobs)),
                   "--format=JobIDRaw,State%40,ElapsedRaw,AllocTRES%1000,ExitCode,Start,End"]
        raw = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30).stdout
    else:
        command, raw = None, ""
    rows = [r for r in parse_sacct(raw) if r["job_id"] in jobs]
    missing = sorted(jobs - {r["job_id"] for r in rows})
    report = {"created_utc": utc_now(), "experiment_sha256": sha256_file(experiment_path),
              "scope": "Distinct new-round allocations (training attempts, verification and discovered fixed-loss exports); historical reused VCS cost is separate.",
              "requested_jobs": sorted(jobs), "missing_jobs": missing, "command": command,
              "attempt_receipts": receipts, "jobs": rows,
              "gpu_allocation_hours_known": sum(r["gpu_allocation_hours"] for r in rows if r["gpu_allocation_hours"] is not None),
              "complete_terminal_accounting": not missing and bool(jobs) and all(r["terminal"] and r["gpu_allocation_hours"] is not None for r in rows),
              "note": "One seed allocation may contain three arm processes; allocation duration is counted once. Incomplete sacct fields are unknown, never zero."}
    out.mkdir(parents=True, exist_ok=True)
    (out / "sacct_raw.txt").write_text(raw, encoding="utf-8")
    write_json(out / "allocation_accounting.json", report)
    write_csv(out / "allocation_accounting.csv", rows, ["job_id", "state", "elapsed_s", "gpu_count", "gpu_allocation_hours", "exit_code", "start", "end", "terminal", "alloc_tres"])
    return report
