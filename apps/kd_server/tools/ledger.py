#!/usr/bin/env python
"""Per-Slurm-job provenance ledger for the KD server runs (stdlib only).

start: writes <KD_ROOT>/ledger/<kind>/<tag>__slurm<id>.json with node, partition, GPU name/UUID/driver,
       throttle reasons, Slurm request, worktree commit/dirty state and helper-script hashes.
end:   adds exit code, stage and end time to the same file.
Nothing here touches a lock, receipt or run directory.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import socket
import subprocess
from pathlib import Path

KD_ROOT = Path(os.environ.get("KD_ROOT", "/home/infres/yinwang/CS_QMI/outputs_apps/kd_20261020"))
WORKTREE = Path("/home/infres/yinwang/CS_QMI/ssl_pilot_app_kd")
SERVER = WORKTREE / "apps" / "kd_server"
SLURM_KEYS = ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_PARTITION", "SLURMD_NODENAME", "SLURM_JOB_NODELIST",
              "SLURM_JOB_QOS", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "SLURM_RESTART_COUNT",
              "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "CUDA_VISIBLE_DEVICES", "SLURM_SUBMIT_DIR", "SLURM_NICE")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(cmd, timeout=30):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"UNAVAILABLE: {type(exc).__name__}: {exc}"


def _query(fields):
    out = run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"])
    names = fields.split(",")
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        rows.append(dict(zip(names, parts)) if len(parts) == len(names) else {"raw": line})
    return rows


def gpu_record():
    rows = _query("index,name,uuid,driver_version,memory.total,pci.bus_id")
    # Field names changed across driver versions; an unknown field fails the whole query, so ask separately.
    for extra in ("clocks_event_reasons.active", "clocks_throttle_reasons.active"):
        more = _query(extra)
        bad = any(str(r.get(extra, "")).startswith("UNAVAILABLE") or "not a valid field" in str(r) for r in more)
        if more and not bad and all(extra in r for r in more) and len(more) == len(rows):
            for r, m in zip(rows, more):
                r[extra] = m[extra]
            break
    return rows


def path_for(kind, tag):
    job = os.environ.get("SLURM_JOB_ID", "nojob")
    return KD_ROOT / "ledger" / kind / f"{tag}__slurm{job}.json"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def start(args):
    helpers = sorted(p for p in SERVER.rglob("*") if p.is_file() and p.suffix in (".py", ".sh", ".sbatch", ".json"))
    script = os.environ.get("KD_BATCH_SCRIPT")
    record = {
        "schema": "kd_server_ledger_v1", "kind": args.kind, "tag": args.tag, "started_utc": now(),
        "hostname": socket.gethostname(), "slurm": {k: os.environ.get(k) for k in SLURM_KEYS},
        "gpus": gpu_record(),
        "worktree_head": run(["git", "-C", str(WORKTREE), "rev-parse", "HEAD"]),
        "worktree_dirty_paths": run(["git", "-C", str(WORKTREE), "status", "--porcelain"]).splitlines(),
        "batch_script_sha256": sha256(script) if script and Path(script).is_file() else None,
        "server_file_sha256": {str(p.relative_to(WORKTREE)): sha256(p) for p in helpers},
        "lock": args.lock, "lock_sha256_file": sha256(args.lock) if args.lock and Path(args.lock).is_file() else None,
        "extra": json.loads(args.extra) if args.extra else None,
    }
    write(path_for(args.kind, args.tag), record)
    print(f"ledger: {path_for(args.kind, args.tag)}", flush=True)


def end(args):
    path = path_for(args.kind, args.tag)
    record = json.loads(path.read_text()) if path.is_file() else {"schema": "kd_server_ledger_v1", "missing_start": True}
    record.update(ended_utc=now(), exit_code=args.rc, final_stage=args.stage,
                  gpus_at_end=gpu_record())
    if args.extra:
        record["end_extra"] = json.loads(args.extra)
    write(path, record)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("start", "end"))
    p.add_argument("--kind", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--lock")
    p.add_argument("--rc", type=int)
    p.add_argument("--stage")
    p.add_argument("--extra", help="JSON string")
    args = p.parse_args()
    (start if args.action == "start" else end)(args)


if __name__ == "__main__":
    main()
