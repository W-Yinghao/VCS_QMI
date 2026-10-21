"""Pre-lock GPU health probe: record the GPU actually used and fail fast (exit 4) before any lock receipt exists.

A CUDA failure inside `vcs_common.experiment train` would leave a FAILED receipt, which under the package's lock
rules needs an amendment with a fresh grid; probing first keeps node faults outside the experiment record.
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch


def main():
    out = sys.argv[1]
    rec = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": os.uname().nodename,
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "partition": os.environ.get("SLURM_JOB_PARTITION"),
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "torch": torch.__version__, "cuda_build": torch.version.cuda}
    ok = False
    try:
        rec["cuda_available"] = torch.cuda.is_available()
        rec["device_count"] = torch.cuda.device_count()
        p = torch.cuda.get_device_properties(0)
        rec["gpu"] = {"name": p.name, "capability": [p.major, p.minor], "total_memory": p.total_memory,
                      "uuid": str(getattr(p, "uuid", "")), "multi_processor_count": p.multi_processor_count}
        x = torch.randn(2048, 2048, device="cuda:0")
        y = (x @ x.T).sum()
        torch.cuda.synchronize()
        rec["matmul_finite"] = bool(torch.isfinite(y))
        free, total = torch.cuda.mem_get_info(0)
        rec["mem_free_bytes"], rec["mem_total_bytes"] = free, total
        ok = rec["cuda_available"] and rec["matmul_finite"] and free > 16 * 2**30
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"[:1000]
    rec["status"] = "PASS" if ok else "FAIL"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps(rec))
    sys.exit(0 if ok else 4)


if __name__ == "__main__":
    main()
