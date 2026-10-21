#!/usr/bin/env python
"""Dry run of the job-ID derivation in `vcs_common.experiment register` for a KD grid. Writes NO lock.

It repeats register()'s per-row steps with the package's own helpers (_effective_config, _family, _candidate,
_path) and launch cwd = KD_ROOT, so the IDs equal the registered ones as long as the grid files and the outputs
layout are unchanged. After registration, emit_cal_sbatch.py --lock ... --precomputed ... verifies equality.
Standard library only (vcs_common.experiment imports no third-party module).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

KD_ROOT = Path(os.environ.get("KD_ROOT", "/home/infres/yinwang/CS_QMI/outputs_apps/kd_20261020"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid", default="protocol/kd_cal_grid/grid_manifest.json")
    p.add_argument("--run-root", default="runs/kd_cal")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    os.chdir(KD_ROOT)
    from vcs_common import experiment as ex
    cwd = Path.cwd()
    grid = Path(args.grid).resolve()
    rows = json.loads(grid.read_text())["rows"]
    jobs = []
    for row in rows:
        raw_path = Path(row["config"])
        options = [raw_path] if raw_path.is_absolute() else [grid.parent / raw_path, cwd / raw_path]
        config_path = next(q.resolve() for q in options if q.is_file())
        cfg = ex._effective_config(json.loads(config_path.read_text()), "kd", cwd)
        family, seed = ex._family(cfg, "kd"), cfg["seed"]
        if row.get("method", family) != family or int(row.get("seed", seed)) != seed:
            raise ValueError(f"grid row/config mismatch: {row}")
        candidate = ex._candidate(cfg, "kd")
        jobs.append({"job_id": f"{family}__{candidate[:16]}__seed{seed}", "family": family,
                     "coefficient": row["coefficient"], "seed": seed, "config_path": str(config_path),
                     "config_stem": config_path.stem, "config_file_sha256": ex.file_hash(config_path),
                     "effective_config_sha256": ex.canonical_hash(cfg),
                     "run_dir": str(ex._path(Path(args.run_root) / config_path.stem, cwd))})
    if len({j["job_id"] for j in jobs}) != len(jobs) or len({j["run_dir"] for j in jobs}) != len(jobs):
        raise ValueError("duplicate job IDs or run dirs")
    body = {"schema": "kd_server_precomputed_job_ids_v1", "dry_run_of": "vcs_common.experiment.register (KD rows)",
            "launch_cwd": str(cwd), "grid": str(grid), "grid_sha256": ex.file_hash(grid),
            "run_root": args.run_root, "n_jobs": len(jobs), "jobs": sorted(jobs, key=lambda j: j["job_id"])}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"n_jobs": len(jobs), "grid_sha256": body["grid_sha256"], "out": args.out}))


if __name__ == "__main__":
    main()
