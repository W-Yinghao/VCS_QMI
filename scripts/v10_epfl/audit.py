#!/usr/bin/env python3
"""V10 EPFL activation gates 1-4 (decision 2026-10-17 section 6.2 items 1-4), CPU only.  Run with python -I through Slurm (cpu-high):

  /home/infres/yinwang/CS_QMI/env/bin/python -I scripts/v10_epfl/audit.py run [--out DIR] [--gates 1,2,3,4]

Records (JSON / CSV, no image content) go to <out>/records; dataset-derived images (overlays, review panels) to <out>/overlays and
<out>/neardup_panels with their sha256 in the records; nothing is written into the repository or the data / R6 / V9 directories.
Gates 2-4 reuse gate 1's saved arrays (<out>/gate1_arrays.npz, <out>/records/ITEM1_*.json) when gate 1 is not re-run.
Exit 0 when every requested gate's required checks pass, 1 otherwise (the records are written in both cases).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
import numpy as np  # noqa: E402
from v10.epfl import audit as AU  # noqa: E402


def provenance():
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()   # noqa: E731
    return {"host": socket.gethostname(), "utc": AU.utc(), "python": platform.python_version(), "numpy": np.__version__,
            "executable": sys.executable, "git_commit": git("rev-parse", "HEAD"), "git_dirty": git("status", "--porcelain", "--", "src/v10/epfl", "scripts/v10_epfl"),
            "audit_sha256": AU.sha256_file(AU.__file__), "script_sha256": AU.sha256_file(__file__), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"), "cpus": os.environ.get("SLURM_CPUS_PER_TASK")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["run", "sheets"]); ap.add_argument("--out", type=Path, default=AU.OUT_DEFAULT); ap.add_argument("--gates", default="1,2,3,4")
    ap.add_argument("--extra-pairs", type=Path, default=None, help="sheets: JSON list of {a, b, ncc, ...} pairs (cluster expansion)")
    a = ap.parse_args(); gates = {int(g) for g in a.gates.split(",") if g}
    if a.cmd == "sheets":
        return sheets_main(a.out.resolve(), a.extra_pairs)
    out = a.out.resolve(); rec_dir = out / "records"; rec_dir.mkdir(parents=True, exist_ok=True)
    if any(bad in str(out) for bad in AU.FORBIDDEN) or str(out).startswith(str(AU.DATA_ROOT)) or str(out).startswith(str(AU.P3_ROOT)):
        raise SystemExit(f"refused output location {out}")
    log = lambda m: print(f"{AU.utc()} {m}", flush=True)   # noqa: E731
    prov = provenance(); log(f"provenance {json.dumps(prov)}"); status = {}
    if 1 in gates:
        rec1, files, pairs, arrays = AU.gate1_integrity(out, log); rec1["provenance"] = prov
        AU.write_json(rec_dir / "ITEM1_INTEGRITY.json", rec1); AU.write_csv(rec_dir / "ITEM1_FILES.csv", files); AU.write_csv(rec_dir / "ITEM1_PAIRS.csv", pairs)
        with open(out / "gate1_arrays.pkl", "wb") as f:
            pickle.dump({"files": files, "pairs": pairs, "arrays": arrays}, f)
        status[1] = rec1["summary"]["status"]; log(f"[gate1] {rec1['summary']}")
    else:
        with open(out / "gate1_arrays.pkl", "rb") as f:
            saved = pickle.load(f)
        files, pairs, arrays = saved["files"], saved["pairs"], saved["arrays"]
    if 2 in gates:
        rec2, lock, new = AU.gate2_replay(out, pairs, arrays, log); rec2["provenance"] = prov
        AU.write_json(rec_dir / "ITEM2_REPLAY.json", rec2)
        AU.write_json(rec_dir / "EPFL_DATA_LOCK.json", {"schema": "v10.epfl.data_lock.v1", "utc": AU.utc(), "p3_root": str(AU.P3_ROOT),
                                                        "p3_manifest": {"path": str(AU.P3_ROOT / "manifest.json"), "sha256": AU.PINS["p3_manifest"]},
                                                        "p3_perturbations": {"path": str(AU.P3_ROOT / "perturbations.json"), "sha256": AU.PINS["p3_perturbations"]},
                                                        "v10_manifest": {"path": str(AU.V10_MANIFEST), "sha256": AU.PINS["v10_manifest"]},
                                                        "windows": lock, "n_windows": len(lock),
                                                        "replayed_bitwise": rec2["summary"]["status"] == "PASS"})
        status[2] = rec2["summary"]["status"]; log(f"[gate2] {rec2['summary']}")
    else:
        rec2 = json.loads((rec_dir / "ITEM2_REPLAY.json").read_text())
    if 3 in gates:
        rec3 = AU.gate3_overlays(out, files, pairs, rec2, log); rec3["provenance"] = prov
        AU.write_json(rec_dir / "ITEM3_OVERLAYS.json", rec3); status[3] = rec3["summary"]["status"]; log(f"[gate3] {rec3['summary']}")
    if 4 in gates:
        rec4 = AU.gate4_near_duplicates(out, arrays, pairs, rec2, log); rec4 = AU.refine_candidates(out, rec4, log); rec4["provenance"] = prov
        AU.write_json(rec_dir / "ITEM4_NEAR_DUPLICATES.json", rec4)
        rows = rec4["candidates"] + rec4["review_band_top"]
        if rows:
            AU.write_csv(rec_dir / "ITEM4_CANDIDATES.csv", rows, fields=sorted({k for r in rows for k in r}))
        status[4] = "COMPUTED_REVIEW_PENDING"; log(f"[gate4] candidates {rec4['n_candidates']} (not REG-P3-linked {rec4['n_candidates_not_p3_linked']}), band {rec4['n_review_band']}")
    AU.write_json(rec_dir / f"RUN_{os.environ.get('SLURM_JOB_ID', 'local')}.json", {"provenance": prov, "gates": sorted(gates), "status": status})
    return 0 if all(v in ("PASS", "COMPUTED_REVIEW_PENDING") for v in status.values()) else 1




def sheets_main(out: Path, extra_pairs: Path | None = None) -> int:
    """Gate-4 review set (AU.REVIEW_RULE) and contact sheets; gate-3 flag sheets.  Writes records/ITEM4_REVIEW_SET.json and
    records/REVIEW_SHEETS.json; images under <out>/review_sheets."""
    rec_dir = out / "records"; rec4 = json.loads((rec_dir / "ITEM4_NEAR_DUPLICATES.json").read_text())
    rec3 = json.loads((rec_dir / "ITEM3_OVERLAYS.json").read_text())
    parts, role, grp = AU.review_set(rec4, out / "neardup_best_ncc.npz")
    todo = []
    for setname in ("split_relevant", "recall_extension"):
        for p in sorted(parts[setname], key=lambda p: (-p["ncc"], p["a"], p["b"])):
            todo.append({**p, "set": setname, "role_a": role[p["a"]], "role_b": role[p["b"]], "unit_a": grp[p["a"]], "unit_b": grp[p["b"]]})
    if extra_pairs:
        for p in json.loads(Path(extra_pairs).read_text()):
            todo.append({**p, "set": "cluster_expansion", "role_a": role[p["a"]], "role_b": role[p["b"]], "unit_a": grp[p["a"]], "unit_b": grp[p["b"]]})
    sheets = AU.review_sheets(out, todo, "neardup" if not extra_pairs else "neardup_expansion")
    flag_sheets = [] if extra_pairs else AU.overlay_sheets(out, rec3["overlays"], {"saturation", "constant_margin", "support_zero_content", "excluded_size"}, "flags")
    AU.write_json(rec_dir / ("ITEM4_REVIEW_SET.json" if not extra_pairs else "ITEM4_REVIEW_SET_EXPANSION.json"),
                  {"rule": AU.REVIEW_RULE, "counts": {k: len(v) for k, v in parts.items()}, "to_review": todo, "parts": parts, "utc": AU.utc()})
    AU.write_json(rec_dir / ("REVIEW_SHEETS.json" if not extra_pairs else "REVIEW_SHEETS_EXPANSION.json"),
                  {"neardup_sheets": sheets, "flag_sheets": flag_sheets, "images_in_git": False, "utc": AU.utc()})
    counts = {k: len(v) for k, v in parts.items()}
    print(f"{AU.utc()} review set {counts} -> {len(todo)} pairs to review, {len(sheets)} pair sheets, {len(flag_sheets)} flag sheets", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
