#!/usr/bin/env python3
"""Write the 36 R22-C unit records (BF / EPFL x 6 families x seeds 1001-1003) from the historical FINAL_DEV COMPLETE.json files.

Each unit pins the CAL-selected checkpoint of that final (selected.checkpoint, by the frozen lexicographic CAL rule; nothing
is chosen here), the stored CAL rows of that checkpoint and the stored DEV rows (artifact hashes from COMPLETE.json), the data
lock and, for EPFL, the stage-ACTIVATED gate record and the frozen group record v2.  BF units -> v10/r22_pmi/units/, EPFL units
-> v10/epfl/r22_pmi/units/ (EPFL records stay private).  Reads only JSON metadata; no image, no TEST.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

WT = Path(__file__).resolve().parents[2]
PROJECT = WT.parent
OUT_ROOT = PROJECT / "handoff" / "v10_r22_pmi_20261022"
FAMILIES = {"MF_VCS_M1": ("M1", "MF_M1"), "MF_VCS_M0": ("M0", "MF_M0"), "MF_VCS_perm1": ("MFp1", "MF_perm1"),
            "MLP_VCS_perm1": ("MLPp1", "MLP_perm1"), "MLP_VCS_perm8": ("MLPp8", "MLP_perm8"),
            "InfoNCE_host[J_perm1]": ("InfoNCE", "InfoNCE_host")}
BF_ROOTS = {"MF_VCS_M1": "v10_bf1_20261017/results/FINAL_DEV", "MF_VCS_M0": "v10_bf1_20261017/results/FINAL_DEV",
            "MF_VCS_perm1": "v10_bf1_20261017/results/FINAL_DEV", "MLP_VCS_perm1": "v10_bf1_20261017/results/FINAL_DEV",
            "MLP_VCS_perm8": "v10_method_pipeline_20261019/families/0/results/FINAL_DEV",
            "InfoNCE_host[J_perm1]": "v10_method_pipeline_20261019/families/1/results/FINAL_DEV"}
EPFL_ROOT = "v10_epfl_production_20261019/results/FINAL_DEV"
BF_DATA_LOCK = PROJECT / "handoff/v10_bf1_20261017/DATA_LOCK.json"
EPFL_DIR = WT / "v10/epfl"
EPFL_RECORD = EPFL_DIR / "EPFL_ACTIVATION_RECORD_ACTIVATED.json"
EPFL_GROUP = EPFL_DIR / "EPFL_GROUP_RECORD_V2_FROZEN.json"
EPFL_DATA_LOCK = EPFL_DIR / "EPFL_DATA_LOCK.json"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stored_gpu(d, c):
    if c.get("hardware", {}).get("gpu_name"):
        return c["hardware"]["gpu_name"]
    job = str(c.get("job") or c.get("job_id"))
    return json.loads((d / f"ENV_{job}.json").read_text()).get("gpu_name")


def unit(ds, family, seed):
    key, method = FAMILIES[family]
    root = PROJECT / "handoff" / (BF_ROOTS[family] if ds == "bf" else EPFL_ROOT)
    d = root / family / f"seed_{seed}"
    c = json.loads((d / "COMPLETE.json").read_text())
    ident = c["identity"]
    assert c["status"] == "COMPLETE" and ident["family"] == family and ident["seed"] == seed and ident["phase"] == "FINAL_DEV"
    sel = c["selected"]; step = sel["step"]
    ck = d / sel["checkpoint"]
    assert sha(ck) == sel["checkpoint_sha256"] == c["artifacts"][sel["checkpoint"]], ck
    cal = d / "CAL" / f"step_{step:04d}.jsonl"; dev = d / "DEV.jsonl"
    assert sha(cal) == c["artifacts"][f"CAL/step_{step:04d}.jsonl"] and sha(dev) == c["artifacts"]["DEV.jsonl"], d
    if ds == "bf":
        lock = {"path": str(BF_DATA_LOCK), "sha256": sha(BF_DATA_LOCK)}
        candidate = ident.get("candidate") or {"lr": ident["lr"], "cap": ident.get("cap")}
        guard = None
    else:
        lock = {"path": str(EPFL_DATA_LOCK), "sha256": sha(EPFL_DATA_LOCK)}
        candidate = ident["candidate"]
        guard = {"record": str(EPFL_RECORD), "record_sha256": sha(EPFL_RECORD), "group_record": str(EPFL_GROUP),
                 "group_record_sha256": sha(EPFL_GROUP)}
        assert guard["group_record_sha256"] == ident["group_record_sha256"]
    assert lock["sha256"] == ident["data_lock_sha256"], (d, lock)
    name = f"r22_pmi_{ds}_{key}_s{seed}"
    return name, {
        "schema": "v10.r22_pmi.unit.v1", "dataset": "BF_SHG" if ds == "bf" else "EPFL_RGB_NIR", "ds": ds,
        "family": family, "key": key, "method": method, "seed": seed, "candidate": candidate,
        "unit_dir": str(d), "complete_sha256": sha(d / "COMPLETE.json"),
        "selected_step": step, "checkpoint": str(ck), "checkpoint_sha256": sel["checkpoint_sha256"],
        "stored": {"CAL": {"path": str(cal), "sha256": sha(cal)}, "DEV": {"path": str(dev), "sha256": sha(dev)}},
        "stored_job": str(c.get("job") or c.get("job_id")), "stored_gpu": stored_gpu(d, c),
        "data_lock": lock, "guard": guard, "job_name": name,
        "out": str(OUT_ROOT / "runs" / ds / key / f"seed_{seed}"),
        "selection_note": "selected.checkpoint of the historical FINAL_DEV COMPLETE.json (frozen CAL lexicographic rule); "
                          "no selection in R22-C"}


def main():
    names = []
    for ds in ("bf", "epfl"):
        for family in FAMILIES:
            for seed in (1001, 1002, 1003):
                name, u = unit(ds, family, seed)
                target = (WT / "v10/r22_pmi/units" if ds == "bf" else EPFL_DIR / "r22_pmi/units") / f"{name}.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(u, indent=1) + "\n")
                names.append(str(target.relative_to(WT)))
    print(len(names), "units written")


if __name__ == "__main__":
    main()
