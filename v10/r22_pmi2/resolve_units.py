#!/usr/bin/env python3
"""Write the 18 R22-C2 unit records (BF / EPFL x MLP_VCS_perm1, MLP_VCS_perm8, InfoNCE_host[J_perm1] x seeds 1001-1003).

Each record extends the R22-C unit record of the same final (v10/r22_pmi/resolve_units.unit: historical FINAL_DEV COMPLETE.json,
CAL-selected checkpoint, stored CAL / DEV rows, data lock, EPFL gate) with every saved checkpoint of that final (steps 250..3000,
sha256 from COMPLETE.json artifacts), its stored J CAL rows (sha256) and the CAL metrics COMPLETE.json recorded for it.  The
original (J) selection is COMPLETE.json `selected`.  BF -> v10/r22_pmi2/units/, EPFL -> v10/epfl/r22_pmi2/units/ (private).
Reads only JSON metadata and file hashes; no image, no TEST.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = HERE.parents[1]
sys.path.insert(0, str(WT / "v10/r22_pmi"))
import resolve_units as CR  # noqa: E402  (R22-C resolver: same historical units, same checks)

OUT_ROOT = CR.PROJECT / "handoff" / "v10_r22_pmi2_20261023"
FAMILIES = ("MLP_VCS_perm1", "MLP_VCS_perm8", "InfoNCE_host[J_perm1]")
STEPS = list(range(250, 3001, 250))
TIME = {("bf", "MLP_VCS_perm8"): "03:00:00", ("epfl", "MLP_VCS_perm8"): "04:30:00"}
TIME_DEFAULT = {"bf": "02:00:00", "epfl": "03:00:00"}


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def unit(ds, family, seed):
    c_name, cu = CR.unit(ds, family, seed)
    d = Path(cu["unit_dir"])
    c = json.loads((d / "COMPLETE.json").read_text())
    hist = c.get("cal_checkpoints") or c["checkpoints"]              # supplement / EPFL: cal_checkpoints; BF1: checkpoints
    assert [h["step"] for h in hist] == STEPS, d
    cks = []
    for h in hist:
        step = h["step"]
        ck = f"checkpoints/step_{step:04d}.pt"; cal = f"CAL/step_{step:04d}.jsonl"
        assert h["checkpoint"] == ck and c["artifacts"][ck] == h["checkpoint_sha256"] == CR.sha(d / ck), (d, step)
        assert c["artifacts"][cal] == CR.sha(d / cal), (d, cal)
        jobs = {str(r["job"]) for r in rows(d / cal) if "job" in r}
        assert jobs <= {cu["stored_job"]}, (d, cal, jobs)              # one training job (= stored_gpu) wrote every CAL row
        cks.append({"step": step, "checkpoint": str(d / ck), "checkpoint_sha256": h["checkpoint_sha256"],
                    "stored_cal": {"path": str(d / cal), "sha256": c["artifacts"][cal]}, "stored_cal_metrics": h["metrics"]})
    sel = c["selected"]
    assert sel["step"] == cu["selected_step"] and sel == next(
        {k: h[k] for k in ("step", "metrics", "checkpoint", "checkpoint_sha256")} for h in hist if h["step"] == sel["step"])
    name = f"r22_pmi2_{ds}_{cu['key']}_s{seed}"
    u = {k: cu[k] for k in ("dataset", "ds", "family", "key", "method", "seed", "candidate", "unit_dir", "complete_sha256",
                            "stored_job", "stored_gpu", "data_lock", "guard")}
    u.update(schema="v10.r22_pmi2.unit.v1", job_name=name, time=TIME.get((ds, family), TIME_DEFAULT[ds]),
             out=str(OUT_ROOT / "runs" / ds / cu["key"] / f"seed_{seed}"),
             original={"step": sel["step"], "checkpoint": cu["checkpoint"], "checkpoint_sha256": sel["checkpoint_sha256"],
                       "cal_metrics": sel["metrics"]},
             stored_dev=cu["stored"]["DEV"], checkpoints=cks, c_unit=c_name,
             selection_rule="min over the 12 checkpoints of (-success42, -success9, median_err, step) on the CAL registrations "
                            "(v10.supplement.records.selection_key = scripts/v10_bf1/common.selection_key; one candidate, so the "
                            "order term is constant)")
    return name, u


def main():
    n = 0
    for ds in ("bf", "epfl"):
        for family in FAMILIES:
            for seed in (1001, 1002, 1003):
                name, u = unit(ds, family, seed)
                target = (HERE / "units" if ds == "bf" else WT / "v10/epfl/r22_pmi2/units") / f"{name}.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(u, indent=1) + "\n"); n += 1
    print(n, "units written")


if __name__ == "__main__":
    main()
