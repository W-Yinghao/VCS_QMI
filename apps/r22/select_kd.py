"""KD lambda selection on CAL (seed 27182) for vcscrd and dkd_vcscrd, and the 6 final lines of the selected lambdas.

Rule (fixed before any S1 result): per arm, among the lambda candidates whose run completed, the highest best-epoch
CAL top-1; ties -> lower CAL NLL -> smaller lambda. A candidate whose run.json says "failed" (e.g. a nonfinite loss)
is reported and not selectable; any other non-complete state stops the selection (needs a look by hand).
Reads only run.json and best_cal_metrics.json.
Usage: python select_kd.py S1_CODE_DIR S2_CODE_DIR OUT_JSON OUT_LINES
  S1_CODE_DIR: snapshot the S1 jobs ran from; S2_CODE_DIR: snapshot holding configs/kd/JOBS_S2_ALL.csv.
Exit 0 = selected, 2 = not all candidates finished yet, 3 = needs a manual decision.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    s1, s2, out_json, out_lines = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3]), Path(sys.argv[4])
    rows = [r for r in csv.DictReader((s1 / "configs/kd/JOBS_S1.csv").open(newline="")) if r["arm"] in ("vcscrd", "dkd_vcscrd")]
    table, chosen = [], {}
    for arm in ("vcscrd", "dkd_vcscrd"):
        cands = []
        for r in (r for r in rows if r["arm"] == arm):
            run = Path(r["run_dir"])
            rj = json.loads((run / "run.json").read_text()) if (run / "run.json").is_file() else None
            status = rj["status"] if rj else "not_started"
            entry = {"arm": arm, "lambda": float(r["lam"]), "seed": int(r["seed"]), "job_name": r["job_name"],
                     "status": status, "run_dir": str(run)}
            if status == "complete":
                if sha(run / "best_cal_metrics.json") != rj["best_cal_metrics_sha256"]:
                    print(f"{r['job_name']}: best_cal_metrics changed after the run"); raise SystemExit(3)
                best = json.loads((run / "best_cal_metrics.json").read_text())
                entry.update(cal_top1=best["top1"], cal_nll=best["nll"], selected_epoch=best["selected_epoch"],
                             best_cal_metrics_sha256=rj["best_cal_metrics_sha256"])
            elif status == "failed":
                entry["failure"] = rj.get("failure")
            cands.append(entry)
        if len(cands) != 3:
            print(f"{arm}: expected 3 candidates, found {len(cands)}"); raise SystemExit(3)
        if any(c["status"] in ("not_started", "running") for c in cands):
            print(f"{arm}: waiting for {[c['job_name'] for c in cands if c['status'] in ('not_started', 'running')]}")
            raise SystemExit(2)
        done = [c for c in cands if c["status"] == "complete"]
        if not done:
            print(f"{arm}: no completed candidate"); raise SystemExit(3)
        chosen[arm] = sorted(done, key=lambda c: (-c["cal_top1"], c["cal_nll"], c["lambda"]))[0]["lambda"]
        table += cands
    lam_tag = lambda v: f"{v:g}"
    finals = [r for r in csv.DictReader((s2 / "configs/kd/JOBS_S2_ALL.csv").open(newline=""))
              if r["lam"] == lam_tag(chosen[r["arm"]])]
    if len(finals) != 6:
        print(f"expected 6 final rows, found {len(finals)}"); raise SystemExit(3)
    body = {"schema": "r22_kd_lambda_selection_v1",
            "rule": "per arm: max best-epoch CAL top-1 among completed candidates; tie lower CAL NLL; tie smaller lambda",
            "cal_seed": 27182, "selected_lambda": chosen, "candidates": table, "s1_code_dir": str(s1), "s2_code_dir": str(s2),
            "final_job_names": [r["job_name"] for r in finals], "evidence_role": "PILOT/DEVELOPMENT", "test_read": False}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(body, indent=1) + "\n")
    lines = subprocess.run([sys.executable, "-I", str(HERE / "server" / "make_lines.py"), str(s2), "configs/kd/JOBS_S2_ALL.csv"],
                           check=True, capture_output=True, text=True).stdout.splitlines()
    names = set(body["final_job_names"])
    keep = [l for l in lines if any(f"--job-name={n} " in l for n in names)]
    out_lines.write_text("\n".join(keep) + "\n")
    print(json.dumps({"selected_lambda": chosen, "finals": len(keep), "record": str(out_json), "lines": str(out_lines)}))


if __name__ == "__main__":
    main()
