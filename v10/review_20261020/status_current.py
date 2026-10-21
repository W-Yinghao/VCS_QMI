#!/usr/bin/env python3
"""V10_STATUS_CURRENT.json (review package §6 item 1): one entry per planned unit with state, job and evidence path.

States: COMPLETE_VERIFIED, RUNNING, QUEUED, FAILED, BLOCKED, NOT_SUBMITTED. Sources: the frozen TEST_REPEAT134 manifest (BF
selections), the seed-extension verify.py, the EPFL official analysis, the TEST_REPEAT v5 status.py and squeue.
Usage: status_current.py OUT.json
"""
import json, subprocess, sys, time
from pathlib import Path

HO = Path("/home/infres/yinwang/CS_QMI/handoff"); PY = "/home/infres/yinwang/CS_QMI/env/bin/python"


def queue():
    out = subprocess.run(["squeue", "-u", "yinwang", "-h", "-o", "%i %j %T"], capture_output=True, text=True).stdout.split("\n")
    return {l.split()[1]: (l.split()[0], l.split()[2]) for l in out if l.strip()}


def main():
    q = queue(); units = []
    for u in json.load(open(HO / "v10_test_repeat134_20261020/TEST_REPEAT134_MANIFEST.json"))["units"]:
        units.append({"line": "BF selections (frozen in TEST_REPEAT134 manifest)", "unit": u["key"], "tier": u["tier"],
                      "state": "COMPLETE_VERIFIED", "evidence": u["unit_dir"]})
    ext = subprocess.run([PY, str(HO / "v10_seed_ext_1006_1008_20261020/verify.py")], capture_output=True, text=True,
                         cwd="/home/infres/yinwang/CS_QMI").stdout
    for l in ext.splitlines():
        p = l.split()
        if len(p) >= 4 and p[0].startswith("m") and "_seed_" in p[0]:
            name = f"v10sx_{p[0][1]}_{p[0].split('_')[-1]}"
            st = p[3] if p[3] != "NOT_SUBMITTED" or name not in q else ("RUNNING" if q[name][1] == "RUNNING" else "QUEUED")
            units.append({"line": "BF seed extension 1006-1008 (secondary DEV)", "unit": p[0], "method": p[1], "state": st,
                          "job": q.get(name, (p[4] if len(p) > 4 and p[4].isdigit() else None,))[0],
                          "evidence": str(HO / "v10_seed_ext_1006_1008_20261020/results" / p[0])})
    ep = json.load(open(HO / "v10_epfl_production_20261019/analysis/EPFL_RESULTS.json"))
    for fam, v in ep["families"].items():
        for s, f in (v.get("finals") or {}).items():
            units.append({"line": "EPFL finals", "unit": f"{fam}__{s}", "state": "COMPLETE_VERIFIED" if f else "NOT_SUBMITTED",
                          "evidence": str(HO / "v10_epfl_production_20261019/analysis/EPFL_RESULTS.json")})
    for T, v in ep["rulsif_v2"].items():
        for s in (1001, 1002, 1003):
            units.append({"line": f"EPFL RuLSIF_exact_v2 T{T} ({v['role']})", "unit": f"seed_{s}",
                          "state": "COMPLETE_VERIFIED" if s in v["summary"].get("seeds_complete", []) else "NOT_SUBMITTED",
                          "evidence": str(HO / f"v10_epfl_rulsif_v2_T{T}_20261020")})
    evn_names = {}
    for l in open(HO / "v10_epfl_eval_noise_20261020/JOB_LINES.txt"):
        if "--job-name=" in l:
            t = l.split(); evn_names[f"{t[-2].strip(chr(39))}/{t[-1]}"] = l.split("--job-name=")[1].split()[0]
    for k, v in (ep.get("evaluation_noise") or {}).get("units", {}).items():
        name = evn_names.get(k)
        st = "COMPLETE_VERIFIED" if v["status"] == "COMPLETE" else (q[name][1] if name in q and q[name][1] == "RUNNING" else
                                                                    "QUEUED" if name in q else "NOT_SUBMITTED")
        units.append({"line": "EPFL evaluation-noise re-evaluation (post hoc, DEV)", "unit": k, "state": st,
                      "job": q.get(name, (None,))[0], "evidence": str(HO / "v10_epfl_eval_noise_20261020")})
    st = json.loads(subprocess.run([PY, "/home/infres/yinwang/CS_QMI/ssl_pilot_v10_test_repeat_exec_v6_20261020/scripts/v10_test_repeat/status.py",
                                    "--json"], capture_output=True, text=True, cwd="/home/infres/yinwang/CS_QMI").stdout)
    names = {l.split(" test ")[1].strip(): l.split("--job-name=")[1].split()[0]
             for f in ("SBATCH_LINES.txt", "SBATCH_LINES_V6.txt") for l in open(HO / "v10_test_repeat134_20261020" / f) if " test " in l}
    for u, name in names.items():
        state = ("COMPLETE_VERIFIED" if u in st["units_done"] else "BLOCKED" if u in st["units_blocked"] else
                 q[name][1] if name in q and q[name][1] in ("RUNNING",) else "QUEUED" if name in q else "NOT_SUBMITTED")
        units.append({"line": "TEST_REPEAT134 v5 + v6 CoMIR (same-job DEV32 then TEST134)", "unit": u, "state": state,
                      "job": q.get(name, (None,))[0], "evidence": str(HO / "v10_test_repeat134_20261020")})
    counts = {}
    for x in units:
        counts.setdefault(x["line"], {}).setdefault(x["state"], 0); counts[x["line"]][x["state"]] += 1
    Path(sys.argv[1]).write_text(json.dumps({"schema": "v10.review.status_current.v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                             "input_lock_v5": st["input_lock"], "counts": counts, "units": units}, indent=1) + "\n")
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
