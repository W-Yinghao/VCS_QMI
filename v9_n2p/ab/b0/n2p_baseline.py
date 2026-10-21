"""N2P NCR_A / L2RM_A unit driver for the B0 gate and the B1 units (V9 N2P package, system protocol V9_N2P_SYSTEM_FULLDEV_v1).

    python n2p_baseline.py run --baseline ncr|l2rm --schedule gate|full --seed S --rho 0.4 --unit <new unit dir>

prepare  copy the official clone, apply v9/patches/<b>_anchor_assisted.diff then patches/<b>_n2p_devonly.diff, build a data view
         holding only train_* / dev_* links, export the V9 observed mapping with scripts/v9/n2_official.py export (read guard on)
train    the patched official trainer (native schedule; --n2p_dev_only saves every candidate and skips the native TEST evalrank),
         under n2p_guard.py policy train.  NCR has no co-training resume: a failed attempt is moved to attempts/ and restarted.
forward  n2p_forward.py under n2p_guard.py policy dev (DEV allowlist): full-DEV score matrices + native200 parity
score    the package scorer v9/evidence/evaluate_dev_scores.py once per candidate
select   best full-DEV rSum (exact ties -> earlier candidate) + the two native200 sensitivity rows -> RESULT.json;
         gate: HEALTH.json (pre-registered rule, scripts/v9/n2_ncr_health.py verdict) and B0_DEV_ADAPTER_REPORT.json
Never calls n2_official.py collect; never writes outside --unit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path("/home/infres/yinwang/CS_QMI/ssl_pilot_v9_n2p_ab")
PKG_EVIDENCE = REPO / "v9_n2p/VCS_V9_N2P_V10_Review_Server_Package_v1_20261020/v9/evidence"
PY_DRIVER = "/home/infres/yinwang/CS_QMI/env/bin/python"
DATA_FILES = ("train_ims.npy", "train_caps.txt", "dev_ims.npy", "dev_caps.txt", "dev_ids.txt")
RECALLS = ("i2t_R1", "i2t_R5", "i2t_R10", "t2i_R1", "t2i_R5", "t2i_R10")


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jload(path):
    return json.loads(Path(path).read_text())


def jdump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1))


def gpu_name():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True)
    except OSError:
        return None
    return r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else None


def guard_cmd(py, unit, trace, policy, frozen, script, args):
    return [py, "-I", "-u", str(unit / "adapter/n2p_guard.py"), str(unit / trace), policy, frozen["paths"]["data_dir"], str(script), *args]


def prepare(a, frozen, unit):
    cfg = {"baseline": a.baseline, "schedule": a.schedule, "seed": a.seed, "rho": a.rho}
    if (unit / "unit.json").exists():
        old = jload(unit / "unit.json")
        if {k: old[k] for k in cfg} != cfg:
            raise SystemExit(f"{unit} belongs to another configuration: {old}")
        return old
    if unit.exists() and any(unit.iterdir()):
        raise SystemExit(f"refusing to use a non-empty directory without unit.json: {unit}")
    unit.mkdir(parents=True, exist_ok=True)
    shutil.copytree(HERE, unit / "adapter", ignore=shutil.ignore_patterns("__pycache__"))
    jdump(unit / "adapter/B0_FROZEN.json", frozen)                # the effective frozen config of this unit
    for p in (REPO / "v9_n2p/ab/patches").glob("*.diff"):
        shutil.copy(p, unit / "adapter" / p.name)
    b = a.baseline
    code = unit / "code" / b
    shutil.copytree(frozen["paths"]["clones"][b], code,
                    ignore=shutil.ignore_patterns(".git", "__pycache__", "runs", "output", "noise_index", "*.png", "cc152k"))
    patches = [REPO / f"v9/patches/{b}_anchor_assisted.diff", unit / f"adapter/{b}_n2p_devonly.diff"]
    for p in patches:
        r = subprocess.run(["patch", "-p1", "-s", "-i", str(p)], cwd=code, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"patch failed: {p}\n{r.stdout}{r.stderr}")
    view = unit / "data_view/f30k_precomp"
    view.mkdir(parents=True)
    for f in DATA_FILES:
        (view / f).symlink_to(Path(frozen["paths"]["data_dir"]) / f)
    shared = frozen["paths"]["shared_stage"].format(rho=f"{a.rho:g}", seed=a.seed)
    r = subprocess.run(guard_cmd(PY_DRIVER, unit, "export_trace.jsonl", "train", frozen, REPO / "scripts/v9/n2_official.py",
                                 ["export", "--shared", shared, "--data-root", str(view), "--out", str(unit / "v9_export")]),
                       capture_output=True, text=True)
    (unit / "export.log").write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        raise SystemExit("export failed; see export.log")
    git = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    sched = frozen["schedules"][b][a.schedule]
    info = {**cfg, "unit": str(unit), "method": frozen["families"][b]["method"], "prepared_utc": utc(), "adapter_git_head": git,
            "shared_stage": shared, "patch_sha256": {p.name: sha(p) for p in patches},
            "effective_code_sha256": {str(p.relative_to(code)): sha(p) for p in sorted(code.rglob("*.py"))},
            "export_manifest": jload(unit / "v9_export/v9_export_manifest.json"), "schedule_values": sched,
            "lr_multiplier": frozen["lr_multiplier"]["value"], "frozen_config_sha256": sha(unit / "adapter/B0_FROZEN.json")}
    jdump(unit / "unit.json", info)
    return info


def trainer_args(a, frozen, unit):
    b, s = a.baseline, frozen["schedules"][a.baseline]
    sched = s[a.schedule]
    common = ["--data_path", str(unit / "data_view"), "--vocab_path", frozen["paths"]["vocab_dir"], "--output_dir", str(unit / "official_out"),
              "--num_epochs", str(sched["num_epochs"]), "--lr_update", str(sched["lr_update"]), "--warmup_epoch", str(sched["warmup_epoch"]),
              "--seed", str(a.seed), "--observed_index_file", str(unit / "v9_export/v9_observed_t2i.npy"),
              "--anchor_mask_file", str(unit / "v9_export/v9_anchor_mask.npy"), "--n2p_dev_only", str(unit / "candidates")]
    script = unit / "code" / b / frozen["families"][b]["trainer"]
    return script, s["fixed_args"] + common


def candidates_done(frozen, a, unit):
    want = {c["candidate_id"] for c in frozen["candidate_maps"][a.baseline][a.schedule]}
    led = unit / "candidates/candidates.jsonl"
    got = {json.loads(x)["candidate_id"] for x in led.read_text().splitlines()} if led.exists() else set()
    return want - got


def train(a, frozen, unit):
    rec = unit / "train_receipt.json"
    if rec.exists() and jload(rec).get("rc") == 0 and not candidates_done(frozen, a, unit):
        print("[train] complete, skipped")
        return 0
    if (unit / "official_out").exists() or (unit / "candidates").exists():     # an unfinished attempt: keep it aside, never mix
        k = len(list((unit / "attempts").glob("*"))) if (unit / "attempts").exists() else 0
        dst = unit / "attempts" / f"attempt_{k:02d}"
        dst.mkdir(parents=True)
        for name in ("official_out", "candidates", "train.log", "train_trace.jsonl", "train_receipt.json"):
            if (unit / name).exists():
                (unit / name).rename(dst / name)
    script, args = trainer_args(a, frozen, unit)
    py = frozen["paths"]["envs"][a.baseline] + "/bin/python"
    env = {**os.environ, "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    cmd = guard_cmd(py, unit, "train_trace.jsonl", "train", frozen, script, args)
    t0, start = time.time(), utc()
    with open(unit / "train.log", "a") as log:
        rc = subprocess.run(cmd, cwd=unit / "code" / a.baseline, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    missing = sorted(candidates_done(frozen, a, unit))
    jdump(rec, {"rc": rc, "start_utc": start, "end_utc": utc(), "wall_seconds": round(time.time() - t0, 1), "host": socket.gethostname(),
                "gpu": gpu_name(), "slurm_job": os.environ.get("SLURM_JOB_ID"), "cmd": cmd, "missing_candidates": missing})
    return rc if not missing else (rc or 3)


def forward(a, frozen, unit):
    summ = unit / "dev/FORWARD_SUMMARY.json"
    if summ.exists() and not jload(summ)["missing"]:
        print("[forward] complete, skipped")
        return 0
    py = frozen["paths"]["envs"][a.baseline] + "/bin/python"
    cmd = guard_cmd(py, unit, "forward_trace.jsonl", "dev", frozen, unit / "adapter/n2p_forward.py",
                    ["--baseline", a.baseline, "--unit", str(unit), "--frozen", str(unit / "adapter/B0_FROZEN.json"),
                     "--schedule", a.schedule, "--seed", str(a.seed), "--rho", str(a.rho)])
    t0 = time.time()
    with open(unit / "forward.log", "a") as log:
        rc = subprocess.run(cmd, cwd=unit, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}).returncode
    jdump(unit / "forward_receipt.json", {"rc": rc, "wall_seconds": round(time.time() - t0, 1), "host": socket.gethostname(), "gpu": gpu_name(),
                                          "slurm_job": os.environ.get("SLURM_JOB_ID"), "cmd": cmd})
    return rc


def score(a, frozen, unit):
    out = unit / "dev/full_dev"
    out.mkdir(exist_ok=True)
    t0, rc = time.time(), 0
    for c in frozen["candidate_maps"][a.baseline][a.schedule]:
        cfg = unit / "dev/score_inputs" / (c["file"].split(".")[0] + ".json")
        res = out / cfg.name
        if not cfg.exists() or res.exists():
            continue
        r = subprocess.run(guard_cmd(PY_DRIVER, unit, "score_trace.jsonl", "dev", frozen, PKG_EVIDENCE / "evaluate_dev_scores.py",
                                     ["--config", str(cfg), "--out", str(res)]), cwd=unit, capture_output=True, text=True)
        if r.returncode != 0:
            (out / (cfg.stem + ".error.txt")).write_text(r.stdout + r.stderr)
            rc = 4
    jdump(unit / "score_receipt.json", {"rc": rc, "wall_seconds": round(time.time() - t0, 1), "scorer_sha256": sha(PKG_EVIDENCE / "evaluate_dev_scores.py")})
    return rc


def trace_summary(unit, name):
    p = unit / name
    rows = [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
    data = sorted({r["real"] for r in rows if r["dataset_file"]})
    return {"trace": str(p), "trace_sha256": sha(p) if p.exists() else None, "opens_logged": len(rows),
            "dataset_files_opened": data, "denied": [r for r in rows if r["decision"] == "DENY"],
            "test_or_eval_only_opened": [r["real"] for r in rows if "eval_only" in r["real"] or (r["dataset_file"] and Path(r["real"]).name.startswith("test"))]}


def select(a, frozen, unit):
    fam = frozen["families"][a.baseline]
    cmap = frozen["candidate_maps"][a.baseline][a.schedule]
    fwd = jload(unit / "dev/FORWARD_SUMMARY.json")
    fwd_by = {c["candidate_id"]: c for c in fwd["candidates"]}
    ledger = {json.loads(x)["candidate_id"]: json.loads(x) for x in (unit / "candidates/candidates.jsonl").read_text().splitlines()}
    rows, missing = [], []
    for c in cmap:
        res = unit / "dev/full_dev" / (c["file"].split(".")[0] + ".json")
        if not res.exists():
            missing.append(c["candidate_id"])
            continue
        r = jload(res)
        assert r["status"] == "SCORE_METRICS_VERIFIED_ONLY" and r["candidate_id"] == c["candidate_id"]
        rows.append({**c, "metrics": r["metrics"], "result": str(res), "checkpoint_sha256": fwd_by[c["candidate_id"]]["checkpoint_sha256"],
                     "native200": fwd_by[c["candidate_id"]]["native200"], "trainer_native_rsum": (ledger.get(c["candidate_id"], {}).get("native_dev") or {}).get("rsum")})
    if not rows:
        raise SystemExit("no scored candidate")
    best = max(rows, key=lambda r: (r["metrics"]["rsum"], -r["order"]))            # exact ties -> earlier candidate
    mains = [r for r in rows if r["phase"] == "main"]
    nat_pool = mains if a.baseline == "ncr" else rows
    nat_pool = [r for r in nat_pool if r["trainer_native_rsum"] is not None]
    native_sel = max(nat_pool, key=lambda r: (r["trainer_native_rsum"], -r["order"])) if nat_pool else None
    n200 = max(rows, key=lambda r: (r["native200"]["fulldev_subset"]["rsum"], -r["order"]))
    pick = lambda r: None if r is None else {"candidate_id": r["candidate_id"], "phase": r["phase"], "epoch": r["epoch"],
                                              "fullDEV": r["metrics"], "native200_fulldev_subset": r["native200"]["fulldev_subset"],
                                              "trainer_native_rsum": r["trainer_native_rsum"], "checkpoint_sha256": r["checkpoint_sha256"],
                                              "checkpoint": str(unit / "candidates" / r["file"])}
    rec = {n: jload(unit / f) for n, f in (("train", "train_receipt.json"), ("forward", "forward_receipt.json"), ("score", "score_receipt.json"))
           if (unit / f).exists()}
    result = {"schema": "v9-n2p-system-unit-result-v1", "system_protocol_id": frozen["package"]["system_protocol_id"], **jload(unit / "unit.json"),
              "row_label": fam["row_label"], "selection_rule": frozen["selection"]["primary"],
              "candidates_expected": len(cmap), "candidates_scored": len(rows), "missing_candidates": missing,
              "primary_fullDEV_best": pick(best), "last": pick(mains[-1]) if mains else None,
              "mean_last5_rsum": sum(r["metrics"]["rsum"] for r in mains[-5:]) / len(mains[-5:]) if len(mains) >= 5 else None,
              "native_sensitivity": {"official_native_selector": pick(native_sel), "native200_recomputed_best": pick(n200),
                                     "note": "native200 (200 images / 1000 captions) is a separate sensitivity; never compared with fullDEV1014"},
              "trajectory": [{"candidate_id": r["candidate_id"], "fullDEV_rsum": r["metrics"]["rsum"],
                              "native200_subset_rsum": r["native200"]["fulldev_subset"]["rsum"], "trainer_native_rsum": r["trainer_native_rsum"]} for r in rows],
              "costs": {k: {kk: v.get(kk) for kk in ("wall_seconds", "host", "gpu", "slurm_job")} for k, v in rec.items()},
              "read_trace": {s: trace_summary(unit, f"{s}_trace.jsonl") for s in ("export", "train", "forward", "score")}}
    jdump(unit / "RESULT.json", result)
    if a.schedule == "gate":
        gate_report(a, frozen, unit, result, rows, fwd)
    return 0


def gate_report(a, frozen, unit, result, rows, fwd):
    sys.path.insert(0, str(REPO / "scripts/v9"))
    from n2_ncr_health import verdict                   # the pre-registered rule, reused unchanged
    from n2_official_smoke import dev_trajectory
    log = (unit / "train.log").read_text(errors="replace")
    traj = dev_trajectory(log)
    warm = frozen["schedules"][a.baseline]["gate"]["warmup_epoch"]
    n_main = frozen["schedules"][a.baseline]["gate"]["num_epochs"]
    ref_traj = traj if a.baseline == "ncr" else traj[warm - 1:]   # NCR prints one post-warm-up validation; L2RM one per warm-up epoch
    h = verdict(ref_traj, jload(unit / "train_receipt.json")["rc"] == 0)
    h.update(observed_index_loaded="load V9 observed index" in log, anchor_term_lines=log.count("[V9] anchor-supervised term"),
             traceback="Traceback" in log, rule=frozen["health"][a.baseline])
    h["pass"] = bool(h["pass"] and h["observed_index_loaded"] and h["anchor_term_lines"] >= n_main and not h["traceback"])
    jdump(unit / "HEALTH.json", h)
    # hook check: candidates equal the trainer's own checkpoints of the same epoch (NCR warm-up file, best-epoch files)
    import torch
    out, same = unit / "official_out", []
    pairs = []
    if a.baseline == "ncr":
        pairs.append(("warmup:%d" % (warm - 1), out / f"warmup_model_{warm - 1}.pth.tar", ["model_A", "model_B"]))
        pairs += [(f"main:{int(p.name.split('_')[1].split('.')[0])}", p, ["model_A", "model_B"]) for p in sorted(out.glob("checkpoint_*.pth.tar"))]
    else:
        pairs += [(f"warmup:{int(p.name.split('_')[2].split('.')[0])}", p, ["net"]) for p in sorted(out.glob("warm_checkpoint_*.pth.tar"))]
        pairs += [(f"main:{int(p.name.split('_')[1].split('.')[0])}", p, ["net"]) for p in sorted(out.glob("checkpoint_*.pth.tar"))]
    flat = lambda sd: [t for d in (sd if isinstance(sd, list) else [sd]) for t in d.values()]
    for cid, native_file, keys in pairs:
        cfile = unit / "candidates" / ("%s_%02d.pth.tar" % (cid.split(":")[0], int(cid.split(":")[1])))
        c, n = torch.load(cfile, map_location="cpu", weights_only=False), torch.load(native_file, map_location="cpu", weights_only=False)
        eq = all(len(flat(c[k])) == len(flat(n[k])) and all(torch.equal(x, y) for x, y in zip(flat(c[k]), flat(n[k]))) for k in keys)
        same.append({"candidate_id": cid, "native_file": native_file.name, "tensors_equal": eq})
    tol = frozen["native200_parity"]["tolerance_max_abs_similarity"]
    printed = traj                                          # trainer-printed native validations (0.1 precision), in print order
    parity = [{"candidate_id": c["candidate_id"], **c["native200"]} for c in fwd["candidates"]]
    tr = result["read_trace"]
    checks = {
        "1_ids_and_scores": {"pass": all(fwd["dev_identity"].values()) and len(rows) == result["candidates_expected"],
                             "dev_identity": fwd["dev_identity"], "scored": len(rows), "expected": result["candidates_expected"],
                             "scorer_status": "SCORE_METRICS_VERIFIED_ONLY for every scored candidate"},
        "2_candidates_and_hook": {"pass": not result["missing_candidates"] and bool(same) and all(s["tensors_equal"] for s in same),
                                  "networks_per_candidate": frozen["families"][a.baseline]["networks"], "trainer_checkpoint_equality": same},
        "3_native200_parity": {"pass": all(fwd["dev_identity"]["native200_is_exact_id_prefix"] and p["max_abs_sim_diff"] <= tol for p in parity),
                               "tolerance": tol, "per_candidate": parity, "trainer_printed_validations": printed},
        "4_native_fidelity_health": {"pass": h["pass"], "health": h},
        "5_no_test_read": {"pass": all(not v["denied"] and not v["test_or_eval_only_opened"] and v["trace_sha256"] for v in tr.values())
                           and "[N2P] native end-of-training TEST evaluation disabled" in log,
                           "dataset_files_by_process": {k: v["dataset_files_opened"] for k, v in tr.items()}, "collect_called": False},
        "6_costs": {"pass": None, "costs": result["costs"], "per_candidate_forward_seconds": [c["seconds"] for c in fwd["candidates"]],
                    "peak_gpu_mem_mb": max([c["peak_gpu_mem_mb"] for c in fwd["candidates"] if c["peak_gpu_mem_mb"] is not None], default=None),
                    "candidate_bytes": [c["checkpoint_bytes"] for c in fwd["candidates"]]}}
    status = "B0_PASS" if all(v["pass"] for k, v in checks.items() if k != "6_costs") else "B0_FAIL"
    jdump(unit / "B0_DEV_ADAPTER_REPORT.json", {"schema": "v9-n2p-b0-dev-adapter-report-v1", "status": status, "baseline": a.baseline,
                                                "method": frozen["families"][a.baseline]["method"], "unit": str(unit), "checks": checks,
                                                "item6_note": "costs are recorded for the B1 time check; not a pass/fail item"})
    print(status)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("step", choices=["run", "prepare", "train", "forward", "score", "select"])
    ap.add_argument("--baseline", required=True, choices=["ncr", "l2rm"])
    ap.add_argument("--schedule", required=True, help="gate | full (keys of the frozen config)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rho", type=float, default=0.4)
    ap.add_argument("--unit", required=True)
    ap.add_argument("--frozen", default=str(HERE / "B0_FROZEN.json"), help="frozen config (default: the committed B0_FROZEN.json)")
    a = ap.parse_args()
    frozen = jload(a.frozen)
    if a.schedule not in frozen["candidate_maps"][a.baseline]:
        raise SystemExit("unknown schedule " + a.schedule)
    unit = Path(a.unit)
    print(f"[n2p] {utc()} {a.step} {a.baseline} {a.schedule} s{a.seed} rho{a.rho} host={socket.gethostname()} job={os.environ.get('SLURM_JOB_ID')}", flush=True)
    steps = {"prepare": lambda: (prepare(a, frozen, unit), 0)[1], "train": lambda: train(a, frozen, unit), "forward": lambda: forward(a, frozen, unit),
             "score": lambda: score(a, frozen, unit), "select": lambda: select(a, frozen, unit)}
    order = ["prepare", "train", "forward", "score", "select"] if a.step == "run" else [a.step]
    if a.step != "prepare" and a.step != "run":
        prepare(a, frozen, unit)
    for s in order:
        rc = steps[s]()
        print(f"[n2p] {utc()} {s} rc={rc}", flush=True)
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
