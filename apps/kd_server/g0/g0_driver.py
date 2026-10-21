#!/usr/bin/env python
"""KD G0 CUDA preflight driver. Pre-declared stages, seed and gates: G0_PLAN.json (same directory).

Uses only the package's own entry points and functions (`kd make-grid`, `kd train`, `kd train --resume`,
kd.runner, vcs_common); changes no package file; never loads the official CIFAR-100 TEST. Each stage records
PASS/FAIL/WARN and never aborts the later stages; G0_SUMMARY.json is rewritten after every stage.

    python g0_driver.py --out DIR --device cuda [--stages all | env,gpu_smoke,...]
    python g0_driver.py --out DIR --device cpu --selftest     # S1 CPU job: exercises this driver's own code

--selftest: the real-data stages run on the gpu_smoke synthetic fixture (tiny teacher/students, CPU) and the
microbenchmark is shortened; results only show that the driver mechanics work, never anything about CIFAR.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLAN_PATH = HERE / "G0_PLAN.json"
PLAN = json.loads(PLAN_PATH.read_text())
KD_ROOT = Path(os.environ.get("KD_ROOT", "/home/infres/yinwang/CS_QMI/outputs_apps/kd_20261020"))
KD_PKG = Path(os.environ.get("KD_PKG", "/home/infres/yinwang/CS_QMI/ssl_pilot_app_kd/apps/VCS_QMI_KD_AIGIQA_Server_Package_v1_20261019"))
PRIMARY = KD_PKG / "kd" / "configs" / "primary_wrn40_2_to_wrn16_2.json"
SEED = int(PLAN["instrumentation_seed"])
FULL_EPOCHS = 240
INSTRUMENT = ["dkd", "dkd_bce", "dkd_vcs_exact"]
TIMING = ["student", "kd", "dkd_mse", "dkd_cosine", "dkd_vid_pooled", "dkd_vcs_sampled", "dkd_vcs_m1",
          "dkd_infonce", "crd_original", "dkd_crd_uniform", "itrd_reference", "dkd_itrd"]
LAMBDA_FAMILIES = ["dkd_mse", "dkd_cosine", "dkd_vid_pooled", "dkd_bce", "dkd_vcs_exact", "dkd_vcs_sampled",
                   "dkd_vcs_m1", "dkd_infonce", "dkd_itrd"]
BOUNDED = ("bce", "vcs_exact", "vcs_sampled", "vcs_m1")
STAGES = ["env", "gpu_smoke", "cuda_math", "instrument", "resume", "timing", "stability", "microbench"]
GATED = ["env", "gpu_smoke", "instrument", "resume", "timing"]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def jread(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def jwrite(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def child_env():
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", f"{KD_PKG}{os.pathsep}{KD_PKG / 'shared'}")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run_logged(cmd, log, kill_when=None, poll=0.2):
    """Run a package CLI command from KD_ROOT; timestamp every output line; optionally SIGKILL when kill_when()."""
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    lines = []
    proc = subprocess.Popen([str(c) for c in cmd], cwd=KD_ROOT, env=child_env(), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)

    def reader():
        with log.open("a") as fh:
            fh.write(f"# {now()} cwd={KD_ROOT} cmd={' '.join(map(str, cmd))}\n")
            for line in proc.stdout:
                t = time.time() - t0
                fh.write(f"[{t:9.2f}] {line}")
                fh.flush()
                lines.append((t, line.rstrip("\n")))

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    killed = False
    while proc.poll() is None:
        if kill_when is not None and not killed and kill_when():
            os.kill(proc.pid, signal.SIGKILL)
            killed = True
        time.sleep(poll)
    thread.join(timeout=60)  # orphaned DataLoader workers may hold the pipe for a few seconds
    epochs = []
    for t, line in lines:
        if line.startswith("{") and '"cal_top1"' in line:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            epochs.append({"epoch": d.get("epoch"), "t": t})
    return {"rc": proc.returncode, "seconds": time.time() - t0, "killed": killed, "epoch_lines": epochs,
            "log": str(log), "tail": [l for _, l in lines[-15:]]}


def make_configs(base, out_dir, arms, coefficient, epochs, log_dir):
    """The package's own grid expansion; the G0-only copies differ from it only in training.epochs."""
    res = run_logged([sys.executable, "-m", "kd", "make-grid", "--base", base, "--out", out_dir, "--arms", *arms,
                      "--seeds", str(SEED), "--coefficients", f"{coefficient:g}"], Path(log_dir) / f"make_grid_{Path(out_dir).name}.log")
    if res["rc"] != 0:
        raise RuntimeError(f"make-grid failed: {res['tail']}")
    configs = {}
    for row in jread(Path(out_dir) / "grid_manifest.json")["rows"]:
        path = Path(out_dir) / row["config"]
        cfg = json.loads(path.read_text())
        cfg["training"]["epochs"] = int(epochs)
        path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
        configs[row["method"]] = path
    return configs


def train_run(config, run_dir, log, resume=False, kill_when=None, poll=0.2):
    cmd = [sys.executable, "-m", "kd", "train", "--config", config, "--out", run_dir] + (["--resume"] if resume else [])
    return run_logged(cmd, log, kill_when=kill_when, poll=poll)


def summarize(run_dir, proc):
    run_dir = Path(run_dir)
    run = jread(run_dir / "run.json", {}) or {}
    hist = jread(run_dir / "history.json", []) or []
    files = {q.name: q.stat().st_size for q in sorted(run_dir.iterdir()) if q.is_file()} if run_dir.is_dir() else {}
    times = [e["t"] for e in proc["epoch_lines"]]
    deltas = [b - a for a, b in zip(times, times[1:])]
    steady = statistics.mean(deltas) if deltas else None
    return {
        "status": run.get("status"), "failure": run.get("failure"), "rc": proc["rc"], "log": proc["log"],
        "epochs_done": len(hist),
        "train_epoch_seconds": [h.get("train_epoch_wall_seconds") for h in hist],
        "cal_inference_seconds": [h.get("cal", {}).get("inference_loader_wall_seconds") for h in hist],
        "epoch_line_times": times, "steady_epoch_wall_seconds": steady,
        "startup_plus_first_epoch_seconds": times[0] if times else None,
        "estimated_240_epoch_seconds": (times[0] + (FULL_EPOCHS - 1) * steady) if steady else None,
        "process_seconds": proc["seconds"],
        "cuda_peak_allocated_bytes": run.get("cuda_peak_allocated_bytes"),
        "parameters_student": run.get("parameters_student"), "parameters_auxiliary": run.get("parameters_auxiliary"),
        "student_initial_state_sha256": run.get("student_initial_state_sha256"),
        "teacher_frozen_verified": run.get("teacher_frozen_verified"),
        "file_bytes": files,
        "history_bytes_per_epoch": (files.get("history.json", 0) / len(hist)) if hist else None,
        "gradient_probe": [h.get("gradient_probe") for h in hist],
        "dependence_last_batch": [h.get("dependence") for h in hist],
        "dependence_aggregate": [h.get("dependence_aggregate") for h in hist],
        "loss_parts": [h.get("loss_parts") for h in hist],
        "next_lr": [h.get("next_lr") for h in hist],
        "cal_top1": [h.get("cal", {}).get("top1") for h in hist],
        "cal_nll": [h.get("cal", {}).get("nll") for h in hist],
        "fit_loss": [h.get("fit_loss") for h in hist],
        "environment": run.get("environment"),
    }


def head_info(run_dir):
    import torch
    path = Path(run_dir) / "last.pt"
    if not path.is_file():
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    w = ckpt["losses"].get("dependence.head.W")
    if w is None:
        return None
    return {"checkpoint_epoch": int(ckpt["epoch"]), "W_frobenius": float(w.norm()), "W_absmax": float(w.abs().max()),
            "W_nonzero_fraction": float((w != 0).float().mean()), "W_initial": "zeros (SimplexBilinearCritic)"}


def cleanup(run_dir):
    removed = {}
    for name in ("last.pt", "best.pt", "best_cal_predictions.csv"):
        path = Path(run_dir) / name
        if path.is_file():
            removed[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
            path.unlink()
    return removed


class Ctx:
    def __init__(self, out, device, selftest):
        self.out, self.device, self.selftest = out, device, selftest
        self.base = PRIMARY
        self.rows = {}
        self.instrument_configs = {}


# ----------------------------------------------------------------------------- stages

def stage_env(ctx):
    import torch
    from vcs_common.runtime import runtime_record
    record = runtime_record()
    ok = ctx.device == "cpu" or (torch.cuda.is_available() and torch.cuda.device_count() >= 1)
    return {"status": "PASS" if ok else "FAIL", "runtime": record, "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_partition": os.environ.get("SLURM_JOB_PARTITION")}


def stage_gpu_smoke(ctx):
    """The package `kd smoke` recipe with an explicit device, plus the two official-CRD arms."""
    from kd.__main__ import ARMS, arm_config
    from kd.data import prepare, write_json
    from kd.runner import compare, evaluate, freeze, train
    out = ctx.out / "gpu_smoke"
    out.mkdir(parents=True, exist_ok=False)
    prepare(out / "data", out / "roles.json", synthetic=True)
    base = {"schema": "kd_config_v1", "name": "smoke", "seed": 27182, "device": ctx.device, "threads": 1,
            "track": "synthetic_smoke", "data": {"root": str(out / "data"), "manifest": str(out / "roles.json")},
            "teacher": {"backend": "tiny", "name": "tiny_teacher"}, "student": {"backend": "tiny", "name": "tiny_student"},
            "loss": {"anchor": "none", "feature": "none"},
            "optimizer": {"lr": .03, "momentum": .9, "weight_decay": .0005, "head_weight_decay": 0., "milestones": [2], "gamma": .1},
            "training": {"epochs": 2, "batch_size": 16, "workers": 0}}
    write_json(out / "teacher_config.json", base)
    train(out / "teacher_config.json", out / "teacher", teacher_training=True)
    base["teacher"]["checkpoint"] = str(out / "teacher" / "teacher.pt")
    base["teacher"]["provenance"] = str(out / "teacher" / "teacher_provenance.json")
    write_json(out / "selftest_base.json", base)
    arms = [a for a in ARMS if "crd" not in a] + [a for a in ARMS if "crd" in a]
    runs, errors, checks = {}, {}, {}
    for arm in arms:
        cfg = arm_config(base, arm, .1)
        if "dependence" in cfg["loss"]:
            cfg["loss"]["dependence"].update(rank=4, solver_max_iter=10)
        write_json(out / f"{arm}.json", cfg)
        try:
            train(out / f"{arm}.json", out / arm)
            runs[arm] = out / arm
        except Exception as exc:
            errors[arm] = f"{type(exc).__name__}: {exc}"
    evaluated = False
    if runs and not errors:
        freeze(list(runs.values()), out / "freeze.json")
        for path in runs.values():
            evaluate(path, out / "freeze.json")  # synthetic fixture TEST only
        compare(list(runs.values()), out / "comparison.json")
        evaluated = True
    for arm, folder in runs.items():
        run, hist = jread(folder / "run.json", {}), jread(folder / "history.json", [])
        c = {"complete": run.get("status") == "complete", "device": run.get("config", {}).get("device") == ctx.device,
             "test_evaluated": (folder / "test_metrics.json").is_file()}
        if arm != "student":
            c["teacher_frozen_verified"] = run.get("teacher_frozen_verified") is True
        method = run.get("config", {}).get("loss", {}).get("dependence", {}).get("method")
        if method in BOUNDED:
            c["head_bounded"] = all(e.get("dependence", {}).get("head_abs_max", 2.) <= 1 + 1e-6 for e in hist)
        if method == "vcs_m1":
            c["m1_solver_stats"] = all(e.get("dependence_aggregate", {}).get("solver_n", 0) > 0 for e in hist)
        if arm not in ("student", "kd", "dkd"):
            c["gradient_probe_finite"] = all(e.get("gradient_probe") and math.isfinite(e["gradient_probe"]["weighted_representation_gradient_norm"]) for e in hist)
        checks[arm] = c
    ok = not errors and evaluated and len(runs) == len(arms) and all(all(v for v in c.values()) for c in checks.values())
    return {"status": "PASS" if ok else "FAIL", "arms": arms, "errors": errors, "checks": checks,
            "teacher_cuda_peak": jread(out / "teacher" / "run.json", {}).get("cuda_peak_allocated_bytes")}


def stage_cuda_math(ctx):
    import torch
    from vcs_common.mf_backend import SceneBatch, dense_objective_terms, objective_terms
    dev = torch.device(ctx.device)
    gen = torch.Generator().manual_seed(20261020)
    rows = []
    for n in (64, 256, 1024):
        r = 32
        a32 = torch.softmax(torch.randn(n, r, generator=gen), -1)
        b32 = torch.softmax(torch.randn(n, r, generator=gen), -1)
        w32 = torch.rand(r, r, generator=gen) * 2 - 1
        # Independent float64 dense formula on the SAME float32-rounded inputs (U: all i != j pairs).
        a64, b64, w64 = (x.double().requires_grad_(True) for x in (a32, b32, w32))
        s = a64 @ w64 @ b64.T
        t, off = s.diagonal(), s[~torch.eye(n, dtype=torch.bool)]
        j64 = t.mean() - off.mean() - 0.5 * (t.square().mean() + off.square().mean())
        g64 = torch.autograd.grad(j64, (a64, b64, w64))
        for name, fn in (("exact_mf", objective_terms), ("dense", dense_objective_terms)):
            a, b, w = (x.to(dev).clone().requires_grad_(True) for x in (a32, b32, w32))
            terms = fn([SceneBatch(a, b, q_mode="u")], w, weights=[1.0])
            grads = torch.autograd.grad(terms.j, (a, b, w))
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            rel = [float((x.detach().double().cpu() - y).norm() / y.norm()) for x, y in zip(grads, g64)]
            rows.append({"n": n, "rank": r, "impl": name, "dtype": "float32", "device": str(dev), "q_mode": "u",
                         "J_reference_float64": float(j64), "abs_dJ": abs(float(terms.j) - float(j64)),
                         "rel_grad_a": rel[0], "rel_grad_b": rel[1], "rel_grad_W": rel[2]})
    worst_dj = max(r["abs_dJ"] for r in rows)
    worst_rel = max(max(r["rel_grad_a"], r["rel_grad_b"], r["rel_grad_W"]) for r in rows)
    return {"status": "PASS" if worst_dj <= 1e-5 and worst_rel <= 1e-4 else "WARN", "rows": rows,
            "worst_abs_dJ": worst_dj, "worst_rel_grad": worst_rel,
            "scope": "fixed random simplex features; not a guarantee for arbitrary CUDA reductions (INSTALL §5)"}


def _base_for_real_stages(ctx):
    if ctx.selftest:
        base = ctx.out / "gpu_smoke" / "selftest_base.json"
        if not base.is_file():
            raise RuntimeError("selftest needs the gpu_smoke stage first")
        return base
    return PRIMARY


def stage_instrument(ctx):
    root = ctx.out / "instrument"
    configs = make_configs(_base_for_real_stages(ctx), root / "configs", INSTRUMENT, 1, 3, root / "logs")
    ctx.instrument_configs = configs
    rows = {}
    for arm in INSTRUMENT:
        run_dir = root / "runs" / arm
        proc = train_run(configs[arm], run_dir, root / "logs" / f"{arm}.log")
        rows[arm] = summarize(run_dir, proc)
        rows[arm]["head_final"] = head_info(run_dir)
        rows[arm]["removed_files"] = cleanup(run_dir)
    ctx.rows["instrument"] = rows
    gates = {"all_complete_3_epochs": all(r["status"] == "complete" and r["epochs_done"] == 3 for r in rows.values()),
             "teacher_frozen_verified": all(r["teacher_frozen_verified"] is True for r in rows.values()),
             "same_student_initial_state_hash": len({r["student_initial_state_sha256"] for r in rows.values()}) == 1}
    info = {}
    for arm in ("dkd_bce", "dkd_vcs_exact"):
        r = rows[arm]
        probes, dep = r["gradient_probe"], r["dependence_last_batch"]
        norms = [p["weighted_representation_gradient_norm"] if p else None for p in probes]
        info[arm] = {"aux_grad_norm_first_batch_per_epoch": norms,
                     "aux_over_ce_first_batch_per_epoch": [p["aux_over_ce"] if p else None for p in probes],
                     "W_absmax_last_batch_per_epoch": [d.get("head_abs_max") if d else None for d in dep],
                     "raw_j_mean_per_epoch": [a.get("raw_j_mean") if a else None for a in r["dependence_aggregate"]],
                     "head_final": r["head_final"],
                     "note": "epoch-0 first batch has W=0, where the bilinear critic's feature gradient is exactly 0 by construction"}
        gates[f"{arm}_aux_grad_positive_after_head_updates"] = len(norms) >= 3 and all(v is not None and v > 0 for v in norms[1:3])
        gates[f"{arm}_W_moved_after_epoch0"] = bool(dep) and (dep[0] or {}).get("head_abs_max", 0) > 0
        gates[f"{arm}_J_finite"] = bool(dep) and all(d and math.isfinite(d.get("raw_j", float("nan"))) for d in dep)
    return {"status": "PASS" if all(gates.values()) else "FAIL", "gates": gates, "dependence_evidence": info, "runs": rows}


def stage_resume(ctx):
    import torch
    if "dkd_vcs_exact" not in ctx.instrument_configs or "instrument" not in ctx.rows:
        raise RuntimeError("resume needs the instrument stage (same config, uninterrupted reference)")
    root = ctx.out / "resume"
    config = ctx.instrument_configs["dkd_vcs_exact"]
    run_dir = root / "run"

    def kill_when():
        hist = jread(run_dir / "history.json")
        return isinstance(hist, list) and len(hist) >= 2 and (run_dir / "last.pt").is_file()

    poll = 0.02 if ctx.selftest else 0.2
    first = train_run(config, run_dir, root / "logs" / "first.log", kill_when=kill_when, poll=poll)
    pre_hist = jread(run_dir / "history.json", []) or []
    pre_run = jread(run_dir / "run.json", {}) or {}
    pre_epoch = int(torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)["epoch"]) if (run_dir / "last.pt").is_file() else None
    if not first["killed"]:
        return {"status": "NOT_EXERCISED" if ctx.selftest else "FAIL", "reason": "run finished before the kill point",
                "first": first}
    second = train_run(config, run_dir, root / "logs" / "resume.log", resume=True)
    post_hist = jread(run_dir / "history.json", []) or []
    post_run = jread(run_dir / "run.json", {}) or {}
    ref = ctx.rows["instrument"]["dkd_vcs_exact"]
    checks = {"killed_mid_run": first["killed"] and first["rc"] != 0,
              "pre_kill_receipt_state_running": pre_run.get("status") == "running",
              "pre_kill_checkpoint_epoch_1_with_2_history_entries": pre_epoch == 1 and len(pre_hist) == 2,
              "resume_rc_0": second["rc"] == 0,
              "resumed_complete_3_epochs": post_run.get("status") == "complete" and post_run.get("epochs_completed") == 3 and len(post_hist) == 3,
              "pre_kill_history_preserved_exactly": post_hist[:2] == pre_hist,
              "lr_schedule_equals_uninterrupted": [h.get("next_lr") for h in post_hist] == ref["next_lr"]}
    deltas = []
    for e, h in enumerate(post_hist):
        if e < len(ref["cal_top1"]):
            deltas.append({"epoch": e, "abs_d_cal_top1": abs(h["cal"]["top1"] - ref["cal_top1"][e]),
                           "abs_d_cal_nll": abs(h["cal"]["nll"] - ref["cal_nll"][e]),
                           "abs_d_fit_loss": abs(h["fit_loss"] - ref["fit_loss"][e])})
    removed = cleanup(run_dir)
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "metric_differences_vs_uninterrupted": deltas,
            "note": "epochs 0-1 come from the killed invocation; differences there measure GPU run-to-run nondeterminism",
            "removed_files": removed, "kill_seconds": first["seconds"], "resume_seconds": second["seconds"]}


def _family_runs(ctx, stage, arms, coefficient, epochs):
    root = ctx.out / stage
    configs = make_configs(_base_for_real_stages(ctx), root / "configs", arms, coefficient, epochs, root / "logs")
    rows = {}
    for arm in arms:
        run_dir = root / "runs" / arm
        proc = train_run(configs[arm], run_dir, root / "logs" / f"{arm}.log")
        rows[arm] = summarize(run_dir, proc)
        rows[arm]["removed_files"] = cleanup(run_dir)
    ctx.rows[stage] = rows
    return rows


def stage_timing(ctx):
    rows = _family_runs(ctx, "timing", TIMING, 1, 2)
    failed = {a: (r["failure"] or f"rc={r['rc']}") for a, r in rows.items()
              if not (r["status"] == "complete" and r["epochs_done"] == 2)}
    return {"status": "PASS" if not failed else "FAIL", "failed": failed, "runs": rows}


def stage_stability(ctx):
    rows = _family_runs(ctx, "stability", LAMBDA_FAMILIES, 100, 2)
    outcome = {a: ("complete" if r["status"] == "complete" and r["epochs_done"] == 2 else
                   f"{r['status']}: {(r['failure'] or {}).get('type')}: {(r['failure'] or {}).get('message')}") for a, r in rows.items()}
    return {"status": "REPORTED", "lambda": 100, "outcome": outcome, "runs": rows,
            "meaning": "finiteness over 2 epochs only; a failure here predicts a FAILED w100 CAL candidate, which blocks seal"}


def stage_microbench(ctx):
    out = ctx.out / "microbench.json"
    extra = ["--sizes", "32", "128", "--ranks", "16", "--repeats", "3", "--warmup", "1"] if ctx.selftest else []
    res = run_logged([sys.executable, "-m", "vcs_common.probes", "benchmark", "--device", ctx.device, "--output", out, *extra],
                     ctx.out / "logs" / "microbench.log")
    return {"status": "PASS" if res["rc"] == 0 and out.is_file() else "FAIL", "output": str(out), "rc": res["rc"],
            "scope": "dependence branch only (projection+loss+backward); excludes encoder/optimizer"}


FUNCS = {"env": stage_env, "gpu_smoke": stage_gpu_smoke, "cuda_math": stage_cuda_math, "instrument": stage_instrument,
         "resume": stage_resume, "timing": stage_timing, "stability": stage_stability, "microbench": stage_microbench}


def family_table(ctx):
    keys = ("status", "epochs_done", "steady_epoch_wall_seconds", "startup_plus_first_epoch_seconds",
            "estimated_240_epoch_seconds", "train_epoch_seconds", "cal_inference_seconds", "cuda_peak_allocated_bytes",
            "file_bytes", "history_bytes_per_epoch", "parameters_student", "parameters_auxiliary")
    table = {"w1": {}, "w100": {}}
    for stage in ("instrument", "timing"):
        for arm, r in ctx.rows.get(stage, {}).items():
            table["w1"][arm] = dict({k: r.get(k) for k in keys}, source_stage=stage)
    for arm, r in ctx.rows.get("stability", {}).items():
        table["w100"][arm] = {k: r.get(k) for k in keys}
    return table


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--stages", default="all")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    stages = STAGES if args.stages == "all" else [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise SystemExit(f"unknown stages {unknown}")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(KD_ROOT)
    ctx = Ctx(out, args.device, args.selftest)
    summary = {"schema": "kd_server_g0_summary_v1", "started_utc": now(), "plan_sha256": sha256(PLAN_PATH),
               "driver_sha256": sha256(__file__), "device": args.device, "selftest": args.selftest,
               "hostname": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
               "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"), "instrumentation_seed": SEED,
               "stages_requested": stages, "stages": {}}
    for stage in stages:
        t0 = time.time()
        print(f"== G0 stage {stage} {now()}", flush=True)
        try:
            result = FUNCS[stage](ctx)
        except Exception as exc:
            result = {"status": "FAIL", "exception": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        result["seconds"] = round(time.time() - t0, 1)
        summary["stages"][stage] = result
        summary["families"] = family_table(ctx)
        jwrite(out / "G0_SUMMARY.json", summary)
        print(f"== G0 stage {stage}: {result['status']} ({result['seconds']} s)", flush=True)
    gated = [s for s in GATED if s in stages]
    summary["overall"] = "PASS" if gated and all(summary["stages"][s]["status"] == "PASS" for s in gated) else "FAIL"
    if args.selftest:
        summary["overall_meaning"] = "selftest: driver mechanics on the synthetic CPU fixture only"
    summary["finished_utc"] = now()
    jwrite(out / "G0_SUMMARY.json", summary)
    print(json.dumps({"overall": summary["overall"], "stages": {s: summary["stages"][s]["status"] for s in stages}}), flush=True)
    raise SystemExit(0 if summary["overall"] == "PASS" else 1)


if __name__ == "__main__":
    main()
