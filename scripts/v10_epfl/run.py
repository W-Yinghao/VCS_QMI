#!/usr/bin/env python3
"""V10 EPFL production runner (main-session decisions 2026-10-19; release form of scripts/v10_supplement/run.py).

  python -I scripts/v10_epfl/run.py cal   <family>               --release <RELEASE.json>     # seed 1017, frozen candidate order, budget ledger
  python -I scripts/v10_epfl/run.py final <family> --seed <s>    --release <RELEASE.json>     # s in 1001-1003, after the family's CAL selection
      (the release may be the 10-family release, its finals-only hardware version, or a hardware diagnostic release: L40S reruns of the
       RTX PRO 6000 finals, or their L40S-vs-L40S control, into their own results root, CAL selection read from the parent results, label
       in the identity)
  python -I scripts/v10_epfl/run.py smoke --device cpu --out <dir> [--families F1,F2]          # CPU code-path smoke (relabelled roles, no DEV read)

One CAL family or one final seed per job; no submission or queue mutation here.  CAL: every candidate trains 3000 steps (or its native
schedule) and is read out on CAL (10 windows x 10 perturbations, pooled by registration) at every prescribed checkpoint; the lexicographic
selection is over complete candidates only; the 8 L40S GPU-h family cap (allocated from Slurm StartTime, earlier jobs included) stops the
family at the first limit, the partial candidate is excluded and marked.  Final: the CAL-selected hyperparameters, a new training seed,
CAL read-out at every checkpoint, its own CAL-selected checkpoint, then DEV (74 windows x 10) only at that checkpoint; DEV statistics by
EPFL scene unit and the frozen sensitivity sets.  Resume (not needed within the job limits): checkpoints are loaded on CPU and the training
log is truncated to the checkpoint step.  Every completed unit is verified (identity, artifacts, selection) before reuse.
"""
from __future__ import annotations

import argparse
import math
import os
import signal
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from v10.epfl import production as PR  # noqa: E402  (standard library only)
from v10.epfl import hardware as HW  # noqa: E402  (standard library only; the finals hardware release version)
from v10.supplement.records import require_identity  # noqa: E402

PHASES = ("CAL", "FINAL_DEV")


def plain(value):
    import numpy as np
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


class Context:
    """Everything a unit needs: guard, device, frozen group record, perturbation indices, the frozen candidate / schedule functions."""

    def __init__(self, guard, device, group, *, pert_indices=tuple(range(PR.N_PERT)), schedule_fn=None, candidates_fn=None, smoke=False,
                 interrupt_after=None):
        self.guard, self.device, self.group, self.pert_indices = guard, device, group, tuple(pert_indices)
        self.schedule_fn, self.candidates_fn, self.smoke, self.interrupt_after = schedule_fn, candidates_fn, smoke, interrupt_after
        self.hardware = None          # hardware release version: {gpu_name, label, node, partition, job, xhw_gate} of the running final
        self.replicate = None         # hardware replicate release: the diagnostic label carried in the unit identity


class Interrupted(RuntimeError):
    """Smoke-only: simulated job interruption after a given step (resume test)."""


# ------------------------------------------------------------------------------------------------------------ evaluation
def evaluate(ctx, trainer, role, windows, path, ident, checkpoint, allocation):
    """Complete, resumable registration block of one checkpoint on one role; truth used only here (error, F_true)."""
    import torch
    from v10 import adapter as A
    from v10.epfl import host as H
    from v10.supplement import checkpoints as CP
    path = Path(path)
    expected = {"unit": ident, "checkpoint_sha256": checkpoint["sha256"], "role": role, "windows": windows,
                "perturbations": list(ctx.pert_indices), "searcher": "unchanged R6-REG-P2" if not ctx.smoke else "SMOKE reduced grid"}
    id_path = path.with_suffix(".identity.json")
    if id_path.exists() and PR.read(id_path) != expected:
        raise RuntimeError("Evaluation namespace belongs to another identity")
    if not id_path.exists():
        PR.write(id_path, expected)
    records = PR.rows(path); seen = {}
    for row in records:
        key = row["window"], row["perturbation"]
        PR.require(key not in seen and row["window"] in windows and row["perturbation"] in ctx.pert_indices, "Duplicate or unexpected evaluation row")
        PR.require(row["role"] == role and row["checkpoint_sha256"] == checkpoint["sha256"], "Evaluation mixes roles or checkpoints")
        PR.require(all(math.isfinite(row[k]) for k in ("err_full", "F_hat", "F_true", "theta_hat", "seconds")), "Non-finite stored row")
        seen[key] = row
    summary_path = path.with_suffix(".summary.json")
    if summary_path.exists():
        s = PR.read(summary_path)
        PR.require(s.get("status") == "COMPLETE" and s["identity"] == expected and len(seen) == len(windows) * len(ctx.pert_indices)
                   and s["rows_sha256"] == PR.sha(path) and s["metrics"] == PR.metrics(records), "Invalid evaluation summary or coverage")
        return s
    unit_of = ctx.group["window_unit"]
    with CP.evaluation_state(trainer):
        session = H.EPFLRegistrationSession(ident["family"], trainer, ctx.guard, role, windows, ctx.device, ident["candidate"])
        for w in windows:
            for i in ctx.pert_indices:
                if (w, i) in seen:
                    continue
                allocation.check()
                energy, prep = session.energy(w, i)
                A.synchronize(trainer.device); t0 = time.perf_counter()
                rec = plain(A.registration_record(energy, trainer.device, session.perturbations[w][i]))
                A.synchronize(trainer.device); rec["seconds"] = time.perf_counter() - t0
                PR.require(all(math.isfinite(rec[k]) for k in ("err_full", "F_hat", "F_true", "theta_hat", "seconds")), "Non-finite registration")
                rec.update(prep)
                rec.update(window=w, unit=unit_of[w], perturbation=i, role=role, err_native_px=rec["err_full"] / 4.0,
                           full_registration_seconds=prep["preparation_seconds"] + rec["seconds"], checkpoint_sha256=checkpoint["sha256"],
                           job=allocation.job)
                PR.append(path, rec); seen[(w, i)] = rec
                del energy
            if trainer.device.type == "cuda":
                torch.cuda.empty_cache()
    PR.require(len(seen) == len(windows) * len(ctx.pert_indices), "Incomplete registration block")
    summary = {"status": "COMPLETE", "identity": expected, "metrics": PR.metrics(list(seen.values())), "rows_sha256": PR.sha(path),
               "pooling": "registration-level (as V10 BF)", "measured_complete_registration_seconds": sum(r["full_registration_seconds"] for r in seen.values())}
    PR.write(summary_path, summary)
    return summary


# ------------------------------------------------------------------------------------------------------------ one unit
def run_unit(ctx, directory, ident, allocation, environment):
    from v10.epfl import host as H
    from v10.supplement import checkpoints as CP
    sched = ident["schedule"]; directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    with PR.lock(directory / "RUN.lock"):
        previous = PR.verified_complete(directory, ident, sched)
        if previous is not None:
            return previous
        allocation.check()
        require_identity(directory, ident)
        PR.write(directory / f"ENV_{allocation.job}.json", environment)
        guard = ctx.guard
        cal = guard.included("CAL")
        trainer = H.new_trainer(ident["family"], ident["candidate"], ident["seed"], guard, ctx.device)
        config = CP.canonical_config(trainer)
        if (directory / "CONFIG.json").exists():
            PR.require(PR.read(directory / "CONFIG.json") == config, "Existing resolved configuration changed")
        PR.write(directory / "CONFIG.json", config)
        cal_checkpoints = []

        def cal_at(step):
            cp_path = directory / "checkpoints" / f"step_{step:04d}.pt"
            cp = CP.load(trainer, cp_path, ident)                 # CP.load maps the archive to CPU first (mixed-device optimizer state)
            PR.require(cp["step"] == step, "CAL checkpoint filename / step mismatch")
            r = evaluate(ctx, trainer, "CAL", cal, directory / "CAL" / f"step_{step:04d}.jsonl", ident, cp, allocation)
            cal_checkpoints.append({"step": step, "metrics": r["metrics"], "checkpoint": str(cp_path.relative_to(directory)), "checkpoint_sha256": cp["sha256"]})
        try:
            existing = sorted((directory / "checkpoints").glob("step_*.json"))
            if existing:                                          # resume: load on CPU, truncate the log to the checkpoint step
                last = max(existing, key=lambda p: PR.read(p)["step"])
                latest = CP.load(trainer, last.with_suffix(".pt"), ident)["step"]
                PR.require(latest in sched["save_steps"], "Unexpected resume checkpoint")
                removed = PR.truncate_steps(directory / "TRAIN_STEPS.jsonl", latest)
                allocation.record("RESUME", unit=directory.name, step=latest, log_rows_truncated=removed)
                for step in sched["save_steps"]:
                    if step <= latest:
                        PR.require(CP.metadata(directory / "checkpoints" / f"step_{step:04d}.pt", ident)["step"] == step, "Checkpoint sequence mismatch")
                for step in sched["cal_steps"]:
                    if step <= latest:
                        cal_at(step)
                CP.load(trainer, last.with_suffix(".pt"), ident)
            elif sched["steps"] == 0:
                CP.save(trainer, directory / "checkpoints/step_0000.pt", ident)
                cal_at(0)
            while trainer.n_steps < sched["steps"]:
                allocation.check()
                rec = plain(trainer.step())
                for key in ("loss", "J", "grad_norm_preclip"):
                    if key in rec and not math.isfinite(rec[key]):
                        raise RuntimeError("Non-finite training value: " + key)
                rec.update(job=allocation.job, utc=PR.utc())
                PR.append(directory / "TRAIN_STEPS.jsonl", rec)
                step = trainer.n_steps
                if step in sched["save_steps"]:
                    CP.save(trainer, directory / "checkpoints" / f"step_{step:04d}.pt", ident)
                    allocation.record("CHECKPOINT", unit=directory.name, step=step)
                if step in sched["cal_steps"]:
                    cal_at(step)
                if ctx.interrupt_after is not None and step == ctx.interrupt_after:
                    raise Interrupted(f"smoke interruption after step {step}")
            PR.require([c["step"] for c in cal_checkpoints] == sched["cal_steps"], "Prescribed CAL checkpoint set missing")
            chosen = min(cal_checkpoints, key=PR.selection_key)
            extra = {}
            if ident["phase"] == "FINAL_DEV":
                guard_check_data(ctx, {"DEV"})
                dev = guard.included("DEV")
                cp = CP.load(trainer, directory / chosen["checkpoint"], ident)
                r = evaluate(ctx, trainer, "DEV", dev, directory / "DEV.jsonl", ident, cp, allocation)
                records = PR.rows(directory / "DEV.jsonl")
                PR.require({x["checkpoint_sha256"] for x in records} == {chosen["checkpoint_sha256"]}, "DEV must be read only at the selected checkpoint")
                extra = {"DEV": r["metrics"], "DEV_statistics": PR.dev_statistics(records, ctx.group), "selected_checkpoint_sha256": chosen["checkpoint_sha256"]}
            PR.write(directory / "DATA_READS.json", guard.reads)
            result = {"status": "COMPLETE", "identity": ident, "steps": sched["steps"], "work_kind": sched["kind"], "cal_checkpoints": cal_checkpoints,
                      "selected": chosen, "utc": PR.utc(), "job": allocation.job, **extra}
            if ctx.hardware is not None:   # hardware release: actual GPU, node, partition and this job's allocated time
                result.update(hardware=ctx.hardware, allocated_seconds_this_job=allocation.elapsed())
            result["artifacts"] = {str(p.relative_to(directory)): PR.sha(p) for p in directory.rglob("*")
                                   if p.is_file() and p.name not in ("RUN.lock", "COMPLETE.json") and not p.name.startswith(".")}
            PR.write(directory / "COMPLETE.json", result)
            return PR.verified_complete(directory, ident, sched)
        except BaseException:
            PR.write(directory / f"READS_INCOMPLETE_{allocation.job}.json", guard.reads)
            raise


def guard_check_data(ctx, roles):
    ctx.guard.verify_data_lock(roles)


# ------------------------------------------------------------------------------------------------------------ CAL family and selection
def checked_selection(root, family, release_sha, data_sha, group_sha, ctx, require_exit=False):
    path = Path(root) / "CAL" / family / "SELECTION.json"
    sel = PR.read(path)
    PR.require(sel.get("status") == "SELECTED" and sel.get("family") == family and sel.get("release_sha256") == release_sha, "Selection identity mismatch")
    if require_exit:
        receipt = PR.read(Path(root).parent / "exits" / f"{sel['job']}.json")
        PR.require(receipt.get("job") == sel["job"] and receipt.get("exit_code") == 0 and receipt.get("release_sha256") == release_sha,
                   "CAL selection lacks a successful exit for this release")
    planned = ctx.candidates_fn(family); complete = sel["completed_candidates"]
    PR.require(bool(complete) and [r["candidate"] for r in complete] == planned[:len(complete)], "CAL candidate sequence incomplete or reordered")
    if len(complete) != len(planned):
        PR.require(sel.get("budget_exhausted") is True, "Missing CAL candidates without budget evidence")
        b = PR.read(path.parent / "BUDGET_EXHAUSTED.json")
        PR.require(math.isfinite(b["allocated_seconds"]) and b["allocated_seconds"] >= PR.CAP_SECONDS - PR.CAP_MARGIN - 60 or ctx.smoke,
                   "Budget-limited selection lacks actual cap exhaustion")
    sched = ctx.schedule_fn(family)
    for r in complete:
        unit = Path(root) / "CAL" / family / f"candidate_{r['candidate']['order']:02d}"
        ident = PR.identity(release_sha, family, r["candidate"], 1017, "CAL", data_sha, group_sha, sched)
        res = PR.verified_complete(unit, ident, sched)
        PR.require(res is not None and PR.sha(unit / "COMPLETE.json") == r["complete_sha256"]
                   and {**res["selected"], "order": r["candidate"]["order"]} == r["choice"], "Selection dependency changed")
    PR.require(min(complete, key=lambda r: PR.selection_key(r["choice"])) == sel["selected"], "CAL selection differs from the fixed lexicographic rule")
    return sel, PR.sha(path)


def run_cal(ctx, root, family, release_sha, data_sha, group_sha, allocation, environment):
    directory = Path(root) / "CAL" / family
    if (directory / "SELECTION.json").exists():
        return checked_selection(root, family, release_sha, data_sha, group_sha, ctx)
    sched = ctx.schedule_fn(family); completed, exhausted = [], False
    for cand in ctx.candidates_fn(family)[:PR.MAX_CANDIDATES]:
        unit = directory / f"candidate_{cand['order']:02d}"
        ident = PR.identity(release_sha, family, cand, 1017, "CAL", data_sha, group_sha, sched)
        try:
            result = run_unit(ctx, unit, ident, allocation, environment)
        except PR.BudgetExceeded:
            exhausted = True
            signal.setitimer(signal.ITIMER_REAL, 0)
            PR.write(directory / "BUDGET_EXHAUSTED.json", {"utc": PR.utc(), "candidate": cand, "allocated_seconds": allocation.previous + allocation.elapsed(),
                                                          "partial_candidate_excluded": True, "rule": "decision 4: 8 L40S GPU-h, first limit reached, no extension"})
            break
        completed.append({"candidate": cand, "complete_sha256": PR.sha(unit / "COMPLETE.json"), "choice": {**result["selected"], "order": cand["order"]}})
    PR.require(bool(completed), "BLOCKED: no complete candidate within the CAL budget")
    sel = {"status": "SELECTED", "family": family, "release_sha256": release_sha, "completed_candidates": completed, "budget_exhausted": exhausted,
           "planned_candidates": len(ctx.candidates_fn(family)), "selected": min(completed, key=lambda r: PR.selection_key(r["choice"])),
           "utc": PR.utc(), "job": allocation.job, "selection_scope": "complete candidates with every prescribed CAL readout"}
    PR.write(directory / "SELECTION.json", sel)
    return checked_selection(root, family, release_sha, data_sha, group_sha, ctx)


def run_final(ctx, root, family, seed, release_sha, data_sha, group_sha, allocation, environment, require_exit=True, cal_release_sha=None,
              selection_pin=None, cal_root=None):
    """cal_release_sha: the release whose CAL selection is consumed (hardware release: the parent 3dae6434); selection_pin: its sha256 when
    the hardware release pinned it.  With ctx.hardware the unit identity carries the GPU name: a COMPLETE unit is reused on any hardware,
    an interrupted one resumes only on its own GPU type."""
    sel, dep = checked_selection(cal_root or root, family, cal_release_sha or release_sha, data_sha, group_sha, ctx, require_exit=require_exit)
    if selection_pin is not None:
        PR.require(dep == selection_pin, "CAL selection differs from the hash pinned by the release")
    cand = sel["selected"]["candidate"]
    ident = PR.identity(release_sha, family, cand, seed, "FINAL_DEV", data_sha, group_sha, ctx.schedule_fn(family), dep)
    unit = Path(root) / "FINAL_DEV" / family / f"seed_{seed}"
    if getattr(ctx, "replicate", None) is not None:   # post-hoc hardware sensitivity diagnostic: its own unit, never a primary one
        primary = Path(cal_root or root).resolve()
        PR.require(cal_root is not None and primary != Path(root).resolve() and primary not in unit.resolve().parents,
                   "A hardware replicate never writes into, replaces or reuses the primary results")
        ident = {**ident, "diagnostic": ctx.replicate}
    if ctx.hardware is not None:
        done = unit / "COMPLETE.json"
        stored = PR.read(done).get("identity") if done.exists() else None
        if stored is not None and "hardware" in stored and {k: v for k, v in stored.items() if k != "hardware"} == ident:
            ident = stored
        else:
            ident = {**ident, "hardware": {"gpu_name": ctx.hardware["gpu_name"]}}
    return run_unit(ctx, unit, ident, allocation, environment)


# ------------------------------------------------------------------------------------------------------------ production entry
def production(args) -> int:
    schema = PR.read(args.release).get("schema")
    replicate = schema in HW.DIAGNOSTICS                           # post-hoc hardware diagnostic: L40S reruns (of RTX finals; L40S control)
    hw = schema == HW.RELEASE_SCHEMA_HW or replicate
    if replicate:
        release = HW.require_release_replicate(args.release, args.family, args.seed, phase=args.phase)
    elif hw:                                                       # finals-only hardware release version (parent 3dae6434)
        release = HW.require_release_hw(args.release, args.family, phase=args.phase)
    else:
        release = PR.require_release(args.release, args.family)    # pure-record validation before importing torch or touching data
    from v10.epfl import host as H
    entry = release["families"][args.family]
    PR.require(entry["candidates"] == H.production_candidates(args.family) and entry["schedule"] == H.production_schedule(args.family),
               "Release candidates / schedule differ from the frozen definitions in this code")
    base = Path(args.release).resolve().parent
    root = Path(release["results_root"]) if hw else base / "results"
    cal_root = Path(release["cal_results_root"]) if replicate else root
    release_sha = PR.sha(args.release); data_sha = release["data_lock"]["sha256"]; group_sha = release["group_record"]["sha256"]
    cal_release_sha = release["cal_release"]["sha256"] if hw else release_sha
    selection_pin = (release.get("selection_pins") or {}).get(args.family) if hw else None
    sched = H.production_schedule(args.family)
    if args.phase == "final":
        PR.require(args.seed in sched["final_seeds"], "final needs one of the family's final seeds")
    from v10 import adapter as A
    from v10.epfl import guard as GD, runtime as RT
    act = PR._abs(release["activation_record"]["path"])
    guard = GD.EPFLDataGuard(record=act, expect_record_sha256=release["activation_record"]["sha256"],
                             data_lock=PR.read(PR._abs(release["data_lock"]["path"])))
    PR.require(guard.stage == "ACTIVATED" and guard.group_sha256 == group_sha, "Gate is not at stage ACTIVATED with the frozen group record")
    group = guard.group_record
    ctx = Context(guard, "cuda", group, schedule_fn=H.production_schedule, candidates_fn=H.production_candidates)
    if replicate:
        ctx.replicate = HW.replicate_label(release, args.family, args.seed)

    def terminated(signum, frame):
        raise InterruptedError(f"Slurm signal {signum}; the last verified checkpoint is kept")
    signal.signal(signal.SIGTERM, terminated); signal.signal(signal.SIGUSR1, terminated)
    lock_path = root / "CAL" / args.family / "FAMILY.lock" if args.phase == "cal" else root / "FINAL_DEV" / args.family / f"seed_{args.seed}.lock"
    with PR.lock(lock_path):
        allocation = PR.Allocation(base, args.family, "CAL" if args.phase == "cal" else "FINAL_DEV")
        status = "FAIL"
        try:
            env = RT.runtime_environment(A)
            if hw:     # owner 2026-10-19: final on RTX6000PRO / L40S / H100; CAL stays on L40S (refused here: the parent runs CAL)
                PR.require(HW.gpu_accepted(args.phase, env.get("gpu_name")) and (not replicate or env.get("gpu_name") == HW.CAL_GPU),
                           f"{env.get('gpu_name')} is not accepted for EPFL {args.phase}" + (" (diagnostic reruns run on L40S only)" if replicate else ""))
                gate = HW.xhw_gate(release, env["gpu_name"], release["source_commit"])
                ctx.hardware = {**HW.hardware_record(env), "xhw_gate": gate}
                allocation.record("HARDWARE", **ctx.hardware)
            else:
                PR.require(env.get("gpu_name") == PR.BUDGET_GPU, "EPFL CAL and finals run on L40S only (decision 5)")
            guard.verify_data_lock({"FIT", "CAL"})
            if args.phase == "cal":
                def alarm(signum, frame):
                    raise PR.BudgetExceeded("CAL allocated-time alarm reached")
                signal.signal(signal.SIGALRM, alarm)
                if allocation.remaining() > 0:
                    signal.setitimer(signal.ITIMER_REAL, allocation.remaining())
                run_cal(ctx, root, args.family, release_sha, data_sha, group_sha, allocation, env)
            else:
                run_final(ctx, root, args.family, args.seed, release_sha, data_sha, group_sha, allocation, env, cal_release_sha=cal_release_sha,
                          selection_pin=selection_pin, cal_root=cal_root)
            status = "COMPLETE"
        except BaseException as e:
            PR.write(base / "failures" / f"{allocation.job}.json", {"utc": PR.utc(), "family": args.family, "phase": args.phase, "seed": args.seed,
                                                                  "exception": repr(e), "traceback": traceback.format_exc()})
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            allocation.record("END", status=status, **({k: ctx.hardware[k] for k in ("gpu_name", "node", "partition")} if ctx.hardware else {}))
    return 0


# ------------------------------------------------------------------------------------------------------------ CPU smoke
class SmokeGuard:
    """Smoke roles on the real gate: FIT = the first 4 FIT windows, CAL = the first 2 CAL windows, 'DEV' = the next 2 CAL windows relabelled
    (no DEV window is read: every 'DEV' access is checked as a CAL read of that window and logged as such)."""
    smoke = True

    def __init__(self, base):
        self.base, self.r6_root, self.reads, self.removed, self.p3_roles = base, base.r6_root, base.reads, base.removed, base.p3_roles
        cal = base.included("CAL")
        self.roles = {"FIT": base.included("FIT")[:4], "CAL": cal[:2], "DEV": cal[2:4]}
        self.stage, self.data_lock = "SMOKE_" + base.stage, base.data_lock

    def included(self, role):
        return list(self.roles[role])

    def expected_fit_windows(self):
        return list(self.roles["FIT"])

    def unit_of(self, w):
        return self.base.unit_of(w)

    def check(self, w, allowed, purpose):
        allowed = set(allowed)
        role = next((r for r, ws in self.roles.items() if w in ws), None)
        if role is None or role not in allowed:
            raise PermissionError(f"{w}: not a smoke window of roles {sorted(allowed)}")
        real = "CAL" if role == "DEV" else role
        return self.base.check(w, {real}, purpose + (" [smoke: CAL window relabelled DEV]" if role == "DEV" else ""))

    def log_rule(self, w, allowed, purpose):
        self.check(w, allowed, purpose)
        self.reads[-1].update({"kind": "rule", "path": None})

    def verify_data_lock(self, roles):
        n = 0
        for role in roles:
            for w in self.roles[role]:
                path = self.check(w, {role}, "smoke data-lock integrity")
                PR.require(PR.sha(path) == self.data_lock[w]["sha256"], f"{w}: cache file changed since the data lock")
                n += 1
        return {"checked": n}


def reset_official_comir_imports():
    """Smoke only.  Production builds one CoMIR trainer per process (CoMIR has one CAL candidate per CAL job and each final seed is its own
    job); comir.verify_official_source() refuses any already-imported 'models' module without a file under the pinned root, and the
    official package's subpackage models.layers is a namespace package (__file__ None), so a second construction in one process is refused.
    The smoke runs CAL and the final in one process, so it drops the official package's modules between the two stages to emulate the
    fresh process of production; the pinned-file and namespace checks still run in full on every construction."""
    for name in [n for n in sys.modules if n == "models" or n.startswith("models.")]:
        del sys.modules[name]


def smoke_group(guard):
    windows = guard.roles["DEV"]; unit = {w: guard.unit_of(w) for r in guard.roles.values() for w in r}
    units = sorted({unit[w] for w in windows}); s = {"windows": windows, "units": units}
    return {"window_unit": unit, "sensitivity_sets": {name: dict(s) for name in PR.DEV_SETS}}


def smoke_schedule_fn(H):
    def f(family):
        s = dict(H.production_schedule(family))
        if s["kind"] == "gradient_training":
            s.update(steps=4, save_steps=[2, 4], cal_steps=[2, 4])
        elif s["kind"] == "native_gradient_training":
            s.update(steps=4, save_steps=[2, 4], cal_steps=[4])
        return s
    return f


def smoke(args) -> int:
    """CPU code path of the whole production flow on smoke roles, a reduced search grid and 2 perturbations; never a scientific record."""
    import json
    import subprocess
    out = Path(args.out).resolve()
    PR.require(args.device == "cpu", "the smoke runs on CPU only")
    PR.require(not str(out).startswith(str(PR.BUNDLE_DEFAULT)), "the smoke never writes into the production bundle")
    from v10 import adapter as A
    from v10.epfl import guard as GD, host as H
    A.Q.TH_STEP, A.P.GRID_STEP, A.P.N_ADAM = 15.0, 17.0, 2          # smoke-only reduced search (5 angles x 29 shifts, 2 Adam steps)
    rec = GD.ACTIVATION_RECORD; gr = GD.GROUP_RECORD_V2
    base = GD.EPFLDataGuard(record=rec, expect_record_sha256=PR.sha(rec), group_record=gr, expect_group_sha256=PR.sha(gr),
                            data_lock=PR.read(PR.EPFL_DIR / "EPFL_DATA_LOCK.json"))
    PR.require(len(base.included("FIT")) == 334 and "water_0006" in base.removed and "street_0020" in base.removed, "group record not applied")
    guard = SmokeGuard(base)
    families = [f for f in (args.families.split(",") if args.families else H.PRODUCTION_FAMILIES)]
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    release_sha = "SMOKE-" + commit; data_sha = PR.sha(PR.EPFL_DIR / "EPFL_DATA_LOCK.json"); group_sha = PR.sha(gr)
    report = {"schema": "v10.epfl.production_smoke.v1", "status": "RUNNING", "commit": commit, "utc": PR.utc(), "families": {},
              "note": "code paths only: smoke roles (4 FIT, 2 CAL, 2 CAL windows relabelled DEV), reduced search grid, 2 perturbations; not a result"}
    sched_fn = smoke_schedule_fn(H)
    for fam in families:
        t0 = time.time(); entry = {}
        try:
            cands = H.production_candidates(fam)
        except H.RuLSIFV2Unavailable as e:
            report["families"][fam] = {"status": "BLOCKED_AS_DECIDED", "reason": str(e)}; continue
        ctx = Context(guard, "cpu", smoke_group(guard), pert_indices=(0, 1), schedule_fn=sched_fn, candidates_fn=lambda f, c=cands: c, smoke=True)
        root = out / "results"
        alloc = PR.Allocation(out, fam, "CAL", job=f"smoke-{os.getpid()}", started=time.time(), release_sha=release_sha, cap_seconds=10 ** 9)
        try:
            sel, sel_sha = run_cal(ctx, root, fam, release_sha, data_sha, group_sha, alloc, {"smoke": True})
            entry.update(cal_candidates=len(sel["completed_candidates"]), selected=sel["selected"]["candidate"])
            alloc_f = PR.Allocation(out, fam, "FINAL_DEV", job=f"smoke-{os.getpid()}", started=time.time(), release_sha=release_sha)
            if fam == "CoMIR_native":
                reset_official_comir_imports(); entry["comir_imports_reset_between_stages"] = True
            res = run_final(ctx, root, fam, 1001, release_sha, data_sha, group_sha, alloc_f, {"smoke": True}, require_exit=False)
            again = run_final(ctx, root, fam, 1001, release_sha, data_sha, group_sha, alloc_f, {"smoke": True}, require_exit=False)
            entry.update(final_complete=res["status"] == "COMPLETE", reuse_identical=again == res,
                         dev_at_selected_only={r["checkpoint_sha256"] for r in PR.rows(root / "FINAL_DEV" / fam / "seed_1001" / "DEV.jsonl")} == {res["selected"]["checkpoint_sha256"]},
                         dev_sets=sorted(res["DEV_statistics"]), status="PASS")
        except Exception as e:  # noqa: BLE001
            entry.update(status="FAIL", error=repr(e)[:800], traceback=traceback.format_exc()[-3000:])
        entry["seconds"] = time.time() - t0; report["families"][fam] = entry
        print(f"[smoke] {fam} {entry['status']} ({entry['seconds']:.0f}s)", flush=True)
    # resume: interrupt a final unit after step 3 (checkpoint at 2), then resume -> the log is truncated to step 2 and the unit completes
    fam = "MF_VCS_perm1"
    if fam in families and report["families"].get(fam, {}).get("status") == "PASS":
        try:
            cands = H.production_candidates(fam)
            ctx = Context(guard, "cpu", smoke_group(guard), pert_indices=(0, 1), schedule_fn=sched_fn, candidates_fn=lambda f, c=cands: c,
                          smoke=True, interrupt_after=3)
            alloc = PR.Allocation(out, fam, "FINAL_DEV", job=f"smoke-resume-{os.getpid()}", started=time.time(), release_sha=release_sha)
            try:
                run_final(ctx, out / "results", fam, 1002, release_sha, data_sha, group_sha, alloc, {"smoke": True}, require_exit=False)
                interrupted = False
            except Interrupted:
                interrupted = True
            log = out / "results" / "FINAL_DEV" / fam / "seed_1002" / "TRAIN_STEPS.jsonl"
            before = [r["step"] for r in PR.rows(log)]
            ctx.interrupt_after = None
            res = run_final(ctx, out / "results", fam, 1002, release_sha, data_sha, group_sha, alloc, {"smoke": True}, require_exit=False)
            after = [r["step"] for r in PR.rows(log)]
            ok = interrupted and before == [1, 2, 3] and after == [1, 2, 3, 4] and res["status"] == "COMPLETE"
            report["resume"] = {"status": "PASS" if ok else "FAIL", "interrupted": interrupted, "steps_before": before, "steps_after": after,
                                "note": "checkpoint at step 2, interruption after step 3: the step-3 row is truncated and recomputed on resume"}
        except Exception as e:  # noqa: BLE001
            report["resume"] = {"status": "FAIL", "error": repr(e)[:800], "traceback": traceback.format_exc()[-3000:]}
    roles = sorted({r["role"] for r in guard.reads}); report["roles_read"] = roles
    report["dev_window_read"] = any(r["role"] == "DEV" for r in guard.reads)
    ok = all(v["status"] in ("PASS", "BLOCKED_AS_DECIDED") for v in report["families"].values()) and report.get("resume", {}).get("status", "PASS") == "PASS" \
        and not report["dev_window_read"]
    report["status"] = "SMOKE_PASS_NOT_A_RESULT" if ok else "SMOKE_FAIL"
    (out / "SMOKE.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(f"[smoke] {report['status']} -> {out / 'SMOKE.json'}", flush=True)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=("cal", "final", "smoke")); ap.add_argument("family", nargs="?")
    ap.add_argument("--release", type=Path); ap.add_argument("--seed", type=int)
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu")); ap.add_argument("--out", type=Path); ap.add_argument("--families", default="")
    a = ap.parse_args()
    if a.phase == "smoke":
        if a.out is None:
            ap.error("smoke needs --out")
        return smoke(a)
    if a.family is None or a.release is None:
        ap.error("cal / final need a family and --release")
    if a.phase == "cal" and a.seed is not None:
        ap.error("the CAL seed is fixed at 1017")
    if a.phase == "final" and a.seed is None:
        ap.error("final needs --seed")
    return production(a)


if __name__ == "__main__":
    sys.exit(main())
