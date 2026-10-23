#!/usr/bin/env python3
"""R22-C2 (PILOT/DEVELOPMENT): the PMI readout through CAL checkpoint selection and DEV evaluation, one job per unit, no training.

  run_pmi2.py run --unit U.json

  1 CAL   every saved checkpoint (steps 250..3000) registers the CAL problems with the PMI energy (R22-C: mean 2 atanh(clip(T,
          +-(1-1e-4))), pmi_energy.py unchanged); before each search the J energy at the truth (F_true_J) is evaluated on the same
          fresh energy instance and compared with the stored historical J CAL row of that checkpoint (inputs / perturbation / model
          identity: bitwise on the GPU type of the stored rows, T01 tolerance otherwise).
  2 SELECT the unit's original CAL rule (v10.supplement.records.selection_key on records.metrics: -success42, -success9,
          median_err, step) on the PMI CAL rows -> PMI-selected checkpoint; SELECTION.json is written before any DEV read.
  3 DEV   per DEV problem, on the originally (J-)selected checkpoint: J, then PMI on the same energy instance (as R22-C; J row compared
          with the stored historical DEV row); if the PMI-selected checkpoint differs: PMI on it.  E keys J_orig, PMI_orig[, PMI_sel].
          DEV is descriptive only: nothing is chosen after it.

The registration path is R22-C's (run_unit.py: pinned host numerics, BF1 / supplement / EPFL sessions, guards and data locks; both
R22-C files are pinned by sha256 below).  Rows are appended per registration; a rerun of the same line verifies and keeps them.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import signal
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = HERE.parents[1]
C_DIR = WT / "v10/r22_pmi"
C_SHA256 = {"run_unit.py": "6ab06f450adf928a752ae5d9188239258ade9b208d0aba48bf6b3a5b538379cc",
            "pmi_energy.py": "5df9dfa8ebd4b52d6fff82bfffdcc9058fad23e23cf0951ca8ea1254bb6edc6f"}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


for _name, _h in C_SHA256.items():
    if sha(C_DIR / _name) != _h:
        raise RuntimeError("R22-C file differs from the pinned bytes: " + _name)
sys.path.insert(0, str(C_DIR))
import run_unit as R  # noqa: E402  (verifies the 23 pinned host files, puts src on sys.path)

A, PE, CP, torch, np = R.A, R.PE, R.CP, R.torch, R.np
from v10.supplement import records as REC  # noqa: E402  (pinned: metrics, selection_key)

require, rows, write, append, plain, utc = R.require, R.rows, R.write, R.append, R.plain, R.utc
ROLES = ("CAL", "DEV")
DEV_ARMS = ("J_orig", "PMI_orig", "PMI_sel")


# ------------------------------------------------------------------------------------------------------------ unit, model
def load_unit(path):
    u = json.loads(Path(path).read_text())
    require(u.get("schema") == "v10.r22_pmi2.unit.v1" and u["family"] in R.METHOD and u["seed"] in (1001, 1002, 1003),
            "Not an R22-C2 unit")
    d = Path(u["unit_dir"])
    require(sha(d / "COMPLETE.json") == u["complete_sha256"], "Historical COMPLETE.json changed")
    c = json.loads((d / "COMPLETE.json").read_text())
    require(c["status"] == "COMPLETE" and c["identity"]["family"] == u["family"] and c["identity"]["seed"] == u["seed"]
            and c["identity"]["phase"] == "FINAL_DEV" and c["selected"]["step"] == u["original"]["step"]
            and c["selected"]["checkpoint_sha256"] == u["original"]["checkpoint_sha256"], "Historical unit identity differs")
    require([k["step"] for k in u["checkpoints"]] == list(range(250, 3001, 250)), "Unit must list the 12 saved checkpoints")
    for k in u["checkpoints"]:
        require(sha(k["checkpoint"]) == k["checkpoint_sha256"], "Checkpoint bytes differ: " + k["checkpoint"])
        require(sha(k["stored_cal"]["path"]) == k["stored_cal"]["sha256"], "Stored CAL rows changed: " + k["stored_cal"]["path"])
    require(sha(u["stored_dev"]["path"]) == u["stored_dev"]["sha256"], "Stored DEV rows changed")
    require(sha(u["data_lock"]["path"]) == u["data_lock"]["sha256"], "Data lock changed")
    return u


def checkpoint(u, step):
    return next(k for k in u["checkpoints"] if k["step"] == step)


def trainer_for(u, step, guard, H, device):
    """R22-C's build_trainer on one saved checkpoint (config / method / seed / model-hash / step checks)."""
    k = checkpoint(u, step)
    return R.build_trainer({**u, "checkpoint": k["checkpoint"], "selected_step": step}, guard, H, device)


def key_of(u):
    return "core" if u["ds"] == "bf" else "window"


def index(path, k):
    return {(r[k], r["perturbation"]): r for r in rows(path)}


# ------------------------------------------------------------------------------------------------------------ registrations
def pmi_with_j_truth(energy, device, truth):
    """F_true_J (host J energy at the truth, fresh instance, before any search), then the PMI registration on the same instance."""
    PE.set_variant(PE.inner(energy), "J")
    f_true_j = energy.value(truth[0], (truth[1], truth[2]))
    rec = PE.register_variants(energy, device, truth, ("PMI",))["PMI"]
    return f_true_j, rec


def close(new, old):
    atol, rtol = R.T01_FP32
    return abs(new - old) <= atol + rtol * abs(old)


def fields(rec):
    return {f: rec[f] for f in R.FIELDS}


def finite(rec):
    return all(math.isfinite(rec[f]) for f in ("err_full", "F_hat", "F_true", "theta_hat", "seconds"))


def register_cal(u, step, guard, H, problems, path, device, job, gpu, deadline=None):
    """PMI CAL registrations of one checkpoint; rows already in path are verified and kept."""
    k = key_of(u)
    done = {}
    for r in rows(path):
        key = (r[k], r["perturbation"])
        require(key not in done and key in problems and r["role"] == "CAL" and r["step"] == step and set(r["E"]) == {"PMI"}
                and finite(r["E"]["PMI"]), "Unexpected or invalid stored C2 CAL row")
        done[key] = r
    todo = [p for p in problems if p not in done]
    if todo:
        old = index(checkpoint(u, step)["stored_cal"]["path"], k)
        trainer, _ = trainer_for(u, step, guard, H, device)
        with CP.evaluation_state(trainer):
            session = R.make_session(u, trainer, guard, H, "CAL", sorted({c for c, _ in problems}), device)
            last = None
            for core, i in todo:
                if core != last and trainer.device.type == "cuda":
                    torch.cuda.empty_cache()
                last = core
                if deadline is not None and time.monotonic() > deadline:
                    raise TimeoutError("Allocation nearly over; finished rows are kept")
                energy, prep = session.energy(core, i)
                truth = session.perturbations[core][i]
                f_true_j, rec = pmi_with_j_truth(energy, device, truth)
                rec = plain(rec); f_true_j = float(f_true_j)
                require(finite(rec) and math.isfinite(f_true_j), "Non-finite CAL registration")
                o = old[(core, i)]
                row = {k: core, "perturbation": i, "role": "CAL", "step": step, "truth": truth, "E": {"PMI": fields(rec)},
                       "J_F_true": f_true_j,
                       "J_truth_vs_stored": {"F_true_bitwise": f_true_j == o["F_true"], "F_true_within_t01": close(f_true_j, o["F_true"]),
                                             "d_F_true": f_true_j - o["F_true"], "n_grid_equal": rec["n_grid"] == o["n_grid"],
                                             "stored_err_full": o["err_full"]},
                       "prep": prep, "job": job, "gpu": gpu, "utc": utc()}
                if k == "window":
                    row["unit"] = guard.unit_of(core)
                append(path, row); done[(core, i)] = row
                del energy
        del trainer
    return [done[p] for p in problems]


def select(errs_by_step):
    """The original CAL rule on {step: [err_full, ...]}: (chosen {step, metrics}, all checkpoints)."""
    cks = [{"step": s, "metrics": REC.metrics([{"err_full": e} for e in errs])} for s, errs in sorted(errs_by_step.items())]
    return min(cks, key=REC.selection_key), cks


def register_dev(u, sel_step, guard, H, problems, path, device, job, gpu, deadline=None):
    """J and PMI on the original checkpoint (same instance), PMI on the PMI-selected one when it differs."""
    k = key_of(u)
    orig = u["original"]["step"]
    same = sel_step == orig
    arms = {"J_orig", "PMI_orig"} | (set() if same else {"PMI_sel"})
    done = {}
    for r in rows(path):
        key = (r[k], r["perturbation"])
        require(key not in done and key in problems and r["role"] == "DEV" and set(r["E"]) == arms
                and r["steps"] == {"orig": orig, "sel": sel_step} and all(finite(r["E"][a]) for a in arms),
                "Unexpected or invalid stored C2 DEV row")
        done[key] = r
    todo = [p for p in problems if p not in done]
    if todo:
        old = index(u["stored_dev"]["path"], k)
        cores = sorted({c for c, _ in problems})
        t_orig, _ = trainer_for(u, orig, guard, H, device)
        t_sel = None if same else trainer_for(u, sel_step, guard, H, device)[0]
        with contextlib.ExitStack() as stack:
            stack.enter_context(CP.evaluation_state(t_orig))
            s_orig = R.make_session(u, t_orig, guard, H, "DEV", cores, device)
            s_sel = None
            if not same:
                stack.enter_context(CP.evaluation_state(t_sel))
                s_sel = R.make_session(u, t_sel, guard, H, "DEV", cores, device)
            last = None
            for core, i in todo:
                if core != last and t_orig.device.type == "cuda":
                    torch.cuda.empty_cache()
                last = core
                if deadline is not None and time.monotonic() > deadline:
                    raise TimeoutError("Allocation nearly over; finished rows are kept")
                energy, prep = s_orig.energy(core, i)
                truth = s_orig.perturbations[core][i]
                recs = plain(PE.register_variants(energy, device, truth, ("J", "PMI")))
                E = {"J_orig": fields(recs["J"]), "PMI_orig": fields(recs["PMI"])}
                del energy
                if not same:
                    require(s_sel.perturbations[core][i] == truth, "Perturbation differs between sessions")
                    energy, _ = s_sel.energy(core, i)
                    E["PMI_sel"] = fields(plain(PE.register_variants(energy, device, truth, ("PMI",)))["PMI"])
                    del energy
                require(all(finite(E[a]) for a in E), "Non-finite DEV registration")
                row = {k: core, "perturbation": i, "role": "DEV", "truth": truth, "steps": {"orig": orig, "sel": sel_step}, "E": E,
                       "J_vs_stored": R.compare_stored(recs["J"], old[(core, i)]), "prep": prep, "job": job, "gpu": gpu, "utc": utc()}
                if k == "window":
                    row["unit"] = guard.unit_of(core)
                append(path, row); done[(core, i)] = row
        del t_orig, t_sel
    return [done[p] for p in problems]


def dev_metrics(recs):
    out = {}
    for a in DEV_ARMS:
        if a in recs[0]["E"]:
            out[a] = REC.metrics([{"err_full": r["E"][a]["err_full"]} for r in recs])
    s = [r["J_vs_stored"] for r in recs]
    out["J_vs_stored"] = {"n": len(s), **{f: sum(x[f] for x in s) for f in ("exact", "F_true_bitwise", "F_true_within_t01", "n_grid_equal")}}
    return out


# ------------------------------------------------------------------------------------------------------------ one unit
def identity(u):
    return {"unit": u, "pmi_eps": PE.EPS, "host_sha256": R.HOST_SHA256, "c_sha256": C_SHA256,
            "run_pmi2_sha256": sha(Path(__file__)), "schema": "v10.r22_pmi2.identity.v1"}


def selection_record(u, sel, cks, cal_dir):
    k = checkpoint(u, sel["step"])
    return {"schema": "v10.r22_pmi2.selection.v1", "rule": u["selection_rule"], "energy": "PMI",
            "pmi_selected": {"step": sel["step"], "metrics": sel["metrics"], "checkpoint": k["checkpoint"],
                             "checkpoint_sha256": k["checkpoint_sha256"]},
            "original": u["original"], "same_checkpoint": sel["step"] == u["original"]["step"], "pmi_cal_checkpoints": cks,
            "cal_rows_sha256": {str(s["step"]): sha(cal_dir / f"step_{s['step']:04d}.jsonl") for s in cks}}


def run_unit(u, out, deadline=None, device="cuda", cal_problems=None, dev_problems=None, steps=None):
    """Full unit (GPU).  The keyword overrides exist only for the CPU check (subset of problems / checkpoints)."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if device == "cuda":
        require(torch.cuda.is_available(), "R22-C2 registrations run on a GPU (Slurm)")
    job = os.environ.get("SLURM_JOB_ID", "local")
    env = R.environment(); env["run_pmi2_sha256"] = sha(Path(__file__)); gpu = env.get("gpu_name")
    write(out / f"ENV_{job}.json", env)
    ident = identity(u)
    ip = out / "IDENTITY.json"
    if ip.exists():
        require(json.loads(ip.read_text()) == json.loads(json.dumps(ident)), "Output directory belongs to another identity")
    else:
        write(ip, ident)
    t0 = time.time()
    guard, H = R.make_guard(u)
    cal_problems = cal_problems or R.problems_of(guard, "CAL", False)
    dev_problems = dev_problems or R.problems_of(guard, "DEV", False)
    steps = steps or [k["step"] for k in u["checkpoints"]]
    cal_dir = out / "CAL"; cal_dir.mkdir(exist_ok=True)
    cal = {}
    for step in steps:
        cal[step] = register_cal(u, step, guard, H, cal_problems, cal_dir / f"step_{step:04d}.jsonl", device, job, gpu, deadline)
    sel, cks = select({s: [r["E"]["PMI"]["err_full"] for r in recs] for s, recs in cal.items()})
    srec = selection_record(u, sel, cks, cal_dir)
    sp = out / "SELECTION.json"
    if sp.exists():
        require(json.loads(sp.read_text()) == json.loads(json.dumps(plain(srec))), "SELECTION.json differs from the CAL rows")
    else:
        write(sp, srec)                                                   # fixed before any DEV registration
    dev = register_dev(u, sel["step"], guard, H, dev_problems, out / "DEV.jsonl", device, job, gpu, deadline)
    roles_read = sorted({r["role"] for r in guard.reads})
    require(set(roles_read) <= set(ROLES), "A role other than CAL / DEV was read")
    cal_id = [r["J_truth_vs_stored"] for recs in cal.values() for r in recs]
    result = {"status": "COMPLETE", "identity_sha256": sha(ip), "job": job, "gpu": gpu, "stored_gpu": u["stored_gpu"],
              "selection": {"pmi_step": sel["step"], "original_step": u["original"]["step"], "same": srec["same_checkpoint"],
                            "selection_sha256": sha(sp)},
              "CAL_J_truth_vs_stored": {"n": len(cal_id), **{f: sum(x[f] for x in cal_id)
                                                            for f in ("F_true_bitwise", "F_true_within_t01", "n_grid_equal")}},
              "DEV": dev_metrics(dev), "seconds_this_job": time.time() - t0, "utc": utc(),
              "rows_sha256": {**{f"CAL/step_{s:04d}.jsonl": sha(cal_dir / f"step_{s:04d}.jsonl") for s in steps},
                              "DEV.jsonl": sha(out / "DEV.jsonl")},
              "data_reads": len(guard.reads), "roles_read": roles_read}
    write(out / "COMPLETE.json", result)
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("run",)); ap.add_argument("--unit", required=True); ap.add_argument("--out")
    ap.add_argument("--stop-after", type=float, default=None, help="seconds of this job after which no new registration starts")
    a = ap.parse_args()
    require(os.environ.get("SLURM_JOB_ID"), "R22-C2 runs only through Slurm")

    def terminated(signum, frame):
        raise InterruptedError(f"signal {signum}; finished rows are kept")
    signal.signal(signal.SIGTERM, terminated); signal.signal(signal.SIGUSR1, terminated)
    deadline = time.monotonic() + a.stop_after if a.stop_after else None
    u = load_unit(a.unit)
    out = Path(a.out or u["out"])
    try:
        r = run_unit(u, out, deadline=deadline)
    except BaseException as e:
        write(out / f"FAIL_{os.environ.get('SLURM_JOB_ID')}.json", {"utc": utc(), "exception": repr(e), "traceback": traceback.format_exc()})
        raise
    print("COMPLETE", u["job_name"], json.dumps({"selection": r["selection"], "DEV": r["DEV"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
