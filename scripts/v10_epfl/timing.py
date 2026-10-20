#!/usr/bin/env python3
"""V10 EPFL measured full cost (decision 2026-10-17 section 6.2 item 5: "then measure the full EPFL cost; the old proxy cost is never
relabelled as measured"), Slurm only, stage GATE_CHECKS (FIT / CAL windows; DEV is never read).

  python -I scripts/v10_epfl/timing.py [--scope cycle|full_candidate] [--families F1,F2]

Every operation timed is the common protocol's own operation on EPFL inputs (EPFL trainer on the full FIT336, the EPFL registration session,
the unchanged host searcher), synchronised on the GPU:
  scope cycle (default)  one complete checkpoint cycle per family: build (data to GPU) + 250 training steps + one complete resumable
                         checkpoint + one complete CAL pass (all 10 CAL windows x 10 perturbations, preparation + full search + the extra
                         exact-energy evaluation that the protocol makes for F_true, here at a fixed non-truth parameter).  MF_VCS_M1 runs
                         one cycle per iteration cap (10 / 30 / 100).  CoMIR (23 x 32 steps, CAL at the end), RuLSIF (fit + CAL) and the
                         training-free methods (CAL) are timed as their complete candidate.  The full candidate / family / final costs are
                         then assembled from these measured parts (EPFL_COST_MEASURED.json, formulas recorded): measured parts,
                         assembled totals.  Registration outputs (estimates, errors, F values) are not recorded.
  scope full_candidate   the first candidate of each listed family run end to end (3000 steps, 12 checkpoints, 12 complete CAL passes):
                         an end-to-end measurement of one candidate (for the owner, if the assembled totals are not accepted).
--smoke --device cpu: a CPU code-path check (3 steps, one window x one perturbation, one grid angle instead of the full search) under
<V10_EPFL_OUT>/smoke_cpu/; never a cost record.
The validation seed 1117 is used (never 1017 or 1001-1003); nothing is selected.  Costs are labelled with the actual GPU; only L40S rows are
in the decision-8.2 budget unit.  M1's per-step cost depends on the relative-rule iteration counts, which change during training: the cycle
measures the first 250 steps; the per-cap iteration-bound cost (every batch at the cap) is recorded beside it.
"""
from __future__ import annotations

import argparse
import gc
import math
import statistics
import sys
import tempfile
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from v10.epfl import runtime as RT  # noqa: E402

CYCLE_STEPS, WARMUP = 250, 10
SMOKE = {"on": False, "device": "cuda"}
STANDIN_PARAMETER = (0.0, (0.0, 0.0))     # the F_true evaluation of the protocol, at a fixed non-truth point (same cost, no truth used)


def sync():
    import torch
    if SMOKE["device"] == "cuda":
        torch.cuda.synchronize()


def cal_pass(H, A, CP, guard, trainer, family, candidate):
    """One complete CAL pass (10 windows x 10 perturbations) through the EPFL session and the unchanged searcher; timing rows only."""
    import torch
    rows = []
    cal = guard.included("CAL")[:1] if SMOKE["on"] else guard.included("CAL")
    sync(); t0 = time.perf_counter()
    with CP.evaluation_state(trainer):
        session = H.EPFLRegistrationSession(family, trainer, guard, "CAL", cal, SMOKE["device"], candidate)
        for w in cal:
            for i in range(1 if SMOKE["on"] else 10):
                energy, prep = session.energy(w, i)
                if SMOKE["on"]:
                    sync(); g0 = time.perf_counter()
                    with torch.no_grad():
                        energy.grid_values(-9.0, torch.tensor(A.Q.grid_points()[1][:32], dtype=torch.float32, device=SMOKE["device"]))
                    sync(); timed = {"seconds": time.perf_counter() - g0, "grid_seconds": 0.0, "refine_and_final_seconds": 0.0, "n_grid": 32, "exact_energy_calls": 0}
                else:
                    timed = A.register_timed(energy, SMOKE["device"])
                sync(); s0 = time.perf_counter(); energy.value(*STANDIN_PARAMETER); sync(); standin = time.perf_counter() - s0
                rows.append({"window": w, "perturbation": i, "preparation_seconds": prep["preparation_seconds"], "search_seconds": timed["seconds"],
                             "grid_seconds": timed["grid_seconds"], "refine_and_final_seconds": timed["refine_and_final_seconds"],
                             "f_true_standin_seconds": standin, "n_grid": timed["n_grid"], "exact_energy_calls": timed["exact_energy_calls"]})
                del energy
    sync(); total = time.perf_counter() - t0
    per = [r["preparation_seconds"] + r["search_seconds"] + r["f_true_standin_seconds"] for r in rows]
    return {"wall_seconds": total, "n_registrations": len(rows), "sum_registration_seconds": sum(per), "median_registration_seconds": statistics.median(per),
            "max_registration_seconds": max(per), "rows": rows}


def train_steps(trainer, n, every_save=None, save=None):
    times, saves, m1 = [], [], []
    for k in range(n):
        sync(); t0 = time.perf_counter(); rec = trainer.step(); sync(); times.append(time.perf_counter() - t0)
        for key in ("loss", "grad_norm_preclip"):
            if key in rec and not math.isfinite(rec[key]):
                raise RuntimeError(f"non-finite {key} at step {trainer.n_steps}")
        if "m1_iterations" in rec:
            m1.append({"iterations": rec["m1_iterations"], "stop": rec["m1_stop_reason"]})
        if every_save and trainer.n_steps % every_save == 0:
            sync(); s0 = time.perf_counter(); save(); sync(); saves.append(time.perf_counter() - s0)
    return times, saves, m1


def time_family(H, A, CP, guard, family, candidate, scope, tmp, label=None):
    import torch
    if SMOKE["device"] == "cuda":
        torch.cuda.reset_peak_memory_stats()
    sched = H.schedule(family); kind = sched["kind"]
    steps_cycle = 3 if SMOKE["on"] else CYCLE_STEPS
    ident = {"schema": "v10.epfl.timing.v1", "family": family, "candidate": candidate, "seed": RT.VALIDATION_SEED, "phase": "EPFL_TIMING"}
    rec = {"family": family, "label": label or family, "candidate": candidate, "kind": kind, "scope": scope}
    sync(); t0 = time.perf_counter(); trainer = H.new_trainer(family, candidate, RT.VALIDATION_SEED, guard, SMOKE["device"]); sync()
    rec["build_seconds"] = time.perf_counter() - t0
    ckpt = Path(tmp) / f"{family.replace('[', '_').replace(']', '_')}.pt"
    save = lambda: CP.save(trainer, ckpt, ident)   # noqa: E731
    if kind == "gradient_training":
        passes = []
        if scope == "cycle":
            times, saves, m1 = train_steps(trainer, steps_cycle, steps_cycle, save)
            passes.append(cal_pass(H, A, CP, guard, trainer, family, candidate))
        else:
            times, saves, m1 = [], [], []
            for _ in range(sched["steps"] // 250):
                t, s, m = train_steps(trainer, 250, 250, save); times += t; saves += s; m1 += m
                passes.append(cal_pass(H, A, CP, guard, trainer, family, candidate))
        steady = times[WARMUP:] or times
        rec.update({"steps_timed": len(times), "step_seconds_mean_after_warmup": statistics.fmean(steady), "step_seconds_median_after_warmup": statistics.median(steady),
                    "step_seconds_first": times[:WARMUP], "train_seconds_total": sum(times), "checkpoint_seconds": saves,
                    "cal_passes": [{k: v for k, v in p.items() if k != "rows"} for p in passes], "cal_pass_rows_first": passes[0]["rows"]})
        if m1:
            its = [x["iterations"] for x in m1]
            rec["m1"] = {"cap": candidate.get("cap"), "iterations_mean": statistics.fmean(its), "iterations_max": max(its),
                         "share_at_cap": sum(x["stop"] == "iteration_budget_exhausted" for x in m1) / len(m1),
                         "stop_reasons": {s: sum(x["stop"] == s for x in m1) for s in {x["stop"] for x in m1}}}
            if scope == "cycle":   # iteration-bound step cost: the same cap with a relative tolerance no batch can meet (timing only)
                trainer.cfg.m1 = A.M1Config(max_iterations=candidate["cap"], relative_tolerance=1e-12)
                bt, _, bm = train_steps(trainer, 3 if SMOKE["on"] else 20)
                rec["m1"].update({"capbound_step_seconds_mean": statistics.fmean(bt[2:]), "capbound_iterations": [x["iterations"] for x in bm][:5],
                                  "capbound_note": "20 steps after the cycle with relative_tolerance 1e-12 (every batch runs to the cap); timing only"})
    elif kind == "native_gradient_training":                          # CoMIR: the whole recipe is the candidate
        times, saves, _ = train_steps(trainer, 3 if SMOKE["on"] else sched["steps"], 3 if SMOKE["on"] else 32, save)
        p = cal_pass(H, A, CP, guard, trainer, family, candidate)
        rec.update({"steps_timed": len(times), "train_seconds_total": sum(times), "step_seconds_mean_after_warmup": statistics.fmean(times[WARMUP:] or times),
                    "checkpoint_seconds": saves, "cal_passes": [{k: v for k, v in p.items() if k != "rows"}], "cal_pass_rows_first": p["rows"]})
    elif kind == "kernel_fit":
        sync(); f0 = time.perf_counter(); info = trainer.step(); sync(); rec["fit_seconds"] = time.perf_counter() - f0
        rec["fit_info"] = {k: info[k] for k in ("scene_count", "negative_coefficients", "linear_solve_residual")}; rec["centres"] = trainer.fit_metadata["centres"]
        sync(); s0 = time.perf_counter(); save(); sync(); rec["checkpoint_seconds"] = [time.perf_counter() - s0]
        p = cal_pass(H, A, CP, guard, trainer, family, candidate)
        rec.update({"cal_passes": [{k: v for k, v in p.items() if k != "rows"}], "cal_pass_rows_first": p["rows"]})
    else:
        sync(); s0 = time.perf_counter(); save(); sync(); rec["checkpoint_seconds"] = [time.perf_counter() - s0]
        p = cal_pass(H, A, CP, guard, trainer, family, candidate)
        rec.update({"cal_passes": [{k: v for k, v in p.items() if k != "rows"}], "cal_pass_rows_first": p["rows"]})
    rec["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30 if SMOKE["device"] == "cuda" else None
    del trainer; gc.collect()
    if SMOKE["device"] == "cuda":
        torch.cuda.empty_cache()
    return rec


def assemble(H, recs: dict) -> dict:
    """Candidate / family / final-seed GPU hours from the measured parts (formulas recorded with every number)."""
    out = {}
    for fam in H.FAMILIES:
        rows = [r for r in recs.values() if r["family"] == fam and r["scope"] == "cycle"]
        if not rows:
            continue
        cands = H.candidates(fam); sched = H.schedule(fam); n_ckpt = len(sched["cal_steps"])
        dev_factor = H.EPFL_DEV_WINDOWS / H.EPFL_CAL_WINDOWS

        def cand_seconds(r):
            cal = r["cal_passes"][0]["wall_seconds"]; ck = statistics.fmean(r["checkpoint_seconds"]) if r["checkpoint_seconds"] else 0.0
            if r["kind"] == "gradient_training":
                return r["build_seconds"] + sched["steps"] * r["step_seconds_mean_after_warmup"] + n_ckpt * (ck + cal), cal
            if r["kind"] == "native_gradient_training":
                return r["build_seconds"] + r["train_seconds_total"] + sum(r["checkpoint_seconds"]) + cal, cal
            if r["kind"] == "kernel_fit":
                return r["build_seconds"] + r["fit_seconds"] + ck + cal, cal
            return r["build_seconds"] + ck + cal, cal
        if fam == "MF_VCS_M1":
            per_cap = {r["candidate"]["cap"]: cand_seconds(r) for r in rows}
            cal_seconds = sum(per_cap[c["cap"]][0] for c in cands)
            final_by_cap = {cap: cs + cal_ * dev_factor for cap, (cs, cal_) in per_cap.items()}
            bound = {r["candidate"]["cap"]: r["build_seconds"] + sched["steps"] * r["m1"]["capbound_step_seconds_mean"]
                     + n_ckpt * (statistics.fmean(r["checkpoint_seconds"]) + r["cal_passes"][0]["wall_seconds"]) for r in rows if "m1" in r}
            out[fam] = {"candidates": len(cands), "candidate_hours_by_cap": {c: v[0] / 3600 for c, v in per_cap.items()},
                        "candidate_hours_by_cap_iteration_bound": {c: v / 3600 for c, v in bound.items()},
                        "cal_family_hours": cal_seconds / 3600, "cal_family_hours_iteration_bound": sum(bound[c["cap"]] for c in cands) / 3600 if len(bound) == 3 else None,
                        "final_seed_hours_by_selected_cap": {c: v / 3600 for c, v in final_by_cap.items()},
                        "family_total_hours_by_selected_cap": {c: (cal_seconds + len(sched["final_seeds"]) * v) / 3600 for c, v in final_by_cap.items()},
                        "within_8h_cal_cap": cal_seconds / 3600 <= 8.0, "m1_iteration_note": "first-250-step iteration counts; later steps may differ; "
                        "the iteration bound (every batch at the cap) is the upper reference"}
            continue
        s, cal = cand_seconds(rows[0])
        cal_seconds = len(cands) * s; final = s + cal * dev_factor
        out[fam] = {"candidates": len(cands), "candidate_hours": s / 3600, "cal_family_hours": cal_seconds / 3600, "final_seed_hours": final / 3600,
                    "final_seeds": len(sched["final_seeds"]), "family_total_hours": (cal_seconds + len(sched["final_seeds"]) * final) / 3600,
                    "within_8h_cal_cap": cal_seconds / 3600 <= 8.0}
    tot = sum(v.get("family_total_hours", max(v.get("family_total_hours_by_selected_cap", {0: 0}).values())) for v in out.values())
    return {"families": out, "total_hours_upper_over_m1_caps": tot,
            "formulas": {"gradient_candidate": "build + 3000 x mean step (after 10 warm-up steps) + 12 x (checkpoint + complete CAL pass)",
                         "comir_candidate": "measured whole recipe (build + 736 steps + 23 checkpoints + CAL pass)",
                         "rulsif_candidate": "build (FIT patches) + fit + checkpoint + CAL pass", "training_free_candidate": "build + checkpoint + CAL pass",
                         "cal_family": "candidates x candidate (M1: sum over its 9 lr x cap candidates, lr does not change the cost)",
                         "final_seed": "candidate + DEV pass, DEV pass = CAL pass x 74 / 10 (same window size and searcher; DEV not read at this stage)",
                         "family_total": "cal_family + final seeds x final_seed"},
            "excludes": ["queue wait", "failed / retried jobs", "job start-up outside the timed operations (Python import, CUDA init)",
                         "the DEV pass itself (assembled from CAL per-registration cost)", "M1 iteration counts after step 250"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--scope", choices=("cycle", "full_candidate"), default="cycle")
    ap.add_argument("--families", default=""); ap.add_argument("--device", default="cuda", choices=("cuda", "cpu")); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); pins = RT.job_pins()
    if a.smoke != (a.device == "cpu"):
        raise SystemExit("--smoke runs on --device cpu only, and the CPU device only as a smoke")
    SMOKE.update({"on": a.smoke, "device": a.device})
    import torch
    from v10 import adapter as A
    from v10.epfl import host as H
    from v10.supplement import checkpoints as CP
    env = RT.runtime_environment(A) if not a.smoke else {"smoke": True, **A.environment(None), "gpu_name": None, "budget_unit_gpu": False}
    guard = RT.guard_from_pins(pins)
    root = RT.OUT_ROOT / ("smoke_cpu/timing" if a.smoke else "timing") / pins["job"]; root.mkdir(parents=True, exist_ok=True)
    fams = [f for f in a.families.split(",") if f] or list(H.FAMILIES)
    plan = []
    for fam in fams:
        if fam == "MF_VCS_M1" and a.scope == "cycle":
            plan += [(fam, c, f"MF_VCS_M1_cap{c['cap']}") for c in H.candidates(fam) if c["lr"] == 1e-4]
        else:
            plan.append((fam, H.candidates(fam)[0], fam))
    recs, failures = {}, {}
    with tempfile.TemporaryDirectory(dir=str(RT.OUT_ROOT)) as tmp:
        for fam, cand, label in plan:
            print(f"[timing] {label} start {A.utc_now()}", flush=True)
            try:
                recs[label] = time_family(H, A, CP, guard, fam, cand, a.scope, tmp, label)
                RT.write(root / f"TIMING_{label.replace('[', '_').replace(']', '_')}.json", {"environment": env, "pins": pins, **recs[label]})
                print(f"[timing] {label} done", flush=True)
            except Exception as e:  # noqa: BLE001
                failures[label] = {"error": repr(e)[:2000], "traceback": traceback.format_exc()[-6000:]}
                print(f"[timing] {label} FAILED {e!r}", flush=True)
                gc.collect()
                if not a.smoke:
                    torch.cuda.empty_cache()
    roles = sorted({r["role"] for r in guard.reads})
    summary = {"schema": "v10.epfl.cost.v1", "decision": "V10-phase1-decisions-20261017 section 6.2 item 5", "scope": a.scope, "pins": pins,
               "environment": env, "gpu": env["gpu_name"], "budget_unit_gpu": env["budget_unit_gpu"], "data_roles_read": roles,
               "status": ("SMOKE_OK_NOT_A_COST_RECORD" if a.smoke else "MEASURED") if recs and not failures and roles == ["CAL", "FIT"] else "INCOMPLETE",
               "failures": failures,
               "measured": {k: {x: v[x] for x in v if x not in ("cal_pass_rows_first",)} for k, v in recs.items()},
               "assembled": assemble(H, recs) if a.scope == "cycle" else None,
               "old_proxy_not_used": "the phase-1 BUDGET EPFL proxy (BF timing x registration count) is not used anywhere here"}
    RT.write(root / "EPFL_COST_MEASURED.json", summary)
    print(f"[timing] {summary['status']} -> {root / 'EPFL_COST_MEASURED.json'}", flush=True)
    return 0 if summary["status"] in ("MEASURED", "SMOKE_OK_NOT_A_COST_RECORD") else 1


if __name__ == "__main__":
    sys.exit(main())
