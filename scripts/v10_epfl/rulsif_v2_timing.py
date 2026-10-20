#!/usr/bin/env python3
"""V10 EPFL measured cost of RuLSIF_exact_v2 (EPFL mapping of EPFL_RULSIF_V2_ADDENDUM.json), Slurm only, L40S; data gate = the GATE_CHECKS
record + the frozen group record v2 (FIT / CAL windows; DEV is never read).  Validation seed 1117; nothing is selected.

  python -I scripts/v10_epfl/rulsif_v2_timing.py --centres 1024|2048 [--registrations 100]
Each T has its own job and out-root (<V10_EPFL_OUT>/rulsif_v2_timing/T<T>/<job>); a centre-provenance sidecar is written beside the record.

Measured parts (synchronised on the GPU), for candidate order 0 (the cost does not depend on the multiplier or the ridge):
  build        trainer construction: FIT patches of the 334 windows (4096 disk positions each), T paired centres (addendum revision 3:
               floor(T / 334) per window, the extras by the fixed seed-independent order), FIT-only medians
  fit          the complete analytic fit: fp64 exact V moments over 334 scenes and the fp64 solve of T + 1 coefficients
  checkpoint   one complete checkpoint save
  registration k complete registrations on CAL windows (default: one complete CAL pass, 10 windows x 10 perturbations): EPFL session
               preparation + the unchanged
               searcher (21 x 508 grid + 3 x 100 Adam) + the protocol's extra exact-energy evaluation (F_true stand-in at a fixed
               non-truth point); no registration error is computed
Assembled (formulas recorded): candidate = build + fit + checkpoint + 100 x mean registration (CAL 10 x 10); CAL family = 8 eligible
candidates; final seed = candidate + 740 x mean registration (DEV 74 x 10, not read here); the 8 L40S GPU-h cap analysis (complete
candidates that fit, resource-limited flag) and the job-length consequence for finals (Slurm limit 23:59; the runner resumes).
--smoke --device cpu: CPU code path (smoke roles, reduced search, 1 registration) under <V10_EPFL_OUT>/smoke_cpu/; never a cost record.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import math
import statistics
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from v10.epfl import runtime as RT  # noqa: E402

FAMILY = "RuLSIF_exact_v2"
CAP_SECONDS, CAP_MARGIN = 8 * 3600, 30
SLURM_MAX_SECONDS = 23 * 3600 + 59 * 60
N_CAL, N_DEV = 100, 740
STANDIN_PARAMETER = (0.0, (0.0, 0.0))


def assemble(m: dict, n_candidates: int, n_final_seeds: int = 3) -> dict:
    """Candidate / family / final-seed seconds from the measured parts (pure arithmetic; tested on CPU)."""
    reg = m["registration_seconds_mean"]
    cand = m["build_seconds"] + m["fit_seconds"] + m["checkpoint_seconds"] + N_CAL * reg
    cal_family = n_candidates * cand
    budget = CAP_SECONDS - CAP_MARGIN
    within = int(budget // cand) if cand > 0 else n_candidates
    complete = min(n_candidates, within)
    final = cand + N_DEV * reg
    jobs_per_final = max(1, math.ceil(final / (SLURM_MAX_SECONDS - 600)))
    return {"candidate_hours": cand / 3600, "cal_pass_hours": N_CAL * reg / 3600, "cal_family_hours_all_candidates": cal_family / 3600,
            "cal_family_hours_charged": min(cal_family, budget) / 3600, "complete_candidates_within_cap": complete,
            "resource_limited": cal_family > budget, "final_seed_hours": final / 3600, "final_seeds": n_final_seeds,
            "finals_hours": n_final_seeds * final / 3600, "jobs_per_final_seed": jobs_per_final,
            "family_total_hours": (min(cal_family, budget) + n_final_seeds * final) / 3600,
            "formulas": {"candidate": "build + fit + checkpoint + 100 x mean registration (CAL 10 windows x 10 perturbations)",
                         "cal_family": f"{n_candidates} eligible candidates x candidate; charged at most 8 h - 30 s (decision 4)",
                         "complete_candidates_within_cap": "floor((8 h - 30 s) / candidate), at most the number of candidates; job start-up excluded",
                         "final_seed": "candidate + 740 x mean registration (DEV 74 windows x 10 perturbations at the selected checkpoint)",
                         "jobs_per_final_seed": "ceil(final_seed / (23:59 - 10 min)); the runner resumes the resumable DEV block"}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--registrations", type=int, default=100)
    ap.add_argument("--centres", type=int, required=True, choices=(1024, 2048))
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu")); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); dev = a.device
    if a.smoke != (dev == "cpu"):
        raise SystemExit("--smoke runs on --device cpu only, and the CPU device only as a smoke")
    pins = RT.job_pins()
    RT.require(pins.get("group_record_sha256"), "the v2 jobs need the group-record pin V10_EPFL_GROUP_SHA256")
    import torch
    from v10 import adapter as A
    from v10.epfl import host as H
    from v10.supplement import checkpoints as CP
    V2, C = H.rulsif_v2_module()
    if a.smoke:
        A.Q.TH_STEP, A.P.GRID_STEP, A.P.N_ADAM = 15.0, 17.0, 2
    env = RT.runtime_environment(A) if not a.smoke else {"smoke": True, **A.environment(None), "gpu_name": None, "budget_unit_gpu": False}
    if not a.smoke:
        RT.require(env["gpu_name"] == RT.BUDGET_GPU, "the v2 timing is measured on L40S only (budget unit)")
    T = a.centres
    root = RT.OUT_ROOT / ("smoke_cpu/rulsif_v2_timing" if a.smoke else "rulsif_v2_timing") / f"T{T}" / pins["job"]; root.mkdir(parents=True, exist_ok=True)
    base = RT.guard_from_pins(pins)
    RT.require(base.stage == "GATE_CHECKS", "the v2 timing runs at stage GATE_CHECKS (DEV refused)")
    if a.smoke:
        spec = importlib.util.spec_from_file_location("epfl_run", REPO / "scripts" / "v10_epfl" / "run.py")
        run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run); guard = run.SmokeGuard(base)
    else:
        guard = base
    sync = (lambda: torch.cuda.synchronize()) if dev == "cuda" else (lambda: None)
    cands = C.candidates(FAMILY); cand = cands[0]; seed = RT.VALIDATION_SEED
    ident = {"schema": "v10.epfl.rulsif_v2_timing.v1", "family": FAMILY, "candidate": cand, "seed": seed, "phase": "EPFL_RULSIF_V2_TIMING"}
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    m = {"family": FAMILY, "candidate": cand, "n_fit_windows": len(H.fit_windows(guard)), "centres_total": T}
    sync(); t0 = time.perf_counter(); trainer = H.new_trainer(FAMILY, cand, seed, guard, dev, centres_total=T); sync(); m["build_seconds"] = time.perf_counter() - t0
    H.verify_centre_provenance(trainer.provenance, trainer.fit_metadata, T); RT.write(root / "CENTRE_PROVENANCE.json", trainer.provenance)
    m["provenance_summary"] = {k: trainer.provenance[k] for k in ("cx_sha256", "cy_sha256", "indices_sha256", "fit_patches_sha256", "extra_windows")}
    sync(); t0 = time.perf_counter(); info = trainer.step(); sync(); m["fit_seconds"] = time.perf_counter() - t0
    m["fit_info"] = {k: info[k] for k in ("scene_count", "coefficient_count", "linear_solve_residual", "negative_coefficients", "mixture_mean_of_g")}
    m["centres"] = trainer.fit_metadata["centres"]
    with tempfile.TemporaryDirectory(dir=str(RT.OUT_ROOT)) as tmp:
        sync(); t0 = time.perf_counter(); CP.save(trainer, Path(tmp) / "v2.pt", ident); sync(); m["checkpoint_seconds"] = time.perf_counter() - t0
    cal = guard.included("CAL")[:1] if a.smoke else guard.included("CAL")
    k = 1 if a.smoke else a.registrations
    plan = [(w, i) for w in cal for i in range(10)][:k]   # window-major, as the CAL pass
    rows = []
    with CP.evaluation_state(trainer):
        session = H.EPFLRegistrationSession(FAMILY, trainer, guard, "CAL", sorted({w for w, _ in plan}), dev, cand)
        for w, i in plan:
            energy, prep = session.energy(w, i)
            if a.smoke:
                sync(); g0 = time.perf_counter()
                with torch.no_grad():
                    energy.grid_values(-9.0, torch.tensor(A.Q.grid_points()[1][:16], dtype=torch.float32, device=dev))
                sync(); timed = {"seconds": time.perf_counter() - g0, "grid_seconds": 0.0, "refine_and_final_seconds": 0.0, "n_grid": 16, "exact_energy_calls": 0}
            else:
                timed = A.register_timed(energy, dev)
            sync(); s0 = time.perf_counter(); energy.value(*STANDIN_PARAMETER); sync(); standin = time.perf_counter() - s0
            rows.append({"window": w, "perturbation": i, "preparation_seconds": prep["preparation_seconds"], "search_seconds": timed["seconds"],
                         "grid_seconds": timed["grid_seconds"], "refine_and_final_seconds": timed["refine_and_final_seconds"],
                         "f_true_standin_seconds": standin, "n_grid": timed["n_grid"], "exact_energy_calls": timed["exact_energy_calls"]})
            print(f"[r2timing] registration {w}#{i}: {rows[-1]['preparation_seconds'] + rows[-1]['search_seconds'] + standin:.1f}s", flush=True)
            del energy
    per = [r["preparation_seconds"] + r["search_seconds"] + r["f_true_standin_seconds"] for r in rows]
    m.update({"registrations": rows, "registration_seconds_mean": statistics.fmean(per), "registration_seconds_median": statistics.median(per),
              "registration_seconds_max": max(per), "n_registrations_timed": len(per)})
    m["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30 if dev == "cuda" else None
    roles = sorted({r["role"] for r in base.reads})
    summary = {"schema": "v10.epfl.rulsif_v2_cost.v1", "family": FAMILY, "centres_total": T, "kernel_role": H.RULSIF_V2_CENTRES[T],
               "pins": pins, "environment": env, "gpu": env.get("gpu_name"),
               "budget_unit_gpu": env.get("budget_unit_gpu"), "data_roles_read": roles, "candidates": len(cands), "measured": m,
               "assembled": assemble(m, len(cands), len(C.schedule(FAMILY)["final_seeds"])),
               "excludes": ["queue wait", "job start-up outside the timed operations (imports, CUDA init, data-lock hashing)", "failed / retried jobs"],
               "status": ("SMOKE_OK_NOT_A_COST_RECORD" if a.smoke else "MEASURED") if roles == ["CAL", "FIT"] else "INCOMPLETE"}
    RT.write(root / "EPFL_RULSIF_V2_COST_MEASURED.json", summary)
    print(f"[r2timing] {summary['status']} -> {root / 'EPFL_RULSIF_V2_COST_MEASURED.json'}; assembled {summary['assembled']}", flush=True)
    del trainer, session; gc.collect()
    return 0 if summary["status"] in ("MEASURED", "SMOKE_OK_NOT_A_COST_RECORD") else 1


if __name__ == "__main__":
    sys.exit(main())
