#!/usr/bin/env python3
"""V10 EPFL GPU gate rows for RuLSIF_exact_v2 (the v2 code merged unchanged from v10/rulsif-exact-v2 2e6ea58; EPFL mapping of
EPFL_RULSIF_V2_ADDENDUM.json).  Slurm only; data gate = the GATE_CHECKS activation record + the frozen group record v2 (FIT 334, CAL 10;
DEV refused).  Validation seed 1117 (never 1017 or 1001-1003); no registration error, no selection, no scientific result.

  python -I scripts/v10_epfl/rulsif_v2_gpu_gate.py [--parts env,numerics,family,capacity2048,guard]

Parts (PASS = every required row of every part passes):
  env       the accepted common Torch / CUDA / cuDNN / precision environment; the GPU must be an L40S (budget unit)
  numerics  tests/v10/test_rulsif_v2.py on CUDA (the v2 numerical identities, as the BF v2 GPU gate)
  family    (mapping rows) the EPFL mapping (addendum revision 3): FIT set 334, 4096 disk-support positions per FIT window, 1024 centres in
            total (22 windows with 4 by the fixed seed-independent extra order, 312 with 3), centre provenance (sidecar = resolved_kernel =
            the model's centre arrays), the EPFL stream tag, host
            constants (R_OMEGA 129, CEN 255.5, SIZE 512), per-modality widths, copula transform, the FIT input (cache 'bf' / 'shg')
            equal to the registration input at the zero perturbation (REG-P3 reference / linear-NIR moving path)
            then, for candidate order 0: the complete analytic fit on FIT334 (fp64 solve: 1025 coefficients), checkpoint round trip and
            identity refusal, frozen evaluation state, one CAL-window energy with a finite geometry gradient vs fp32 central
            differences, one full common search (21 x 508 grid + 3 x 100 Adam) timed, FIT / CAL reads only, peak GPU memory
  capacity2048  the pre-registered 2048-centre sensitivity: complete fit (44 windows with 7, 290 with 6; the 44 include the 22), the FIT
            positions identical to the 1024 fit at the same seed, provenance, one full common search timed
  guard     DEV refused, the two group-record removals refused, excluded pairs / BF records / BF held cores refused, data lock FIT + CAL
Outputs <V10_EPFL_OUT>/rulsif_v2_gpu_gate/<job>/{GATE.json, <part>.json}.  --smoke --device cpu: CPU code-path check on smoke roles
(4 FIT windows, 256 centres each, reduced search) under <V10_EPFL_OUT>/smoke_cpu/rulsif_v2_gpu_gate/<job>/; never a gate result.
"""
from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from v10.epfl import runtime as RT  # noqa: E402

FAMILY = "RuLSIF_exact_v2"
PARTS = ("env", "numerics", "family", "capacity2048", "guard")


def summarize(rows):
    req = [r for r in rows if r.get("required", True)]
    return {"status": "PASS" if req and all(r["status"] == "PASS" for r in req) else "FAIL", "n": len(rows), "n_required": len(req),
            "required_fail": [r["name"] for r in req if r["status"] != "PASS"],
            "diagnostic_fail": [r["name"] for r in rows if not r.get("required", True) and r["status"] != "PASS"]}


def smoke_guard(base):
    """The production runner's SmokeGuard (4 FIT, 2 CAL, 2 CAL windows relabelled DEV; no DEV window read)."""
    spec = importlib.util.spec_from_file_location("epfl_run", REPO / "scripts" / "v10_epfl" / "run.py")
    run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)
    return run.SmokeGuard(base)


def allocation_rows(H, guard, trainer, total, ref=None):
    """Revision-3 allocation and centre provenance (sidecar = resolved_kernel = the model's centre arrays)."""
    from v10.epfl.geometry import row
    md, prov = trainer.fit_metadata, trainer.provenance; fit = H.fit_windows(guard); smoke = getattr(guard, "smoke", False)
    counts = H.centre_counts(fit, total); extra = H.extra_centre_order(fit)[:total % len(fit)]
    hist = {str(c): counts.count(c) for c in sorted(set(counts))}
    want = {1024: {"3": 312, "4": 22}, 2048: {"6": 290, "7": 44}}[total]
    rows = [row(f"alloc{total}.counts_revision3", prov["counts"] == counts and md["total_centres"] == total == sum(counts)
                and md["centres_per_window_histogram"] == hist and (smoke or hist == want), histogram=hist),
            row(f"alloc{total}.extra_windows_fixed_order", prov["extra_windows"] == extra == md["extra_windows"]
                and (smoke or len(extra) == {1024: 22, 2048: 44}[total]), extra_windows=extra)]
    if ref is not None:
        rows.append(row(f"alloc{total}.extra_windows_include_the_1024_extras", set(ref["extra_windows"]) <= set(extra)))
        rows.append(row(f"alloc{total}.fit_positions_identical_to_the_1024_fit", prov["fit_patches_sha256"] == ref["fit_patches_sha256"]
                        and prov["stream_tag"] == ref["stream_tag"], note="positions are drawn before the centres on the same per-seed stream"))
    try:
        H.verify_centre_provenance(prov, md, total); ok = True; err = None
    except RuntimeError as e:
        ok, err = False, str(e)
    rows.append(row(f"alloc{total}.provenance_consistent", ok and H.array_sha256(trainer.ratio.centres_left) == prov["cx_sha256"]
                    and H.array_sha256(trainer.ratio.centres_right) == prov["cy_sha256"], error=err,
                    cx_sha256=prov["cx_sha256"], cy_sha256=prov["cy_sha256"], indices_sha256=prov["indices_sha256"]))
    return rows


def write_sidecar(root, trainer, name):
    RT.write(Path(root) / "provenance" / f"{name}.json", trainer.provenance)
    return str(Path(root) / "provenance" / f"{name}.json")


def mapping_rows(H, A, V2, guard, trainer, candidate, seed, device):
    import numpy as np
    import torch
    from v10.epfl.geometry import row
    rows = []; md = trainer.fit_metadata; fit = H.fit_windows(guard); n_fit = len(fit)
    rows.append(row("mapping.fit_set_is_the_group_record_fit", md["fit_cores"] == fit and (getattr(guard, "smoke", False) or n_fit == 334),
                    n_fit_windows=n_fit))
    rows.append(row("mapping.removed_windows_not_in_fit", not ({"water_0006", "street_0020"} & set(md["fit_cores"]))))
    rows += allocation_rows(H, guard, trainer, 1024)
    rows.append(row("mapping.positions_4096_per_window", md["fit_positions_per_core"] == 4096 and len(trainer.scenes) == n_fit
                    and all(tuple(s[0].shape) == (4096, 81) and tuple(s[1].shape) == (4096, 81) for s in trainer.scenes)))
    rows.append(row("mapping.stream_tag_epfl", md["stream_tag"] == H.rulsif_v2_stream_tag(seed), stream_tag=md["stream_tag"]))
    rows.append(row("mapping.support_disk_r129", md["fit_support"] == "disk_r129" and candidate["fit_support"] == "disk_r129"))
    rows.append(row("mapping.host_constants", A.Q.R_OMEGA == 129 and float(A.Q.CEN) == 255.5 and A.P.SIZE == 512,
                    R_OMEGA=A.Q.R_OMEGA, CEN=float(A.Q.CEN), SIZE=A.P.SIZE))
    pos = V2.support_positions(np.random.default_rng(20261019), 4096, "disk_r129")
    radius = np.sqrt(((pos - A.Q.CEN) ** 2).sum(-1))
    rows.append(row("mapping.disk_positions_inside_support_and_patches_inside_window", bool(radius.max() <= A.Q.R_OMEGA + 1e-9)
                    and bool(pos.min() - 4 >= 0) and bool(pos.max() + 4 <= A.P.SIZE - 1), max_radius=float(radius.max())))
    m = trainer.ratio; mult = candidate["bandwidth_multiplier"]
    rows.append(row("mapping.per_modality_widths", md["bandwidth_rule"] == "per_modality_median"
                    and math.isclose(m.bandwidth_left, md["bandwidth_median_left"] * mult, rel_tol=1e-12)
                    and math.isclose(m.bandwidth_right, md["bandwidth_median_right"] * mult, rel_tol=1e-12),
                    h_x=m.bandwidth_left, h_y=m.bandwidth_right))
    rows.append(row("mapping.copula_transform", m.input_transform == "copula" and md["input_transform"] == "copula"))
    rows.append(row("mapping.intensity_input_path", candidate["shg_input"] == "intensity" and md["shg_input"] == "intensity"))
    w = fit[0]
    inputs = H.EPFLGuardedInputs(guard, {"FIT"}, "V10 EPFL RuLSIF_exact_v2 gate: FIT input vs registration input", {w: [[0.0, 0.0, 0.0]]}, device)
    bf, shg = A.P.load_img(guard.r6_root, w, ("bf", "shg"))
    ref = inputs.reference(w)[0].detach().cpu().double().numpy()[0]; mov = inputs.moving(w, 0)[0].detach().cpu().double().numpy()[0]
    d_ref, d_mov = float(np.abs(ref - bf).max()), float(np.abs(mov - shg).max())
    rows.append(row("mapping.fit_reference_equals_registration_reference", d_ref == 0.0, max_abs=d_ref, window=w))
    rows.append(row("mapping.fit_moving_equals_registration_moving_at_zero_perturbation", d_mov <= 1e-5, max_abs=d_mov, window=w,
                    note="cache 'shg' (linear NIR robust-norm) vs REG-P3 Inputs3 moving at (0, 0, 0)"))
    del inputs
    return rows


def family_rows(H, A, CP, V2, guard, device, smoke, root):
    import numpy as np
    import torch
    from v10.epfl.geometry import row
    from v10.supplement.training import rng_state
    rows, info = [], {"family": FAMILY}
    candidate = H.production_candidates(FAMILY)[0]; seed = RT.VALIDATION_SEED
    ident = {"schema": "v10.epfl.rulsif_v2_gate.v1", "family": FAMILY, "candidate": candidate, "seed": seed, "phase": "EPFL_RULSIF_V2_GATE"}
    cuda = str(device).startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter(); trainer = H.new_trainer(FAMILY, candidate, seed, guard, device, centres_total=1024); A.synchronize(device)
    info["build_seconds"] = time.perf_counter() - t0
    rows += mapping_rows(H, A, V2, guard, trainer, candidate, seed, device)
    info["provenance_sidecar"] = write_sidecar(root, trainer, f"T1024_order{candidate['order']}_seed{seed}")
    info["provenance_summary"] = {k: trainer.provenance[k] for k in ("stream_tag", "extra_order_tag", "extra_windows", "cx_sha256", "cy_sha256",
                                                                      "indices_sha256", "centre_yx_sha256", "fit_patches_sha256", "medians", "median_pairs")}
    t0 = time.perf_counter(); fit_info = trainer.step(); A.synchronize(device); info["fit_seconds"] = time.perf_counter() - t0
    n = len(H.fit_windows(guard)); coeff = trainer.ratio.coefficients
    rows.append(row("family.kernel_fit_complete", fit_info["scene_count"] == n and fit_info["coefficient_count"] == 1025 and trainer.n_steps == 1,
                    scene_count=fit_info["scene_count"], coefficient_count=fit_info["coefficient_count"]))
    rows.append(row("family.fit_solve_finite", bool(torch.isfinite(coeff).all()) and math.isfinite(fit_info["linear_solve_residual"])
                    and math.isfinite(fit_info["J_fit_moments"]), residual=fit_info["linear_solve_residual"], J_fit=fit_info["J_fit_moments"],
                    negative_coefficients=fit_info["negative_coefficients"]))
    rows.append(row("family.mixture_mean_of_g_is_one", abs(fit_info["mixture_mean_of_g"] - 1.0) <= 1e-6, value=fit_info["mixture_mean_of_g"],
                    note="theta_0 outside the ridge: E_{(P+Q)/2}[g] = 1 exactly on FIT"))
    info["fit_info"] = {k: fit_info[k] for k in ("scene_count", "coefficient_count", "linear_solve_residual", "J_fit_moments", "mixture_mean_of_g",
                                                  "negative_coefficients", "constant_coefficient", "bandwidth_left", "bandwidth_right")}
    with tempfile.TemporaryDirectory(dir=str(RT.OUT_ROOT)) as tmp:
        cp = Path(tmp) / "v2.pt"; before = coeff.detach().cpu().clone()
        CP.save(trainer, cp, ident); trainer.ratio.coefficients.zero_(); CP.load(trainer, cp, ident)
        rows.append(row("family.checkpoint_round_trip_bitwise", torch.equal(trainer.ratio.coefficients.detach().cpu(), before)))
        try:
            CP.load(trainer, cp, {**ident, "seed": 999}); wrong = False
        except RuntimeError:
            wrong = True
        rows.append(row("family.checkpoint_refuses_other_identity", wrong))
    st = copy.deepcopy(rng_state(trainer)); h0 = A.state_hash(*trainer.modules())
    with CP.evaluation_state(trainer):
        cal = guard.included("CAL")[0]
        session = H.EPFLRegistrationSession(FAMILY, trainer, guard, "CAL", [cal], device, candidate)
        energy, prep = session.energy(cal, 0); info["preparation"] = prep
        raw = energy.energy
        rows.append(row("family.energy_is_v2_class", isinstance(raw, V2.RuLSIFV2RigidEnergy)))
        rows.append(row("family.moving_copula_values_in_open_unit_interval", float(raw.mov.min()) > 0.0 and float(raw.mov.max()) < 1.0))
        p = torch.tensor([.37, .23, -.41], device=device, requires_grad=True)
        value = energy(p); grad = torch.autograd.grad(value, p)[0]
        rows.append(row("family.geometry_gradient_finite_nonzero", bool(torch.isfinite(grad).all()) and float(grad.abs().max()) > 0, grad=grad.tolist()))
        h = .01; eye = torch.eye(3, device=device) * h
        with torch.no_grad():
            fd = torch.stack([(energy(p.detach() + e) - energy(p.detach() - e)) / (2 * h) for e in eye])
        rows.append(row("family.geometry_gradient_vs_fp32_central_difference", bool(torch.allclose(grad, fd, atol=5e-4, rtol=.05)),
                        grad=grad.tolist(), fd=fd.tolist(), atol=5e-4, rtol=.05))
        if smoke:
            with torch.no_grad():
                gv = energy.grid_values(-9.0, torch.tensor(A.Q.grid_points()[1][:16], dtype=torch.float32, device=device))
            rows.append(row("family.smoke_grid_values_finite", bool(torch.isfinite(gv).all()) and len(gv) == 16))
            info["full_search_timing"] = None
        else:
            timing = A.register_timed(energy, device)
            rows.append(row("family.full_common_search_grid", timing["n_grid"] == 21 * 508 and math.isfinite(timing["F_hat"]),
                            n_grid=timing["n_grid"], seconds=timing["seconds"]))
            info["full_search_timing"] = {k: timing[k] for k in ("seconds", "grid_seconds", "refine_and_final_seconds", "grid_calls", "exact_energy_calls", "n_grid")}
    rows.append(row("family.evaluation_restores_rng", rng_state(trainer) == st))
    rows.append(row("family.evaluation_did_not_change_the_model", A.state_hash(*trainer.modules()) == h0))
    roles = {r["role"] for r in guard.reads}
    rows.append(row("family.data_roles_read_FIT_CAL_only", roles <= {"FIT", "CAL"}, roles=sorted(roles)))
    info["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else None
    del trainer, session, energy, value, grad; gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    return rows, info


def capacity_rows(H, A, V2, guard, device, smoke, root, ref):
    """The 2048-centre sensitivity at the gate: complete fit, allocation / provenance, positions identical to the 1024 fit, one full search."""
    import torch
    from v10.epfl.geometry import row
    from v10.supplement import checkpoints as CP
    rows, info = [], {"family": FAMILY, "centres_total": 2048}
    candidate = H.production_candidates(FAMILY)[0]; seed = RT.VALIDATION_SEED; cuda = str(device).startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter(); trainer = H.new_trainer(FAMILY, candidate, seed, guard, device, centres_total=2048); A.synchronize(device)
    info["build_seconds"] = time.perf_counter() - t0
    rows += allocation_rows(H, guard, trainer, 2048, ref=ref)
    info["provenance_sidecar"] = write_sidecar(root, trainer, f"T2048_order{candidate['order']}_seed{seed}")
    t0 = time.perf_counter(); fit_info = trainer.step(); A.synchronize(device); info["fit_seconds"] = time.perf_counter() - t0
    n = len(H.fit_windows(guard))
    rows.append(row("capacity2048.kernel_fit_complete", fit_info["scene_count"] == n and fit_info["coefficient_count"] == 2049
                    and bool(torch.isfinite(trainer.ratio.coefficients).all()) and abs(fit_info["mixture_mean_of_g"] - 1.0) <= 1e-6,
                    coefficient_count=fit_info["coefficient_count"], mixture_mean_of_g=fit_info["mixture_mean_of_g"],
                    residual=fit_info["linear_solve_residual"]))
    with CP.evaluation_state(trainer):
        cal = guard.included("CAL")[0]
        session = H.EPFLRegistrationSession(FAMILY, trainer, guard, "CAL", [cal], device, candidate)
        energy, prep = session.energy(cal, 0); info["preparation"] = prep
        p = torch.tensor([.37, .23, -.41], device=device, requires_grad=True)
        grad = torch.autograd.grad(energy(p), p)[0]
        rows.append(row("capacity2048.geometry_gradient_finite_nonzero", bool(torch.isfinite(grad).all()) and float(grad.abs().max()) > 0))
        if smoke:
            with torch.no_grad():
                gv = energy.grid_values(-9.0, torch.tensor(A.Q.grid_points()[1][:16], dtype=torch.float32, device=device))
            rows.append(row("capacity2048.smoke_grid_values_finite", bool(torch.isfinite(gv).all())))
            info["full_search_timing"] = None
        else:
            timing = A.register_timed(energy, device)
            rows.append(row("capacity2048.full_common_search_grid", timing["n_grid"] == 21 * 508 and math.isfinite(timing["F_hat"]),
                            n_grid=timing["n_grid"], seconds=timing["seconds"]))
            info["full_search_timing"] = {k: timing[k] for k in ("seconds", "grid_seconds", "refine_and_final_seconds", "n_grid")}
    rows.append(row("capacity2048.data_roles_read_FIT_CAL_only", {r["role"] for r in guard.reads} <= {"FIT", "CAL"}))
    info["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else None
    del trainer, session, energy
    if cuda:
        torch.cuda.empty_cache()
    return rows, info


def guard_rows(guard, smoke):
    from v10.epfl.geometry import row
    out = []

    def refused(fn):
        try:
            fn(); return False
        except PermissionError:
            return True
    base = getattr(guard, "base", guard)
    out.append(row("guard.dev_refused_at_gate_checks", refused(lambda: base.check(base.included("DEV")[0], {"DEV"}, "negative check"))))
    for w in ("water_0006", "street_0020"):
        out.append(row(f"guard.group_record_removal_{w}_refused", refused(lambda w=w: base.check(w, {"FIT"}, "negative check"))))
    out.append(row("guard.excluded_alignment_pair_refused", refused(lambda: base.check("water_0083", {"FIT", "CAL"}, "negative check"))))
    out.append(row("guard.bf_record_refused", refused(lambda: base.check("1B_A2", {"FIT"}, "negative check"))))
    out.append(row("guard.bf_held_test_core_refused", refused(lambda: base.check("1B_A1", {"FIT", "CAL"}, "negative check"))))
    out.append(row("guard.group_record_applied", len(base.included("FIT")) == 334 and base.group_sha256 is not None, fit=len(base.included("FIT"))))
    if not smoke:
        out.append(row("guard.data_lock_fit_cal_verified", base.verify_data_lock({"FIT", "CAL"})["checked"] == 344))
    return out


def numerics_rows(device):
    from v10.epfl.geometry import row
    env = {**os.environ, "V10_TEST_DEVICE": "cuda" if str(device).startswith("cuda") else "cpu", "PYTHONPATH": str(REPO / "src")}
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/v10/test_rulsif_v2.py"], cwd=REPO, env=env,
                       capture_output=True, text=True)
    tail = (r.stdout + r.stderr)[-3000:]
    return [row("numerics.test_rulsif_v2_on_device", r.returncode == 0, device=env["V10_TEST_DEVICE"], seconds=time.time() - t0, tail=tail)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--parts", default=",".join(PARTS))
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu")); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--xhw", action="store_true", help="cross-hardware run (owner 2026-10-19: finals on RTX6000PRO / L40S / H100): the GPU must be "
                    "one of the three accepted final types instead of L40S; output under rulsif_v2_gpu_gate_xhw/<label>/<job>; the precondition "
                    "for RuLSIF_exact_v2 finals on that GPU type")
    a = ap.parse_args(); parts = [p for p in a.parts.split(",") if p]; dev = a.device
    if a.smoke != (dev == "cpu"):
        raise SystemExit("--smoke runs on --device cpu only, and the CPU device only as a smoke")
    pins = RT.job_pins()
    RT.require(pins.get("group_record_sha256"), "the v2 jobs need the group-record pin V10_EPFL_GROUP_SHA256")
    import torch  # noqa: F401
    from v10 import adapter as A
    from v10.epfl import host as H
    from v10.supplement import checkpoints as CP
    V2, C = H.rulsif_v2_module()
    if a.smoke:
        A.Q.TH_STEP, A.P.GRID_STEP, A.P.N_ADAM = 15.0, 17.0, 2
    env = RT.runtime_environment(A) if not a.smoke else {"smoke": True, **A.environment(None)}
    final_gpus = ("NVIDIA L40S", "NVIDIA RTX PRO 6000 Blackwell Server Edition", "NVIDIA H100 NVL")
    labels = {"NVIDIA L40S": "L40S", "NVIDIA RTX PRO 6000 Blackwell Server Edition": "RTX6000PRO", "NVIDIA H100 NVL": "H100"}
    sub = ("rulsif_v2_gpu_gate_xhw/" + labels.get(env.get("gpu_name"), "smoke")) if a.xhw else "rulsif_v2_gpu_gate"
    root = RT.OUT_ROOT / (("smoke_cpu/" + sub) if a.smoke else sub) / pins["job"]; root.mkdir(parents=True, exist_ok=True)
    base = RT.guard_from_pins(pins)
    RT.require(base.stage == "GATE_CHECKS", "the v2 gate runs at stage GATE_CHECKS (DEV refused)")
    guard = smoke_guard(base) if a.smoke else base
    report = {"schema": "v10.epfl.rulsif_v2_gpu_gate.v1", "family": FAMILY, "pins": pins, "environment": env, "stage": base.stage, "parts": {},
              "cross_hardware": bool(a.xhw),
              "candidates": C.candidates(FAMILY), "amendment_sha256": RT.sha256_file(C.RULSIF_V2_AMENDMENT),
              "preregistration_sha256": RT.sha256_file(C.RULSIF_V2_RECORD), "mapping": H.RULSIF_V2_EPFL_MAPPING,
              "addendum_sha256": RT.sha256_file(REPO / "v10" / "epfl" / "EPFL_RULSIF_V2_ADDENDUM_REV3.json"), "centres_totals": [1024, 2048],
              "purpose": "implementation / execution gate of RuLSIF_exact_v2 on EPFL; no selection, no scientific result",
              "registration_errors_computed": False, "validation_seed": RT.VALIDATION_SEED}
    ref1024 = None
    for part in parts:
        t0 = time.time(); rows, extra = [], {}
        try:
            if part == "env":
                from v10.epfl.geometry import row
                rows = [row("env.accepted_common_environment", True, gpu=env.get("gpu_name"), smoke=a.smoke),
                        (row("env.accepted_final_gpu", a.smoke or env.get("gpu_name") in final_gpus, gpu=env.get("gpu_name"))
                         if a.xhw else row("env.l40s_budget_unit", a.smoke or env.get("gpu_name") == RT.BUDGET_GPU, gpu=env.get("gpu_name")))]
            elif part == "numerics":
                rows = numerics_rows(dev)
            elif part == "family":
                rows, extra["info"] = family_rows(H, A, CP, V2, guard, dev, a.smoke, root)
                ref1024 = extra["info"].get("provenance_summary")
            elif part == "capacity2048":
                rows, extra["info"] = capacity_rows(H, A, V2, guard, dev, a.smoke, root, ref1024)
            elif part == "guard":
                rows = guard_rows(guard, a.smoke)
        except Exception as e:  # noqa: BLE001
            from v10.epfl.geometry import row
            rows.append(row(f"{part}.exception", False, error=repr(e)[:1000], traceback=traceback.format_exc()[-4000:]))
        res = {"rows": rows, "summary": summarize(rows), "seconds": time.time() - t0, **extra}
        RT.write(root / f"{part}.json", res); report["parts"][part] = res["summary"]
        print(f"[r2gate] part {part} {res['summary']} ({res['seconds']:.0f}s)", flush=True)
    report["data_reads_roles"] = sorted({r["role"] for r in base.reads}); report["n_data_reads"] = len(base.reads)
    report["dev_read"] = "DEV" in report["data_reads_roles"]
    ok = report["parts"] and all(v["status"] == "PASS" for v in report["parts"].values()) and not report["dev_read"]
    report["status"] = ("SMOKE_PASS_NOT_A_GATE_RESULT" if ok else "SMOKE_FAIL_NOT_A_GATE_RESULT") if a.smoke else ("PASS" if ok else "FAIL")
    RT.write(root / "GATE.json", report)
    print(f"[r2gate] {report['status']} -> {root / 'GATE.json'}", flush=True)
    return 0 if report["status"] in ("PASS", "SMOKE_PASS_NOT_A_GATE_RESULT") else 1


if __name__ == "__main__":
    sys.exit(main())
