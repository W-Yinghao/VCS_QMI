#!/usr/bin/env python3
"""V10 EPFL item-5 GPU gate (decision 2026-10-17 section 6.2 item 5), Slurm only, stage GATE_CHECKS (FIT / CAL windows; DEV refused).

  python -I scripts/v10_epfl/gpu_gate.py [--parts env,units,warp,support,gradient,checkpoint,families,guard] [--families F1,F2]

Parts (every row PASS / FAIL; PASS = every required row of every part passes):
  env         the accepted common Torch / CUDA / cuDNN / precision environment, actual GPU recorded
  units       pixel units: err_full = 4 x native px, thresholds 10.5 / 2.25 native px, corner patch, search set, manifests (down 1, resize 1.0)
  warp        perturb_map conventions on a real FIT and CAL window (identity, integer shift, rotation direction, truth round trip minimum),
              the EPFL moving path = REG-P3 Inputs3 (linear NIR), the BF log path refused, reference / training inputs = the cache
  support     effective support by sentinel poisoning (V level; critic / MF / MI / NMI / MIND / RuLSIF energy levels), fp64 on CUDA
  gradient    T01's geometry part on an EPFL FIT window: fp64 finite differences vs autograd for every MF term, dense vs moments
  checkpoint  the indivisible MF pair save / load: energy values and geometry gradients bitwise after reload
  families    every family of the 11-method set on EPFL: trainer on the full FIT336, resume identity (1 step, save, step, load, step), frozen
              evaluation state, one CAL-window energy with finite geometry gradient vs finite differences, host permutation identities, one
              full common search (21 x 508 grid + 3 x 100 Adam) timed; registration error is NOT computed
  guard       negative admission checks: DEV at GATE_CHECKS, excluded pairs, BF records and held BF TEST cores, forbidden paths
Outputs: <V10_EPFL_OUT>/gpu_gate/<job>/{GATE.json, <part>.json, families/<family>.json}.  No scientific result is produced.
--smoke --device cpu: a CPU code-path check of the same parts (no full search, no environment gate), written under
<V10_EPFL_OUT>/smoke_cpu/<job>/ with status SMOKE_..._NOT_A_GATE_RESULT; it never counts as the item-5 gate.
"""
from __future__ import annotations

import argparse
import copy
import gc
import math
import sys
import tempfile
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from v10.epfl import runtime as RT  # noqa: E402

PARTS = ("env", "units", "warp", "support", "gradient", "checkpoint", "families", "guard")


def summarize(rows):
    req = [r for r in rows if r.get("required", True)]
    return {"status": "PASS" if req and all(r["status"] == "PASS" for r in req) else "FAIL", "n": len(rows), "n_required": len(req),
            "required_fail": [r["name"] for r in req if r["status"] != "PASS"], "diagnostic_fail": [r["name"] for r in rows if not r.get("required", True) and r["status"] != "PASS"]}


def _cuda(device):
    return str(device).startswith("cuda")


def family_gate(H, A, CP, guard, family, device="cuda", smoke=False):
    """The BF supplement execution gate (scripts/v10_supplement/validate.py gate) on EPFL inputs, for one family."""
    import numpy as np
    import torch
    from v10.epfl.geometry import row
    from v10.supplement.geometry_validation import native_gradient_check
    from v10.supplement.learned import MLPRigidEnergy
    from v10.supplement.training import rng_state
    rows = []; info = {"family": family}
    candidate = H.candidates(family)[0]
    ident = {"schema": "v10.epfl.gate.v1", "family": family, "candidate": candidate, "seed": RT.VALIDATION_SEED, "phase": "EPFL_ITEM5_GATE"}
    t0 = time.time()
    if _cuda(device):
        torch.cuda.reset_peak_memory_stats()
    trainer = H.new_trainer(family, candidate, RT.VALIDATION_SEED, guard, device); info["build_seconds"] = time.time() - t0
    kind = H.schedule(family)["kind"]; gradient = kind in ("gradient_training", "native_gradient_training")
    with tempfile.TemporaryDirectory(dir=str(RT.OUT_ROOT)) as tmp:
        cp = Path(tmp) / "resume.pt"
        if gradient:
            first = trainer.step(); CP.save(trainer, cp, ident); expected = trainer.step()
            model = [{k: v.detach().cpu().clone() for k, v in m.state_dict().items()} for m in trainer.modules()]
            opts = copy.deepcopy([o.state_dict() for o in trainer.optimizers]); local = copy.deepcopy(rng_state(trainer))
            cpu_rng, cuda_rng = torch.get_rng_state(), (torch.cuda.get_rng_state_all() if _cuda(device) else [])
            CP.load(trainer, cp, ident); actual = trainer.step()
            same = all(torch.equal(v.detach().cpu(), s[k]) for m, s in zip(trainer.modules(), model) for k, v in m.state_dict().items())
            close = all(torch.allclose(v.detach().cpu(), s[k], atol=1e-5, rtol=1e-4) for m, s in zip(trainer.modules(), model) for k, v in m.state_dict().items())
            rows.append(row(f"{family}.resume_state_after_reload_step", close, bitwise=same))
            rows.append(row(f"{family}.resume_rng_restored", rng_state(trainer) == local and torch.equal(cpu_rng, torch.get_rng_state())
                            and all(torch.equal(x, y) for x, y in zip(cuda_rng, torch.cuda.get_rng_state_all() if _cuda(device) else []))))
            rows.append(row(f"{family}.resume_step_count", trainer.n_steps == 2))
            fin = all(math.isfinite(r.get(k, 0.0)) for r in (first, expected, actual) for k in ("loss", "grad_norm_preclip"))
            rows.append(row(f"{family}.training_values_finite", fin, first={k: first.get(k) for k in ("loss", "J", "grad_norm_preclip", "m1_iterations")}))
            if family == "InfoNCE_host[J_perm1]":
                rows.append(row(f"{family}.infonce_log_label", "J" not in first and first.get("training_objective") == "InfoNCE_host"))
            info["n_fit_windows"] = len(getattr(trainer, "imgs", getattr(trainer, "images", {})))
            del model, opts
        elif family == "RuLSIF_exact":
            fit_info = trainer.step(); CP.save(trainer, cp, ident); CP.load(trainer, cp, ident)
            rows.append(row(f"{family}.kernel_fit_full_fit336", len(trainer.fit_metadata["fit_windows"]) == 336 and fit_info["scene_count"] == 336,
                            centres=trainer.fit_metadata["centres"], residual=fit_info["linear_solve_residual"]))
        else:
            CP.save(trainer, cp, ident); CP.load(trainer, cp, ident)
            rows.append(row(f"{family}.training_free_no_updates", trainer.n_steps == 0))
        try:
            CP.load(trainer, cp, {**ident, "seed": 999}); wrong = False
        except RuntimeError:
            wrong = True
        rows.append(row(f"{family}.checkpoint_refuses_other_identity", wrong))
    before = copy.deepcopy(rng_state(trainer)); flags = [(m.training, [p.requires_grad for p in m.parameters()]) for m in trainer.modules()]
    h_before = A.state_hash(*trainer.modules()) if trainer.modules() else None
    with CP.evaluation_state(trainer):
        cal = guard.included("CAL")[0]
        session = H.EPFLRegistrationSession(family, trainer, guard, "CAL", [cal], device, candidate)
        energy, prep = session.energy(cal, 0)
        p = torch.tensor([.37, .23, -.41], device=device, requires_grad=True)
        value = energy(p); grad = torch.autograd.grad(value, p)[0]
        rows.append(row(f"{family}.geometry_gradient_finite_nonzero", bool(torch.isfinite(grad).all()) and float(grad.abs().max()) > 0, grad=grad.tolist()))
        if family in ("MI_native", "NMI_native", "CoMIR_native"):
            try:
                chk = native_gradient_check(energy.energy, p, grad); ok = True
            except AssertionError as e:
                chk, ok = {"error": str(e)[:500]}, False
            rows.append(row(f"{family}.geometry_gradient_vs_fp64_local_fd", ok, check=chk))
        else:
            h = .01; eye = torch.eye(3, device=device) * h
            with torch.no_grad():
                fd = torch.stack([(energy(p.detach() + e) - energy(p.detach() - e)) / (2 * h) for e in eye])
            ok = bool(torch.allclose(grad, fd, atol=5e-4, rtol=.05))
            rows.append(row(f"{family}.geometry_gradient_vs_fp32_central_difference", ok, grad=grad.tolist(), fd=fd.tolist(), atol=5e-4, rtol=.05))
        if family in ("MLP_VCS_perm8", "InfoNCE_host[J_perm1]"):
            raw = energy.energy; r1, r2 = np.random.default_rng(8117), np.random.default_rng(8117)
            other = MLPRigidEnergy(session.reference, raw.mov, r1, encoder=trainer.eV, critic=trainer.critic, draws=1)
            native = A.Q.RigidEnergy(torch, torch.nn.functional, "critic", session.reference, raw.mov, r2, eV=trainer.eV, critic=trainer.critic)
            rows.append(row(f"{family}.perm1_energy_equals_host_critic_energy", bool(torch.allclose(other(p), native(p), atol=1e-5, rtol=1e-4))
                            and r1.bit_generator.state == r2.bit_generator.state))
            extra = MLPRigidEnergy(session.reference, raw.mov, np.random.default_rng(8117), encoder=trainer.eV, critic=trainer.critic, draws=8, extra_seed=917)
            rows.append(row(f"{family}.perm8_first_draw_is_host_permutation", bool(torch.equal(other.perm, extra.permutations[0]))))
        if family in H.BF1_METHODS and trainer.mf:
            rows.append(row(f"{family}.mf_energy_frozen_W_estimator", energy.energy.estimator == ("perm1" if family == "MF_VCS_perm1" else "moments")))
        if smoke:
            with torch.no_grad():
                gv = energy.grid_values(-9.0, torch.tensor(A.Q.grid_points()[1][:64], dtype=torch.float32, device=device))
            rows.append(row(f"{family}.smoke_grid_values_finite", bool(torch.isfinite(gv).all()) and len(gv) == 64))
            info.update({"preparation": prep, "full_search_timing": None})
        else:
            timing = A.register_timed(energy, device)
            rows.append(row(f"{family}.full_common_search_grid", timing["n_grid"] == 21 * 508, n_grid=timing["n_grid"], seconds=timing["seconds"]))
            info.update({"preparation": prep, "full_search_timing": timing})
    rows.append(row(f"{family}.evaluation_restores_rng_and_flags", rng_state(trainer) == before
                    and flags == [(m.training, [p.requires_grad for p in m.parameters()]) for m in trainer.modules()]))
    if h_before is not None:
        rows.append(row(f"{family}.evaluation_did_not_change_the_model", A.state_hash(*trainer.modules()) == h_before))
    if gradient:
        trainer.step(); rows.append(row(f"{family}.training_continues_after_evaluation", trainer.n_steps == 3))
    roles = {r["role"] for r in guard.reads}
    rows.append(row(f"{family}.data_roles_read_FIT_CAL_only", roles <= {"FIT", "CAL"}, roles=sorted(roles)))
    info["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30 if _cuda(device) else None
    del trainer, session, energy, value, grad; gc.collect()
    if _cuda(device):
        torch.cuda.empty_cache()
    return rows, info


def guard_rows(guard):
    from v10.epfl.geometry import row
    out = []

    def refused(fn):
        try:
            fn(); return False
        except PermissionError:
            return True
    dev = guard.included("DEV")[0]
    out.append(row("guard.dev_refused_at_gate_checks", refused(lambda: guard.check(dev, {"DEV"}, "negative check"))))
    out.append(row("guard.excluded_alignment_pair_refused", refused(lambda: guard.check("water_0083", {"FIT", "CAL"}, "negative check"))))
    out.append(row("guard.excluded_size_pair_refused", refused(lambda: guard.check("country_0007", {"FIT", "CAL"}, "negative check"))))
    out.append(row("guard.bf_record_refused", refused(lambda: guard.check("1B_A2", {"FIT"}, "negative check"))))
    out.append(row("guard.bf_held_test_core_refused", refused(lambda: guard.check("1B_A1", {"FIT", "CAL"}, "negative check"))))
    out.append(row("guard.role_mismatch_refused", refused(lambda: guard.check(guard.included("CAL")[0], {"FIT"}, "negative check"))))
    out.append(row("guard.unknown_role_refused", refused(lambda: guard.check(guard.included("FIT")[0], {"TEST"}, "negative check"))))
    out.append(row("guard.data_lock_fit_cal_verified", guard.verify_data_lock({"FIT", "CAL"})["checked"] == 346))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--parts", default=",".join(PARTS)); ap.add_argument("--families", default="")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu")); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); parts = [p for p in a.parts.split(",") if p]; dev = a.device
    if a.smoke != (dev == "cpu"):
        raise SystemExit("--smoke runs on --device cpu only, and the CPU device only as a smoke")
    pins = RT.job_pins()
    import torch
    from v10 import adapter as A
    from v10.epfl import host as H, geometry as G
    from v10.supplement import checkpoints as CP
    env = RT.runtime_environment(A) if not a.smoke else {"smoke": True, **A.environment(None)}
    root = RT.OUT_ROOT / ("smoke_cpu" if a.smoke else "gpu_gate") / pins["job"]; root.mkdir(parents=True, exist_ok=True)
    guard = RT.guard_from_pins(pins)
    report = {"schema": "v10.epfl.item5_gate.v1", "decision": "V10-phase1-decisions-20261017 section 6.2 item 5", "pins": pins, "environment": env,
              "stage": guard.stage, "parts": {}, "purpose": "implementation / geometry gate on EPFL; no selection, no scientific result",
              "registration_errors_computed": False}
    fit0, cal0 = guard.included("FIT")[0], guard.included("CAL")[0]
    for part in parts:
        t0 = time.time(); rows = []; extra = {}
        try:
            if part == "env":
                rows = [G.row("env.accepted_common_environment", True, gpu=env.get("gpu_name"), budget_unit_gpu=env.get("budget_unit_gpu"), smoke=a.smoke)]
            elif part == "units":
                rows = G.pixel_unit_rows() + G.manifest_unit_rows(guard) + G.support_constant_rows()
            elif part == "warp":
                rows = G.warp_rows(guard, fit0, "FIT", dev) + G.warp_rows(guard, cal0, "CAL", dev)
            elif part == "support":
                rows = support_part(G, H, A, guard, cal0, dev)
            elif part == "gradient":
                rows = G.gradient_rows(guard, fit0, dev, torch.float64, states=("sharpened_x10",) if a.smoke else ("host_init", "sharpened_x10"))
                if not a.smoke:
                    rows += [dict(r, required=False, name="fp32_diag." + r["name"]) for r in G.gradient_rows(guard, fit0, dev, torch.float32, states=("sharpened_x10",))]
            elif part == "checkpoint":
                with tempfile.TemporaryDirectory(dir=str(RT.OUT_ROOT)) as tmp:
                    rows = G.checkpoint_rows(guard, fit0, dev, tmp)
            elif part == "families":
                fams = [f for f in a.families.split(",") if f] or list(H.FAMILIES)
                extra["families"] = {}
                for fam in fams:
                    try:
                        fr, finfo = family_gate(H, A, CP, guard, fam, dev, a.smoke)
                    except Exception as e:  # noqa: BLE001 - recorded as a failed family, the others continue
                        fr, finfo = [G.row(f"{fam}.exception", False, error=repr(e)[:1000], traceback=traceback.format_exc()[-4000:])], {"family": fam}
                    rows += fr; extra["families"][fam] = finfo
                    RT.write(root / "families" / f"{fam}.json", {"rows": fr, "info": finfo, "summary": summarize(fr)})
                    print(f"[gate] family {fam} {summarize(fr)['status']}", flush=True)
            elif part == "guard":
                rows = guard_rows(guard)
        except Exception as e:  # noqa: BLE001
            rows.append(G.row(f"{part}.exception", False, error=repr(e)[:1000], traceback=traceback.format_exc()[-4000:]))
        res = {"rows": rows, "summary": summarize(rows), "seconds": time.time() - t0, **extra}
        RT.write(root / f"{part}.json", res); report["parts"][part] = res["summary"]
        print(f"[gate] part {part} {res['summary']} ({res['seconds']:.0f}s)", flush=True)
    report["data_reads_roles"] = sorted({r["role"] for r in guard.reads}); report["n_data_reads"] = len(guard.reads)
    report["status"] = "PASS" if report["parts"] and all(v["status"] == "PASS" for v in report["parts"].values()) else "FAIL"
    if a.smoke:
        report["status"] = "SMOKE_" + report["status"] + "_NOT_A_GATE_RESULT"
    RT.write(root / "GATE.json", report)
    print(f"[gate] {report['status']} -> {root / 'GATE.json'}", flush=True)
    return 0 if report["status"] in ("PASS", "SMOKE_PASS_NOT_A_GATE_RESULT") else 1


def support_part(G, H, A, guard, window, device="cuda"):
    """Sentinel poisoning of the moving image outside each energy's read radius, on one CAL window, float32 CUDA (relative tolerance 1e-6:
    cuDNN may choose global (FFT) convolution algorithms; the CPU tests check exact equality)."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    import sys as _s
    _s.path.insert(0, str(A.REPO / "scripts" / "v10"))
    import t01_validate as T   # noqa: E402
    from v10.supplement.mind import DescriptorRigidEnergy, MINDConfig, mind2d
    dev = torch.device(device); dt = torch.float32      # production dtype (Q.RigidEnergy positions are float32)
    pert = H.perturbations_for(guard, [window], "CAL", "V10 EPFL support part")
    inputs = H.EPFLGuardedInputs(guard, {"CAL"}, "V10 EPFL support part", pert, dev)
    ref_l, ref_rgb = inputs.reference(window); mov = inputs.moving(window, 2)[0].to(dt)
    eU, eV, cr, head = T.build_models_seeded(T.SEED, dev, dt); A.freeze_for_test(eU, eV, cr, head)
    DU = A.reference_descriptors(eU, inputs, window, dt)
    rng = lambda: np.random.default_rng(A.Q.seed_of(f"R6-REG-P2|pos|{window}|2"))   # noqa: E731
    rg = lambda M: tuple(float(x) for x in np.percentile(M.cpu().numpy(), [0.5, 99.5]))   # noqa: E731
    ranges = (rg(ref_l), rg(mov))
    mind_ref = mind2d(ref_l.to(dt), MINDConfig())

    def mk(fn):
        return lambda m: fn(mov if m is None else m)
    energies = {"MF_moments": (mk(lambda m: A.MFRigidEnergy(DU, m, rng(), eV=eV, head=head, estimator="moments", dtype=dt)), G.R_VP_CRITIC),
                "MF_perm1": (mk(lambda m: A.MFRigidEnergy(DU, m, rng(), eV=eV, head=head, estimator="perm1", dtype=dt)), G.R_VP_CRITIC),
                "MLP_critic": (mk(lambda m: A.Q.RigidEnergy(torch, F, "critic", DU, m, rng(), eV=eV, critic=cr)), G.R_VP_CRITIC),
                "MI": (mk(lambda m: A.Q.RigidEnergy(torch, F, "mi", ref_l.to(dt), m, rng(), ranges=ranges)), G.R_VP_POINT),
                "NMI": (mk(lambda m: A.Q.RigidEnergy(torch, F, "nmi", ref_l.to(dt), m, rng(), ranges=ranges)), G.R_VP_POINT)}
    rows = G.support_rows(guard, window, "CAL", dev, energies, dt)
    md = lambda m: DescriptorRigidEnergy(mind_ref, mind2d(m, MINDConfig()), rng())   # noqa: E731  (descriptor of the poisoned image)
    base = G._values(md(mov), [(0.0, 0.0, 0.0), (A.Q.UMAX, 36.0, -36.0)], dev, dt)
    r_mind = G.R_VP_POINT + 2 * math.sqrt(2)
    pois = G._values(md(G._poison_outside(mov, r_mind + 1e-6, 1e3)), [(0.0, 0.0, 0.0), (A.Q.UMAX, 36.0, -36.0)], dev, dt)
    tol = 0.0 if dev.type == "cpu" else 1e-6
    rows.append(G.row(f"support.energy_unchanged_outside_r{r_mind:.2f}_MIND2D", G._rel_diff(pois, base) <= tol, max_rel_diff=G._rel_diff(pois, base), values=base[:3]))
    rows.append(G.row("support.comir_encode_once_reads_the_whole_moving_image", True, required=False,
                      note="CoMIR encodes the whole perturbed moving image (global receptive field, incl. reflection-padded border): no local "
                           "support statement; disclosed as in BF"))
    return rows


if __name__ == "__main__":
    sys.exit(main())
