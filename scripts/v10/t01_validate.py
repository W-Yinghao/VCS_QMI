"""V10 T01 - mathematics and real-host adapter validation (package V10_DESIGN 6.1, docs/HOST_ADAPTER_CN.md).

Parts (each writes <out-root>/T01/parts/<part>_<cpu|cuda>.json and a copy in its run directory):
  verify_torch  the package core/verify_torch.py, unchanged, as a subprocess next to an unchanged copy of mf_vcs.py
                (sha256 of both checked against the package MANIFEST_SHA256.json); its own internal limits apply.
  core_tol      the same moment / dense / HVP / permutation identities on random simplex features with the package's design
                tolerances (allclose: fp64 atol 1e-10 rtol 1e-8, fp32 atol 1e-5 rtol 1e-4), reference magnitudes recorded; the
                HVP also against autograd of the DENSE objective; NaN / inf / infeasible-W / non-simplex rejections; n = 1 (V
                allowed, U rejected); constant features (zero feature difference: Delta = 0, J = -tau^2); and a small dense QP:
                the package box solver against an independent fp64 pair-feature oracle (explicit Hessian, SciPy L-BFGS-B +
                active-set polishing, certified by its own FW gap).
  adapter       acceptance on the REAL host: a real BF-SHG FIT batch through the host encoders in two feature states - the host
                init (nearly uniform features, J ~ 1e-6) and sharpened_x10 (the same descriptors, feature maps x 10: spread
                features, so the comparisons have power) - dense vs moments values and A / B / W / encoder gradients,
                unequal-n scene weights, Q modes, M1 strict / training modes, MLP perm1 / perm8 / dense; one-update checks
                (same positive terms with a non-zero W, boundary-W projection, AdamW decay in fp64); the real warp chain in
                both states (batched energy vs the package backend, fixed-W state hash across geometry candidates, checkpoint
                save / restore).  M1 (owner decision 2026-10-17 section 2, relative zero-head rule): reference = ||Delta_b||_1 of
                the batch's own snapshot, threshold 1% of it, warm-start gap separate, stop rule consistent with the true FW gap,
                equality with the package profiled objective at tau_b; the arguments the solver call itself received (tolerance =
                0.01 x the independently recomputed G_b(0), initial W = the warm start, max_iterations = the cap) recorded by an
                independent wrapper, since with the cap binding the solved W does not show the tolerance; on the real host-init
                batch the zero-head start iterates, moves W and trains the encoders, the next batch warm-starts, cap 10 returns the
                approximate W with its true gap, HVP / moment-pass counters (working dtype, fp64 diagnostic and per-batch totals)
                equal independent counts; an exact-zero Delta fixture returns W = 0 with zero_reference_gap without calling the
                solver; the fp64 same-input diagnostic and a large-cap run that reaches the 1% threshold on real features
                (non-gating); rule 'absolute' through the adapter is by default the strict package call (require_converged True,
                bitwise; raises when unconverged) and the Trainer refuses it without absolute_reference_only; the old absolute
                target 1e-4 from W = 0 (degenerate at the host init) and the absolute cap-binding rows stay as reference rows
                (require_converged False, explicit).
  geometry      fp64 central finite differences of every energy term (P: p1, p2; Q: q1, q2 by moments, dense and perm1)
                w.r.t. (u, ty, tx) on the real warp chain at two non-zero rigid transforms whose warped sampling grid lies
                on a rational lattice (Pythagorean rotation) at distance >= 1/(2L) from every bilinear kink; two step sizes;
                zero kink crossings verified for every step.  Plus a non-gating fp32-vs-fp64 gradient diagnostic.
  host_replay   (CPU) the UNCHANGED host trainer G.cmd_train run for 2 steps in a subprocess (V9_REG_ROOT redirected under
                outputs_v10/T01/host_replay) against the V10 Trainer MLP_perm1: encoder / critic states after 2 steps.
  merge         -> v10/T01/TORCH_VALIDATION.json and v10/T01/GEOMETRY_GRADIENT_CHECK.json from whatever parts exist
                (missing parts are NOT_RUN); host_adapter_verified only when every expected part is PASS AND all parts ran
                the code of the current tree (code hash, host hash, backend sha256; clean tree).
Data: BF-SHG FIT cores only, through adapter.DataGuard (V10 manifest include rows); T01 never reads TEST, CAL or DEV.
Statuses: PASS / FAIL / BLOCKED (a precondition of a required check is not met, or a required check could not run) / NOT_RUN;
an exception inside a check group is a FAIL row with its traceback.  Part status: FAIL > BLOCKED > PASS; NOT_RUN = no part file.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import inspect
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from v10 import adapter as A
from v10 import mf_vcs as MF

SEED = 10101                     # V10_PROTOCOL compute.seed (T01 / T02 only; training seeds 1001-1003 and tuning seed 1017 unused)
K_SMOKE = 16                     # V10_PROTOCOL head.default_width_for_interface_smoke
T00_RNG_EVIDENCE = A.OUT_ROOT / "T00" / "host_audit_evidence" / "perm_fixed_points.json"
REPO_T01 = A.REPO / "v10" / "T01"
# Geometry check: rotations with rational sin / cos (Pythagorean triples a, b, L: sin = a / L, cos = b / L) map the integer box
# grid onto (1 / L)-lattices; the translation is shifted by < 1 / (2L) so every warped coordinate sits exactly 1 / (2L) from an
# integer (bilinear kink).  Both transforms are inside the host's allowed set (|theta| <= 30 deg, |t| <= 51).
GEOMETRY_TRANSFORMS = (("T1_rot+16.26deg", (7, 24, 25), +1, (5.3, -3.9)), ("T2_rot-22.62deg", (5, 12, 13), -1, (-17.3, 22.1)))
FD_STEPS = (1e-3, 2.5e-4)        # in u (= theta * rho, px-like) and px; max point displacement 2.26 h << 1 / (2L) >= 0.02
FD_ATOL, FD_RTOL = 1e-10, 1e-5   # pre-registered: |g_fd - g_ad| <= FD_ATOL + FD_RTOL * max_d |g_ad[d]| for both steps
EXPECTED_TORCH_PARTS = (("verify_torch", "cpu"), ("verify_torch", "cuda"), ("core_tol", "cpu"), ("core_tol", "cuda"),
                        ("adapter", "cpu"), ("adapter", "cuda"), ("host_replay", "cpu"))
EXPECTED_GEOMETRY_PARTS = (("geometry", "cpu"), ("geometry", "cuda"))
PROVENANCE_KEYS = ("code_hash_src_scripts_v10", "host_code_hash", "mf_vcs_sha256")
FEATURE_STATES = ("host_init", "sharpened_x10")   # real descriptors; MF feature maps as built, or x 10 (spread simplex features)
POWER_STATE = ("float64", "sharpened_x10")        # where the detectability checks gate (effects >> fp64 rounding bound)
DETECT_NOTE = ("detectability: the effect of the wrong rule must exceed 100 x the equality bound (atol + rtol |J|) of the same "
               "dtype.  Gating only in float64 / sharpened_x10; at the host init (J ~ 1e-6) and in float32 (bound >= 1e-5) the "
               "effect is below the bound by construction, so those rows are diagnostics with effect_over_tol recorded")
BOUNDARY_TOL, BOUNDARY_ITERS = 1e-12, 3000        # boundary-W fixture: the batch's box QP solved far below the protocol target
QP_ORACLE_CERT = 1e-12                            # the oracle's own FW gap must certify F* to this before it is compared
HOST_REPLAY_SEED, HOST_REPLAY_STEPS = 901, 2      # G.cmd_train(uid, 'vcs', 1.0, 0.0, 901, steps=2) vs V10 Trainer MLP_perm1
M1_LARGE_CAP = 5000                               # diagnostic: the relative stop rule from W = 0 on real features with a large cap
SOLVE_SIGNATURE = inspect.signature(MF.solve_box_head)   # the unwrapped backend solver (binds the recorded call arguments)


# ------------------------------------------------------------------------------------------------------------ recorder
class Recorder:
    STATUSES = ("PASS", "FAIL", "BLOCKED", "NOT_RUN")

    def __init__(self, part: str, device: str):
        self.part, self.device, self.rows = part, device, []
        self.t0, self.started = time.time(), A.utc_now()

    def add(self, name, status, required=True, **fields):
        if status not in self.STATUSES:
            raise ValueError(f"status {status} not in {self.STATUSES}")
        row = {"name": name, "status": status, "required": bool(required), **fields}
        self.rows.append(row)
        brief = {k: fields[k] for k in ("max_abs_err", "worst_err_over_bound", "value", "reason", "error") if k in fields}
        print(f"[{self.part}/{self.device}] {status:7s} {'' if required else '(diag) '}{name} {json.dumps(brief, default=str)[:300]}", flush=True)
        return row

    def cmp(self, name, x, y, dtype_name, required=True, **extra):
        r = A.compare(x, y, dtype_name)
        return self.add(name, "PASS" if r["allclose"] else "FAIL", required, **({} if r["allclose"] else {"failure_kind": "tolerance"}), **r, **extra)

    def cmp_many(self, name, xs, ys, dtype_name, required=True, **extra):
        r = A.compare_many(list(xs), list(ys), dtype_name)
        return self.add(name, "PASS" if r["allclose"] else "FAIL", required, **({} if r["allclose"] else {"failure_kind": "tolerance"}), **r, **extra)

    def expect(self, name, cond, required=True, **fields):
        ok = bool(cond)
        return self.add(name, "PASS" if ok else "FAIL", required, **({} if ok else {"failure_kind": "assertion"}), **fields)

    def blocked(self, name, reason, required=True, **fields):
        return self.add(name, "BLOCKED", required, reason=reason, **fields)

    def raises(self, name, fn, exc, required=True):
        try:
            fn()
        except exc as e:  # noqa: BLE001
            return self.add(name, "PASS", required, raised=type(e).__name__, message=str(e)[:300])
        except Exception as e:  # noqa: BLE001
            return self.add(name, "FAIL", required, failure_kind="wrong_exception", raised=type(e).__name__, message=str(e)[:300])
        return self.add(name, "FAIL", required, failure_kind="no_exception", expected=exc.__name__)

    @contextlib.contextmanager
    def group(self, name, required=True):
        try:
            yield
        except Exception as e:  # noqa: BLE001 - a failing group is recorded, the part continues
            self.add(f"{name}:exception", "FAIL", required, failure_kind="exception", error=repr(e)[:1000], traceback=traceback.format_exc()[-6000:])

    def result(self, extra=None) -> dict:
        req = [r for r in self.rows if r["required"]]
        by = {s: [r["name"] for r in req if r["status"] == s] for s in self.STATUSES}
        if by["FAIL"]:
            status = "FAIL"
        elif by["BLOCKED"] or by["NOT_RUN"]:
            status = "BLOCKED"
        elif req:
            status = "PASS"
        else:
            status = "NOT_RUN"
        out = {"part": self.part, "device": self.device, "status": status, "n_checks": len(self.rows), "n_required": len(req),
               **{f"n_{s.lower()}": sum(r["status"] == s for r in self.rows) for s in self.STATUSES},
               "required_failures": by["FAIL"], "required_blocked": by["BLOCKED"], "required_not_run": by["NOT_RUN"],
               "diagnostic_failures": [r["name"] for r in self.rows if not r["required"] and r["status"] == "FAIL"],
               "started_utc": self.started, "ended_utc": A.utc_now(), "seconds": time.time() - self.t0, "checks": self.rows}
        out.update(extra or {})
        return out


def dtype_name(dtype) -> str:
    return str(dtype).split(".")[-1]


def build_models_seeded(seed, device, dtype, K=K_SMOKE):
    """Host encoders + host critic (host init order after torch.manual_seed) + MF head; W ~ U(-0.9, 0.9) from a separate CPU
    generator (W = 0, the package training init, would make every feature / geometry gradient vanish)."""
    torch.manual_seed(seed)
    eU, eV, cr = A.P.build_models(nn)
    head = A.MFHead(A.DESC, A.DESC, K, K, temperature=1.0)
    for m in (eU, eV, cr, head):
        m.to(torch.device(device), dtype)
    gen = torch.Generator(device="cpu").manual_seed(A.Q.seed_of(f"V10|T01|W|{seed}|{K}"))
    with torch.no_grad():
        head.W.copy_(torch.empty(K, K, dtype=torch.float64).uniform_(-0.9, 0.9, generator=gen))
    return eU, eV, cr, head


def per_tensor(names, xs, ys, dn: str) -> list:
    """Per-tensor comparison summary (which tensor carries the worst error)."""
    out = []
    for n, x, y in zip(names, xs, ys):
        c = A.compare(x, y, dn)
        out.append({"tensor": n, "allclose": c["allclose"], "max_abs_err": c["max_abs_err"], "ref_max_abs": c["ref_max_abs"],
                    "worst_err_over_bound": c["worst_err_over_bound"], "numel": c["numel"]})
    return out


def grads(out, inputs, retain=True):
    """Gradients of a scalar w.r.t. inputs; an input the output does not depend on gets an explicit zero tensor."""
    inputs = list(inputs)
    gs = torch.autograd.grad(out, inputs, retain_graph=retain, allow_unused=True)
    return [torch.zeros_like(x) if g is None else g for g, x in zip(gs, inputs)]


def close_rel(x: float, y: float, dn: str) -> bool:
    """|x - y| <= rtol(dtype) |y| (the design rtol, atol = 0): for two code paths over the same inputs that should agree to rounding
    (stricter than the design rule, whose fp32 atol 1e-5 exceeds the host-init gaps ~3e-5)."""
    return abs(x - y) <= A.TOLERANCES[dn][1] * abs(y)


def distinct_rel(x: float, y: float, dn: str) -> bool:
    """x and y differ by more than rtol(dtype) max(|x|, |y|): a check that tells the two apart has power at that dtype's rounding."""
    return abs(x - y) > A.TOLERANCES[dn][1] * max(abs(x), abs(y))


@contextlib.contextmanager
def independent_m1_counters():
    """Counts, independently of the adapter's own counters, what an M1 call really does: QuadraticHead constructions (moment
    snapshots, by dtype), QuadraticHead._hvp calls, objective_terms calls (outer moment passes; both the backend name and the
    adapter's imported name) and solve_box_head calls, with the arguments each solver call actually received (tolerance,
    max_iterations and a clone of the initial W, bound against the backend signature; also when the call comes from the package
    profiled_objective).  The wrappers change no value and are removed after the block.  Log with counts_record()."""
    counts = {"heads": {}, "hvp": 0, "objective_terms": 0, "solve_box_head": 0, "solve_box_head_args": []}
    init0, hvp0, obj0, obj_a0, solve0 = MF.QuadraticHead.__init__, MF.QuadraticHead._hvp, MF.objective_terms, A.objective_terms, MF.solve_box_head

    def init(self, scenes, *args, **kw):
        init0(self, scenes, *args, **kw)
        key = dtype_name(self.delta.dtype)
        counts["heads"][key] = counts["heads"].get(key, 0) + 1

    def hvp(self, direction):
        counts["hvp"] += 1
        return hvp0(self, direction)

    def obj(*args, **kw):
        counts["objective_terms"] += 1
        return obj0(*args, **kw)

    def solve(*args, **kw):
        counts["solve_box_head"] += 1
        bound = SOLVE_SIGNATURE.bind(*args, **kw)
        bound.apply_defaults()
        ini = bound.arguments["initial"]
        counts["solve_box_head_args"].append({"tolerance": bound.arguments["tolerance"], "max_iterations": bound.arguments["max_iterations"],
                                              "initial": None if ini is None else ini.detach().clone()})
        return solve0(*args, **kw)
    MF.QuadraticHead.__init__, MF.QuadraticHead._hvp = init, hvp
    MF.objective_terms, A.objective_terms, MF.solve_box_head = obj, obj, solve
    try:
        yield counts
    finally:
        MF.QuadraticHead.__init__, MF.QuadraticHead._hvp = init0, hvp0
        MF.objective_terms, A.objective_terms, MF.solve_box_head = obj0, obj_a0, solve0


def counts_record(cnt: dict) -> dict:
    """JSON record of independent_m1_counters: the counts, and per solver call its tolerance, cap and a summary of the initial W."""
    out = {k: v for k, v in cnt.items() if k != "solve_box_head_args"}
    out["solve_box_head_args"] = [{"tolerance": a["tolerance"], "max_iterations": a["max_iterations"],
                                   "initial": None if a["initial"] is None else {"is_zero": not bool(a["initial"].any()),
                                                                                 "absmax": float(a["initial"].abs().max()),
                                                                                 "dtype": dtype_name(a["initial"].dtype)}}
                                  for a in cnt["solve_box_head_args"]]
    return out


def m1_solver_call_ok(cnt: dict, info: dict, *, g_ref: float, w_start, cap: int, dn: str) -> tuple[bool, dict]:
    """The M1 solver call itself, recorded independently of the adapter's log (independent_m1_counters): exactly one solve_box_head
    call; its tolerance equals the logged tau_b bitwise and 0.01 x the independently recomputed G_b(0) (g_ref) to rtol; its initial W
    equals the warm start bitwise (W = 0 on a first batch); its max_iterations equals the cap.  With the cap binding (host init) the
    solved W does not depend on the tolerance, so only this records what the solver was given."""
    calls = cnt["solve_box_head_args"]
    a = calls[0] if len(calls) == 1 else None
    tau_ref = A.M1_RELATIVE_TOLERANCE * g_ref
    ok = (a is not None and a["tolerance"] == info["m1_tolerance_effective"] and close_rel(a["tolerance"], tau_ref, dn)
          and a["initial"] is not None and a["initial"].shape == w_start.shape and torch.equal(a["initial"], w_start)
          and a["max_iterations"] == cap == info["m1_max_iterations"])
    return ok, {"solver_calls": len(calls), "solver_tolerance": None if a is None else a["tolerance"],
                "tolerance_from_independent_reference": tau_ref, "solver_max_iterations": None if a is None else a["max_iterations"],
                "solver_initial_equals_warm_start_bitwise": None if a is None or a["initial"] is None else bool(
                    a["initial"].shape == w_start.shape and torch.equal(a["initial"], w_start)),
                "solver_initial_is_zero": None if a is None or a["initial"] is None else not bool(a["initial"].any())}


def m1_stop_consistent(info: dict) -> bool:
    """The stop reason agrees with the logged numbers: gap_tolerance <=> gap_final <= threshold within the cap; iteration_budget_exhausted <=>
    the cap was used and the gap is still above the threshold (approximate head); zero_reference_gap <=> G_b(0) = 0 and W = 0."""
    s, it, cap = info["m1_stop_reason"], info["m1_iterations"], info["m1_max_iterations"]
    if s == "gap_tolerance":
        return info["m1_converged"] and info["m1_gap_final"] <= info["m1_tolerance_effective"] and it <= cap
    if s == "iteration_budget_exhausted":
        return (not info["m1_converged"]) and it == cap and info["m1_gap_final"] > info["m1_tolerance_effective"]
    if s == "zero_reference_gap":
        return info["m1_gap_ref_zero"] == 0.0 and it == 0 and info["m1_W_final_absmax"] == 0.0 and info["m1_solver_status"] is None
    return False


def m1_fields(info: dict) -> dict:
    """The logged M1 fields of a row (the solved W itself is not recorded)."""
    return {k: v for k, v in info.items() if k != "m1_W"}


def exact_zero_delta_scenes(n_sc: int, n_pos: int, K: int, dtype, dev, seed: int) -> list:
    """Scenes whose Delta is exactly zero in floating point: every a row is the one-hot e_0 and the b rows are one-hot of random
    classes.  With n_pos a power of two, A^T B / n and (sum a)(sum b)^T / n^2 are the same exact numbers (integers over powers of
    two), and the weights 1 / n_sc are powers of two, so Delta = 0 exactly (a constructed fixture: real features never give it)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for s in range(n_sc):
        a = torch.zeros(n_pos, K, dtype=dtype)
        a[:, 0] = 1
        b = torch.zeros(n_pos, K, dtype=dtype)
        b[torch.arange(n_pos), torch.randint(0, K, (n_pos,), generator=g)] = 1
        out.append(MF.SceneBatch(a.to(dev), b.to(dev), "v", scene_id=str(s)))
    return out


# ------------------------------------------------------------------------------------------------------------ part: verify_torch
def part_verify_torch(device: str, run_dir: Path) -> dict:
    rec = Recorder("verify_torch", device)
    vt = run_dir / "verify_torch"
    vt.mkdir(parents=True, exist_ok=True)
    extra = {}
    with rec.group("package_verify_torch"):
        manifest = json.loads((A.PACKAGE / "MANIFEST_SHA256.json").read_text())
        src_backend, src_verify = A.REPO / "src" / "v10" / "mf_vcs.py", A.PACKAGE / "core" / "verify_torch.py"
        sb, sv = A.sha256_file(src_backend), A.sha256_file(src_verify)
        rec.expect("backend_copy_sha256_equals_package", sb == manifest["core/mf_vcs.py"], value=sb)
        rec.expect("verify_torch_sha256_equals_package", sv == manifest["core/verify_torch.py"], value=sv)
        shutil.copy2(src_backend, vt / "mf_vcs.py")
        shutil.copy2(src_verify, vt / "verify_torch.py")
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}       # only the run directory's mf_vcs.py is importable
        t0 = time.time()
        proc = subprocess.run([sys.executable, "verify_torch.py", "--device", device], cwd=vt, capture_output=True, text=True,
                              env=env, timeout=7200)
        secs = time.time() - t0
        (vt / "stdout.log").write_text(proc.stdout)
        (vt / "stderr.log").write_text(proc.stderr)
        report = vt / ("MATH_TORCH_AUDIT_" + str(torch.device(device)).replace(":", "_") + ".json")
        data = json.loads(report.read_text()) if report.exists() else None
        ok = proc.returncode == 0 and data is not None and data.get("status") == "passed" and data.get("n_checks", 0) > 0
        failing = [ln for ln in proc.stderr.splitlines() if "AssertionError" in ln][-1:] or None
        rec.add("package_verify_torch", "PASS" if ok else "FAIL", True, **({} if ok else {"failure_kind": "package_script"}),
                exit_code=proc.returncode, n_checks=(data or {}).get("n_checks"), seconds=secs, report=str(report),
                stdout_tail=proc.stdout[-2000:], stderr_tail=proc.stderr[-4000:], failing_check=failing,
                internal_limits="verify_torch.py: fp64 value 2e-12 / gradient 2e-11, fp32 value 2e-5 / gradient 3e-5 (absolute, L44); "
                                "fixed absolute limits on the other checks (L78-161)")
        extra["package_report"] = data
        if data:
            rows = []
            for r in data.get("checks", []):
                dn = "float32" if "float32" in r["name"] else "float64"
                rows.append({"name": r["name"], "error": r["error"], "package_limit": r["limit"], "design_atol": A.TOLERANCES[dn][0],
                             "within_design_atol": r["error"] <= A.TOLERANCES[dn][0]})
            n_out = sum(not r["within_design_atol"] for r in rows)
            rec.add("package_errors_within_design_atol", "PASS" if n_out == 0 else "FAIL", False,
                    note="design rule is atol + rtol*|ref|; the package script records no reference magnitudes, so rows above atol "
                         "are undetermined here (core_tol re-runs these identities with reference magnitudes)",
                    n_rows=len(rows), n_above_design_atol=n_out, rows_above=[r for r in rows if not r["within_design_atol"]][:20])
    return rec.result(extra)


# ------------------------------------------------------------------------------------------------------------ part: core_tol
def pair_feature_quadratic(scenes, weights, ridge: float):
    """Independent fp64 numpy oracle of the fixed-feature quadratic (the package verify_numpy construction, no mf_vcs code):
    explicit pair features p_i = vec(a_i b_i^T), q_ij = vec(a_i b_j^T) (U: i != j; independent: the separate pools);
    F(w) = <delta, w> - w^T H w / 2 with delta = sum_s weight_s (mean p - mean q), H = sum_s weight_s (P^T P / n_P + Q^T Q / n_Q)
    + ridge I, w = W.reshape(-1) (row-major)."""
    def np64(x):
        return x.detach().to("cpu", torch.float64).numpy()
    r, s = scenes[0].a.shape[1], scenes[0].b.shape[1]
    delta, H = np.zeros(r * s), ridge * np.eye(r * s)
    for sc, wt in zip(scenes, weights):
        a, b = np64(sc.a), np64(sc.b)
        qa, qb = (np64(x) for x in sc.q_pools())
        p = np.einsum("ir,is->irs", a, b).reshape(len(a), -1)
        q = np.einsum("ir,js->ijrs", qa, qb)
        if sc.q_mode == "u":
            q = q[~np.eye(len(a), dtype=bool)]
        q = q.reshape(-1, r * s)
        delta += wt * (p.mean(0) - q.mean(0))
        H += wt * (p.T @ p / len(p) + q.T @ q / len(q))
    return delta, H


def box_qp_gap(delta, H, x) -> float:
    g = delta - H @ x
    return float(np.where(g >= 0, g * (1 - x), -g * (1 + x)).sum())


def box_qp_oracle(delta, H) -> dict:
    """Independent solve of max <delta, x> - x^T H x / 2 over [-1, 1]^d (explicit d x d H; numpy / SciPy fp64): L-BFGS-B from 0,
    then active-set polishing rounds (bound-active coordinates fixed, free block by least squares, step truncated at the box,
    accepted only when the FW gap decreases).  Certificate: its FW gap on (delta, H), an upper bound of F* - F(x) by concavity."""
    from scipy.optimize import minimize
    d = len(delta)
    res = minimize(lambda x: (0.5 * x @ H @ x - delta @ x, H @ x - delta), np.zeros(d), jac=True, method="L-BFGS-B",
                   bounds=[(-1.0, 1.0)] * d, options={"ftol": 1e-16, "gtol": 1e-14, "maxiter": 100000, "maxls": 100})
    x = np.clip(res.x, -1.0, 1.0)
    gap0 = gap = box_qp_gap(delta, H, x)
    rounds = 0
    for rounds in range(1, 51):
        if gap <= 1e-15:
            break
        g = delta - H @ x
        up, lo = (x >= 1 - 1e-9) & (g > 0), (x <= -1 + 1e-9) & (g < 0)
        act = up | lo
        xn = np.where(up, 1.0, np.where(lo, -1.0, x))
        free = ~act
        if free.any():
            rhs = delta[free] - H[np.ix_(free, act)] @ xn[act]
            xn[free] = np.linalg.lstsq(H[np.ix_(free, free)], rhs, rcond=None)[0]
        step = xn - x
        with np.errstate(divide="ignore", invalid="ignore"):
            lim = np.where(step > 0, (1 - x) / step, np.where(step < 0, (-1 - x) / step, np.inf))
        xn = np.clip(x + min(1.0, float(lim.min())) * step, -1.0, 1.0)
        gn = box_qp_gap(delta, H, xn)
        if gn >= gap:
            break
        x, gap = xn, gn
    return {"x": x, "F": float(delta @ x - 0.5 * x @ H @ x), "gap": gap, "lbfgsb_gap": gap0, "polish_rounds": rounds,
            "lbfgsb_status": int(res.status), "lbfgsb_message": str(res.message)}


def part_core_tol(device: str, run_dir: Path) -> dict:
    rec = Recorder("core_tol", device)
    dev = torch.device(device)
    torch.manual_seed(20261016)
    qp_rows = []

    def feats(n, k, dtype):
        z = torch.randn(n, k, device=dev, dtype=dtype, requires_grad=True)
        return z, z.softmax(-1)

    def qp_scene(n, r, s, mode, dtype):
        a = torch.randn(n, r, device=dev, dtype=dtype).softmax(-1)
        b = torch.randn(n, s, device=dev, dtype=dtype).softmax(-1)
        if mode == "independent":
            return MF.SceneBatch(a, b, mode, torch.randn(n + 1, r, device=dev, dtype=dtype).softmax(-1),
                                 torch.randn(n + 3, s, device=dev, dtype=dtype).softmax(-1))
        return MF.SceneBatch(a, b, mode)

    for dtype in (torch.float64, torch.float32):
        dn = dtype_name(dtype)
        atol, rtol = A.TOLERANCES[dn]
        for mode in ("v", "u", "independent"):
            for n, r, s in ((6, 3, 2), (17, 7, 6), (128, 16, 16)):
                tag0 = f"{dn}_{mode}_n{n}_r{r}_s{s}"
                with rec.group(tag0):
                    za, a = feats(n, r, dtype)
                    zb, b = feats(n, s, dtype)
                    w = torch.empty(r, s, device=dev, dtype=dtype).uniform_(-0.8, 0.8).requires_grad_()
                    leaves = [za, zb, w]
                    if mode == "independent":
                        zqa, qa = feats(n + 1, r, dtype)
                        zqb, qb = feats(n + 3, s, dtype)
                        scene = MF.SceneBatch(a, b, mode, qa, qb)
                        leaves += [zqa, zqb]
                    else:
                        scene = MF.SceneBatch(a, b, mode)
                    for ridge in (0.0, 0.07):
                        exact = MF.objective_terms([scene], w, weights=[1.0], ridge=ridge)
                        ge = grads(exact.f, leaves)
                        for chunk, ck in ((None, False), (5, False), (5, True)):
                            tag = f"{tag0}_ridge{ridge}_chunk{chunk}_ckpt{ck}"
                            dense = MF.dense_objective_terms([scene], w, weights=[1.0], ridge=ridge, chunk_size=chunk, checkpoint_blocks=ck)
                            rec.cmp_many(f"terms_moments_vs_dense_{tag}", [exact.p1, exact.p2, exact.q1, exact.q2, exact.j, exact.f],
                                         [dense.p1, dense.p2, dense.q1, dense.q2, dense.j, dense.f], dn)
                            rec.cmp_many(f"gradients_moments_vs_dense_{tag}", ge, grads(dense.f, leaves), dn)
                        head = MF.QuadraticHead([scene], weights=[1.0], ridge=ridge)
                        hv, hg = head.value_and_gradient(w)
                        tg = torch.autograd.grad(exact.f, w, create_graph=True, retain_graph=True)[0]
                        direction = torch.randn_like(w)
                        thv = -torch.autograd.grad((tg * direction).sum(), w, retain_graph=True)[0]
                        dref = MF.dense_objective_terms([scene], w, weights=[1.0], ridge=ridge)        # independent second-order path
                        tgd = torch.autograd.grad(dref.f, w, create_graph=True, retain_graph=True)[0]
                        thv_dense = -torch.autograd.grad((tgd * direction).sum(), w, retain_graph=True)[0]
                        hvp = head.hvp(direction)
                        rec.cmp(f"head_value_{tag0}_ridge{ridge}", hv, exact.f, dn)
                        rec.cmp(f"head_value_vs_dense_{tag0}_ridge{ridge}", hv, dref.f, dn)
                        rec.cmp(f"head_gradient_{tag0}_ridge{ridge}", hg, tg, dn)
                        rec.cmp(f"head_gradient_vs_dense_autograd_{tag0}_ridge{ridge}", hg, tgd, dn)
                        rec.cmp(f"head_HVP_{tag0}_ridge{ridge}", hvp, thv, dn)
                        rec.cmp(f"head_HVP_vs_dense_autograd_HVP_{tag0}_ridge{ridge}", hvp, thv_dense, dn)
        with rec.group(f"{dn}_all_permutations"):
            za, a = feats(5, 3, dtype)
            zb, b = feats(5, 2, dtype)
            w = torch.empty(3, 2, device=dev, dtype=dtype).uniform_(-0.7, 0.7).requires_grad_()
            scene = MF.SceneBatch(a, b, "v")
            perms = torch.tensor(list(itertools.permutations(range(5))), device=dev, dtype=torch.long)
            sampled = MF.sampled_permutation_terms([scene], w, weights=[1.0], permutations=[perms])
            full = MF.objective_terms([scene], w, weights=[1.0])
            rec.cmp(f"all_120_ordinary_permutations_mean_equals_V_value_{dn}", sampled.f, full.f, dn)
            rec.cmp_many(f"all_120_ordinary_permutations_mean_equals_V_gradients_{dn}", grads(sampled.f, (za, zb, w)), grads(full.f, (za, zb, w)), dn)
        with rec.group(f"{dn}_input_rejections"):
            a, b = feats(9, 3, dtype)[1].detach(), feats(9, 2, dtype)[1].detach()
            sc = MF.SceneBatch(a, b, "v")
            ids0 = torch.zeros(9, dtype=torch.long, device=dev)
            for label, bad in (("nan", float("nan")), ("posinf", float("inf")), ("neginf", float("-inf"))):
                a_bad, b_bad = a.clone(), b.clone()
                a_bad[2, 1] = bad
                b_bad[4, 0] = bad
                w_bad = torch.zeros(3, 2, device=dev, dtype=dtype)
                w_bad[1, 1] = bad
                rec.raises(f"SceneBatch_rejects_{label}_in_a_{dn}", lambda: MF.SceneBatch(a_bad, b, "v"), ValueError)
                rec.raises(f"SceneBatch_rejects_{label}_in_b_U_mode_{dn}", lambda: MF.SceneBatch(a, b_bad, "u"), ValueError)
                rec.raises(f"SceneBatch_rejects_{label}_in_independent_pool_{dn}", lambda: MF.SceneBatch(a, b, "independent", a_bad, b), ValueError)
                rec.raises(f"partition_scenes_rejects_{label}_{dn}", lambda: MF.partition_scenes(a_bad, b, ids0), ValueError)
                rec.raises(f"objective_rejects_{label}_in_W_{dn}", lambda: MF.objective_terms([sc], w_bad, weights=[1.0]), ValueError)
                rec.raises(f"dense_objective_rejects_{label}_in_W_{dn}", lambda: MF.dense_objective_terms([sc], w_bad, weights=[1.0]), ValueError)
                rec.raises(f"head_HVP_rejects_{label}_direction_{dn}", lambda: MF.QuadraticHead([sc], weights=[1.0]).hvp(w_bad), ValueError)
            w_out = torch.full((3, 2), 1.01, device=dev, dtype=dtype)
            rec.raises(f"objective_rejects_infeasible_W_{dn}", lambda: MF.objective_terms([sc], w_out, weights=[1.0]), ValueError)
            rec.raises(f"solver_rejects_infeasible_initial_W_{dn}", lambda: MF.solve_box_head(MF.QuadraticHead([sc], weights=[1.0]), initial=w_out), ValueError)
            rec.raises(f"SceneBatch_rejects_negative_entries_{dn}", lambda: MF.SceneBatch(-a, b, "v"), ValueError)
            rec.raises(f"SceneBatch_rejects_unnormalised_rows_{dn}", lambda: MF.SceneBatch(2 * a, b, "v"), ValueError)
            rec.raises(f"U_mode_n1_rejected_{dn}", lambda: MF.SceneBatch(a[:1], b[:1], "u"), ValueError)
            w = torch.empty(3, 2, device=dev, dtype=dtype).uniform_(-0.8, 0.8)
            t1 = ((a[:1] @ w) * b[:1]).sum()
            rec.cmp(f"V_mode_n1_allowed_J_equals_minus_t2_{dn}", MF.objective_terms([MF.SceneBatch(a[:1], b[:1], "v")], w, weights=[1.0]).j, -t1 ** 2, dn)
        with rec.group(f"{dn}_constant_features"):
            # zero feature difference: every row of a scene is the same simplex vector -> t_i = tau for all pairs, Delta = 0,
            # J = -tau^2 (V, U, independent pools of the same vectors), maximum J* = 0 at any W with tau = 0
            n, r, s = 17, 7, 6
            ca = torch.randn(r, device=dev, dtype=dtype).softmax(-1)
            cb = torch.randn(s, device=dev, dtype=dtype).softmax(-1)
            w = torch.empty(r, s, device=dev, dtype=dtype).uniform_(-0.8, 0.8)
            tau = ca @ w @ cb
            tol_c = 1e-12 if dtype == torch.float64 else 1e-6
            for mode in ("v", "u", "independent"):
                a, b = ca.expand(n, r).clone(), cb.expand(n, s).clone()
                scene = (MF.SceneBatch(a, b, mode, ca.expand(n + 2, r).clone(), cb.expand(n + 5, s).clone()) if mode == "independent"
                         else MF.SceneBatch(a, b, mode))
                m, d = MF.objective_terms([scene], w, weights=[1.0]), MF.dense_objective_terms([scene], w, weights=[1.0])
                rec.cmp_many(f"constant_features_moments_vs_dense_{mode}_{dn}", [m.p1, m.p2, m.q1, m.q2, m.j], [d.p1, d.p2, d.q1, d.q2, d.j], dn)
                rec.cmp(f"constant_features_J_equals_minus_tau2_{mode}_{dn}", m.j, -tau ** 2, dn, tau=float(tau))
                head = MF.QuadraticHead([scene], weights=[1.0])
                rec.cmp(f"constant_features_Delta_zero_{mode}_{dn}", head.delta, torch.zeros_like(head.delta), dn,
                        delta_abs_max=float(head.delta.abs().max()))
                L = head.lipschitz_upper_bound
                rec.expect(f"constant_features_lipschitz_bound_finite_positive_{mode}_{dn}", math.isfinite(L) and L > 0, value=L)
                s0 = MF.solve_box_head(head, tolerance=tol_c, max_iterations=20000)
                rec.expect(f"constant_features_solver_from_W0_stops_at_W0_{mode}_{dn}", s0.converged and math.isfinite(s0.fw_gap)
                           and s0.iterations == 0 and float(s0.w.abs().max()) == 0.0, gap=s0.fw_gap, iterations=s0.iterations, f=s0.f)
                s1 = MF.solve_box_head(head, initial=w, tolerance=tol_c, max_iterations=20000)
                rec.expect(f"constant_features_solver_from_W_reaches_J_star_0_{mode}_{dn}", s1.converged and math.isfinite(s1.fw_gap)
                           and -tol_c - atol <= s1.f <= atol, gap=s1.fw_gap, iterations=s1.iterations, f=s1.f, f_start=float(-tau ** 2))
        with rec.group(f"{dn}_small_dense_QP"):
            tol_s, iters_s = (1e-11, 50000) if dtype == torch.float64 else (1e-6, 20000)
            cases = [(f"{mode}_n{n}_r{r}_s{s}", [qp_scene(n, r, s, mode, dtype)], [1.0])
                     for n, r, s in ((6, 3, 2), (17, 7, 6)) for mode in ("v", "u", "independent")]
            cases.append(("weighted_mixed_3_scenes", [qp_scene(4, 3, 2, "v", dtype), qp_scene(7, 3, 2, "u", dtype),
                                                      qp_scene(3, 3, 2, "independent", dtype)], [0.2, 0.3, 0.5]))
            eye = torch.eye(2, device=dev, dtype=dtype)
            cases.append(("active_box_identity_u", [MF.SceneBatch(eye, eye, "u")], [1.0]))
            for name, scenes, weights in cases:
                for ridge in (0.0, 0.07):
                    tag = f"{name}_ridge{ridge}_{dn}"
                    with rec.group(f"QP_{tag}"):
                        head = MF.QuadraticHead(scenes, weights=weights, ridge=ridge)
                        sol = MF.solve_box_head(head, tolerance=tol_s, max_iterations=iters_s)
                        delta, H = pair_feature_quadratic(scenes, weights, ridge)
                        orc = box_qp_oracle(delta, H)
                        xs = sol.w.detach().to("cpu", torch.float64).numpy().reshape(-1)
                        f_at_s, gap_at_s = float(delta @ xs - 0.5 * xs @ H @ xs), box_qp_gap(delta, H, xs)
                        row = {"case": tag, "solver_iterations": sol.iterations, "solver_gap": sol.fw_gap, "solver_F": sol.f,
                               "oracle_F": orc["F"], "oracle_gap": orc["gap"], "oracle_lbfgsb_gap": orc["lbfgsb_gap"],
                               "oracle_polish_rounds": orc["polish_rounds"], "oracle_lbfgsb_message": orc["lbfgsb_message"],
                               "n_active_solver": int((sol.w.abs() == 1).sum()), "n_active_oracle": int((np.abs(orc["x"]) >= 1 - 1e-12).sum()),
                               "dim": len(delta)}
                        qp_rows.append(row)
                        rec.expect(f"QP_package_solver_converged_{tag}", sol.converged and sol.fw_gap <= tol_s, gap=sol.fw_gap,
                                   iterations=sol.iterations, tolerance=tol_s)
                        rec.cmp(f"QP_package_F_at_its_W_vs_pair_feature_oracle_{tag}", torch.tensor(sol.f, dtype=torch.float64),
                                torch.tensor(f_at_s, dtype=torch.float64), dn)
                        rec.cmp(f"QP_package_gap_at_its_W_vs_pair_feature_oracle_{tag}", torch.tensor(sol.fw_gap, dtype=torch.float64),
                                torch.tensor(gap_at_s, dtype=torch.float64), dn)
                        if orc["gap"] <= QP_ORACLE_CERT:
                            rec.cmp(f"QP_optimum_package_vs_independent_oracle_{tag}", torch.tensor(sol.f, dtype=torch.float64),
                                    torch.tensor(orc["F"], dtype=torch.float64), dn, oracle_gap=orc["gap"], solver_gap=sol.fw_gap,
                                    n_active_solver=row["n_active_solver"], n_active_oracle=row["n_active_oracle"])
                        else:
                            rec.blocked(f"QP_optimum_package_vs_independent_oracle_{tag}",
                                        f"oracle not certified: its FW gap {orc['gap']:.3g} > {QP_ORACLE_CERT:g}", oracle=row)
    return rec.result({"tolerances": A.TOLERANCES, "qp_oracle_rows": qp_rows})


# ------------------------------------------------------------------------------------------------------------ geometry helpers
def box1_points() -> torch.Tensor:
    return A.Q.grid_coords(torch, *A.Q.BOX1, torch.device("cpu")).to(torch.float64)


def warped_box(p) -> np.ndarray:
    """Q.apply_T on the exact warp box (BOX1 integer grid) at p = (u, ty, tx) in fp64 -> [M, 2] moving-image coordinates."""
    th = torch.tensor([float(p[0]) / A.Q.RHO], dtype=torch.float64)
    t = torch.tensor([[float(p[1]), float(p[2])]], dtype=torch.float64)
    return A.Q.apply_T(torch, th, t, box1_points())[0].numpy()


def kink_margin(p) -> float:
    pts = warped_box(p)
    return float(np.min(np.abs(pts - np.round(pts))))


def kink_crossings(p, direction: int, h: float) -> int:
    e = np.zeros(3)
    e[direction] = 1.0
    lo, mid, hi = warped_box(p - h * e), warped_box(p), warped_box(p + h * e)
    return int(np.sum(np.any((np.floor(lo) != np.floor(mid)) | (np.floor(hi) != np.floor(mid)), axis=1)))


def lattice_transform(triple, sign, t_target):
    a, b, L = triple
    th = sign * math.atan2(a, b)
    u = th * A.Q.RHO
    pts0 = warped_box(np.array([u, 0.0, 0.0]))
    t = []
    for k in (0, 1):
        r0 = float(np.mod(pts0[0, k], 1.0 / L))
        d = (0.5 / L - r0 - t_target[k]) % (1.0 / L)
        if d > 0.5 / L:
            d -= 1.0 / L
        t.append(t_target[k] + d)
    return np.array([u, t[0], t[1]], dtype=np.float64), th, L


def geometry_specs():
    out = []
    for name, triple, sign, target in GEOMETRY_TRANSFORMS:
        p, th, L = lattice_transform(triple, sign, target)
        out.append({"name": name, "p": p, "theta_deg": math.degrees(th), "L": L, "margin": kink_margin(p), "expected_margin": 0.5 / L})
    return out


def term_values(e: A.MFRigidEnergy, p) -> dict:
    """Every energy term at p through the host chain (e(p) -> Q.RigidEnergy.batch -> warp -> eV -> sample -> j_critic):
    the adapter's batched moments energy, and the package backend's moments / dense / perm1 terms on the same captured B."""
    j_rows = e(p)
    B = e.last["B"][0]
    sc = MF.SceneBatch(e.A, B, "v")
    m = MF.objective_terms([sc], e.W, weights=[1.0])
    d = MF.dense_objective_terms([sc], e.W, weights=[1.0], chunk_size=1024)
    pm = MF.sampled_permutation_terms([sc], e.W, weights=[1.0], permutations=[e.perm[None]])
    return {"J_adapter_batched_moments": j_rows, "p1": m.p1, "p2": m.p2, "q1_moments": m.q1, "q2_moments": m.q2, "J_moments": m.j,
            "q1_dense": d.q1, "q2_dense": d.q2, "J_dense": d.j, "q1_perm1": pm.q1, "q2_perm1": pm.q2, "J_perm1": pm.j}


# ------------------------------------------------------------------------------------------------------------ part: adapter
def sharpened(head):
    """Spread-feature stress state on the SAME real descriptors: a copy of the MF head with feature-map weights and biases x 10
    (far-from-uniform simplex features; W unchanged).  The real host init gives nearly uniform features (J ~ 1e-6), which leaves
    several comparisons without power; this state gives them power without changing any tolerance."""
    h = copy.deepcopy(head)
    with torch.no_grad():
        for lin in (h.left, h.right):
            lin.weight.mul_(10.0)
            lin.bias.mul_(10.0)
    return h


def part_adapter(device: str, run_dir: Path, n_sc_by_dtype: dict) -> dict:
    rec = Recorder("adapter", device)
    dev = torch.device(device)
    guard = A.DataGuard()
    fit = []
    states_info = {}
    with rec.group("host_and_data"):
        rec.expect("host_code_hash_pinned", A.host_code_hash() == A.HOST_CODE_HASH, value=A.host_code_hash())
        rec.expect("v10_manifest_sha256_pinned", guard.manifest_sha256 == A.V10_MANIFEST_SHA256, value=guard.manifest_sha256)
        fit = A.fit_cores(guard, 1.0, SEED)
        rec.expect("fit_cores_v10_include_equals_r6_fit_32", len(fit) == 32, value=len(fit))
        rec.expect("v9_writer_tripwires_installed", all(getattr(getattr(mod, n), "v10_tripwire", False) for mod, _, names in A._V9_WRITERS for n in names))
        held = guard.held()
        rec.expect("held_records_are_the_134_test_cores", len(held) == 134 and all(guard.rows[c]["original_split"] == "test" for c in held), value=len(held))
        n_reads = len(guard.reads)
        rec.raises("guard_refuses_held_test_core", lambda: guard.check(held[0], {"FIT", "CAL", "DEV"}, "probe"), PermissionError)
        rec.raises("guard_refuses_dev_core_when_fit_only", lambda: guard.check(guard.included("DEV")[0], {"FIT"}, "probe"), PermissionError)
        rec.raises("guard_refuses_unknown_record", lambda: guard.check("XX_not_a_core", {"FIT"}, "probe"), PermissionError)
        rec.expect("guard_refusals_not_logged_as_reads", len(guard.reads) == n_reads)
    with rec.group("host_rng_replay"):
        stats = A.replay_permutation_fixed_points(fit, 901, 200)
        if T00_RNG_EVIDENCE.exists():
            ref = json.loads(T00_RNG_EVIDENCE.read_text())["training_seed901_frac1_eps0_first200steps"]
            keys = ("n_sc", "n_perms", "mean_fixed_points", "share_with_>=1_fixed_point", "distinct_cores_per_step")
            rec.expect("host_batch_rng_fixed_point_stats_equal_T00_replay", all(stats[k] == ref[k] for k in keys), value=stats, reference=ref,
                       note="T00 replayed G.cmd_train's numpy draws in a separate numpy-only script (a transcription of the same host "
                            "lines; compared on fixed-point statistics, not the drawn sequence); the host trainer itself is run and "
                            "compared with the V10 trainer in part host_replay (CPU)")
        else:
            rec.add("host_batch_rng_fixed_point_stats_equal_T00_replay", "NOT_RUN", False, reason=f"{T00_RNG_EVIDENCE} missing", value=stats)

    for dn in ("float64", "float32"):
        dtype, n_sc = getattr(torch, dn), int(n_sc_by_dtype[dn])
        base = None
        with rec.group(f"real_batch_{dn}"):
            hb = A.HostBatchSampler(fit, SEED, n_sc=n_sc).draw()
            imgs = A.load_learned_inputs(guard, hb.cores, {"FIT"}, f"T01 adapter real FIT batch ({dn})", dev)
            eU, eV, cr, head = build_models_seeded(SEED, dev, dtype)
            bu, bv = A.assemble_crops(imgs, hb)
            du, dv = A.batch_descriptors(eU, eV, bu.to(dtype), bv.to(dtype), hb.yx)
            del imgs, bu, bv
            base = {"hb": hb, "eU": eU, "eV": eV, "cr": cr, "head0": head, "du": du, "dv": dv, "n_pos": hb.perm.shape[1], "n_sc": n_sc,
                    "wts": A.host_weights(n_sc)}
            rec.expect(f"real_descriptors_finite_{dn}", bool(torch.isfinite(du).all() and torch.isfinite(dv).all()),
                       du_std=float(du.std()), dv_std=float(dv.std()), n_sc=n_sc, n_pos=base["n_pos"], cores=hb.cores)
        if base is None:
            rec.add(f"real_batch_checks_{dn}", "NOT_RUN", True, reason="the real batch could not be built (see the exception row)")
        else:
            n_pos = base["n_pos"]
            for state in FEATURE_STATES:
                ctx = None
                with rec.group(f"real_features_{state}_{dn}"):
                    head_s = base["head0"] if state == "host_init" else sharpened(base["head0"])
                    Af, Bf = head_s.features(base["du"].reshape(-1, A.DESC), base["dv"].reshape(-1, A.DESC))
                    scenes = A.mf_scenes(Af, Bf, n_sc, n_pos)
                    ctx = {**base, "head": head_s, "Af": Af, "Bf": Bf, "scenes": scenes, "state": state, "power": (dn, state) == POWER_STATE}
                    spread = {"a_std_max": float(Af.std(0).max()), "b_std_max": float(Bf.std(0).max()),
                              "a_range": [float(Af.min()), float(Af.max())], "b_range": [float(Bf.min()), float(Bf.max())],
                              "J_moments": float(MF.objective_terms(scenes, head_s.W, weights=base["wts"]).j)}
                    states_info[f"{state}_{dn}"] = spread
                    rec.expect(f"real_features_nonconstant_{state}_{dn}", spread["a_std_max"] > 0 and spread["b_std_max"] > 0
                               and len(torch.unique(Af.detach(), dim=0)) > 1 and len(torch.unique(Bf.detach(), dim=0)) > 1, **spread)
                    rows_ok = all(torch.equal(sc.a, Af[i * n_pos:(i + 1) * n_pos]) and torch.equal(sc.b, Bf[i * n_pos:(i + 1) * n_pos])
                                  for i, sc in enumerate(scenes))
                    rec.expect(f"scene_s_holds_the_rows_of_crop_s_{state}_{dn}", rows_ok and [s_.scene_id for s_ in scenes] == [str(i) for i in range(n_sc)],
                               cores=base["hb"].cores)
                if ctx is None:
                    rec.add(f"real_feature_checks_{state}_{dn}", "NOT_RUN", True, reason="features could not be built (see the exception row)")
                    continue
                for name, fn in (("moments_vs_dense", real_moments_vs_dense), ("unequal_n_and_partition", real_unequal_n),
                                 ("q_modes", real_q_modes), ("m1", real_m1)):
                    with rec.group(f"real_{name}_{state}_{dn}"):
                        fn(rec, ctx, dev, dn)
                del ctx
            with rec.group(f"real_mlp_{dn}"):
                real_mlp(rec, base, dev, dn)
            del base
        with rec.group(f"m1_zero_reference_{dn}"):
            m1_zero_reference_checks(rec, dev, dn)
        for state in FEATURE_STATES:
            with rec.group(f"registration_side_{state}_{dn}"):
                reg_side_checks(rec, guard, fit, dev, dtype, dn, run_dir, state)

    with rec.group("one_update_float32"):
        one_update_checks(rec, guard, fit, dev, int(n_sc_by_dtype["float32"]))
    with rec.group("optimizer_weight_decay_float64"):
        weight_decay_checks(rec, fit, dev)
    return rec.result({"data_reads": guard.reads, "seed": SEED, "K": K_SMOKE, "n_sc_by_dtype": n_sc_by_dtype, "feature_states": states_info,
                       "geometry_transforms": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items()} for s in geometry_specs()]})


def real_moments_vs_dense(rec: Recorder, c: dict, dev, dn: str):
    Af, Bf, W, scenes, wts, sfx = c["Af"], c["Bf"], c["head"].W, c["scenes"], c["wts"], f"{c['state']}_{dn}"
    m = MF.objective_terms(scenes, W, weights=wts)
    d = MF.dense_objective_terms(scenes, W, weights=wts)
    rec.cmp_many(f"real_moments_vs_dense_terms_{sfx}", [m.p1, m.p2, m.q1, m.q2, m.j], [d.p1, d.p2, d.q1, d.q2, d.j], dn,
                 terms="p1,p2,q1,q2,J", J_moments=float(m.j), J_dense=float(d.j))
    leaves = [Af, Bf, W]
    gm, gd = grads(m.j, leaves), grads(d.j, leaves)
    for lbl, x, y in zip(("A", "B", "W"), gm, gd):
        rec.cmp(f"real_moments_vs_dense_grad_{lbl}_{sfx}", x, y, dn)
    enc = [p for mod in (c["eU"], c["eV"], c["head"].left, c["head"].right) for p in mod.parameters()]
    rec.cmp_many(f"real_moments_vs_dense_grad_encoders_featuremaps_{sfx}", grads(m.j, enc), grads(d.j, enc), dn,
                 required=(dn == "float64"),
                 note="end-to-end; gating in fp64; fp32 is a diagnostic (fp32 GPU convolutions run in TF32 by default)")
    dc = MF.dense_objective_terms(scenes, W, weights=wts, chunk_size=32, checkpoint_blocks=True)
    rec.cmp(f"real_moments_vs_dense_chunked_checkpointed_value_{sfx}", m.j, dc.j, dn)
    rec.cmp_many(f"real_moments_vs_dense_chunked_checkpointed_grads_{sfx}", gm, grads(dc.j, leaves), dn)
    s5 = MF.SceneBatch(Af[:5], Bf[:5], "v")                          # all ordinary permutations of a real 5-row subset average to V
    perms = torch.tensor(list(itertools.permutations(range(5))), device=dev, dtype=torch.long)
    samp = MF.sampled_permutation_terms([s5], W, weights=[1.0], permutations=[perms])
    full = MF.objective_terms([s5], W, weights=[1.0])
    rec.cmp(f"real_all_120_permutations_mean_equals_V_value_{sfx}", samp.j, full.j, dn)
    rec.cmp_many(f"real_all_120_permutations_mean_equals_V_grads_{sfx}", grads(samp.j, leaves), grads(full.j, leaves), dn)


def real_unequal_n(rec: Recorder, c: dict, dev, dn: str):
    Af, Bf, W, n_pos, sfx, power = c["Af"], c["Bf"], c["head"].W, c["n_pos"], f"{c['state']}_{dn}", c["power"]
    leaves = [Af, Bf, W]
    sizes, locked = [n_pos, 57, 9, 1], [0.4, 0.3, 0.2, 0.1]
    rows = [slice(s * n_pos, s * n_pos + sizes[s]) for s in range(4)]
    fx = [MF.SceneBatch(Af[rows[s]], Bf[rows[s]], "v", scene_id=str(s)) for s in range(4)]
    mu, dU = MF.objective_terms(fx, W, weights=locked), MF.dense_objective_terms(fx, W, weights=locked)
    rec.cmp_many(f"unequal_n_moments_vs_dense_terms_{sfx}", [mu.p1, mu.p2, mu.q1, mu.q2, mu.j], [dU.p1, dU.p2, dU.q1, dU.q2, dU.j], dn,
                 sizes=sizes, weights=locked)
    rec.cmp_many(f"unequal_n_moments_vs_dense_grads_ABW_{sfx}", grads(mu.j, leaves), grads(dU.j, leaves), dn)
    explicit = sum(wt * MF.objective_terms([fx[s]], W, weights=[1.0]).j for s, wt in enumerate(locked))
    rec.cmp(f"unequal_n_locked_weights_equal_explicit_scene_sum_{sfx}", mu.j, explicit, dn)
    atol, rtol = A.TOLERANCES[dn]
    tol = atol + rtol * abs(float(mu.j))
    jpos = MF.objective_terms(fx, W, weights=[n_ / sum(sizes) for n_ in sizes]).j
    eff = abs(float(jpos - mu.j))
    rec.expect(f"unequal_n_position_weighting_detectable_{sfx}", eff > 100 * tol, required=power, value=float(jpos - mu.j), tol=tol,
               effect_over_tol=eff / tol, note=DETECT_NOTE)
    pooled = MF.SceneBatch(torch.cat([Af[r] for r in rows]), torch.cat([Bf[r] for r in rows]), "v")
    jpool = MF.objective_terms([pooled], W, weights=[1.0]).j
    eff = abs(float(jpool - mu.j))
    rec.expect(f"unequal_n_cross_scene_pooling_detectable_{sfx}", eff > 100 * tol, required=power, value=float(jpool - mu.j), tol=tol,
               effect_over_tol=eff / tol, note=DETECT_NOTE)
    t1 = (Af[rows[3]] @ W * Bf[rows[3]]).sum()
    rec.cmp(f"V_scene_n1_allowed_J_equals_minus_t2_{sfx}", MF.objective_terms([fx[3]], W, weights=[1.0]).j, -t1 ** 2, dn)
    ids = torch.cat([torch.full((sizes[s],), v, device=dev, dtype=torch.long) for s, v in enumerate((5, 2, 9, 7))])
    Ar, Br = torch.cat([Af[r] for r in rows]), torch.cat([Bf[r] for r in rows])
    shuffle = torch.randperm(len(ids), generator=torch.Generator(device="cpu").manual_seed(SEED)).to(dev)
    part = MF.partition_scenes(Ar[shuffle], Br[shuffle], ids[shuffle], q_mode="v")
    rec.expect(f"partition_scenes_sorted_id_order_{sfx}", [s.scene_id for s in part] == ["2", "5", "7", "9"]
               and [len(s.a) for s in part] == [57, n_pos, 1, 9], value=[(s.scene_id, len(s.a)) for s in part])
    rec.cmp(f"partition_scenes_weights_follow_sorted_order_{sfx}", MF.objective_terms(part, W, weights=[0.3, 0.4, 0.1, 0.2]).j, mu.j, dn)


def real_q_modes(rec: Recorder, c: dict, dev, dn: str):
    Af, Bf, W, n_pos, sfx, power = c["Af"], c["Bf"], c["head"].W, c["n_pos"], f"{c['state']}_{dn}", c["power"]
    leaves = [Af, Bf, W]
    atol, rtol = A.TOLERANCES[dn]
    rec.raises(f"U_mode_n1_rejected_{sfx}", lambda: MF.SceneBatch(Af[:1], Bf[:1], "u"), ValueError)
    su, sv = MF.SceneBatch(Af[:n_pos], Bf[:n_pos], "u"), MF.SceneBatch(Af[:n_pos], Bf[:n_pos], "v")
    um, ud = MF.objective_terms([su], W, weights=[1.0]), MF.dense_objective_terms([su], W, weights=[1.0])
    rec.cmp_many(f"U_mode_moments_vs_dense_terms_{sfx}", [um.q1, um.q2, um.j], [ud.q1, ud.q2, ud.j], dn)
    rec.cmp_many(f"U_mode_moments_vs_dense_grads_ABW_{sfx}", grads(um.j, leaves), grads(ud.j, leaves), dn)
    vm = MF.objective_terms([sv], W, weights=[1.0])
    tol = atol + rtol * abs(float(vm.j))
    eff = abs(float(um.j - vm.j))
    rec.expect(f"U_differs_from_V_detectable_{sfx}", eff > 100 * tol, required=power, value=float(um.j - vm.j), tol=tol,
               effect_over_tol=eff / tol, note=DETECT_NOTE)
    si = MF.SceneBatch(Af[:64], Bf[:64], "independent", Af[64:n_pos], Bf[64:n_pos])
    im, idn = MF.objective_terms([si], W, weights=[1.0]), MF.dense_objective_terms([si], W, weights=[1.0])
    rec.cmp_many(f"independent_pool_moments_vs_dense_{sfx}", [im.q1, im.q2, im.j], [idn.q1, idn.q2, idn.j], dn)
    rec.cmp_many(f"independent_pool_moments_vs_dense_grads_ABW_{sfx}", grads(im.j, leaves), grads(idn.j, leaves), dn)


def real_m1(rec: Recorder, c: dict, dev, dn: str):
    Af, Bf, W, scenes, wts, sfx = c["Af"], c["Bf"], c["head"].W, c["scenes"], c["wts"], f"{c['state']}_{dn}"
    tol_strict, iters_strict = (1e-9, 30000) if dn == "float64" else (1e-4, 5000)
    gap_start = float(MF.QuadraticHead(scenes, weights=wts).fw_gap(W.detach()))
    try:                                                             # strict mode keeps require_converged=True (package default)
        prof = MF.profiled_objective(scenes, weights=wts, ridge=0.0, initial=W.detach(), tolerance=tol_strict, max_iterations=iters_strict)
    except RuntimeError as e:
        rec.add(f"M1_strict_converged_{sfx}", "FAIL", True, failure_kind="not_converged", error=str(e)[:500], tolerance=tol_strict,
                max_iterations=iters_strict, gap_at_start=gap_start)
        prof = None
    if prof is not None:
        sol = prof.solution
        rec.expect(f"M1_strict_converged_{sfx}", sol.converged and sol.fw_gap <= tol_strict, gap=sol.fw_gap, iterations=sol.iterations,
                   tolerance=tol_strict, gap_at_start=gap_start, informative=sol.iterations > 0)
        rec.expect(f"M1_strict_W_in_box_and_detached_{sfx}", float(sol.w.abs().max()) <= 1.0 and not sol.w.requires_grad,
                   w_absmax=float(sol.w.abs().max()), n_active=int((sol.w.abs() == 1).sum()))
        manual = MF.objective_terms(scenes, sol.w.detach(), weights=wts)
        g1, g2 = grads(prof.terms.f, [Af, Bf]), grads(manual.f, [Af, Bf])
        rec.expect(f"M1_outer_grad_equals_fixed_W_partial_bitwise_{sfx}", all(torch.equal(x, y) for x, y in zip(g1, g2)))
        rec.cmp(f"M1_outer_F_equals_inner_F_{sfx}", prof.terms.f, torch.tensor(sol.f, dtype=torch.float64), dn)
    rec.raises(f"M1_strict_incomplete_solve_rejected_{sfx}",
               lambda: MF.profiled_objective(scenes, weights=wts, max_iterations=0, tolerance=1e-300), RuntimeError)
    # the adapter's rule 'absolute' with its default require_converged=True is the strict audit mode: the package call, bitwise, at
    # the absolute target 1e-4 from the seeded W (converges within the strict budget in phase 1), and it raises when not converged
    tol_a = A.M1_PROTOCOL_TOLERANCE
    try:
        pkg_a = MF.profiled_objective(scenes, weights=wts, ridge=0.0, initial=W.detach(), tolerance=tol_a, max_iterations=iters_strict)
        terms_a, info_a = A.mf_objective("MF_M1", scenes, W.detach(), wts, m1=A.M1Config(rule="absolute", max_iterations=iters_strict))
    except RuntimeError as e:
        rec.add(f"M1_adapter_rule_absolute_default_is_the_strict_package_call_bitwise_{sfx}", "FAIL", True, failure_kind="not_converged",
                error=str(e)[:500], tolerance=tol_a, max_iterations=iters_strict)
    else:
        rec.expect(f"M1_adapter_rule_absolute_default_is_the_strict_package_call_bitwise_{sfx}",
                   info_a["m1_require_converged"] is True and info_a["m1_converged"] and torch.equal(info_a["m1_W"], pkg_a.solution.w)
                   and torch.equal(terms_a.f, pkg_a.terms.f) and info_a["m1_iterations"] == pkg_a.solution.iterations
                   and info_a["m1_gap_final"] == pkg_a.solution.fw_gap <= tol_a, tolerance=tol_a, iterations=info_a["m1_iterations"],
                   gap=info_a["m1_gap_final"], require_converged=info_a["m1_require_converged"],
                   note="M1Config(rule='absolute') defaults to require_converged=True (package design 7: strict acceptance)")
    rec.raises(f"M1_adapter_rule_absolute_default_rejects_unconverged_head_{sfx}",
               lambda: A.mf_objective("MF_M1", scenes, W.detach(), wts, m1=A.M1Config(rule="absolute", tolerance=1e-300, max_iterations=0)),
               RuntimeError)
    pt = MF.profiled_objective(scenes, weights=wts, initial=W.detach(), tolerance=1e-4, max_iterations=10, require_converged=False)
    rec.expect(f"M1_training_mode_returns_and_logs_gap_{sfx}", math.isfinite(pt.solution.fw_gap) and pt.solution.iterations <= 10,
               gap=pt.solution.fw_gap, iterations=pt.solution.iterations, converged=pt.solution.converged, solver_status=pt.solution.status)
    # relative zero-head rule (decision 2026-10-17 2.1, the protocol rule) from the seeded non-zero W, i.e. a warm start, cap 10
    W0 = W.detach().clone()
    with independent_m1_counters() as cnt_r:
        terms_r, info_r = A.mf_objective("MF_M1", scenes, W0, wts, m1=A.M1Config(max_iterations=10))
    fresh = MF.QuadraticHead(scenes, weights=wts)                      # an independent snapshot of the same features
    g_ref, g_start, g_final = float(fresh.delta.abs().sum()), float(fresh.fw_gap(W0)), float(fresh.fw_gap(info_r["m1_W"]))
    rt = A.M1_RELATIVE_TOLERANCE
    solver_ok, solver = m1_solver_call_ok(cnt_r, info_r, g_ref=g_ref, w_start=W0, cap=10, dn=dn)
    rec.expect(f"M1_relative_reference_is_zero_head_gap_threshold_1pct_warm_start_gap_separate_{sfx}",
               close_rel(info_r["m1_gap_ref_zero"], g_ref, dn) and info_r["m1_tolerance_effective"] == rt * info_r["m1_gap_ref_zero"]
               and close_rel(info_r["m1_gap_start"], g_start, dn) and info_r["m1_gap_start"] != info_r["m1_gap_ref_zero"]
               and info_r["m1_tolerance_effective"] != rt * info_r["m1_gap_start"] and not info_r["m1_W_start_is_zero"]
               and solver_ok and distinct_rel(solver["solver_tolerance"], rt * g_start, dn),
               independent_gap_ref_zero=g_ref, independent_gap_start=g_start, bitwise_gap_ref_zero=info_r["m1_gap_ref_zero"] == g_ref,
               bitwise_gap_start=info_r["m1_gap_start"] == g_start, **solver, independent_counts=counts_record(cnt_r), **m1_fields(info_r),
               note="threshold = 0.01 x G_b(0) of the current batch (not of the warm start); both gaps recomputed on a fresh snapshot; "
                    "the solver call's own tolerance / initial W / cap recorded independently (with the cap binding, W alone would not "
                    "show the tolerance) and distinguishable from 0.01 x the warm-start gap")
    rec.expect(f"M1_relative_stop_rule_consistent_final_gap_is_true_FW_gap_{sfx}", m1_stop_consistent(info_r)
               and close_rel(info_r["m1_gap_final"], g_final, dn) and float(info_r["m1_W"].abs().max()) <= 1.0
               and not info_r["m1_W"].requires_grad, independent_gap_final=g_final, bitwise_gap_final=info_r["m1_gap_final"] == g_final,
               stop_reason=info_r["m1_stop_reason"], iterations=info_r["m1_iterations"])
    ref = MF.profiled_objective(scenes, weights=wts, initial=W0, tolerance=info_r["m1_tolerance_effective"], max_iterations=10,
                                require_converged=False)
    rec.expect(f"M1_relative_equals_package_profiled_objective_at_tau_b_bitwise_{sfx}", torch.equal(info_r["m1_W"], ref.solution.w)
               and torch.equal(terms_r.f, ref.terms.f) and info_r["m1_iterations"] == ref.solution.iterations
               and info_r["m1_gap_final"] == ref.solution.fw_gap,
               note="the relative rule = the package profiled objective with tolerance tau_b = 0.01 G_b(0) computed on the same batch")
    if dn != "float64":
        d = info_r["m1_fp64_diagnostic"]
        rec.expect(f"M1_relative_fp64_same_input_diagnostic_{sfx}", (not d.get("triggered")) or d.get("decisions_agree"), required=False,
                   note="diagnostic: same feature values through the same backend in float64 when g_ref <= 1e4 x 2 eps; PASS = "
                        "not triggered, or the fp32 and fp64 stop decisions at the returned W agree", **d)
    # the relative stop rule exercised on real features (diagnostic: whether 1% of G_b(0) is reached within the cap depends on the
    # feature state): zero-head start W = 0, cap M1_LARGE_CAP
    WZ = torch.zeros_like(W0)
    with independent_m1_counters() as cnt_l:
        _, info_l = A.mf_objective("MF_M1", scenes, WZ, wts, m1=A.M1Config(max_iterations=M1_LARGE_CAP))
    gl = float(fresh.fw_gap(info_l["m1_W"]))
    ok_l, solver_l = m1_solver_call_ok(cnt_l, info_l, g_ref=g_ref, w_start=WZ, cap=M1_LARGE_CAP, dn=dn)
    rec.expect(f"M1_relative_large_cap_from_W0_reaches_the_1pct_threshold_stop_rule_consistent_{sfx}",
               info_l["m1_stop_reason"] == "gap_tolerance" and info_l["m1_iterations"] < M1_LARGE_CAP and m1_stop_consistent(info_l)
               and close_rel(info_l["m1_gap_final"], gl, dn) and ok_l, required=False, independent_gap_final=gl, **solver_l,
               **{k: info_l[k] for k in ("m1_stop_reason", "m1_iterations", "m1_gap_ref_zero", "m1_gap_final", "m1_tolerance_effective",
                                         "m1_gap_relative_to_zero", "m1_converged", "m1_hvp_calls_total", "m1_moments_passes_total")},
               note=f"diagnostic: the gap_tolerance branch of the relative rule on real features (cap {M1_LARGE_CAP}); FAIL here = the cap "
                    "was reached before 1% of G_b(0) (recorded, not gating)")


def m1_zero_reference_checks(rec: Recorder, dev, dn: str):
    """Decision 2026-10-17 2.3: an exactly zero Delta_b returns W = 0 with status zero_reference_gap; nothing is divided by g_ref and the
    solver (which requires a positive tolerance) is not called.  Constructed fixture (exact_zero_delta_scenes), the host batch shape
    (16 scenes x 128 positions, K16), through the adapter path A.mf_objective, from W = 0 and from a non-zero warm start."""
    dtype = getattr(torch, dn)
    n_sc, n_pos, K = 16, 128, K_SMOKE
    scenes, wts = exact_zero_delta_scenes(n_sc, n_pos, K, dtype, dev, SEED), A.host_weights(n_sc)
    head = MF.QuadraticHead(scenes, weights=wts)
    names = (f"M1_relative_zero_reference_from_W0_returns_W0_zero_reference_gap_{dn}",
             f"M1_relative_zero_reference_from_warm_start_returns_W0_{dn}")
    if bool(head.delta.any()):
        for nm in names:
            rec.blocked(nm, "precondition not met: the constructed fixture's Delta is not exactly zero", delta_absmax=float(head.delta.abs().max()))
        return
    W0 = torch.zeros(K, K, dtype=dtype, device=dev)
    with independent_m1_counters() as cnt:
        terms, info = A.mf_objective("MF_M1", scenes, W0, wts, m1=A.M1Config())
    diag_heads = 1 if info["m1_fp64_diagnostic"].get("triggered") else 0
    rec.expect(names[0], info["m1_stop_reason"] == "zero_reference_gap" and torch.equal(info["m1_W"], W0) and info["m1_gap_ref_zero"] == 0.0
               and info["m1_gap_relative_to_zero"] is None and info["m1_tolerance_effective"] is None and info["m1_iterations"] == 0
               and info["m1_converged"] and info["m1_solver_status"] is None and info["m1_hvp_calls"] == 0 and info["m1_moments_passes"] == 2
               and float(terms.f) == 0.0 and cnt["solve_box_head"] == 0 and cnt["objective_terms"] == 1
               and cnt["heads"].get(dn) == 1 and sum(cnt["heads"].values()) == 1 + diag_heads
               and info["m1_hvp_calls_total"] == cnt["hvp"] and info["m1_moments_passes_total"] == sum(cnt["heads"].values()) + cnt["objective_terms"],
               independent_counts=counts_record(cnt), f=float(terms.f),
               **m1_fields(info), note="constructed exact-zero fixture; W = 0 is the maximiser of the negative semidefinite empirical "
                                       "head objective; not a statement about population independence")
    gen = torch.Generator(device="cpu").manual_seed(A.Q.seed_of(f"V10|T01|zero-ref-warm|{dn}"))
    Wp = torch.empty(K, K, dtype=torch.float64).uniform_(-0.9, 0.9, generator=gen).to(dev, dtype)
    with independent_m1_counters() as cnt2:
        terms2, info2 = A.mf_objective("MF_M1", scenes, Wp, wts, m1=A.M1Config())
    g_start = float(head.fw_gap(Wp))
    rec.expect(names[1], info2["m1_stop_reason"] == "zero_reference_gap" and torch.equal(info2["m1_W"], torch.zeros_like(Wp))
               and info2["m1_tolerance_effective"] is None and info2["m1_gap_relative_to_zero"] is None and not info2["m1_W_start_is_zero"]
               and close_rel(info2["m1_gap_start"], g_start, dn) and info2["m1_hvp_calls"] == 1 and cnt2["solve_box_head"] == 0
               and info2["m1_hvp_calls_total"] == cnt2["hvp"] and info2["m1_moments_passes_total"] == sum(cnt2["heads"].values()) + cnt2["objective_terms"]
               and float(terms2.f) == 0.0 and info2["m1_W_update_absmax"] == float(Wp.abs().max()), independent_gap_start=g_start,
               independent_counts=counts_record(cnt2), **m1_fields(info2),
               note="the warm start is replaced by W = 0 (the optimum); its gap is logged (1 HVP)")
    rec.raises(f"M1_solver_rejects_zero_tolerance_{dn}", lambda: MF.solve_box_head(head, tolerance=0.0), ValueError)


def real_mlp(rec: Recorder, c: dict, dev, dn: str):
    du, dv, cr, hb, n_sc = c["du"], c["dv"], c["cr"], c["hb"], c["n_sc"]
    perm = torch.as_tensor(hb.perm, device=dev)
    dvq = torch.stack([dv[s, perm[s]] for s in range(n_sc)])                       # host G L171-173 transcribed
    host_loss = A.P.loss_fn(torch, F, "vcs", cr(torch.cat([du, dv], -1)).squeeze(-1), cr(torch.cat([du, dvq], -1)).squeeze(-1))
    ad_loss, _ = A.mlp_objective("MLP_perm1", cr, du, dv, perm=perm)
    rec.cmp(f"MLP_perm1_adapter_equals_host_formula_{dn}", ad_loss, host_loss, dn, bitwise=bool(torch.equal(host_loss, ad_loss)),
            note="against the host lines transcribed here; the host trainer itself is compared in part host_replay (CPU)")
    crp = list(cr.parameters())
    perms = torch.tensor(list(itertools.permutations(range(5))), device=dev, dtype=torch.long)
    du5, dv5 = du[:1, :5], dv[:1, :5]
    _, jd5 = A.mlp_objective("MLP_dense", cr, du5, dv5)
    jp5 = torch.stack([A.mlp_objective("MLP_perm1", cr, du5, dv5, perm=p[None])[1] for p in perms]).mean()
    rec.cmp(f"MLP_dense_equals_all_120_perm1_mean_value_{dn}", jp5, jd5, dn)
    rec.cmp_many(f"MLP_dense_equals_all_120_perm1_mean_grads_{dn}", grads(jp5, [du, dv] + crp), grads(jd5, [du, dv] + crp), dn)
    _, jfull = A.mlp_objective("MLP_dense", cr, du, dv)
    _, jblk = A.mlp_objective("MLP_dense", cr, du, dv, block=32, checkpoint=True)
    names = ["du", "dv"] + [f"critic.{n}" for n, _ in cr.named_parameters()]
    g_blk, g_full = grads(jblk, [du, dv] + crp), grads(jfull, [du, dv] + crp)
    rec.cmp(f"MLP_dense_blocked_checkpointed_vs_full_value_{dn}", jblk, jfull, dn)
    # Main-session decision 2026-10-17 (job 1043657 row: 2.94e-5 = 2.83 x the fp32 bound on critic.4.weight): in fp64 the
    # blocked-vs-full comparison gates the implementation (agrees to ~3e-11 relative).  In fp32 both sides are fp32 implementations
    # and the full-graph sum over all n_sc x n x n pairs is the less accurate one (2.9e-5 vs fp64, blocked 2.9e-7), so the fp32 gate is
    # each fp32 implementation against the fp64 value of the same objective at the same fp32-rounded inputs and weights; the
    # blocked/checkpointed path (the one used for training and timing) is required, the full-graph fp32 path and the fp32
    # blocked-vs-full difference are reported as diagnostics.  The tolerance itself is unchanged (package fp32 atol 1e-5, rtol 1e-4).
    rec.cmp_many(f"MLP_dense_blocked_checkpointed_vs_full_grads_{dn}", g_blk, g_full, dn, required=(dn == "float64"),
                 per_tensor=per_tensor(names, g_blk, g_full, dn), n_pairs=int(du.shape[0] * du.shape[1] * dv.shape[1]),
                 note=None if dn == "float64" else "diagnostic in fp32: both sides fp32; gated against fp64 in the next rows")
    if dn == "float32":
        du64, dv64 = du.detach().double().requires_grad_(), dv.detach().double().requires_grad_()
        cr64 = copy.deepcopy(cr).double()
        _, j64 = A.mlp_objective("MLP_dense", cr64, du64, dv64)
        g64 = grads(j64, [du64, dv64] + list(cr64.parameters()))
        for label, g in (("full", g_full), ("blocked_checkpointed", g_blk)):
            rec.cmp_many(f"MLP_dense_{label}_float32_grads_vs_float64_value_same_inputs", g, g64, "float32",
                         required=(label == "blocked_checkpointed"), per_tensor=per_tensor(names, g, g64, "float32"),
                         note=("gate: the fp32 blocked/checkpointed implementation against fp64 at the same inputs and weights"
                               if label == "blocked_checkpointed" else
                               "diagnostic: fp32 accumulation error of the full-graph dense MLP critic (all n_sc x n x n pairs; P and Q "
                               "means nearly cancel at init) against fp64 at the same fp32-rounded inputs and weights"))
    _, j8same = A.mlp_objective("MLP_perm8", cr, du, dv, perms=perm[:, None].expand(-1, 8, -1).contiguous())
    rec.cmp(f"MLP_perm8_normalisation_equals_perm1_when_rows_repeat_{dn}", j8same, -ad_loss, dn)


def reg_side_checks(rec: Recorder, guard, fit, dev, dtype, dn: str, run_dir: Path, state: str):
    c0 = fit[0]
    sfx = f"{state}_{dn}"
    pert = {c0: A.Q.perturbations_p2(c0)}                    # REG-P2's rule applied to a FIT core: no CAL / DEV / TEST read
    inputs = A.GuardedInputs(guard, {"FIT"}, f"T01 registration-side ({sfx})", pert, dev)
    eU, eV, cr, head = build_models_seeded(SEED, dev, dtype)
    if state == "sharpened_x10":
        head = sharpened(head)
    A.freeze_for_test(eU, eV, cr, head)
    DU = A.reference_descriptors(eU, inputs, c0, dtype)
    mov = inputs.moving(c0, 0)[1]
    rng = lambda: np.random.default_rng(A.Q.seed_of(f"R6-REG-P2|pos|{c0}|0"))  # noqa: E731 - the host's per-(image, perturbation) stream

    def energy(est, record=True, eV_=eV, head_=head, DU_=DU):
        return A.MFRigidEnergy(DU_, mov, rng(), eV=eV_, head=head_, estimator=est, dtype=dtype, record=record)

    specs = geometry_specs()
    h0 = A.state_hash(eU, eV, head)
    host = A.Q.RigidEnergy(torch, F, "critic", DU.to(torch.float32), mov, rng(), eV=eV, critic=cr)
    e_probe = energy("moments")
    rec.expect(f"registration_positions_and_permutation_are_host_{sfx}", torch.equal(e_probe.perm, host.perm)
               and torch.equal(e_probe.x.to(torch.float32), host.x), n_pos=int(len(host.x)))
    values = {}
    for spec in specs:
        name = spec["name"]
        e = energy("moments")
        p = torch.tensor(spec["p"], dtype=dtype, device=dev, requires_grad=True)
        Ev = e(p)
        dv_host, B = e.last["dv"][0], e.last["B"][0]
        dv_explicit = A.host_moving_descriptors(e, p)
        rec.cmp(f"host_chain_dv_equals_explicit_chain_{name}_{sfx}", dv_host, dv_explicit, dn, bitwise=bool(torch.equal(dv_host, dv_explicit)))
        sc = MF.SceneBatch(e.A, B, "v")
        pm = MF.objective_terms([sc], head.W, weights=[1.0])
        pdn = MF.dense_objective_terms([sc], head.W, weights=[1.0], chunk_size=1024)
        rec.cmp(f"registration_batched_vs_package_moments_value_{name}_{sfx}", Ev, pm.j, dn)
        rec.cmp_many(f"registration_package_moments_vs_dense_terms_{name}_{sfx}", [pm.p1, pm.p2, pm.q1, pm.q2, pm.j],
                     [pdn.p1, pdn.p2, pdn.q1, pdn.q2, pdn.j], dn)
        g_rows, g_pm, g_pd = grads(Ev, [p])[0], grads(pm.j, [p])[0], grads(pdn.j, [p])[0]
        rec.cmp(f"registration_geometry_grad_batched_vs_package_moments_{name}_{sfx}", g_rows, g_pm, dn)
        rec.cmp(f"registration_geometry_grad_moments_vs_dense_{name}_{sfx}", g_pm, g_pd, dn, grad=g_pm.tolist())
        rec.expect(f"registration_geometry_grad_finite_nonzero_{name}_{sfx}", bool(torch.isfinite(g_rows).all()) and float(g_rows.abs().max()) > 0,
                   grad=g_rows.tolist())
        e1 = energy("perm1")
        p1 = torch.tensor(spec["p"], dtype=dtype, device=dev, requires_grad=True)
        E1 = e1(p1)
        pp = MF.sampled_permutation_terms([MF.SceneBatch(e1.A, e1.last["B"][0], "v")], head.W, weights=[1.0], permutations=[e1.perm[None]])
        rec.cmp(f"registration_batched_vs_package_perm1_value_{name}_{sfx}", E1, pp.j, dn)
        rec.cmp(f"registration_batched_vs_package_perm1_geometry_grad_{name}_{sfx}", grads(E1, [p1])[0], grads(pp.j, [p1])[0], dn)
        values[name] = (float(Ev), g_rows.detach().clone())
    # fixed W across geometry candidates: no refit, no parameter change, frozen contract, geometry gradients kept
    e = energy("moments", record=False)
    tt = torch.tensor(A.Q.grid_points()[1][:256], dtype=torch.float32, device=dev)
    n_cand = 0
    with torch.no_grad():
        for th in (-9.0, 12.0):
            e.grid_values(th, tt)
            n_cand += len(tt)
    gnorms = []
    for k in range(5):
        p = torch.tensor([0.37 * k * A.Q.RHO * math.pi / 180.0 * 5, 2.0 - k, -3.0 + 1.5 * k], dtype=dtype, device=dev, requires_grad=True)
        (-e(p)).backward()
        gnorms.append(float(p.grad.abs().max()))
        n_cand += 1
    full_search = None
    if dev.type == "cuda" and dn == "float32":
        th_hat, t_hat, n_grid, f_hat = A.Q.optimise_rigid(torch, e, dev)          # one full host search (grid + 3 x 100 Adam steps)
        full_search = {"n_grid": int(n_grid), "theta_hat_deg": float(th_hat), "t_hat": [float(x) for x in t_hat]}
        n_cand += int(n_grid) + 300
    h1 = A.state_hash(eU, eV, head)
    rec.expect(f"fixed_W_state_hash_unchanged_across_candidates_{sfx}", h0 == h1, before=h0, after=h1, n_candidates=n_cand, full_search=full_search)
    mods = (eU, eV, head)
    rec.expect(f"test_time_params_frozen_eval_no_grad_{sfx}", all(not m.training for m in mods)
               and all((not q.requires_grad) and q.grad is None for m in mods for q in m.parameters()))
    rec.expect(f"test_time_geometry_grads_finite_nonzero_{sfx}", all(math.isfinite(g) and g > 0 for g in gnorms), grad_absmax=gnorms)
    trainable = copy.deepcopy(eV).train()
    for q in trainable.parameters():
        q.requires_grad_(True)
    rec.raises(f"test_time_energy_refuses_trainable_models_{sfx}", lambda: energy("moments", eV_=trainable), ValueError)
    # checkpoint save / restore of the indivisible (encoders, feature maps, W) pair
    path = run_dir / f"smoke_pair_{sfx}.pt"
    sha = A.save_pair(path, eU, eV, head, {"purpose": "T01 smoke checkpoint round trip", "seed": SEED, "dtype": dn, "K": K_SMOKE, "state": state})
    eU2, eV2, head2, meta = A.load_pair(path, dev, dtype)
    A.freeze_for_test(eU2, eV2, head2)
    rec.expect(f"checkpoint_roundtrip_state_hash_equal_{sfx}", A.state_hash(eU2, eV2, head2) == h0, sha256=sha, bytes=path.stat().st_size, meta=meta)
    DU2 = A.reference_descriptors(eU2, inputs, c0, dtype)
    for spec in specs:
        e2 = energy("moments", eV_=eV2, head_=head2, DU_=DU2)
        p = torch.tensor(spec["p"], dtype=dtype, device=dev, requires_grad=True)
        E2 = e2(p)
        g2 = grads(E2, [p])[0]
        v0, g0 = values[spec["name"]]
        rec.cmp(f"checkpoint_roundtrip_energy_value_{spec['name']}_{sfx}", E2, torch.tensor(v0, dtype=torch.float64), dn,
                bitwise=float(E2) == v0)
        rec.cmp(f"checkpoint_roundtrip_geometry_grad_{spec['name']}_{sfx}", g2, g0, dn, bitwise=bool(torch.equal(g2, g0)))


def one_update_checks(rec: Recorder, guard, fit, dev, n_sc: int, *, imgs=None, build_models=None):
    """imgs / build_models: injected only by the unit tests (synthetic images, tiny encoders); T01 reads the real FIT batch through
    the guard and builds the host models."""
    replay = A.HostBatchSampler(fit, SEED, n_sc=n_sc)               # the trainers' own draws (same seed): batches 1 and 2
    first, second = replay.draw(), replay.draw()
    if imgs is None:                                                # batch 2: the M1 warm-start check (3b) takes a second step
        imgs = A.load_learned_inputs(guard, sorted(set(first.cores) | set(second.cores)), {"FIT"},
                                     "T01 one-update checks (float32; the first two host batches)", dev)
    n_pos = first.perm.shape[1]

    def trainer(method, **kw):
        return A.Trainer(A.TrainConfig(method=method, seed=SEED, n_sc=n_sc, **kw), imgs, fit, dev, build_models=build_models)

    def enc_params(t):
        return [p for m in (t.eU, t.eV) for p in m.parameters()]

    def fm_params(t):
        return list(t.head.left.parameters()) + list(t.head.right.parameters())

    def grad_norm(params):
        gs = [p.grad for p in params if p.grad is not None]
        return float(torch.sqrt(sum((g.double() ** 2).sum() for g in gs))) if gs else 0.0

    def max_change(before, params):
        return max(float((p.detach() - b).abs().max()) for p, b in zip(params, before))

    # (1) MF_perm1 vs MF_M0 on the same first batch, both with the SAME non-zero seeded W (with the package start W = 0 every
    #     positive term is 0 for any features, so a W = 0 comparison would carry no information)
    tA, tB = trainer("MF_perm1", w_init="uniform"), trainer("MF_M0", w_init="uniform")
    rec.expect("MF_perm1_vs_M0_identical_initial_state_nonzero_W", A.state_hash(tA.eU, tA.eV, tA.head) == A.state_hash(tB.eU, tB.eV, tB.head)
               and float(tA.head.W.abs().max()) > 0, W_absmax=float(tA.head.W.abs().max()))
    bA, bB = tA.draw(), tB.draw()
    rec.expect("same_first_batch_across_methods", bA[0].cores == bB[0].cores == first.cores and np.array_equal(bA[0].yx, bB[0].yx)
               and np.array_equal(bA[0].perm, bB[0].perm) and torch.equal(bA[1], bB[1]) and torch.equal(bA[2], bB[2]),
               note="cores, positions, permutations, BF crops (bu) and SHG crops (bv)")
    _, oA = tA.forward(*bA)
    _, oB = tB.forward(*bB)
    same = {k: bool(torch.equal(oA[k], oB[k])) for k in ("du", "dv", "A", "B")}
    rec.expect("MF_perm1_vs_M0_same_descriptors_and_features_bitwise", all(same.values()), **same)
    tpos_a = ((oA["A"] @ tA.head.W) * oA["B"]).sum(-1)
    tpos_b = ((oB["A"] @ tB.head.W) * oB["B"]).sum(-1)
    rec.expect("MF_perm1_vs_M0_same_positive_scores_bitwise_nonzero", bool(torch.equal(tpos_a, tpos_b)) and float(tpos_a.abs().max()) > 0,
               t_absmax=float(tpos_a.abs().max()))
    pa, pb = (oA["terms"].p1, oA["terms"].p2), (oB["terms"].p1, oB["terms"].p2)
    rec.expect("MF_perm1_vs_M0_same_positive_terms_bitwise_nonzero", bool(torch.equal(pa[0], pb[0]) and torch.equal(pa[1], pb[1]))
               and abs(float(pa[0])) > 0 and float(pa[1]) > 0, p1=float(pa[0]), p2=float(pa[1]))
    order = [s.scene_id for s in oA["scenes"]] == [s.scene_id for s in oB["scenes"]] == [str(i) for i in range(n_sc)]
    rows_ok = all(torch.equal(sc.a, oA["A"][i * n_pos:(i + 1) * n_pos]) and torch.equal(sc.b, oA["B"][i * n_pos:(i + 1) * n_pos])
                  for i, sc in enumerate(oA["scenes"]))
    from_crop = A.compare_many([tA.head.features_a(oA["du"][i]) for i in range(n_sc)] + [tA.head.features_b(oA["dv"][i]) for i in range(n_sc)],
                               [oA["A"][i * n_pos:(i + 1) * n_pos] for i in range(n_sc)] + [oA["B"][i * n_pos:(i + 1) * n_pos] for i in range(n_sc)],
                               "float32")["allclose"]
    rec.expect("MF_perm1_vs_M0_same_scene_order_scene_s_is_crop_s", order and rows_ok and from_crop, order=order, rows_bitwise=rows_ok,
               features_from_crop_s=from_crop, cores=first.cores)
    del oA, oB, tA, tB

    # (2) one step from the package start W = 0, then one step from a boundary head (the batch's box QP: active entries)
    for method in ("MF_perm1", "MF_M0"):
        t = trainer(method)
        W0 = t.head.W.detach().clone()
        r = t.step(t.draw())
        rec.expect(f"{method}_step_from_W0_moves_W_inside_box", float((t.head.W - W0).abs().max()) > 0 and r["w_absmax"] <= 1.0,
                   w_absmax=r["w_absmax"], J=r["J"])
        rec.expect(f"{method}_encoder_grad_exactly_zero_at_W0_expected", grad_norm(enc_params(t)) == 0.0, required=False,
                   note="dJ/dA and dJ/dB are linear in W: the package init W = 0 gives zero feature / encoder gradients at the first "
                        "step (encoders then only decay by AdamW weight decay); recorded as a structural fact",
                   encoder_grad_norm=grad_norm(enc_params(t)))
        t = trainer(method)
        batch = t.draw()
        with torch.no_grad():
            _, out = t.forward(*batch)
            prof = MF.profiled_objective(out["scenes"], weights=A.host_weights(n_sc), tolerance=BOUNDARY_TOL, max_iterations=BOUNDARY_ITERS,
                                         require_converged=False)
            t.head.W.copy_(prof.solution.w)
        n_active = int((t.head.W.abs() == 1).sum())
        pre = {"n_active_before": n_active, "boundary_solve": {"tolerance": BOUNDARY_TOL, "max_iterations": BOUNDARY_ITERS,
                                                               "iterations": prof.solution.iterations, "gap": prof.solution.fw_gap,
                                                               "gap_at_W0_equals_Delta_l1": prof.solution.history[0]["fw_gap"],
                                                               "w_absmax": float(prof.solution.w.abs().max())}}
        names = (f"{method}_W_projection_active_and_feasible", f"{method}_encoder_grad_nonzero_and_params_change",
                 f"{method}_feature_maps_change", f"{method}_step_features_nonconstant")
        if n_active == 0:
            for nm in names:
                rec.blocked(nm, "precondition not met: the box-QP head of this batch has no active entry (n_active = 0)", **pre)
            del out
            continue
        before_enc = [p.detach().clone() for p in enc_params(t)]
        before_fm = [p.detach().clone() for p in fm_params(t)]
        r = t.step(batch, keep=True)
        rec.expect(names[0], r["w_absmax_preprojection"] > 1.0 and r["w_absmax"] <= 1.0, w_absmax_preprojection=r["w_absmax_preprojection"],
                   w_absmax=r["w_absmax"], **pre)
        rec.expect(names[1], grad_norm(enc_params(t)) > 0 and max_change(before_enc, enc_params(t)) > 0,
                   encoder_grad_norm_postclip=grad_norm(enc_params(t)), encoder_max_change=max_change(before_enc, enc_params(t)),
                   grad_norm_preclip=r["grad_norm_preclip"])
        rec.expect(names[2], max_change(before_fm, fm_params(t)) > 0, max_change=max_change(before_fm, fm_params(t)))
        o = r["_out"]
        rec.expect(names[3], float(o["A"].std(0).max()) > 0 and float(o["B"].std(0).max()) > 0,
                   a_std_max=float(o["A"].std(0).max()), b_std_max=float(o["B"].std(0).max()))
        del r, o, out

    # (3) M1 (decision 2026-10-17 2.1 / 2.3: the relative zero-head rule is the protocol; rule 'absolute' = strict numeric audit by
    #     default (require_converged True); its approximate pre-decision mode only in the reference rows 3d-3f, explicitly).
    # (3a) zero-head start at the real host init: the package start W = 0, protocol rule, cap 30 (M1Config default)
    wts = A.host_weights(n_sc)
    rt = A.M1_RELATIVE_TOLERANCE
    rec.raises("MF_M1_trainer_refuses_rule_absolute_without_the_reference_flag", lambda: trainer("MF_M1", m1=A.M1Config(rule="absolute")),
               ValueError)
    t = trainer("MF_M1")
    cap_default = t.cfg.m1.max_iterations
    before_enc = [p.detach().clone() for p in enc_params(t)]
    before_fm = [p.detach().clone() for p in fm_params(t)]
    W_start1 = t.head.W.detach().clone()
    with independent_m1_counters() as cnt1:
        r1 = t.step(t.draw(), keep=True)
    i1 = r1["_out"]["info"]
    gn, enc_chg, fm_chg = grad_norm(enc_params(t)), max_change(before_enc, enc_params(t)), max_change(before_fm, fm_params(t))
    spot = r1.get("encoder_gradient_diagnostic") or {}
    rec.expect("MF_M1_relative_from_W0_iterates_moves_W_and_trains_encoders", i1["m1_W_start_is_zero"] and r1["m1_iterations"] > 0
               and r1["w_absmax"] > 0 and r1["m1_W_update_absmax"] > 0 and gn > 0 and enc_chg > 0 and fm_chg > 0
               and spot.get("encoders_grad_norm", 0.0) > 0, encoder_grad_norm_postclip=gn, encoder_max_change=enc_chg,
               feature_map_max_change=fm_chg, encoder_gradient_diagnostic=spot, J=r1["J"], **m1_fields(i1),
               note="the absolute target 1e-4 stopped this batch at iteration 0 (row MF_M1_absolute_tolerance_1e-4_from_W0_...)")
    fresh1 = MF.QuadraticHead(r1["_out"]["scenes"], weights=wts)
    g_ref1 = float(fresh1.delta.abs().sum())
    ok1, solver1 = m1_solver_call_ok(cnt1, i1, g_ref=g_ref1, w_start=W_start1, cap=cap_default, dn="float32")
    rec.expect("MF_M1_relative_reference_is_l1_delta_of_the_batch_snapshot_and_gap_at_W0", close_rel(i1["m1_gap_ref_zero"], g_ref1, "float32")
               and i1["m1_tolerance_effective"] == rt * i1["m1_gap_ref_zero"] and close_rel(i1["m1_gap_start"], i1["m1_gap_ref_zero"], "float32")
               and i1["m1_moments_passes"] == 2 and m1_stop_consistent(i1) and ok1 and solver1["solver_initial_is_zero"] is True,
               independent_gap_ref_zero=g_ref1, bitwise_vs_fresh_snapshot=i1["m1_gap_ref_zero"] == g_ref1,
               bitwise_gap_start_at_W0=i1["m1_gap_start"] == i1["m1_gap_ref_zero"], **solver1, independent_counts=counts_record(cnt1),
               note="G_b(0) = ||Delta_b||_1 of the batch's own snapshot; from W = 0 the solver's iteration-0 gap is the same quantity; the "
                    "solver call itself (recorded independently) got tau_b = 0.01 G_b(0), the start W = 0 and the cap")
    d1 = i1["m1_fp64_diagnostic"]
    rec.expect("MF_M1_relative_fp64_same_input_diagnostic_host_init", bool(d1.get("triggered")) and bool(d1.get("decisions_agree")),
               required=False, note="diagnostic: g_ref at the host init is near fp32 resolution; same feature values in float64", **d1)
    # (3b) warm start: the next batch starts from this batch's W; its start gap differs from its own zero-head reference
    W_prev = t.head.W.detach().clone()
    rec.expect("MF_M1_relative_next_batch_warm_starts_from_previous_W", torch.equal(W_prev, i1["m1_W"]) and float(W_prev.abs().max()) > 0,
               w_absmax=float(W_prev.abs().max()))
    with independent_m1_counters() as cnt2:
        r2 = t.step(t.draw(), keep=True)
    i2 = r2["_out"]["info"]
    fresh2 = MF.QuadraticHead(r2["_out"]["scenes"], weights=wts)
    g_ref2, g_start2 = float(fresh2.delta.abs().sum()), float(fresh2.fw_gap(W_prev))
    ok2, solver2 = m1_solver_call_ok(cnt2, i2, g_ref=g_ref2, w_start=W_prev, cap=cap_default, dn="float32")
    rec.expect("MF_M1_relative_warm_start_gap_differs_from_zero_head_reference_threshold_uses_reference",
               (not i2["m1_W_start_is_zero"]) and i2["m1_W_start_absmax"] == float(W_prev.abs().max())
               and close_rel(i2["m1_gap_ref_zero"], g_ref2, "float32") and close_rel(i2["m1_gap_start"], g_start2, "float32")
               and i2["m1_gap_start"] != i2["m1_gap_ref_zero"] and i2["m1_tolerance_effective"] == rt * i2["m1_gap_ref_zero"]
               and i2["m1_tolerance_effective"] != rt * i2["m1_gap_start"] and m1_stop_consistent(i2)
               and ok2 and distinct_rel(solver2["solver_tolerance"], rt * g_start2, "float32"),
               independent_gap_ref_zero=g_ref2, independent_gap_start=g_start2, tolerance_from_warm_start_gap=rt * g_start2, **solver2,
               independent_counts=counts_record(cnt2), **m1_fields(i2),
               note="the solver call itself (recorded independently) got tau_b = 0.01 G_b(0) of this batch, distinguishable from 0.01 x "
                    "the warm-start gap, and the previous batch's W as its start; the cap binds here, so W alone would not show it")
    del r1, r2, i1, i2, fresh1, fresh2
    # (3c) cap reached at the host init (cap 10): approximate W returned with its true FW gap, kept as the warm start
    t = trainer("MF_M1", m1=A.M1Config(max_iterations=10))
    W_startc = t.head.W.detach().clone()
    with independent_m1_counters() as cntc:
        rc = t.step(t.draw(), keep=True)
    ic = rc["_out"]["info"]
    name = "MF_M1_relative_cap_10_reached_returns_approximate_W_with_true_gap"
    if ic["m1_stop_reason"] != "iteration_budget_exhausted":
        rec.blocked(name, "precondition not met: the relative threshold was reached within 10 iterations at the host init", **m1_fields(ic))
    else:
        headc = MF.QuadraticHead(rc["_out"]["scenes"], weights=wts)
        true_gap = float(headc.fw_gap(ic["m1_W"]))
        okc, solverc = m1_solver_call_ok(cntc, ic, g_ref=float(headc.delta.abs().sum()), w_start=W_startc, cap=10, dn="float32")
        rec.expect(name, ic["m1_iterations"] == 10 and not ic["m1_converged"] and ic["m1_gap_final"] > ic["m1_tolerance_effective"]
                   and close_rel(ic["m1_gap_final"], true_gap, "float32") and torch.equal(t.head.W, ic["m1_W"])
                   and 0 < float(ic["m1_W"].abs().max()) <= 1.0 and math.isfinite(rc["J"]) and okc, independent_true_gap=true_gap,
                   bitwise_true_gap=ic["m1_gap_final"] == true_gap, **solverc, independent_counts=counts_record(cntc), **m1_fields(ic))
    del rc, ic
    # (3d) counters against independent counts (relative rule, and rule 'absolute' as a cap-binding reference row), and the relative
    #      path = the package call at tau_b; the per-batch totals include the fp64 diagnostic (decision 2.3)
    for label, cfg_m1, ref_only in (("relative", A.M1Config(), False),
                                    ("absolute_cap_binding", A.M1Config(rule="absolute", tolerance=A.M1_CAP_BINDING_TOLERANCE,
                                                                        require_converged=False), True)):
        t = trainer("MF_M1", m1=cfg_m1, absolute_reference_only=ref_only)
        batch = t.draw()
        with torch.no_grad(), independent_m1_counters() as cnt:
            _, out = t.forward(*batch)
        info = out["info"]
        diag = info["m1_fp64_diagnostic"]
        diag_hvp = diag.get("hvp_calls", 0) if diag.get("triggered") else 0
        diag_mom = diag.get("moments_passes", 0) if diag.get("triggered") else 0
        restarts = info["m1_momentum_restarts"]
        rec.expect(f"MF_M1_{label}_counters_equal_independent_count", info["m1_hvp_calls"] + diag_hvp == cnt["hvp"]
                   and info["m1_hvp_calls"] == 2 + 3 * info["m1_iterations"] + restarts and 0 <= restarts <= info["m1_iterations"]
                   and info["m1_moments_passes"] == 2 and cnt["objective_terms"] == 1 and cnt["solve_box_head"] == 1
                   and cnt["heads"].get("float32") == 1 and cnt["heads"].get("float64", 0) == diag_mom
                   and info["m1_fp64_diag_hvp_calls"] == diag_hvp and info["m1_fp64_diag_moments_passes"] == diag_mom
                   and info["m1_hvp_calls_total"] == cnt["hvp"] and info["m1_moments_passes_total"] == sum(cnt["heads"].values()) + cnt["objective_terms"],
                   independent_counts=counts_record(cnt), diag_hvp_calls=diag_hvp, diag_moments_passes=diag_mom, **m1_fields(info),
                   note="moments passes = QuadraticHead snapshots in the working dtype + outer objective_terms calls; the fp64 "
                        "diagnostic (when triggered) counts its own snapshot / HVPs (m1_fp64_diag_*), and the per-batch totals "
                        "(m1_*_total) equal the independent counts of everything the call did")
        if label == "relative":
            with torch.no_grad():
                ref = MF.profiled_objective(out["scenes"], weights=wts, ridge=0.0, initial=t.head.W.detach(),
                                            tolerance=info["m1_tolerance_effective"], max_iterations=30, require_converged=False)
            rec.expect("MF_M1_relative_adapter_path_equals_package_profiled_objective_at_tau_b_bitwise", torch.equal(info["m1_W"], ref.solution.w)
                       and torch.equal(out["terms"].f, ref.terms.f) and info["m1_iterations"] == ref.solution.iterations
                       and info["m1_gap_final"] == ref.solution.fw_gap, tau_b=info["m1_tolerance_effective"])
        del out, info
    # (3e) the absolute target 1e-4 from W = 0 (the pre-decision training setting: require_converged False): DIAGNOSTIC of the
    #      host-init degeneracy (a reference row: absolute_reference_only)
    t = trainer("MF_M1", m1=A.M1Config(rule="absolute", require_converged=False), absolute_reference_only=True)
    before_enc = [p.detach().clone() for p in enc_params(t)]
    r = t.step(t.draw())
    gn = grad_norm(enc_params(t))
    rec.expect("MF_M1_absolute_tolerance_1e-4_from_W0_moves_W_and_trains_encoders", r["w_absmax"] > 0 and gn > 0, required=False,
               note="DIAGNOSTIC kept from phase 1 (formerly MF_M1_protocol_tolerance_from_W0_moves_W_and_trains_encoders): with the "
                    "absolute target 1e-4 the first FW gap from W = 0 equals ||Delta||_1; below 1e-4 the head stops at iteration 0, W "
                    "stays 0 and the encoder gradient is exactly 0.  The protocol now uses the relative zero-head rule (rows above)",
               m1_gap_start_equals_Delta_l1=r["m1_gap_start"], m1_absolute_tolerance=r["m1_absolute_tolerance"], m1_iterations=r["m1_iterations"],
               m1_converged=r["m1_converged"], m1_stop_reason=r["m1_stop_reason"], m1_require_converged=r["m1_require_converged"],
               w_absmax=r["w_absmax"], encoder_grad_norm=gn, J=r["J"], encoder_max_change=max_change(before_enc, enc_params(t)))
    # (3f) rule 'absolute' with a binding iteration cap (the phase-1 T02 cost configuration; reference rows, require_converged False)
    m1cap = A.M1Config(rule="absolute", tolerance=A.M1_CAP_BINDING_TOLERANCE, max_iterations=30, require_converged=False)
    t = trainer("MF_M1", m1=m1cap, absolute_reference_only=True)
    before_enc = [p.detach().clone() for p in enc_params(t)]
    before_fm = [p.detach().clone() for p in fm_params(t)]
    r = t.step(t.draw(), keep=True)
    keys = ("m1_gap_final", "m1_iterations", "m1_converged", "m1_solver_status", "m1_stop_reason", "m1_gap_start", "m1_hvp_calls",
            "m1_momentum_restarts")
    rec.expect("MF_M1_cap_binding_step_logs_gap_iterations_hvp", all(k in r for k in keys) and math.isfinite(r["m1_gap_final"])
               and r["m1_gap_scope"] == "inner_pre_outer_batch" and not r["m1_require_converged"] and r["m1_iterations"] == 30
               and r["m1_momentum_restarts"] >= 0 and not r["m1_converged"] and r["m1_stop_reason"] == "iteration_budget_exhausted",
               **{k: r.get(k) for k in keys})
    rec.expect("MF_M1_cap_binding_W_is_inner_solution_nonzero_in_box", torch.equal(t.head.W, r["_out"]["info"]["m1_W"])
               and 0 < r["w_absmax"] <= 1.0 and not t.head.W.requires_grad, w_absmax=r["w_absmax"])
    rec.expect("MF_M1_encoder_grad_nonzero_and_params_change", grad_norm(enc_params(t)) > 0 and max_change(before_enc, enc_params(t)) > 0,
               encoder_grad_norm_postclip=grad_norm(enc_params(t)), encoder_max_change=max_change(before_enc, enc_params(t)))
    rec.expect("MF_M1_feature_maps_change", max_change(before_fm, fm_params(t)) > 0, max_change=max_change(before_fm, fm_params(t)))
    rec.expect("MF_M1_W_not_an_optimiser_parameter", all(p is not t.head.W for g in t.opt.param_groups for p in g["params"]))
    del r
    t = trainer("MF_M1", m1=m1cap, absolute_reference_only=True)
    batch = t.draw()
    with torch.no_grad():
        _, out = t.forward(*batch)
        ref = MF.profiled_objective(out["scenes"], weights=A.host_weights(n_sc), ridge=0.0, initial=t.head.W.detach(),
                                    tolerance=m1cap.tolerance, max_iterations=m1cap.max_iterations, require_converged=False)
    restored = (MF.QuadraticHead.__dict__["_hvp"].__module__ == MF.__name__ and MF.QuadraticHead.__dict__["__init__"].__module__ == MF.__name__
                and MF.solve_box_head.__module__ == MF.__name__ and MF.objective_terms.__module__ == MF.__name__
                and A.objective_terms is MF.objective_terms)
    rec.expect("MF_M1_adapter_path_equals_package_profiled_objective_bitwise", torch.equal(out["info"]["m1_W"], ref.solution.w)
               and torch.equal(out["terms"].f, ref.terms.f) and out["info"]["m1_iterations"] == ref.solution.iterations and restored,
               wrappers_restored=restored, note="rule 'absolute' with require_converged False (reference mode, explicit); the "
                                                "HVP-counting wrapper and the independent counters change no value and are removed "
                                                "after the call")
    del out, ref
    # (4) host MLP critic
    t = trainer("MLP_perm1")
    before_enc = [p.detach().clone() for p in enc_params(t)]
    r = t.step(t.draw())
    rec.expect("MLP_perm1_encoder_grad_nonzero_and_params_change", grad_norm(enc_params(t)) > 0 and max_change(before_enc, enc_params(t)) > 0,
               J=r["J"], grad_norm_preclip=r["grad_norm_preclip"])
    t = trainer("MLP_perm8")
    hb, _, _ = t.draw()
    pr = t.extra_perms(hb)
    rec.expect("MLP_perm8_row0_is_host_permutation_rows_are_permutations", pr.shape == (n_sc, 8, hb.perm.shape[1])
               and np.array_equal(pr[:, 0], hb.perm) and all(np.array_equal(np.sort(row), np.arange(hb.perm.shape[1])) for blk in pr for row in blk))


def weight_decay_checks(rec: Recorder, fit, dev):
    """AdamW decoupled decay with zero gradients in float64, so that lr * wd * |p| (~1e-8) is resolved (in fp32 it is ~1 ulp):
    W must not decay (wd 0); feature maps and encoders decay by exactly (1 - lr * wd).  No data is read (no batch drawn)."""
    t = A.Trainer(A.TrainConfig(method="MF_M0", seed=SEED, w_init="uniform", dtype="float64"), {}, fit, dev)
    groups = {g["name"]: g for g in t.opt.param_groups}
    snap = {name: [p.detach().clone() for p in g["params"]] for name, g in groups.items()}
    for p in t.clip_params:
        p.grad = torch.zeros_like(p)
    t.opt.step()
    for name, g in groups.items():
        lr, wd, new, old = g["lr"], g["weight_decay"], g["params"], snap[name]
        if wd == 0:
            rec.expect(f"weight_decay_{name}_zero_exact_float64", all(torch.equal(n_, o) for n_, o in zip(new, old)), lr=lr, weight_decay=wd)
            continue
        err = max(float((n_ - o * (1 - lr * wd)).abs().max()) for n_, o in zip(new, old))
        ratio = sum(float((n_ - o).abs().sum()) for n_, o in zip(new, old)) / (lr * wd * sum(float(o.abs().sum()) for o in old))
        rec.expect(f"weight_decay_{name}_is_host_value_float64", err <= 1e-15 and abs(ratio - 1) <= 1e-6, lr=lr, weight_decay=wd,
                   max_err_vs_expected=err, decay_ratio=ratio,
                   note="decay_ratio = sum|p_new - p_old| / (lr wd sum|p_old|): 1 for the host decay 1e-4, 0 without decay, 10 for 1e-3")
    rec.expect("weight_decay_groups_are_the_protocol_values", {n: g["weight_decay"] for n, g in groups.items()} == {"encoders": 1e-4, "feature_maps": 1e-4, "W": 0.0},
               groups={n: {"lr": g["lr"], "weight_decay": g["weight_decay"]} for n, g in groups.items()})


# ------------------------------------------------------------------------------------------------------------ part: geometry
def part_geometry(device: str, run_dir: Path) -> dict:
    rec = Recorder("geometry", device)
    dev, dt = torch.device(device), torch.float64
    guard = A.DataGuard()
    fd_rows = []
    extra = {"fd_steps": FD_STEPS, "fd_atol": FD_ATOL, "fd_rtol": FD_RTOL, "parameterisation": "p = (u, ty, tx), theta = u / rho, rho = R_OMEGA / sqrt(2) = %.6f; ty, tx in working px (512 grid)" % A.Q.RHO}
    with rec.group("lattice_transforms"):
        specs = geometry_specs()
        for s in specs:
            rec.expect(f"kink_margin_{s['name']}", s["margin"] >= s["expected_margin"] - 1e-9, margin=s["margin"], expected=s["expected_margin"],
                       theta_deg=s["theta_deg"], p=s["p"].tolist(), within_allowed_set=abs(s["theta_deg"]) <= 30 and math.hypot(*s["p"][1:]) <= 51)
            for d in range(3):
                for h in FD_STEPS:
                    n = kink_crossings(s["p"], d, h)
                    rec.expect(f"zero_kink_crossings_{s['name']}_dir{d}_h{h:g}", n == 0, value=n)
        extra["transforms"] = [{**{k: v for k, v in s.items() if k != "p"}, "p": s["p"].tolist()} for s in specs]
    fit = A.fit_cores(guard, 1.0, SEED)
    c0 = fit[0]
    pert = {c0: A.Q.perturbations_p2(c0)}
    inputs = A.GuardedInputs(guard, {"FIT"}, "T01 geometry gradient check", pert, dev)
    mov = inputs.moving(c0, 0)[1]
    rng = lambda: np.random.default_rng(A.Q.seed_of(f"R6-REG-P2|pos|{c0}|0"))  # noqa: E731
    extra.update({"core": c0, "core_role": "FIT", "moving_perturbation": pert[c0][0]})
    for state in ("host_init", "sharpened_x10"):
        with rec.group(f"state_{state}"):
            eU, eV, cr, head = build_models_seeded(SEED, dev, dt)
            if state == "sharpened_x10":       # synthetic stress state: feature-map weights x 10 -> far-from-uniform simplex features
                with torch.no_grad():
                    for lin in (head.left, head.right):
                        lin.weight.mul_(10.0)
                        lin.bias.mul_(10.0)
            A.freeze_for_test(eU, eV, cr, head)
            DU = A.reference_descriptors(eU, inputs, c0, dt)
            e = A.MFRigidEnergy(DU, mov, rng(), eV=eV, head=head, estimator="moments", dtype=dt, record=True)
            for s in specs:
                p = torch.tensor(s["p"], dtype=dt, device=dev, requires_grad=True)
                tv = term_values(e, p)
                B = e.last["B"][0]
                g_ad = {k: grads(v, [p])[0].detach().cpu().numpy() for k, v in tv.items()}
                vals = {k: float(v) for k, v in tv.items()}
                rec.cmp_many(f"dense_vs_moments_values_{state}_{s['name']}", [tv["q1_moments"], tv["q2_moments"], tv["J_moments"], tv["J_adapter_batched_moments"]],
                             [tv["q1_dense"], tv["q2_dense"], tv["J_dense"], tv["J_dense"]], "float64", values=vals)
                rec.cmp_many(f"dense_vs_moments_geometry_grads_{state}_{s['name']}",
                             [torch.tensor(g_ad[k]) for k in ("q1_moments", "q2_moments", "J_moments", "J_adapter_batched_moments")],
                             [torch.tensor(g_ad[k]) for k in ("q1_dense", "q2_dense", "J_dense", "J_dense")], "float64")
                for k in ("p1", "p2", "q1_moments", "q2_moments"):
                    rec.expect(f"geometry_grad_nonzero_{k}_{state}_{s['name']}", float(np.abs(g_ad[k]).max()) > 0, grad=g_ad[k].tolist())
                fd = {}
                p0 = s["p"]
                with torch.no_grad():
                    for d in range(3):
                        for h in FD_STEPS:
                            ev = np.zeros(3)
                            ev[d] = h
                            plus = term_values(e, torch.tensor(p0 + ev, dtype=dt, device=dev))
                            minus = term_values(e, torch.tensor(p0 - ev, dtype=dt, device=dev))
                            for k in tv:
                                fd[(k, d, h)] = (float(plus[k]) - float(minus[k])) / (2 * h)
                for k in tv:
                    scale = float(np.abs(g_ad[k]).max())
                    ok_all, worst = True, 0.0
                    for d in range(3):
                        for h in FD_STEPS:
                            err = abs(fd[(k, d, h)] - float(g_ad[k][d]))
                            ok = err <= FD_ATOL + FD_RTOL * scale
                            ok_all &= ok
                            worst = max(worst, err / (FD_ATOL + FD_RTOL * scale))
                            fd_rows.append({"state": state, "transform": s["name"], "term": k, "direction": ("u", "ty", "tx")[d], "h": h,
                                            "g_autograd": float(g_ad[k][d]), "g_fd": fd[(k, d, h)], "abs_err": err, "scale": scale,
                                            "bound": FD_ATOL + FD_RTOL * scale, "status": "PASS" if ok else "FAIL"})
                    rec.expect(f"fd_vs_autograd_{k}_{state}_{s['name']}", ok_all, worst_err_over_bound=worst, scale=scale,
                               informative=scale >= 1e-6, g_autograd=g_ad[k].tolist(),
                               g_fd={f"{('u', 'ty', 'tx')[d]}_h{h:g}": fd[(k, d, h)] for d in range(3) for h in FD_STEPS})
                del tv, B
        with rec.group(f"fp32_diagnostic_{state}", required=False):
            # fp32 production dtype vs fp64 (diagnostic, not gating): same weights cast to fp32, host float32 chain
            for label, conv_prec in ((("default_precision", None), ("cudnn_conv_ieee", "ieee")) if dev.type == "cuda" else (("default_precision", None),)):
                with A.cudnn_conv_precision(conv_prec):
                    eU32, eV32, h32 = (copy.deepcopy(m).float() for m in (eU, eV, head))
                    DU32 = A.reference_descriptors(eU32, inputs, c0, torch.float32)
                    e32 = A.MFRigidEnergy(DU32, mov, rng(), eV=eV32, head=h32, estimator="moments", dtype=torch.float32)
                    for s in specs:
                        p32 = torch.tensor(s["p"], dtype=torch.float32, device=dev, requires_grad=True)
                        g32 = grads(e32(p32), [p32])[0]
                        p64 = torch.tensor(s["p"], dtype=dt, device=dev, requires_grad=True)
                        g64 = grads(e(p64), [p64])[0]
                        rec.cmp(f"fp32_vs_fp64_geometry_grad_{label}_{state}_{s['name']}", g32, g64, "float32", required=False,
                                note="diagnostic: production fp32 chain vs fp64 reference at the same p32-rounded weights; "
                                     "cudnn conv fp32_precision=%s" % A._safe(lambda: torch.backends.cudnn.conv.fp32_precision),
                                g32=g32.tolist(), g64=g64.tolist())
    extra["fd_rows"] = fd_rows
    extra["data_reads"] = guard.reads
    return rec.result(extra)


# ------------------------------------------------------------------------------------------------------------ part: host_replay
def part_host_replay(device: str, run_dir: Path, out_root: Path) -> dict:
    """The host trainer itself against the V10 trainer.  G.cmd_train (unchanged) runs in a subprocess that never imports
    v10.adapter (so the V9 writer tripwires are absent there), with V9_REG_ROOT redirected to <out-root>/T01/host_replay/<run>,
    for 2 steps (seed 901, fraction 1, eps 0, objective vcs) on CPU; then the V10 Trainer MLP_perm1 (seed 901, the V10 FIT set)
    runs 2 steps here.  The encoder and critic states after step 2 must agree at the fp32 design tolerance (bitwise recorded),
    and the 2-step update must exceed 100 x atol (power: a different batch, loss or optimiser would show)."""
    rec = Recorder("host_replay", device)
    uid = f"V10-T01-host-replay-vcs-frac1-eps0-s{HOST_REPLAY_SEED}"
    root = out_root / "T01" / "host_replay" / run_dir.name
    extra = {"uid": uid, "replay_root": str(root), "seed": HOST_REPLAY_SEED, "steps": HOST_REPLAY_STEPS}
    if torch.device(device).type != "cpu":
        rec.add("host_replay", "NOT_RUN", False, reason="CPU only (the host trainer uses CUDA whenever it is visible)")
        return rec.result(extra)
    ckpt_path = root / "units" / uid / "run" / f"ckpt_{HOST_REPLAY_STEPS:05d}.pt"
    with rec.group("host_trainer_subprocess"):
        if root.exists():
            raise RuntimeError(f"{root} exists: the host replay needs a fresh root")
        env = {**os.environ, "V9_REG_ROOT": str(root), "CUDA_VISIBLE_DEVICES": ""}
        cmd = [sys.executable, str(A.REPO / "scripts" / "v10" / "t01_host_replay.py"), "--root", str(root), "--uid", uid,
               "--seed", str(HOST_REPLAY_SEED), "--steps", str(HOST_REPLAY_STEPS)]
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
        (run_dir / "host_replay_stdout.log").write_text(proc.stdout)
        (run_dir / "host_replay_stderr.log").write_text(proc.stderr)
        lines = proc.stdout.strip().splitlines()
        info = json.loads(lines[-1]) if proc.returncode == 0 and lines else {}
        extra["host_subprocess"] = {"command": " ".join(cmd), "exit_code": proc.returncode, "seconds": time.time() - t0, "info": info}
        rec.expect("host_trainer_subprocess_ok", proc.returncode == 0, exit_code=proc.returncode, stderr_tail=proc.stderr[-3000:],
                   stdout_tail=proc.stdout[-1500:])
        rec.expect("host_trainer_is_the_pinned_unwrapped_host", info.get("host_code_hash") == A.HOST_CODE_HASH
                   and info.get("v10_adapter_imported") is False and info.get("cmd_train_is_tripwire") is False, value=info)
        files = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
        rec.expect("host_trainer_wrote_only_its_run_files_under_the_replay_root", info.get("v9_root") == str(root.resolve())
                   and files == [f"units/{uid}/run/ckpt_{HOST_REPLAY_STEPS:05d}.pt", f"units/{uid}/run/train.json"], files=files)
    if not ckpt_path.exists():
        rec.blocked("host_vs_v10_MLP_perm1_state_after_2_steps", f"host checkpoint {ckpt_path} missing (see the subprocess rows)")
        return rec.result(extra)
    with rec.group("v10_trainer_vs_host"):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        tj = json.loads((ckpt_path.parent / "train.json").read_text())
        guard = A.DataGuard()
        fit = A.fit_cores(guard, 1.0, HOST_REPLAY_SEED)
        rec.expect("host_fit_set_and_recipe_equal_v10", tj.get("fit") == fit and tj.get("steps") == HOST_REPLAY_STEPS and tj.get("objective") == "vcs"
                   and tj.get("fit_fraction") == 1.0 and tj.get("train_mismatch") == 0.0, n_fit=len(fit), host_code_hash_g1_internal=tj.get("code_hash"))
        imgs = A.load_learned_inputs(guard, fit, {"FIT"}, "T01 host replay: V10 trainer side (2 steps, as the host)", "cpu")
        t = A.Trainer(A.TrainConfig(method="MLP_perm1", seed=HOST_REPLAY_SEED), imgs, fit, "cpu")
        mods = (("eU", t.eU), ("eV", t.eV), ("critic", t.critic))
        init = {k: [v.detach().clone() for v in m.state_dict().values()] for k, m in mods}
        recs = [t.step() for _ in range(HOST_REPLAY_STEPS)]
        xs, ys, bit = [], [], True
        for key, mod in mods:
            sd = mod.state_dict()
            if list(sd) != list(ck[key]):
                raise RuntimeError(f"{key}: state_dict keys differ from the host checkpoint")
            for name, v in sd.items():
                xs.append(v)
                ys.append(ck[key][name])
                bit &= bool(torch.equal(v, ck[key][name]))
        change = max(float((v - i).abs().max()) for key, mod in mods for v, i in zip(mod.state_dict().values(), init[key]))
        rec.cmp_many("host_vs_v10_MLP_perm1_state_after_2_steps", xs, ys, "float32", bitwise=bit, update_max_abs=change)
        rec.expect("host_vs_v10_comparison_has_power", change > 100 * A.TOLERANCES["float32"][0], update_max_abs=change)
        rec.expect("host_vs_v10_state_bitwise", bit, required=False, note="bitwise equality recorded; the gate is the fp32 design tolerance")
        host_loss = [e["loss"] for e in tj.get("log", []) if e.get("step") == HOST_REPLAY_STEPS]
        rec.cmp(f"host_vs_v10_loss_at_step_{HOST_REPLAY_STEPS}", torch.tensor(recs[-1]["loss"], dtype=torch.float64),
                torch.tensor(host_loss[0] if host_loss else float("nan"), dtype=torch.float64), "float32",
                v10_loss=recs[-1]["loss"], host_loss=host_loss[0] if host_loss else None)
        extra["data_reads"] = guard.reads
        extra["host_side_reads"] = f"G.cmd_train read the {len(tj.get('fit', []))} FIT cores of the R6 cache itself (P.learned_inputs, read-only)"
        extra["host_train_json"] = {k: tj.get(k) for k in ("unit", "objective", "seed", "steps", "fit_fraction", "train_mismatch", "log", "seconds", "code_hash")}
    return rec.result(extra)


# ------------------------------------------------------------------------------------------------------------ run / merge
def parts_dir(out_root: Path) -> Path:
    return out_root / "T01" / "parts"


def run(args) -> int:
    pin = A.check_job_pin()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable: nothing run", flush=True)
        return 3
    devtype = torch.device(device).type
    label = os.environ.get("SLURM_JOB_ID") or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = args.out_root / "T01" / "runs" / f"{devtype}_{label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    env = A.environment(device)
    print(json.dumps({"t01_run": str(run_dir), **{k: env[k] for k in ("host", "git_commit", "code_hash_src_scripts_v10", "device", "torch")},
                      "gpu_name": env.get("gpu_name"), "job_pin": pin}), flush=True)
    print(json.dumps({"precision_record": env["precision"]}, default=str), flush=True)
    settings0 = A.precision_settings(env["precision"])
    n_sc = {"float64": args.n_sc_fp64 if args.n_sc_fp64 else (4 if devtype == "cpu" else 16), "float32": args.n_sc_fp32}
    rc = 0
    for part in [p for p in args.parts.split(",") if p]:
        t0 = time.time()
        try:
            if part == "verify_torch":
                res = part_verify_torch(device, run_dir)
            elif part == "core_tol":
                res = part_core_tol(device, run_dir)
            elif part == "adapter":
                res = part_adapter(device, run_dir, n_sc)
            elif part == "geometry":
                res = part_geometry(device, run_dir)
            elif part == "host_replay":
                res = part_host_replay(device, run_dir, args.out_root)
            else:
                raise ValueError(f"unknown part {part}")
        except Exception as e:  # noqa: BLE001
            res = {"part": part, "device": devtype, "status": "FAIL", "failure_kind": "exception", "error": repr(e)[:2000],
                   "traceback": traceback.format_exc()[-8000:], "checks": []}
        prec_after = A.precision_record(device)
        res.update({"provenance": env, "job_pin": pin, "run_dir": str(run_dir), "wall_seconds": time.time() - t0, "device_type": devtype,
                    "precision_after_part": prec_after, "precision_settings_unchanged": A.precision_settings(prec_after) == settings0})
        A.write_json_atomic(run_dir / f"{part}_{devtype}.json", res)
        A.write_json_atomic(parts_dir(args.out_root) / f"{part}_{devtype}.json", res)
        print(f"[t01] part {part} on {devtype}: {res['status']} ({res.get('n_pass', 0)} pass / {res.get('n_fail', 0)} fail / "
              f"{res.get('n_blocked', 0)} blocked, {time.time() - t0:.0f} s) required failures: {res.get('required_failures', [res.get('error')])} "
              f"blocked: {res.get('required_blocked', [])}", flush=True)
        rc = rc or (0 if res["status"] == "PASS" else 1)
    return rc


def _summary(part: dict | None) -> dict:
    if part is None:
        return {"status": "NOT_RUN"}
    return part


def overall_status(parts: dict) -> str:
    st = [p["status"] for p in parts.values()]
    if any(s == "FAIL" for s in st):
        return "FAIL"
    if any(s == "BLOCKED" for s in st):
        return "BLOCKED"
    if st and all(s == "PASS" for s in st):
        return "PASS"
    return "NOT_RUN"


def provenance_check(parts: dict) -> tuple[dict, list]:
    """Every part that ran must carry the code identity of the CURRENT tree (code hash of src/v10 + scripts/v10, host code hash,
    backend sha256) and a clean tree; commits that touch only results / docs do not change this identity."""
    current = {"code_hash_src_scripts_v10": A.code_hash(), "host_code_hash": A.host_code_hash(), "mf_vcs_sha256": A.sha256_file(MF.__file__)}
    rows, mismatches = {}, []
    for key, p in parts.items():
        if p.get("status") == "NOT_RUN" and "provenance" not in p:
            continue
        pv = p.get("provenance") or {}
        rows[key] = {k: pv.get(k) for k in ("git_commit", "submit_commit", "slurm_job_id", "slurm_partition", "host", "gpu_name", "utc",
                                            "job_pin_expected", "job_code_hash", "git_dirty_code", *PROVENANCE_KEYS)}
        prec = pv.get("precision")                     # decision 2026-10-17 section 3: recorded, not gating (absent in phase-1 parts)
        rows[key]["precision"] = None if prec is None else {"settings": A.precision_settings(prec), "env": prec.get("env"),
                                                            "gpu_capability": prec.get("gpu_capability"),
                                                            "unchanged_during_part": p.get("precision_settings_unchanged")}
        bad = [k for k in PROVENANCE_KEYS if pv.get(k) != current[k]] + (["git_dirty_code"] if pv.get("git_dirty_code") else [])
        if bad:
            mismatches.append({"part": key, "fields": bad, **{k: pv.get(k) for k in bad}})
    return {"current": current, "parts": rows}, mismatches


def merge(args) -> int:
    A.check_job_pin()
    pdir = parts_dir(args.out_root)
    load = lambda name, dev: json.loads((pdir / f"{name}_{dev}.json").read_text()) if (pdir / f"{name}_{dev}.json").exists() else None  # noqa: E731
    torch_parts = {f"{n}_{d}": _summary(load(n, d)) for n, d in EXPECTED_TORCH_PARTS}
    geom_parts = {f"{n}_{d}": _summary(load(n, d)) for n, d in EXPECTED_GEOMETRY_PARTS}
    allp = {**torch_parts, **geom_parts}
    prov, mismatches = provenance_check(allp)
    verified = all(p["status"] == "PASS" for p in allp.values()) and not mismatches
    common = {"generated_utc": A.utc_now(), "generator": "scripts/v10/t01_validate.py merge", "parts_dir": str(pdir),
              "status_vocabulary": "PASS / FAIL / BLOCKED / NOT_RUN (missing part file = NOT_RUN); a check-group exception is a FAIL row",
              "host_adapter_verified": verified,
              "host_adapter_verified_rule": "true only when every expected part (verify_torch, core_tol, adapter on CPU and CUDA; host_replay "
                                            "on CPU; geometry on CPU and CUDA) is PASS AND every part ran the code of the current tree "
                                            "(code hash of src/v10 + scripts/v10, host code hash, mf_vcs sha256) from a clean tree",
              "provenance": prov, "provenance_mismatches": mismatches,
              "pending_parts": [k for k, p in allp.items() if p["status"] == "NOT_RUN"],
              "failed_parts": [k for k, p in allp.items() if p["status"] == "FAIL"],
              "blocked_parts": [k for k, p in allp.items() if p["status"] == "BLOCKED"],
              "tolerances_design": {k: {"atol": v[0], "rtol": v[1]} for k, v in A.TOLERANCES.items()}}
    tv = {"schema": "V10-T01-TORCH_VALIDATION-v2", "overall_status": overall_status(torch_parts), **common, "parts": torch_parts}
    gg = {"schema": "V10-T01-GEOMETRY_GRADIENT_CHECK-v2", "overall_status": overall_status(geom_parts), **common,
          "method": {"transforms": [t[0] for t in GEOMETRY_TRANSFORMS], "fd_steps": FD_STEPS, "fd_atol": FD_ATOL, "fd_rtol": FD_RTOL,
                     "kink_rule": "warped BOX1 grid on a (1/L)-lattice at distance 1/(2L) from every integer coordinate; zero floor "
                                  "changes of any warped coordinate over [p - h e, p + h e] verified for every direction and step"},
          "parts": geom_parts}
    args.repo_out.mkdir(parents=True, exist_ok=True)
    A.write_json_atomic(args.repo_out / "TORCH_VALIDATION.json", tv)
    A.write_json_atomic(args.repo_out / "GEOMETRY_GRADIENT_CHECK.json", gg)
    print(json.dumps({"TORCH_VALIDATION": tv["overall_status"], "GEOMETRY_GRADIENT_CHECK": gg["overall_status"],
                      "host_adapter_verified": verified, "pending": common["pending_parts"], "failed": common["failed_parts"],
                      "blocked": common["blocked_parts"], "provenance_mismatches": mismatches}), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--device", choices=("cpu", "cuda"), required=True)
    r.add_argument("--parts", default="verify_torch,core_tol,adapter,geometry,host_replay")
    r.add_argument("--out-root", type=Path, default=A.OUT_ROOT)
    r.add_argument("--n-sc-fp64", type=int, default=0, help="scenes in the fp64 real batch (default 4 on CPU, 16 on CUDA)")
    r.add_argument("--n-sc-fp32", type=int, default=16, help="scenes in the fp32 real batch (host: 16)")
    m = sub.add_parser("merge")
    m.add_argument("--out-root", type=Path, default=A.OUT_ROOT)
    m.add_argument("--repo-out", type=Path, default=REPO_T01)
    args = ap.parse_args()
    return run(args) if args.cmd == "run" else merge(args)


if __name__ == "__main__":
    sys.exit(main())
