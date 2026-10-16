"""V10 T01 - mathematics and real-host adapter validation (package V10_DESIGN 6.1, docs/HOST_ADAPTER_CN.md).

Parts (each writes <out-root>/T01/parts/<part>_<cpu|cuda>.json and a copy in its run directory):
  verify_torch  the package core/verify_torch.py, unchanged, as a subprocess next to an unchanged copy of mf_vcs.py
                (sha256 of both checked against the package MANIFEST_SHA256.json); its own internal limits apply.
  core_tol      the same moment / dense / HVP / permutation identities on random simplex features with the package's design
                tolerances (allclose: fp64 atol 1e-10 rtol 1e-8, fp32 atol 1e-5 rtol 1e-4), reference magnitudes recorded.
  adapter       acceptance on the REAL host: a real BF-SHG FIT batch through the host encoders (dense vs moments values and
                A / B / W gradients, unequal-n scene weights, Q modes, M1 strict / training modes, MLP perm1 / perm8 / dense),
                one-update checks, the real warp chain (batched energy vs the package backend, fixed-W state hash across
                geometry candidates, checkpoint save / restore).
  geometry      fp64 central finite differences of every energy term (P: p1, p2; Q: q1, q2 by moments, dense and perm1)
                w.r.t. (u, ty, tx) on the real warp chain at two non-zero rigid transforms whose warped sampling grid lies
                on a rational lattice (Pythagorean rotation) at distance >= 1/(2L) from every bilinear kink; two step sizes;
                zero kink crossings verified for every step.  Plus a non-gating fp32-vs-fp64 gradient diagnostic.
  merge         -> v10/T01/TORCH_VALIDATION.json and v10/T01/GEOMETRY_GRADIENT_CHECK.json from whatever parts exist
                (missing parts are NOT_RUN); host_adapter_verified only when every expected part on CPU and CUDA is PASS.
Data: BF-SHG FIT cores only, through adapter.DataGuard (V10 manifest include rows); T01 never reads TEST, CAL or DEV.
Statuses: PASS / FAIL / NOT_RUN per check; an exception inside a check group is a FAIL row with its traceback.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
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
                        ("adapter", "cpu"), ("adapter", "cuda"))
EXPECTED_GEOMETRY_PARTS = (("geometry", "cpu"), ("geometry", "cuda"))


# ------------------------------------------------------------------------------------------------------------ recorder
class Recorder:
    def __init__(self, part: str, device: str):
        self.part, self.device, self.rows = part, device, []
        self.t0, self.started = time.time(), A.utc_now()

    def add(self, name, status, required=True, **fields):
        row = {"name": name, "status": status, "required": bool(required), **fields}
        self.rows.append(row)
        brief = {k: fields[k] for k in ("max_abs_err", "worst_err_over_bound", "value", "error") if k in fields}
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
        fails = [r["name"] for r in req if r["status"] != "PASS"]
        status = "PASS" if req and not fails else "FAIL"
        out = {"part": self.part, "device": self.device, "status": status, "n_checks": len(self.rows), "n_required": len(req),
               "n_pass": sum(r["status"] == "PASS" for r in self.rows), "n_fail": sum(r["status"] == "FAIL" for r in self.rows),
               "n_not_run": sum(r["status"] == "NOT_RUN" for r in self.rows), "required_failures": fails,
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


def grads(out, inputs, retain=True):
    """Gradients of a scalar w.r.t. inputs; an input the output does not depend on gets an explicit zero tensor."""
    inputs = list(inputs)
    gs = torch.autograd.grad(out, inputs, retain_graph=retain, allow_unused=True)
    return [torch.zeros_like(x) if g is None else g for g, x in zip(gs, inputs)]


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
def part_core_tol(device: str, run_dir: Path) -> dict:
    rec = Recorder("core_tol", device)
    dev = torch.device(device)
    torch.manual_seed(20261016)

    def feats(n, k, dtype):
        z = torch.randn(n, k, device=dev, dtype=dtype, requires_grad=True)
        return z, z.softmax(-1)

    for dtype in (torch.float64, torch.float32):
        dn = dtype_name(dtype)
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
                        rec.cmp(f"head_value_{tag0}_ridge{ridge}", hv, exact.f, dn)
                        rec.cmp(f"head_gradient_{tag0}_ridge{ridge}", hg, tg, dn)
                        rec.cmp(f"head_HVP_{tag0}_ridge{ridge}", head.hvp(direction), thv, dn)
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
    return rec.result({"tolerances": A.TOLERANCES})


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
def part_adapter(device: str, run_dir: Path, n_sc_by_dtype: dict) -> dict:
    rec = Recorder("adapter", device)
    dev = torch.device(device)
    guard = A.DataGuard()
    fit = []
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
            rec.expect("host_batch_rng_equals_T00_independent_replay", all(stats[k] == ref[k] for k in keys), value=stats, reference=ref,
                       note="T00 audit replayed G.cmd_train's numpy draws independently (seed 901, fraction 1, 200 steps)")
        else:
            rec.add("host_batch_rng_equals_T00_independent_replay", "NOT_RUN", False, reason=f"{T00_RNG_EVIDENCE} missing", value=stats)

    for dn in ("float64", "float32"):
        dtype, n_sc = getattr(torch, dn), int(n_sc_by_dtype[dn])
        ctx = None
        with rec.group(f"real_batch_{dn}"):
            hb = A.HostBatchSampler(fit, SEED, n_sc=n_sc).draw()
            imgs = A.load_learned_inputs(guard, hb.cores, {"FIT"}, f"T01 adapter real FIT batch ({dn})", dev)
            eU, eV, cr, head = build_models_seeded(SEED, dev, dtype)
            bu, bv = A.assemble_crops(imgs, hb)
            du, dv = A.batch_descriptors(eU, eV, bu.to(dtype), bv.to(dtype), hb.yx)
            del imgs, bu, bv
            n_pos = hb.perm.shape[1]
            Af, Bf = head.features(du.reshape(-1, A.DESC), dv.reshape(-1, A.DESC))
            ctx = {"hb": hb, "eU": eU, "eV": eV, "cr": cr, "head": head, "du": du, "dv": dv, "Af": Af, "Bf": Bf, "n_pos": n_pos,
                   "scenes": A.mf_scenes(Af, Bf, n_sc, n_pos), "wts": A.host_weights(n_sc), "n_sc": n_sc}
            rec.expect(f"real_descriptors_finite_{dn}", bool(torch.isfinite(du).all() and torch.isfinite(dv).all()),
                       du_std=float(du.std()), dv_std=float(dv.std()), n_sc=n_sc, n_pos=n_pos, cores=hb.cores)
            rec.expect(f"real_features_nonconstant_{dn}", float(Af.std(0).max()) > 0 and float(Bf.std(0).max()) > 0
                       and len(torch.unique(Af.detach(), dim=0)) > 1 and len(torch.unique(Bf.detach(), dim=0)) > 1,
                       a_std_max=float(Af.std(0).max()), b_std_max=float(Bf.std(0).max()),
                       a_range=[float(Af.min()), float(Af.max())], b_range=[float(Bf.min()), float(Bf.max())])
            rec.expect(f"scene_order_is_crop_order_{dn}", [s_.scene_id for s_ in ctx["scenes"]] == [str(i) for i in range(n_sc)])
        if ctx is None:
            rec.add(f"real_batch_checks_{dn}", "NOT_RUN", True, reason="the real batch could not be built (see the exception row)")
        else:
            for name, fn in (("moments_vs_dense", real_moments_vs_dense), ("unequal_n_and_partition", real_unequal_n),
                             ("q_modes", real_q_modes), ("m1", real_m1), ("mlp", real_mlp)):
                with rec.group(f"real_{name}_{dn}"):
                    fn(rec, ctx, dev, dn)
            del ctx

        with rec.group(f"registration_side_{dn}"):
            reg_side_checks(rec, guard, fit, dev, dtype, dn, run_dir)

    with rec.group("one_update_float32"):
        one_update_checks(rec, guard, fit, dev, int(n_sc_by_dtype["float32"]))
    return rec.result({"data_reads": guard.reads, "seed": SEED, "K": K_SMOKE, "n_sc_by_dtype": n_sc_by_dtype,
                       "geometry_transforms": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items()} for s in geometry_specs()]})


def real_moments_vs_dense(rec: Recorder, c: dict, dev, dn: str):
    Af, Bf, W, scenes, wts = c["Af"], c["Bf"], c["head"].W, c["scenes"], c["wts"]
    m = MF.objective_terms(scenes, W, weights=wts)
    d = MF.dense_objective_terms(scenes, W, weights=wts)
    rec.cmp_many(f"real_moments_vs_dense_terms_{dn}", [m.p1, m.p2, m.q1, m.q2, m.j], [d.p1, d.p2, d.q1, d.q2, d.j], dn,
                 terms="p1,p2,q1,q2,J", J_moments=float(m.j), J_dense=float(d.j))
    leaves = [Af, Bf, W]
    gm, gd = grads(m.j, leaves), grads(d.j, leaves)
    for lbl, x, y in zip(("A", "B", "W"), gm, gd):
        rec.cmp(f"real_moments_vs_dense_grad_{lbl}_{dn}", x, y, dn)
    enc = [p for mod in (c["eU"], c["eV"], c["head"].left, c["head"].right) for p in mod.parameters()]
    rec.cmp_many(f"real_moments_vs_dense_grad_encoders_featuremaps_{dn}", grads(m.j, enc), grads(d.j, enc), dn,
                 required=(dn == "float64"),
                 note="end-to-end; gating in fp64; fp32 is a diagnostic (fp32 GPU convolutions run in TF32 by default)")
    dc = MF.dense_objective_terms(scenes, W, weights=wts, chunk_size=32, checkpoint_blocks=True)
    rec.cmp(f"real_moments_vs_dense_chunked_checkpointed_value_{dn}", m.j, dc.j, dn)
    rec.cmp_many(f"real_moments_vs_dense_chunked_checkpointed_grads_{dn}", gm, grads(dc.j, leaves), dn)
    s5 = MF.SceneBatch(Af[:5], Bf[:5], "v")                          # all ordinary permutations of a real 5-row subset average to V
    perms = torch.tensor(list(itertools.permutations(range(5))), device=dev, dtype=torch.long)
    samp = MF.sampled_permutation_terms([s5], W, weights=[1.0], permutations=[perms])
    full = MF.objective_terms([s5], W, weights=[1.0])
    rec.cmp(f"real_all_120_permutations_mean_equals_V_value_{dn}", samp.j, full.j, dn)
    rec.cmp_many(f"real_all_120_permutations_mean_equals_V_grads_{dn}", grads(samp.j, leaves), grads(full.j, leaves), dn)


def real_unequal_n(rec: Recorder, c: dict, dev, dn: str):
    Af, Bf, W, n_pos = c["Af"], c["Bf"], c["head"].W, c["n_pos"]
    leaves = [Af, Bf, W]
    sizes, locked = [n_pos, 57, 9, 1], [0.4, 0.3, 0.2, 0.1]
    rows = [slice(s * n_pos, s * n_pos + sizes[s]) for s in range(4)]
    fx = [MF.SceneBatch(Af[rows[s]], Bf[rows[s]], "v", scene_id=str(s)) for s in range(4)]
    mu, dU = MF.objective_terms(fx, W, weights=locked), MF.dense_objective_terms(fx, W, weights=locked)
    rec.cmp_many(f"unequal_n_moments_vs_dense_terms_{dn}", [mu.p1, mu.p2, mu.q1, mu.q2, mu.j], [dU.p1, dU.p2, dU.q1, dU.q2, dU.j], dn,
                 sizes=sizes, weights=locked)
    rec.cmp_many(f"unequal_n_moments_vs_dense_grads_ABW_{dn}", grads(mu.j, leaves), grads(dU.j, leaves), dn)
    explicit = sum(wt * MF.objective_terms([fx[s]], W, weights=[1.0]).j for s, wt in enumerate(locked))
    rec.cmp(f"unequal_n_locked_weights_equal_explicit_scene_sum_{dn}", mu.j, explicit, dn)
    atol, rtol = A.TOLERANCES[dn]
    tol = atol + rtol * abs(float(mu.j))
    jpos = MF.objective_terms(fx, W, weights=[n_ / sum(sizes) for n_ in sizes]).j
    rec.expect(f"unequal_n_position_weighting_detectable_{dn}", abs(float(jpos - mu.j)) > 100 * tol, value=float(jpos - mu.j), tol=tol)
    pooled = MF.SceneBatch(torch.cat([Af[r] for r in rows]), torch.cat([Bf[r] for r in rows]), "v")
    jpool = MF.objective_terms([pooled], W, weights=[1.0]).j
    rec.expect(f"unequal_n_cross_scene_pooling_detectable_{dn}", abs(float(jpool - mu.j)) > 100 * tol, value=float(jpool - mu.j), tol=tol)
    t1 = (Af[rows[3]] @ W * Bf[rows[3]]).sum()
    rec.cmp(f"V_scene_n1_allowed_J_equals_minus_t2_{dn}", MF.objective_terms([fx[3]], W, weights=[1.0]).j, -t1 ** 2, dn)
    ids = torch.cat([torch.full((sizes[s],), v, device=dev, dtype=torch.long) for s, v in enumerate((5, 2, 9, 7))])
    Ar, Br = torch.cat([Af[r] for r in rows]), torch.cat([Bf[r] for r in rows])
    shuffle = torch.randperm(len(ids), generator=torch.Generator(device="cpu").manual_seed(SEED)).to(dev)
    part = MF.partition_scenes(Ar[shuffle], Br[shuffle], ids[shuffle], q_mode="v")
    rec.expect(f"partition_scenes_sorted_id_order_{dn}", [s.scene_id for s in part] == ["2", "5", "7", "9"]
               and [len(s.a) for s in part] == [57, n_pos, 1, 9], value=[(s.scene_id, len(s.a)) for s in part])
    rec.cmp(f"partition_scenes_weights_follow_sorted_order_{dn}", MF.objective_terms(part, W, weights=[0.3, 0.4, 0.1, 0.2]).j, mu.j, dn)


def real_q_modes(rec: Recorder, c: dict, dev, dn: str):
    Af, Bf, W, n_pos = c["Af"], c["Bf"], c["head"].W, c["n_pos"]
    leaves = [Af, Bf, W]
    atol, rtol = A.TOLERANCES[dn]
    rec.raises(f"U_mode_n1_rejected_{dn}", lambda: MF.SceneBatch(Af[:1], Bf[:1], "u"), ValueError)
    su, sv = MF.SceneBatch(Af[:n_pos], Bf[:n_pos], "u"), MF.SceneBatch(Af[:n_pos], Bf[:n_pos], "v")
    um, ud = MF.objective_terms([su], W, weights=[1.0]), MF.dense_objective_terms([su], W, weights=[1.0])
    rec.cmp_many(f"U_mode_moments_vs_dense_terms_{dn}", [um.q1, um.q2, um.j], [ud.q1, ud.q2, ud.j], dn)
    rec.cmp_many(f"U_mode_moments_vs_dense_grads_ABW_{dn}", grads(um.j, leaves), grads(ud.j, leaves), dn)
    vm = MF.objective_terms([sv], W, weights=[1.0])
    rec.expect(f"U_differs_from_V_{dn}", abs(float(um.j - vm.j)) > 100 * (atol + rtol * abs(float(vm.j))), value=float(um.j - vm.j))
    si = MF.SceneBatch(Af[:64], Bf[:64], "independent", Af[64:n_pos], Bf[64:n_pos])
    im, idn = MF.objective_terms([si], W, weights=[1.0]), MF.dense_objective_terms([si], W, weights=[1.0])
    rec.cmp_many(f"independent_pool_moments_vs_dense_{dn}", [im.q1, im.q2, im.j], [idn.q1, idn.q2, idn.j], dn)
    rec.cmp_many(f"independent_pool_moments_vs_dense_grads_ABW_{dn}", grads(im.j, leaves), grads(idn.j, leaves), dn)


def real_m1(rec: Recorder, c: dict, dev, dn: str):
    Af, Bf, W, scenes, wts = c["Af"], c["Bf"], c["head"].W, c["scenes"], c["wts"]
    tol_strict, iters_strict = (1e-9, 30000) if dn == "float64" else (1e-4, 5000)
    try:                                                             # strict mode keeps require_converged=True (package default)
        prof = MF.profiled_objective(scenes, weights=wts, ridge=0.0, initial=W.detach(), tolerance=tol_strict, max_iterations=iters_strict)
    except RuntimeError as e:
        rec.add(f"M1_strict_converged_{dn}", "FAIL", True, failure_kind="not_converged", error=str(e)[:500], tolerance=tol_strict,
                max_iterations=iters_strict)
        prof = None
    if prof is not None:
        sol = prof.solution
        rec.expect(f"M1_strict_converged_{dn}", sol.converged and sol.fw_gap <= tol_strict, gap=sol.fw_gap, iterations=sol.iterations,
                   tolerance=tol_strict)
        rec.expect(f"M1_strict_W_in_box_and_detached_{dn}", float(sol.w.abs().max()) <= 1.0 and not sol.w.requires_grad,
                   w_absmax=float(sol.w.abs().max()), n_active=int((sol.w.abs() == 1).sum()))
        manual = MF.objective_terms(scenes, sol.w.detach(), weights=wts)
        g1, g2 = grads(prof.terms.f, [Af, Bf]), grads(manual.f, [Af, Bf])
        rec.expect(f"M1_outer_grad_equals_fixed_W_partial_bitwise_{dn}", all(torch.equal(x, y) for x, y in zip(g1, g2)))
        rec.cmp(f"M1_outer_F_equals_inner_F_{dn}", prof.terms.f, torch.tensor(sol.f, dtype=torch.float64), dn)
    rec.raises(f"M1_strict_incomplete_solve_rejected_{dn}",
               lambda: MF.profiled_objective(scenes, weights=wts, max_iterations=0, tolerance=1e-300), RuntimeError)
    pt = MF.profiled_objective(scenes, weights=wts, initial=W.detach(), tolerance=1e-4, max_iterations=10, require_converged=False)
    rec.expect(f"M1_training_mode_returns_and_logs_gap_{dn}", math.isfinite(pt.solution.fw_gap) and pt.solution.iterations <= 10,
               gap=pt.solution.fw_gap, iterations=pt.solution.iterations, converged=pt.solution.converged, status=pt.solution.status)


def real_mlp(rec: Recorder, c: dict, dev, dn: str):
    du, dv, cr, hb, n_sc = c["du"], c["dv"], c["cr"], c["hb"], c["n_sc"]
    perm = torch.as_tensor(hb.perm, device=dev)
    dvq = torch.stack([dv[s, perm[s]] for s in range(n_sc)])                       # host G L171-173 transcribed
    host_loss = A.P.loss_fn(torch, F, "vcs", cr(torch.cat([du, dv], -1)).squeeze(-1), cr(torch.cat([du, dvq], -1)).squeeze(-1))
    ad_loss, _ = A.mlp_objective("MLP_perm1", cr, du, dv, perm=perm)
    rec.cmp(f"MLP_perm1_adapter_equals_host_formula_{dn}", ad_loss, host_loss, dn, bitwise=bool(torch.equal(host_loss, ad_loss)))
    crp = list(cr.parameters())
    perms = torch.tensor(list(itertools.permutations(range(5))), device=dev, dtype=torch.long)
    du5, dv5 = du[:1, :5], dv[:1, :5]
    _, jd5 = A.mlp_objective("MLP_dense", cr, du5, dv5)
    jp5 = torch.stack([A.mlp_objective("MLP_perm1", cr, du5, dv5, perm=p[None])[1] for p in perms]).mean()
    rec.cmp(f"MLP_dense_equals_all_120_perm1_mean_value_{dn}", jp5, jd5, dn)
    rec.cmp_many(f"MLP_dense_equals_all_120_perm1_mean_grads_{dn}", grads(jp5, [du, dv] + crp), grads(jd5, [du, dv] + crp), dn)
    _, jfull = A.mlp_objective("MLP_dense", cr, du, dv)
    _, jblk = A.mlp_objective("MLP_dense", cr, du, dv, block=32, checkpoint=True)
    rec.cmp(f"MLP_dense_blocked_checkpointed_vs_full_value_{dn}", jblk, jfull, dn)
    rec.cmp_many(f"MLP_dense_blocked_checkpointed_vs_full_grads_{dn}", grads(jblk, [du, dv] + crp), grads(jfull, [du, dv] + crp), dn)
    _, j8same = A.mlp_objective("MLP_perm8", cr, du, dv, perms=perm[:, None].expand(-1, 8, -1).contiguous())
    rec.cmp(f"MLP_perm8_normalisation_equals_perm1_when_rows_repeat_{dn}", j8same, -ad_loss, dn)


def reg_side_checks(rec: Recorder, guard, fit, dev, dtype, dn: str, run_dir: Path):
    c0 = fit[0]
    pert = {c0: A.Q.perturbations_p2(c0)}                    # REG-P2's rule applied to a FIT core: no CAL / DEV / TEST read
    inputs = A.GuardedInputs(guard, {"FIT"}, f"T01 registration-side ({dn})", pert, dev)
    eU, eV, cr, head = build_models_seeded(SEED, dev, dtype)
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
    rec.expect(f"registration_positions_and_permutation_are_host_{dn}", torch.equal(e_probe.perm, host.perm)
               and torch.equal(e_probe.x.to(torch.float32), host.x), n_pos=int(len(host.x)))
    values = {}
    for spec in specs:
        name = spec["name"]
        e = energy("moments")
        p = torch.tensor(spec["p"], dtype=dtype, device=dev, requires_grad=True)
        Ev = e(p)
        dv_host, B = e.last["dv"][0], e.last["B"][0]
        dv_explicit = A.host_moving_descriptors(e, p)
        rec.cmp(f"host_chain_dv_equals_explicit_chain_{name}_{dn}", dv_host, dv_explicit, dn, bitwise=bool(torch.equal(dv_host, dv_explicit)))
        sc = MF.SceneBatch(e.A, B, "v")
        pm = MF.objective_terms([sc], head.W, weights=[1.0])
        pdn = MF.dense_objective_terms([sc], head.W, weights=[1.0], chunk_size=1024)
        rec.cmp(f"registration_batched_vs_package_moments_value_{name}_{dn}", Ev, pm.j, dn)
        rec.cmp_many(f"registration_package_moments_vs_dense_terms_{name}_{dn}", [pm.p1, pm.p2, pm.q1, pm.q2, pm.j],
                     [pdn.p1, pdn.p2, pdn.q1, pdn.q2, pdn.j], dn)
        g_rows, g_pm, g_pd = grads(Ev, [p])[0], grads(pm.j, [p])[0], grads(pdn.j, [p])[0]
        rec.cmp(f"registration_geometry_grad_batched_vs_package_moments_{name}_{dn}", g_rows, g_pm, dn)
        rec.cmp(f"registration_geometry_grad_moments_vs_dense_{name}_{dn}", g_pm, g_pd, dn, grad=g_pm.tolist())
        rec.expect(f"registration_geometry_grad_finite_nonzero_{name}_{dn}", bool(torch.isfinite(g_rows).all()) and float(g_rows.abs().max()) > 0,
                   grad=g_rows.tolist())
        e1 = energy("perm1")
        p1 = torch.tensor(spec["p"], dtype=dtype, device=dev, requires_grad=True)
        E1 = e1(p1)
        pp = MF.sampled_permutation_terms([MF.SceneBatch(e1.A, e1.last["B"][0], "v")], head.W, weights=[1.0], permutations=[e1.perm[None]])
        rec.cmp(f"registration_batched_vs_package_perm1_value_{name}_{dn}", E1, pp.j, dn)
        rec.cmp(f"registration_batched_vs_package_perm1_geometry_grad_{name}_{dn}", grads(E1, [p1])[0], grads(pp.j, [p1])[0], dn)
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
    rec.expect(f"fixed_W_state_hash_unchanged_across_candidates_{dn}", h0 == h1, before=h0, after=h1, n_candidates=n_cand, full_search=full_search)
    mods = (eU, eV, head)
    rec.expect(f"test_time_params_frozen_eval_no_grad_{dn}", all(not m.training for m in mods)
               and all((not q.requires_grad) and q.grad is None for m in mods for q in m.parameters()))
    rec.expect(f"test_time_geometry_grads_finite_nonzero_{dn}", all(math.isfinite(g) and g > 0 for g in gnorms), grad_absmax=gnorms)
    trainable = copy.deepcopy(eV).train()
    for q in trainable.parameters():
        q.requires_grad_(True)
    rec.raises(f"test_time_energy_refuses_trainable_models_{dn}", lambda: energy("moments", eV_=trainable), ValueError)
    # checkpoint save / restore of the indivisible (encoders, feature maps, W) pair
    path = run_dir / f"smoke_pair_{dn}.pt"
    sha = A.save_pair(path, eU, eV, head, {"purpose": "T01 smoke checkpoint round trip", "seed": SEED, "dtype": dn, "K": K_SMOKE})
    eU2, eV2, head2, meta = A.load_pair(path, dev, dtype)
    A.freeze_for_test(eU2, eV2, head2)
    rec.expect(f"checkpoint_roundtrip_state_hash_equal_{dn}", A.state_hash(eU2, eV2, head2) == h0, sha256=sha, bytes=path.stat().st_size, meta=meta)
    DU2 = A.reference_descriptors(eU2, inputs, c0, dtype)
    for spec in specs:
        e2 = energy("moments", eV_=eV2, head_=head2, DU_=DU2)
        p = torch.tensor(spec["p"], dtype=dtype, device=dev, requires_grad=True)
        E2 = e2(p)
        g2 = grads(E2, [p])[0]
        v0, g0 = values[spec["name"]]
        rec.cmp(f"checkpoint_roundtrip_energy_value_{spec['name']}_{dn}", E2, torch.tensor(v0, dtype=torch.float64), dn,
                bitwise=float(E2) == v0)
        rec.cmp(f"checkpoint_roundtrip_geometry_grad_{spec['name']}_{dn}", g2, g0, dn, bitwise=bool(torch.equal(g2, g0)))


def one_update_checks(rec: Recorder, guard, fit, dev, n_sc: int):
    first = A.HostBatchSampler(fit, SEED, n_sc=n_sc).draw()
    imgs = A.load_learned_inputs(guard, first.cores, {"FIT"}, "T01 one-update checks (float32)", dev)

    def trainer(method, **kw):
        return A.Trainer(A.TrainConfig(method=method, seed=SEED, n_sc=n_sc, **kw), imgs, fit, dev)

    def enc_params(t):
        return [p for m in (t.eU, t.eV) for p in m.parameters()]

    def grad_norm(params):
        gs = [p.grad for p in params if p.grad is not None]
        return float(torch.sqrt(sum((g.double() ** 2).sum() for g in gs))) if gs else 0.0

    def max_change(before, params):
        return max(float((p.detach() - b).abs().max()) for p, b in zip(params, before))

    tA, tB = trainer("MF_perm1"), trainer("MF_M0")
    bA, bB = tA.draw(), tB.draw()
    rec.expect("same_first_batch_across_methods", bA[0].cores == bB[0].cores == first.cores and np.array_equal(bA[0].yx, bB[0].yx)
               and np.array_equal(bA[0].perm, bB[0].perm) and torch.equal(bA[1], bB[1]))
    rec.expect("MF_init_W_zero_package_default", float(tA.head.W.abs().max()) == 0.0 and float(tB.head.W.abs().max()) == 0.0)
    _, oA = tA.forward(*bA)
    _, oB = tB.forward(*bB)
    rec.cmp_many("MF_perm1_vs_M0_same_positive_terms", [oA["terms"].p1, oA["terms"].p2], [oB["terms"].p1, oB["terms"].p2], "float32",
                 bitwise=bool(torch.equal(oA["terms"].p1, oB["terms"].p1) and torch.equal(oA["terms"].p2, oB["terms"].p2)))
    rec.expect("MF_perm1_vs_M0_same_scene_order", [s.scene_id for s in oA["scenes"]] == [s.scene_id for s in oB["scenes"]] == [str(i) for i in range(n_sc)])
    del oA, oB
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
        # boundary variant: W := the box-QP head of this batch (active entries have outward gradients by KKT), then one step
        t = trainer(method)
        batch = t.draw()
        with torch.no_grad():
            _, out = t.forward(*batch)
            prof = MF.profiled_objective(out["scenes"], weights=A.host_weights(n_sc), tolerance=1e-4, max_iterations=2000, require_converged=False)
            t.head.W.copy_(prof.solution.w)
        n_active = int((t.head.W.abs() == 1).sum())
        before_enc = [p.detach().clone() for p in enc_params(t)]
        before_fm = [p.detach().clone() for p in list(t.head.left.parameters()) + list(t.head.right.parameters())]
        r = t.step(batch, keep=True)
        rec.expect(f"{method}_W_projection_active_and_feasible", r["w_absmax_preprojection"] > 1.0 and r["w_absmax"] <= 1.0,
                   w_absmax_preprojection=r["w_absmax_preprojection"], w_absmax=r["w_absmax"], n_active_before=n_active,
                   gap_of_start=prof.solution.fw_gap)
        rec.expect(f"{method}_encoder_grad_nonzero_and_params_change", grad_norm(enc_params(t)) > 0 and max_change(before_enc, enc_params(t)) > 0,
                   encoder_grad_norm_postclip=grad_norm(enc_params(t)), encoder_max_change=max_change(before_enc, enc_params(t)),
                   grad_norm_preclip=r["grad_norm_preclip"])
        fm = list(t.head.left.parameters()) + list(t.head.right.parameters())
        rec.expect(f"{method}_feature_maps_change", max_change(before_fm, fm) > 0, max_change=max_change(before_fm, fm))
        o = r["_out"]
        rec.expect(f"{method}_step_features_nonconstant", float(o["A"].std(0).max()) > 0 and float(o["B"].std(0).max()) > 0,
                   a_std_max=float(o["A"].std(0).max()), b_std_max=float(o["B"].std(0).max()))
        del r, o, out
    t = trainer("MF_M1")
    before_enc = [p.detach().clone() for p in enc_params(t)]
    r = t.step(t.draw(), keep=True)
    rec.expect("MF_M1_step_logs_gap_iterations_convergence", all(k in r for k in ("m1_gap", "m1_iterations", "m1_converged", "m1_status"))
               and math.isfinite(r["m1_gap"]) and r["m1_gap_scope"] == "inner_pre_outer_batch" and not r["m1_require_converged"],
               gap=r.get("m1_gap"), iterations=r.get("m1_iterations"), converged=r.get("m1_converged"), status=r.get("m1_status"))
    rec.expect("MF_M1_W_is_inner_solution_in_box", torch.equal(t.head.W, r["_out"]["info"]["m1_W"]) and r["w_absmax"] <= 1.0
               and not t.head.W.requires_grad, w_absmax=r["w_absmax"])
    rec.expect("MF_M1_encoder_grad_nonzero_and_params_change", grad_norm(enc_params(t)) > 0 and max_change(before_enc, enc_params(t)) > 0,
               encoder_grad_norm_postclip=grad_norm(enc_params(t)), encoder_max_change=max_change(before_enc, enc_params(t)))
    rec.expect("MF_M1_W_not_an_optimiser_parameter", all(p is not t.head.W for g in t.opt.param_groups for p in g["params"]))
    del r
    t = trainer("MF_M0", w_init="uniform")
    W0, f0 = t.head.W.detach().clone(), t.head.left.weight.detach().clone()
    for p in t.clip_params:
        p.grad = torch.zeros_like(p)
    t.opt.step()
    lr, wd = t.cfg.head_lr, t.cfg.weight_decay
    rec.expect("W_weight_decay_zero_functional", torch.equal(t.head.W, W0) and [g["weight_decay"] for g in t.opt.param_groups if g["name"] == "W"] == [0.0],
               groups={g["name"]: {"lr": g["lr"], "weight_decay": g["weight_decay"]} for g in t.opt.param_groups})
    rec.cmp("feature_map_weight_decay_is_host_value", t.head.left.weight, f0 * (1 - lr * wd), "float32")
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


# ------------------------------------------------------------------------------------------------------------ run / merge
def parts_dir(out_root: Path) -> Path:
    return out_root / "T01" / "parts"


def run(args) -> int:
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
                      "gpu_name": env.get("gpu_name")}), flush=True)
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
            else:
                raise ValueError(f"unknown part {part}")
        except Exception as e:  # noqa: BLE001
            res = {"part": part, "device": devtype, "status": "FAIL", "failure_kind": "exception", "error": repr(e)[:2000],
                   "traceback": traceback.format_exc()[-8000:], "checks": []}
        res.update({"provenance": env, "run_dir": str(run_dir), "wall_seconds": time.time() - t0, "device_type": devtype})
        A.write_json_atomic(run_dir / f"{part}_{devtype}.json", res)
        A.write_json_atomic(parts_dir(args.out_root) / f"{part}_{devtype}.json", res)
        print(f"[t01] part {part} on {devtype}: {res['status']} ({res.get('n_pass', 0)} pass / {res.get('n_fail', 0)} fail, "
              f"{time.time() - t0:.0f} s) required failures: {res.get('required_failures', [res.get('error')])}", flush=True)
        rc = rc or (0 if res["status"] == "PASS" else 1)
    return rc


def _summary(part: dict | None) -> dict:
    if part is None:
        return {"status": "NOT_RUN"}
    return part


def merge(args) -> int:
    pdir = parts_dir(args.out_root)
    load = lambda name, dev: json.loads((pdir / f"{name}_{dev}.json").read_text()) if (pdir / f"{name}_{dev}.json").exists() else None  # noqa: E731
    torch_parts = {f"{n}_{d}": _summary(load(n, d)) for n, d in EXPECTED_TORCH_PARTS}
    geom_parts = {f"{n}_{d}": _summary(load(n, d)) for n, d in EXPECTED_GEOMETRY_PARTS}

    def overall(parts):
        st = [p["status"] for p in parts.values()]
        if any(s == "FAIL" for s in st):
            return "FAIL"
        if all(s == "PASS" for s in st):
            return "PASS"
        return "NOT_RUN"
    allp = {**torch_parts, **geom_parts}
    verified = all(p["status"] == "PASS" for p in allp.values())
    common = {"generated_utc": A.utc_now(), "generator": "scripts/v10/t01_validate.py merge", "parts_dir": str(pdir),
              "status_vocabulary": "PASS / FAIL / NOT_RUN (missing part file = NOT_RUN); a check-group exception is a FAIL row",
              "host_adapter_verified": verified,
              "host_adapter_verified_rule": "true only when every expected part (verify_torch, core_tol, adapter, geometry) is PASS "
                                            "on CPU and on CUDA; otherwise false",
              "pending_parts": [k for k, p in allp.items() if p["status"] == "NOT_RUN"],
              "failed_parts": [k for k, p in allp.items() if p["status"] == "FAIL"],
              "tolerances_design": {k: {"atol": v[0], "rtol": v[1]} for k, v in A.TOLERANCES.items()}}
    tv = {"schema": "V10-T01-TORCH_VALIDATION-v1", "overall_status": overall(torch_parts), **common, "parts": torch_parts}
    gg = {"schema": "V10-T01-GEOMETRY_GRADIENT_CHECK-v1", "overall_status": overall(geom_parts), **common,
          "method": {"transforms": [t[0] for t in GEOMETRY_TRANSFORMS], "fd_steps": FD_STEPS, "fd_atol": FD_ATOL, "fd_rtol": FD_RTOL,
                     "kink_rule": "warped BOX1 grid on a (1/L)-lattice at distance 1/(2L) from every integer coordinate; zero floor "
                                  "changes of any warped coordinate over [p - h e, p + h e] verified for every direction and step"},
          "parts": geom_parts}
    args.repo_out.mkdir(parents=True, exist_ok=True)
    A.write_json_atomic(args.repo_out / "TORCH_VALIDATION.json", tv)
    A.write_json_atomic(args.repo_out / "GEOMETRY_GRADIENT_CHECK.json", gg)
    print(json.dumps({"TORCH_VALIDATION": tv["overall_status"], "GEOMETRY_GRADIENT_CHECK": gg["overall_status"],
                      "host_adapter_verified": verified, "pending": common["pending_parts"], "failed": common["failed_parts"]}), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--device", choices=("cpu", "cuda"), required=True)
    r.add_argument("--parts", default="verify_torch,core_tol,adapter,geometry")
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
