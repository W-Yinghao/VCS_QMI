"""EPFL geometry checks for the V10 host (decision 2026-10-17 section 6.2 item 5: warp, geometry gradient, pixel units, effective support,
frozen checkpoint).  Torch; every function takes a device and returns check rows {"name", "status", "required", ...}, so the CPU tests
(tests/v10/test_epfl_geometry.py) and the GPU gate (scripts/v10_epfl/gpu_gate.py) run the same code.

Data come only through EPFLDataGuard at stage GATE_CHECKS (FIT / CAL windows); no DEV window, no registration error on a CAL / DEV
window is computed here.  The one end-to-end registration (pixel units) runs on a FIT window with a known synthetic perturbation.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F

from v10 import adapter as A
from . import host as H

Q, P = A.Q, A.P
R_VP_CRITIC = Q.R_OMEGA + Q.RF_BOX * math.sqrt(2) + Q.TMAX + math.sqrt(2)     # radius about c read in V_p by critic / MF energies (204.04)
R_VP_POINT = Q.R_OMEGA + Q.TMAX + math.sqrt(2)                                # MI / NMI / encode-once latents: point samples (181.41)
R_V = R_VP_CRITIC + Q.TMAX                                                     # radius read in the raw window V through V_p (255.04)


def row(name, ok, required=True, **kw):
    return {"name": name, "status": "PASS" if ok else "FAIL", "required": required, **kw}


# ------------------------------------------------------------------------------------------------------------ pixel units (no data)
def pixel_unit_rows() -> list:
    out = []
    errs = [(d, Q.corner_error_full(0.0, np.array([d, 0.0]), 0.0, (0.0, 0.0))) for d in (0.25, 1.0, 2.25, 10.5, 51.0)]
    out.append(row("units.translation_error_d_px_gives_err_full_4d", all(abs(e - 4 * d) < 1e-9 for d, e in errs), values=errs))
    rot = [(t, Q.corner_error_full(t, np.zeros(2), 0.0, (0.0, 0.0)), 4 * H.mean_corner_error_native(t, (0.0, 0.0))) for t in (0.1, 1.0, 5.0)]
    out.append(row("units.rotation_error_matches_independent_corner_formula", all(abs(a - b) < 1e-9 for _, a, b in rot), values=rot))
    mix = Q.corner_error_full(3.0, np.array([1.5, -2.0]), 1.0, (-0.5, 0.25)); ind = 4 * H.mean_corner_error_native(3.0, (1.5, -2.0), 1.0, (-0.5, 0.25))
    out.append(row("units.rotation_plus_translation_independent_formula", abs(mix - ind) < 1e-9 * max(1.0, ind), err_full=mix, independent=ind,
                   note="truth theta* = 1 deg, t* = (-0.5, 0.25); estimate 3 deg, (1.5, -2.0); both transforms about the window centre"))
    pure = Q.corner_error_full(7.0, np.array([3.0, 4.0]), 7.0, (0.0, 0.0))
    out.append(row("units.same_rotation_pure_translation_error_is_4_x_shift", abs(pure - 20.0) < 1e-9, err_full=pure,
                   note="equal rotations, translation error (3, 4) native px -> mean corner error 5 native px = 20 err_full"))
    thr = H.success_thresholds_native()
    out.append(row("units.thresholds_42_9_err_full_are_10.5_2.25_native_px", (thr["primary_native_px"], thr["strict_native_px"]) == (10.5, 2.25), **thr))
    out.append(row("units.corner_patch_half_width_104.125_native_px", Q.HALF834 == 104.125 and P.DOWN == 4, half=Q.HALF834,
                   note="CoMIR's 834-px BF patch at x4 downsampling; on the native EPFL window the same 512-frame patch is 208.25 native px"))
    out.append(row("units.search_set_in_native_px", (Q.TMAX, Q.THMAX, P.GRID_STEP, Q.TH_STEP) == (51.0, 30.0, 4.0, 3.0), t_max_native=Q.TMAX,
                   theta_max_deg=Q.THMAX, grid=[P.GRID_STEP, Q.TH_STEP]))
    return out


def manifest_unit_rows(guard) -> list:
    import json
    p3 = json.loads((guard.r6_root / "manifest.json").read_text())
    inc = [r for r in guard.rows.values() if r["inclusion"] == "include"]
    return [row("units.p3_manifest_down_1_size_512", p3["down"] == 1 and p3["size"] == 512 and p3["dn_scale"] == 255.0, down=p3["down"]),
            row("units.v10_manifest_resize_factor_1_for_all_420_windows", all(r["resize_factor"] == "1.0" for r in inc), n=len(inc)),
            row("units.perturbation_translations_within_51_native_px", all(math.hypot(t[1], t[2]) <= Q.TMAX + 1e-9 and abs(t[0]) <= Q.THMAX
                                                                          for w in H.pinned_perturbations(guard).values() for t in w))]


def support_constant_rows() -> list:
    return [row("support.analytic_critic_radius_in_V_le_255.5", R_V <= Q.CEN, r_vp=R_VP_CRITIC, r_v=R_V, cen=Q.CEN,
                formula="R_OMEGA + RF_BOX sqrt2 + TMAX (candidate) + sqrt2 (bilinear) + TMAX (perturbation)"),
            row("support.boxes", (Q.BOX1, Q.BOX2, Q.R_OMEGA) == ((110, 401), (59, 452), 129), box1=Q.BOX1, box2=Q.BOX2, r_omega=Q.R_OMEGA)]


# ------------------------------------------------------------------------------------------------------------ warp (real EPFL window)
def raw_nir(guard, window, role, device, dtype=torch.float32):
    path = guard.check(window, {role}, "V10 EPFL geometry check (raw NIR window)")
    return torch.tensor(np.load(path)["nir"], dtype=dtype, device=torch.device(device))[None]


def warp_rows(guard, window, role, device) -> list:
    """perturb_map conventions on a real EPFL window, in the host's float32 (perturb_map builds float32 coordinates): identity and integer
    translation exact, the 90-degree direction (float32 cos / sin: tolerance 1e-4 on [0, 1] intensities), the truth round trip
    V_p(T*(z)) = V(z) beating every nearby wrong transform for each of the window's 10 frozen perturbations, and the EPFL moving path =
    REG-P3 Inputs3 (linear NIR, both paths) while the BF log path cannot run on the EPFL cache."""
    dev = torch.device(device); out = []
    V = raw_nir(guard, window, role, dev)
    ident = Q.perturb_map(torch, F, V, 0.0, (0.0, 0.0))
    out.append(row("warp.identity_exact", bool(torch.equal(ident, V)), max_abs=float((ident - V).abs().max())))
    sh = Q.perturb_map(torch, F, V, 0.0, (7.0, -12.0))
    out.append(row("warp.integer_translation_exact", bool(torch.equal(sh[0, 7:, :-12], V[0, :-7, 12:])), note="V_p(y) = V(y - t*) for t* = (7, -12)"))
    r90 = Q.perturb_map(torch, F, V, 90.0, (0.0, 0.0)); expect = torch.flip(V[0].T, (0,))   # V_p(y, x) = V(x, 511 - y)
    err90 = float((r90[0] - expect).abs().max())
    out.append(row("warp.rotation_direction_90deg", err90 <= 1e-4, max_abs=err90, data_range=[float(V.min()), float(V.max())],
                   convention="V_p(y) = V(R(-theta*)(y - c) + c); R(theta) = [[cos, -sin], [sin, cos]] on (y, x); theta* > 0 turns the content"))
    pert = H.perturbations_for(guard, [window], role, "V10 EPFL warp round-trip check")[window]
    yy, xx = torch.meshgrid(torch.arange(512, dtype=torch.float64, device=dev), torch.arange(512, dtype=torch.float64, device=dev), indexing="ij")
    z = torch.stack([yy.reshape(-1), xx.reshape(-1)], -1); z = z[((z - Q.CEN) ** 2).sum(-1) <= Q.R_OMEGA ** 2]
    ref = V[0].reshape(-1)[(z[:, 0] * 512 + z[:, 1]).long()].double()
    rt = []
    for i, (th, ty, tx) in enumerate(pert):
        Vp = Q.perturb_map(torch, F, V, th, (ty, tx))

        def err(dth, dt):
            pts = Q.apply_T(torch, torch.tensor([math.radians(th + dth)], dtype=torch.float64, device=dev),
                            torch.tensor([[ty + dt[0], tx + dt[1]]], dtype=torch.float64, device=dev), z).to(Vp.dtype)
            return float((Q.sample_pts(torch, F, Vp, pts)[0, :, 0].double() - ref).abs().mean())
        e0 = err(0.0, (0.0, 0.0)); worse = [err(dth, dt) for dth, dt in ((1.0, (0, 0)), (-1.0, (0, 0)), (0.0, (1.0, 0)), (0.0, (-1.0, 0)), (0.0, (0, 1.0)), (0.0, (0, -1.0)))]
        rt.append({"perturbation": i, "theta": th, "t": [ty, tx], "mean_abs_err_at_truth": e0, "min_err_at_off_truth": min(worse)})
    out.append(row("warp.truth_round_trip_is_the_minimum_for_every_perturbation", all(r["mean_abs_err_at_truth"] < r["min_err_at_off_truth"] for r in rt),
                   rows=rt, data_scale=float(V.std())))
    I3 = H.EPFLGuardedInputs(guard, {role}, "V10 EPFL warp: moving path", {window: pert}, dev)
    mv = I3.moving(window, 3)
    Vp32 = Q.perturb_map(torch, F, V, pert[3][0], (pert[3][1], pert[3][2]))[0].cpu().numpy().astype(np.float64)
    exp = torch.tensor(P.robust_norm(Vp32), device=dev)[None]
    out.append(row("warp.epfl_moving_path_is_linear_nir_robust_norm_both_paths", bool(torch.equal(mv[0][0], exp[0])) and mv[0] is mv[1]))
    bf_in = Q.Inputs(torch, F, guard.r6_root, {window: pert}, dev)
    try:
        bf_in.moving(window, 3); bf_refused = False
    except KeyError:
        bf_refused = True
    out.append(row("warp.bf_log_moving_path_cannot_run_on_the_epfl_cache", bf_refused, note="Q.Inputs reads 'shg_log', absent from the REG-P3 cache"))
    ref_t = I3.reference(window); z_ = np.load(guard.r6_root / "cache" / f"{window}.npz")
    out.append(row("warp.reference_is_cached_luminance_and_rgb", np.array_equal(ref_t[0][0].cpu().numpy(), z_["bf"])
                   and np.array_equal(ref_t[1].permute(1, 2, 0).cpu().numpy(), z_["bf_rgb_n"])))
    li = P.learned_inputs(torch, guard.r6_root, window, dev)
    out.append(row("warp.training_inputs_rgb3_linear_nir1", tuple(li[0].shape) == (3, 512, 512) and tuple(li[1].shape) == (1, 512, 512)
                   and np.array_equal(li[1][0].cpu().numpy(), z_["shg_log_n"]) and np.array_equal(z_["shg_log_n"], z_["shg"])))
    return out


# ------------------------------------------------------------------------------------------------------------ effective support
def _poison_outside(m, radius, value):
    H_, W_ = m.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(H_, device=m.device, dtype=torch.float64), torch.arange(W_, device=m.device, dtype=torch.float64), indexing="ij")
    out = m.clone(); out[..., ((yy - Q.CEN) ** 2 + (xx - Q.CEN) ** 2) > radius ** 2] = value
    return out


def _rel_diff(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12)))


def support_rows(guard, window, role, device, energies: dict, dtype=torch.float32, sentinel=1.0e3, rtol=None) -> list:
    """Effective support on a real window, in the production float32: for each energy constructor make(moving) -> energy (values at a fixed
    parameter list incl. the search-set extremes and two grid angles), the moving image is replaced OUTSIDE the method's read radius by a
    sentinel; values must be unchanged (rtol 0 on CPU; on CUDA 1e-6, cuDNN may pick global (FFT) convolution algorithms), and replacing it
    inside a smaller radius must change them by far more (power).  At the V level, V_p within R_VP_CRITIC is bitwise unchanged when V is
    poisoned outside R_V.  The reflected fraction of V_p (entering only the whole-image robust normalisation) is recorded."""
    dev = torch.device(device); out = []; rtol = (0.0 if dev.type == "cpu" else 1e-6) if rtol is None else rtol
    V = raw_nir(guard, window, role, dev); pert = H.perturbations_for(guard, [window], role, "V10 EPFL support check")[window]
    for i in (0, 4, 8):
        th, ty, tx = pert[i]
        clean = Q.perturb_map(torch, F, V, th, (ty, tx)); pois = Q.perturb_map(torch, F, _poison_outside(V, R_V + 1e-6, sentinel), th, (ty, tx))
        inner = _poison_outside(torch.zeros_like(clean), R_VP_CRITIC, 1.0)[0] == 0
        out.append(row(f"support.V_level_pert{i}", bool(torch.equal(clean[0][inner], pois[0][inner])), r_v=R_V, r_vp=R_VP_CRITIC,
                       reflected_fraction_of_Vp=float(_source_outside(th, ty, tx, dev).float().mean())))
    params = [(0.0, 0.0, 0.0), (Q.UMAX, 36.0, -36.0), (-Q.UMAX, -51.0, 0.0), (0.4 * Q.UMAX, 0.0, 51.0), (-0.2 * Q.UMAX, 30.5, 41.2)]
    for name, (make, radius) in energies.items():
        e0 = make(None)
        base = _values(e0, params, dev, dtype)
        poisoned = _values(make(_poison_outside(e0.mov, radius + 1e-6, sentinel)), params, dev, dtype)
        control = _values(make(_poison_outside(e0.mov, radius - 25.0, sentinel)), params, dev, dtype)
        d_out, d_in = _rel_diff(poisoned, base), _rel_diff(control, base)
        out.append(row(f"support.energy_unchanged_outside_r{radius:.2f}_{name}", d_out <= rtol, max_rel_diff=d_out, rtol=rtol, values=base[:3]))
        out.append(row(f"support.power_control_changes_inside_{name}", d_in > max(1e-3, 100 * rtol), max_rel_diff=d_in))
    return out


def _source_outside(th, ty, tx, dev):
    """Grid points of V_p whose source R(-theta*)(y - c - t*) + c falls outside the 512 window (reflection-padded): they enter only the
    whole-image robust normalisation of the moving image, never the sampled energy (support rows)."""
    y = Q.grid_coords(torch, 0, Q.S - 1, dev).double() - torch.tensor([ty, tx], dtype=torch.float64, device=dev)
    src = Q.apply_T(torch, torch.tensor([-math.radians(th)], dtype=torch.float64, device=dev), torch.zeros(1, 2, dtype=torch.float64, device=dev), y)[0]
    return ((src < -0.5) | (src > Q.S - 0.5)).any(-1)


def _values(energy, params, dev, dtype):
    vals = []
    with torch.no_grad():
        for p in params:
            vals.append(float(energy(torch.tensor(p, dtype=dtype, device=dev))))
        tt = torch.tensor(Q.grid_points()[1][::37], dtype=dtype, device=dev)
        for th in (-30.0, 21.0):
            vals += [float(v) for v in energy.grid_values(th, tt)]
    return vals


# ------------------------------------------------------------------------------------------------------------ geometry gradient (T01 on EPFL)
def gradient_rows(guard, window, device, dtype=torch.float64, states=("host_init", "sharpened_x10")) -> list:
    """T01's geometry part (scripts/v10/t01_validate.py: lattice transforms with a known kink margin, fp64 finite differences vs autograd for
    every MF energy term, dense vs moments values and gradients) on an EPFL FIT window, plus the checkpoint round trip of the indivisible
    (encoders, feature maps, W) pair and the frozen-model contract."""
    import sys
    sys.path.insert(0, str(A.REPO / "scripts" / "v10"))
    import t01_validate as T   # noqa: E402  (helpers only; nothing runs on import)
    dev = torch.device(device); out = []
    pert = {window: Q.perturbations_p2(window)}
    guard.log_rule(window, {"FIT"}, "V10 EPFL geometry gradient (perturbation rule on a FIT window)")
    inputs = H.EPFLGuardedInputs(guard, {"FIT"}, "V10 EPFL geometry gradient check", pert, dev)
    mov = inputs.moving(window, 0)[1]
    rng = lambda: np.random.default_rng(Q.seed_of(f"R6-REG-P2|pos|{window}|0"))   # noqa: E731
    specs = T.geometry_specs()
    for s in specs:
        out.append(row(f"gradient.kink_margin_{s['name']}", s["margin"] >= s["expected_margin"] - 1e-9, margin=s["margin"]))
    for state in states:
        eU, eV, cr, head = T.build_models_seeded(T.SEED, dev, dtype)
        if state == "sharpened_x10":
            head = T.sharpened(head)
        A.freeze_for_test(eU, eV, cr, head)
        h0 = A.state_hash(eU, eV, head)
        DU = A.reference_descriptors(eU, inputs, window, dtype)
        e = A.MFRigidEnergy(DU, mov, rng(), eV=eV, head=head, estimator="moments", dtype=dtype, record=True)
        for s in specs:
            p = torch.tensor(s["p"], dtype=dtype, device=dev, requires_grad=True)
            tv = T.term_values(e, p); g = {k: T.grads(v, [p])[0].detach().cpu().numpy() for k, v in tv.items()}
            c1 = A.compare_many([tv["q1_moments"], tv["q2_moments"], tv["J_moments"], tv["J_adapter_batched_moments"]],
                                [tv["q1_dense"], tv["q2_dense"], tv["J_dense"], tv["J_dense"]], T.dtype_name(dtype))
            out.append(row(f"gradient.dense_vs_moments_values_{state}_{s['name']}", c1["allclose"], max_abs_err=c1["max_abs_err"]))
            c2 = A.compare_many([torch.tensor(g[k]) for k in ("q1_moments", "q2_moments", "J_moments", "J_adapter_batched_moments")],
                                [torch.tensor(g[k]) for k in ("q1_dense", "q2_dense", "J_dense", "J_dense")], T.dtype_name(dtype))
            out.append(row(f"gradient.dense_vs_moments_geometry_grads_{state}_{s['name']}", c2["allclose"], max_abs_err=c2["max_abs_err"]))
            if dtype == torch.float64:
                worst = 0.0; ok = True
                with torch.no_grad():
                    for d in range(3):
                        for h in T.FD_STEPS:
                            ev = np.zeros(3); ev[d] = h
                            plus = T.term_values(e, torch.tensor(s["p"] + ev, dtype=dtype, device=dev)); minus = T.term_values(e, torch.tensor(s["p"] - ev, dtype=dtype, device=dev))
                            for k in tv:
                                scale = float(np.abs(g[k]).max()); fd = (float(plus[k]) - float(minus[k])) / (2 * h)
                                bound = T.FD_ATOL + T.FD_RTOL * scale; err = abs(fd - float(g[k][d])); ok &= err <= bound; worst = max(worst, err / bound)
                out.append(row(f"gradient.fd_vs_autograd_all_terms_{state}_{s['name']}", ok, worst_err_over_bound=worst, fd_steps=list(T.FD_STEPS)))
            nz = all(float(np.abs(g[k]).max()) > 0 for k in ("p1", "p2", "q1_moments", "q2_moments"))
            out.append(row(f"gradient.nonzero_geometry_gradient_{state}_{s['name']}", nz))
        out.append(row(f"frozen.state_hash_unchanged_after_energy_and_gradients_{state}", A.state_hash(eU, eV, head) == h0))
        out.append(row(f"frozen.eval_no_grad_{state}", all(not m.training and all(not q.requires_grad for q in m.parameters()) for m in (eU, eV, head))))
        trainable = copy.deepcopy(eV).train()
        for q in trainable.parameters():
            q.requires_grad_(True)
        try:
            A.MFRigidEnergy(DU, mov, rng(), eV=trainable, head=head, estimator="moments", dtype=dtype); refused = False
        except ValueError:
            refused = True
        out.append(row(f"frozen.energy_refuses_trainable_models_{state}", refused))
    return out


def checkpoint_rows(guard, window, device, tmpdir, dtype=torch.float32) -> list:
    """The indivisible (encoders, feature maps, W) pair round trip (A.save_pair / A.load_pair) with energy values and geometry gradients on
    an EPFL FIT window: bitwise equal after reload."""
    import sys
    sys.path.insert(0, str(A.REPO / "scripts" / "v10"))
    import t01_validate as T   # noqa: E402
    dev = torch.device(device); out = []
    guard.log_rule(window, {"FIT"}, "V10 EPFL checkpoint round trip (perturbation rule on a FIT window)")
    inputs = H.EPFLGuardedInputs(guard, {"FIT"}, "V10 EPFL checkpoint round trip", {window: Q.perturbations_p2(window)}, dev)
    mov = inputs.moving(window, 1)[1]; rng = lambda: np.random.default_rng(Q.seed_of(f"R6-REG-P2|pos|{window}|1"))   # noqa: E731
    eU, eV, cr, head = T.build_models_seeded(T.SEED, dev, dtype); A.freeze_for_test(eU, eV, head)
    vals = []
    for s in T.geometry_specs():
        e = A.MFRigidEnergy(A.reference_descriptors(eU, inputs, window, dtype), mov, rng(), eV=eV, head=head, estimator="moments", dtype=dtype)
        p = torch.tensor(s["p"], dtype=dtype, device=dev, requires_grad=True); v = e(p); vals.append((float(v), T.grads(v, [p])[0].detach().clone()))
    path = f"{tmpdir}/epfl_pair.pt"; sha = A.save_pair(path, eU, eV, head, {"purpose": "V10 EPFL checkpoint round trip", "window": window})
    eU2, eV2, head2, meta = A.load_pair(path, dev, dtype); A.freeze_for_test(eU2, eV2, head2)
    out.append(row("checkpoint.state_hash_equal_after_reload", A.state_hash(eU2, eV2, head2) == A.state_hash(eU, eV, head), sha256=sha))
    for s, (v0, g0) in zip(T.geometry_specs(), vals):
        e2 = A.MFRigidEnergy(A.reference_descriptors(eU2, inputs, window, dtype), mov, rng(), eV=eV2, head=head2, estimator="moments", dtype=dtype)
        p = torch.tensor(s["p"], dtype=dtype, device=dev, requires_grad=True); v = e2(p); g = T.grads(v, [p])[0]
        out.append(row(f"checkpoint.energy_and_gradient_bitwise_{s['name']}", float(v) == v0 and bool(torch.equal(g, g0))))
    return out
