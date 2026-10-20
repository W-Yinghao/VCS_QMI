#!/usr/bin/env python3
"""Cross-hardware DIAGNOSTIC of the item-5 checkpoint round trip (the H100 gate 1050341 failed checkpoint.energy_and_gradient_bitwise_T1 /
_T2).  Numbers only: it changes no release, gate verdict or rule, and it is never a gate record.  Slurm only; FIT data only.

For each case (the gate's EPFL FIT window, and the BF1 / T01 FIT core) and each settings variant (the gate's settings; cudnn.deterministic;
torch.use_deterministic_algorithms(True, warn_only=True); cudnn disabled), with the gate's seeded MF models (T01 build_models_seeded, fp32):
  runs   mem0, mem1, mem2 (the same model objects, energy rebuilt each time, as the gate), deepcopy (fresh allocations, no file),
         reload0, reload1 (A.save_pair / A.load_pair, as the gate)
  each   for both geometry specs (T1_rot+16.26deg, T2_rot-22.62deg): the gate's energy value and geometry gradient, and the same candidate
         through the explicit host chain with every intermediate: reference descriptors DU, reference samples u, A = features_a(u),
         warp coordinates, warped moving box Wk (grid_sample), eV(Wk), moving samples dv (grid_sample of eV(Wk)), B = features_b(dv),
         the moment terms and J; backward d J / d (B, dv, eV(Wk), Wk, coordinates, p)
  output every run against mem0: bitwise flag, max |abs| and max relative difference per tensor and per gradient component; the first
         forward / backward stage that differs; the torch backend flags, deterministic-algorithms state, CUBLAS_WORKSPACE_CONFIG and any
         determinism warnings; the T01 cross-hardware round-trip rows (L40S / RTX PRO 6000 / H100) read from their records
  python -I scripts/v10_epfl/xhw_diag.py --out <dir>
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from v10.epfl import runtime as RT  # noqa: E402

FWD = ("DU", "u", "A", "coords", "Wk", "eV_Wk", "dv", "B", "terms_p1", "terms_p2", "terms_q1", "terms_q2", "J_chain")
BWD = ("dJ_dB", "dJ_ddv", "dJ_deV_Wk", "dJ_dWk", "dJ_dcoords", "dJ_dp_chain")
T01_RECORDS = {"L40S (accepted T01)": "/projects/EEG-foundation-model/yinghao/outputs_v10/T01/parts/adapter_cuda.json",
               "RTX PRO 6000 (1048500)": "/projects/EEG-foundation-model/yinghao/outputs_v10/xhw_t01/rtx_pro_6000/T01/parts/adapter_cuda.json",
               "H100 NVL nodesumo01 (1048596)": "/projects/EEG-foundation-model/yinghao/outputs_v10/xhw_t01/h100_nvl_nodesumo01/T01/parts/adapter_cuda.json"}


def diff(a, b):
    import torch
    a, b = a.detach().double().cpu(), b.detach().double().cpu()
    d = (a - b).abs()
    rel = (d / b.abs().clamp_min(1e-300))
    return {"bitwise": bool(torch.equal(a, b)), "max_abs": float(d.max()) if d.numel() else 0.0,
            "max_rel": float(rel[d > 0].max()) if bool((d > 0).any()) else 0.0, "ref_max_abs": float(b.abs().max()) if b.numel() else 0.0}


def flags():
    import torch
    return {"cudnn_enabled": torch.backends.cudnn.enabled, "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic, "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32, "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_algorithms_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")}


def evaluate(A, T, models, inputs, core, mov, rng, specs, dev):
    """The gate's two numbers per spec, plus the explicit chain's intermediates and backward tensors."""
    import torch
    import torch.nn.functional as F
    eU, eV, head = models; dtype = torch.float32; Q, P = A.Q, A.P; out = {}
    for s in specs:
        DU = A.reference_descriptors(eU, inputs, core, dtype)
        e = A.MFRigidEnergy(DU, mov, rng(), eV=eV, head=head, estimator="moments", dtype=dtype)
        p = torch.tensor(s["p"], dtype=dtype, device=dev, requires_grad=True); v = e(p); g = T.grads(v, [p])[0].detach().clone()
        p2 = torch.tensor(s["p"], dtype=dtype, device=dev, requires_grad=True)
        th, t = p2[0:1] / Q.RHO, p2[1:3][None]; n = Q.BOX1[1] - Q.BOX1[0] + 1
        coords = Q.apply_T(torch, th, t, e.box1)
        Wk = Q.sample_pts(torch, F, e.mov, coords)[0, :, 0].reshape(1, 1, n, n)
        Fv = e.kw["eV"](Wk)
        dv = P.sample_map(torch, F, Fv[0], e.x - Q.BOX1[0])
        B = head.features_b(dv[None])
        J, terms = A.mf_energy_rows(e.A, B, e.W, "moments", perm=e.perm, perms=e.perms, dense_block=e.dense_block, return_terms=True)
        J = J[0]
        gB, gdv, gFv, gWk, gc, gp = torch.autograd.grad(J, [B, dv, Fv, Wk, coords, p2], retain_graph=False)
        tv = {k: (terms[k] if torch.is_tensor(terms[k]) else torch.tensor(float(terms[k]))) for k in ("p1", "p2", "q1", "q2")} if isinstance(terms, dict) else {}
        out[s["name"]] = {"energy": v.detach().clone(), "grad": g,
                          "fwd": {"DU": DU, "u": e.u, "A": e.A, "coords": coords, "Wk": Wk, "eV_Wk": Fv, "dv": dv, "B": B,
                                  **{f"terms_{k}": tv[k] for k in tv}, "J_chain": J},
                          "bwd": {"dJ_dB": gB, "dJ_ddv": gdv, "dJ_deV_Wk": gFv, "dJ_dWk": gWk, "dJ_dcoords": gc, "dJ_dp_chain": gp},
                          "chain_equals_gate_energy_bitwise": bool(torch.equal(J.detach(), v.detach()))}
    return out


def compare(run, ref):
    res = {}
    for name in ref:
        a, b = run[name], ref[name]
        e = diff(a["energy"], b["energy"]); g = diff(a["grad"], b["grad"])
        comp = [diff(a["grad"][i], b["grad"][i]) for i in range(len(b["grad"]))]
        fwd = {k: diff(a["fwd"][k], b["fwd"][k]) for k in b["fwd"]}
        bwd = {k: diff(a["bwd"][k], b["bwd"][k]) for k in b["bwd"]}
        first_f = next((k for k in FWD if k in fwd and not fwd[k]["bitwise"]), None)
        first_b = next((k for k in BWD if k in bwd and not bwd[k]["bitwise"]), None)
        res[name] = {"energy": e, "grad": g, "grad_components": {c: comp[i] for i, c in enumerate(("u", "ty", "tx"))},
                     "first_forward_difference": first_f, "first_backward_difference_from_J": first_b, "forward": fwd, "backward": bwd,
                     "gate_row_would_pass": bool(e["bitwise"] and g["bitwise"]), "chain_equals_gate_energy_bitwise": a["chain_equals_gate_energy_bitwise"]}
    return res


def run_case(A, T, case, dev, tmp):
    import numpy as np
    import torch
    specs = T.geometry_specs(); out = {"core": case["core"], "variants": {}}
    for variant in ("gate_settings", "cudnn_deterministic", "deterministic_algorithms_warn_only", "cudnn_disabled"):
        saved = (torch.backends.cudnn.deterministic, torch.backends.cudnn.enabled, torch.are_deterministic_algorithms_enabled(),
                 torch.is_deterministic_algorithms_warn_only_enabled())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if variant == "cudnn_deterministic":
                torch.backends.cudnn.deterministic = True
            elif variant == "deterministic_algorithms_warn_only":
                torch.use_deterministic_algorithms(True, warn_only=True)
            elif variant == "cudnn_disabled":
                torch.backends.cudnn.enabled = False
            fl = flags()
            eU, eV, cr, head = T.build_models_seeded(T.SEED, dev, torch.float32); A.freeze_for_test(eU, eV, head)
            rng = lambda: np.random.default_rng(A.Q.seed_of(case["rng_tag"]))   # noqa: E731
            runs = {}
            for k in range(3):
                runs[f"mem{k}"] = evaluate(A, T, (eU, eV, head), case["inputs"], case["core"], case["mov"], rng, specs, dev)
            dU, dV, dH = copy.deepcopy(eU), copy.deepcopy(eV), copy.deepcopy(head); A.freeze_for_test(dU, dV, dH)
            runs["deepcopy"] = evaluate(A, T, (dU, dV, dH), case["inputs"], case["core"], case["mov"], rng, specs, dev)
            path = Path(tmp) / f"pair_{case['name']}_{variant}.pt"
            A.save_pair(path, eU, eV, head, {"purpose": "xhw diagnostic"})
            eU2, eV2, head2, _ = A.load_pair(path, dev, torch.float32); A.freeze_for_test(eU2, eV2, head2)
            same_state = A.state_hash(eU2, eV2, head2) == A.state_hash(eU, eV, head)
            for k in range(2):
                runs[f"reload{k}"] = evaluate(A, T, (eU2, eV2, head2), case["inputs"], case["core"], case["mov"], rng, specs, dev)
            ptr = lambda m: sorted({int(q.data_ptr()) % 256 for q in m.parameters()})   # noqa: E731
            out["variants"][variant] = {
                "flags": fl, "state_hash_equal_after_reload": same_state,
                "param_data_ptr_mod256": {"original": ptr(eV) + ptr(head), "deepcopy": ptr(dV) + ptr(dH), "reloaded": ptr(eV2) + ptr(head2)},
                "vs_mem0": {r: compare(runs[r], runs["mem0"]) for r in runs if r != "mem0"},
                "values_mem0": {s: {"energy": float(runs["mem0"][s]["energy"]), "grad": [float(x) for x in runs["mem0"][s]["grad"]]} for s in runs["mem0"]},
                "determinism_warnings": sorted({str(w.message)[:300] for w in caught})}
            del runs, eU, eV, head, dU, dV, dH, eU2, eV2, head2
        torch.backends.cudnn.deterministic, torch.backends.cudnn.enabled = saved[0], saved[1]
        torch.use_deterministic_algorithms(saved[2], warn_only=saved[3])
        torch.cuda.empty_cache()
    return out


def t01_rows():
    out = {}
    for tag, p in T01_RECORDS.items():
        if not os.path.exists(p):
            out[tag] = None; continue
        d = json.load(open(p)); rows = []

        def walk(o):
            if isinstance(o, dict):
                if str(o.get("name", "")).startswith("checkpoint_roundtrip_") and "bitwise" in o:
                    rows.append({"row": o["name"], "status": o.get("status"), "bitwise": o["bitwise"], "max_abs_err": o.get("max_abs_err"),
                                 "max_rel_err": o.get("max_rel_err")})
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(d); out[tag] = {"path": p, "rows": rows, "pass_rule": "T01 rec.cmp: allclose at the dtype tolerance; 'bitwise' is recorded, not required"}
    return out


def summary(results):
    s = {}
    for case, c in results.items():
        for variant, v in c["variants"].items():
            for run, cmp in v["vs_mem0"].items():
                for spec, r in cmp.items():
                    s[f"{case}|{variant}|{run}|{spec}"] = {"energy_bitwise": r["energy"]["bitwise"], "grad_bitwise": r["grad"]["bitwise"],
                                                           "energy_max_abs": r["energy"]["max_abs"], "grad_max_abs": r["grad"]["max_abs"],
                                                           "grad_max_rel": r["grad"]["max_rel"],
                                                           "first_forward": r["first_forward_difference"],
                                                           "first_backward": r["first_backward_difference_from_J"]}
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--out", required=True)
    a = ap.parse_args(); t0 = time.time()
    pins = RT.job_pins()
    import torch
    from v10 import adapter as A
    from v10.epfl import host as H
    sys.path.insert(0, str(A.REPO / "scripts" / "v10"))
    import t01_validate as T  # noqa: E402
    env = RT.runtime_environment(A)
    dev = torch.device("cuda"); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    guard = RT.guard_from_pins(pins)
    w = guard.included("FIT")[0]
    guard.log_rule(w, {"FIT"}, "xhw diagnostic (perturbation rule on a FIT window)")
    e_inputs = H.EPFLGuardedInputs(guard, {"FIT"}, "xhw diagnostic", {w: A.Q.perturbations_p2(w)}, dev)
    cases = [{"name": "epfl_gate_window", "core": w, "inputs": e_inputs, "mov": e_inputs.moving(w, 1)[1], "rng_tag": f"R6-REG-P2|pos|{w}|1"}]
    bf_guard = A.DataGuard(); c0 = A.fit_cores(bf_guard, 1.0, T.SEED)[0]
    b_inputs = A.GuardedInputs(bf_guard, {"FIT"}, "xhw diagnostic (BF1 / T01 core)", {c0: A.Q.perturbations_p2(c0)}, dev)
    cases.append({"name": "bf1_t01_core", "core": c0, "inputs": b_inputs, "mov": b_inputs.moving(c0, 0)[1], "rng_tag": f"R6-REG-P2|pos|{c0}|0"})
    results = {}
    with tempfile.TemporaryDirectory(dir=str(out)) as tmp:
        for case in cases:
            results[case["name"]] = run_case(A, T, case, dev, tmp)
    roles = sorted({r["role"] for r in guard.reads} | {r.get("role") for r in getattr(bf_guard, "reads", []) if isinstance(r, dict)})
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    rep = {"schema": "v10.epfl.xhw_checkpoint_diagnostic.v1", "status": "DIAGNOSTIC_NOT_A_GATE_RESULT", "pins": pins, "environment": env,
           "nvidia_smi": smi, "torch_cuda": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
           "device_capability": list(torch.cuda.get_device_capability()), "flags_at_start": flags(),
           "question_map": {"1_run_to_run_vs_reload": "vs_mem0: mem1 / mem2 (no reload), deepcopy, reload0 / reload1",
                            "2_size_and_first_stage": "energy / grad / grad_components and first_forward_difference / first_backward_difference_from_J",
                            "3_cause": "the four settings variants, determinism_warnings, flags, param_data_ptr_mod256",
                            "4_bf1_and_t01": "case bf1_t01_core, and t01_records"},
           "summary": summary(results), "cases": results, "t01_records": t01_rows(), "data_roles_read": roles,
           "seconds": time.time() - t0}

    def plain(o):
        if isinstance(o, dict):
            return {k: plain(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [plain(v) for v in o]
        if hasattr(o, "detach"):
            return None
        if isinstance(o, float) and not math.isfinite(o):
            return str(o)
        return o
    (out / "XHW_DIAG.json").write_text(json.dumps(plain(rep), indent=1, default=str) + "\n")
    print(f"[xhw-diag] -> {out / 'XHW_DIAG.json'} ({time.time() - t0:.0f}s)", flush=True)
    for k, v in rep["summary"].items():
        if k.split("|")[1] == "gate_settings":
            print(f"[xhw-diag] {k} {v}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
