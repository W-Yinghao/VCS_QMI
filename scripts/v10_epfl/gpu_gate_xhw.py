#!/usr/bin/env python3
"""Cross-hardware item-5 gate on RTX6000PRO / H100 under the amended acceptance (v10/epfl/EPFL_XHW_GATE_AMENDMENT_20261020.json; owner
2026-10-20 「什么快什么来吧」).  The unchanged scripts/v10_epfl/gpu_gate.py, every part, with ONE delta: in the checkpoint part the two rows
checkpoint.energy_and_gradient_bitwise_<spec> (T1_rot+16.26deg, T2_rot-22.62deg) are judged with T01's round-trip tolerance for the same rows
(adapter.compare: |x - y| <= atol + rtol |y| elementwise, y = the value before reload; float32 atol 1e-5, rtol 1e-4; the energy and the
geometry gradient each, as T01's checkpoint_roundtrip_energy_value_<spec> / _geometry_grad_<spec>), with the measured max abs / max relative
difference and the bitwise flags recorded; checkpoint.state_hash_equal_after_reload stays bitwise and required.  Refuses L40S (its gate is
item-5 gate 1048825, unchanged).  Slurm only.
  python -I scripts/v10_epfl/gpu_gate_xhw.py            (same arguments as gpu_gate.py)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "v10_epfl"))

CRITERION = {"id": "EPFL_XHW_GATE_AMENDMENT_20261020", "record": "v10/epfl/EPFL_XHW_GATE_AMENDMENT_20261020.json",
             "rule": "checkpoint.energy_and_gradient_bitwise_<spec>: T01 round-trip tolerance (adapter.compare, float32 atol 1e-5, rtol 1e-4) on "
                     "the energy and on the geometry gradient; checkpoint.state_hash_equal_after_reload bitwise; every other row unchanged"}


def checkpoint_rows_xhw(guard, window, device, tmpdir, dtype=None):
    """geometry.checkpoint_rows (unchanged procedure), with the two energy / gradient rows judged by T01's tolerance and their measured
    differences recorded; the state-hash row is the original bitwise row."""
    import numpy as np
    import torch
    from v10 import adapter as A
    from v10.epfl import geometry as G, host as H
    Q = A.Q
    dtype = dtype or torch.float32
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
    out.append(G.row("checkpoint.state_hash_equal_after_reload", A.state_hash(eU2, eV2, head2) == A.state_hash(eU, eV, head), sha256=sha,
                     criterion="bitwise (unchanged)"))
    dn = str(dtype).split(".")[-1]
    for s, (v0, g0) in zip(T.geometry_specs(), vals):
        e2 = A.MFRigidEnergy(A.reference_descriptors(eU2, inputs, window, dtype), mov, rng(), eV=eV2, head=head2, estimator="moments", dtype=dtype)
        p = torch.tensor(s["p"], dtype=dtype, device=dev, requires_grad=True); v = e2(p); g = T.grads(v, [p])[0]
        ce = A.compare(v.detach(), torch.tensor(v0, dtype=torch.float64), dn); cg = A.compare(g, g0, dn)
        bit_e, bit_g = float(v) == v0, bool(torch.equal(g, g0))
        out.append(G.row(f"checkpoint.energy_and_gradient_bitwise_{s['name']}", ce["allclose"] and cg["allclose"],
                         criterion=CRITERION["id"] + ": T01 round-trip tolerance (adapter.compare, float32 atol 1e-5, rtol 1e-4)",
                         energy={**ce, "bitwise": bit_e}, gradient={**cg, "bitwise": bit_g}, bitwise_energy_and_gradient=bit_e and bit_g))
    return out


def main() -> int:
    from v10 import adapter as A
    from v10.epfl import geometry as G, runtime as RT
    import gpu_gate as GG
    env = RT.runtime_environment(A)
    if env.get("gpu_name") == RT.BUDGET_GPU:
        raise SystemExit("L40S keeps the unchanged item-5 gate (1048825); the amended cross-hardware gate runs on RTX6000PRO / H100 only")
    G.checkpoint_rows = checkpoint_rows_xhw               # the one delta (the procedure itself is identical)
    rc = GG.main()
    pins = RT.job_pins()
    p = RT.OUT_ROOT / "gpu_gate" / pins["job"] / "GATE.json"
    rep = json.loads(p.read_text()); rep["xhw_criterion"] = CRITERION
    cp = json.loads((p.parent / "checkpoint.json").read_text())
    rep["checkpoint_rows_measured"] = {r["name"]: {k: r.get(k) for k in ("status", "criterion", "energy", "gradient", "bitwise_energy_and_gradient", "sha256")}
                                       for r in cp["rows"]}
    RT.write(p, rep)
    print(f"[xhw-gate] {rep['status']} under {CRITERION['id']} -> {p}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
