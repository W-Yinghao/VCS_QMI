#!/usr/bin/env python3
"""R22-C CPU check on synthetic encoders (host architectures, random weights, synthetic smooth images; no data read).

For each host energy class used by the six families (MFRigidEnergy moments / perm1, Q.RigidEnergy 'critic', MLPRigidEnergy
draws 1 / 8):
  1 J unchanged: an instance switched to PMI and EPT and back to J gives bit-identical J values, gradients and grid values to an
    untouched instance; the instance's j_critic is the class method again
  2 PMI / EPT = brute force: per position T_n recomputed independently in float64 numpy (linear + softmax + bilinear W for MF;
    linear + erf-GELU + tanh for the MLP), then mean 2 atanh(clip(T_n, -1+1e-4, 1-1e-4)) / mean T_n, against the fp32 variant
  3 clipping: T = +-1 exactly (MF W = +-1; MLP output bias +-100) gives the fixed bound 2 atanh(1 - 1e-4), finite gradient
  4 float64 MF PMI pose gradient = central finite difference
  5 runner path: pmi_energy.register_variants (J, PMI, EPT on one instance) through the unchanged host searcher on a reduced grid
    (CPU only: 15 deg x 17 px grid, 2 Adam steps): its J record equals A.registration_record of an untouched instance bit for bit
Writes v10/r22_pmi/results/CPU_CHECK.json.  Exit 0 only if every check passes.
"""
from __future__ import annotations

import copy
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_unit as R  # noqa: E402  (verifies the pinned host files, puts src on sys.path)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

A, PE = R.A, R.PE
from v10.supplement.learned import MLPRigidEnergy  # noqa: E402

torch.manual_seed(20261022)
np.random.seed(20261022)
RESULT = {"schema": "v10.r22_pmi.cpu_check.v1", "utc": R.utc(), "torch": torch.__version__, "eps": PE.EPS, "checks": {}}
OK = True


def record(name, passed, **info):
    global OK
    OK = OK and bool(passed)
    RESULT["checks"][name] = {"pass": bool(passed), **R.plain(info)}
    print(("PASS " if passed else "FAIL ") + name, json.dumps(R.plain(info))[:300], flush=True)


def smooth(c, seed):
    g = torch.Generator().manual_seed(seed)
    return F.interpolate(torch.randn(1, c, 32, 32, generator=g), size=(512, 512), mode="bilinear", align_corners=False)[0]


eU, eV, critic = A.P.build_models(nn)
head = A.MFHead(16, 16, 16, 16)
with torch.no_grad():
    head.W.copy_(torch.empty(16, 16).uniform_(-1, 1))
    for m in (head.left, head.right):
        m.weight.mul_(4.0)                                     # sharper simplex features -> |T| spread over (-1, 1)
for m in (eU, eV, critic, head):
    A.freeze_for_test(m)
ref_rgb, mov = smooth(3, 1), smooth(1, 2)
with torch.no_grad():
    ref = eU(ref_rgb[None])[0]
BUILDERS = {
    "MF_moments": lambda rng, hd=head: A.MFRigidEnergy(ref, mov, rng, eV=eV, head=hd, estimator="moments"),
    "MF_perm1": lambda rng, hd=head: A.MFRigidEnergy(ref, mov, rng, eV=eV, head=hd, estimator="perm1"),
    "MLP_host_perm1": lambda rng, cr=critic: A.Q.RigidEnergy(torch, F, "critic", ref, mov, rng, eV=eV, critic=cr),
    "MLP_perm8": lambda rng, cr=critic: MLPRigidEnergy(ref, mov, rng, encoder=eV, critic=cr, draws=8, extra_seed=77),
    "InfoNCE_perm1": lambda rng, cr=critic: MLPRigidEnergy(ref, mov, rng, encoder=eV, critic=cr, draws=1, extra_seed=78),
}
POSES = [torch.tensor([0.0, 0.0, 0.0]), torch.tensor([math.radians(7.0) * A.Q.RHO, 5.5, -12.25]),
         torch.tensor([math.radians(-21.0) * A.Q.RHO, -30.0, 18.0])]


def build(kind, **kw):
    return BUILDERS[kind](np.random.default_rng(A.Q.seed_of(f"R22-CPU|{kind}")), **kw)


def value_and_grad(e, p):
    p = p.clone().requires_grad_(True)
    v = e(p); v.backward()
    return v.detach(), p.grad.detach()


# numpy float64 brute force ------------------------------------------------------------------------------------------
def np_lin(m, x):
    return x @ m.weight.detach().double().numpy().T + m.bias.detach().double().numpy()


def np_softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)


_erf = np.vectorize(math.erf)


def np_gelu(x):
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def brute_T(e, dv):
    u = e.u.detach().double().numpy(); d = dv.detach().double().numpy(); out = np.empty(len(u))
    if isinstance(e, A.MFRigidEnergy):
        hd = e.head; W = hd.W.detach().double().numpy()
        a = np_softmax(np_lin(hd.left, u) / hd.temperature); b = np_softmax(np_lin(hd.right, d) / hd.temperature)
        for n in range(len(u)):                                 # one position at a time
            out[n] = float(a[n] @ W @ b[n])
        return out
    cr = e.kw["critic"]
    for n in range(len(u)):
        h = np.concatenate([u[n], d[n]])[None]
        h = np_gelu(np_lin(cr[0], h)); h = np_gelu(np_lin(cr[2], h)); out[n] = math.tanh(float(np_lin(cr[4], h)[0, 0]))
    return out


def brute_pmi(T):
    lo, hi = -1.0 + PE.EPS, 1.0 - PE.EPS
    return float(np.mean([2.0 * math.atanh(min(max(t, lo), hi)) for t in T]))


atol, rtol = R.T01_FP32
for kind in BUILDERS:
    e0, e1 = build(kind), build(kind)
    # 1 J unchanged after switching variants
    for v in ("PMI", "EPT", "J"):
        PE.set_variant(e1, v)
        if v != "J":
            e1(POSES[1])
    same = "j_critic" not in e1.__dict__
    ts = torch.tensor([[0.0, 0.0], [4.0, -8.0], [-12.0, 20.0], [36.0, 32.0]])
    vals, grads, grid = [], [], []
    for p in POSES:
        (v0, g0), (v1, g1) = value_and_grad(e0, p), value_and_grad(e1, p)
        vals.append(torch.equal(v0, v1)); grads.append(torch.equal(g0, g1))
    with torch.no_grad():
        grid.append(torch.equal(e0.grid_values(3.0, ts), e1.grid_values(3.0, ts)))
    record(f"{kind}:J_unchanged_bitwise", same and all(vals) and all(grads) and all(grid),
           class_method_restored=same, values_equal=vals, grads_equal=grads, grid_equal=grid)
    # 2 PMI / EPT against the brute force
    e2 = build(kind)
    errs = {}
    for p in POSES:
        with torch.no_grad():
            dv = A.host_moving_descriptors(e2, p)
            Tf = PE.positive_scores(e2, dv[None])[0].double().numpy()
            PE.set_variant(e2, "PMI"); pmi = float(e2(p))
            PE.set_variant(e2, "EPT"); ept = float(e2(p))
            PE.set_variant(e2, "J")
        Tb = brute_T(e2, dv)
        bp, be = brute_pmi(Tb), float(np.mean(Tb))
        errs.setdefault("T_maxabs", []).append(float(np.abs(Tf - Tb).max()))
        errs.setdefault("PMI_abs", []).append(abs(pmi - bp)); errs.setdefault("EPT_abs", []).append(abs(ept - be))
        errs.setdefault("PMI", []).append(pmi); errs.setdefault("PMI_brute", []).append(bp)
        errs.setdefault("T_range", []).append([float(Tb.min()), float(Tb.max())])
        errs.setdefault("ok", []).append(abs(pmi - bp) <= atol + rtol * abs(bp) and abs(ept - be) <= atol + rtol * abs(be))
    record(f"{kind}:PMI_EPT_equal_brute_force", all(errs["ok"]), **errs)

# 3 clipping at T = +-1
bound = 2.0 * math.atanh(1.0 - PE.EPS)
for sign in (1.0, -1.0):
    hd = copy.deepcopy(head)
    with torch.no_grad():
        hd.W.fill_(sign)
    e = build("MF_moments", hd=hd); PE.set_variant(e, "PMI")
    v, g = value_and_grad(e, POSES[1])
    record(f"MF_clip_{'pos' if sign > 0 else 'neg'}", abs(float(v) - sign * bound) <= 1e-3 and bool(torch.isfinite(g).all()),
           value=float(v), bound=sign * bound, grad=g)
    cr = copy.deepcopy(critic)
    with torch.no_grad():
        cr[4].bias.fill_(100.0 * sign)
    e = build("MLP_host_perm1", cr=cr); PE.set_variant(e, "PMI")
    v, g = value_and_grad(e, POSES[1])
    record(f"MLP_clip_{'pos' if sign > 0 else 'neg'}", abs(float(v) - sign * bound) <= 1e-3 and bool(torch.isfinite(g).all()),
           value=float(v), bound=sign * bound, grad=g)

# 4 float64 MF PMI pose gradient vs central finite difference
eV64, hd64 = copy.deepcopy(eV).double(), copy.deepcopy(head).double()
with torch.no_grad():
    ref64 = copy.deepcopy(eU).double()(ref_rgb[None].double())[0]
e64 = A.MFRigidEnergy(ref64, mov.double(), np.random.default_rng(A.Q.seed_of("R22-CPU|fd")), eV=eV64, head=hd64,
                      estimator="moments", dtype=torch.float64)
PE.set_variant(e64, "PMI")
p0 = POSES[1].double()
_, g = value_and_grad(e64, p0)
h = 1e-6; fd = []
with torch.no_grad():
    for k in range(3):
        d = torch.zeros(3, dtype=torch.float64); d[k] = h
        fd.append(float((e64(p0 + d) - e64(p0 - d)) / (2 * h)))
fd = np.array(fd); rel = float(np.abs(fd - g.numpy()).max() / max(1e-12, np.abs(fd).max()))
record("MF_PMI_float64_grad_vs_fd", rel < 1e-4, autograd=g, finite_difference=fd, max_rel=rel)

# 5 runner path through the host searcher (reduced CPU grid)
A.Q.TH_STEP, A.P.GRID_STEP, A.P.N_ADAM = 15.0, 17.0, 2
truth = (7.0, 5.5, -12.25)
for kind in BUILDERS:
    t0 = time.time()
    ref_rec = R.plain(A.registration_record(build(kind), "cpu", truth))
    e = build(kind)
    recs = R.plain(PE.register_variants(e, "cpu", truth))
    keys = ("theta_hat", "t_hat", "err_full", "F_hat", "F_true", "n_grid")
    same = all(recs["J"][k] == ref_rec[k] for k in keys)
    PE.set_variant(e, "PMI"); f_true_pmi = e.value(truth[0], truth[1:]); PE.set_variant(e, "J")
    fin = all(math.isfinite(recs[v][k]) for v in PE.VARIANTS for k in ("err_full", "F_hat", "F_true"))
    record(f"{kind}:runner_J_equals_host_record", same and fin and "j_critic" not in e.__dict__
           and recs["PMI"]["F_true"] == f_true_pmi,
           J=recs["J"], host=ref_rec, PMI=recs["PMI"], EPT=recs["EPT"], seconds=time.time() - t0)

RESULT["status"] = "PASS" if OK else "FAIL"
out = HERE / "results" / "CPU_CHECK.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(R.plain(RESULT), indent=1, allow_nan=False) + "\n")
print("CPU_CHECK", RESULT["status"], out, flush=True)
sys.exit(0 if OK else 1)
