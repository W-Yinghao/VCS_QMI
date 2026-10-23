#!/usr/bin/env python3
"""R22-C2 CPU check (Slurm cpu-high; no GPU, no TEST).

  1 selection: for each of the 18 units, run_pmi2.select on the stored historical J CAL rows of the 12 checkpoints gives exactly the
    per-checkpoint CAL metrics of the historical COMPLETE.json and the historical selected step (and its metrics); plus a tie case
    (equal metrics -> earliest step)
  2 PMI path = R22-C: run_pmi2 uses R22-C's pmi_energy module (pinned file); on synthetic host energies (Q.RigidEnergy 'critic',
    MLPRigidEnergy draws 1 / 8, bare and in the supplement FiniteEnergy wrapper; host searcher on a reduced CPU grid) the C2 CAL call
    (F_true_J, then PMI) and the C2 DEV call (J, PMI) equal R22-C's register_variants (J, PMI, EPT on one instance) bit for bit on
    theta_hat, t_hat, err_full, F_hat, F_true, n_grid; F_true_J = R22-C's J F_true; the instance is back on J
  3 real-stack dry run (reduced CPU grid, 2 CAL + 2 DEV problems, the original and one other checkpoint) on three units, one per
    evaluator path (BF1 MLP perm1, supplement InfoNCE, EPFL MLP perm8): run_unit completes, SELECTION.json follows the CAL rows,
    the 'PMI-selected differs' DEV path writes the three arms, a rerun keeps every row (same sha256), F_true_J is within 1e-3 of the
    stored GPU row (CPU vs GPU numerics; the GPU jobs compare bitwise / T01)
Writes v10/r22_pmi2/results/CPU_CHECK.json and dry-run rows under handoff/v10_r22_pmi2_20261023/cpu_check/.  Exit 0 only if all pass.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_pmi2 as RP  # noqa: E402

R, A, PE, torch, np = RP.R, RP.A, RP.PE, RP.torch, RP.np
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402
from v10.supplement.learned import MLPRigidEnergy  # noqa: E402
from v10.supplement.registration import FiniteEnergy  # noqa: E402

PROJECT = RP.WT.parent
DRY = PROJECT / "handoff/v10_r22_pmi2_20261023/cpu_check"
UNITS = sorted((HERE / "units").glob("*.json")) + sorted((RP.WT / "v10/epfl/r22_pmi2/units").glob("*.json"))
RESULT = {"schema": "v10.r22_pmi2.cpu_check.v1", "utc": R.utc(), "torch": torch.__version__, "checks": {}}
OK = True
KEYS = ("theta_hat", "t_hat", "err_full", "F_hat", "F_true", "n_grid")


def record(name, passed, **info):
    global OK
    OK = OK and bool(passed)
    RESULT["checks"][name] = {"pass": bool(passed), **R.plain(info)}
    print(("PASS " if passed else "FAIL ") + name, json.dumps(R.plain(info))[:400], flush=True)


# 1 selection reproduction ---------------------------------------------------------------------------------------------
record("units_18", len(UNITS) == 18, units=[p.name for p in UNITS])
for p in UNITS:
    u = RP.load_unit(p)
    n = 80 if u["ds"] == "bf" else 100
    errs = {}
    for k in u["checkpoints"]:
        rs = R.rows(k["stored_cal"]["path"])
        errs[k["step"]] = [r["err_full"] for r in rs]
    chosen, cks = RP.select(errs)
    m_equal = [c["metrics"] == k["stored_cal_metrics"] for c, k in zip(cks, u["checkpoints"], strict=True)]
    record(f"select_J_reproduces:{u['job_name']}",
           all(len(e) == n for e in errs.values()) and all(m_equal) and chosen["step"] == u["original"]["step"]
           and chosen["metrics"] == u["original"]["cal_metrics"],
           chosen_step=chosen["step"], original_step=u["original"]["step"], metrics_equal=sum(m_equal))
tie, _ = RP.select({500: [1.0, 2.0, 50.0], 250: [1.0, 2.0, 50.0], 750: [1.0, 2.0, 60.0]})
record("select_tie_earliest_step", tie["step"] == 250, chosen=tie)

# 2 PMI path = R22-C ---------------------------------------------------------------------------------------------------
record("pmi_module_is_R22C", Path(PE.__file__).resolve() == (RP.C_DIR / "pmi_energy.py").resolve()
       and RP.sha(PE.__file__) == RP.C_SHA256["pmi_energy.py"] and PE.EPS == 1e-4, module=PE.__file__)
torch.manual_seed(20261023)


def smooth(c, seed):
    g = torch.Generator().manual_seed(seed)
    return F.interpolate(torch.randn(1, c, 32, 32, generator=g), size=(512, 512), mode="bilinear", align_corners=False)[0]


eU, eV, critic = A.P.build_models(nn)
for m in (eU, eV, critic):
    A.freeze_for_test(m)
ref_rgb, mov = smooth(3, 1), smooth(1, 2)
with torch.no_grad():
    ref = eU(ref_rgb[None])[0]
BUILDERS = {
    "MLP_host_perm1": lambda rng: A.Q.RigidEnergy(torch, F, "critic", ref, mov, rng, eV=eV, critic=critic),
    "MLP_perm8": lambda rng: MLPRigidEnergy(ref, mov, rng, encoder=eV, critic=critic, draws=8, extra_seed=77),
    "InfoNCE_perm1": lambda rng: MLPRigidEnergy(ref, mov, rng, encoder=eV, critic=critic, draws=1, extra_seed=78),
}
saved_grid = (A.Q.TH_STEP, A.P.GRID_STEP, A.P.N_ADAM)
A.Q.TH_STEP, A.P.GRID_STEP, A.P.N_ADAM = 15.0, 17.0, 2
truth = [7.0, 5.5, -12.25]
for kind, b in BUILDERS.items():
    for wrapped in (False, True):
        build = (lambda b=b: FiniteEnergy(b(np.random.default_rng(A.Q.seed_of(f"R22-CPU|{kind}"))))) if wrapped else \
                (lambda b=b: b(np.random.default_rng(A.Q.seed_of(f"R22-CPU|{kind}"))))
        t0 = time.time()
        c_ref = R.plain(PE.register_variants(build(), "cpu", truth))                 # R22-C: J, PMI, EPT on one instance
        e = build()
        ftj, pmi = RP.pmi_with_j_truth(e, "cpu", truth); pmi = R.plain(pmi)
        back_on_j = "j_critic" not in PE.inner(e).__dict__
        dev = R.plain(PE.register_variants(build(), "cpu", truth, ("J", "PMI")))
        same = (all(pmi[k] == c_ref["PMI"][k] for k in KEYS) and float(ftj) == c_ref["J"]["F_true"]
                and all(dev[v][k] == c_ref[v][k] for v in ("J", "PMI") for k in KEYS))
        record(f"{kind}{':finite_wrapper' if wrapped else ''}:C2_equals_R22C", same and back_on_j,
               F_true_J=float(ftj), C_J_F_true=c_ref["J"]["F_true"], PMI=pmi, C_PMI=c_ref["PMI"], seconds=time.time() - t0)

# 3 real-stack dry run (reduced grid still in place) ---------------------------------------------------------------------
for name in ("r22_pmi2_bf_MLPp1_s1001", "r22_pmi2_bf_InfoNCE_s1001", "r22_pmi2_epfl_MLPp8_s1001"):
    t0 = time.time()
    try:
        path = next(p for p in UNITS if p.stem == name)
        u = RP.load_unit(path)
        out = DRY / name
        if out.exists():
            shutil.rmtree(out)
        guard, H = R.make_guard(u)
        cal_cores, dev_cores = guard.included("CAL"), guard.included("DEV")
        calp = [(cal_cores[0], 0), (cal_cores[1], 3)]
        devp = [(dev_cores[0], 0), (dev_cores[-1], 7)]
        orig = u["original"]["step"]
        other = next(k["step"] for k in u["checkpoints"] if k["step"] != orig)
        steps = sorted({orig, other})
        r1 = RP.run_unit(u, out, device="cpu", cal_problems=calp, dev_problems=devp, steps=steps)
        sel = json.loads((out / "SELECTION.json").read_text())
        cal_rows = {s: R.rows(out / "CAL" / f"step_{s:04d}.jsonl") for s in steps}
        sel_ok = sel["pmi_selected"]["step"] == RP.select({s: [x["E"]["PMI"]["err_full"] for x in v] for s, v in cal_rows.items()})[0]["step"]
        dev_rows = R.rows(out / "DEV.jsonl")
        arms_ok = all(set(x["E"]) == ({"J_orig", "PMI_orig"} if sel["same_checkpoint"] else set(RP.DEV_ARMS)) for x in dev_rows)
        # the other DEV path, forced: PMI-selected = the other checkpoint when C2 chose the original, else the original
        forced = other if sel["same_checkpoint"] else orig
        fp = out / f"DEV_forced_sel_{forced}.jsonl"
        fr = RP.register_dev(u, forced, guard, H, devp, fp, "cpu", "local", None)
        forced_ok = all(set(x["E"]) == ({"J_orig", "PMI_orig"} if forced == orig else set(RP.DEV_ARMS)) for x in fr)
        # resume: a rerun registers nothing new and gives the same rows
        before = {q.name: RP.sha(q) for q in sorted(out.rglob("*.jsonl"))}
        r2 = RP.run_unit(u, out, device="cpu", cal_problems=calp, dev_problems=devp, steps=steps)
        RP.register_dev(u, forced, guard, H, devp, fp, "cpu", "local", None)
        after = {q.name: RP.sha(q) for q in sorted(out.rglob("*.jsonl"))}
        d_ft = [abs(x["J_truth_vs_stored"]["d_F_true"]) for v in cal_rows.values() for x in v]
        d_dev = [abs(x["J_vs_stored"]["d_F_true"]) for x in dev_rows]
        record(f"dry_run:{name}", sel_ok and arms_ok and forced_ok and before == after and r1["rows_sha256"] == r2["rows_sha256"]
               and max(d_ft + d_dev) < 1e-3 and r1["roles_read"] == ["CAL", "DEV"],
               steps=steps, pmi_step=sel["pmi_selected"]["step"], same=sel["same_checkpoint"], forced_sel=forced,
               max_abs_d_F_true_cal=max(d_ft), max_abs_d_F_true_dev=max(d_dev), resume_rows_identical=before == after,
               roles_read=r1["roles_read"], seconds=time.time() - t0)
    except Exception as e:  # noqa: BLE001
        import traceback
        record(f"dry_run:{name}", False, error=repr(e)[:800], traceback=traceback.format_exc()[-3000:])
A.Q.TH_STEP, A.P.GRID_STEP, A.P.N_ADAM = saved_grid

RESULT["status"] = "PASS" if OK else "FAIL"
out = HERE / "results" / "CPU_CHECK.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(R.plain(RESULT), indent=1, allow_nan=False) + "\n")
print("CPU_CHECK", RESULT["status"], out, flush=True)
sys.exit(0 if OK else 1)
