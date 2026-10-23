#!/usr/bin/env python3
"""R22 alternative C (PILOT/DEVELOPMENT): register CAL and DEV with a fixed V10 final checkpoint, each registration problem with
the host J energy, then the PMI energy, then E_P[T] (pmi_energy.py), all on the same energy instance in the same job.

  run_unit.py run   --unit U.json                      full CAL + DEV of one unit (resumable: finished rows are kept)
  run_unit.py smoke --out DIR U1.json [U2.json ...]    a few fixed registrations per unit, J compared with the stored rows

The model is the historical CAL-selected checkpoint of the final (unit record; no selection here, no training).  The J path is the
evaluator of the run that produced the stored rows: BF MF / MLP perm1 as scripts/v10_bf1/run.py evaluate(), BF MLP perm8 /
InfoNCE through supplement RegistrationSession, EPFL through EPFLRegistrationSession; the host numerics files are pinned below
(identical bytes to the pipeline / EPFL hw2 trees; BF1's adapter differs only in training code).  Roles: CAL and DEV only (BF
DataGuard refuses the held TEST cores; the EPFL gate is the stage-ACTIVATED record), data locks verified before reading.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import signal
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = HERE.parents[1]
HOST_SHA256 = {
    "src/v10/__init__.py": "1d84649b206398c3ae36b1ae3bf69159baf8fe01675603e1868672da1cc1c15a",
    "src/v10/adapter.py": "9c2d1d968584b006d6c7d767e03bac335a8556d71cc462252dec6d1135699fa4",
    "src/v10/mf_vcs.py": "cdf5df2e43bacb74b963f7b6b519a962f91fc8e41f28b4f6a4e2696816024973",
    "src/v9/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "src/v9/reg/__init__.py": "458a701ac93ddd9a4541546787c8481b7d47593297ff536b1bc14b9658059db4",
    "src/v9/reg/g1.py": "b6122bc731d3d6b5b72e9fca15963da5d206b71f8e3b9174992e68ae44adbfff",
    "src/v9/reg/r6_reg_pilot.py": "c044adca00034853ab508e33f57f87c5aff55dcd5070e8ecb45bf0ca5f71791d",
    "src/v9/reg/r6_reg_p2.py": "0e248cac8cd5c72b0707b32f7e3de3ecd0b3d160764d1c188b418d2ad43c974c",
    "src/v9/reg/r6_reg_p3.py": "5bbc3ebaf11b0cdca37663e0f0205bf00ab797fb523b3f8c7a38dd48b2e9c7a2",
    "src/v10/supplement/__init__.py": "c98f60c1c8fb8f23dda8576ebda9058c5105596f7bdd6b8aaad66680d983de01",
    "src/v10/supplement/learned.py": "6776b1564a5da3332d81ddf834a62820ea327f8da51c955a06b80afd1bb2cc66",
    "src/v10/supplement/registration.py": "7e125ff626d413cf2ab0cd2430231660c0c85dcf913a398b7ab94f5f0987b806",
    "src/v10/supplement/records.py": "e9c2cef276cf64cc43bd12dd2febdddec8ac18308ae8944cd479e8b97be3dbcf",
    "src/v10/supplement/checkpoints.py": "7c19e2a4cd87a979b0176400454fcff675f0e4740e946958c918602ea17c943f",
    "src/v10/supplement/training.py": "5403dceeb76a445a0cce9d329ae476d6a4015a3223837899fc0826ed23d263e2",
    "src/v10/supplement/contract.py": "7beb675f43afa5848ac2983cbb5d0cc4e67e9d1104f74548dfe430ffee95930b",
    "src/v10/supplement/mind.py": "ea0e45bfc1709a06c21b4a74d7ef784f90219b3e19b0e0bd7599a9f208a01a18",
    "src/v10/supplement/rulsif.py": "cfc90545a7317a0a77164bc77278f832557e94c0d2cc480db14469126844bdcf",
    "src/v10/supplement/comir.py": "40ecebff8cca7923c9fe76f9e8c255570bc9f5df49d46bb4f8090830911693bd",
    "src/v10/epfl/__init__.py": "8efb9271aa6d06b963a7e187207b97415f71d4095c7fbc3d73af44a736134b39",
    "src/v10/epfl/host.py": "aef45559278b0b0bdffdfa2f40b838310a0f77a203168b359fa002c9139320cf",
    "src/v10/epfl/guard.py": "c57455fe64cb8ef3b8f5cb0bd97304e839f4b64f65c189024d37a78cd9223ab0",
    "src/v10/epfl/production.py": "ccb0e60179a4700dc4187601375f2e7046866af21ef3e0d4f09196b2a1c51b9a",
}
SUPPLEMENT_BF = ("MLP_VCS_perm8", "InfoNCE_host[J_perm1]")
METHOD = {"MF_VCS_M1": "MF_M1", "MF_VCS_M0": "MF_M0", "MF_VCS_perm1": "MF_perm1", "MLP_VCS_perm1": "MLP_perm1",
          "MLP_VCS_perm8": "MLP_perm8", "InfoNCE_host[J_perm1]": "InfoNCE_host"}
ROLES = ("CAL", "DEV")
N_PERT = 10
SMOKE_SUBSET = {"CAL": ((0, 0), (1, 0)), "DEV": ((0, 0), (0, 1), (1, 0))}   # (index into the sorted role list, perturbation)
T01_FP32 = (1e-5, 1e-4)                                                      # (atol, rtol) of the V10 fp32 acceptance


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def require(ok, why):
    if not ok:
        raise RuntimeError(why)


for _rel, _h in HOST_SHA256.items():
    require(sha(WT / _rel) == _h, "Pinned host file differs: " + _rel)
sys.path.insert(0, str(WT / "src"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from v10 import adapter as A  # noqa: E402
from v10.supplement import checkpoints as CP  # noqa: E402
from v10.supplement.registration import RegistrationSession  # noqa: E402
import pmi_energy as PE  # noqa: E402


def plain(x):
    if isinstance(x, torch.Tensor): return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, dict): return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [plain(v) for v in x]
    return x


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(plain(value), indent=1, allow_nan=False) + "\n"); os.replace(tmp, path)


def append(path, value):
    with open(path, "a") as f:
        f.write(json.dumps(plain(value), allow_nan=False) + "\n"); f.flush(); os.fsync(f.fileno())


def rows(path):
    path = Path(path)
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ------------------------------------------------------------------------------------------------------------ unit, data, model
def load_unit(path):
    u = json.loads(Path(path).read_text())
    require(u.get("schema") == "v10.r22_pmi.unit.v1" and u["family"] in METHOD and u["seed"] in (1001, 1002, 1003), "Not an R22-C unit")
    d = Path(u["unit_dir"])
    require(sha(d / "COMPLETE.json") == u["complete_sha256"], "Historical COMPLETE.json changed")
    c = json.loads((d / "COMPLETE.json").read_text())
    require(c["status"] == "COMPLETE" and c["identity"]["family"] == u["family"] and c["identity"]["seed"] == u["seed"]
            and c["identity"]["phase"] == "FINAL_DEV", "Historical unit identity differs")
    require(c["selected"]["checkpoint_sha256"] == u["checkpoint_sha256"] and c["selected"]["step"] == u["selected_step"]
            and d / c["selected"]["checkpoint"] == Path(u["checkpoint"]), "Unit is not the historical CAL-selected checkpoint")
    require(sha(u["checkpoint"]) == u["checkpoint_sha256"], "Checkpoint bytes differ")
    for role in ROLES:
        require(sha(u["stored"][role]["path"]) == u["stored"][role]["sha256"], "Stored rows changed: " + role)
    require(sha(u["data_lock"]["path"]) == u["data_lock"]["sha256"], "Data lock changed")
    return u


def make_guard(u):
    lock = json.loads(Path(u["data_lock"]["path"]).read_text())
    if u["ds"] == "bf":
        guard = A.DataGuard()
        files = {r["core"]: r for r in lock["files"]}
        for role in ROLES:
            cores = guard.included(role)
            require(sorted(c for c, r in files.items() if r["role"] == role) == cores, "BF data lock role set differs: " + role)
            for core in cores:
                require(sha(files[core]["path"]) == files[core]["sha256"], "BF input changed since the data lock: " + core)
        return guard, None
    from v10.epfl import guard as GD, host as H
    g = u["guard"]
    guard = GD.EPFLDataGuard(record=g["record"], expect_record_sha256=g["record_sha256"], data_lock=lock)
    require(guard.stage == "ACTIVATED" and guard.group_sha256 == g["group_record_sha256"], "EPFL gate is not the ACTIVATED record with group v2")
    guard.verify_data_lock(set(ROLES))
    return guard, H


def build_trainer(u, guard, H, device):
    saved = torch.load(u["checkpoint"], map_location="cpu", weights_only=True)
    cfg = dict(saved["config"]); cfg["m1"] = A.M1Config(**cfg["m1"])
    config = A.TrainConfig(**cfg)
    require(config.method == METHOD[u["family"]] and config.seed == u["seed"], "Checkpoint method / seed differs from the unit")
    require(json.loads(json.dumps(dataclasses.asdict(config))) == json.loads(json.dumps(saved["config"])), "Config round trip differs")
    fit = A.fit_cores(guard, 1.0, u["seed"]) if H is None else H.fit_windows(guard)        # names only: no FIT image is read
    t = A.Trainer(config, {}, fit, device)
    for module, state in zip(t.modules(), saved["modules"], strict=True):
        module.load_state_dict(state, strict=True)
    expected = saved.get("model_state_hash") or saved.get("model_sha256")
    require(A.state_hash(*t.modules()) == expected, "Restored model differs from the checkpoint's model hash")
    require(int(saved["step"]) == u["selected_step"], "Checkpoint step differs")
    return t, {"model_state_sha256": expected, "step": int(saved["step"]), "format": saved.get("format") or saved.get("schema")}


class BF1Session:
    """scripts/v10_bf1/run.py evaluate(): MF -> A.MFRigidEnergy (perm1: host permutation; M0 / M1: exact V moments),
    MLP perm1 -> Q.RigidEnergy 'critic'; reference descriptors once per core; no FiniteEnergy wrapper (finite records required)."""

    def __init__(self, trainer, guard, role, cores, device):
        self.trainer, self.device, self.core, self.ref = trainer, torch.device(device), None, None
        self.perturbations = {}
        for c in cores:
            guard.log_rule(c, {role}, "R22-C BF " + role + " frozen perturbation rule")
            self.perturbations[c] = A.Q.perturbations_p2(c)
        self.inputs = A.GuardedInputs(guard, {role}, "R22-C BF " + role + " registration", self.perturbations, self.device)

    def energy(self, core, i):
        t = self.trainer
        if self.core != core:
            self.inputs.ref.clear(); self.inputs.mov.clear()
            self.ref = A.reference_descriptors(t.eU, self.inputs, core); self.core = core
        rng = np.random.default_rng(A.Q.seed_of(f"R6-REG-P2|pos|{core}|{i}"))
        mov = self.inputs.moving(core, i)[1]
        if t.mf:
            est = "perm1" if t.cfg.method == "MF_perm1" else "moments"
            e = A.MFRigidEnergy(self.ref, mov, rng, eV=t.eV, head=t.head, estimator=est)
        else:
            e = A.Q.RigidEnergy(torch, F, "critic", self.ref, mov, rng, eV=t.eV, critic=t.critic)
        self.inputs.mov.pop((core, i), None)
        return e, {}


def make_session(u, trainer, guard, H, role, cores, device):
    if u["ds"] == "bf":
        if u["family"] in SUPPLEMENT_BF:
            return RegistrationSession(u["family"], trainer, guard, role, cores, device, u["candidate"])
        return BF1Session(trainer, guard, role, cores, device)
    return H.EPFLRegistrationSession(u["family"], trainer, guard, role, cores, device, u["candidate"])


# ------------------------------------------------------------------------------------------------------------ registration
FIELDS = ("theta_hat", "t_hat", "err_full", "rot_err_deg", "trans_err_full", "F_hat", "F_true", "n_grid", "seconds", "wall_seconds")


def stored_index(u, role):
    k = "core" if u["ds"] == "bf" else "window"
    return {(r[k], r["perturbation"]): r for r in rows(u["stored"][role]["path"])}


def compare_stored(new, old):
    """J re-evaluation against the stored historical row of the same registration (same checkpoint and problem)."""
    exact = all(new[k] == old[k] for k in ("theta_hat", "t_hat", "err_full", "F_hat", "F_true", "n_grid"))
    atol, rtol = T01_FP32
    return {"exact": exact, "F_true_bitwise": new["F_true"] == old["F_true"],
            "F_true_within_t01": abs(new["F_true"] - old["F_true"]) <= atol + rtol * abs(old["F_true"]),
            "n_grid_equal": new["n_grid"] == old["n_grid"], "d_err_full": new["err_full"] - old["err_full"],
            "d_F_true": new["F_true"] - old["F_true"], "stored_err_full": old["err_full"]}


def register_role(u, trainer, guard, H, role, problems, path, device, job, gpu, deadline=None):
    """problems: [(core, perturbation)] in order; rows already in path are verified and skipped."""
    k = "core" if u["ds"] == "bf" else "window"
    done = {}
    for r in rows(path):
        key = (r[k], r["perturbation"])
        require(key not in done and key in problems and r["role"] == role and set(r["E"]) == set(PE.VARIANTS)
                and all(math.isfinite(r["E"][v][f]) for v in PE.VARIANTS for f in ("err_full", "F_hat", "F_true")),
                "Unexpected or invalid stored R22 row")
        done[key] = r
    old = stored_index(u, role)
    cores = sorted({c for c, _ in problems})
    with CP.evaluation_state(trainer):
        session = make_session(u, trainer, guard, H, role, cores, device)
        last_core = None
        for core, i in problems:
            if (core, i) in done:
                continue
            if core != last_core and trainer.device.type == "cuda":
                torch.cuda.empty_cache()
            last_core = core
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError("Allocation nearly over; finished rows are kept")
            energy, prep = session.energy(core, i)
            truth = session.perturbations[core][i]
            recs = plain(PE.register_variants(energy, device, truth))
            for v, rec in recs.items():
                for f in ("err_full", "F_hat", "F_true", "theta_hat", "seconds"):
                    require(math.isfinite(rec[f]), f"Non-finite registration ({v}, {f})")
            row = {k: core, "perturbation": i, "role": role, "truth": truth,
                   "E": {v: {f: recs[v][f] for f in FIELDS} for v in PE.VARIANTS},
                   "J_vs_stored": compare_stored(recs["J"], old[(core, i)]), "prep": prep, "job": job, "gpu": gpu, "utc": utc()}
            if k == "window":
                row["unit"] = guard.unit_of(core)
            append(path, row); done[(core, i)] = row
            del energy
    return [done[p] for p in problems]


def metrics(errs):
    return {"n": len(errs), "success42": sum(e < 42 for e in errs) / len(errs), "success9": sum(e < 9 for e in errs) / len(errs),
            "median_err": float(np.median(errs))}


def summary(u, records, group=None):
    out = {}
    for v in PE.VARIANTS:
        out[v] = metrics([r["E"][v]["err_full"] for r in records])
    s = [r["J_vs_stored"] for r in records]
    out["J_vs_stored"] = {"n": len(s), "exact": sum(x["exact"] for x in s), "F_true_bitwise": sum(x["F_true_bitwise"] for x in s),
                          "F_true_within_t01": sum(x["F_true_within_t01"] for x in s), "n_grid_equal": sum(x["n_grid_equal"] for x in s),
                          "stored": metrics([x["stored_err_full"] for x in s])}
    if group is not None:
        from v10.epfl import production as PR
        out["DEV_statistics"] = {v: PR.dev_statistics([{"window": r["window"], "err_full": r["E"][v]["err_full"]} for r in records], group)
                                 for v in PE.VARIANTS}
    return out


def environment():
    env = A.environment("cuda" if torch.cuda.is_available() else None)
    env["pmi_eps"] = PE.EPS; env["variants"] = list(PE.VARIANTS); env["pmi_energy_sha256"] = sha(HERE / "pmi_energy.py")
    env["run_unit_sha256"] = sha(Path(__file__))
    return env


def problems_of(guard, role, smoke):
    cores = guard.included(role)
    if smoke:
        return [(cores[j], i) for j, i in SMOKE_SUBSET[role]]
    return [(c, i) for c in cores for i in range(N_PERT)]


def run_unit(u, out, smoke=False, deadline=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    require(torch.cuda.is_available(), "R22-C registrations run on a GPU (Slurm)")
    job = os.environ.get("SLURM_JOB_ID", "local")
    env = environment(); gpu = env.get("gpu_name")
    write(out / f"ENV_{job}.json", env)
    ident = {"unit": u, "variants": list(PE.VARIANTS), "pmi_eps": PE.EPS, "host_sha256": HOST_SHA256,
             "pmi_energy_sha256": env["pmi_energy_sha256"], "smoke": smoke}
    ip = out / "IDENTITY.json"
    if ip.exists():
        require(json.loads(ip.read_text()) == json.loads(json.dumps(ident)), "Output directory belongs to another identity")
    else:
        write(ip, ident)
    t0 = time.time()
    guard, H = make_guard(u)
    trainer, model = build_trainer(u, guard, H, device)
    result = {"status": "COMPLETE", "identity_sha256": sha(ip), "model": model, "job": job, "gpu": gpu, "roles": {}}
    group = None if H is None else guard.group_record
    for role in ROLES:
        recs = register_role(u, trainer, guard, H, role, problems_of(guard, role, smoke), out / f"{role}.jsonl", device, job, gpu, deadline)
        result["roles"][role] = summary(u, recs, group if (role == "DEV" and not smoke) else None)
    result.update(seconds_this_job=time.time() - t0, utc=utc(), rows_sha256={r: sha(out / f"{r}.jsonl") for r in ROLES},
                  data_reads=len(guard.reads), roles_read=sorted({r["role"] for r in guard.reads}))
    require(set(result["roles_read"]) <= {"CAL", "DEV"}, "A role other than CAL / DEV was read")
    write(out / "COMPLETE.json", result)
    del trainer
    torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("run", "smoke")); ap.add_argument("units", nargs="*")
    ap.add_argument("--unit"); ap.add_argument("--out"); ap.add_argument("--stop-after", type=float, default=None,
                                                                         help="seconds of this job after which no new registration starts")
    a = ap.parse_args()
    require(os.environ.get("SLURM_JOB_ID"), "R22-C runs only through Slurm")

    def terminated(signum, frame):
        raise InterruptedError(f"signal {signum}; finished rows are kept")
    signal.signal(signal.SIGTERM, terminated); signal.signal(signal.SIGUSR1, terminated)
    deadline = time.monotonic() + a.stop_after if a.stop_after else None
    if a.mode == "run":
        u = load_unit(a.unit)
        out = Path(a.out or u["out"])
        try:
            r = run_unit(u, out, deadline=deadline)
        except BaseException as e:
            write(out / f"FAIL_{os.environ.get('SLURM_JOB_ID')}.json", {"utc": utc(), "exception": repr(e), "traceback": traceback.format_exc()})
            raise
        print("COMPLETE", u["job_name"], json.dumps({role: {v: r["roles"][role][v] for v in PE.VARIANTS} for role in ROLES}), flush=True)
        return 0
    out = Path(a.out); report = {"schema": "v10.r22_pmi.smoke.v1", "utc": utc(), "units": {}}
    ok = True
    for path in a.units:
        u = load_unit(path); t0 = time.time()
        try:
            r = run_unit(u, out / u["job_name"], smoke=True)
            recs = [x for role in ROLES for x in rows(out / u["job_name"] / f"{role}.jsonl")]
            s = [x["J_vs_stored"] for x in recs]
            sec = {v: float(np.mean([x["E"][v]["wall_seconds"] for x in recs])) for v in PE.VARIANTS}
            entry = {"registrations": len(recs), "J_exact": sum(x["exact"] for x in s), "F_true_bitwise": sum(x["F_true_bitwise"] for x in s),
                     "F_true_within_t01": sum(x["F_true_within_t01"] for x in s), "n_grid_equal": sum(x["n_grid_equal"] for x in s),
                     "max_abs_d_err_full": max(abs(x["d_err_full"]) for x in s), "max_abs_d_F_true": max(abs(x["d_F_true"]) for x in s),
                     "mean_wall_seconds": sec, "gpu": r["gpu"], "stored_gpu": u["stored_gpu"], "model": r["model"]}
            # Smoke 1056093/1056094: F_true is bitwise only on the GPU type of the stored rows; across types it differs at the fp32
            # conv level (up to 2.3e-4 there), so the identity gate is bitwise F_true on the same type and n_grid everywhere.
            same_type = r["gpu"] == u["stored_gpu"]
            entry["same_gpu_type_as_stored"] = same_type
            entry["status"] = "PASS" if entry["n_grid_equal"] == len(s) and (not same_type or entry["F_true_bitwise"] == len(s)) else "FAIL"
        except Exception as e:  # noqa: BLE001
            entry = {"status": "FAIL", "error": repr(e)[:800], "traceback": traceback.format_exc()[-3000:]}
        entry["seconds"] = time.time() - t0
        ok = ok and entry["status"] == "PASS"
        report["units"][u["job_name"]] = entry
        print("[smoke]", u["job_name"], entry["status"], json.dumps({k: entry.get(k) for k in ("J_exact", "F_true_bitwise", "mean_wall_seconds")}), flush=True)
    report["status"] = "PASS" if ok else "FAIL"
    write(out / "SMOKE.json", report)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
