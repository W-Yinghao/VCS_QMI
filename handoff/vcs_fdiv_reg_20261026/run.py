"""VCS as a drop-in for MI in intensity-based multimodal rigid registration (training-free; BF-SHG and EPFL RGB-NIR, V10 problems).

Same host pipeline as the V10 MI_native baseline (supplement RegistrationSession: role-gated inputs, positions, 0.5/99.5 percentile
ranges, Parzen joint histogram, host grid + top-3 starts + 100 Adam); only the dependence measure of the joint histogram changes
(fdiv.py).  Modes:
  check                          CPU unit tests (host MI/NMI bitwise, VCS closed form = numeric per-bin sup J, finiteness)
  gate                           GPU: host MI/NMI default reproduce the historical V10 CAL rows; every measure x config finite on 1 problem
  cal --ds D --divs a,b          CAL, all 4 configs of the given measures, + landscape / capture diagnostics
  select                         LOCK.json by the pre-registered CAL rule (no DEV read)
  dev --ds D --divs a,b          DEV, locked + default configs of the given measures, and the host MI default in the same job
  analyze                        ANALYSIS.json + ANALYSIS_CN.txt
"""
import argparse
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
sys.path.insert(0, str(BASE / "ssl_pilot_v10_r22c/v10/r22_pmi"))
import run_unit as R  # noqa: E402  (pins the host files)

A, torch, F = R.A, R.torch, R.F
Q = A.Q
from v10.supplement.registration import FiniteEnergy  # noqa: E402

sys.path.insert(0, str(HERE))
import fdiv as FD  # noqa: E402

UNIT = {ds: BASE / f"handoff/r22_c3_codex_20261023/units/r22_pmi3_{ds}_MLPp1_s1001.json" for ds in ("bf", "epfl")}
STORED = {
    ("bf", "mi"): BASE / "handoff/v10_native_geometry_recovery_20261019/families/2/results/FINAL_DEV/MI_native/seed_1001",
    ("bf", "nmi"): BASE / "handoff/v10_method_pipeline_20261019/families/3/results/FINAL_DEV/NMI_native/seed_1001",
    ("epfl", "mi"): BASE / "handoff/v10_epfl_production_20261019/results/FINAL_DEV/MI_native/seed_1001",
    ("epfl", "nmi"): BASE / "handoff/v10_epfl_production_20261019/results/FINAL_DEV/NMI_native/seed_1001",
}
MIND_DEV = {"bf": BASE / "handoff/v10_method_pipeline_20261019/families/6/results/FINAL_DEV/MIND_2D_adapted/seed_1001/DEV.jsonl",
            "epfl": BASE / "handoff/v10_epfl_production_20261019/results/FINAL_DEV/MIND_2D_adapted/seed_1001/DEV.jsonl"}
FIELDS = ("theta_hat", "t_hat", "err_full", "F_hat", "F_true", "n_grid", "seconds")


def verify():
    for p, h in json.loads((HERE / "FREEZE.json").read_text())["files"].items():
        R.require(R.sha(BASE / p) == h, "Frozen file changed: " + p)
    return R.sha(HERE / "FREEZE.json")


def key_of(ds):
    return "core" if ds == "bf" else "window"


def context(ds):
    u = R.load_unit(UNIT[ds]); guard, H = R.make_guard(u)
    return guard, H


def session(ds, guard, H, role, cores, device):
    if ds == "bf":
        return R.RegistrationSession("MI_native", None, guard, role, cores, device, None)
    return H.EPFLRegistrationSession("MI_native", None, guard, role, cores, device, None)


def cfg_name(div, bins, sigma):
    return f"{div}_b{bins}_s{sigma:g}"


def energy_for(base, div, bins, sigma):
    if div == "mi" and (bins, sigma) == FD.DEFAULT:
        return base                                                # the host MI energy itself (paired reference)
    return FD.make_energy(Q, torch, F, base, div, bins, sigma)


def register(base, div, bins, sigma, device, truth):
    e = energy_for(base, div, bins, sigma)
    rec = R.plain(A.registration_record(FiniteEnergy(e), device, truth))
    for f in ("err_full", "F_hat", "F_true"):
        R.require(math.isfinite(rec[f]), f"Non-finite registration {div} {f}")
    return rec


# ------------------------------------------------------------------------------------------------------------ diagnostics
def basin_errors(truth):
    ths, ts = Q.grid_points()
    return np.array([[Q.corner_error_full(float(th), t, truth[0], (truth[1], truth[2])) for t in ts] for th in ths])


def grid_diagnostics(e, truth, err):
    """Landscape smoothness and basin capture on the host coarse grid (21 angles x 508 translations); err = basin_errors(truth)."""
    ths, ts = Q.grid_points(); tt = torch.tensor(ts, dtype=torch.float32, device=e.x.device)
    with torch.no_grad():
        vals = np.stack([e.grid_values(float(th), tt).float().cpu().numpy() for th in ths])          # [21, 508]
    vals = np.nan_to_num(vals, nan=-np.inf)
    index = {(int(round(y)), int(round(x))): j for j, (y, x) in enumerate(ts)}; step = int(Q.P.GRID_STEP)
    nb = [[index[k] for k in ((int(round(y)) + dy, int(round(x)) + dx) for dy, dx in ((step, 0), (-step, 0), (0, step), (0, -step))) if k in index]
          for y, x in ts]
    local = 0
    for a in range(len(ths)):
        for j in range(len(ts)):
            v = vals[a, j]; neigh = [vals[a, k] for k in nb[j]] + [vals[b, j] for b in (a - 1, a + 1) if 0 <= b < len(ths)]
            local += all(v > w for w in neigh)
    basin = err < 42; order = np.argsort(vals.ravel())[::-1]
    return {"local_max_fraction": local / vals.size, "n_local_max": int(local), "global_max_in_basin": bool(basin.ravel()[order[0]]),
            "top3_has_basin": bool(basin.ravel()[order[:3]].any()), "basin_nodes": int(basin.sum()),
            "init_err_full": Q.corner_error_full(0.0, (0.0, 0.0), truth[0], (truth[1], truth[2]))}


# ------------------------------------------------------------------------------------------------------------ modes
def cpu_check():
    torch.set_num_threads(4); g = torch.Generator().manual_seed(7); checks = {}
    u = torch.rand(4096, generator=g, dtype=torch.float64); v = (0.6 * u[None] + 0.4 * torch.rand(5, 4096, generator=g, dtype=torch.float64))
    rng = ((0.0, 1.0), (0.0, 1.0))
    for nmi in (False, True):
        host = Q.soft_mi_batch(torch, u, v, rng, nmi=nmi); mine = FD.similarity(torch, u, v, rng, "nmi" if nmi else "mi", 64, 1.0)
        R.require(torch.equal(host, mine), "host soft_mi differs from fdiv " + ("nmi" if nmi else "mi"))
    checks["host_mi_nmi_bitwise"] = True
    # VCS closed form = numeric per-bin sup over T in [-1, 1] of (p - q)T - (p + q)T^2/2, and equals J(T*) of the original objective
    Pj = FD.joint_hist(torch, u, v, rng, 32, 1.0); p = Pj; q = Pj.sum(2)[:, :, None] * Pj.sum(1)[:, None, :]
    grid = torch.linspace(-1, 1, 40001, dtype=torch.float64)
    num = torch.stack([(((p[k] - q[k])[..., None] * grid) - 0.5 * ((p[k] + q[k])[..., None] * grid ** 2)).max(-1).values.sum() for k in range(len(p))])
    closed = FD.divergence(torch, Pj, "vcs")
    Ts = (p - q) / (p + q); J = (p * Ts).sum((1, 2)) - (q * Ts).sum((1, 2)) - 0.5 * (p * Ts ** 2).sum((1, 2)) - 0.5 * (q * Ts ** 2).sum((1, 2))
    R.require(torch.allclose(num, closed, rtol=1e-6, atol=1e-12) and torch.allclose(J, closed, rtol=1e-12), "VCS closed form differs")
    R.require(bool((Ts.abs() <= 1 + 1e-12).all()), "T* outside [-1, 1]")
    checks["vcs_closed_form"] = {"max_rel_numeric": float(((num - closed).abs() / closed).max()), "J_of_Tstar_equal": True}
    # independence -> ~0, identity -> larger; finite values and gradients for every measure and config
    vi = torch.rand(1, 4096, generator=g, dtype=torch.float64); vd = u[None].clone()
    for div in FD.DIVERGENCES:
        for bins, sig in FD.CONFIGS:
            x = v.clone().requires_grad_(True); val = FD.similarity(torch, u, x, rng, div, bins, sig)
            gr, = torch.autograd.grad(val.sum(), x)
            R.require(bool(torch.isfinite(val).all() and torch.isfinite(gr).all()), f"non-finite {div} {bins} {sig}")
        indep = float(FD.similarity(torch, u, vi, rng, div)); ident = float(FD.similarity(torch, u, vd, rng, div))
        R.require(ident > indep, f"{div}: identity not above independence")
        checks[div] = {"independent": indep, "identical": ident}
    R.write(HERE / "CPU_CHECK.json", {"status": "PASS", "freeze_sha256": verify(), "checks": checks})
    print("FDIV CPU PASS", flush=True)


def gate():
    freeze = verify(); R.require(json.loads((HERE / "CPU_CHECK.json").read_text())["status"] == "PASS", "CPU check not PASS")
    device = "cuda"; torch.set_num_threads(4); out = {"freeze_sha256": freeze, "gpu": torch.cuda.get_device_name(), "datasets": {}}
    for ds in ("bf", "epfl"):
        guard, H = context(ds); k = key_of(ds); cores = guard.included("CAL")[:2]; probs = [(c, i) for c in cores for i in (0, 1)]
        rows = {}
        for div in ("mi", "nmi"):
            stored = {(r[k], r["perturbation"]): r for r in R.rows(STORED[(ds, div)] / "CAL/step_0000.jsonl")}
            s = session(ds, guard, H, "CAL", cores, device)
            for c, i in probs:
                ef, _ = s.energy(c, i); base = ef.energy; truth = s.perturbations[c][i]
                if div == "nmi":
                    base = Q.RigidEnergy.__new__(Q.RigidEnergy); base.__dict__.update(ef.energy.__dict__); base.kind = "nmi"
                rec = R.plain(A.registration_record(FiniteEnergy(base), device, truth))
                mine = R.plain(A.registration_record(FiniteEnergy(FD.make_energy(Q, torch, F, ef.energy, div, 64, 1.0)), device, truth))
                old = stored[(c, i)]
                rows[f"{div}|{c}|{i}"] = {"host_exact_vs_stored": all(rec[f] == old[f] for f in ("theta_hat", "t_hat", "err_full", "F_true", "n_grid")),
                                          "fdiv_exact_vs_host": all(mine[f] == rec[f] for f in ("theta_hat", "t_hat", "err_full", "F_true", "n_grid")),
                                          "d_F_true_vs_stored": rec["F_true"] - old["F_true"], "d_err_vs_stored": rec["err_full"] - old["err_full"]}
        s = session(ds, guard, H, "CAL", cores[:1], device); ef, _ = s.energy(cores[0], 0); truth = s.perturbations[cores[0]][0]; timing = {}
        for div in FD.DIVERGENCES:
            for bins, sig in FD.CONFIGS:
                t0 = time.time(); rec = register(ef.energy, div, bins, sig, device, truth); timing[cfg_name(div, bins, sig)] = {"seconds": time.time() - t0, "err_full": rec["err_full"]}
        out["datasets"][ds] = {"reproduction": rows, "all_configs_one_problem": timing, "roles_read": sorted({x["role"] for x in guard.reads})}
        R.require(set(out["datasets"][ds]["roles_read"]) <= {"FIT", "CAL", "DEV"}, "role outside FIT/CAL/DEV")
        R.require(all(r["fdiv_exact_vs_host"] for r in rows.values()), f"{ds}: fdiv MI/NMI differs from the host energy")
    exact = all(r["host_exact_vs_stored"] for d in out["datasets"].values() for r in d["reproduction"].values())
    out["status"] = "PASS" if exact else "PASS_TOLERANCE" if all(abs(r["d_F_true_vs_stored"]) <= 1e-5 + 1e-4 * 1 for d in out["datasets"].values()
                                                                 for r in d["reproduction"].values()) else "FAIL"
    R.write(HERE / "GATE.json", out); print("FDIV GATE", out["status"], flush=True)
    R.require(out["status"] != "FAIL", "gate FAIL")


def run_role(ds, role, divs, configs_of):
    freeze = verify(); R.require(os.environ.get("SLURM_JOB_ID") and torch.cuda.is_available(), "GPU Slurm job required")
    gate_rec = json.loads((HERE / "GATE.json").read_text()); R.require(gate_rec["status"] in ("PASS", "PASS_TOLERANCE"), "gate not passed")
    device = "cuda"; torch.set_num_threads(4); job = os.environ["SLURM_JOB_ID"]; started = time.time()
    guard, H = context(ds); k = key_of(ds); cores = guard.included(role)
    tag = f"{role.lower()}_{ds}_" + "-".join(divs); out = HERE / "runs" / tag; out.mkdir(parents=True, exist_ok=True)
    R.require(not (out / "COMPLETE.json").exists() and not list(out.glob("ATTEMPT_*.json")), "No silent rerun: " + tag)
    R.write(out / f"ATTEMPT_{job}.json", {"status": "RUNNING", "job": job, "start": started, "freeze_sha256": freeze})
    s = session(ds, guard, H, role, cores, device)
    for c in cores:
        for i in range(R.N_PERT):
            ef, _ = s.energy(c, i); base = ef.energy; truth = s.perturbations[c][i]
            row = {k: c, "perturbation": i, "role": role, "truth": truth, "E": {}, "diag": {}}
            err = basin_errors(truth) if role == "CAL" else None
            if ds == "epfl":
                row["unit"] = guard.unit_of(c)
            for div in divs:
                for bins, sig in configs_of(div):
                    name = cfg_name(div, bins, sig); rec = register(base, div, bins, sig, device, truth)
                    row["E"][name] = {f: rec[f] for f in FIELDS}
                    if role == "CAL":
                        row["diag"][name] = grid_diagnostics(energy_for(base, div, bins, sig), truth, err)
            R.append(out / "ROWS.jsonl", row)
        print(tag, c, flush=True)
    R.require({x["role"] for x in guard.reads} <= {"FIT", "CAL", "DEV"}, "role outside FIT/CAL/DEV")
    res = {"status": "COMPLETE", "job": job, "ds": ds, "role": role, "divs": divs, "seconds": time.time() - started, "gpu": torch.cuda.get_device_name(),
           "freeze_sha256": freeze, "rows_sha256": R.sha(out / "ROWS.jsonl"), "n_rows": len(R.rows(out / "ROWS.jsonl")),
           "roles_read": sorted({x["role"] for x in guard.reads}), "environment": R.environment()}
    R.write(out / "COMPLETE.json", res); R.write(out / f"ATTEMPT_{job}.json", {"status": "COMPLETE", "job": job, "seconds": res["seconds"]})
    print("FDIV COMPLETE", tag, flush=True)


def unit_metrics(rows, name, ds):
    by = defaultdict(list)
    for r in rows:
        by[r["unit"] if ds == "epfl" else r["core"]].append(r["E"][name]["err_full"])
    errs = [e for v in by.values() for e in v]
    return {"strict": statistics.mean(statistics.mean(e < 9 for e in v) for v in by.values()),
            "loose": statistics.mean(statistics.mean(e < 42 for e in v) for v in by.values()),
            "median": float(np.median(errs)), "p95": float(np.percentile(errs, 95)), "n": len(errs), "units": len(by)}, by


def cal_rows(ds):
    return [r for d in job_dirs(f"cal_{ds}_") if (d / "COMPLETE.json").exists() for r in R.rows(d / "ROWS.jsonl")]


def merge(rows_list, k):
    """First occurrence of each config wins; callers pass jobs containing vcs first, so the MI reference paired with vcs/js
    comes from the same job (same GPU)."""
    m = {}
    for r in rows_list:
        key = (r[k], r["perturbation"])
        if key not in m:
            m[key] = json.loads(json.dumps(r)); continue
        for f in ("E", "diag"):
            for name, v in r.get(f, {}).items():
                m[key][f].setdefault(name, v)
    return list(m.values())


def job_dirs(prefix):
    return sorted((HERE / "runs").glob(prefix + "*"), key=lambda d: ("-vcs" not in d.name, d.name))


def select():
    verify(); lock = {"utc": R.utc(), "rule": "per dataset x measure: max (CAL loose<42, CAL strict<9, -CAL median); ties -> default (64,1), "
                      "then fewer bins, then smaller sigma", "datasets": {}}
    for ds in ("bf", "epfl"):
        rows = merge(cal_rows(ds), key_of(ds)); n = 80 if ds == "bf" else 100
        R.require(len(rows) == n and all(len(r["E"]) == len(FD.DIVERGENCES) * len(FD.CONFIGS) for r in rows), f"{ds}: CAL incomplete")
        lock["datasets"][ds] = {}
        for div in FD.DIVERGENCES:
            scored = []
            for bins, sig in FD.CONFIGS:
                m, _ = unit_metrics(rows, cfg_name(div, bins, sig), ds)
                scored.append(((m["loose"], m["strict"], -m["median"], (bins, sig) == FD.DEFAULT, -bins, -sig), (bins, sig), m))
            best = max(scored, key=lambda z: z[0])
            lock["datasets"][ds][div] = {"bins": best[1][0], "sigma": best[1][1], "cal": {cfg_name(div, b, s): m for _, (b, s), m in scored}}
    R.write(HERE / "LOCK.json", lock); print(json.dumps({ds: {d: (v["bins"], v["sigma"]) for d, v in x.items()} for ds, x in lock["datasets"].items()}))


def dev_configs(ds):
    lock = json.loads((HERE / "LOCK.json").read_text())["datasets"][ds]
    def f(div):
        c = [FD.DEFAULT] if div == "mi" else []
        for cfg in ((lock[div]["bins"], lock[div]["sigma"]), FD.DEFAULT):
            if cfg not in c:
                c.append(cfg)
        return c
    return f


def analyze():
    verify(); lock = json.loads((HERE / "LOCK.json").read_text()); out = {"utc": R.utc(), "lock": {ds: {d: (v["bins"], v["sigma"]) for d, v in x.items()}
                                                                                                   for ds, x in lock["datasets"].items()}, "datasets": {}}
    for ds in ("bf", "epfl"):
        k = key_of(ds)
        rows = merge([r for d in job_dirs(f"dev_{ds}_") if (d / "COMPLETE.json").exists() for r in R.rows(d / "ROWS.jsonl")], k)
        n = 320 if ds == "bf" else 740; R.require(len(rows) == n, f"{ds}: DEV incomplete ({len(rows)}/{n})")
        res = {"locked": {}, "default": {}, "contrasts": {}}
        L = lock["datasets"][ds]; byname = {}
        for div in FD.DIVERGENCES:
            for label, cfg in (("locked", (L[div]["bins"], L[div]["sigma"])), ("default", FD.DEFAULT)):
                name = cfg_name(div, *cfg); m, by = unit_metrics(rows, name, ds); res[label][div] = dict(m, config=name); byname[(label, div)] = by
        for label in ("locked", "default"):
            for other in ("mi", "js", "nmi"):
                a, b = byname[(label, "vcs")], byname[(label, other)]
                d_strict = [statistics.mean(x < 9 for x in a[u]) - statistics.mean(x < 9 for x in b[u]) for u in a]
                d_loose = [statistics.mean(x < 42 for x in a[u]) - statistics.mean(x < 42 for x in b[u]) for u in a]
                c = {"strict_pp": 100 * statistics.mean(d_strict), "loose_pp": 100 * statistics.mean(d_loose),
                     "units_not_worse_strict": sum(x >= 0 for x in d_strict) / len(d_strict)}
                if other == "mi":
                    c["signal"] = c["strict_pp"] >= 3 and c["units_not_worse_strict"] >= 0.7 and c["loose_pp"] >= -1
                res["contrasts"][f"{label}: vcs - {other}"] = c
        mind = [r for r in R.rows(MIND_DEV[ds])]
        by = defaultdict(list)
        for r in mind:
            by[r.get("unit", r.get(k)) if ds == "epfl" else r["core"]].append(r["err_full"])
        res["MIND_historical"] = {"strict": statistics.mean(statistics.mean(e < 9 for e in v) for v in by.values()),
                                  "loose": statistics.mean(statistics.mean(e < 42 for e in v) for v in by.values()),
                                  "median": float(np.median([e for v in by.values() for e in v])), "n": sum(len(v) for v in by.values())}
        hist = R.rows(STORED[(ds, "mi")] / "DEV.jsonl"); hidx = {(r[k], r["perturbation"]): r for r in hist}
        res["mi_default_vs_historical"] = {"n": len(rows), "exact_err": sum(r["E"][cfg_name("mi", *FD.DEFAULT)]["err_full"] == hidx[(r[k], r["perturbation"])]["err_full"] for r in rows)}
        cal = merge(cal_rows(ds), k); diag = {}
        for div in FD.DIVERGENCES:
            name = cfg_name(div, L[div]["bins"], L[div]["sigma"]); d = [r["diag"][name] for r in cal]
            init = sorted(x["init_err_full"] for x in d); t1, t2 = init[len(init) // 3], init[2 * len(init) // 3]
            terc = {}
            for lab, lo, hi in (("small", -1, t1), ("medium", t1, t2), ("large", t2, 1e9)):
                sel = [r["E"][name]["err_full"] for r, x in zip(cal, d) if lo < x["init_err_full"] <= hi]
                terc[lab] = {"n": len(sel), "loose": statistics.mean(e < 42 for e in sel), "strict": statistics.mean(e < 9 for e in sel)}
            diag[div] = {"config": name, "local_max_fraction_mean": statistics.mean(x["local_max_fraction"] for x in d),
                         "global_max_in_basin": statistics.mean(x["global_max_in_basin"] for x in d),
                         "top3_has_basin": statistics.mean(x["top3_has_basin"] for x in d), "by_initial_displacement": terc}
        res["cal_diagnostics"] = diag
        out["datasets"][ds] = res
    out["cost_gpuh"] = sum(json.loads((d / "COMPLETE.json").read_text())["seconds"] for d in (HERE / "runs").glob("*") if (d / "COMPLETE.json").exists()) / 3600
    R.write(HERE / "ANALYSIS.json", out); print(json.dumps({ds: {"locked": {d: (round(v["strict"], 4), round(v["loose"], 4)) for d, v in x["locked"].items()},
                                                                 "contrasts": x["contrasts"]} for ds, x in out["datasets"].items()}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["check", "gate", "cal", "select", "dev", "analyze"])
    ap.add_argument("--ds", choices=("bf", "epfl")); ap.add_argument("--divs", default=",".join(FD.DIVERGENCES)); a = ap.parse_args()
    divs = [d for d in a.divs.split(",") if d]; R.require(set(divs) <= set(FD.DIVERGENCES), "unknown measure")
    if a.mode == "check": cpu_check()
    elif a.mode == "gate": gate()
    elif a.mode == "cal": run_role(a.ds, "CAL", divs, lambda d: FD.CONFIGS)
    elif a.mode == "select": select()
    elif a.mode == "dev": run_role(a.ds, "DEV", divs, dev_configs(a.ds))
    else: analyze()
