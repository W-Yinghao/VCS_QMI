#!/usr/bin/env python3
"""R22-C analysis (PILOT/DEVELOPMENT): recompute every number from the per-registration rows of the 36 units.

  python3 v10/r22_pmi/analyze.py            write the outputs below
  python3 v10/r22_pmi/analyze.py --check    recompute and compare with the committed SUMMARY.json (exit 1 on any difference)
  python3 v10/r22_pmi/analyze.py --partial  interim readout over the families whose 3 seeds are COMPLETE (status INTERIM)

Inputs: handoff/v10_r22_pmi_20261022/runs/<ds>/<key>/seed_<s>/{CAL,DEV}.jsonl and COMPLETE.json (rows sha256 checked); every unit
must have all CAL (BF 80, EPFL 100) and DEV (BF 320, EPFL 740) problems with J, PMI and EPT.  Nothing is taken from COMPLETE
summaries.  Metrics: BF err_full (full-resolution px), success < 42 / < 9, registration-pooled (BF DEV32 convention);
EPFL native px = err_full / 4, success < 10.5 / < 2.25 native px, CAL registration-pooled, DEV unit-macro over the 72 scene units
(mean of unit success fractions; median of unit medians), as the existing EPFL analysis.  Paired per registration: two-way flips
at both thresholds between energies, and between the J re-evaluation and the stored historical row (re-evaluation reference).
Seed means; paired unit bootstrap (B 10000, seed 20261022) of the 3-seed mean DEV strict-success difference.
Outputs: v10/r22_pmi/results/SUMMARY.json and TABLES.md (aggregates only), v10/epfl/r22_pmi/EPFL_PER_UNIT.json (EPFL scene units,
private), handoff/v10_r22_pmi_20261022/ANALYSIS.json (everything).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np

WT = Path(__file__).resolve().parents[2]
PROJECT = WT.parent
RUNS = PROJECT / "handoff/v10_r22_pmi_20261022/runs"
OUT_SUMMARY = WT / "v10/r22_pmi/results/SUMMARY.json"
OUT_TABLES = WT / "v10/r22_pmi/results/TABLES.md"
OUT_EPFL = WT / "v10/epfl/r22_pmi/EPFL_PER_UNIT.json"
OUT_ALL = PROJECT / "handoff/v10_r22_pmi_20261022/ANALYSIS.json"
VARIANTS = ("J", "PMI", "EPT")
KEYS = ("M1", "M0", "MFp1", "MLPp1", "MLPp8", "InfoNCE")
SEEDS = (1001, 1002, 1003)
N = {"bf": {"CAL": 80, "DEV": 320}, "epfl": {"CAL": 100, "DEV": 740}}
THR = (42.0, 9.0)
B, BOOT_SEED = 10000, 20261022


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def unit_record(ds, key, seed):
    d = (WT / "v10/r22_pmi/units") if ds == "bf" else (WT / "v10/epfl/r22_pmi/units")
    return json.loads((d / f"r22_pmi_{ds}_{key}_s{seed}.json").read_text())


def load(ds, key, seed):
    u = unit_record(ds, key, seed)
    d = RUNS / ds / key / f"seed_{seed}"
    c = json.loads((d / "COMPLETE.json").read_text())
    out = {"family": u["family"], "gpu": c["gpu"], "stored_gpu": u["stored_gpu"], "job": c["job"]}
    k = "core" if ds == "bf" else "window"
    for role in ("CAL", "DEV"):
        p = d / f"{role}.jsonl"
        if sha(p) != c["rows_sha256"][role]:
            raise RuntimeError(f"{d}: {role} rows differ from COMPLETE.json")
        r = rows(p)
        keys = {(x[k], x["perturbation"]) for x in r}
        if len(r) != N[ds][role] or len(keys) != len(r) or any(set(x["E"]) != set(VARIANTS) for x in r):
            raise RuntimeError(f"{d}: incomplete or duplicated {role} rows")
        out[role] = r
    return out


def err(x, v):
    return x["E"][v]["err_full"] if v != "stored" else x["J_vs_stored"]["stored_err_full"]


def reg_metrics(r, v, ds):
    e = [err(x, v) for x in r]
    m = {"n": len(e), "success42": sum(z < THR[0] for z in e) / len(e), "success9": sum(z < THR[1] for z in e) / len(e),
         "median_err_full": statistics.median(e)}
    if ds == "epfl":
        m["median_native_px"] = m["median_err_full"] / 4
    return m


def unit_macro(r, v):
    per = {}
    for x in r:
        per.setdefault(x["unit"], []).append(err(x, v))
    um = {u: {"n": len(e), "success42": sum(z < THR[0] for z in e) / len(e), "success9": sum(z < THR[1] for z in e) / len(e),
              "median_native_px": statistics.median(e) / 4} for u, e in sorted(per.items())}
    return {"n_units": len(um), "success42": statistics.fmean(m["success42"] for m in um.values()),
            "success9": statistics.fmean(m["success9"] for m in um.values()),
            "median_of_unit_medians_native_px": statistics.median(m["median_native_px"] for m in um.values())}, um


def flips(r, a, b):
    """b relative to a, per registration: fail->success, success->fail, net, at both thresholds."""
    out = {}
    for name, t in (("lt42", THR[0]), ("lt9", THR[1])):
        fs = sum(err(x, a) >= t and err(x, b) < t for x in r); sf = sum(err(x, a) < t and err(x, b) >= t for x in r)
        out[name] = {"fail_to_success": fs, "success_to_fail": sf, "net": fs - sf}
    return out


def failure_attribution(r, v):
    """Among registrations failing < 42 with energy v: how many have F_hat >= F_true (the energy prefers the wrong pose) vs a search
    miss (a higher energy at the truth was not found)."""
    fail = [x for x in r if x["E"][v]["err_full"] >= THR[0]]
    return {"failures42": len(fail), "energy_prefers_wrong_pose": sum(x["E"][v]["F_hat"] >= x["E"][v]["F_true"] for x in fail)}


def analyse_unit(ds, data):
    res = {"family": data["family"], "gpu": data["gpu"], "stored_gpu": data["stored_gpu"], "job": data["job"]}
    for role in ("CAL", "DEV"):
        r = data[role]
        m = {v: reg_metrics(r, v, ds) for v in VARIANTS + ("stored",)}
        block = {"registrations": m,
                 "flips": {"PMI_vs_J": flips(r, "J", "PMI"), "EPT_vs_J": flips(r, "J", "EPT"), "PMI_vs_EPT": flips(r, "EPT", "PMI"),
                           "Jreeval_vs_stored": flips(r, "stored", "J")},
                 "J_vs_stored_exact": sum(x["J_vs_stored"]["exact"] for x in r),
                 "F_true_within_t01": sum(x["J_vs_stored"]["F_true_within_t01"] for x in r),
                 "failure_attribution": {v: failure_attribution(r, v) for v in VARIANTS},
                 "seconds": {v: sum(x["E"][v]["wall_seconds"] for x in r) for v in VARIANTS}}
        if ds == "epfl" and role == "DEV":
            block["unit_macro"], per = {}, {}
            for v in VARIANTS + ("stored",):
                block["unit_macro"][v], per[v] = unit_macro(r, v)
            if block["unit_macro"]["J"]["n_units"] != 72:
                raise RuntimeError("EPFL DEV must have 72 scene units")
            block["_per_unit"] = per
        res[role] = block
    return res


def headline(ds, role, block, v):
    """The reported metrics of one (unit, role, energy): BF registration-pooled; EPFL CAL pooled, DEV unit-macro."""
    if ds == "epfl" and role == "DEV":
        m = block["unit_macro"][v]
        return {"s42": m["success42"], "s9": m["success9"], "median": m["median_of_unit_medians_native_px"]}
    m = block["registrations"][v]
    return {"s42": m["success42"], "s9": m["success9"], "median": m["median_native_px"] if ds == "epfl" else m["median_err_full"]}


def unit_success(data, role, v, thr):
    per = {}
    k = "core" if "core" in data[role][0] else "unit"
    for x in data[role]:
        per.setdefault(x[k], []).append(err(x, v) < thr)
    return {u: float(np.mean(s)) for u, s in per.items()}


def bootstrap(ds, datas, key, a, b, thr):
    """Paired unit bootstrap of the 3-seed mean DEV success difference b - a (units: BF cores, EPFL scene units)."""
    D = []
    for s in SEEDS:
        ua, ub = unit_success(datas[(key, s)], "DEV", a, thr), unit_success(datas[(key, s)], "DEV", b, thr)
        units = sorted(ua)
        D.append([ub[u] - ua[u] for u in units])
    D = np.array(D)                                                               # [3 seeds, U units], same units in every seed
    g = np.random.default_rng(BOOT_SEED)
    idx = g.integers(0, D.shape[1], (B, D.shape[1]))
    boots = D[:, idx].mean(-1).mean(0)
    return {"estimate": float(D.mean(-1).mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "units": int(D.shape[1]), "B": B, "seed": BOOT_SEED}


def family_summary(ds, per_seed, datas, key):
    out = {}
    for role in ("CAL", "DEV"):
        f = {}
        for v in VARIANTS + ("stored",):
            h = [headline(ds, role, per_seed[s][role], v) for s in SEEDS]
            f[v] = {m: statistics.fmean(x[m] for x in h) for m in ("s42", "s9", "median")}
        for v in ("PMI", "EPT"):
            d9 = [headline(ds, role, per_seed[s][role], v)["s9"] - headline(ds, role, per_seed[s][role], "J")["s9"] for s in SEEDS]
            d42 = [headline(ds, role, per_seed[s][role], v)["s42"] - headline(ds, role, per_seed[s][role], "J")["s42"] for s in SEEDS]
            dm = [headline(ds, role, per_seed[s][role], v)["median"] - headline(ds, role, per_seed[s][role], "J")["median"] for s in SEEDS]
            f[f"{v}_minus_J"] = {"s9_per_seed": d9, "s42_per_seed": d42, "median_per_seed": dm,
                                 "s9_mean": statistics.fmean(d9), "s42_mean": statistics.fmean(d42), "median_mean": statistics.fmean(dm),
                                 "s9_seeds_positive": sum(x > 0 for x in d9), "s9_seeds_negative": sum(x < 0 for x in d9)}
        if role == "DEV":
            f["bootstrap_s9"] = {f"{v}_minus_J": bootstrap(ds, datas, key, "J", v, THR[1]) for v in ("PMI", "EPT")}
            f["bootstrap_s42"] = {f"{v}_minus_J": bootstrap(ds, datas, key, "J", v, THR[0]) for v in ("PMI", "EPT")}
        out[role] = f
    return out


def cost():
    """All attempts from the job receipts (Slurm StartTime to end, one GPU each), by mode and GPU; the CPU check is not a GPU job."""
    out = {"jobs": [], "gpu_hours": {}, "gpu_hours_total": 0.0}
    for p in sorted((PROJECT / "handoff/v10_r22_pmi_20261022/receipts").glob("*.json")):
        r = json.loads(p.read_text())
        gpu = (r.get("gpu") or "").split(",")[0]
        h = r["allocated_seconds"] / 3600
        out["jobs"].append({"job": r["job"], "name": r["name"], "mode": r["mode"], "gpu": gpu, "exit_code": r["exit_code"], "hours": h})
        key = f"{r['mode']}|{gpu}"
        out["gpu_hours"][key] = out["gpu_hours"].get(key, 0.0) + h
        out["gpu_hours_total"] += h
    return out


def pct(x):
    return f"{100 * x:.2f}"


def dpp(x):
    return f"{100 * x:+.2f}"


def tables(summary):
    L = [f"状态：{summary['status']}" + (f"；尚未完成的家族：{', '.join(summary['pending_families'])}" if summary["pending_families"] else "")]
    for ds, title, unit in (("bf", "BF-SHG", "err_full"), ("epfl", "EPFL RGB-NIR", "native px")):
        s1, s2 = ("<42", "<9") if ds == "bf" else ("<10.5 px", "<2.25 px")
        for role in ("CAL", "DEV"):
            agg = "registration-pooled" if (ds == "bf" or role == "CAL") else "72 scene-unit macro; median of unit medians"
            L.append(f"\n### {title} {role}（{agg}；中位误差单位 {unit}）\n")
            L.append(f"| 家族 | seed | J {s2} | PMI−J {s2} (pp) | EPT−J {s2} (pp) | PMI−J {s1} (pp) | PMI vs J {s2} 翻转 失败→成功/成功→失败 (净) | "
                     f"J 中位 | PMI 中位 | EPT 中位 | J 重评 vs 记录 {s2} 翻转 (净) |")
            L.append("|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|")
            for key in [k for k in KEYS if k in summary[ds]["families"]]:
                fam = summary[ds]["families"][key]
                for s in SEEDS:
                    u = summary[ds]["units"][f"{key}_s{s}"][role]
                    h = {v: u["headline"][v] for v in VARIANTS}
                    fl = u["flips"]["PMI_vs_J"]["lt9"]; fr = u["flips"]["Jreeval_vs_stored"]["lt9"]
                    L.append(f"| {key} | {s} | {pct(h['J']['s9'])} | {dpp(h['PMI']['s9'] - h['J']['s9'])} | {dpp(h['EPT']['s9'] - h['J']['s9'])} | "
                             f"{dpp(h['PMI']['s42'] - h['J']['s42'])} | {fl['fail_to_success']}/{fl['success_to_fail']} ({fl['net']:+d}) | "
                             f"{h['J']['median']:.3f} | {h['PMI']['median']:.3f} | {h['EPT']['median']:.3f} | "
                             f"{fr['fail_to_success']}/{fr['success_to_fail']} ({fr['net']:+d}) |")
                f = fam[role]
                L.append(f"| **{key} 均值** | | {pct(f['J']['s9'])} | {dpp(f['PMI_minus_J']['s9_mean'])} | {dpp(f['EPT_minus_J']['s9_mean'])} | "
                         f"{dpp(f['PMI_minus_J']['s42_mean'])} | +{f['PMI_minus_J']['s9_seeds_positive']}/−{f['PMI_minus_J']['s9_seeds_negative']} seeds | "
                         f"{f['J']['median']:.3f} | {f['PMI']['median']:.3f} | {f['EPT']['median']:.3f} | |")
        L.append(f"\n### {title} DEV：3 seed 均值差的配对单元 bootstrap（B={B}），{s2} 与 {s1}\n")
        L.append(f"| 家族 | PMI−J {s2} (pp) [95%] | EPT−J {s2} (pp) [95%] | PMI−J {s1} (pp) [95%] | EPT−J {s1} (pp) [95%] |")
        L.append("|---|---|---|---|---|")
        for key in [k for k in KEYS if k in summary[ds]["families"]]:
            bs9, bs42 = summary[ds]["families"][key]["DEV"]["bootstrap_s9"], summary[ds]["families"][key]["DEV"]["bootstrap_s42"]
            cell = lambda b: f"{dpp(b['estimate'])} [{dpp(b['ci95'][0])}, {dpp(b['ci95'][1])}]"
            L.append(f"| {key} | {cell(bs9['PMI_minus_J'])} | {cell(bs9['EPT_minus_J'])} | {cell(bs42['PMI_minus_J'])} | {cell(bs42['EPT_minus_J'])} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--partial", action="store_true", help="interim readout: families with all 3 seeds COMPLETE only")
    a = ap.parse_args()
    summary, full, epfl_private, pending = {}, {}, {}, []
    for ds in ("bf", "epfl"):
        datas, units, fams = {}, {}, {}
        for key in KEYS:
            if not all((RUNS / ds / key / f"seed_{s}" / "COMPLETE.json").exists() for s in SEEDS):
                if not a.partial:
                    raise RuntimeError(f"{ds} {key}: not all seeds COMPLETE (use --partial for an interim readout)")
                pending.append(f"{ds}_{key}")
                continue
            per_seed = {}
            for s in SEEDS:
                datas[(key, s)] = load(ds, key, s)
                res = analyse_unit(ds, datas[(key, s)])
                for role in ("CAL", "DEV"):
                    res[role]["headline"] = {v: headline(ds, role, res[role], v) for v in VARIANTS + ("stored",)}
                per_seed[s] = res
                full[f"{ds}_{key}_s{s}"] = res
                pub = json.loads(json.dumps({k: v for k, v in res.items()}))
                for role in ("CAL", "DEV"):
                    pub[role].pop("_per_unit", None)
                units[f"{key}_s{s}"] = pub
                if ds == "epfl":
                    epfl_private[f"{key}_s{s}"] = res["DEV"]["_per_unit"]
            fams[key] = family_summary(ds, per_seed, datas, key)
        summary[ds] = {"units": units, "families": fams}
    summary["cost"] = cost()
    summary["status"] = "INTERIM" if pending else "COMPLETE"
    summary["pending_families"] = pending
    summary["meta"] = {"schema": "v10.r22_pmi.summary.v1", "label": "PILOT/DEVELOPMENT; DEV descriptive, no selection after DEV",
                       "variants": {"J": "host J energy (per-window product reference)", "PMI": "mean 2 atanh(clip(T, +-(1-1e-4)))",
                                    "EPT": "mean T on aligned pairs"},
                       "thresholds": "BF err_full < 42 / < 9; EPFL < 10.5 / < 2.25 native px (= err_full < 42 / < 9)"}
    text = json.dumps(summary, indent=1, sort_keys=True, allow_nan=False) + "\n"
    if a.check:
        same = OUT_SUMMARY.exists() and OUT_SUMMARY.read_text() == text
        print("CHECK", "IDENTICAL" if same else "DIFFERENT")
        return 0 if same else 1
    for p, t in ((OUT_SUMMARY, text), (OUT_TABLES, tables(summary)),
                 (OUT_EPFL, json.dumps(epfl_private, indent=1, sort_keys=True) + "\n"),
                 (OUT_ALL, json.dumps(full, indent=1, sort_keys=True) + "\n")):
        p.parent.mkdir(parents=True, exist_ok=True); p.write_text(t)
    print("wrote", OUT_SUMMARY, OUT_TABLES, OUT_EPFL, OUT_ALL)
    return 0


if __name__ == "__main__":
    sys.exit(main())
