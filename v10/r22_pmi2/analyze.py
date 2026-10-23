#!/usr/bin/env python3
"""R22-C2 analysis (PILOT/DEVELOPMENT; DEV descriptive, nothing selected after DEV): every number from the per-registration rows.

  python3 v10/r22_pmi2/analyze.py            write the outputs below
  python3 v10/r22_pmi2/analyze.py --check    recompute and compare with the committed SUMMARY.json (exit 1 on any difference)
  python3 v10/r22_pmi2/analyze.py --partial  interim readout over the families whose 3 seeds are COMPLETE (status INTERIM)

Inputs: handoff/v10_r22_pmi2_20261023/runs/<ds>/<key>/seed_<s>/{CAL/step_*.jsonl, SELECTION.json, DEV.jsonl, COMPLETE.json} (rows
sha256 checked against COMPLETE.json; the PMI selection is recomputed from the CAL rows with the original rule and must equal
SELECTION.json).  DEV arms per registration: J_orig (J energy, originally J-selected checkpoint), PMI_orig (PMI energy, same
checkpoint, same energy instance), PMI_sel (PMI energy, PMI-selected checkpoint; = PMI_orig when both rules pick the same one),
stored (the historical J DEV row).
  total      PMI pipeline - J pipeline   = PMI_sel - J_orig
  readout    PMI_orig - J_orig           (energy at the J-selected checkpoint; R22-C's comparison, re-run in this job)
  selection  PMI_sel - PMI_orig          (checkpoint choice, PMI energy throughout);  total = readout + selection
  reeval     J_orig - stored             (same checkpoint, J re-evaluated: the re-evaluation reference)
Metrics as R22-C: BF err_full, success < 42 / < 9, registration-pooled; EPFL native px = err_full / 4, success < 10.5 / < 2.25 px,
DEV unit-macro over the 72 scene units (mean of unit success fractions; median of unit medians).  Two-way flips per registration.
Seed means; paired unit bootstrap (B 10000, seed 20261022, R22-C's procedure) of the 3-seed mean DEV success difference.
Outputs: v10/r22_pmi2/results/SUMMARY.json and TABLES.md (aggregates), v10/epfl/r22_pmi2/EPFL_PER_UNIT.json (EPFL scene units,
private), handoff/v10_r22_pmi2_20261023/ANALYSIS.json (everything).
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
BASE = PROJECT / "handoff/v10_r22_pmi2_20261023"
RUNS = BASE / "runs"
OUT_SUMMARY = WT / "v10/r22_pmi2/results/SUMMARY.json"
OUT_TABLES = WT / "v10/r22_pmi2/results/TABLES.md"
OUT_EPFL = WT / "v10/epfl/r22_pmi2/EPFL_PER_UNIT.json"
OUT_ALL = BASE / "ANALYSIS.json"
KEYS = ("MLPp1", "MLPp8", "InfoNCE")
SEEDS = (1001, 1002, 1003)
STEPS = tuple(range(250, 3001, 250))
N = {"bf": {"CAL": 80, "DEV": 320}, "epfl": {"CAL": 100, "DEV": 740}}
THR = (42.0, 9.0)
ARMS = ("J_orig", "PMI_orig", "PMI_sel", "stored")
CONTRASTS = {"total": ("J_orig", "PMI_sel"), "readout": ("J_orig", "PMI_orig"), "selection": ("PMI_orig", "PMI_sel"),
             "reeval": ("stored", "J_orig")}
B, BOOT_SEED = 10000, 20261022


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def cal_metrics(errs):
    """v10.supplement.records.metrics (the original CAL rule's metrics)."""
    return {"n": len(errs), "success42": sum(e < 42 for e in errs) / len(errs), "success9": sum(e < 9 for e in errs) / len(errs),
            "median_err": statistics.median(errs)}


def selection_key(c):
    m = c["metrics"]
    return -m["success42"], -m["success9"], m["median_err"], c["step"]


def unit_record(ds, key, seed):
    d = (WT / "v10/r22_pmi2/units") if ds == "bf" else (WT / "v10/epfl/r22_pmi2/units")
    return json.loads((d / f"r22_pmi2_{ds}_{key}_s{seed}.json").read_text())


def load(ds, key, seed):
    u = unit_record(ds, key, seed)
    d = RUNS / ds / key / f"seed_{seed}"
    c = json.loads((d / "COMPLETE.json").read_text())
    for name, h in c["rows_sha256"].items():
        if sha(d / name) != h:
            raise RuntimeError(f"{d}: {name} differs from COMPLETE.json")
    if set(c["rows_sha256"]) != {f"CAL/step_{s:04d}.jsonl" for s in STEPS} | {"DEV.jsonl"}:
        raise RuntimeError(f"{d}: COMPLETE.json does not list the 12 CAL files and DEV")
    k = "core" if ds == "bf" else "window"
    cal = {s: rows(d / f"CAL/step_{s:04d}.jsonl") for s in STEPS}
    for s, r in cal.items():
        if len(r) != N[ds]["CAL"] or len({(x[k], x["perturbation"]) for x in r}) != len(r) or any(x["step"] != s for x in r):
            raise RuntimeError(f"{d}: incomplete CAL step {s}")
    sel = json.loads((d / "SELECTION.json").read_text())
    cks = [{"step": s, "metrics": cal_metrics([x["E"]["PMI"]["err_full"] for x in cal[s]])} for s in STEPS]
    best = min(cks, key=selection_key)
    if best["step"] != sel["pmi_selected"]["step"] or best["metrics"] != sel["pmi_selected"]["metrics"] \
            or sel["original"]["step"] != u["original"]["step"]:
        raise RuntimeError(f"{d}: SELECTION.json does not follow the CAL rows")
    same = best["step"] == u["original"]["step"]
    dev = rows(d / "DEV.jsonl")
    arms = {"J_orig", "PMI_orig"} | (set() if same else {"PMI_sel"})
    if len(dev) != N[ds]["DEV"] or len({(x[k], x["perturbation"]) for x in dev}) != len(dev) or any(set(x["E"]) != arms for x in dev):
        raise RuntimeError(f"{d}: incomplete or inconsistent DEV rows")
    for x in dev:
        x["E"].setdefault("PMI_sel", x["E"]["PMI_orig"])
    return {"family": u["family"], "gpu": c["gpu"], "stored_gpu": u["stored_gpu"], "job": c["job"], "unit": u, "CAL": cal,
            "pmi_cal": cks, "selection": {"pmi_step": best["step"], "j_step": u["original"]["step"], "same": same}, "DEV": dev}


def err(x, a):
    return x["J_vs_stored"]["stored_err_full"] if a == "stored" else x["E"][a]["err_full"]


def reg_metrics(r, a, ds):
    e = [err(x, a) for x in r]
    m = {"n": len(e), "success42": sum(z < THR[0] for z in e) / len(e), "success9": sum(z < THR[1] for z in e) / len(e),
         "median_err_full": statistics.median(e)}
    if ds == "epfl":
        m["median_native_px"] = m["median_err_full"] / 4
    return m


def unit_macro(r, a):
    per = {}
    for x in r:
        per.setdefault(x["unit"], []).append(err(x, a))
    um = {u: {"n": len(e), "success42": sum(z < THR[0] for z in e) / len(e), "success9": sum(z < THR[1] for z in e) / len(e),
              "median_native_px": statistics.median(e) / 4} for u, e in sorted(per.items())}
    return {"n_units": len(um), "success42": statistics.fmean(m["success42"] for m in um.values()),
            "success9": statistics.fmean(m["success9"] for m in um.values()),
            "median_of_unit_medians_native_px": statistics.median(m["median_native_px"] for m in um.values())}, um


def flips(r, a, b):
    out = {}
    for name, t in (("lt42", THR[0]), ("lt9", THR[1])):
        fs = sum(err(x, a) >= t and err(x, b) < t for x in r); sf = sum(err(x, a) < t and err(x, b) >= t for x in r)
        out[name] = {"fail_to_success": fs, "success_to_fail": sf, "net": fs - sf}
    return out


def headline(ds, block, a):
    if ds == "epfl":
        m = block["unit_macro"][a]
        return {"s42": m["success42"], "s9": m["success9"], "median": m["median_of_unit_medians_native_px"]}
    m = block["registrations"][a]
    return {"s42": m["success42"], "s9": m["success9"], "median": m["median_err_full"]}


def analyse_unit(ds, data):
    r = data["DEV"]
    dev = {"registrations": {a: reg_metrics(r, a, ds) for a in ARMS},
           "flips": {c: flips(r, a, b) for c, (a, b) in CONTRASTS.items()},
           "J_vs_stored": {f: sum(x["J_vs_stored"][f] for x in r) for f in ("exact", "F_true_bitwise", "F_true_within_t01", "n_grid_equal")},
           "seconds": {a: sum(x["E"][a]["wall_seconds"] for x in r) for a in ("J_orig", "PMI_orig")}}
    # PMI_sel is an alias of PMI_orig (no separate registration) when both rules pick the same checkpoint
    dev["seconds"]["PMI_sel"] = 0.0 if data["selection"]["same"] else sum(x["E"]["PMI_sel"]["wall_seconds"] for x in r)
    per = {}
    if ds == "epfl":
        dev["unit_macro"] = {}
        for a in ARMS:
            dev["unit_macro"][a], per[a] = unit_macro(r, a)
        if dev["unit_macro"]["J_orig"]["n_units"] != 72:
            raise RuntimeError("EPFL DEV must have 72 scene units")
    dev["headline"] = {a: headline(ds, dev, a) for a in ARMS}
    dev["differences"] = {c: {m: dev["headline"][b][m] - dev["headline"][a][m] for m in ("s42", "s9", "median")}
                          for c, (a, b) in CONTRASTS.items()}
    hist = {k["step"]: k["stored_cal_metrics"] for k in data["unit"]["checkpoints"]}
    pmi = {c["step"]: c["metrics"] for c in data["pmi_cal"]}
    cid = [x["J_truth_vs_stored"] for s in STEPS for x in data["CAL"][s]]
    cal = {"selection": data["selection"],
           "pmi_cal_at": {"pmi_step": pmi[data["selection"]["pmi_step"]], "j_step": pmi[data["selection"]["j_step"]]},
           "j_cal_at": {"pmi_step": hist[data["selection"]["pmi_step"]], "j_step": hist[data["selection"]["j_step"]]},
           "pmi_cal_per_step": pmi, "j_cal_per_step_historical": hist,
           "J_truth_vs_stored": {"n": len(cid), **{f: sum(x[f] for x in cid) for f in ("F_true_bitwise", "F_true_within_t01", "n_grid_equal")},
                                 "max_abs_d_F_true": max(abs(x["d_F_true"]) for x in cid)},
           "seconds_PMI": sum(x["E"]["PMI"]["wall_seconds"] for s in STEPS for x in data["CAL"][s])}
    return {"family": data["family"], "gpu": data["gpu"], "stored_gpu": data["stored_gpu"], "job": data["job"],
            "same_gpu_type_as_stored": data["gpu"] == data["stored_gpu"], "CAL": cal, "DEV": dev}, per


def unit_success(data, a, thr):
    per = {}
    k = "core" if "core" in data["DEV"][0] else "unit"
    for x in data["DEV"]:
        per.setdefault(x[k], []).append(err(x, a) < thr)
    return {u: float(np.mean(s)) for u, s in per.items()}


def bootstrap(datas, key, a, b, thr):
    """R22-C's paired unit bootstrap of the 3-seed mean DEV success difference b - a (units: BF cores, EPFL scene units)."""
    D = []
    for s in SEEDS:
        ua, ub = unit_success(datas[(key, s)], a, thr), unit_success(datas[(key, s)], b, thr)
        D.append([ub[u] - ua[u] for u in sorted(ua)])
    D = np.array(D)
    g = np.random.default_rng(BOOT_SEED)
    idx = g.integers(0, D.shape[1], (B, D.shape[1]))
    boots = D[:, idx].mean(-1).mean(0)
    return {"estimate": float(D.mean(-1).mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "units": int(D.shape[1]), "B": B, "seed": BOOT_SEED}


def family_summary(per_seed, datas, key):
    f = {"same_checkpoint_seeds": sum(per_seed[s]["CAL"]["selection"]["same"] for s in SEEDS),
         "steps": {str(s): [per_seed[s]["CAL"]["selection"]["j_step"], per_seed[s]["CAL"]["selection"]["pmi_step"]] for s in SEEDS},
         "headline_mean": {a: {m: statistics.fmean(per_seed[s]["DEV"]["headline"][a][m] for s in SEEDS) for m in ("s42", "s9", "median")}
                           for a in ARMS}}
    for c in CONTRASTS:
        d = {m: [per_seed[s]["DEV"]["differences"][c][m] for s in SEEDS] for m in ("s42", "s9", "median")}
        f[c] = {**{f"{m}_per_seed": v for m, v in d.items()}, **{f"{m}_mean": statistics.fmean(v) for m, v in d.items()},
                "s9_seeds_positive": sum(x > 0 for x in d["s9"]), "s9_seeds_negative": sum(x < 0 for x in d["s9"])}
        f[c]["bootstrap_s9"] = bootstrap(datas, key, *CONTRASTS[c], THR[1])
        f[c]["bootstrap_s42"] = bootstrap(datas, key, *CONTRASTS[c], THR[0])
    return f


def cost():
    out = {"jobs": [], "gpu_hours": {}, "gpu_hours_total": 0.0}
    for p in sorted((BASE / "receipts").glob("*.json")):
        r = json.loads(p.read_text())
        gpu = (r.get("gpu") or "").split(",")[0]
        h = r["allocated_seconds"] / 3600
        out["jobs"].append({"job": r["job"], "name": r["name"], "gpu": gpu, "exit_code": r["exit_code"], "hours": h})
        out["gpu_hours"][gpu] = out["gpu_hours"].get(gpu, 0.0) + h
        out["gpu_hours_total"] += h
    return out


def dpp(x):
    return f"{100 * x:+.2f}"


def tables(summary):
    L = [f"状态：{summary['status']}" + (f"；尚未完成：{', '.join(summary['pending_families'])}" if summary["pending_families"] else ""),
         "", "全流程 = PMI 选 checkpoint + PMI 评 DEV；参照 = 原 J 选中的 checkpoint 用 J 评 DEV（同一作业）。"
         "读数 = 同一 checkpoint 上 PMI−J；选择 = PMI 下两个 checkpoint 之差；全流程 = 读数 + 选择。重评 = 本作业 J − 历史 J 记录。"]
    for ds, title, unit in (("bf", "BF-SHG", "err_full"), ("epfl", "EPFL RGB-NIR", "native px")):
        if not summary[ds]["families"]:
            continue
        s1, s2 = ("<42", "<9") if ds == "bf" else ("<10.5 px", "<2.25 px")
        agg = "registration-pooled" if ds == "bf" else "72 scene-unit macro；median of unit medians"
        L.append(f"\n### {title} DEV（{agg}；中位误差单位 {unit}）\n")
        L.append(f"| 家族 | seed | J/PMI 选中 step | J 全流程 {s2} | 全流程 {s2} (pp) | 读数 {s2} | 选择 {s2} | 全流程 {s1} | "
                 f"全流程 {s2} 翻转 失败→成功/成功→失败 | J 中位 | PMI 全流程中位 | 重评 {s2} 翻转 (净) |")
        L.append("|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|")
        for key in [k for k in KEYS if k in summary[ds]["families"]]:
            for s in SEEDS:
                u = summary[ds]["units"][f"{key}_s{s}"]
                sel, d, h = u["CAL"]["selection"], u["DEV"]["differences"], u["DEV"]["headline"]
                fl, fr = u["DEV"]["flips"]["total"]["lt9"], u["DEV"]["flips"]["reeval"]["lt9"]
                L.append(f"| {key} | {s} | {sel['j_step']}/{sel['pmi_step']}{'' if not sel['same'] else ' (同)'} | {100 * h['J_orig']['s9']:.2f} | "
                         f"{dpp(d['total']['s9'])} | {dpp(d['readout']['s9'])} | {dpp(d['selection']['s9'])} | {dpp(d['total']['s42'])} | "
                         f"{fl['fail_to_success']}/{fl['success_to_fail']} | {h['J_orig']['median']:.3f} | {h['PMI_sel']['median']:.3f} | "
                         f"{fr['fail_to_success']}/{fr['success_to_fail']} ({fr['net']:+d}) |")
            f = summary[ds]["families"][key]
            L.append(f"| **{key} 均值** | | 同 {f['same_checkpoint_seeds']}/3 | {100 * f['headline_mean']['J_orig']['s9']:.2f} | "
                     f"{dpp(f['total']['s9_mean'])} (+{f['total']['s9_seeds_positive']}/−{f['total']['s9_seeds_negative']}) | "
                     f"{dpp(f['readout']['s9_mean'])} | {dpp(f['selection']['s9_mean'])} | {dpp(f['total']['s42_mean'])} | | "
                     f"{f['headline_mean']['J_orig']['median']:.3f} | {f['headline_mean']['PMI_sel']['median']:.3f} | |")
        L.append(f"\n### {title} DEV：3 seed 均值差的配对单元 bootstrap（B={B}）\n")
        L.append(f"| 家族 | 全流程 {s2} [95%] | 读数 {s2} [95%] | 选择 {s2} [95%] | 全流程 {s1} [95%] | 重评 {s2} [95%] |")
        L.append("|---|---|---|---|---|---|")
        cell = lambda b: f"{dpp(b['estimate'])} [{dpp(b['ci95'][0])}, {dpp(b['ci95'][1])}]"  # noqa: E731
        for key in [k for k in KEYS if k in summary[ds]["families"]]:
            f = summary[ds]["families"][key]
            L.append(f"| {key} | {cell(f['total']['bootstrap_s9'])} | {cell(f['readout']['bootstrap_s9'])} | {cell(f['selection']['bootstrap_s9'])} | "
                     f"{cell(f['total']['bootstrap_s42'])} | {cell(f['reeval']['bootstrap_s9'])} |")
    c = summary["cost"]
    L.append(f"\n成本：{len(c['jobs'])} 次作业尝试，共 {c['gpu_hours_total']:.2f} GPU 小时（"
             + "，".join(f"{g} {h:.2f}" for g, h in sorted(c["gpu_hours"].items())) + "）。")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--partial", action="store_true")
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
                res, per = analyse_unit(ds, datas[(key, s)])
                per_seed[s] = res
                full[f"{ds}_{key}_s{s}"] = {**res, "_per_unit": per}
                units[f"{key}_s{s}"] = res
                if ds == "epfl":
                    epfl_private[f"{key}_s{s}"] = per
            fams[key] = family_summary(per_seed, datas, key)
        summary[ds] = {"units": units, "families": fams}
    summary["cost"] = cost()
    summary["status"] = "INTERIM" if pending else "COMPLETE"
    summary["pending_families"] = pending
    summary["meta"] = {"schema": "v10.r22_pmi2.summary.v1", "label": "PILOT/DEVELOPMENT; DEV descriptive, no selection after DEV",
                       "pmi": "mean 2 atanh(clip(T, +-(1-1e-4))) (R22-C)", "contrasts": {c: f"{b} - {a}" for c, (a, b) in CONTRASTS.items()},
                       "selection_rule": "min (-success42, -success9, median_err, step) over the 12 CAL checkpoints",
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
