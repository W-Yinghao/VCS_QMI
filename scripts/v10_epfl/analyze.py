#!/usr/bin/env python3
"""Official EPFL V10 results analysis (frozen EPFL protocol, EPFL_PROTOCOL_FROZEN.json; frozen group record v2).  CPU, result files only.

Reads only JSON / JSONL result and record files: the finals' COMPLETE.json, DEV.jsonl and ENV_<job>.json, the CAL SELECTION.json files, the
exit receipts and cost ledgers of the releases, the frozen records in v10/epfl.  It reads no image, no cache file and no held-out data (EPFL has
no TEST role; BF TEST134 is never touched), and submits nothing.

  python -I scripts/v10_epfl/analyze.py            -> <analysis>/EPFL_RESULTS.json and EPFL_RESULTS_CN.md
  python -I scripts/v10_epfl/analyze.py --check    -> recompute and compare with the stored JSON (exit 0 iff identical and inputs unchanged)

Hardware replicates (post-hoc hardware sensitivity diagnostic, EPFL_HW_REPLICATE_20261020): for each RTX PRO 6000 final of d9d67cf2 rerun on
L40S under the replicate release (<hw2 bundle>/hw_replicates), the pair RTX run vs L40S rerun of the same seed: per registration (written to
<hw_replicates>/analysis/HW_PAIR_<family>_seed_<seed>.jsonl), the unit-macro differences L40S - RTX on every set with paired unit bootstrap
intervals (the families' own draws), the per-registration and per-unit agreement, the CAL trajectories / selected checkpoints / model hashes,
the training-log divergence and the L40S seed spread.  Reported beside the primary results; the primary values never change.

Statistics (frozen protocol): per final, on each sensitivity set of the group record (DEV72_primary, DEV_minus_weaker_relations,
DEV_minus_boundary_and_saturated, DEV_minus_all): per EPFL scene unit the fractions err_full < 42 and < 9 (internal units = 10.5 / 2.25 native px)
and the median err_full over all the unit's registrations (windows x perturbations; perturbations within a unit are not independent); the
unit-macro = mean over units (success) / median over units (median); registration-level pooled values beside them; per category.  Family values
are seed means (training-free families have one configuration and one final).  Bootstrap: units resampled with replacement, the same draws for
every family (paired by unit), B = 10000, numpy default_rng(20261020), percentile 95% intervals - interval disclosure only, no tests.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO.parent
HANDOFF = PROJECT / "handoff"
PARENT = HANDOFF / "v10_epfl_production_20261019"
BUNDLES = {"3dae6434": PARENT, "7cbd1c21": HANDOFF / "v10_epfl_production_hw_20261019", "d9d67cf2": HANDOFF / "v10_epfl_production_hw2_20261020"}
R2 = {1024: HANDOFF / "v10_epfl_rulsif_v2_T1024_20261020", 2048: HANDOFF / "v10_epfl_rulsif_v2_T2048_20261020"}
EPFL = REPO / "v10" / "epfl"
OUT = PARENT / "analysis"
SETS = ("DEV72_primary", "DEV_minus_weaker_relations", "DEV_minus_boundary_and_saturated", "DEV_minus_all")
FAMILIES = ("MF_VCS_perm1", "MF_VCS_M0", "MF_VCS_M1", "MLP_VCS_perm1", "MLP_VCS_perm8", "InfoNCE_host[J_perm1]", "MI_native", "NMI_native",
            "CoMIR_native", "MIND_2D_adapted")
TRAINING_FREE = ("MI_native", "NMI_native", "MIND_2D_adapted")
B_BOOT, BOOT_SEED = 10000, 20261020
L40S = "NVIDIA L40S"
OWNER = {"2026-10-19": "final 放开，CAL 留 L40S", "2026-10-20": "什么快什么来吧"}
REPLICATES = BUNDLES["d9d67cf2"] / "hw_replicates"          # post-hoc hardware sensitivity diagnostic (its own release and results root)
REPLICATE_SCHEMA = "v10.epfl.release_hw_replicate.v1"
CONTROLS = BUNDLES["d9d67cf2"] / "hw_controls"              # its L40S-vs-L40S control (its own release and results root)
CONTROL_SCHEMA = "v10.epfl.release_hw_control.v1"
EVAL_NOISE = HANDOFF / "v10_epfl_eval_noise_20261020"        # post-hoc evaluation-noise diagnostic (RELEASE_<ref8>.json, results/, exits/)
EVAL_NOISE_SCHEMA, EVN_LABEL = "v10.epfl.release_eval_noise.v1", "POST-HOC EVALUATION-NOISE DIAGNOSTIC"
RULSIF_T1024 = "RuLSIF_exact_v2[T1024]"
DIAG_LABEL = "POST-HOC HARDWARE SENSITIVITY DIAGNOSTIC"


class Inputs:
    """Every file read, with its sha256 (the --check mode verifies them)."""

    def __init__(self):
        self.files = {}

    def json(self, path):
        path = Path(path); data = path.read_bytes(); self.files[str(path)] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    def jsonl(self, path):
        path = Path(path); data = path.read_bytes(); self.files[str(path)] = hashlib.sha256(data).hexdigest()
        return [json.loads(line) for line in data.decode().splitlines() if line.strip()]


def unit_values(rows, windows, unit_of):
    """{unit: (success42, success9, median err_full, n)} over the rows of the given windows."""
    per = {}
    for r in rows:
        if r["window"] in windows:
            per.setdefault(unit_of[r["window"]], []).append(float(r["err_full"]))
    return {u: (sum(e < 42 for e in v) / len(v), sum(e < 9 for e in v) / len(v), statistics.median(v), len(v)) for u, v in sorted(per.items())}


def set_stats(rows, s, unit_of):
    windows = set(s["windows"]); uv = unit_values(rows, windows, unit_of)
    if sorted(uv) != sorted(s["units"]):
        raise RuntimeError("DEV rows do not cover the set's units")
    regs = [float(r["err_full"]) for r in rows if r["window"] in windows]
    med_u = statistics.median(v[2] for v in uv.values())
    return {"n_units": len(uv), "n_registrations": len(regs),
            "unit_macro": {"success42": statistics.fmean(v[0] for v in uv.values()), "success9": statistics.fmean(v[1] for v in uv.values()),
                           "median_err_full": med_u, "median_native_px": med_u / 4},
            "registration_level": {"success42": sum(e < 42 for e in regs) / len(regs), "success9": sum(e < 9 for e in regs) / len(regs),
                                   "median_err_full": statistics.median(regs), "median_native_px": statistics.median(regs) / 4},
            "_units": uv}


def receipts(inp):
    out = {}
    for tag, b in BUNDLES.items():
        for p in sorted((b / "exits").glob("*.json")) if (b / "exits").exists() else []:
            r = inp.json(p); out[(tag, r["job"])] = r
    return out


def final_record(inp, unit_dir, group, rec, release_tag):
    c = inp.json(unit_dir / "COMPLETE.json")
    rows = inp.jsonl(unit_dir / "DEV.jsonl")
    dev_w = set(group["roles"]["DEV"])
    keys = {(r["window"], r["perturbation"]) for r in rows}
    if len(rows) != 740 or len(keys) != 740 or {w for w, _ in keys} != dev_w or any(r.get("role") != "DEV" for r in rows):
        raise RuntimeError(f"{unit_dir}: DEV rows are not the complete 74 x 10 block")
    if {r["checkpoint_sha256"] for r in rows} != {c["selected_checkpoint_sha256"]}:
        raise RuntimeError(f"{unit_dir}: DEV rows not at the selected checkpoint")
    env_path = unit_dir / f"ENV_{c['job']}.json"
    env = inp.json(env_path) if env_path.exists() else {}
    hw = c.get("hardware") or {"gpu_name": env.get("gpu_name"), "node": env.get("slurm_nodelist") or env.get("host"), "partition": env.get("slurm_partition")}
    r = rec.get((release_tag, c["job"])) or {}
    stats = {name: set_stats(rows, group["sensitivity_sets"][name], group["window_unit"]) for name in SETS}
    mism = []
    for name in SETS:     # the run-time DEV statistics (production.dev_statistics) must agree with this recomputation
        rt = (c.get("DEV_statistics") or {}).get(name) or {}
        um = rt.get("unit_macro") or {}
        if um and (abs(um["success42"] - stats[name]["unit_macro"]["success42"]) > 1e-12 or abs(um["success9"] - stats[name]["unit_macro"]["success9"]) > 1e-12
                   or abs(um["median_of_unit_medians_native_px"] - stats[name]["unit_macro"]["median_native_px"]) > 1e-9):
            mism.append(name)
    return {"job": c["job"], "release": release_tag, "hardware": {k: hw.get(k) for k in ("gpu_name", "node", "partition")},
            "allocated_hours": (r.get("allocated_seconds") or c.get("allocated_seconds_this_job") or 0) / 3600 or None,
            "selected": {"candidate": c["identity"]["candidate"], "step": c["selected"]["step"], "cal_metrics": c["selected"]["metrics"]},
            "stats": stats, "runtime_statistics_agree": not mism, "runtime_mismatch_sets": mism,
            "per_category": per_category(stats["DEV72_primary"]["_units"])}


def per_category(uv):
    cats = {}
    for u, v in uv.items():
        cats.setdefault(u.split("/")[0], []).append(v)
    return {c: {"n_units": len(v), "success42": statistics.fmean(x[0] for x in v), "success9": statistics.fmean(x[1] for x in v),
                "median_err_full": statistics.median(x[2] for x in v)} for c, v in sorted(cats.items())}


def family_summary(finals, sets_units):
    """Seed means (unit-macro and registration-level), per seed, and the paired unit bootstrap of the seed means."""
    import numpy as np
    done = {s: f for s, f in finals.items() if f is not None}
    out = {"seeds_complete": sorted(done), "per_seed": {}, "seed_mean": {}, "bootstrap95": {}}
    if not done:
        return out
    for s, f in done.items():
        out["per_seed"][str(s)] = {name: {"unit_macro": f["stats"][name]["unit_macro"], "registration_level": f["stats"][name]["registration_level"]}
                                   for name in SETS}
    cats = sorted({c for f in done.values() for c in f["per_category"]})
    out["per_category_seed_mean"] = {c: {k: statistics.fmean(f["per_category"][c][k] for f in done.values()) for k in ("success42", "success9", "median_err_full")}
                                     | {"n_units": next(iter(done.values()))["per_category"][c]["n_units"]} for c in cats}
    rng = np.random.default_rng(BOOT_SEED)
    for name in SETS:
        units = sets_units[name]
        idx = rng.integers(0, len(units), size=(B_BOOT, len(units)))
        arr = {k: np.array([[f["stats"][name]["_units"][u][k] for u in units] for f in done.values()]) for k in (0, 1, 2)}   # [seeds, units]
        mean = {"success42": float(arr[0].mean()), "success9": float(arr[1].mean()), "median_err_full": float(np.mean(np.median(arr[2], axis=1)))}
        mean["median_native_px"] = mean["median_err_full"] / 4
        reg = {k: statistics.fmean(f["stats"][name]["registration_level"][k] for f in done.values()) for k in ("success42", "success9", "median_err_full")}
        reg["median_native_px"] = reg["median_err_full"] / 4
        out["seed_mean"][name] = {"unit_macro": mean, "registration_level": reg}
        b42 = arr[0][:, idx].mean(axis=2).mean(axis=0); b9 = arr[1][:, idx].mean(axis=2).mean(axis=0)
        bmed = np.median(arr[2][:, idx], axis=2).mean(axis=0)
        q = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]   # noqa: E731
        out["bootstrap95"][name] = {"success42": q(b42), "success9": q(b9), "median_err_full": q(bmed), "median_native_px": [v / 4 for v in q(bmed)]}
    return out


def strata(finals):
    done = {s: f for s, f in finals.items() if f is not None}
    l40 = {s: f for s, f in done.items() if f["hardware"]["gpu_name"] == L40S}
    m = lambda fs, k: statistics.fmean(f["stats"]["DEV72_primary"]["unit_macro"][k] for f in fs.values()) if fs else None   # noqa: E731
    return {"hardware_per_seed": {str(s): f["hardware"]["gpu_name"] for s, f in done.items()},
            "pooled": {k: m(done, k) for k in ("success42", "success9", "median_err_full")},
            "L40S_only": {k: m(l40, k) for k in ("success42", "success9", "median_err_full")}, "n_pooled": len(done), "n_L40S": len(l40)}


def selected_cal_metrics(inp, results_root, family, sel):
    """The selected candidate's CAL read-out (n, <42, <9, median err_full, checkpoint step) from SELECTION.json, verified against the selected
    candidate's COMPLETE.json (sha256 = the selection's complete_sha256, and its own selected checkpoint = the choice)."""
    chosen = sel["selected"]; choice = chosen["choice"]; m = choice["metrics"]
    cp = Path(results_root) / "CAL" / family / f"candidate_{chosen['candidate']['order']:02d}" / "COMPLETE.json"
    comp = inp.json(cp)
    ok = (inp.files[str(cp)] == chosen["complete_sha256"] and comp["selected"]["step"] == choice["step"] and comp["selected"]["metrics"] == m
          and comp["selected"]["checkpoint_sha256"] == choice["checkpoint_sha256"])
    if not ok:
        raise RuntimeError(f"{family}: CAL selection does not match its candidate COMPLETE.json")
    return {"n": m["n"], "success42": m["success42"], "success9": m["success9"], "median_err": m["median_err"], "step": choice["step"],
            "checkpoint_sha256": choice["checkpoint_sha256"], "candidate_complete_sha256_verified": True}


def cal_info(inp, results_root, family, rec, ledger_jobs, predicted):
    p = Path(results_root) / "CAL" / family / "SELECTION.json"
    if not p.exists():
        return {"status": "PENDING"}
    sel = inp.json(p)
    jobs = sorted(ledger_jobs.get((family, "CAL"), set()))
    hours = sum((rec.get(("3dae6434", j)) or {}).get("allocated_seconds", 0) for j in jobs) / 3600 if jobs else None
    return {"status": "SELECTED", "selected": sel["selected"]["candidate"], "selected_step": sel["selected"]["choice"]["step"],
            "selected_cal_metrics": selected_cal_metrics(inp, results_root, family, sel),
            "completed_candidates": len(sel["completed_candidates"]), "planned_candidates": sel["planned_candidates"],
            "budget_exhausted": sel["budget_exhausted"], "cal_jobs": jobs, "cal_allocated_l40s_hours": hours, "predicted": predicted}


def boot_draws(sets_units):
    """The families' bootstrap draws per set (family_summary draws them in SETS order from one default_rng(BOOT_SEED))."""
    import numpy as np
    rng = np.random.default_rng(BOOT_SEED)
    return {name: rng.integers(0, len(sets_units[name]), size=(B_BOOT, len(sets_units[name]))) for name in SETS}


def _q(x):
    import numpy as np
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


UM = ("success42", "success9", "median_err_full")
ROW_FIELDS = ("theta", "t", "theta_hat", "t_hat", "err_full", "rot_err_deg", "trans_err_full", "F_hat", "F_true", "n_grid")   # a bitwise row


def unit_macro_difference(prim_units, rep_units, units, idx):
    """rerun - primary unit-macro differences on one set, with the paired unit bootstrap interval (the same unit draws for both runs)."""
    import numpy as np
    a = {k: np.array([prim_units[u][k] for u in units]) for k in (0, 1, 2)}
    b = {k: np.array([rep_units[u][k] for u in units]) for k in (0, 1, 2)}
    val = lambda d: {"success42": float(d[0].mean()), "success9": float(d[1].mean()), "median_err_full": float(np.median(d[2]))}   # noqa: E731
    pv, rv = val(a), val(b)
    delta = {k: rv[k] - pv[k] for k in pv}
    d42 = (b[0] - a[0])[idx].mean(axis=1); d9 = (b[1] - a[1])[idx].mean(axis=1)
    dmed = np.median(b[2][idx], axis=1) - np.median(a[2][idx], axis=1)
    return {"primary": pv, "rerun": rv, "difference_rerun_minus_primary": delta, "abs_difference": {k: abs(v) for k, v in delta.items()},
            "bootstrap95": {"success42": _q(d42), "success9": _q(d9), "median_err_full": _q(dmed)},
            "units_differing": {"success42": int(np.sum(a[0] != b[0])), "success9": int(np.sum(a[1] != b[1])), "median_err_full": int(np.sum(a[2] != b[2]))},
            "n_units": len(units)}


def per_registration(fam, seed, prim_rows, rep_rows):
    """Rows joined by (window, perturbation); the truth (theta, t) must be identical (the same registration problems)."""
    import math
    a = {(r["window"], r["perturbation"]): r for r in prim_rows}; b = {(r["window"], r["perturbation"]): r for r in rep_rows}
    if sorted(a) != sorted(b):
        raise RuntimeError(f"{fam} seed {seed}: the two DEV blocks cover different registrations")
    out = []
    for k in sorted(a):
        x, y = a[k], b[k]
        if x["theta"] != y["theta"] or x["t"] != y["t"] or x["unit"] != y["unit"]:
            raise RuntimeError(f"{fam} seed {seed} {k}: different registration problem (truth differs)")
        dt = [y["t_hat"][i] - x["t_hat"][i] for i in range(2)]
        out.append({"family": fam, "seed": seed, "window": k[0], "unit": x["unit"], "perturbation": k[1],
                    "err_full_primary": x["err_full"], "err_full_rerun": y["err_full"], "delta_err_full": y["err_full"] - x["err_full"],
                    "success42_primary": x["err_full"] < 42, "success42_rerun": y["err_full"] < 42, "success9_primary": x["err_full"] < 9,
                    "success9_rerun": y["err_full"] < 9, "theta_hat_primary": x["theta_hat"], "theta_hat_rerun": y["theta_hat"],
                    "delta_theta_hat_deg": y["theta_hat"] - x["theta_hat"], "t_hat_primary": x["t_hat"], "t_hat_rerun": y["t_hat"],
                    "delta_t_hat_norm": math.hypot(*dt), "F_hat_primary": x["F_hat"], "F_hat_rerun": y["F_hat"],
                    "estimate_bitwise_equal": x["theta_hat"] == y["theta_hat"] and x["t_hat"] == y["t_hat"], "F_true_equal": x["F_true"] == y["F_true"],
                    "row_bitwise_equal": all(x[f] == y[f] for f in ROW_FIELDS)})
    return out


def agreement(rows):
    import numpy as np
    n = len(rows); d = np.array([r["delta_err_full"] for r in rows]); ad = np.abs(d)

    def table(k):
        both = sum(r[k + "_primary"] and r[k + "_rerun"] for r in rows); neither = sum(not r[k + "_primary"] and not r[k + "_rerun"] for r in rows)
        p_only = sum(r[k + "_primary"] and not r[k + "_rerun"] for r in rows); r_only = sum(r[k + "_rerun"] and not r[k + "_primary"] for r in rows)
        return {"both": both, "neither": neither, "primary_only": p_only, "rerun_only": r_only, "disagree": p_only + r_only,
                "agreement": (both + neither) / n}
    return {"n_registrations": n, "estimate_bitwise_equal": sum(r["estimate_bitwise_equal"] for r in rows),
            "F_true_equal": sum(r["F_true_equal"] for r in rows), "row_bitwise_equal": sum(r["row_bitwise_equal"] for r in rows),
            "delta_err_full": {"mean": float(d.mean()), "median": float(np.median(d)), "median_abs": float(np.median(ad)),
                               "p95_abs": float(np.percentile(ad, 95)), "max_abs": float(ad.max())},
            "abs_delta_err_full_at_most": {"0.1": int(np.sum(ad <= 0.1)), "1": int(np.sum(ad <= 1.0)), "9": int(np.sum(ad <= 9.0))},
            "delta_theta_hat_deg_max_abs": float(max(abs(r["delta_theta_hat_deg"]) for r in rows)),
            "delta_t_hat_norm_max": float(max(r["delta_t_hat_norm"] for r in rows)),
            "success42": table("success42"), "success9": table("success9")}


def training_divergence(prim_steps, rep_steps):
    a = {r["step"]: r for r in prim_steps}; b = {r["step"]: r for r in rep_steps}
    steps = sorted(set(a) & set(b))
    diff = [s for s in steps if a[s]["loss"] != b[s]["loss"]]
    dl = [abs(b[s]["loss"] - a[s]["loss"]) for s in steps]
    return {"steps_compared": len(steps), "steps_primary": len(a), "steps_rerun": len(b), "first_step_loss_differs": diff[0] if diff else None,
            "steps_loss_bitwise_equal": len(steps) - len(diff), "max_abs_delta_loss": max(dl) if dl else None,
            "loss_step_1": [a[1]["loss"], b[1]["loss"]] if 1 in a and 1 in b else None,
            "loss_last": [a[steps[-1]]["loss"], b[steps[-1]]["loss"]] if steps else None}


def cal_trajectory(inp, prim_dir, rep_dir, prim_c, rep_c):
    out = []
    rc = {c["step"]: c for c in rep_c["cal_checkpoints"]}
    for c in prim_c["cal_checkpoints"]:
        s = c["step"]; r = rc[s]
        ma = inp.json(prim_dir / "checkpoints" / f"step_{s:04d}.json")["model_sha256"]
        mb = inp.json(rep_dir / "checkpoints" / f"step_{s:04d}.json")["model_sha256"]
        out.append({"step": s, "primary": c["metrics"], "rerun": r["metrics"], "model_sha256_equal": ma == mb})
    return out


def diagnostic_pairs(inp, bundle, schema, group, sets_units, families, problems):
    """The pairs of one post-hoc hardware diagnostic release (the RTX-vs-L40S replicates or their L40S-vs-L40S control): primary unit vs
    its L40S rerun of the same seed; PENDING until the rerun unit is COMPLETE."""
    rp = Path(bundle) / "RELEASE.json"
    if not rp.exists():
        return {"status": "NO_RELEASE"}, {}
    rel = inp.json(rp); rsha = inp.files[str(rp)]
    if rel.get("schema") != schema:
        raise RuntimeError(f"{rp} is not a release of schema {schema}")
    rec = {}
    for p in sorted((Path(bundle) / "exits").glob("*.json")) if (Path(bundle) / "exits").exists() else []:
        r = inp.json(p); rec[("diag", r["job"])] = r
    draws = boot_draws(sets_units); pairs, files = {}, {}
    for key, pair in rel["pairs"].items():
        fam, seed = pair["family"], pair["seed"]
        prim_dir = Path(pair["primary_unit"]); rep_dir = Path(rel["results_root"]) / "FINAL_DEV" / fam / f"seed_{seed}"
        prim_c = inp.json(prim_dir / "COMPLETE.json")
        if inp.files[str(prim_dir / "COMPLETE.json")] != pair["primary_complete_sha256"]:
            problems.append(f"hardware diagnostic {key} ({schema}): the primary COMPLETE.json changed since the release")
        base = {"family": fam, "seed": seed, "primary_job": pair["primary_job"], "primary_gpu": pair["primary_gpu"],
                "rerun_gpu": pair["replicate_gpu"], "rerun_unit": str(rep_dir)}
        if not (rep_dir / "COMPLETE.json").exists():
            pairs[key] = {**base, "status": "PENDING (rerun not COMPLETE)" + (" - started" if rep_dir.exists() else "")}
            continue
        rep_c = inp.json(rep_dir / "COMPLETE.json"); idn = rep_c["identity"]; pidn = prim_c["identity"]
        same = {k: idn.get(k) == pidn.get(k) for k in ("family", "candidate", "seed", "phase", "schedule", "data_lock_sha256", "group_record_sha256",
                                                        "selection_dependency_sha256")}
        ok = (all(same.values()) and idn.get("release_sha256") == rsha and (idn.get("diagnostic") or {}).get("kind") == DIAG_LABEL
              and (idn.get("hardware") or {}).get("gpu_name") == L40S and (rep_c.get("hardware") or {}).get("gpu_name") == L40S)
        if not ok:
            problems.append(f"hardware diagnostic {key} ({schema}): identity is not the same unit on L40S under its release ({same})")
        receipt = rec.get(("diag", rep_c["job"]))
        prim = final_record(inp, prim_dir, group, receipts(inp), "d9d67cf2")
        rep = final_record(inp, rep_dir, group, rec, "diag")
        rows = per_registration(fam, seed, inp.jsonl(prim_dir / "DEV.jsonl"), inp.jsonl(rep_dir / "DEV.jsonl"))
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        fpath = Path(bundle) / "analysis" / f"HW_PAIR_{fam}_seed_{seed}.jsonl"; files[str(fpath)] = text
        diffs = {name: unit_macro_difference(prim["stats"][name]["_units"], rep["stats"][name]["_units"], sets_units[name], draws[name]) for name in SETS}
        entry = {**base, "status": "COMPLETE" + ("" if receipt and receipt.get("exit_code") == 0 else " (exit receipt pending)"),
                 "identity_checks": {"same_unit_fields": same, "release_sha256_is_diagnostic": idn.get("release_sha256") == rsha,
                                     "diagnostic_label": idn.get("diagnostic"), "gpu_l40s": (idn.get("hardware") or {}).get("gpu_name") == L40S},
                 "primary": {"job": prim["job"], "hardware": prim["hardware"], "selected": prim["selected"], "allocated_hours": prim["allocated_hours"]},
                 "rerun": {"job": rep["job"], "hardware": rep["hardware"], "selected": rep["selected"], "allocated_hours": rep["allocated_hours"],
                           "exit_code": (receipt or {}).get("exit_code"), "runtime_statistics_agree": rep["runtime_statistics_agree"]},
                 "selected_step": {"primary": prim["selected"]["step"], "rerun": rep["selected"]["step"]},
                 "unit_macro_difference": diffs, "per_registration_agreement": agreement(rows),
                 "per_registration_file": {"path": str(fpath), "sha256": hashlib.sha256(text.encode()).hexdigest(), "rows": len(rows)},
                 "cal_trajectory": cal_trajectory(inp, prim_dir, rep_dir, prim_c, rep_c),
                 "training_log": training_divergence(inp.jsonl(prim_dir / "TRAIN_STEPS.jsonl"), inp.jsonl(rep_dir / "TRAIN_STEPS.jsonl"))}
        if schema == REPLICATE_SCHEMA and fam in families:     # the L40S seeds of the family beside the RTX seed (numbers only)
            fin = families[fam]["finals"]
            l40 = {s: f["stats"]["DEV72_primary"]["unit_macro"] for s, f in fin.items() if f and f["hardware"]["gpu_name"] == L40S}
            l40_all = {f"{seed}_rerun": rep["stats"]["DEV72_primary"]["unit_macro"], **l40}
            mean = lambda vs, k: statistics.fmean(v[k] for v in vs)   # noqa: E731
            entry["seed_context_DEV72_primary"] = {
                "rtx_primary": prim["stats"]["DEV72_primary"]["unit_macro"], "l40s_seeds": l40_all,
                "rerun_minus_primary_same_seed": diffs["DEV72_primary"]["difference_rerun_minus_primary"],
                "primary_minus_mean_of_primary_l40s_seeds": {k: prim["stats"]["DEV72_primary"]["unit_macro"][k] - mean(l40.values(), k) for k in UM} if l40 else None,
                "rerun_minus_mean_of_primary_l40s_seeds": {k: rep["stats"]["DEV72_primary"]["unit_macro"][k] - mean(l40.values(), k) for k in UM} if l40 else None,
                "l40s_seed_range_with_rerun": {k: max(v[k] for v in l40_all.values()) - min(v[k] for v in l40_all.values()) for k in UM},
                "diagnostic_l40s_seed_mean_with_rerun": {k: mean(l40_all.values(), k) for k in UM},
                "primary_seed_mean_pooled": families[fam]["strata"]["pooled"], "primary_l40s_only": families[fam]["strata"]["L40S_only"],
                "note": "arithmetic only: (primary - mean of the primary L40S seeds) = (primary - rerun) + (rerun - mean of the primary L40S "
                        "seeds); the diagnostic mean is never a primary value"}
        pairs[key] = entry
        if not rep["runtime_statistics_agree"]:
            problems.append(f"hardware diagnostic {key} ({schema}): run-time DEV statistics differ from the recomputation")
    return {"status": "COMPLETE" if all(v["status"].startswith("COMPLETE") for v in pairs.values()) else "PENDING",
            "role": rel.get("role"), "label": rel["label"], "release_sha256": rsha, "owner_rules_verbatim": rel["owner_rules_verbatim"],
            "record": rel.get("replicate_record") or rel.get("control_record"), "results_root": rel["results_root"], "partition": rel["partition"],
            "exclude": rel["exclude"], "bootstrap": "paired by unit, the families' draws per set (B 10000, default_rng(20261020)); percentile 95%",
            "pairs": pairs}, files


def replicate_pairs(inp, group, sets_units, families, problems):
    return diagnostic_pairs(inp, REPLICATES, REPLICATE_SCHEMA, group, sets_units, families, problems)


def control_pairs(inp, group, sets_units, families, problems):
    return diagnostic_pairs(inp, CONTROLS, CONTROL_SCHEMA, group, sets_units, families, problems)


def eval_noise_unit(inp, rsha, rel, key, spec, group, sets_units, draws, rec, problems):
    """One recorded final vs its re-evaluation (eval-noise diagnostic); PENDING / FAIL until a verified COMPLETE re-evaluation exists."""
    fam, seed = spec["family"], spec["seed"]; u = Path(spec["recorded_unit"]); out = Path(spec["out"])
    rec_c = inp.json(u / "COMPLETE.json")
    if inp.files[str(u / "COMPLETE.json")] != spec["complete_sha256"]:
        problems.append(f"eval noise {key}: the recorded COMPLETE.json changed since the release")
    base = {"family": fam, "seed": seed, "release": rsha[:8], "recorded_job": spec["recorded_job"], "gpu_name": spec["gpu_name"],
            "recorded_selected_step": spec["selected_step"]}
    fails = sorted(out.glob("FAIL_*.json")) if out.exists() else []
    for f in fails:
        inp.json(f)
    if fails:                                         # a failed bitwise check: no result of this unit is used (sticky, no retry)
        return {**base, "status": "FAIL (no result used): " + ", ".join(f.stem for f in fails)}, None
    if not (out / "COMPLETE.json").exists():
        return {**base, "status": "PENDING (started)" if out.exists() else "PENDING"}, None
    c = inp.json(out / "COMPLETE.json"); idn = c["identity"]; dg = idn.get("diagnostic") or {}
    new = inp.jsonl(out / "DEV.jsonl"); old = inp.jsonl(u / "DEV.jsonl")
    dev_w = set(group["roles"]["DEV"]); keys = {(r["window"], r["perturbation"]) for r in new}
    ok = (c.get("status") == "COMPLETE" and dg.get("kind") == EVN_LABEL and dg.get("release_sha256") == rsha
          and {k: v for k, v in idn.items() if k != "diagnostic"} == rec_c["identity"] and (c.get("hardware") or {}).get("gpu_name") == spec["gpu_name"]
          and all(x["status"] == "PASS" for x in c.get("checks", [])) and c["artifacts"].get("DEV.jsonl") == inp.files[str(out / "DEV.jsonl")]
          and len(new) == len(keys) == 740 and {w for w, _ in keys} == dev_w
          and {r["checkpoint_sha256"] for r in new} == {spec["selected_checkpoint_sha256"]})
    receipt = rec.get(c["job"]) or {}
    if not ok or receipt.get("exit_code") != 0:
        problems.append(f"eval noise {key}: the re-evaluation is not a verified COMPLETE unit (identity, checks, rows, GPU or exit receipt)")
        return {**base, "status": "UNVERIFIED (not used)"}, None
    stats_old = {name: set_stats(old, group["sensitivity_sets"][name], group["window_unit"]) for name in SETS}
    stats_new = {name: set_stats(new, group["sensitivity_sets"][name], group["window_unit"]) for name in SETS}
    rows = per_registration(fam, seed, old, new)
    ag = agreement(rows)
    if ag["F_true_equal"] != 740:
        problems.append(f"eval noise {key}: F_true differs in {740 - ag['F_true_equal']} rows despite the job's bitwise checks")
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    fpath = EVAL_NOISE / "analysis" / f"EVN_{fam}_seed_{seed}.jsonl"
    entry = {**base, "status": "COMPLETE", "job": c["job"], "hardware": c.get("hardware"), "allocated_hours": (receipt.get("allocated_seconds") or 0) / 3600,
             "unit_macro_difference": {name: unit_macro_difference(stats_old[name]["_units"], stats_new[name]["_units"], sets_units[name], draws[name])
                                       for name in SETS},
             "registration_level_difference": {name: {k: stats_new[name]["registration_level"][k] - stats_old[name]["registration_level"][k] for k in
                                                      ("success42", "success9", "median_err_full")} for name in SETS},
             "per_registration_agreement": ag, "per_registration_file": {"path": str(fpath), "sha256": hashlib.sha256(text.encode()).hexdigest(), "rows": len(rows)}}
    return entry, (str(fpath), text)


def eval_noise(inp, group, sets_units, families, rulsif, problems):
    """Post-hoc evaluation-noise diagnostic (EPFL_EVAL_NOISE_20261020): recorded vs re-evaluated DEV of every final; per family the noise floor
    (max |re-evaluated - recorded| unit-macro over its finals, each metric and set) beside the between-family differences of the seed means."""
    rels = sorted(EVAL_NOISE.glob("RELEASE_*.json"))
    if not rels:
        return {"status": "NO_RELEASE"}, {}
    rec = {}
    for p in sorted((EVAL_NOISE / "exits").glob("*.json")) if (EVAL_NOISE / "exits").exists() else []:
        r = inp.json(p); rec[r["job"]] = r
    draws = boot_draws(sets_units); units, files, releases = {}, {}, {}
    for rp in rels:
        rel = inp.json(rp); rsha = inp.files[str(rp)]
        if rel.get("schema") != EVAL_NOISE_SCHEMA:
            raise RuntimeError(f"{rp} is not an eval-noise release")
        releases[rp.name] = {"sha256": rsha, "reference_commit": rel["reference_commit"], "recorded_release": rel["recorded_release"]["sha256"],
                             "units": len(rel["units"])}
        for key, spec in rel["units"].items():
            units[key], f = eval_noise_unit(inp, rsha, rel, key, spec, group, sets_units, draws, rec, problems)
            if f:
                files[f[0]] = f[1]
    fams = sorted({v["family"] for v in units.values()})
    floors = {}
    for fam in fams:
        mine = [v for v in units.values() if v["family"] == fam]; done = [v for v in mine if v["status"] == "COMPLETE"]
        name = RULSIF_T1024 if fam == "RuLSIF_exact_v2" else fam
        if not done:
            floors[name] = {"status": "PENDING", "n_finals": len(mine), "n_reevaluated": 0}; continue
        um = lambda f: {n: {k: f(abs(v["unit_macro_difference"][n]["difference_rerun_minus_primary"][k]) for v in done) for k in UM} for n in SETS}   # noqa: E731
        ags = [v["per_registration_agreement"] for v in done]
        floors[name] = {"status": "COMPLETE" if len(done) == len(mine) else f"PARTIAL ({len(done)}/{len(mine)})", "n_finals": len(mine),
                        "n_reevaluated": len(done), "floor_max_abs": um(max), "mean_abs": um(lambda xs: statistics.fmean(list(xs))),
                        "per_registration": {"max_disagree42": max(a["success42"]["disagree"] for a in ags), "max_disagree9": max(a["success9"]["disagree"] for a in ags),
                                             "max_median_abs_delta_err_full": max(a["delta_err_full"]["median_abs"] for a in ags),
                                             "max_p95_abs_delta_err_full": max(a["delta_err_full"]["p95_abs"] for a in ags),
                                             "max_abs_delta_err_full": max(a["delta_err_full"]["max_abs"] for a in ags),
                                             "min_row_bitwise_equal": min(a["row_bitwise_equal"] for a in ags), "min_F_true_equal": min(a["F_true_equal"] for a in ags)}}
    means = {fam: v["summary"]["seed_mean"] for fam, v in families.items() if v["summary"]["seed_mean"]}
    r1024 = (rulsif.get("1024") or {}).get("summary") or {}
    if r1024.get("seed_mean"):
        means[RULSIF_T1024] = r1024["seed_mean"]
    between = {}
    for n in SETS:
        between[n] = {}
        for k in UM:
            rows = []
            for a, b in itertools.combinations(sorted(means), 2):
                d = means[a][n]["unit_macro"][k] - means[b][n]["unit_macro"][k]
                fl = [floors.get(x, {}).get("floor_max_abs", {}).get(n, {}).get(k) for x in (a, b)]
                floor = max(fl) if all(x is not None for x in fl) else None
                rows.append({"a": a, "b": b, "difference": d, "observed_repeat_change_max": floor})
            between[n][k] = rows
    complete = bool(floors) and all(v["status"] == "COMPLETE" for v in floors.values())
    return {"status": "COMPLETE" if complete else "PENDING", "label": EVN_LABEL, "releases": releases, "units": units, "family_floors": floors,
            "family_seed_means": {fam: {n: means[fam][n]["unit_macro"] for n in SETS} for fam in means},
            "between_family_differences": between,
            "rule": "observed repeat change of a family = the largest |re-evaluated - recorded| unit-macro value over its re-evaluated finals "
                    "(each metric, each set; one re-evaluation per final); shown beside each between-family difference",
            "statement": "an observed change from one re-evaluation per final, not a calibrated noise distribution, noise floor, "
                         "equivalence bound or significance threshold (review package 2026-10-20 section 4); differences are not "
                         "declared interpretable or uninterpretable from it", "owner_context_verbatim": "什么快什么来吧"}, files


def replicate_vs_control(rep, ctl):
    """Per family: the RTX-vs-L40S pair beside the L40S-vs-L40S pair - |delta| of the unit-macro values (every set) and the per-registration
    agreement.  Numbers only."""
    out = {}
    for sec, kind in ((rep, "rtx_vs_l40s"), (ctl, "l40s_vs_l40s")):
        for key, v in (sec.get("pairs") or {}).items():
            row = {"pair": key, "seed": v["seed"], "status": v["status"], "primary_job": v["primary_job"], "primary_gpu": v["primary_gpu"],
                   "rerun_gpu": v["rerun_gpu"]}
            if v["status"].startswith("COMPLETE"):
                ag = v["per_registration_agreement"]
                row.update(rerun_job=v["rerun"]["job"], selected_step=v["selected_step"],
                           abs_unit_macro_difference={n: v["unit_macro_difference"][n]["abs_difference"] for n in SETS},
                           unit_macro_difference={n: {"difference_rerun_minus_primary": v["unit_macro_difference"][n]["difference_rerun_minus_primary"],
                                                      "bootstrap95": v["unit_macro_difference"][n]["bootstrap95"],
                                                      "units_differing": v["unit_macro_difference"][n]["units_differing"]} for n in SETS},
                           per_registration={"n": ag["n_registrations"], "estimate_bitwise_equal": ag["estimate_bitwise_equal"],
                                             "agreement42": ag["success42"]["agreement"], "disagree42": ag["success42"]["disagree"],
                                             "agreement9": ag["success9"]["agreement"], "disagree9": ag["success9"]["disagree"],
                                             "median_abs_delta_err_full": ag["delta_err_full"]["median_abs"],
                                             "max_abs_delta_err_full": ag["delta_err_full"]["max_abs"]},
                           first_step_loss_differs=v["training_log"]["first_step_loss_differs"])
            out.setdefault(v["family"], {})[kind] = row
    return {"families": out, "statement": "numbers only: |delta| of the RTX-vs-L40S pair (seed 1001) beside |delta| of the L40S-vs-L40S pair "
                                          "(seed 1002) for each family, one pair of each kind; no claim beyond these numbers"}


def analyse(inp):
    group = inp.json(EPFL / "EPFL_GROUP_RECORD_V2_FROZEN.json")
    budget = inp.json(EPFL / "EPFL_BUDGET_CHECK.json")
    rel = {tag: inp.json(b / "RELEASE.json") for tag, b in BUNDLES.items()}
    rec = receipts(inp)
    ledger_jobs = {}
    for b in (PARENT,):
        for r in inp.jsonl(b / "COST_LEDGER.jsonl"):
            ledger_jobs.setdefault((r["family"], r["phase"]), set()).add(r["job"])
    sets_units = {name: sorted(group["sensitivity_sets"][name]["units"]) for name in SETS}
    tag_of = {}
    for tag in BUNDLES:
        tag_of[hashlib.sha256((BUNDLES[tag] / "RELEASE.json").read_bytes()).hexdigest()] = tag
    families, problems = {}, []
    for fam in FAMILIES:
        seeds = rel["3dae6434"]["families"][fam]["schedule"]["final_seeds"]
        finals = {}
        for s in seeds:
            d = PARENT / "results" / "FINAL_DEV" / fam / f"seed_{s}"
            if not (d / "COMPLETE.json").exists():
                finals[s] = None; problems.append(f"{fam} seed {s}: not COMPLETE"); continue
            idn = json.loads((d / "COMPLETE.json").read_text())["identity"]
            finals[s] = final_record(inp, d, group, rec, tag_of.get(idn["release_sha256"], idn["release_sha256"][:8]))
            if not finals[s]["runtime_statistics_agree"]:
                problems.append(f"{fam} seed {s}: run-time DEV statistics differ from the recomputation in {finals[s]['runtime_mismatch_sets']}")
        bf = budget["families"].get(fam, {})
        families[fam] = {"kind": "training_free" if fam in TRAINING_FREE else "learned", "finals": {str(s): strip(f) for s, f in finals.items()},
                         "summary": family_summary(finals, sets_units), "strata": strata(finals),
                         "cal": cal_info(inp, PARENT / "results", fam, rec, ledger_jobs,
                                         {"cal_family_hours_measured_assembly": bf.get("cal_family_hours_measured_assembly"),
                                          "cal_family_hours_iteration_bound": bf.get("cal_family_hours_iteration_bound"), "budget_check": bf.get("status"),
                                          "final_seed_hours_max": bf.get("final_seed_hours_max")}),
                         "finals_allocated_hours": {str(s): (f or {}).get("allocated_hours") for s, f in finals.items()}}
    rulsif = {}
    for T, b in R2.items():
        if not (b / "RELEASE.json").exists():
            rulsif[str(T)] = {"status": "NO_RELEASE"}; continue
        r2rel = inp.json(b / "RELEASE.json"); finals = {}
        for s in r2rel["schedule"]["final_seeds"]:
            d = b / "results" / "FINAL_DEV" / "RuLSIF_exact_v2" / f"seed_{s}"
            finals[s] = final_record(inp, d, group, {}, "r2_T%d" % T) if (d / "COMPLETE.json").exists() else None
        sel_p = b / "results" / "CAL" / "RuLSIF_exact_v2" / "SELECTION.json"
        sel = inp.json(sel_p) if sel_p.exists() else None
        n_done = sum(f is not None for f in finals.values())
        rulsif[str(T)] = {"role": r2rel["kernel_dictionary"]["role"], "release_sha256": hashlib.sha256((b / "RELEASE.json").read_bytes()).hexdigest(),
                          "status": "COMPLETE" if n_done == 3 else f"PENDING ({n_done}/3 finals complete)",
                          "cal": None if sel is None else {"selected": sel["selected"]["candidate"], "completed_candidates": len(sel["completed_candidates"]),
                                                           "budget_exhausted": sel["budget_exhausted"],
                                                           "selected_cal_metrics": selected_cal_metrics(inp, b / "results", "RuLSIF_exact_v2", sel)},
                          "measured_cost_timing": r2rel["measured_cost"], "finals": {str(s): strip(f) for s, f in finals.items()},
                          "summary": family_summary(finals, sets_units) if n_done else None, "strata": strata(finals) if n_done else None}
    deltas = {fam: {name: {k: v["summary"]["seed_mean"][name]["unit_macro"][k] - v["summary"]["seed_mean"]["DEV72_primary"]["unit_macro"][k]
                            for k in ("success42", "success9", "median_err_full")} for name in SETS[1:]}
              for fam, v in families.items() if v["summary"]["seed_mean"]}
    xhw = {}
    for lab in ("RTX6000PRO", "H100"):
        for p in sorted((BUNDLES["d9d67cf2"] / "xhw_gates" / lab / "gpu_gate").glob("*/GATE.json")):
            g = inp.json(p)
            xhw[lab] = {"job": g["pins"]["job"], "status": g["status"], "criterion": (g.get("xhw_criterion") or {}).get("id"),
                        "checkpoint_rows_measured": {n: {"status": r["status"], "energy_max_abs": (r.get("energy") or {}).get("max_abs_err"),
                                                          "energy_max_rel": (r.get("energy") or {}).get("max_rel_err"),
                                                          "gradient_max_abs": (r.get("gradient") or {}).get("max_abs_err"),
                                                          "gradient_max_rel": (r.get("gradient") or {}).get("max_rel_err")}
                                                     for n, r in (g.get("checkpoint_rows_measured") or {}).items() if r.get("energy")}}
    hw_rep, pair_files = replicate_pairs(inp, group, sets_units, families, problems)
    hw_ctl, ctl_files = control_pairs(inp, group, sets_units, families, problems)
    evn, evn_files = eval_noise(inp, group, sets_units, families, rulsif, problems)
    pair_files = {**pair_files, **ctl_files, **evn_files}
    m1 = families["MF_VCS_M1"]["cal"]
    disclosures = {
        "benchmark": "EPFL is a development benchmark (DEV, 72 scene units), reported separately from BF and never pooled with it; no method-superiority "
                     "claim; bootstrap intervals are interval disclosure only (no significance tests)",
        "selection_asymmetry": "learned families select among several candidates (3 or 9) and 12 CAL checkpoints per seed; the training-free families "
                               "(MI, NMI, MIND) have one configuration and one final; CoMIR has one candidate and one checkpoint",
        "m1_cost": {"predicted_cal_hours": m1["predicted"]["cal_family_hours_measured_assembly"], "actual_cal_allocated_l40s_hours": m1["cal_allocated_l40s_hours"],
                    "completed_candidates": m1["completed_candidates"], "budget_exhausted": m1["budget_exhausted"],
                    "note": "the measured assembly predicted 10.92 h (RESOURCE_LIMITED_DISCLOSED); the actual CAL used 7.49 allocated L40S h and all 9 "
                            "candidates completed within the 8 h cap"},
        "hardware": {"amendments": {"EPFL_HARDWARE_AMENDMENT_20261019": "6f48992a6cb7de5f7016e4bdab4493319e2e2df02a6fc944306708e90fed7d2e",
                                    "EPFL_XHW_GATE_AMENDMENT_20261020": "cd673b7cbd5b87bb974be3b797d8686b6efe3c0a59ff497d7be2707484bdc7c1"},
                     "owner_quotes": OWNER, "cross_hardware_gates": xhw,
                     "rule": "finals on RTX6000PRO / L40S / H100 after a PASS cross-hardware gate; CAL on L40S; the RTX6000PRO / H100 checkpoint "
                             "energy / gradient rows judged with T01's round-trip tolerance (post-failure, owner-approved criterion change); results by "
                             "hardware stratum and pooled; no claim of cross-hardware equivalence"},
        "hardware_replicates": "post-hoc hardware sensitivity diagnostic (EPFL_HW_REPLICATE_20261020): the two RTX PRO 6000 finals rerun on L40S "
                               "with the same science, CAL selection and seed in their own results root; reported beside the primary results, "
                               "never pooled into them; deterministic algorithms are off, so each pair measures hardware together with run-to-run "
                               "non-determinism",
        "evaluation_noise": "post-hoc evaluation-noise diagnostic (EPFL_EVAL_NOISE_20261020): every final's CAL-selected checkpoint re-evaluated once "
                            "on DEV with the unchanged evaluator on its recorded GPU type (deterministic algorithms off, so the pose search is not "
                            "bitwise repeatable; BF TEST_REPEAT134 DEV gates); per family the noise floor beside the between-family differences; "
                            "differences below the evaluation noise are not interpretable",
        "hardware_controls": "its L40S-vs-L40S control (EPFL_HW_CONTROL_20261020): two L40S finals (seed 1002) rerun on L40S the same way in a "
                             "third results root, shown beside the RTX-vs-L40S pairs; one pair of each kind per family, at different seeds; no "
                             "claim beyond the numbers",
        "rulsif_v2": "RuLSIF_exact_v2 runs under amendment A1 (post-gate, level-A eligibility, candidate 8 excluded) and the EPFL addendum revision 3: "
                     "kernel dictionary 10688 -> 1024 centres (mean per window 32 -> about 3.07), which changes expressive capacity - "
                     "「容量与成本共同受约束的核基线」; exact empirical moments of the 1024-centre model under the per-window V reference; the "
                     "pre-registered 2048-centre capacity sensitivity is shown beside T1024 and never replaces it; same size as BF v2 means the same "
                     "resource configuration, not sufficient capacity; no claim about RuLSIF's best capability",
        "statistical_unit": "DEV statistics are unit-macro over EPFL scene units (72; near-duplicate links grouped); perturbations within a unit are "
                            "not independent; paired across methods by unit (the same bootstrap draws for every family)",
        "thresholds": "success < 42 and < 9 err_full = 10.5 and 2.25 native px (err_full = 4 x native px on the 512 native window)",
    }
    return {"schema": "v10.epfl.results.v1", "protocol": {"frozen_protocol_sha256": hashlib.sha256((EPFL / "EPFL_PROTOCOL_FROZEN.json").read_bytes()).hexdigest(),
                                                           "group_record_sha256": hashlib.sha256((EPFL / "EPFL_GROUP_RECORD_V2_FROZEN.json").read_bytes()).hexdigest()},
            "bootstrap": {"B": B_BOOT, "seed": BOOT_SEED, "generator": "numpy.random.default_rng", "resampled": "units, with replacement, the same draws for "
                          "every family within a set", "interval": "percentile 2.5 / 97.5"},
            "families": families, "rulsif_v2": rulsif, "sensitivity_deltas_vs_primary": deltas, "hardware_replicates": hw_rep,
            "hardware_controls": hw_ctl, "hardware_replicate_vs_control": replicate_vs_control(hw_rep, hw_ctl), "evaluation_noise": evn,
            "disclosures": disclosures, "problems": problems}, pair_files


def strip(f):
    if f is None:
        return None
    return {**{k: v for k, v in f.items() if k != "stats"}, "stats": {n: {k: v for k, v in s.items() if k != "_units"} for n, s in f["stats"].items()}}


def markdown(res) -> str:
    L = ["# EPFL V10 结果（官方分析，冻结协议）", "", "开发基准（DEV，72 个场景单元）；与 BF 分开报告、不合并；不作方法优劣结论；区间仅为披露（单元自助法，配对于单元）。", "",
         f"Bootstrap：B = {res['bootstrap']['B']}，seed {res['bootstrap']['seed']}（numpy default_rng），百分位 95%。阈值 <42 / <9 err_full = 10.5 / 2.25 原生像素。", "",
         "## DEV72_primary（单元宏平均，种子均值 [95% 区间]）", "", "| 族 | 种子 | 硬件 | <42 | <9 | 单元中位 err_full（原生 px） |", "|---|---|---|---|---|---|"]
    for fam, v in res["families"].items():
        sm = v["summary"]["seed_mean"].get("DEV72_primary"); ci = v["summary"]["bootstrap95"].get("DEV72_primary")
        if not sm:
            continue
        hw = ", ".join(f"{s}:{(g or '').replace('NVIDIA ', '').replace(' Blackwell Server Edition', '')}" for s, g in v["strata"]["hardware_per_seed"].items())
        u = sm["unit_macro"]
        L.append(f"| {fam} | {len(v['summary']['seeds_complete'])} | {hw} | {u['success42']:.3f} [{ci['success42'][0]:.3f}, {ci['success42'][1]:.3f}] | "
                 f"{u['success9']:.3f} [{ci['success9'][0]:.3f}, {ci['success9'][1]:.3f}] | {u['median_err_full']:.2f} ({u['median_native_px']:.2f}) "
                 f"[{ci['median_err_full'][0]:.2f}, {ci['median_err_full'][1]:.2f}] |")
    L += ["", "## DEV72_primary 配准层面（全部 740 次配准汇总，种子均值）", "", "| 族 | <42 | <9 | 中位 err_full（原生 px） |", "|---|---|---|---|"]
    for fam, v in res["families"].items():
        r = (v["summary"]["seed_mean"].get("DEV72_primary") or {}).get("registration_level")
        if r:
            L.append(f"| {fam} | {r['success42']:.3f} | {r['success9']:.3f} | {r['median_err_full']:.2f} ({r['median_native_px']:.2f}) |")
    cats = sorted({c for v in res["families"].values() for c in (v["summary"].get("per_category_seed_mean") or {})})
    L += ["", "## DEV72_primary 按类别（单元宏平均 <42 / <9，种子均值；每类单元数见 JSON）", "", "| 族 | " + " | ".join(cats) + " |", "|---|" + "---|" * len(cats)]
    for fam, v in res["families"].items():
        pc = v["summary"].get("per_category_seed_mean") or {}
        if pc:
            L.append(f"| {fam} | " + " | ".join(f"{pc[c]['success42']:.2f} / {pc[c]['success9']:.2f}" for c in cats) + " |")
    L += ["", "## 敏感性集合（相对 DEV72_primary 的差，单元宏平均种子均值）", "", "| 族 | 集合 | Δ<42 | Δ<9 | Δ中位 err_full |", "|---|---|---|---|---|"]
    for fam, d in res["sensitivity_deltas_vs_primary"].items():
        for name, x in d.items():
            L.append(f"| {fam} | {name} | {x['success42']:+.4f} | {x['success9']:+.4f} | {x['median_err_full']:+.3f} |")
    L += ["", "## 硬件分层（DEV72_primary 单元宏平均）", "", "| 族 | pooled n | <42 pooled | <42 仅 L40S | <9 pooled | <9 仅 L40S |", "|---|---|---|---|---|---|"]
    for fam, v in res["families"].items():
        s = v["strata"]
        if s["n_pooled"] and s["n_pooled"] != s["n_L40S"]:
            L.append(f"| {fam} | {s['n_pooled']} (L40S {s['n_L40S']}) | {s['pooled']['success42']:.3f} | {s['L40S_only']['success42']:.3f} | "
                     f"{s['pooled']['success9']:.3f} | {s['L40S_only']['success9']:.3f} |")
    L += ["", "其余族的全部最终种子都在 L40S 上（分层与合并相同）。不声称跨硬件等价。", "", "## CAL 选择与实测成本（L40S 分配 GPU 小时）", "",
          "| 族 | 选中 | step | CAL n | CAL <42 | CAL <9 | CAL 中位 err_full | CAL 完成/计划 | CAL 实测 h | CAL 预测 h | 最终各种子 h |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for fam, v in res["families"].items():
        c = v["cal"]; m = c.get("selected_cal_metrics") or {}
        fh = ", ".join(f"{s}:{h:.2f}" for s, h in v["finals_allocated_hours"].items() if h)
        L.append(f"| {fam} | {json.dumps(c.get('selected'), ensure_ascii=False)} | {m.get('step')} | {m.get('n')} | {m.get('success42', 0):.3f} | "
                 f"{m.get('success9', 0):.3f} | {m.get('median_err', 0):.2f} | {c.get('completed_candidates')}/{c.get('planned_candidates')} | "
                 f"{(c.get('cal_allocated_l40s_hours') or 0):.2f} | {(c.get('predicted') or {}).get('cal_family_hours_measured_assembly') or 0:.2f} | {fh} |")
    hr = res.get("hardware_replicates") or {}; hc = res.get("hardware_controls") or {}
    gs = lambda g: (g or "").replace("NVIDIA ", "").replace(" Blackwell Server Edition", "")   # noqa: E731
    L += ["", "## 硬件诊断复跑（事后硬件敏感性诊断，EPFL_HW_REPLICATE_20261020 / EPFL_HW_CONTROL_20261020；不替代、不并入主结果）", ""]
    if hr.get("status", "NO_RELEASE") == "NO_RELEASE":
        L.append("- 尚无复跑 release。")
    else:
        L += [f"同一 release 科学代码、同一 CAL 选择、同一种子，在 L40S（`--partition={hr['partition']} --exclude={hr['exclude']}`）上重跑："
              f"RTX PRO 6000 的最终单元（种子 1001，RTX vs L40S）以及对照 L40S 的最终单元（种子 1002，L40S vs L40S）。"
              f"所有者规则「{hr['owner_rules_verbatim']['2026-10-19']}」「{hr['owner_rules_verbatim']['2026-10-20']}」。"
              f"差值为复跑 − 主单元；区间：{hr['bootstrap']}。", "",
              "| 族 | 对 | 种子 | 主作业（GPU） | 复跑作业 | 状态 | 选中 step 主 / 复跑 | Δ<42 [95%] | Δ<9 [95%] | Δ单元中位 err_full [95%] | 估计逐位相同 | <42 一致（不一致数） | <9 一致（不一致数） | 中位/最大 abs Δerr_full | 首个不同训练 step |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for sec, kind in ((hr, "RTX vs L40S"), (hc, "L40S vs L40S")):
            for key, v in (sec.get("pairs") or {}).items():
                head = f"| {v['family']} | {kind} | {v['seed']} | {v['primary_job']}（{gs(v['primary_gpu'])}） |"
                if not v["status"].startswith("COMPLETE"):
                    L.append(f"{head} - | {v['status']} | | | | | | | | | |")
                    continue
                d = v["unit_macro_difference"]["DEV72_primary"]; dd = d["difference_rerun_minus_primary"]; ci = d["bootstrap95"]
                ag = v["per_registration_agreement"]; a42, a9 = ag["success42"], ag["success9"]
                L.append(f"{head} {v['rerun']['job']} | {v['status']} | {v['selected_step']['primary']} / {v['selected_step']['rerun']} | "
                         f"{dd['success42']:+.4f} [{ci['success42'][0]:+.4f}, {ci['success42'][1]:+.4f}] | {dd['success9']:+.4f} [{ci['success9'][0]:+.4f}, {ci['success9'][1]:+.4f}] | "
                         f"{dd['median_err_full']:+.3f} [{ci['median_err_full'][0]:+.3f}, {ci['median_err_full'][1]:+.3f}] | {ag['estimate_bitwise_equal']}/{ag['n_registrations']} | "
                         f"{a42['agreement']:.3f} ({a42['disagree']}) | {a9['agreement']:.3f} ({a9['disagree']}) | "
                         f"{ag['delta_err_full']['median_abs']:.3f} / {ag['delta_err_full']['max_abs']:.2f} | {v['training_log']['first_step_loss_differs']} |")
        cmp_ = (res.get("hardware_replicate_vs_control") or {}).get("families") or {}
        L += ["", "### |Δ(RTX vs L40S)| 与 |Δ(L40S vs L40S)| 并列（DEV72_primary 单元宏平均；逐次配准一致性）", "",
              "| 族 | 对 | 种子 | abs Δ<42 | abs Δ<9 | abs Δ单元中位 err_full | <42 不一致 / 740 | <9 不一致 / 740 | 中位 abs Δerr_full | 最大 abs Δerr_full |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for fam, kinds in cmp_.items():
            for kind, label in (("rtx_vs_l40s", "RTX vs L40S"), ("l40s_vs_l40s", "L40S vs L40S")):
                r = kinds.get(kind)
                if r is None:
                    L.append(f"| {fam} | {label} | - | 无 release | | | | | | |")
                elif "abs_unit_macro_difference" not in r:
                    L.append(f"| {fam} | {label} | {r['seed']} | {r['status']} | | | | | | |")
                else:
                    a = r["abs_unit_macro_difference"]["DEV72_primary"]; g = r["per_registration"]
                    L.append(f"| {fam} | {label} | {r['seed']} | {a['success42']:.4f} | {a['success9']:.4f} | {a['median_err_full']:.3f} | {g['disagree42']} | "
                             f"{g['disagree9']} | {g['median_abs_delta_err_full']:.3f} | {g['max_abs_delta_err_full']:.2f} |")
        L.append("")
        L.append("仅列出数值，不据此作进一步结论（每族每类各 1 对；两类对的种子不同：1001 / 1002）。确定性算法关闭。")
        for key, v in (hr.get("pairs") or {}).items():
            if v["status"].startswith("COMPLETE") and v.get("seed_context_DEV72_primary"):
                c = v["seed_context_DEV72_primary"]
                seeds = ", ".join(f"{s}: {x['success42']:.3f} / {x['success9']:.3f}" for s, x in c["l40s_seeds"].items())
                L.append(f"- {v['family']} 种子 {v['seed']}：RTX {c['rtx_primary']['success42']:.3f} / {c['rtx_primary']['success9']:.3f}；L40S（<42 / <9）{seeds}；"
                         f"逐次配准差异见 `{v['per_registration_file']['path']}`（sha256 {v['per_registration_file']['sha256'][:12]}…）。")
        for key, v in (hc.get("pairs") or {}).items():
            if v["status"].startswith("COMPLETE"):
                L.append(f"- 对照 {v['family']} 种子 {v['seed']}：逐次配准差异见 `{v['per_registration_file']['path']}`（sha256 {v['per_registration_file']['sha256'][:12]}…）。")
    ev = res.get("evaluation_noise") or {}
    L += ["", "## 评估噪声（事后评估噪声诊断，EPFL_EVAL_NOISE_20261020；不改变任何主结果）", ""]
    if ev.get("status", "NO_RELEASE") == "NO_RELEASE":
        L.append("- 尚无评估噪声 release。")
    else:
        L += ["每个最终单元的 CAL 选中检查点在 DEV（74 × 10）上用不变的评估器、guard 与激活重评一次，在其记录 DEV 的 GPU 类型上；不重新训练。"
              "观察到的重评变动 = 该族各最终单元 |重评 − 记录| 单元宏平均差的最大值（每个指标、每个集合）。它只是每个最终单元一次重评的观察值，不是已标定的噪声分布、噪声下限、等价界或显著性阈值，不据此判定族间差异能否解释。"
              f"所有者背景「{ev['owner_context_verbatim']}」。状态：{ev['status']}。", "",
              "### 各族观察到的重评变动（DEV72_primary 单元宏平均；逐次配准）", "",
              "| 族 | 重评 / 最终 | 状态 | 最大 abs Δ<42 | 最大 abs Δ<9 | 最大 abs Δ单元中位 err_full | 均值 abs Δ<9 | 最多 <42 翻转 | 最多 <9 翻转 | 最大中位 abs Δerr_full | 最大 abs Δerr_full | 最少逐位相同行 / 740 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for fam, f in ev["family_floors"].items():
            if "floor_max_abs" not in f:
                L.append(f"| {fam} | {f['n_reevaluated']} / {f['n_finals']} | {f['status']} | | | | | | | | | |"); continue
            fl = f["floor_max_abs"]["DEV72_primary"]; mn = f["mean_abs"]["DEV72_primary"]; g = f["per_registration"]
            L.append(f"| {fam} | {f['n_reevaluated']} / {f['n_finals']} | {f['status']} | {fl['success42']:.4f} | {fl['success9']:.4f} | {fl['median_err_full']:.3f} | "
                     f"{mn['success9']:.4f} | {g['max_disagree42']} | {g['max_disagree9']} | {g['max_median_abs_delta_err_full']:.3f} | {g['max_abs_delta_err_full']:.2f} | "
                     f"{g['min_row_bitwise_equal']} |")
        L += ["", "### 每个最终单元（DEV72_primary，重评 − 记录）", "",
              "| 族 | 种子 | GPU | 状态 | Δ<42 | Δ<9 | Δ单元中位 err_full | <42 翻转 | <9 翻转 | abs Δerr_full 中位 / p95 / 最大 | 逐位相同行 | F_true 相同 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for key, v in sorted(ev["units"].items()):
            g = (v["gpu_name"] or "").replace("NVIDIA ", "").replace(" Blackwell Server Edition", "")
            if v["status"] != "COMPLETE":
                L.append(f"| {v['family']} | {v['seed']} | {g} | {v['status']} | | | | | | | | |"); continue
            d = v["unit_macro_difference"]["DEV72_primary"]["difference_rerun_minus_primary"]; a = v["per_registration_agreement"]; e = a["delta_err_full"]
            L.append(f"| {v['family']} | {v['seed']} | {g} | {v['status']} | {d['success42']:+.4f} | {d['success9']:+.4f} | {d['median_err_full']:+.3f} | "
                     f"{a['success42']['disagree']} | {a['success9']['disagree']} | {e['median_abs']:.3f} / {e['p95_abs']:.3f} / {e['max_abs']:.2f} | "
                     f"{a['row_bitwise_equal']} | {a['F_true_equal']} |")
        L += ["", "### 族间差异与观察到的重评变动（DEV72_primary 种子均值，A − B；并列给出，不作判定）", "",
              "| A | B | Δ<9 | 两族中较大的重评变动 | Δ单元中位 err_full | 两族中较大的重评变动 |", "|---|---|---|---|---|---|"]
        b9 = ev["between_family_differences"]["DEV72_primary"]["success9"]; bm = ev["between_family_differences"]["DEV72_primary"]["median_err_full"]
        fm = lambda x, f: "待定" if x is None else format(x, f)       # noqa: E731
        for x, y in zip(b9, bm):
            L.append(f"| {x['a']} | {x['b']} | {x['difference']:+.4f} | {fm(x['observed_repeat_change_max'], '.4f')} | {y['difference']:+.3f} | "
                     f"{fm(y['observed_repeat_change_max'], '.3f')} |")
        L.append("")
        L.append("规则：" + ev["rule"] + "；" + ev["statement"] + "。每个最终单元只重评一次；不作方差模型，不据此作进一步结论。")
    L += ["", "## RuLSIF_exact_v2（A1；核字典 10688 → 1024，「容量与成本共同受约束的核基线」；T2048 为预注册容量敏感性，并列、不替代）", ""]
    for T, v in res["rulsif_v2"].items():
        L.append(f"- T{T}（{v.get('role')}）：{v.get('status')}")
        if v.get("summary") and v["summary"]["seed_mean"]:
            u = v["summary"]["seed_mean"]["DEV72_primary"]["unit_macro"]
            note = "3 个种子均值" if v.get("status") == "COMPLETE" else "部分结果，待全部完成"
            L.append(f"  - 已完成种子 {v['summary']['seeds_complete']}：<42 {u['success42']:.3f}，<9 {u['success9']:.3f}，单元中位 {u['median_err_full']:.2f}（{note}）")
    L += ["", "## 披露", ""] + [f"- {k}: {json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v}" for k, v in res["disclosures"].items()]
    L += ["", "## 发现的问题", ""] + ([f"- {p}" for p in res["problems"]] or ["- 无"])
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    inp = Inputs(); res, pair_files = analyse(inp); res["inputs"] = inp.files
    if a.check:
        old = json.loads((OUT / "EPFL_RESULTS.json").read_text())
        changed = sorted(p for p, h in old["inputs"].items() if not Path(p).exists() or hashlib.sha256(Path(p).read_bytes()).hexdigest() != h)
        same = {k: v for k, v in old.items() if k != "utc"} == json.loads(json.dumps({k: v for k, v in res.items() if k != "utc"}))
        files_same = all(Path(p).exists() and Path(p).read_text() == t for p, t in pair_files.items())
        print(f"[check] inputs changed: {changed}; recomputation identical: {same}; per-registration pair files identical: {files_same}")
        return 0 if same and files_same and not changed else 1
    for p, t in pair_files.items():       # per-registration differences of the hardware replicate pairs (diagnostic root, never git)
        Path(p).parent.mkdir(parents=True, exist_ok=True); tmp = Path(p).with_name("." + Path(p).name + ".tmp"); tmp.write_text(t); os.replace(tmp, p)
    res["utc"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / ".EPFL_RESULTS.json.tmp"; tmp.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n"); os.replace(tmp, OUT / "EPFL_RESULTS.json")
    (OUT / "EPFL_RESULTS_CN.md").write_text(markdown(res))
    print(f"-> {OUT / 'EPFL_RESULTS.json'} ({len(inp.files)} input files); problems: {res['problems']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
