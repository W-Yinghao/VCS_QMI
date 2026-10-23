"""r22 summaries (PILOT/DEVELOPMENT, FIT/CAL only): per-run CAL results, paired per-seed differences, cost of all attempts.

KD: r22 S1/S2 runs plus the previous round's runs at the same seeds (only run.json and best_cal_metrics.json are read;
    the previous finals' TEST files are never opened). AIG: r22 runs plus round-21 task_only / alternating / joint runs
    (result.json, history.json, cal_predictions.csv), with a per-seed paired prompt bootstrap (prompt groups resampled;
    not training-seed uncertainty) and per-prompt regret wins/losses.
Usage: python summarize.py --app kd|aigiqa --code CODE_DIR --out DIR [--bootstrap 1000]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
import sys
from pathlib import Path

CS = Path("/home/infres/yinwang/CS_QMI")
REC = CS / "outputs_apps" / "r22" / "records"
KD_PREV = CS / "outputs_apps" / "kd_20261020" / "runs"
AIG_R21 = CS / "outputs_apps" / "aigiqa_round_20261021"
KD_METRICS = ["top1", "top5", "nll", "ece_15bins", "brier"]
AIG_METRICS = ["srcc", "plcc", "within_prompt_pairwise_macro", "within_prompt_pairwise_micro", "within_prompt_srcc_macro",
               "normalized_regret"]
LOWER = {"nll", "ece_15bins", "brier", "normalized_regret"}
KD_PAIRS = [("vcscrd", "crd_original", "VCS-CRD vs the previous CRD (objective and Q sampling both differ)"),
            ("vcscrd", "crd_uniform", "objective only (same uniform Q, same heads/memory)"),
            ("crd_uniform", "crd_original", "Q sampling only (official NCE)"),
            ("dkd_vcscrd", "dkd", "application value over DKD"),
            ("dkd_vcscrd", "dkd_crd_uniform", "objective only, on DKD (exact matched control)"),
            ("dkd_crd_uniform", "dkd", "context: previous round"),
            ("dkd_vcscrd", "vcscrd", "DKD anchor added")]
AIG_PAIRS = [("exact_bank_mosw", "bce_bank_mosw", "primary: VCS vs matched BCE, same P_w and banked Q"),
             ("exact_bank_mosw", "exact_bank", "MOS-weighted P vs uniform P"),
             ("exact_bank", "exact_inbatch", "bank vs in-batch Q, lambda 0.1"),
             ("exact_bank_mosw", "r1_task_only", "application value"),
             ("bce_bank_mosw", "r1_task_only", "application value (BCE)"),
             ("exact_bank", "r1_task_only", "application value"),
             ("exact_inbatch", "r1_task_only", "application value"),
             ("exact_inbatch", "r1_alternating_target_sg_exact", "lambda 0.1 vs 0.01, in-batch exact"),
             ("exact_bank_mosw", "r1_alternating_target_sg_exact", "r22 recipe vs round-21 alternating exact"),
             ("bce_bank_mosw", "r1_alternating_target_sg_bce", "r22 recipe vs round-21 alternating BCE"),
             ("r1_alternating_target_sg_exact", "r1_task_only", "context: round 21")]


def load(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def mean_sd(values):
    values = [v for v in values if v is not None]
    return (statistics.mean(values) if values else None, statistics.stdev(values) if len(values) > 1 else None)


def attempts(job_name):
    out = []
    for p in sorted(glob.glob(f"{REC}/{job_name}/slurm_*/slurm.json")):
        rec = load(p); start, end = rec.get("start", {}), rec.get("end", {})
        smi = (start.get("nvidia_smi") or "").split(";")[0].split(",")
        out.append({"slurm_job_id": start.get("slurm", {}).get("SLURM_JOB_ID"), "node": start.get("slurm", {}).get("SLURMD_NODENAME"),
                    "partition": start.get("slurm", {}).get("SLURM_JOB_PARTITION"),
                    "gpu_name": smi[1].strip() if len(smi) > 3 else None, "exit_code": end.get("exit_code"),
                    "elapsed_seconds": end.get("elapsed_seconds"), "start_utc": start.get("utc"), "end_utc": end.get("utc")})
    return out


def pairs_table(units, pairs, metrics):
    table = []
    for a, b, role in pairs:
        seeds = sorted(s for (f, s), u in units.items() if f == a and u["ok"] and units.get((b, s), {}).get("ok"))
        entry = {"a": a, "b": b, "role": role, "paired_seeds": seeds, "metrics": {}}
        for m in metrics:
            d = [None if units[(a, s)]["cal"].get(m) is None or units[(b, s)]["cal"].get(m) is None
                 else units[(a, s)]["cal"][m] - units[(b, s)]["cal"][m] for s in seeds]
            mu, sd = mean_sd(d)
            better = [x < 0 if m in LOWER else x > 0 for x in d if x is not None]
            entry["metrics"][m] = {"delta_per_seed": d, "mean": mu, "sample_sd": sd, "n_a_better": sum(better),
                                   "better_direction": "lower" if m in LOWER else "higher"}
        table.append(entry)
    return table


def kd_units(code):
    units = {}
    for jobs in ("JOBS_S1.csv", "JOBS_S2.csv"):
        path = code / "configs" / "kd" / jobs
        if not path.is_file():
            continue
        for r in csv.DictReader(path.open(newline="")):
            run = Path(r["run_dir"]); rj = load(run / "run.json"); best = load(run / "best_cal_metrics.json")
            hist = load(run / "history.json") or []
            ok = bool(rj and rj.get("status") == "complete" and best)
            probes = [h.get("gradient_probe") or {} for h in hist]
            ratio = [p.get("aux_over_ce") for p in probes]
            last = (hist[-1].get("loss_parts", {}) if hist else {})
            fam = f"{r['arm']}_l{r['lam']}" if r["seed"] == "27182" and r["arm"] != "crd_uniform" else r["arm"]
            key = (fam, int(r["seed"]))
            units[key] = {"family": fam, "arm": r["arm"], "lambda": float(r["lam"]), "seed": int(r["seed"]),
                          "source": "r22", "job_name": r["job_name"], "status": rj["status"] if rj else "MISSING", "ok": ok,
                          "selected_epoch": (best or {}).get("selected_epoch"), "cal": {m: (best or {}).get(m) for m in KD_METRICS},
                          "epochs_completed": (rj or {}).get("epochs_completed"),
                          "aux_over_ce_first_batch_mean": mean_sd(ratio)[0],
                          "last_epoch_loss_parts": last, "fit_wall_seconds_sum": (rj or {}).get("fit_wall_seconds_sum"),
                          "cuda_peak_allocated_bytes": (rj or {}).get("cuda_peak_allocated_bytes"),
                          "teacher_frozen_verified": (rj or {}).get("teacher_frozen_verified"),
                          "best_checkpoint_sha256": (rj or {}).get("best_checkpoint_sha256"),
                          "config_sha256": r["config_sha256"], "attempts": attempts(r["job_name"])}
    prev = [("crd_original", "crd_original"), ("dkd", "dkd"), ("dkd_crd_uniform", "dkd_crd_uniform"), ("student", "student"), ("kd", "kd")]
    for fam, name in prev:
        for seed in (31415, 16180, 57721):
            run = KD_PREV / "kd_final" / f"{name}_seed{seed}"
            best = load(run / "best_cal_metrics.json")
            units[(fam, seed)] = {"family": fam, "seed": seed, "source": "previous KD round final (kd_20261020)", "ok": bool(best),
                                  "selected_epoch": (best or {}).get("selected_epoch"), "cal": {m: (best or {}).get(m) for m in KD_METRICS},
                                  "run_dir": str(run)}
        run = KD_PREV / "kd_cal" / f"{name}_w1_seed27182"
        best = load(run / "best_cal_metrics.json")
        if best:
            units[(fam, 27182)] = {"family": fam, "seed": 27182, "source": "previous KD round CAL run (kd_20261020)", "ok": True,
                                   "selected_epoch": best.get("selected_epoch"), "cal": {m: best.get(m) for m in KD_METRICS},
                                   "run_dir": str(run)}
    return units


def kd(args):
    units = kd_units(Path(args.code))
    cal_seed = sorted(((k, u) for k, u in units.items() if k[1] == 27182), key=lambda x: -(x[1]["cal"]["top1"] or 0))
    cal_table = [{"family": k[0], "top1": u["cal"]["top1"], "nll": u["cal"]["nll"], "selected_epoch": u["selected_epoch"],
                  "source": u["source"], "status": u.get("status", "complete" if u["ok"] else "MISSING")} for k, u in cal_seed]
    sel = load(Path(__file__).resolve().parent / "records" / "kd_lambda_selection.json")
    finals = {k: u for k, u in units.items() if k[1] != 27182}
    pairs = pairs_table(finals, KD_PAIRS, KD_METRICS)
    per_family = {}
    for f in sorted({k[0] for k in finals}):
        vals = [finals[(f, s)]["cal"]["top1"] for s in (31415, 16180, 57721) if (f, s) in finals and finals[(f, s)]["ok"]]
        per_family[f] = {"cal_top1_per_seed_31415_16180_57721": [finals[(f, s)]["cal"]["top1"] if (f, s) in finals else None
                                                                 for s in (31415, 16180, 57721)],
                         "mean": mean_sd(vals)[0], "sample_sd": mean_sd(vals)[1], "n": len(vals)}
    write(args, "KD", units, {"cal_seed_27182_table": cal_table, "lambda_selection": sel, "finals_cal_top1": per_family,
                              "paired_differences_finals": pairs})


def aig(args):
    sys.path[:0] = [str(Path(args.code) / "engine"), str(Path(args.code) / "engine" / "shared")]
    from aigiqa.metrics import metrics as aig_metrics, paired_bootstrap
    units = {}
    rows = list(csv.DictReader((Path(args.code) / "configs" / "aigiqa" / "JOBS.csv").open(newline="")))
    sources = [(r["arm"], int(r["seed"]), Path(r["run_dir"]), r["job_name"], "r22") for r in rows]
    for fam in ("r1_task_only", "r1_alternating_target_sg_exact", "r1_alternating_target_sg_bce",
                "r1_joint_scheduled_exact", "r1_joint_scheduled_bce"):
        sources += [(fam, s, AIG_R21 / f"{fam}__seed{s}", None, "round 21") for s in (4101, 4102, 4103)]
    for fam, seed, run, job, src in sources:
        res, hist, failed = load(run / "result.json"), load(run / "history.json") or [], load(run / "FAILED.json")
        ok = bool(res and res.get("status") == "COMPLETED_FIT_CAL_ONLY" and not failed)
        active = [h["train_diagnostics"] for h in hist if h["train_diagnostics"].get("active_aux_weight", 0) > 0]
        keys = sorted({k for d in active for k in d if k.startswith("text_") or k.startswith("fusion_") or "critic_fit" in k})
        diag = {k + "_mean_active_epochs": mean_sd([d.get(k) for d in active])[0] for k in keys}
        units[(fam, seed)] = {"family": fam, "seed": seed, "source": src, "job_name": job, "ok": ok,
                              "status": res["status"] if res else ("FAILED" if failed else "MISSING"),
                              "best_epoch": (res or {}).get("best_epoch"),
                              "cal": {m: (res or {}).get("cal_metrics", {}).get(m) for m in AIG_METRICS},
                              "srcc_by_epoch": [h["cal_metrics"]["srcc"] for h in hist],
                              "fit_seconds": (res or {}).get("fit_seconds_excludes_warmup"),
                              "peak_allocated_bytes": (res or {}).get("peak_allocated_bytes"),
                              "checkpoint_sha256": (res or {}).get("checkpoint_sha256"), "train_diagnostics_active_epochs": diag,
                              "attempts": attempts(job) if job else None,
                              "cal_predictions": str(run / "cal_predictions.csv") if ok else None}
    pairs = pairs_table(units, AIG_PAIRS, AIG_METRICS)
    preds = {k: list(csv.DictReader(open(u["cal_predictions"]))) for k, u in units.items() if u["cal_predictions"]}
    for entry in pairs:
        entry["per_seed_prompt_bootstrap"], entry["per_prompt_regret"] = {}, {}
        for s in entry["paired_seeds"]:
            ra, rb = preds[(entry["a"], s)], preds[(entry["b"], s)]
            entry["per_seed_prompt_bootstrap"][s] = paired_bootstrap(ra, rb, repetitions=args.bootstrap)["differences"]
            deltas = []
            for g in sorted({x["prompt_group"] for x in ra}):
                ga = [x for x in ra if x["prompt_group"] == g]; gb = [x for x in rb if x["prompt_group"] == g]
                if len(ga) >= 2:
                    deltas.append(aig_metrics(ga)["normalized_regret"] - aig_metrics(gb)["normalized_regret"])
            entry["per_prompt_regret"][s] = {"n_groups": len(deltas), "a_lower_regret": sum(d < 0 for d in deltas),
                                             "a_higher_regret": sum(d > 0 for d in deltas), "equal": sum(d == 0 for d in deltas)}
    per_family = {}
    for f in sorted({k[0] for k in units}):
        vals = [units[(f, s)]["cal"]["srcc"] for s in (4101, 4102, 4103) if units.get((f, s), {}).get("ok")]
        per_family[f] = {"cal_srcc_per_seed_4101_4102_4103": [units.get((f, s), {}).get("cal", {}).get("srcc") for s in (4101, 4102, 4103)],
                         "mean": mean_sd(vals)[0], "sample_sd": mean_sd(vals)[1], "n": len(vals)}
    write(args, "AIG", units, {"cal_srcc": per_family, "paired_differences": pairs})


def write(args, tag, units, extra):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    att = [a for u in units.values() for a in (u.get("attempts") or [])]
    body = {"schema": f"r22_{tag.lower()}_summary_v1", "evidence_role": "PILOT/DEVELOPMENT", "test_read": False,
            "n_r22_units": sum(u["source"] == "r22" for u in units.values()),
            "n_r22_complete": sum(u["source"] == "r22" and u["ok"] for u in units.values()),
            "cost_r22_all_attempts": {"attempts": len(att), "gpu_hours": sum(a["elapsed_seconds"] or 0 for a in att) / 3600,
                                      "by_gpu": {g: sum(a["elapsed_seconds"] or 0 for a in att if a["gpu_name"] == g) / 3600
                                                 for g in sorted({a["gpu_name"] or "?" for a in att})}},
            **extra, "units": [dict(u, key=list(k)) for k, u in sorted(units.items(), key=lambda x: (x[0][0], x[0][1]))]}
    (out / f"R22_{tag}_SUMMARY.json").write_text(json.dumps(body, indent=1, default=str) + "\n")
    flat = [{"family": u["family"], "seed": u["seed"], "source": u["source"], "ok": u["ok"],
             **{f"cal_{m}": u["cal"].get(m) for m in (KD_METRICS if tag == "KD" else AIG_METRICS)},
             "selected_epoch": u.get("selected_epoch", u.get("best_epoch")),
             "gpu": ";".join(a["gpu_name"] or "?" for a in (u.get("attempts") or [])),
             "attempt_seconds": sum(a["elapsed_seconds"] or 0 for a in (u.get("attempts") or []))} for u in units.values()]
    with (out / f"R22_{tag}_PER_RUN.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)
    m = "top1" if tag == "KD" else "srcc"
    print(json.dumps({"tag": tag, "r22_complete": body["n_r22_complete"], "r22_units": body["n_r22_units"],
                      "gpu_hours_r22": round(body["cost_r22_all_attempts"]["gpu_hours"], 3)}))
    for p in extra.get("paired_differences_finals", extra.get("paired_differences", [])):
        print(f"{p['a']} - {p['b']}: {m} {p['metrics'][m]['delta_per_seed']} mean {p['metrics'][m]['mean']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--app", choices=["kd", "aigiqa"], required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()
    (kd if args.app == "kd" else aig)(args)


if __name__ == "__main__":
    main()
