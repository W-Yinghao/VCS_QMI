"""Round 20261021 apps summaries: per-seed CAL results, pre-registered paired differences, cost of every attempt.

PILOT/DEVELOPMENT, FIT/CAL only: reads the fixed run dirs (result/run/history JSON, best-epoch CAL predictions) and
the per-job Slurm records. No TEST file exists for these runs and none is opened. Missing/failed runs stay null.
Pairs follow APPS_SERVER_CN §4-5 and 03_RESULT_ANALYSIS §2; three seeds give per-seed deltas, mean and sample SD
only. AIG also gets a per-seed paired prompt bootstrap (package aigiqa.metrics.paired_bootstrap; prompt groups are
the resampling unit, not training-seed uncertainty) and per-prompt regret wins/losses.
Usage: python summarize.py --app aigiqa|kd --stage STAGE --apps PACKAGE_APPS_DIR --records R21_REC --out DIR
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
import sys
from pathlib import Path

AIG_PAIRS = [("r1_alternating_target_sg_exact", "r1_joint_scheduled_exact", "primary: recipe effect on exact"),
             ("r1_alternating_target_sg_bce", "r1_joint_scheduled_bce", "primary: recipe effect on BCE"),
             ("r1_alternating_target_sg_exact", "r1_alternating_target_sg_bce", "objective difference under the new recipe"),
             ("r1_alternating_target_sg_exact", "r1_task_only", "application value"),
             ("r1_joint_scheduled_exact", "r1_task_only", "auxiliary arm minus task-only"),
             ("r1_joint_scheduled_bce", "r1_task_only", "auxiliary arm minus task-only"),
             ("r1_alternating_target_sg_bce", "r1_task_only", "auxiliary arm minus task-only")]
AIG_METRICS = ["srcc", "plcc", "within_prompt_pairwise_macro", "within_prompt_pairwise_micro",
               "within_prompt_srcc_macro", "normalized_regret", "rmse_normalized"]
KD_PAIRS = [("r1_kd_alternating_target_sg_exact", "r1_kd_dkd_anchor", "primary"),
            ("r1_kd_alternating_target_sg_exact", "r1_kd_alternating_target_sg_bce", "primary"),
            ("r1_kd_alternating_target_sg_bce", "r1_kd_dkd_anchor", "secondary")]
KD_METRICS = ["top1", "top5", "nll", "ece_15bins", "brier"]
LOWER_BETTER = {"normalized_regret", "rmse_normalized", "nll", "ece_15bins", "brier"}


def load(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def mean_sd(values):
    values = [v for v in values if v is not None]
    return (statistics.mean(values) if values else None, statistics.stdev(values) if len(values) > 1 else None)


def attempts(records, app, row):
    out = []
    for p in sorted(glob.glob(f"{records}/{app}/{row}/slurm_*/slurm.json")):
        rec = load(p)
        end, start = rec.get("end", {}), rec.get("start", {})
        smi = (start.get("nvidia_smi") or "").split(";")[0].split(",")
        out.append({"slurm_job_id": start.get("slurm", {}).get("SLURM_JOB_ID"), "node": start.get("slurm", {}).get("SLURMD_NODENAME"),
                    "partition": start.get("slurm", {}).get("SLURM_JOB_PARTITION"),
                    "gpu_name": smi[1].strip() if len(smi) > 3 else None, "driver": smi[3].strip() if len(smi) > 3 else None,
                    "exit_code": end.get("exit_code"), "elapsed_seconds": end.get("elapsed_seconds"),
                    "start_utc": start.get("utc"), "end_utc": end.get("utc")})
    return out


def pair_table(units, pairs, metrics):
    table = []
    for a, b, role in pairs:
        seeds = sorted(s for (f, s), u in units.items() if f == a and u["status_ok"] and units.get((b, s), {}).get("status_ok"))
        entry = {"a": a, "b": b, "role": role, "paired_seeds": seeds, "metrics": {}}
        for m in metrics:
            d = [units[(a, s)]["cal"][m] - units[(b, s)]["cal"][m] if units[(a, s)]["cal"].get(m) is not None
                 and units[(b, s)]["cal"].get(m) is not None else None for s in seeds]
            mu, sd = mean_sd(d)
            entry["metrics"][m] = {"delta_per_seed": d, "mean": mu, "sample_sd": sd,
                                   "n_positive": sum(x > 0 for x in d if x is not None),
                                   "better_direction": "lower" if m in LOWER_BETTER else "higher"}
        table.append(entry)
    return table


def aig(args, rows, out):
    from aigiqa.metrics import metrics as aig_metrics, paired_bootstrap
    units, flat = {}, []
    for r in rows:
        key = (r["family"], int(r["seed"]))
        run = Path(r["run_dir"]); res = load(run / "result.json"); hist = load(run / "history.json") or []
        failed = load(run / "FAILED.json")
        ok = bool(res and res.get("status") == "COMPLETED_FIT_CAL_ONLY" and not failed)
        active = [h["train_diagnostics"] for h in hist if h["train_diagnostics"].get("active_aux_weight", 0) > 0]
        diag = {}
        for k in ("fusion_weighted_aux_over_anchor", "fusion_anchor_aux_cosine"):
            diag[k + "_mean_active_epochs"] = mean_sd([d.get(k) for d in active])[0]
        crit = sorted({k for d in active for k in d if "critic_fit_loss" in k or k.endswith("raw_j")})
        for k in crit:
            diag[k + "_mean_active_epochs"] = mean_sd([d.get(k) for d in active])[0]
        u = {"family": r["family"], "seed": int(r["seed"]), "status": res["status"] if res else ("FAILED" if failed else "MISSING"),
             "status_ok": ok, "best_epoch": res.get("best_epoch") if res else None,
             "cal": {m: (res or {}).get("cal_metrics", {}).get(m) for m in AIG_METRICS},
             "srcc_by_epoch": [h["cal_metrics"]["srcc"] for h in hist],
             "fit_seconds": (res or {}).get("fit_seconds_excludes_warmup"), "peak_allocated_bytes": (res or {}).get("peak_allocated_bytes"),
             "checkpoint_sha256": (res or {}).get("checkpoint_sha256"), "config_sha256": r["config_sha256"],
             "critic_updates_total": (res or {}).get("critic_updates_total"), "train_diagnostics": diag,
             "attempts": attempts(args.records, "aigiqa", f"{r['family']}__seed{r['seed']}"),
             "cal_predictions": str(run / "cal_predictions.csv") if ok else None}
        units[key] = u
        flat.append({k: v for k, v in u.items() if k not in ("cal", "srcc_by_epoch", "train_diagnostics", "attempts")}
                    | {f"cal_{m}": u["cal"][m] for m in AIG_METRICS}
                    | {"gpu": ";".join(a["gpu_name"] or "?" for a in u["attempts"]),
                       "attempt_seconds": sum(a["elapsed_seconds"] or 0 for a in u["attempts"])})
    pairs = pair_table(units, AIG_PAIRS, AIG_METRICS)
    preds = {k: list(csv.DictReader(open(u["cal_predictions"]))) for k, u in units.items() if u["cal_predictions"]}
    for entry in pairs:
        entry["per_seed_prompt_bootstrap"], entry["per_prompt_regret"] = {}, {}
        for s in entry["paired_seeds"]:
            ra, rb = preds[(entry["a"], s)], preds[(entry["b"], s)]
            entry["per_seed_prompt_bootstrap"][s] = paired_bootstrap(ra, rb, repetitions=args.bootstrap)["differences"]
            groups = sorted({x["prompt_group"] for x in ra})
            deltas = []
            for g in groups:
                ga = [x for x in ra if x["prompt_group"] == g]; gb = [x for x in rb if x["prompt_group"] == g]
                if len(ga) < 2:
                    continue
                deltas.append((aig_metrics(ga)["normalized_regret"] - aig_metrics(gb)["normalized_regret"], g))
            entry["per_prompt_regret"][s] = {
                "n_groups": len(deltas), "a_lower_regret": sum(d < 0 for d, _ in deltas),
                "a_higher_regret": sum(d > 0 for d, _ in deltas), "equal": sum(d == 0 for d, _ in deltas),
                "largest_a_worse": [{"prompt_group": g, "regret_delta": d} for d, g in sorted(deltas, reverse=True)[:5] if d > 0]}
    did = {}
    for m in AIG_METRICS:
        ex = dict(zip(pairs[0]["paired_seeds"], pairs[0]["metrics"][m]["delta_per_seed"]))
        bc = dict(zip(pairs[1]["paired_seeds"], pairs[1]["metrics"][m]["delta_per_seed"]))
        seeds = sorted(set(ex) & set(bc))
        d = [ex[s] - bc[s] if ex[s] is not None and bc[s] is not None else None for s in seeds]
        did[m] = {"seeds": seeds, "per_seed": d, "mean": mean_sd(d)[0], "sample_sd": mean_sd(d)[1]}
    write(out, "AIG", units, flat, pairs, {"exact_change_minus_bce_change (descriptive only)": did})


def kd(args, rows, out):
    units, flat = {}, []
    for r in rows:
        key = (r["family"], int(r["seed"]))
        run = Path(r["run_dir"]); rj = load(run / "run.json"); best = load(run / "best_cal_metrics.json")
        hist = load(run / "history.json") or []
        ok = bool(rj and rj.get("status") == "complete" and best)
        probes = [(h["epoch"], h.get("gradient_probe") or {}) for h in hist]
        act = [p for e, p in probes if e >= 20 and p.get("weighted_aux_over_anchor") is not None]
        last = hist[-1] if hist else {}
        dep = last.get("dependence_aggregate", {})
        u = {"family": r["family"], "seed": int(r["seed"]), "status": rj["status"] if rj else "MISSING", "status_ok": ok,
             "selected_epoch": (best or {}).get("selected_epoch"), "cal": {m: (best or {}).get(m) for m in KD_METRICS},
             "top1_by_epoch": [h["cal"]["top1"] for h in hist],
             "gradient_probe_active_epochs": {"n_epochs": len(act),
                 "weighted_aux_over_anchor_mean": mean_sd([p["weighted_aux_over_anchor"] for p in act])[0],
                 "anchor_aux_cosine_mean": mean_sd([p.get("anchor_aux_cosine") for p in act])[0],
                 "scope": "first FIT batch of each epoch, epochs 21-40, anchor CE+actual DKD"},
             "final_epoch_dependence": {k: dep.get(k) for k in ("dependence_critic_fit_loss_before_mean",
                                        "dependence_critic_fit_loss_after_mean", "raw_j_mean", "raw_j_min", "raw_j_max",
                                        "dependence_x_batch_std_mean_mean", "dependence_y_batch_std_mean_mean")},
             "fit_wall_seconds_sum": (rj or {}).get("fit_wall_seconds_sum"), "cuda_peak_allocated_bytes": (rj or {}).get("cuda_peak_allocated_bytes"),
             "best_checkpoint_sha256": (rj or {}).get("best_checkpoint_sha256"), "config_sha256": r["config_sha256"],
             "teacher_frozen_verified": (rj or {}).get("teacher_frozen_verified"),
             "critic_updates_total": (rj or {}).get("critic_updates_total"),
             "attempts": attempts(args.records, "kd", f"{r['family']}__seed{r['seed']}")}
        units[key] = u
        flat.append({k: v for k, v in u.items() if k not in ("cal", "top1_by_epoch", "gradient_probe_active_epochs", "final_epoch_dependence", "attempts")}
                    | {f"cal_{m}": u["cal"][m] for m in KD_METRICS}
                    | {"grad_ratio_active_mean": u["gradient_probe_active_epochs"]["weighted_aux_over_anchor_mean"],
                       "grad_cos_active_mean": u["gradient_probe_active_epochs"]["anchor_aux_cosine_mean"],
                       "gpu": ";".join(a["gpu_name"] or "?" for a in u["attempts"]),
                       "attempt_seconds": sum(a["elapsed_seconds"] or 0 for a in u["attempts"])})
    write(out, "KD", units, flat, pair_table(units, KD_PAIRS, KD_METRICS), {})


def write(out, tag, units, flat, pairs, extra):
    out.mkdir(parents=True, exist_ok=True)
    n_att = sum(len(u["attempts"]) for u in units.values())
    gpu_h = sum(a["elapsed_seconds"] or 0 for u in units.values() for a in u["attempts"]) / 3600
    body = {"schema": f"r21_{tag.lower()}_summary_v1", "evidence_role": "PILOT/DEVELOPMENT", "test_read": False,
            "n_units": len(units), "n_complete": sum(u["status_ok"] for u in units.values()),
            "cost": {"attempts": n_att, "gpu_hours_all_attempts": gpu_h},
            "units": [units[k] for k in sorted(units)], "paired_differences": pairs, **extra}
    (out / f"{tag}_SUMMARY.json").write_text(json.dumps(body, indent=1, default=str) + "\n")
    with (out / f"{tag}_PER_SEED.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)
    print(json.dumps({"tag": tag, "complete": body["n_complete"], "units": len(units), "gpu_hours": round(gpu_h, 3)}))
    for p in pairs:
        m = "srcc" if tag == "AIG" else "top1"
        print(f"{p['a']} - {p['b']}: {m} per seed {p['metrics'][m]['delta_per_seed']} mean {p['metrics'][m]['mean']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--app", choices=["aigiqa", "kd"], required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--apps", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()
    sys.path[:0] = [str(Path(args.stage).resolve()), str(Path(args.stage).resolve() / "shared")]
    sub = "aigiqa" if args.app == "aigiqa" else "kd_probe"
    rows = list(csv.DictReader(open(Path(args.apps) / "configs" / sub / "JOBS.csv")))
    (aig if args.app == "aigiqa" else kd)(args, rows, Path(args.out))


if __name__ == "__main__":
    main()
