"""P140 — v7 V7-CORE tables and status files, built ONLY from raw per-run files (never from rounded numbers in reports).

Plan: VCS_SSL_Server_Plan_v7_CN.md §3 (tables A / B / C-input), §8 (grid sensitivity, costs), §11 (minimal records, delivery files).
Sources per run (outputs/<run_id>/): config.resolved.yaml (method / loss / scorer / pairing / lr / dims), evaluations/evaluation_epoch_<E>.json
(final readout at the run's own epochs E — never a best epoch), summary.json (cumulative train / eval seconds, peak memory), logs/epochs.jsonl
(per-epoch utc -> wall-clock segments / requeues), status.json; slurm_logs/unit_*.out banners (host / partition per job; a requeued job keeps its
id and its log is truncated, so earlier segments come from slurm_logs/gpu_upgrade.log); squeue for the live map.
Writes reports/results_v7.json (table A), reports/paired_comparisons_v7.json (table B + selected-vs-selected + grid sensitivity + cross-checks),
reports/runtime_v7.json (costs), reports/status_v7.md (summary + maps + incidents).  Analysis only.
    python scripts/p140_v7_tables.py
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("/home/infres/yinwang/CS_QMI/outputs")
LOGS = Path("/home/infres/yinwang/CS_QMI/slurm_logs")
REP = ROOT / "reports"
LINES = ("P35", "P41", "P91", "P104", "P107", "P111", "P112", "P114", "P115", "P120", "P120A1", "P126", "P127", "P129", "P130", "P133", "P135", "P136", "P137", "P138", "P145", "P146", "P147", "P148")
RUN_RE = re.compile(r"^(%s)_" % "|".join(sorted(LINES, key=len, reverse=True)))
CLOSE, THR = 0.3, {"cifar10": 0.30, "cifar100": 0.50}  # P114 labels; P127 / P129 / P135 replacement thresholds
GRID = [(1.5, 0.5), (2.0, 0.25), (2.0, 0.5), (2.0, 0.75), (3.0, 0.5), (3.0, 0.25)]
NEIGH = {(1.5, 0.5): "a-", (3.0, 0.5): "a+", (2.0, 0.25): "kappa-", (2.0, 0.75): "kappa+"}  # one-axis neighbours of the default (2, 0.5)


def ts(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def iso(t):
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


# ---------------------------------------------------------------------------------------------------------------- slurm log index
def index_unit_logs():
    """run_id -> list of {job, host, partition, start, end, exit, cuda_missing, file} from unit_*.out banners (last segment per job id)."""
    idx = defaultdict(list)
    for f in glob.glob(str(LOGS / "unit_*.out")):
        try:
            txt = open(f, errors="ignore").read()
        except Exception:
            continue
        m = re.search(r"unit (\S+): cfg", txt[:3000])
        if not m:
            continue
        h = re.search(r"host=(\S+) job=(\d+) partition=(\S+) gpus=\S+ start=(\S+)", txt[:3000])
        e = re.findall(r"exit=(\d+) end=(\S+)", txt[-3000:])
        idx[m.group(1)].append({"file": os.path.basename(f), "job": h.group(2) if h else None, "host": h.group(1) if h else None,
                                "partition": h.group(3) if h else None, "start": h.group(4) if h else None,
                                "exit": int(e[-1][0]) if e else None, "end": e[-1][1] if e else None,
                                "cuda_missing": "no CUDA device visible" in txt})
    return idx


def parse_moves():
    """gpu_upgrade.log: requeue / move events (source host of the interrupted segment)."""
    ev = []
    p = LOGS / "gpu_upgrade.log"
    if not p.exists():
        return ev
    for line in open(p):
        m = re.match(r"(\S+Z) (.*)", line.strip())
        if not m:
            continue
        t, msg = m.groups()
        mm = re.search(r"(?:move|rotate) (\d+) \((\S+), (\d+) epochs left\)\s*(\S+?)/(\S+?):gpu(\d)", msg)
        if mm:
            ev.append({"utc": t, "job": mm.group(1), "run_id": mm.group(2), "epochs_left": int(mm.group(3)), "from_partition": mm.group(4),
                       "from_node": mm.group(5), "from_gpu": int(mm.group(6)), "text": msg})
            continue
        mr = re.search(r"rotate (\d+) \((\S+), (\d+) epochs left, (\S+?)/(\S+?):gpu(\d)", msg)
        if mr:
            ev.append({"utc": t, "job": mr.group(1), "run_id": mr.group(2), "epochs_left": int(mr.group(3)), "from_partition": mr.group(4),
                       "from_node": mr.group(5), "from_gpu": int(mr.group(6)), "text": msg}); continue
        mm2 = re.search(r"(\d{7}) (?:\(([^)]*)\) )?requeued off (?:(\S+) )?(node\d+)(?::gpu(\d))?", msg)
        if mm2:
            ev.append({"utc": t, "job": mm2.group(1), "run_id": None, "from_partition": mm2.group(3), "from_node": mm2.group(4),
                       "from_gpu": int(mm2.group(5)) if mm2.group(5) else None, "text": msg}); continue
        if "manual move" in msg or "requeued" in msg or "stopped pid" in msg or "BadConstraints" in msg:
            ev.append({"utc": t, "text": msg})
    return ev


GPU_CLASS_S_PER_EPOCH = [(25, "RTX6000PRO healthy"), (36.5, "H100"), (41.5, "L40S"), (1e9, "RTX6000PRO throttled (node60 / node61 GPU1)")]


def infer_gpu(s_per_epoch, rec):
    """Only for ResNet-18, 4 views, B 256 units (calibrated there); otherwise None."""
    if s_per_epoch is None or not (rec["backbone"] == "resnet18_cifar" and rec["views"] == 4 and rec["batch_images"] == 256):
        return None
    for lim, name in GPU_CLASS_S_PER_EPOCH:
        if s_per_epoch < lim:
            return name + " (inferred from s/epoch)"


# ---------------------------------------------------------------------------------------------------------------- per-run record
def describe(rid, cfg):
    c = cfg
    m, loss = c["run"].get("method"), c["objective"]["loss"]
    cr = c["model"]["critic"]
    fam = {"negative_J": "VCS", "js_matched_logistic": "JS", "nt_xent": "SimCLR", "variance_invariance_covariance": "VICReg"}.get(loss, loss)
    pr = c["pairing"]
    scorer = None
    if fam in ("VCS", "JS") and cr.get("enabled"):
        a, b = cr.get("cosine_scale_init"), cr.get("cosine_bias_init")
        scorer = {"affine_mode": cr.get("affine_mode") or "learned/recipe", "a": a, "b": b,
                  "kappa": (round(-b / a, 6) if (a and b is not None) else None), "curvature_lambda": cr.get("curvature_lambda"),
                  "trainable": (cr.get("affine_mode") not in ("fixed", "fixed_curved"))}
    elif fam == "SimCLR":
        scorer = {"temperature": c["objective"].get("simclr_temperature")}
    pairing = {"pair_scope": pr.get("pair_scope", "cross_view_k"), "k": pr.get("k"), "negative_detach": pr.get("negative_detach"),
               "momentum_encoder": pr.get("momentum_encoder", False), "moco_consistent": pr.get("moco_consistent", False),
               "moco_use_queue": pr.get("moco_use_queue", True if pr.get("moco_consistent") else None)}
    proj = c["model"]["projector"]
    return {"family": fam, "method_field": m, "loss": loss, "scorer": scorer, "pairing": pairing,
            "dataset": c["data"]["name"], "backbone": c["model"]["backbone"], "h_dim": c["model"]["h_dim"],
            "projector": f"{c['model']['h_dim']}->{proj['hidden_dim']}->{proj['output_dim']}", "views": c["views"]["count"],
            "crop_min": c["views"]["random_resized_crop"]["scale"][0], "epochs": c["train"]["epochs"],
            "batch_images": c["train"]["batch_size_images"], "lr": c["optimizer"]["lr"], "optimizer": c["optimizer"]["name"],
            "seed": c["run"]["seed"], "stage": c["run"].get("stage")}


def segments(rid):
    p = OUT / rid / "logs" / "epochs.jsonl"
    if not p.exists():
        return [], None
    recs = {}
    for line in open(p):
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("epoch", 0) >= 1 and e.get("utc"):
            recs[e["epoch"]] = e  # resume rewrites an epoch: keep the last record
    if not recs:
        return [], None
    eps = sorted(recs)
    t = np.array([ts(recs[k]["utc"]) for k in eps]); tr = np.array([recs[k]["train_seconds"] for k in eps], float)
    gaps = np.diff(t)
    med = float(np.median(gaps)) if len(gaps) else None
    cut = [0] + [i + 1 for i, g in enumerate(gaps) if g > max(600.0, 10 * (med or 0))] + [len(eps)]
    segs = []
    for a, b in zip(cut[:-1], cut[1:]):
        d = np.diff(tr[a:b]); d = d[d > 0]
        spe = float(np.median(d)) if len(d) else None
        segs.append({"first_epoch": eps[a], "last_epoch": eps[b - 1], "n_epochs": b - a, "start_utc": iso(t[a] - (spe or 0)), "end_utc": iso(t[b - 1]),
                     "s_per_epoch": round(spe, 2) if spe else None})
    return segs, med


def collect():
    logs = index_unit_logs(); moves = parse_moves()
    job2run = {b["job"]: r for r, bs in logs.items() for b in bs if b["job"]}
    move_by_run = defaultdict(list)
    for e in moves:
        if e.get("job") and not e.get("run_id"):
            e["run_id"] = job2run.get(e["job"])
        if e.get("run_id"):
            move_by_run[e["run_id"]].append(e)
    runs = {}
    for d in sorted(OUT.iterdir()):
        rid = d.name
        if not RUN_RE.match(rid) or not (d / "config.resolved.yaml").exists() or "SMOKE" in rid.upper() or "GATE" in rid.upper():
            continue
        try:
            cfg = yaml.safe_load(open(d / "config.resolved.yaml"))
        except Exception:
            continue
        rec = {"run_id": rid, "line": RUN_RE.match(rid).group(1)} | describe(rid, cfg)
        st = load_json(d / "status.json") or {}
        rec["status"] = st.get("status"); rec["completed_epoch"] = st.get("completed_epoch")
        ev = load_json(d / "evaluations" / f"evaluation_epoch_{rec['epochs']:03d}.json") or load_json(d / "evaluations" / f"evaluation_epoch_{rec['epochs']}.json")
        rec["final"] = ({"epoch": rec["epochs"], "linear": ev["linear_val_top1_pct"], "knn": ev["knn_val_top1_pct"],
                         "eval_file": str((d / "evaluations").relative_to(OUT.parent))} if ev else None)
        if rec["epochs"] == 1600:
            e8 = load_json(d / "evaluations" / "evaluation_epoch_800.json")
            rec["intermediate_800"] = {"linear": e8["linear_val_top1_pct"], "knn": e8["knn_val_top1_pct"]} if e8 else None
        sm = load_json(d / "summary.json") or {}
        segs, _ = segments(rid)
        banners = sorted(logs.get(rid, []), key=lambda x: x["start"] or "")
        for s in segs:  # attach host: the banner whose start lies within (segment start − 15 min, segment end)
            s["host"] = s["partition"] = None; s["host_source"] = None
            for bnr in banners:
                if bnr["start"] and ts(s["start_utc"]) - 900 <= ts(bnr["start"]) <= ts(s["end_utc"]):
                    s["host"], s["partition"], s["job"], s["host_source"] = bnr["host"], bnr["partition"], bnr["job"], "unit log banner"
            if s["host"] is None:
                for mv in move_by_run.get(rid, []):
                    if ts(s["start_utc"]) <= ts(mv["utc"]) <= ts(s["end_utc"]) + 900:
                        s["host"] = mv["from_node"] + (f":gpu{mv['from_gpu']}" if mv.get("from_gpu") is not None else ""); s["partition"] = mv.get("from_partition"); s["host_source"] = "gpu_upgrade.log move"
            if s["host"] is None:
                s["gpu_inferred"] = infer_gpu(s["s_per_epoch"], rec)
            if s.get("host") == "node61" and s["s_per_epoch"] and s["s_per_epoch"] > 40 and rec["backbone"] == "resnet18_cifar":
                s["host"] = "node61:gpu1 (throttled; GPU index inferred from s/epoch)"
        compute_s = (sm.get("train_seconds") or 0) + (sm.get("eval_seconds") or 0)
        rec["cost"] = {"gpu_hours_train_plus_periodic_eval": round(compute_s / 3600, 3) if compute_s else None,
                       "peak_allocated_gb": round(sm["peak_allocated_mb"] / 1024, 2) if sm.get("peak_allocated_mb") else None,
                       "peak_reserved_gb": round(sm["peak_reserved_mb"] / 1024, 2) if sm.get("peak_reserved_mb") else None,
                       "n_segments": len(segs), "segments": segs,
                       "wall_hours": round(sum(ts(s["end_utc"]) - ts(s["start_utc"]) for s in segs) / 3600, 2) if segs else None}
        rec["infra_events"] = [b for b in logs.get(rid, []) if b["cuda_missing"]] + move_by_run.get(rid, [])
        rec["role"] = "selection (seed 0)" if rec["seed"] == 0 else "confirmation seed"
        runs[rid] = rec
    return runs, logs, moves


# ---------------------------------------------------------------------------------------------------------------- statistics
def paired(runs, pairs, key, planned=None):
    """pairs: list of (seed, run_a, run_b). Returns per-seed values and A − B statistics on key ('linear' / 'knn')."""
    rows, d = [], []
    for s, ra, rb in pairs:
        A, B = runs.get(ra), runs.get(rb)
        if not (A and B and A["final"] and B["final"]):
            rows.append({"seed": s, "a": ra, "b": rb, "missing": [r for r, x in ((ra, A), (rb, B)) if not (x and x["final"])]})
            continue
        va, vb = A["final"][key], B["final"][key]; d.append(va - vb)
        rows.append({"seed": s, "a": ra, "b": rb, "A": va, "B": vb, "delta": round(va - vb, 4)})
    out = {"per_seed": rows, "n": len(d)}
    if d:
        x = np.array(d); m = float(x.mean()); out["mean_delta"] = round(m, 4)
        A_vals = [r["A"] for r in rows if "A" in r]; B_vals = [r["B"] for r in rows if "B" in r]
        out["A_mean"], out["B_mean"] = round(float(np.mean(A_vals)), 4), round(float(np.mean(B_vals)), 4)
        if len(d) > 1:
            sd = float(x.std(ddof=1)); h = float(stats.t.ppf(0.975, len(d) - 1) * sd / np.sqrt(len(d)))
            out.update({"sd_delta": round(sd, 4), "ci95": [round(m - h, 4), round(m + h, 4)],
                        "A_sd": round(float(np.std(A_vals, ddof=1)), 4), "B_sd": round(float(np.std(B_vals, ddof=1)), 4),
                        "A_range_max_minus_min": round(max(A_vals) - min(A_vals), 4), "B_range_max_minus_min": round(max(B_vals) - min(B_vals), 4)})
            out["label"] = "close" if abs(m) < CLOSE else ("clear" if (m - h > 0 or m + h < 0) else "inconclusive")
            if planned and len(d) < planned:
                out["label"] = f"INCOMPLETE ({len(d)}/{planned} seeds; label not final)"
        else:
            out["label"] = "single seed (no interval)"
    return out


def config_diff(runs, ra, rb):
    """Keys of config.resolved.yaml that differ between two runs, ignoring run identity, stage and the JS markers / loss."""
    def flat(d, p=""):
        o = {}
        for k, v in d.items():
            if isinstance(v, dict):
                o |= flat(v, p + k + ".")
            else:
                o[p + k] = v
        return o
    try:
        fa = flat(yaml.safe_load(open(OUT / ra / "config.resolved.yaml"))); fb = flat(yaml.safe_load(open(OUT / rb / "config.resolved.yaml")))
    except Exception:
        return None
    skip = re.compile(r"^(run\.(seed|stage|run_id|name|notes?)|objective\.(loss|js_fixed_scorer|js_all_view_tokens)|logging\.|execution\.)")
    return sorted(k for k in set(fa) | set(fb) if not skip.match(k) and fa.get(k) != fb.get(k))


def g(v):
    return f"{v:g}"


def tables(runs):
    res = {}
    # ---- Table B: same-configuration VCS vs matched JS (paired by seed)
    B = {}
    B["P114_cifar10_shared_(2,0.5)_lr1e-3"] = [(s, f"P107_AP3_views4_800ep_seed{s}", f"P114_JSAP3_views4_800ep_seed{s}") for s in range(3)]
    B["P120+A1_cifar100_shared_(2,0.5)_lr1e-3"] = [(s, (f"P107_AP3_c100_views4_800ep_seed{s}" if s < 3 else f"P120A1_AP3_c100_views4_800ep_seed{s}"),
                                                   f"P120_JS_AP3_c100_views4_800ep_seed{s}") for s in range(5)]
    for ds in ("c10", "c100"):
        for lr in ("lr0.5x", "lr2x"):
            B[f"P135_{ds}_{lr}_(2,0.5)"] = [(0, f"P135_vcs_{ds}_{lr}_seed0", f"P135_js_{ds}_{lr}_seed0")]
        for a, k in GRID:
            if (a, k) == (2.0, 0.5):
                continue
            B[f"P127vsP129_{ds}_({g(a)},{g(k)})_lr1e-3"] = [(0, f"P127_AP3_{ds}_a{g(a)}_k{g(k)}_seed0", f"P129_JS_{ds}_a{g(a)}_k{g(k)}_seed0")]
    tb = {}
    for name, prs in B.items():
        tb[name] = {"what": "VCS − matched JS, same structure (only the loss differs)", "linear": paired(runs, prs, "linear", len(prs)), "knn": paired(runs, prs, "knn", len(prs)),
                    "config_diff_besides_loss": sorted({k for _, a, b in prs for k in (config_diff(runs, a, b) or [])})}
    res["table_B_same_configuration"] = tb
    # ---- selected-vs-selected (P127 rule-selected VCS cell vs P129 rule-selected JS cell), seeds paired
    sel = {"cifar10": ("(2,0.5)", "(2,0.25)", [(s, f"P107_AP3_views4_800ep_seed{s}", ("P129_JS_c10_a2_k0.25_seed0" if s == 0 else f"P129_JS_c10_a2_k0.25_seed{s}")) for s in range(3)]),
           "cifar100": ("(2,0.5)", "(3,0.5)", [(s, f"P107_AP3_c100_views4_800ep_seed{s}", f"P129_JS_c100_a3_k0.5_seed{s}") for s in range(3)])}
    res["selected_vs_selected"] = {ds: {"label": "SELECTED-vs-SELECTED (each loss with its own rule-selected scorer) — not a single-factor contrast",
                                        "vcs_cell": v, "js_cell": j, "linear": paired(runs, p, "linear", 3), "knn": paired(runs, p, "knn", 3)}
                                   for ds, (v, j, p) in sel.items()}
    # ---- grid sensitivity (P127 VCS, P129 JS, P135 lr)
    sens = {}
    for fam, prefix, base in (("VCS", "P127_AP3", {"c10": "P107_AP3_views4_800ep_seed0", "c100": "P107_AP3_c100_views4_800ep_seed0"}),
                              ("JS", "P129_JS", {"c10": "P114_JSAP3_views4_800ep_seed0", "c100": "P120_JS_AP3_c100_views4_800ep_seed0"})):
        for ds in ("c10", "c100"):
            dname = "cifar10" if ds == "c10" else "cifar100"
            cells = {"(2,0.5)": base[ds]} | {f"({g(a)},{g(k)})": f"{prefix}_{ds}_a{g(a)}_k{g(k)}_seed0" for a, k in GRID if (a, k) != (2.0, 0.5)}
            vals = {c: runs[r]["final"] for c, r in cells.items() if runs.get(r) and runs[r]["final"]}
            if len(vals) < len(cells):
                sens[f"{fam}_{dname}_(a,kappa)"] = {"incomplete": sorted(set(cells) - set(vals))}; continue
            dflt = vals["(2,0.5)"]["linear"]; best = max(vals, key=lambda c: vals[c]["linear"]); gain = vals[best]["linear"] - dflt
            sel_cell = best if (best != "(2,0.5)" and gain >= THR[dname]) else "(2,0.5)"
            sens[f"{fam}_{dname}_(a,kappa)"] = {
                "default_(2,0.5)": vals["(2,0.5)"], "highest_tested": {"cell": best, **vals[best], "gain_vs_default": round(gain, 4)},
                "rule_selected": {"cell": sel_cell, "threshold": THR[dname], **vals[sel_cell]},
                "full_range_linear": [round(min(v["linear"] for v in vals.values()), 2), round(max(v["linear"] for v in vals.values()), 2)],
                "full_range_knn": [round(min(v["knn"] for v in vals.values()), 2), round(max(v["knn"] for v in vals.values()), 2)],
                "local_neighbourhood_of_default": {NEIGH[(a, k)] + f" ({g(a)},{g(k)})": round(vals[f"({g(a)},{g(k)})"]["linear"] - dflt, 4)
                                                   for (a, k) in NEIGH},
                "all_cells": vals, "seed": 0}
    for fam, base in (("VCS", {"c10": "P107_AP3_views4_800ep_seed0", "c100": "P107_AP3_c100_views4_800ep_seed0"}),
                      ("JS", {"c10": "P114_JSAP3_views4_800ep_seed0", "c100": "P120_JS_AP3_c100_views4_800ep_seed0"})):
        for ds in ("c10", "c100"):
            dname = "cifar10" if ds == "c10" else "cifar100"
            cells = {"5e-4": f"P135_{fam.lower()}_{ds}_lr0.5x_seed0", "1e-3": base[ds], "2e-3": f"P135_{fam.lower()}_{ds}_lr2x_seed0"}
            vals = {c: runs[r]["final"] for c, r in cells.items() if runs.get(r) and runs[r]["final"]}
            if len(vals) < 3:
                sens[f"{fam}_{dname}_lr"] = {"incomplete": sorted(set(cells) - set(vals))}; continue
            dflt = vals["1e-3"]["linear"]; best = max(vals, key=lambda c: vals[c]["linear"]); gain = vals[best]["linear"] - dflt
            sens[f"{fam}_{dname}_lr"] = {"default_1e-3": vals["1e-3"], "highest_tested": {"lr": best, **vals[best], "gain_vs_default": round(gain, 4)},
                                         "rule_selected": {"lr": best if (best != "1e-3" and gain >= THR[dname]) else "1e-3", "threshold": THR[dname]},
                                         "full_range_linear": [round(min(v["linear"] for v in vals.values()), 2), round(max(v["linear"] for v in vals.values()), 2)],
                                         "local_neighbourhood_of_default": {k: round(vals[k]["linear"] - dflt, 4) for k in ("5e-4", "2e-3")},
                                         "all_cells": vals, "seed": 0, "scorer": "(2,0.5)"}
    res["grid_sensitivity"] = sens
    # ---- cross-checks of headline numbers quoted in earlier reports (raw recomputation)
    def multi(names):
        v = [runs[n]["final"]["linear"] for n in names if runs.get(n) and runs[n]["final"]]
        k = [runs[n]["final"]["knn"] for n in names if runs.get(n) and runs[n]["final"]]
        return {"n": len(v), "linear_mean": round(float(np.mean(v)), 4) if v else None, "linear_sd": round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else None,
                "linear_range": round(max(v) - min(v), 4) if v else None, "knn_mean": round(float(np.mean(k)), 4) if k else None}
    res["cross_checks"] = {
        "A-P3_C10_5seeds (P107 layer-2: 89.02 ± 0.04)": multi([f"P107_AP3_views4_800ep_seed{s}" for s in range(5)]),
        "SimCLR_C10_5seeds (P107 layer-2: 88.31 ± 0.21)": multi([f"P41_simclr_views4_800ep_seed{s}" for s in range(3)] + [f"P111_simclr_views4_800ep_seed{s}" for s in (3, 4)]),
        "A-P3_C100_5seeds (P120 A1: 60.15 ± 0.38)": multi([f"P107_AP3_c100_views4_800ep_seed{s}" for s in range(3)] + [f"P120A1_AP3_c100_views4_800ep_seed{s}" for s in (3, 4)]),
        "JS_C100_5seeds (P120 A1: 59.35 ± 0.42)": multi([f"P120_JS_AP3_c100_views4_800ep_seed{s}" for s in range(5)]),
        "JS_C10_(2,0.25)_3seeds (P129 A1: 88.97, 'spread' 0.66)": multi([f"P129_JS_c10_a2_k0.25_seed{s}" for s in range(3)]),
        "A-P3_C10_3seeds (P129 A1: 89.03, 'spread' 0.10)": multi([f"P107_AP3_views4_800ep_seed{s}" for s in range(3)]),
        "SimCLR_C100_3seeds (P91: 58.25 / 57.21)": multi([f"P91_c100_simclr_views4_800ep_seed{s}" for s in range(3)]),
        "FREE_C100_3seeds (P120: 56.74)": multi([f"P120_FREE_AP3_c100_views4_800ep_seed{s}" for s in range(3)]),
    }
    return res


# ---------------------------------------------------------------------------------------------------------------- status map
def live_map(runs, logs):
    sq = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", "yinwang"), "-o", "%i|%j|%T|%P|%N|%r|%M"], capture_output=True, text=True).stdout
    q = [dict(zip(("job", "name", "state", "partition", "node", "reason", "elapsed"), l.split("|"))) for l in sq.strip().split("\n") if l]
    units = {"P129": r"^P129_", "P130": r"^P130_", "P133": r"^P133_", "P135": r"^P135_", "P136": r"^P136_", "P137": r"^P137_"}
    jobs_by_prefix = {"P129": r"^p129", "P130": r"^p130", "P133": r"^p133", "P135": r"^p135", "P136": r"^p136", "P137": r"^p137"}
    m = {}
    for u, rx in units.items():
        rr = [r for r in runs.values() if re.match(rx, r["run_id"])]
        m[u] = {"completed": sorted(r["run_id"] for r in rr if r["final"]),
                "running_or_resumable": sorted(f"{r['run_id']} ({r['status']}, epoch {r['completed_epoch']})" for r in rr if not r["final"]),
                "queue": [f"{j['job']} {j['name']} {j['state']} {j['partition']} {j['node'] or ''} {j['reason']}".strip() for j in q if re.match(jobs_by_prefix[u], j["name"])]}
    return m, q


def main():
    runs, logs, moves = collect()
    tb = tables(runs)
    done = {k: v for k, v in runs.items() if v["final"] and v["epochs"] in (800, 1600)}  # table A: full-schedule runs only (800; P136 1600)
    tableA = sorted(({k: v[k] for k in ("run_id", "line", "family", "method_field", "loss", "dataset", "backbone", "projector", "views", "crop_min", "epochs",
                                           "batch_images", "scorer", "pairing", "lr", "optimizer", "seed", "role", "stage", "final", "status")}
                     | ({"intermediate_800": v["intermediate_800"]} if "intermediate_800" in v else {})
                     | {"gpu_hours": v["cost"]["gpu_hours_train_plus_periodic_eval"], "peak_allocated_gb": v["cost"]["peak_allocated_gb"],
                        "segments": len(v["cost"]["segments"])} for v in done.values()), key=lambda r: (r["dataset"], r["line"], r["run_id"]))
    json.dump({"generated_utc": iso(dt.datetime.now(dt.timezone.utc).timestamp()), "source": "raw per-run files (outputs/*), see script docstring",
               "readout": "final frozen-h linear / kNN at the run's own last epoch (development selection split)", "n_finished": len(tableA),
               "table_A": tableA, "unfinished": sorted(f"{k} ({v['status']}, epoch {v['completed_epoch']})" for k, v in runs.items() if not v["final"])},
              open(REP / "results_v7.json", "w"), indent=1)
    json.dump({"generated_utc": iso(dt.datetime.now(dt.timezone.utc).timestamp()), "labels": "P114: close |Δ|<0.3; clear |Δ|≥0.3 and 95% t interval excludes 0; else inconclusive; 'not significant' is never equivalence",
               **tb}, open(REP / "paired_comparisons_v7.json", "w"), indent=1)
    # runtime
    per_class = defaultdict(list)
    for v in done.values():
        for s in v["cost"]["segments"]:
            if v["backbone"] == "resnet18_cifar" and v["views"] == 4 and v["batch_images"] == 256 and s["s_per_epoch"]:
                cls = (s.get("partition") or "?") + ("" if not s.get("host") else f"/{s['host']}")
                per_class[cls].append(s["s_per_epoch"])
    json.dump({"generated_utc": iso(dt.datetime.now(dt.timezone.utc).timestamp()),
               "note": "gpu_hours = cumulative train + periodic-eval seconds (summary.json; restored across resume); wall segments from epochs.jsonl utc gaps; "
                       "a requeued job's unit log is truncated, so earlier segments take their host from gpu_upgrade.log or are inferred from s/epoch (flagged)",
               "per_run": {k: {"gpu_hours": v["cost"]["gpu_hours_train_plus_periodic_eval"], "wall_hours": v["cost"]["wall_hours"], "peak_allocated_gb": v["cost"]["peak_allocated_gb"],
                               "segments": v["cost"]["segments"], "infra_events": v["infra_events"]} for k, v in runs.items()},
               "s_per_epoch_by_partition_host_R18_4v_B256": {k: {"n_segments": len(x), "median": round(float(np.median(x)), 2)} for k, x in sorted(per_class.items())},
               "gpu_upgrade_events": moves,
               "cuda_missing_jobs": sorted([dict(run_id=r, **b) for r, bs in logs.items() for b in bs if b["cuda_missing"]], key=lambda x: x["end"] or "")},
              open(REP / "runtime_v7.json", "w"), indent=1)
    lm, q = live_map(runs, logs)
    write_status(runs, done, tb, lm, logs, moves)
    print(f"results_v7.json: {len(tableA)} finished runs; paired / runtime / status written")


def fmt_pair(p):
    L, K = p["linear"], p["knn"]
    if L.get("n", 0) == 0:
        return "missing"
    s = f"{L['A_mean']:.2f} vs {L['B_mean']:.2f}: Δ {L['mean_delta']:+.2f}"
    if "ci95" in L:
        s += f" [{L['ci95'][0]:+.2f}, {L['ci95'][1]:+.2f}]"
    s += f" ({L['label']}; n={L['n']}); kNN Δ {K['mean_delta']:+.2f}" + (f" ({K['label']})" if K.get("label") else "")
    return s


def write_status(runs, done, tb, lm, logs, moves):
    B, S, G, X = tb["table_B_same_configuration"], tb["selected_vs_selected"], tb["grid_sensitivity"], tb["cross_checks"]
    L = ["# status_v7 — V7-CORE (P140): executive summary, task map, incidents", "",
         f"Generated {iso(dt.datetime.now(dt.timezone.utc).timestamp())} by `scripts/p140_v7_tables.py` from raw per-run files only "
         f"({len(done)} finished runs in table A; details in `results_v7.json`, `paired_comparisons_v7.json`, `runtime_v7.json`).  Development selection split; "
         "official test closed.  Seed-0 selection runs and confirmation seeds are kept apart; different scorer / lr runs are never pooled as seeds.", "",
         "## 1. Executive summary", ""]
    L.append(f"- **Shared configuration, VCS − matched JS** (only the loss differs): CIFAR-10 3 seeds {fmt_pair(B['P114_cifar10_shared_(2,0.5)_lr1e-3'])}; "
             f"CIFAR-100 5 seeds {fmt_pair(B['P120+A1_cifar100_shared_(2,0.5)_lr1e-3'])}.")
    lr_bits = []
    for ds in ("c10", "c100"):
        for lr in ("lr0.5x", "lr2x"):
            p = B[f"P135_{ds}_{lr}_(2,0.5)"]["linear"]
            if p.get("n"):
                lr_bits.append(f"{ds} {lr} {p['mean_delta']:+.2f}")
    base0 = {"c10": B["P114_cifar10_shared_(2,0.5)_lr1e-3"]["linear"]["per_seed"][0].get("delta"), "c100": B["P120+A1_cifar100_shared_(2,0.5)_lr1e-3"]["linear"]["per_seed"][0].get("delta")}
    same = all((B[f"P135_{ds}_{lr}_(2,0.5)"]["linear"].get("mean_delta", 0) > 0) == (base0[ds] > 0) for ds in ("c10", "c100") for lr in ("lr0.5x", "lr2x") if B[f"P135_{ds}_{lr}_(2,0.5)"]["linear"].get("n"))
    L.append(f"- **Same contrast at other learning rates (seed 0):** {'; '.join(lr_bits)} (lr 1e-3 seed 0: c10 {base0['c10']:+.2f}, c100 {base0['c100']:+.2f}) — "
             + ("the sign of VCS − JS matches lr 1e-3 in every cell." if same else "the sign of VCS − JS differs from lr 1e-3 in at least one cell."))
    for ds, v in S.items():
        L.append(f"- **Selected-vs-selected {ds}** (VCS {v['vcs_cell']} vs JS {v['js_cell']}; each loss with its own rule-selected scorer — not a single-factor contrast): {fmt_pair(v)}.")
    for k in ("VCS_cifar10_(a,kappa)", "VCS_cifar100_(a,kappa)", "JS_cifar10_(a,kappa)", "JS_cifar100_(a,kappa)", "VCS_cifar100_lr", "JS_cifar100_lr"):
        v = G.get(k, {})
        if "incomplete" in v:
            L.append(f"- Grid {k}: incomplete ({', '.join(v['incomplete'])})."); continue
        if not v:
            continue
        ht = v["highest_tested"]; rs = v["rule_selected"]
        cell = ht.get("cell", ht.get("lr")); rcell = rs.get("cell", rs.get("lr"))
        L.append(f"- Grid {k}: default {v.get('default_(2,0.5)', v.get('default_1e-3'))['linear']:.2f}; highest tested {cell} {ht['linear']:.2f} ({ht['gain_vs_default']:+.2f}); "
                 f"rule-selected {rcell}; full range {v['full_range_linear'][0]:.2f}–{v['full_range_linear'][1]:.2f}.")
    L += ["", "## 2. Task map (P129, P130, P133, P135, P136, P137)", ""]
    for u, m in lm.items():
        L.append(f"**{u}** — completed {len(m['completed'])}; running / resumable {len(m['running_or_resumable'])}; in queue {len(m['queue'])}.")
        for x in m["running_or_resumable"]:
            L.append(f"  - running/resumable: {x}")
        for x in m["queue"]:
            L.append(f"  - queue: {x}")
    L += ["", "## 3. Infrastructure incidents (kept separate from algorithm results)", ""]
    cm = sorted([dict(run_id=r, **b) for r, bs in logs.items() for b in bs if b["cuda_missing"]], key=lambda x: x["end"] or "")
    hosts = sorted({c["host"] for c in cm if c["host"]})
    L.append(f"- **CUDA device not visible** (node hand-out without a usable GPU): {len(cm)} job logs on host(s) {', '.join(hosts) or '—'}, "
             f"ends {cm[0]['end'] if cm else '—'} … {cm[-1]['end'] if cm else '—'}; all jobs exited before creating a run directory and were resubmitted with "
             "node52 excluded (default exclusion now node51, node52, node60).")
    L.append("- **Throttled GPUs** (nvidia-smi throttle reason 0x88, ~255 W vs ~450 W): node60 GPU 0 and 1, node61 GPU 1 — ≈ 44 s / epoch vs 20 s on healthy RTX6000PRO "
             "(ResNet-18, 4 views, B 256); node60 excluded for new starts.")
    mv = [e for e in moves if e.get("job")]
    L.append(f"- **GPU moves (requeue + epoch-boundary resume from last.pt):** {len(mv)} logged requeues: " + "; ".join(
        f"{e['job']} {e.get('run_id') or '?'} from {e['from_node']}" + (f":gpu{e['from_gpu']}" if e.get('from_gpu') is not None else "") for e in mv)
        + ".  Training results of moved runs are kept (resume restores model, optimizer, RNG at an epoch boundary); their wall time spans several GPU types.")
    L.append("- Mover incidents (2026-10-04): an all-node exclusion made SLURM flag pending jobs `BadConstraints` (reverted, no age loss); an unpinned rotation let a "
             "waiting unit restart on the vacated slow GPU; the automatic mover is off — manual pinned moves only.  RTX node58 / 59 / 61 maintenance drain from 2026-10-05 13:16 local.")
    fails = sorted(r["run_id"] for r in runs.values() if r["status"] == "FAILED")
    L.append(f"- **Algorithm / run failures (status FAILED):** {', '.join(fails) if fails else 'none'}.  Gate failures by design: P133 queue variants "
             "(constant-map J > 0) — not trained.")
    L += ["", "## 4. Cross-checks of numbers quoted in earlier reports (recomputed from raw files)", ""]
    for k, v in X.items():
        L.append(f"- {k}: n={v['n']}, linear mean {v['linear_mean']}, sample sd {v['linear_sd']}, max−min {v['linear_range']}; kNN mean {v['knn_mean']}")
    L += ["", "## 5. Table B detail (same configuration, VCS − JS)", "", "| contrast | linear | config keys differing besides the loss |", "|---|---|---|"]
    for k, v in B.items():
        L.append(f"| {k} | {fmt_pair(v)} | {', '.join(v['config_diff_besides_loss']) or 'none'} |")
    open(REP / "status_v7.md", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
