#!/usr/bin/env python
"""V8.1 fork 1 — D00 inventory, D01 epoch / source audit, D05 log-level curves, D07 external-context table.

Reads: the 48 V8 unit directories (log.json, done.json, pred/*.jsonl row counts, export.json, file sizes — no checkpoint is
deserialised), the 12 SLURM training logs, the trainer source at the four commits the jobs ran at (git show), the frozen
result files v8_eval.json / confirm.json / x40_dev_external.json / V7C-04 primary.json.  No label file is opened.
Writes reports/V8_1_DIAGNOSTICS/results/{inventory.csv, epoch_audit.csv, curves.csv, d05_log_level.json, selection_gap.md,
benchmark_context.csv, d07_context.json, fork1_facts.json}.
"""
from __future__ import annotations
import ast, csv, hashlib, json, math, re, subprocess, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

UNITS = Path("/projects/EEG-foundation-model/yinghao/outputs_v8/units")
SLOGS = Path("/home/infres/yinwang/CS_QMI/slurm_logs")
V8 = Path("/home/infres/yinwang/CS_QMI/ssl_pilot_v8"); V7 = Path("/home/infres/yinwang/CS_QMI/ssl_pilot_v7")
EVAL = Path("/projects/EEG-foundation-model/yinghao/outputs_v8/eval_v1/v8_eval.json")
CONFIRM = V8 / "reports/V8_UTILITY/V8_05_CONFIRMATION_results/confirm.json"
X40 = V7 / "reports/V7_CLOSEOUT/V7C_DEV_results/x40_dev_external.json"
V7C04 = V7 / "reports/V7_CLOSEOUT/V7C_04_results/primary.json"
SNAP_DIR = Path("/projects/EEG-foundation-model/yinghao/outputs_v81/source_snapshots")
OUT = Path(__file__).resolve().parents[2] / "reports/V8_1_DIAGNOSTICS/results"
JOBS = {"FIT": {"U-CS": 1040811, "U-BCE": 1040812, "U-RPS": 1040813, "U-CE": 1040814, "U-MSE": 1040815, "U-L1": 1040816, "U-RANK": 1040817, "U-LIST": 1040818},
        "TUNE": {"U-CS": 1040713, "U-BCE": 1040713, "U-RPS": 1040714, "U-CE": 1040714, "U-MSE": 1040715, "U-L1": 1040715, "U-RANK": 1040716, "U-LIST": 1040716}}
ARMS = ["U-CS", "U-BCE", "U-RPS", "U-CE", "U-MSE", "U-L1", "U-RANK", "U-LIST"]
FORMAL_COMMIT = "cc6299f"
ROLE_N = {"CAL_SELECT": 1653, "CAL_PROB": 1630, "DEV": 3257, "genai_rank": 14400, "genai_bench": 9600, "tifa160": 800}


def warmup_cosine(step, total, lr, warmup=100, start=1e-8):      # copied from scripts/v8/v8_core.py (blob 0542e843, identical at every job commit)
    if step < warmup:
        return start + (lr - start) * step / warmup
    return 0.5 * lr * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*a):
    return subprocess.run(["git", "-C", str(V8), *a], check=True, capture_output=True, text=True).stdout


def parse_utc(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


# ------------------------------------------------------------------------------------------------------------ SLURM logs
def parse_job_logs():
    jobs = {}
    for f in sorted(SLOGS.glob("v8_formal_*_10408*.out")) + sorted(SLOGS.glob("v8_tune_*_10407*.out")):
        jid = int(re.search(r"_(\d+)\.out$", f.name).group(1)); txt = f.read_text(errors="replace")
        h = re.search(r"^host=(\S+) gpu=(.+?), (\d+) MiB start=(\S+) commit=(\S+) code=(\S+)", txt, re.M)
        rec = {"log": str(f), "host": h.group(1), "gpu": h.group(2), "gpu_mib": int(h.group(3)), "job_start": h.group(4), "commit": h.group(5), "code": h.group(6),
               "exit": re.search(r"^exit=(\d+) end=(\S+)", txt, re.M).groups(), "units": {}}
        for m in re.finditer(r"^\[(?:formal|group)\] (FIT|TUNE) (\S+) lr (\S+) seed (\d+) -> (\S+) start (\S+)", txt, re.M):
            rec["units"][m.group(5)] = {"role": m.group(1), "arm": m.group(2), "lr": m.group(3), "seed": int(m.group(4)), "start": m.group(6)}
        for m in re.finditer(r"^\[(?:formal|group)\] (\S+) done \(rc (\d+)\) (\S+)", txt, re.M):
            rec["units"][m.group(1)].update({"rc": int(m.group(2)), "done": m.group(3)})
        # per-epoch printed CAL line and the step prints (every 200 steps) — cross-checks of log.json
        cur = None; ep_lines = defaultdict(list); step_prints = defaultdict(lambda: defaultdict(int))
        for line in txt.splitlines():
            m = re.match(r"^\[(?:formal|group)\] (?:FIT|TUNE) \S+ lr \S+ seed \d+ -> (\S+) start", line)
            if m: cur = m.group(1); continue
            m = re.match(r"^\[v8_train\] epoch (\d+): CAL_SELECT mean-rating MSE ([0-9.]+) \((\d+)s \+ (\d+)s\)", line)
            if m and cur: ep_lines[cur].append((int(m.group(1)), float(m.group(2)), int(m.group(3)), int(m.group(4)))); continue
            m = re.match(r"^\[v8_train\] \S+ s\d+ lr\S+ epoch (\d+) step (\d+) loss", line)
            if m and cur: step_prints[cur][int(m.group(1))] += 1
            if "already complete" in line and cur: rec["units"][cur]["already_complete_message"] = True
        for u in rec["units"]:
            rec["units"][u]["printed_epochs"] = ep_lines.get(u, []); rec["units"][u]["step_prints"] = dict(step_prints.get(u, {}))
        jobs[jid] = rec
    return jobs


# ------------------------------------------------------------------------------------------------------------ source audit
def cmd_train_text(src: str) -> tuple[str, int, int]:
    tree = ast.parse(src); f = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "cmd_train")
    lines = src.splitlines(); return "\n".join(lines[f.lineno - 1:f.end_lineno]), f.lineno, f.end_lineno


def source_audit(commits):
    out = {}
    for c in commits:
        src = git("show", f"{c}:scripts/v8/v8_train.py"); p = SNAP_DIR / f"v8_train_{c}.py"; p.write_text(src)
        blob = git("rev-parse", f"{c}:scripts/v8/v8_train.py").strip(); core = git("rev-parse", f"{c}:scripts/v8/v8_core.py").strip()
        labels = git("rev-parse", f"{c}:scripts/v8/v8_labels.py").strip()
        t, l0, l1 = cmd_train_text(src)
        names = [f for f in git("ls-tree", "--name-only", c, "scripts/v8/").split() if f.endswith(".py")]
        cat = b"".join(subprocess.run(["git", "-C", str(V8), "show", f"{c}:{f}"], check=True, capture_output=True).stdout for f in sorted(names))
        calls = []
        for n in ast.walk(ast.parse(src)):
            if isinstance(n, ast.FunctionDef) and n.name == "cmd_train":
                for m in ast.walk(n):
                    if isinstance(m, ast.Call) and ast.unparse(m.func) in {"scaler.step", "cal_metrics", "predict", "torch.save", "model.load_state_dict", "set_train", "os.replace"}:
                        calls.append((m.lineno, ast.unparse(m.func)))
        out[c] = {"trainer_blob": blob, "trainer_sha256": hashlib.sha256(src.encode()).hexdigest(), "core_blob": core, "labels_blob": labels,
                  "cmd_train_lines": [l0, l1], "cmd_train_sha256": hashlib.sha256(t.encode()).hexdigest(), "code_snapshot_hash_recomputed": hashlib.sha256(cat).hexdigest()[:16],
                  "commit_date": git("show", "-s", "--format=%cI", c).strip(), "cmd_train_calls_in_line_order": sorted(set(calls)), "snapshot_path": str(p)}
    return out


# ------------------------------------------------------------------------------------------------------------ units
def load_units(jobs):
    by_unit_job = {u: (jid, rec) for jid, rec in jobs.items() for u in rec["units"]}
    rows = []
    for d in sorted(UNITS.iterdir()):
        if not d.is_dir() or not (d / "log.json").exists(): continue
        L = json.loads((d / "log.json").read_text()); kind = "FIT" if d.name.startswith("V8-FIT") else "TUNE"
        jid, jrec = by_unit_job[d.name]; ju = jrec["units"][d.name]
        preds = {p.stem: sum(1 for _ in open(p)) for p in sorted((d / "pred").glob("*.jsonl"))}
        exp = json.loads((d / "pred/export.json").read_text()) if (d / "pred/export.json").exists() else {}
        texp = json.loads((d / "pred/tifa160_export.json").read_text()) if (d / "pred/tifa160_export.json").exists() else {}
        ep = L["epochs_log"]; sel = L["selected_epoch"]; e_sel = ep[sel]
        mse = [e["cal_select"]["mean_mse_prompt_macro"] for e in ep]; pa = [e["cal_select"]["within_prompt_pairwise_acc_macro"] for e in ep]
        rps = [e["cal_select"]["rps_micro"] for e in ep]; ce = [e["cal_select"]["ce_micro"] for e in ep]
        steps = [e["steps"] for e in ep]; seen = [e["images_seen"] for e in ep]
        start, done = parse_utc(ju["start"]), parse_utc(ju["done"])
        r = {"unit": d.name, "kind": kind, "arm": L["arm"], "loss": L["loss"], "lr": L["lr"], "seed": L["seed"], "job_id": jid, "host": jrec["host"], "gpu": L["gpu"],
             "source_commit": jrec["commit"], "code_snapshot_hash": jrec["code"], "epochs_planned": L["epochs"], "epochs_logged": len(ep), "selected_epoch": sel,
             "selected_cal_select_mse": L["selected_cal_select_mse"], "done_json": (d / "done.json").exists(), "done_json_selected_epoch": json.loads((d / "done.json").read_text())["selected_epoch"] if (d / "done.json").exists() else None,
             "export_json_selected_epoch": exp.get("selected_epoch"), "tifa160_export_epoch": texp.get("selected_epoch"),
             "best_pt_bytes": (d / "best.pt").stat().st_size if (d / "best.pt").exists() else 0, "resume_pt_bytes": (d / "resume.pt").stat().st_size if (d / "resume.pt").exists() else 0,
             "per_epoch_checkpoints": "not_saved (best.pt / resume.pt overwritten each epoch; source lines 211-215)",
             "pred_rows": preds, "pred_complete": all(preds.get(k if k.startswith("genai") or k == "tifa160" else f"images_{k}", -1) == v for k, v in ROLE_N.items() if kind == "FIT" or k not in ("genai_bench", "tifa160")),
             "steps_per_epoch": steps, "images_seen_per_epoch": seen, "n_fit": L["n_fit"], "total_steps_planned": L["total_steps"], "steps_logged_total": sum(steps),
             "images_complete_every_epoch": all(s == L["n_fit"] for s in seen), "batch_plans": len(L.get("batch_plan_sha256", [])), "smoke": L["smoke"], "resumed_at": L.get("resumed_at", []),
             "cal_mse": mse, "cal_pairacc": pa, "cal_rps": rps, "cal_ce": ce, "train_means": [e["train_means"] for e in ep], "train_s": [e["train_s"] for e in ep], "eval_s": [e["eval_s"] for e in ep],
             "peak_mem_gb": max(e["peak_mem_gb"] for e in ep), "train_seconds": L["seconds"], "unit_start": ju["start"], "unit_done": ju["done"], "unit_rc": ju["rc"],
             "unit_wall_s": (done - start).total_seconds(), "printed_epochs": ju["printed_epochs"], "step_prints": ju["step_prints"], "warmup": L["warmup"], "weight_decay": L["weight_decay"],
             "head_bias_init": L["model"].get("head_bias_init"), "n_trainable": L["model"].get("n_trainable_v8"), "log_sha256": sha(d / "log.json")}
        r["cum_steps_at_selected"] = sum(steps[:sel + 1]); r["cum_images_at_selected"] = sum(seen[:sel + 1])
        r["lr_at_last_step_of_selected_epoch"] = warmup_cosine(r["cum_steps_at_selected"] - 1, L["total_steps"], L["lr"], L["warmup"])
        r["lr_peak_reached_step"] = L["warmup"]
        # cross-checks between the printed log and log.json
        pe = ju["printed_epochs"]; r["printed_epochs_match"] = len(pe) == len(ep) and all(abs(pm - m) < 1e-5 and pi == i for (pi, pm, _, _), (i, m) in zip(pe, zip(range(len(ep)), mse)))
        r["step_prints_match"] = all(ju["step_prints"].get(i, 0) == math.ceil(s / 200) for i, s in enumerate(steps))
        rows.append(r)
    return rows


# ------------------------------------------------------------------------------------------------------------ D05 log level
def argbest(vals, lower=True):
    best = None
    for i, v in enumerate(vals):
        if best is None or (v < vals[best] if lower else v > vals[best]): best = i      # strict: ties keep the earlier epoch
    return best


def d05(rows):
    per_unit = []; per_arm = defaultdict(list)
    for r in rows:
        sel = r["selected_epoch"]; mse, pa, rps, ce = r["cal_mse"], r["cal_pairacc"], r["cal_rps"], r["cal_ce"]
        e_pa, e_rps, e_ce = argbest(pa, lower=False), argbest(rps), argbest(ce)
        loss = [t["loss"] for t in r["train_means"]]; point = [t.get("point") for t in r["train_means"]]; rank = [t.get("ranking") for t in r["train_means"]]
        u = {"unit": r["unit"], "kind": r["kind"], "arm": r["arm"], "lr": r["lr"], "seed": r["seed"], "epoch_by_mse_selected": sel, "epoch_by_pairacc": e_pa, "epoch_by_rps": e_rps, "epoch_by_ce": e_ce,
             "mse_curve": mse, "pairacc_curve": pa, "rps_curve": rps, "ce_curve": ce, "train_loss_curve": loss, "train_point_curve": point, "train_ranking_curve": rank,
             "mse_sel": mse[sel], "mse_e0": mse[0], "mse_last": mse[-1], "mse_min": min(mse), "mse_e0_minus_sel": mse[0] - mse[sel], "mse_last_minus_sel": mse[-1] - mse[sel],
             "pairacc_sel": pa[sel], "pairacc_best": pa[e_pa], "pairacc_gap_sel_minus_best": pa[sel] - pa[e_pa], "mse_at_pairacc_best_minus_sel": mse[e_pa] - mse[sel],
             "rps_sel": rps[sel], "rps_best": rps[e_rps], "rps_gap": rps[sel] - rps[e_rps], "mse_at_rps_best_minus_sel": mse[e_rps] - mse[sel],
             "train_loss_monotone_decreasing": all(b < a for a, b in zip(loss, loss[1:])), "train_loss_e0_to_last": [loss[0], loss[-1]],
             "cal_mse_min_at_epoch": argbest(mse), "cal_mse_rises_after_selected": (sel < len(mse) - 1) and (mse[-1] > mse[sel]),
             "ranking_pairs_per_step": r["train_means"][0].get("ranking_pairs"), "ranking_subgroups_per_step": r["train_means"][0].get("ranking_subgroups")}
        per_unit.append(u); per_arm[(r["kind"], r["arm"])].append(u)
    summ = {}
    for (kind, arm), us in sorted(per_arm.items()):
        n = len(us)
        summ[f"{kind}/{arm}"] = {"n_units": n, "selected_epochs": [u["epoch_by_mse_selected"] for u in us], "pairacc_epochs": [u["epoch_by_pairacc"] for u in us], "rps_epochs": [u["epoch_by_rps"] for u in us],
                                "n_pairacc_epoch_differs": sum(u["epoch_by_pairacc"] != u["epoch_by_mse_selected"] for u in us), "n_rps_epoch_differs": sum(u["epoch_by_rps"] != u["epoch_by_mse_selected"] for u in us),
                                "mean_pairacc_gap": sum(u["pairacc_gap_sel_minus_best"] for u in us) / n, "max_pairacc_gap": min(u["pairacc_gap_sel_minus_best"] for u in us),
                                "mean_mse_at_pairacc_best_minus_sel": sum(u["mse_at_pairacc_best_minus_sel"] for u in us) / n,
                                "mean_mse_curve": [sum(u["mse_curve"][e] for u in us) / n for e in range(5)], "mean_pairacc_curve": [sum(u["pairacc_curve"][e] for u in us) / n for e in range(5)],
                                "mean_rps_curve": [sum(u["rps_curve"][e] for u in us) / n for e in range(5)], "mean_train_loss_curve": [sum(u["train_loss_curve"][e] for u in us) / n for e in range(5)],
                                "mean_mse_e0_minus_sel": sum(u["mse_e0_minus_sel"] for u in us) / n, "mean_mse_last_minus_sel": sum(u["mse_last_minus_sel"] for u in us) / n,
                                "n_train_loss_monotone": sum(u["train_loss_monotone_decreasing"] for u in us), "n_cal_mse_rises_after_selected": sum(u["cal_mse_rises_after_selected"] for u in us)}
    allu = [u for u in per_unit]
    summ["ALL"] = {"n_units": len(allu), "n_pairacc_epoch_differs": sum(u["epoch_by_pairacc"] != u["epoch_by_mse_selected"] for u in allu), "n_rps_epoch_differs": sum(u["epoch_by_rps"] != u["epoch_by_mse_selected"] for u in allu),
                   "n_ce_epoch_differs": sum(u["epoch_by_ce"] != u["epoch_by_mse_selected"] for u in allu), "pairacc_gap_range": [min(u["pairacc_gap_sel_minus_best"] for u in allu), max(u["pairacc_gap_sel_minus_best"] for u in allu)],
                   "mse_cost_of_pairacc_choice_range": [min(u["mse_at_pairacc_best_minus_sel"] for u in allu), max(u["mse_at_pairacc_best_minus_sel"] for u in allu)],
                   "n_selected_epoch0": sum(u["epoch_by_mse_selected"] == 0 for u in allu), "n_selected_last": sum(u["epoch_by_mse_selected"] == 4 for u in allu),
                   "n_train_loss_monotone": sum(u["train_loss_monotone_decreasing"] for u in allu), "n_cal_mse_rises_after_selected": sum(u["cal_mse_rises_after_selected"] for u in allu),
                   "mse_e0_minus_sel_range": [min(u["mse_e0_minus_sel"] for u in allu), max(u["mse_e0_minus_sel"] for u in allu)], "mse_last_minus_sel_range": [min(u["mse_last_minus_sel"] for u in allu), max(u["mse_last_minus_sel"] for u in allu)]}
    return per_unit, summ


# ------------------------------------------------------------------------------------------------------------ D07 context
def d07(rows):
    E = json.loads(EVAL.read_text()); C = json.loads(CONFIRM.read_text()); X = json.loads(X40.read_text()); V = json.loads(V7C04.read_text())
    gr = E["genai_rank"]["primary"]["methods"]; same = E["genai_rank"]["same_conditions"]["methods"]; full = E["genai_rank"]["full_rows"]["methods"]; attr = E["genai_rank"]["attribution"]["methods"]
    def n9(m, gen):
        src = gr.get(m) or same.get(m) or full.get(m) or attr.get(m) or {}
        if gen not in src: return None, None
        s = src[gen]["N9"]["selected"]; g = src[gen]["N9"]["regret"]; return sum(s) / len(s), sum(g) / len(g)
    xo = {r["system"]: r for r in X["overall"]}
    def dev(sysname):
        r = xo.get(sysname); return (r["srcc"], r["pairwise_acc"]) if r else (None, None)
    # V8 cost from the job logs: training seconds (log.json) and export time (unit wall − training); exported images per unit
    cost = defaultdict(list)
    for r in rows:
        if r["kind"] != "FIT": continue
        n_exp = ROLE_N["CAL_SELECT"] + ROLE_N["CAL_PROB"] + ROLE_N["DEV"] + ROLE_N["genai_rank"] + ROLE_N["genai_bench"]
        cost[r["gpu"]].append({"train_min": r["train_seconds"] / 60, "export_min": (r["unit_wall_s"] - r["train_seconds"]) / 60, "export_img_per_s": n_exp / (r["unit_wall_s"] - r["train_seconds"]),
                               "cal_eval_img_per_s": ROLE_N["CAL_SELECT"] / (sum(r["eval_s"]) / len(r["eval_s"])), "peak_mem_gb": r["peak_mem_gb"]})
    cost_s = {g: {k: (min(x[k] for x in v), max(x[k] for x in v)) for k in v[0]} | {"n_units": len(v)} for g, v in cost.items()}
    v8_train_cost = "; ".join(f"{g}: train {c['train_min'][0]:.0f}-{c['train_min'][1]:.0f} min / unit (5 epochs, {ROLE_N['CAL_SELECT']}-image CAL eval each), export of 30,540 images {c['export_min'][0]:.0f}-{c['export_min'][1]:.0f} min ({c['export_img_per_s'][0]:.0f}-{c['export_img_per_s'][1]:.0f} img/s), peak {c['peak_mem_gb'][1]:.1f} GB" for g, c in cost_s.items())
    rows7 = []
    def add(system, host, trained_on, readout, route_status, n9src=None, confirm_key=None, devkey=None, cost_str="", note=""):
        d3, r3 = n9(n9src, "DALLE_3") if n9src else (None, None); sx, rx = n9(n9src, "SDXL_Base") if n9src else (None, None)
        lv = C["levels"].get(confirm_key, {}) if confirm_key else {}
        ds, dp = dev(devkey) if devkey else (None, None)
        rows7.append({"system": system, "host_backbone": host, "trained_on": trained_on, "readout": readout, "route_status": route_status,
                      "genai_rank_N9_selected_DALLE_3": d3, "genai_rank_N9_regret_DALLE_3": r3, "genai_rank_N9_selected_SDXL_Base": sx, "genai_rank_N9_regret_SDXL_Base": rx,
                      "genai_bench850_pooled_spearman": lv.get("spearman_pooled"), "genai_bench850_pairwise": lv.get("pairwise"),
                      "evalmuse_DEV_srcc_descriptive": ds, "evalmuse_DEV_pairwise_descriptive": dp, "cost_recorded": cost_str, "note": note})
    for a in ARMS:
        add(a, "FGA-BLIP2 construction (V7 build_model: BLIP-2 COCO-finetuned init, EVA ViT-g frozen in eval mode; Q-Former + ln_vision + query_tokens trained) + Linear(768->5) head",
            "EvalMuse FIT 26,177 images (overall votes), lr / epoch by CAL_SELECT", "mu = 1 + sum_k u_k from the 5-level histogram head (raw, no temperature)", "V8 formal, 3 seeds, frozen V8-03",
            n9src=a, confirm_key=a, devkey=None, cost_str=v8_train_cost, note="DEV SRCC / agreement for the V8 arms: see D02 / D03 (V81_02, V81_03), not recomputed here; GenAI-Rank N9 = mean over seeds of the per-seed selection (V8-04); GenAI-Bench = seed-mean mu (V8-05)")
    add("B0_FGA_released_O", "FGA-BLIP2 released checkpoint (same BLIP-2 host family)", "all of EvalMuse TRAIN (includes DEV -> DEV exposed)", "native overall score", "released model, frozen V7 freeze v3 predictions",
        n9src="B0_FGA_released_O", confirm_key="B0_FGA_released_O", devkey="B0_FGA_released_O (exposed)", cost_str="native eval in the V7 N0 harness; not timed separately", note="DEV row exposed (trained on DEV)")
    add("B1_VQAScore", "CLIP-FlanT5-XXL (t2v-metrics 3.0)", "none on EvalMuse (zero-shot VQA probability)", "P(yes) native score", "frozen V7 freeze v3 predictions",
        n9src="B1_VQAScore", confirm_key="B1_VQAScore", devkey="B1_VQAScore", cost_str="GenAI-Rank 14,400 images: 39 min wall clock on one L40S (V7C-03)", note="not a 1-5 rating; mean-MSE columns NA by design (D03)")
    add("B4T_TIFA", "LLaMA-2 question generation (released local generator) + UnifiedQA filter + mPLUG-large VQA", "none on EvalMuse", "fraction of questions answered correctly", "official route (V7C-03 fidelity: 91 % element precision, 78 % recall vs released GPT-3.5 questions)",
        n9src="B4T_TIFA", confirm_key="B4T_TIFA", devkey="B4T_TIFA", cost_str="GenAI-Rank 14,400 images: questions 3.5 h + VQA 2.3 h on L40S (V7C-03)", note="66 / 1,600 GenAI-Rank pools score 0 for every candidate (ties, first-index rule)")
    add("B4D_DSG", "DSG prompts with Qwen2.5-32B-Instruct (local adaptation) + mPLUG-large VQA", "none on EvalMuse", "DSG dependency-weighted question accuracy", "LOCAL ADAPTATION (not the released PaLM-2 graphs; V7C-03 fidelity 81-86 % tuple content)",
        n9src="B4D_DSG", confirm_key="B4D_DSG", devkey="B4D_DSG", cost_str="GenAI-Rank 14,400 images: generation ~2.5 GPU-h (H100 NVL / RTX) + VQA ~2.4 GPU-h (3 x L40S) (V7C-03)", note="exposed on TIFA160 through DSG's in-context examples (V8-05 §B)")
    # V7 M2_E (VCS element route) from V7C-04 for context
    m2 = V["methods"].get("M2_E", {})
    def v7n9(gen):
        s = m2[gen]["N9"]["selected"]; g = m2[gen]["N9"]["regret"]; return sum(s) / len(s), sum(g) / len(g)
    d3, r3 = v7n9("DALLE_3"); sx, rx = v7n9("SDXL_Base")
    rows7.append({"system": "V7 M2_E (VCS element route, context)", "host_backbone": "same FGA-BLIP2 construction, element-level heads", "trained_on": "EvalMuse FIT (element labels)", "readout": "E readout (element aggregation)", "route_status": "V7 closeout, first GenAI-Rank read (V7C-04)",
                  "genai_rank_N9_selected_DALLE_3": d3, "genai_rank_N9_regret_DALLE_3": r3, "genai_rank_N9_selected_SDXL_Base": sx, "genai_rank_N9_regret_SDXL_Base": rx, "genai_bench850_pooled_spearman": None, "genai_bench850_pairwise": None,
                  "evalmuse_DEV_srcc_descriptive": None, "evalmuse_DEV_pairwise_descriptive": None, "cost_recorded": "V7 training (not re-timed here)", "note": "not scored on GenAI-Bench 850 (V7 excluded by owner decision); DEV rows exist per seed in x40 (M2_semantic_vcs-s*_E/O), not merged here"})
    for sysname, host, note in [("B2_VQAScore_Qwen25VL7B", "VQAScore with Qwen2.5-VL-7B", "secondary external row"), ("B5_HPSv3", "HPSv3 preference model", "secondary external row"), ("B3_PNVQA_own", "PN-VQA", "secondary external row"),
                                ("CLIPScore_sanity", "CLIP similarity", "sanity row")]:
        rows7.append({"system": sysname, "host_backbone": host, "trained_on": "none on EvalMuse", "readout": "native score", "route_status": "present in V7 freeze v3; not scored in any frozen V8 family (not_scored)",
                      "genai_rank_N9_selected_DALLE_3": None, "genai_rank_N9_regret_DALLE_3": None, "genai_rank_N9_selected_SDXL_Base": None, "genai_rank_N9_regret_SDXL_Base": None, "genai_bench850_pooled_spearman": None, "genai_bench850_pairwise": None,
                      "evalmuse_DEV_srcc_descriptive": dev(sysname)[0], "evalmuse_DEV_pairwise_descriptive": dev(sysname)[1], "cost_recorded": "", "note": note})
    # Table C kernel routes (untrained representation) — separate column class
    tc = E.get("table_c_distribution", {})
    for k, v in tc.items():
        rows7.append({"system": f"Table C {k}", "host_backbone": "UNTRAINED BLIP-2 features (768-d query mean, V7X-20; PCA as sensitivity) + kernel bridge", "trained_on": "EvalMuse FIT (kernel fit only; no network update)", "readout": "natural tails from T (FIT priors)",
                      "route_status": "kernel route — representation differs from the fully fine-tuned arms; shown separately (D07)", "genai_rank_N9_selected_DALLE_3": None, "genai_rank_N9_regret_DALLE_3": None, "genai_rank_N9_selected_SDXL_Base": None, "genai_rank_N9_regret_SDXL_Base": None,
                      "genai_bench850_pooled_spearman": None, "genai_bench850_pairwise": None, "evalmuse_DEV_srcc_descriptive": None, "evalmuse_DEV_pairwise_descriptive": None, "cost_recorded": "CPU kernel fit (V8-04 Table C)", "note": "distribution readouts: " + json.dumps(v)[:300]})
    ctx = {"sources": {"v8_eval.json": {"path": str(EVAL), "sha256": sha(EVAL), "utc": E["utc"]}, "confirm.json": {"path": str(CONFIRM), "sha256": sha(CONFIRM), "utc": C["utc"]}, "x40_dev_external.json": {"path": str(X40), "sha256": sha(X40)}, "V7C-04 primary.json": {"path": str(V7C04), "sha256": sha(V7C04)}},
           "genai_rank_contrasts_U-CS": E["genai_rank"]["primary"]["primary"], "genai_bench_families": C["families"], "v8_cost_by_gpu": cost_s, "rows": rows7}
    return rows7, ctx


# ------------------------------------------------------------------------------------------------------------ main
def main():
    OUT.mkdir(parents=True, exist_ok=True); SNAP_DIR.mkdir(parents=True, exist_ok=True)
    jobs = parse_job_logs(); rows = load_units(jobs)
    commits = sorted({r["source_commit"] for r in rows}); src = source_audit(commits)
    ref = src[FORMAL_COMMIT]["cmd_train_sha256"]
    for r in rows:
        s = src[r["source_commit"]]; r["trainer_blob"] = s["trainer_blob"]; r["trainer_sha256"] = s["trainer_sha256"]; r["cmd_train_identical_to_formal_commit"] = s["cmd_train_sha256"] == ref
        r["code_snapshot_hash_reproduced"] = s["code_snapshot_hash_recomputed"] == r["code_snapshot_hash"]
    # inventory.csv
    inv_cols = ["unit", "kind", "arm", "lr", "seed", "job_id", "host", "gpu", "source_commit", "code_snapshot_hash", "code_snapshot_hash_reproduced", "epochs_planned", "epochs_logged", "selected_epoch", "selected_cal_select_mse",
                "done_json", "done_json_selected_epoch", "export_json_selected_epoch", "tifa160_export_epoch", "best_pt_bytes", "resume_pt_bytes", "per_epoch_checkpoints", "pred_complete", "smoke", "resumed_at", "unit_start", "unit_done", "unit_rc", "train_seconds", "unit_wall_s", "peak_mem_gb", "log_sha256"]
    with open(OUT / "inventory.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=inv_cols + ["pred_rows"]); w.writeheader()
        for r in rows: w.writerow({k: (json.dumps(r[k]) if isinstance(r[k], (list, dict)) else r[k]) for k in inv_cols} | {"pred_rows": json.dumps(r["pred_rows"])})
    # epoch_audit.csv (docs/03 §2 fields)
    ea_cols = ["unit", "kind", "arm", "seed", "lr", "job_id", "source_commit", "trainer_blob", "trainer_sha256", "code_snapshot_hash", "code_snapshot_hash_reproduced", "cmd_train_identical_to_formal_commit", "epoch_index", "completed_epoch_count",
               "planned_batches_per_epoch", "attempted_steps_selected_epoch", "steps_logged_total", "total_steps_planned", "successful_updates", "images_seen_selected_epoch", "n_fit", "images_complete_every_epoch", "smoke", "resume_history",
               "checkpoint_stage_best_pt", "checkpoint_stage_resume_pt", "init_preupdate_saved", "reconstructed_init", "cum_steps_at_selected", "cum_images_at_selected", "lr_at_last_step_of_selected_epoch", "printed_epochs_match", "step_prints_match"]
    with open(OUT / "epoch_audit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ea_cols); w.writeheader()
        for r in rows:
            sel = r["selected_epoch"]
            w.writerow({"unit": r["unit"], "kind": r["kind"], "arm": r["arm"], "seed": r["seed"], "lr": r["lr"], "job_id": r["job_id"], "source_commit": r["source_commit"], "trainer_blob": r["trainer_blob"], "trainer_sha256": r["trainer_sha256"],
                        "code_snapshot_hash": r["code_snapshot_hash"], "code_snapshot_hash_reproduced": r["code_snapshot_hash_reproduced"], "cmd_train_identical_to_formal_commit": r["cmd_train_identical_to_formal_commit"], "epoch_index": sel,
                        "completed_epoch_count": sel + 1, "planned_batches_per_epoch": "/".join(map(str, r["steps_per_epoch"])), "attempted_steps_selected_epoch": r["steps_per_epoch"][sel], "steps_logged_total": r["steps_logged_total"], "total_steps_planned": r["total_steps_planned"],
                        "successful_updates": "unknown (AMP GradScaler skip count not recorded; scaler state only inside resume.pt, not read here)", "images_seen_selected_epoch": r["images_seen_per_epoch"][sel], "n_fit": r["n_fit"], "images_complete_every_epoch": r["images_complete_every_epoch"],
                        "smoke": r["smoke"], "resume_history": json.dumps(r["resumed_at"]), "checkpoint_stage_best_pt": f"AFTER_EPOCH (epoch index {sel}; written after that epoch's full pass + CAL eval; export.json epoch {r['export_json_selected_epoch']})",
                        "checkpoint_stage_resume_pt": f"AFTER_EPOCH (last logged epoch {len(r['steps_per_epoch']) - 1}; inferred from the loop, file not deserialised here)", "init_preupdate_saved": False,
                        "reconstructed_init": "possible: build(seed, fit_hist) is deterministic given the upstream checkpoint bytes (head weight from torch.Generator seed*7919+5, bias = log FIT histogram); RECONSTRUCTED_INIT, not a bitwise copy of the run's state (D06)",
                        "cum_steps_at_selected": r["cum_steps_at_selected"], "cum_images_at_selected": r["cum_images_at_selected"], "lr_at_last_step_of_selected_epoch": f"{r['lr_at_last_step_of_selected_epoch']:.3e}", "printed_epochs_match": r["printed_epochs_match"], "step_prints_match": r["step_prints_match"]})
    # curves.csv
    with open(OUT / "curves.csv", "w", newline="") as f:
        cols = ["unit", "kind", "arm", "lr", "seed", "gpu", "epoch", "cum_steps_end", "steps", "images_seen", "lr_end_of_epoch", "train_loss", "train_point", "train_ranking", "ranking_pairs_per_step", "ranking_subgroups_per_step", "cal_mse_prompt_macro", "cal_pairacc_macro", "cal_rps_micro", "cal_ce_micro", "selected_by_mse", "train_s", "eval_s", "peak_mem_gb"]
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            cum = 0
            for e, (st, se, tm) in enumerate(zip(r["steps_per_epoch"], r["images_seen_per_epoch"], r["train_means"])):
                cum += st
                w.writerow({"unit": r["unit"], "kind": r["kind"], "arm": r["arm"], "lr": r["lr"], "seed": r["seed"], "gpu": r["gpu"], "epoch": e, "cum_steps_end": cum, "steps": st, "images_seen": se, "lr_end_of_epoch": f"{warmup_cosine(cum - 1, r['total_steps_planned'], r['lr'], r['warmup']):.3e}",
                            "train_loss": tm["loss"], "train_point": tm.get("point"), "train_ranking": tm.get("ranking"), "ranking_pairs_per_step": tm.get("ranking_pairs"), "ranking_subgroups_per_step": tm.get("ranking_subgroups"),
                            "cal_mse_prompt_macro": r["cal_mse"][e], "cal_pairacc_macro": r["cal_pairacc"][e], "cal_rps_micro": r["cal_rps"][e], "cal_ce_micro": r["cal_ce"][e], "selected_by_mse": e == r["selected_epoch"], "train_s": r["train_s"][e], "eval_s": r["eval_s"][e], "peak_mem_gb": r["peak_mem_gb"]})
    per_unit, summ = d05(rows)
    (OUT / "d05_log_level.json").write_text(json.dumps({"generated_utc": datetime.now(timezone.utc).strftime("%FT%TZ"), "rule": "log level only (docs/04 D05 level 1): selected = CAL_SELECT prompt-macro mean-rating MSE argmin (ties -> earlier); pairacc = within-prompt pairwise accuracy argmax (ties -> earlier); rps / ce argmin; curves from log.json epochs_log, 48 units",
                                                        "summary": summ, "units": per_unit}, indent=1))
    rows7, ctx = d07(rows)
    with open(OUT / "benchmark_context.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows7[0])); w.writeheader(); [w.writerow(r) for r in rows7]
    (OUT / "d07_context.json").write_text(json.dumps(ctx, indent=1))
    facts = {"generated_utc": datetime.now(timezone.utc).strftime("%FT%TZ"), "n_units": len(rows), "n_fit": sum(r["kind"] == "FIT" for r in rows), "n_tune": sum(r["kind"] == "TUNE" for r in rows),
             "all_done_json": all(r["done_json"] for r in rows), "all_pred_complete": all(r["pred_complete"] for r in rows), "all_images_complete": all(r["images_complete_every_epoch"] for r in rows),
             "all_steps_total_eq_planned": all(r["steps_logged_total"] == r["total_steps_planned"] for r in rows), "all_batch_plans_5": all(r["batch_plans"] == 5 for r in rows), "any_resume": [r["unit"] for r in rows if r["resumed_at"]],
             "any_smoke": [r["unit"] for r in rows if r["smoke"]], "selected_epoch_consistency": all(r["selected_epoch"] == r["export_json_selected_epoch"] == (r["done_json_selected_epoch"]) for r in rows),
             "tifa160_export_epoch_matches": all(r["tifa160_export_epoch"] == r["selected_epoch"] for r in rows if r["kind"] == "FIT"), "all_code_hash_reproduced": all(r["code_snapshot_hash_reproduced"] for r in rows),
             "all_cmd_train_identical": all(r["cmd_train_identical_to_formal_commit"] for r in rows), "all_printed_epochs_match": all(r["printed_epochs_match"] for r in rows), "all_step_prints_match": all(r["step_prints_match"] for r in rows),
             "steps_per_epoch_set": sorted({s for r in rows for s in r["steps_per_epoch"]}), "n_fit_set": sorted({r["n_fit"] for r in rows}), "commits": src, "jobs": {j: {k: v for k, v in rec.items() if k != "units"} | {"units": list(rec["units"])} for j, rec in jobs.items()},
             "head_bias_init_shared": len({json.dumps(r["head_bias_init"]) for r in rows}) == 1, "n_trainable_set": sorted({r["n_trainable"] for r in rows}), "peak_mem_gb_range": [min(r["peak_mem_gb"] for r in rows), max(r["peak_mem_gb"] for r in rows)],
             "selected_epochs": {r["unit"]: r["selected_epoch"] for r in rows}, "v8_cost_by_gpu": ctx["v8_cost_by_gpu"], "d05_summary": summ}
    (OUT / "fork1_facts.json").write_text(json.dumps(facts, indent=1, default=str))
    print(json.dumps({k: v for k, v in facts.items() if k not in ("commits", "jobs", "selected_epochs", "d05_summary", "v8_cost_by_gpu")}, indent=1))
    print("D05 ALL:", json.dumps(summ["ALL"], indent=1)); print("cost:", json.dumps(ctx["v8_cost_by_gpu"], indent=1))
    for k, v in summ.items():
        if k != "ALL": print(k, "sel", v["selected_epochs"], "pa", v["pairacc_epochs"], "rps", v["rps_epochs"], "gap %.4f" % v["mean_pairacc_gap"], "mse_cost %.4f" % v["mean_mse_at_pairacc_best_minus_sel"], "mse_curve", [round(x, 4) for x in v["mean_mse_curve"]], "pa_curve", [round(x, 4) for x in v["mean_pairacc_curve"]], "loss", [round(x, 4) for x in v["mean_train_loss_curve"]], "e0-sel %.4f last-sel %.4f" % (v["mean_mse_e0_minus_sel"], v["mean_mse_last_minus_sel"]))
    print("table_c keys sample:", list(json.loads(EVAL.read_text())["table_c_distribution"].items())[0][1] if json.loads(EVAL.read_text()).get("table_c_distribution") else None)


if __name__ == "__main__":
    main()
