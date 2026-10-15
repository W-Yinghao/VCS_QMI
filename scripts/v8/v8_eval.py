"""V8 evaluation step (V8-03 `evaluation:`): Tables A / B on GenAI-Rank with the FROZEN V7 scorer, the distribution readouts on CAL_PROB / DEV,
the lr / epoch choices, and the Table C file inventory — on the 24 formal units (8 arms x seeds 0-2).

    python -I scripts/v8/v8_eval.py dry-run [--out DIR]          # unit / export / external-file / label-file checks; NO label read
    python -I scripts/v8/v8_eval.py run --allow-scoring --out DIR  # the evaluation; refuses without the request file + a V8 ledger row

Frozen pieces reused unchanged: `scripts/v7/v7_score.py rank` (the GenAI-Rank scorer: official N = 9 pools, first-index ties, prompt-cluster
bootstrap 5,000 seed 2718, Holm per family; identical to the V7 worktree copy, sha256 checked), the V7X-50 v3 external predictions (B0 FGA,
B1 VQAScore, B4T TIFA, B4D DSG; sha256 against the frozen manifest), the sealed GenAI-Rank label file (sha256 against the seal report; never
printed).  Families (each one scorer call = one Holm family): primary U-CS vs {B0, B1, B4T, B4D} x 2 generators (8); same_conditions U-CS vs
{U-RPS, U-MSE, U-RANK, U-LIST} (8); attribution U-CS vs U-BCE (2); full_rows U-CS vs {U-CE, U-L1} (4, descriptive); subset_primary /
subset_same_conditions = the V7X-50 v2 common-defined-subset sensitivity (prompts with a flagged external failure removed; not pre-specified
in V8-03, reported as a sensitivity).  GenAI-Bench and TIFA160 labels are never touched.
Distribution readouts (natural prompt -> image weights w of the role; `labels` in V8-03): per-threshold natural Brier y_k (1 - u_k)^2 +
(1 - y_k) u_k^2, their mean (= the U-RPS risk, "RPS normalised by the 4 thresholds"), histogram log loss - sum_h hist_h log q_h, mean-rating
MSE / MAE prompt-macro (the CAL_SELECT selection unit) and natural-weighted; secondary: the same after a CAL_PROB temperature on the 5 logits.
Outputs are written to a new directory only (refuses to overwrite).  Every number of the report comes from v8_eval.json.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "v7"))
import v8_labels as LB  # noqa: E402

UNITS = Path("/projects/EEG-foundation-model/yinghao/outputs_v8/units")
LR_CHOICE_DIR = Path("/projects/EEG-foundation-model/yinghao/outputs_v8/lr_choice")
KERNELS = Path("/projects/EEG-foundation-model/yinghao/outputs_v8/kernels")
FIT_CSV = Path("/home/infres/yinwang/CS_QMI/outputs/V8/package_ref_v1/VCS_V8_Utility_Aligned_CS_Server_v1_20261014/configs/FIT_UNITS.csv")
DATA = Path("/home/infres/yinwang/CS_QMI/outputs/V7/data")
FREEZE = Path("/projects/EEG-foundation-model/yinghao/outputs_v7/x50_rank_freeze_v3")
MANIFEST = Path("/home/infres/yinwang/CS_QMI/ssl_pilot_v7/reports/V7/V7X_50_RANK_PREDICTION_FREEZE_v3.json")
SEAL_REPORT = DATA / "genai_rank_seal_report.json"
LABELS = Path("/projects/EEG-foundation-model/yinghao/data/genai_rank/closed/genai_rank_labels_sealed.jsonl")
SCORER = HERE.parent / "v7" / "v7_score.py"
SCORER_V7 = Path("/home/infres/yinwang/CS_QMI/ssl_pilot_v7/scripts/v7/v7_score.py")
REQUEST = ROOT / "reports" / "V8_UTILITY" / "V8_EVAL_SCORING_REQUEST.md"
LEDGER = ROOT / "reports" / "V8_UTILITY" / "LABEL_ACCESS_LEDGER.csv"
ARMS = ("U-CS", "U-BCE", "U-RPS", "U-CE", "U-MSE", "U-L1", "U-RANK", "U-LIST"); SEEDS = (0, 1, 2); GENS = ("DALLE_3", "SDXL_Base")
EXTERNAL = ("B0_FGA_released_O", "B1_VQAScore", "B4T_TIFA", "B4D_DSG")
FAMILIES = {"primary": ("U-CS", list(EXTERNAL)), "same_conditions": ("U-CS", ["U-RPS", "U-MSE", "U-RANK", "U-LIST"]),
            "attribution": ("U-CS", ["U-BCE"]), "full_rows": ("U-CS", ["U-CE", "U-L1"])}
ROLES = ("CAL_PROB", "DEV"); EPS = 1e-12


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_jsonl(p):
    with open(p) as fh:
        return [json.loads(l) for l in fh if l.strip()]


# ------------------------------------------------------------------------------------------------------------------------- unit inventory
def unit_dir(arm: str, seed: int, units: Path = UNITS) -> Path:
    return units / f"V8-FIT-{arm}-s{seed}"


def check_units(units: Path = UNITS, fit_csv: Path = FIT_CSV, need_bench: bool = True) -> dict:
    """Every formal unit of FIT_UNITS.csv needs done.json, pred/export.json, pred/genai_rank.jsonl (+ genai_bench.jsonl) and images_{CAL_PROB,DEV}."""
    rows = list(csv.DictReader(fit_csv.read_text(encoding="utf-8-sig").splitlines()))
    if len(rows) != len(ARMS) * len(SEEDS):
        raise SystemExit(f"FIT_UNITS.csv lists {len(rows)} units, expected {len(ARMS) * len(SEEDS)}")
    need = ["done.json", "pred/export.json", "pred/genai_rank.jsonl", "pred/images_CAL_PROB.jsonl", "pred/images_DEV.jsonl"] + (["pred/genai_bench.jsonl"] if need_bench else [])
    missing, present = {}, []
    for r in rows:
        u = units / r["unit_id"]; lack = [f for f in need if not (u / f).is_file()]
        (missing.__setitem__(r["unit_id"], lack) if lack else present.append(r["unit_id"]))
    return {"present": present, "missing": missing, "n_expected": len(rows)}


def require_complete(units: Path = UNITS, fit_csv: Path = FIT_CSV) -> list[str]:
    inv = check_units(units, fit_csv)
    if inv["missing"]:
        raise SystemExit("refusing: formal units incomplete — " + "; ".join(f"{k}: {v}" for k, v in inv["missing"].items()))
    return inv["present"]


# --------------------------------------------------------------------------------------------------------------------- external / labels
def check_external(freeze: Path = FREEZE, manifest: Path = MANIFEST) -> dict:
    man = json.loads(manifest.read_text()); methods = json.loads((freeze / "methods.json").read_text()); out = {"methods_json_sha256": sha(freeze / "methods.json")}
    if out["methods_json_sha256"] != man["methods_json"]["sha256"]:
        raise SystemExit("refusing: methods.json differs from the frozen V7X-50 v3 manifest")
    out["files"] = {}
    for name in EXTERNAL:
        files = methods[name]
        if len(files) != 1:
            raise SystemExit(f"{name}: expected one seedless file, got {len(files)}")
        rec = man["methods"][name]["files"][0]; s = sha(files[0])
        if files[0] != rec["file"] or s != rec["sha256"]:
            raise SystemExit(f"refusing: {name} file or sha256 differs from the frozen manifest")
        out["files"][name] = {"file": files[0], "sha256": s, "rows": sum(1 for _ in open(files[0]))}
    drop = {p for _, p in man["sensitivity_common_defined_subset"]["excluded_pools"]}
    out["subset_excluded_prompts"] = sorted(drop); out["subset_rule"] = man["sensitivity_common_defined_subset"]["rule"]
    return out


def check_labels(read_hash: bool) -> dict:
    seal = json.loads(SEAL_REPORT.read_text()); out = {"label_file": str(LABELS), "exists": LABELS.is_file(), "sealed_sha256": seal["sealed_sha256"], "prompts_sha256": seal["prompts_sha256"], "sealed_rows": seal["sealed_rows"]}
    if read_hash:
        out["label_file_sha256"] = sha(LABELS)
        if out["label_file_sha256"] != seal["sealed_sha256"]:
            raise SystemExit("refusing: the label file is not the sealed GenAI-Rank label file")
    return out


def check_scorer() -> dict:
    s = sha(SCORER)
    if SCORER_V7.is_file() and sha(SCORER_V7) != s:
        raise SystemExit("refusing: scripts/v7/v7_score.py differs from the V7 worktree copy (the frozen scorer)")
    return {"scorer": str(SCORER), "sha256": s, "same_as_v7_worktree": SCORER_V7.is_file()}


# ------------------------------------------------------------------------------------------------------------------- frozen scorer calls
def filter_jsonl(src, dst, drop_prompts: set):
    with open(src) as f, open(dst, "w") as g:
        for line in f:
            if str(json.loads(line)["prompt_id"]) not in drop_prompts:
                g.write(line)


def run_scorer(methods: dict, labels: Path, out_json: Path, ours: str, comps: list, workdir: Path, python: str | None = None) -> dict:
    """One call of the frozen scorer = one Holm family; the scorer refuses to overwrite and never prints a label."""
    mj = workdir / f"methods_{out_json.stem}.json"; mj.write_text(json.dumps(methods, indent=1))
    cmd = [python or sys.executable, "-I", str(SCORER), "rank", "--methods", str(mj), "--labels", str(labels), "--out", str(out_json), "--ours", ours, "--comparators", *comps, "--allow-scoring"]
    r = subprocess.run(cmd, capture_output=True, text=True); (workdir / f"{out_json.stem}.log").write_text(r.stdout + "\n" + r.stderr)
    if r.returncode != 0:
        raise SystemExit(f"frozen scorer failed for {out_json.name}: {r.stderr[-800:]}")
    return json.loads(out_json.read_text())


def v8_methods(present: list[str], units: Path = UNITS) -> dict:
    return {arm: [str(unit_dir(arm, s, units) / "pred" / "genai_rank.jsonl") for s in SEEDS] for arm in ARMS}


# ---------------------------------------------------------------------------------------------------------------- distribution readouts
def _softmax(z):
    z = np.asarray(z, float); z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)


def _tails(q):
    return np.flip(np.cumsum(np.flip(q, -1), -1), -1)[..., 1:]


def distribution_metrics(pred: list[dict], recs: dict, prompt_of: dict | None = None, use: str = "q") -> dict:
    """pred rows: image_id + (q[5] and u[4] and mu) or (u[4], mu) ; recs: image_id -> label record (hist, tail, mean, w, prompt_id).
    use = 'q' (histogram present) or 'u' (tails only: q derived as differences, log loss reported only if the tails are monotone)."""
    by = {}
    for p in pred:
        if p["image_id"] in by:
            raise ValueError(f"duplicate prediction {p['image_id']}")
        by[p["image_id"]] = p
    if set(by) != set(recs):
        raise ValueError(f"prediction / label universes differ: {len(by)} vs {len(recs)} images; no silent dropping")
    ids = sorted(recs); w = np.array([recs[i]["w"] for i in ids]); Y = np.array([recs[i]["tail"] for i in ids]); H = np.array([recs[i]["hist"] for i in ids])
    mean_h = np.array([recs[i]["mean"] for i in ids]); pid = [recs[i]["prompt_id"] for i in ids]
    U = np.array([by[i]["u"] for i in ids], float); mu = np.array([by[i]["mu"] for i in ids], float)
    if use == "q":
        Q = np.array([by[i]["q"] for i in ids], float)
    else:
        full = np.concatenate([np.ones((len(ids), 1)), U, np.zeros((len(ids), 1))], 1); Q = full[:, :-1] - full[:, 1:]
    brier_k = Y * (1 - U) ** 2 + (1 - Y) * U ** 2; wn = w / w.sum()
    monotone = bool(np.all(np.diff(U, axis=1) <= 1e-9)) and bool(np.all(Q >= -1e-9))
    logloss = float((wn * (-(H * np.log(np.clip(Q, EPS, 1))).sum(1)).sum())) if monotone else None
    per_p = defaultdict(list)
    for p, m, h in zip(pid, mu, mean_h):
        per_p[p].append((m, h))
    mse_macro = float(np.mean([np.mean([(a - b) ** 2 for a, b in v]) for v in per_p.values()])); mae_macro = float(np.mean([np.mean([abs(a - b) for a, b in v]) for v in per_p.values()]))
    return {"n_images": len(ids), "n_prompts": len(per_p), "brier_k_natural": [float(x) for x in (wn[:, None] * brier_k).sum(0)], "rps_natural": float((wn * brier_k.mean(1)).sum()),
            "hist_logloss_natural": logloss, "tails_monotone": monotone, "mean_mse_prompt_macro": mse_macro, "mean_mae_prompt_macro": mae_macro,
            "mean_mse_natural": float((wn * (mu - mean_h) ** 2).sum()), "mean_mae_natural": float((wn * np.abs(mu - mean_h)).sum())}


def fit_temperature(pred: list[dict], recs: dict, grid=np.exp(np.linspace(math.log(0.25), math.log(4.0), 121))) -> float:
    """CAL_PROB temperature on the 5 logits: the T on a fixed log grid minimising the natural-weighted histogram log loss (secondary only)."""
    by = {p["image_id"]: p for p in pred}; ids = sorted(recs); Z = np.array([by[i]["logits"] for i in ids], float); H = np.array([recs[i]["hist"] for i in ids]); w = np.array([recs[i]["w"] for i in ids]); w = w / w.sum()
    best = None
    for T in grid:
        ll = float((w * (-(H * np.log(np.clip(_softmax(Z / T), EPS, 1))).sum(1))).sum())
        if best is None or ll < best[1] - 1e-12:
            best = (float(T), ll)
    return best[0]


def apply_temperature(pred: list[dict], T: float, levels=np.arange(1, 6)) -> list[dict]:
    out = []
    for p in pred:
        q = _softmax(np.asarray(p["logits"], float) / T); u = _tails(q); out.append({"image_id": p["image_id"], "q": q.tolist(), "u": u.tolist(), "mu": float(1 + u.sum())})
    return out


def unit_distribution(unit: Path, R: dict) -> dict:
    res = {}; cal = read_jsonl(unit / "pred" / "images_CAL_PROB.jsonl"); T = fit_temperature(cal, R["CAL_PROB"])
    for role in ROLES:
        pred = read_jsonl(unit / "pred" / f"images_{role}.jsonl"); res[role] = {"raw": distribution_metrics(pred, R[role]), "calibrated_T_cal_prob": distribution_metrics(apply_temperature(pred, T), R[role])}
    res["temperature_cal_prob"] = T; return res


def label_records() -> dict:
    R = LB.records(LB.load(DATA)); return {role: {r["image_id"]: r for r in R[role]} for role in ROLES}


def table_c_inventory() -> dict:
    out = {}
    for d in sorted(KERNELS.glob("pca*/K-*")):
        fit = d / "fit.json"; out[f"{d.parent.name}/{d.name}"] = {"fit_json": str(fit) if fit.is_file() else None, "pred_files": sorted(p.name for p in d.glob("pred_*.jsonl")),
                                                                 "selected": (json.loads(fit.read_text()).get("selected") if fit.is_file() else None)}
    return out or {"status": "pending"}


def table_c_distribution(R: dict) -> dict:
    out = {}
    for d in sorted(KERNELS.glob("pca*/K-*")):
        key = f"{d.parent.name}/{d.name}"; out[key] = {}
        for role in ROLES:
            f = d / f"pred_{role}.jsonl"
            if not f.is_file():
                out[key][role] = None; continue
            rows = read_jsonl(f)
            out[key][role] = {col: distribution_metrics([{"image_id": r["image_id"], "u": r[f"u_{col}"], "mu": r["mu" if col == "raw" else "mu_pava"]} for r in rows], R[role], use="u") for col in ("raw", "pava")}
    return out


# ------------------------------------------------------------------------------------------------------------------------------ choices
def lr_choices() -> dict:
    files = sorted(LR_CHOICE_DIR.glob("lr_choice_*.json")); recs = [json.loads(f.read_text()) for f in files]
    if not recs:
        raise SystemExit("no lr_choice records")
    arms = {a: {"lr": r["arms"][a]["lr"], "cal_select_mse": r["arms"][a]["cal_select_mse"], "selected_epoch_tuning": r["arms"][a]["selected_epoch"], "lr_edge": r["arms"][a]["lr_edge"]} for r in recs[:1] for a in r["arms"]}
    for r in recs[1:]:
        for a in r["arms"]:
            if r["arms"][a]["lr"] != arms[a]["lr"]:
                raise SystemExit(f"lr_choice records disagree for {a}")
    return {"files": [str(f) for f in files], "arms": arms}


def unit_choices(present: list[str], units: Path = UNITS) -> dict:
    out = {}
    for u in present:
        d = json.loads((units / u / "done.json").read_text()); log = json.loads((units / u / "log.json").read_text()); ex = json.loads((units / u / "pred" / "export.json").read_text())
        out[u] = {"arm": log["arm"], "seed": log["seed"], "lr": log["lr"], "selected_epoch": d["selected_epoch"], "cal_select_mse": d["cal_select_mse"], "gpu": d.get("gpu"), "export_selected_epoch": ex["selected_epoch"], "seconds": log.get("seconds")}
    return out


# ----------------------------------------------------------------------------------------------------------------------------- markdown
def _fmt(c):
    return f"{c['delta']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] | {c['p_holm']:.3f} | " + " / ".join(f"{x:+.3f}" for x in c["per_seed_delta"])


def markdown(res: dict) -> str:
    L = [f"# V8 evaluation (V8-03 `evaluation:`) — generated from v8_eval.json ({res['utc']})", "",
         f"Units: {res['units']['n_expected']} formal (all present).  Scorer `{Path(res['scorer']['scorer']).name}` sha256 {res['scorer']['sha256'][:16]}; external files = V7X-50 v3 (methods.json {res['external']['methods_json_sha256'][:16]}).  Label file sha256 = seal ({res['labels']['sealed_sha256'][:16]}).", ""]
    for fam, (ours, comps) in FAMILIES.items():
        r = res["genai_rank"].get(fam)
        if not r:
            continue
        L += [f"## {fam}: {ours} vs {', '.join(comps)} (N = 9, official first-index ties; prompt-cluster bootstrap 5,000 seed 2718; Holm over {len(r['primary'])})", "",
              "| contrast | Δ selected human score [95 %] | p Holm | per seed |", "|---|---|---|---|"]
        for k, c in r["primary"].items():
            L.append(f"| {k} | {_fmt(c)} |")
        L += ["", "| method | generator | N1 selected | N3 selected | N9 selected | N9 regret |", "|---|---|---|---|---|---|"]
        for m, g in r["methods"].items():
            for gen in GENS:
                L.append(f"| {m} | {gen} | " + " | ".join(f"{np.mean(g[gen][f'N{n}']['selected']):.3f}" for n in (1, 3, 9)) + f" | {np.mean(g[gen]['N9']['regret']):.3f} |")
        L.append("")
    for fam in ("subset_primary", "subset_same_conditions"):
        r = res["genai_rank"].get(fam)
        if r:
            L += [f"## sensitivity {fam} (V7X-50 v2 common-defined subset, {len(res['external']['subset_excluded_prompts'])} prompts removed; not pre-specified in V8-03)", "", "| contrast | Δ [95 %] | p Holm | per seed |", "|---|---|---|---|"]
            L += [f"| {k} | {_fmt(c)} |" for k, c in r["primary"].items()] + [""]
    L += ["## lr rule and selected epochs", "", "| arm | lr (rule) | tuning CAL_SELECT MSE | lr edge | formal selected epochs s0/s1/s2 | formal CAL_SELECT MSE s0/s1/s2 |", "|---|---|---|---|---|---|"]
    for a in ARMS:
        c = res["lr_choice"]["arms"][a]; us = [res["unit_choices"][f"V8-FIT-{a}-s{s}"] for s in SEEDS]
        L.append(f"| {a} | {c['lr']} | {c['cal_select_mse']:.4f} | {c['lr_edge']} | " + "/".join(str(u["selected_epoch"]) for u in us) + " | " + "/".join(f"{u['cal_select_mse']:.4f}" for u in us) + " |")
    for role in ROLES:
        L += ["", f"## distribution readouts on {role} (natural weights; mean over 3 seeds, raw outputs | after the CAL_PROB temperature)", "",
              "| arm | Brier k=2/3/4/5 | RPS | hist log loss | mean MSE macro | mean MAE macro | mean MSE natural | RPS (cal) | log loss (cal) | MSE macro (cal) |", "|---|---|---|---|---|---|---|---|---|---|"]
        for a in ARMS:
            rs = [res["distribution"][f"V8-FIT-{a}-s{s}"][role] for s in SEEDS]; m = lambda key, col="raw": float(np.mean([r[col][key] for r in rs]))
            bk = np.mean([r["raw"]["brier_k_natural"] for r in rs], 0)
            L.append(f"| {a} | " + "/".join(f"{x:.4f}" for x in bk) + f" | {m('rps_natural'):.4f} | {m('hist_logloss_natural'):.4f} | {m('mean_mse_prompt_macro'):.4f} | {m('mean_mae_prompt_macro'):.4f} | {m('mean_mse_natural'):.4f} | {m('rps_natural', 'calibrated_T_cal_prob'):.4f} | {m('hist_logloss_natural', 'calibrated_T_cal_prob'):.4f} | {m('mean_mse_prompt_macro', 'calibrated_T_cal_prob'):.4f} |")
    if isinstance(res.get("table_c_distribution"), dict):
        L += ["", "## Table C kernel bridges (distribution readouts; raw | PAVA tails; log loss only where the tails are monotone)", "", "| route | role | Brier k=2/3/4/5 (raw) | RPS raw | RPS pava | log loss pava | mean MSE macro raw | mean MSE macro pava |", "|---|---|---|---|---|---|---|---|"]
        for key, roles in res["table_c_distribution"].items():
            for role, r in roles.items():
                if r:
                    ll = r["pava"]["hist_logloss_natural"]
                    L.append(f"| {key} | {role} | " + "/".join(f"{x:.4f}" for x in r["raw"]["brier_k_natural"]) + f" | {r['raw']['rps_natural']:.4f} | {r['pava']['rps_natural']:.4f} | {ll if ll is None else f'{ll:.4f}'} | {r['raw']['mean_mse_prompt_macro']:.4f} | {r['pava']['mean_mse_prompt_macro']:.4f} |")
    return "\n".join(L) + "\n"


# -------------------------------------------------------------------------------------------------------------------------------- main
def cmd_dry_run(a) -> int:
    inv = check_units(need_bench=not a.no_bench); ext = check_external(); lab = check_labels(read_hash=False); sc = check_scorer(); lr = lr_choices() if LR_CHOICE_DIR.is_dir() else {"arms": {}}
    rep = {"utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "units": inv, "external": {k: v for k, v in ext.items() if k != "subset_excluded_prompts"}, "n_subset_excluded_prompts": len(ext["subset_excluded_prompts"]),
           "labels": lab, "scorer": sc, "lr_choice_arms": {k: v["lr"] for k, v in lr["arms"].items()}, "request_file": REQUEST.is_file(), "ledger_rows": (len(list(csv.DictReader(open(LEDGER, encoding="utf-8-sig")))) if LEDGER.is_file() else 0),
           "table_c": table_c_inventory(), "ready": not inv["missing"] and lab["exists"] and REQUEST.is_file()}
    print(json.dumps(rep, indent=1)); print(f"[dry-run] present {len(inv['present'])} / {inv['n_expected']}; missing {len(inv['missing'])}; ready_for_run={rep['ready']}")
    return 0


def cmd_run(a) -> int:
    if not a.allow_scoring:
        raise SystemExit("refusing to read sealed labels without --allow-scoring")
    if not REQUEST.is_file():
        raise SystemExit(f"refusing: the scoring request {REQUEST} must exist (and be committed) before any label is read")
    rows = list(csv.DictReader(open(LEDGER, encoding="utf-8-sig"))) if LEDGER.is_file() else []
    if not any(r.get("dataset", "").startswith("GenAI-Rank") for r in rows):
        raise SystemExit("refusing: no GenAI-Rank row in the V8 label-access ledger")
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}; use a new versioned path")
    present = require_complete(); ext = check_external(); lab = check_labels(read_hash=True); sc = check_scorer()
    out.mkdir(parents=True); work = out / "work"; work.mkdir(); methods = json.loads((FREEZE / "methods.json").read_text()); v8m = v8_methods(present)
    res = {"utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "protocol": "reports/V8_UTILITY/V8_03_PROTOCOL_FROZEN.yaml evaluation:", "units": check_units(), "external": ext, "labels": lab, "scorer": sc,
           "v8_prediction_files": {k: [{"file": f, "sha256": sha(f)} for f in v] for k, v in v8m.items()}, "genai_rank": {}}
    for fam, (ours, comps) in FAMILIES.items():
        m = {ours: v8m[ours], **{c: (v8m[c] if c in v8m else methods[c]) for c in comps}}
        res["genai_rank"][fam] = run_scorer(m, LABELS, out / f"{fam}.json", ours, comps, work, a.python)
    drop = set(ext["subset_excluded_prompts"]); lab_s = work / "labels_subset.jsonl"; filter_jsonl(LABELS, lab_s, drop); lab_s.chmod(0o400)
    for fam, src in (("subset_primary", "primary"), ("subset_same_conditions", "same_conditions")):
        ours, comps = FAMILIES[src]; m = {}
        for k in [ours, *comps]:
            m[k] = []
            for f in (v8m[k] if k in v8m else methods[k]):
                dst = work / f"subset_{Path(f).parent.parent.name if k in v8m else ''}{Path(f).name}"; filter_jsonl(f, dst, drop); m[k].append(str(dst))
        res["genai_rank"][fam] = run_scorer(m, lab_s, out / f"{fam}.json", ours, comps, work, a.python)
    lab_s.unlink()
    R = label_records(); res["distribution"] = {u: unit_distribution(UNITS / u, R) for u in present}; res["lr_choice"] = lr_choices(); res["unit_choices"] = unit_choices(present)
    res["table_c"] = table_c_inventory(); res["table_c_distribution"] = table_c_distribution(R) if KERNELS.is_dir() else None
    (out / "v8_eval.json").write_text(json.dumps(res, indent=1)); (out / "v8_eval.md").write_text(markdown(res))
    print(f"-> {out}/v8_eval.md"); return 0


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dry-run"); d.add_argument("--out", default=None); d.add_argument("--no-bench", action="store_true", help="do not require genai_bench.jsonl in the inventory")
    r = sub.add_parser("run"); r.add_argument("--out", required=True); r.add_argument("--allow-scoring", action="store_true"); r.add_argument("--python", default=None, help="interpreter for the frozen scorer (default: this one)")
    a = ap.parse_args(argv)
    return cmd_dry_run(a) if a.cmd == "dry-run" else cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
