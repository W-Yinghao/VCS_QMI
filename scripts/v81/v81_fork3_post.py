#!/usr/bin/env python
"""V8.1 fork 3 — CPU post-processing of the GPU diagnostics (outputs_v81/fork3/*/summary.json) into the delivery files.

Writes reports/V8_1_DIAGNOSTICS/results/{checkpoint_inference,input_checks,gradient_checks}.{json,csv} and
reports/V8_1_DIAGNOSTICS/V81_05_INPUT_GRADIENT_CHECKS.md (every number in the markdown is read from the JSON written here; facts only).
Also validates every row file of the checkpoint-level inference with the package reference validator (labelled mode).

    python -I scripts/v81/v81_fork3_post.py [--src /projects/EEG-foundation-model/yinghao/outputs_v81/fork3] [--results DIR] [--md FILE] [--no-validate]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC_DEFAULT = Path("/projects/EEG-foundation-model/yinghao/outputs_v81/fork3")
RESULTS_DEFAULT = REPO / "reports" / "V8_1_DIAGNOSTICS" / "results"
MD_DEFAULT = REPO / "reports" / "V8_1_DIAGNOSTICS" / "V81_05_INPUT_GRADIENT_CHECKS.md"
ARMS = ["U-CS", "U-BCE", "U-RPS", "U-CE", "U-MSE", "U-L1", "U-RANK", "U-LIST"]


def job_info(ids: list) -> list:
    """host / gpu / start / end / exit from the SLURM log header and footer of each job (read-only)."""
    import glob, re
    out = []
    for j in ids:
        fs = sorted(glob.glob(f"/home/infres/yinwang/CS_QMI/slurm_logs/*_{j}.out"))
        if not fs:
            out.append({"job": j, "log": None}); continue
        txt = Path(fs[-1]).read_text(errors="replace"); head = txt.splitlines()[0] if txt else ""
        get = lambda pat, t: (re.search(pat, t).group(1) if re.search(pat, t) else None)
        st, en = get(r"start=(\S+)", head), get(r"exit=\d+ end=(\S+)", txt)
        hours = None
        if st and en:
            from datetime import datetime
            hours = (datetime.strptime(en, "%Y-%m-%dT%H:%M:%SZ") - datetime.strptime(st, "%Y-%m-%dT%H:%M:%SZ")).total_seconds() / 3600
        out.append({"job": j, "log": fs[-1], "name": Path(fs[-1]).name.rsplit("_", 1)[0], "host": get(r"host=(\S+)", head), "gpu": get(r"gpu=([^,]+)", head),
                    "start": st, "end": en, "exit": get(r"exit=(\d+) end=", txt), "hours": hours})
    return out


def load_core():
    spec = importlib.util.spec_from_file_location("v81core", REPO / "v8_1" / "package_v81_20261015" / "reference" / "core.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def f(x, d=3):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def g(x, d=3):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}g}"


def mean_or_none(v):
    v = [x for x in v if x is not None and np.isfinite(x)]; return float(np.mean(v)) if v else None


def write_csv(p: Path, rows: list, cols: list):
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader(); [w.writerow(r) for r in rows]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def table(cols: list, rows: list) -> str:
    return "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + "".join("| " + " | ".join(str(c) for c in r) + " |\n" for r in rows)


# ------------------------------------------------------------------------------------------------------------------------ checkpoint inference
UNITS_V8 = Path("/projects/EEG-foundation-model/yinghao/outputs_v8/units")


def vs_log(e: dict) -> dict | None:
    """Re-inferred CAL_SELECT metrics minus the training-time log values of the same epoch (read-only on log.json)."""
    if e["kind"] not in ("best", "resume"):
        return None
    L = json.loads((UNITS_V8 / e["unit"].split("/")[-1] / "log.json").read_text()); ep = L["epochs_log"][e["epoch"]]
    if ep["epoch"] != e["epoch"]:
        raise SystemExit(f"log epoch order mismatch for {e['unit']}")
    m = e["roles"]["CAL_SELECT"]["metrics"]; c = ep["cal_select"]
    return {k: m[k] - c[k] for k in ("mean_mse_prompt_macro", "rps_micro", "ce_micro", "within_prompt_pairwise_acc_macro") if c.get(k) is not None and m.get(k) is not None}


def checkpoint_inference(S: dict, validate) -> tuple[list, dict]:
    rows = []; inv = {}
    for tag, e in S["entries"].items():
        dl = vs_log(e)
        for role in ("FIT_AUDIT", "CAL_SELECT"):
            r = e["roles"][role]; m = r["metrics"]; vs = r.get("vs_frozen_export") or {}
            rows.append({"tag": tag, "method": e["method"], "seed": e["seed"], "checkpoint_id": e["checkpoint_id"], "kind": e["kind"], "epoch": e.get("epoch"), "role": role, "n": r["n"],
                         "mse_prompt_macro": m["mean_mse_prompt_macro"], "rps_micro": m["rps_micro"], "ce_micro": m["ce_micro"], "pairwise_acc_macro": m["within_prompt_pairwise_acc_macro"],
                         "spearman_pooled": m["spearman_pooled"], "spearman_prompt_macro": m["spearman_prompt_macro"], "mu_mean": m["mu_mean"], "mu_std": m["mu_std"],
                         "within_prompt_mu_std": m["within_prompt_mu_std_mean"], "within_prompt_human_std": m["within_prompt_human_std_mean"], "between_prompt_mu_std": m["between_prompt_mu_std"],
                         "vs_export_max_abs_dmu": vs.get("max_abs_dmu"), "vs_export_mean_abs_dmu": vs.get("mean_abs_dmu"), "vs_export_applies": vs.get("applies"),
                         "vs_export_gpu": vs.get("export_gpu"), **({f"vs_log_d_{k}": v for k, v in dl.items()} if (dl and role == "CAL_SELECT") else {}),
                         "best_epoch": e.get("best_epoch"), "resume_epoch": e.get("resume_epoch"), "best_equals_resume_epoch": e.get("best_equals_resume_epoch"),
                         "best_resume_max_abs_param_diff": e.get("best_resume_max_abs_param_diff"), "ckpt_sha256": e.get("sha256"), "ckpt_bytes": e.get("bytes"),
                         "rows_file": r["rows"], "rows_sha256": r["rows_sha256"]})
            if validate is not None:
                recs = [json.loads(l) for l in open(r["rows"])]; validate(recs, labelled=True)
                if hashlib.sha256(Path(r["rows"]).read_bytes()).hexdigest() != r["rows_sha256"]:
                    raise SystemExit(f"rows sha256 mismatch for {tag} {role}")
        if e["kind"] in ("best", "resume"):
            inv.setdefault(e["unit"].split("/")[-1], {}).update({e["kind"]: {"epoch": e["epoch"], "sha256": e["sha256"], "bytes": e["bytes"], "step": e.get("step"), "best_field": e.get("ckpt_best_field"),
                                                                            "cal_select_vs_log": dl},
                                                              "log_selected_epoch": e["log_selected_epoch"], "best_equals_resume_epoch": e.get("best_equals_resume_epoch"),
                                                              "best_resume_max_abs_param_diff": e.get("best_resume_max_abs_param_diff")})
    return rows, inv


def arm_tables(rows: list) -> dict:
    """Per (role, method, kind): per-seed values and the seed mean of the headline metrics (seed policy: per seed, then the metric mean)."""
    keys = ("mse_prompt_macro", "pairwise_acc_macro", "spearman_pooled", "spearman_prompt_macro", "rps_micro", "ce_micro", "within_prompt_mu_std", "mu_std", "vs_export_max_abs_dmu")
    by = defaultdict(list)
    for r in rows:
        by[(r["role"], r["method"], r["kind"])].append(r)
    out = {}
    for (role, method, kind), rs in by.items():
        rs = sorted(rs, key=lambda r: r["seed"])
        out[f"{role}|{method}|{kind}"] = {"role": role, "method": method, "kind": kind, "seeds": [r["seed"] for r in rs], "epochs": [r["epoch"] for r in rs],
                                          **{k: {"per_seed": [r[k] for r in rs], "mean": mean_or_none([r[k] for r in rs])} for k in keys}}
    return out


# ------------------------------------------------------------------------------------------------------------------------------ interventions
def input_checks(S: dict) -> list:
    rows = []
    for tag, e in S["entries"].items():
        n, p, w, gr, gd = e["normal"], e["permuted_across_prompts"], e["within_prompt_rotation"], e["grey_image"], e["gradients"]
        rows.append({"tag": tag, "method": e["method"], "seed": e["seed"], "epoch": e.get("epoch"), "n_rows": e["n_rows"],
                     "normal_spearman_pooled": n["spearman_pooled"], "normal_spearman_prompt_macro": n["spearman_prompt_macro"], "normal_pairwise_acc_macro": n["within_prompt_pairwise_acc_macro"],
                     "normal_mse_prompt_macro": n["mean_mse_prompt_macro"], "normal_within_prompt_mu_std": n["within_prompt_mu_std_mean"], "normal_between_prompt_mu_std": n["between_prompt_mu_std"],
                     "normal_mu_std": n["mu_std"], "human_within_prompt_std": n["within_prompt_human_std_mean"],
                     "perm_abs_dmu_mean": p["abs_dmu_mean"], "perm_abs_dmu_median": p["abs_dmu_median"], "perm_abs_dmu_p90": p["abs_dmu_p90"], "perm_frac_abs_dmu_lt_0.05": p["frac_abs_dmu_lt_0.05"],
                     "perm_frac_abs_dmu_lt_0.2": p["frac_abs_dmu_lt_0.2"], "perm_q_l1_mean": p["q_l1_mean"], "perm_u_abs_mean": p["u_abs_mean"],
                     "perm_rho_text_following": p["spearman_permuted_vs_normal_same_row(text-following)"], "perm_rho_image_following": p["spearman_permuted_vs_normal_of_placed_image(image-following)"],
                     "perm_within_prompt_mu_std": p["within_prompt_mu_std_mean"],
                     "rot_spearman_dmu_vs_dhuman": w["spearman_dmu_vs_dhuman"], "rot_sign_agreement": w["sign_agreement_rate"], "rot_abs_dmu_mean": w["abs_dmu_mean"], "rot_n_pairs_human_differs": w["n_pairs_human_differs"],
                     "grey_mu_mean": gr["mu_mean"], "grey_mu_std": gr["mu_std"], "grey_dmu_vs_prompt_mean": gr["dmu_vs_prompt_mean_normal_mean"], "grey_frac_ge_prompt_max": gr["frac_grey_ge_prompt_max"],
                     "grey_frac_le_prompt_min": gr["frac_grey_le_prompt_min"], "grey_rho_vs_prompt_human_mean": gr["spearman_grey_mu_vs_prompt_human_mean"],
                     "grad_px_normal_mean": gd["normal"]["grad_pixels_l2"]["mean"], "grad_px_normal_median": gd["normal"]["grad_pixels_l2"]["median"],
                     "grad_px_permuted_mean": gd["permuted"]["grad_pixels_l2"]["mean"], "grad_vit_normal_mean": gd["normal"]["grad_vit_output_l2"]["mean"],
                     "grad_vit_normal_median": gd["normal"]["grad_vit_output_l2"]["median"], "grad_vit_permuted_mean": gd["permuted"]["grad_vit_output_l2"]["mean"],
                     "grad_vit_cls_share_normal": gd["normal"]["vit_output_cls_token_share"]["mean"], "grad_vit_max_patch_share_normal": gd["normal"]["vit_output_max_patch_share"]["mean"],
                     "grad_n_images": gd["normal"]["n"], "peak_mem_gb": e.get("peak_mem_gb"), "seconds": e.get("seconds"), "rows_file": e["rows"], "rows_sha256": e["rows_sha256"]})
    return rows


def gradient_checks(S: dict) -> list:
    rows = []
    for tag, e in S["entries"].items():
        r = {"tag": tag, "arm": e["arm"], "loss": e["loss"], "seed": e["seed"], "checkpoint": e["checkpoint"], "epoch": e.get("epoch"), "n_batches": e["n_batches"],
             "plan_matches_unit_log": e["plan"]["matches_unit_log"], "nonfinite_batches": e["nonfinite_batches"], "mu_saturated_frac": e["mu_saturated_frac_mean"], "mu_mean": e["mu_mean"],
             "mu_within_batch_std": e["mu_within_batch_std_mean"], "abs_T_mean": e["abs_T_mean"], "point_loss": e["point_loss_mean"], "ranking_loss": e["ranking_loss_mean"],
             "ranking_pairs": e["ranking_pairs_mean"], "ranking_subgroups": e["ranking_subgroups_mean"]}
        for grp in ("head", "qformer", "ln_vision", "query_tokens", "all_trainable"):
            s = e["summary"][grp]
            for stat in ("point_norm", "rank_norm", "cos_point_rank", "rank_over_point"):
                v = s.get(stat); r[f"{grp}_{stat}"] = None if v is None else v["mean"]; r[f"{grp}_{stat}_median"] = None if v is None else v["median"]
        rows.append(r)
    return rows


# ----------------------------------------------------------------------------------------------------------------------------------- markdown
def param_section(PA: dict) -> list:
    """Section 5 of the markdown from results/param_audit.json (facts only)."""
    L = []; A = L.append; units = PA["units"]
    A("## 5. Parameter-level audit (checkpoints vs RECONSTRUCTED_INIT; CPU job, read-only)\n")
    A("`results/param_audit.{json,csv}` (`scripts/v81/v81_fork3_param_audit.py`).  For each seed V8's `build(seed)` is rebuilt on CPU; the trainable set and the AdamW grouping are "
      "rebuilt with V8's own rule and checked against the optimiser state's tensor counts and shapes.  rel = ‖θ − θ_init‖₂ / ‖θ_init‖₂ over the group.  "
      "AdamW steps = the per-tensor `step` counter in `resume.pt`; global iterations = the trainer's batch counter; their difference = optimiser steps skipped by the fp16 GradScaler.  "
      f"{PA['note_vit']}.\n")
    gm = PA["init_meta"]; s0 = gm[sorted(gm)[0]]
    A(f"Trainable set rebuilt: {s0['n_trainable_tensors']} tensors / {s0['n_trainable_params']:,} parameters ({s0['n_decay_tensors']} with weight decay, {s0['n_no_decay_tensors']} without); "
      f"ViT parameters {s0['n_vit_params']:,}, any requires_grad: {s0['vit_requires_grad_any']}.  Trainable groups (tensor counts): "
      + ", ".join(f"{k} {v}" for k, v in PA["trainable_groups"].items() if not k.startswith("qf.L")) + f"; per-layer Q-former groups {sum(v for k, v in PA['trainable_groups'].items() if k.startswith('qf.L'))} tensors.\n")
    order = sorted(units, key=lambda u: (ARMS.index(units[u]["arm"]), units[u]["seed"]))
    al = PA.get("tied_aliases") or {}
    if al:
        eq = [all(units[u][k].get("tied_alias_equals_target", {}).values()) for u in order for k in ("best", "resume")]
        A("Tied state-dict aliases (one storage, two names; counted once, under the parameter): " + "; ".join(f"`{a}` = `{t}`" for a, t in al.items())
          + f" — alias equal to its parameter in {sum(eq)}/{len(eq)} checkpoints.\n")
    rows = []
    for u in order:
        r = units[u]; b = r["best"]["vs_init"]; o = r["resume"]["optimizer"]; fz = b.get("frozen_all", {"n_identical": 0, "n_tensors": 0, "max_abs_delta": 0.0})
        rv = lambda k: g(b[k]["rel_change"], 3) if k in b and b[k]["rel_change"] is not None else "—"
        rows.append([r["arm"], r["seed"], g(r["lr"], 2), r["best"]["epoch"], rv("trainable_all"), rv("head"), rv("ln_vision"), rv("query_tokens"), rv("qf.embeddings"), rv("qformer_self_attn_all"),
                     rv("qformer_cross_attn_all"), rv("qformer_ffn_query_all"), rv("qformer_ffn_text_all"), rv("qf.cls_head"), f"{fz['n_identical']}/{fz['n_tensors']}",
                     ", ".join(f"{k} {v}" for k, v in o["never_stepped_groups"].items()) or "0", f"{o['adamw_step_max']:.0f}/{o['global_iterations']}", o["skipped_by_gradscaler"],
                     g(o["gradscaler"].get("scale"), 4), g(r["resume_vs_best"]["trainable_all"]["rel_change"], 3) if r["resume_vs_best"]["trainable_all"]["rel_change"] else "0"])
    A(table(["arm", "seed", "lr", "best epoch", "rel all trainable", "rel head", "rel ln_vision", "rel query tokens", "rel Q-former embeddings", "rel self-attn", "rel cross-attn", "rel query FFN",
             "rel text FFN", "rel LM head", "frozen tensors identical", "never-stepped tensors (groups)", "AdamW steps / iterations", "skipped steps", "final loss scale", "resume vs best rel"], rows))
    layers = sorted({k.split(".")[1] for u in units for k in units[u]["best"]["vs_init"] if k.startswith("qf.L") and k.endswith("cross_attn")})
    if layers:
        A("\nCross-attention relative change per Q-former layer (best.pt vs RECONSTRUCTED_INIT; seed mean, per-seed values in the csv):\n")
        rows = []
        for arm in ARMS:
            us = [u for u in order if units[u]["arm"] == arm]
            if not us:
                continue
            rows.append([arm] + [g(mean_or_none([units[u]["best"]["vs_init"].get(f"qf.{l}.cross_attn", {}).get("rel_change") for u in us]), 3) for l in layers]
                        + [g(mean_or_none([units[u]["best"]["vs_init"].get(f"qf.{l}.self_attn", {}).get("rel_change") for u in us]), 3) for l in ("L00", "L11")])
        A(table(["arm"] + [f"{l} cross" for l in layers] + ["L00 self", "L11 self"], rows))
    ns = [(u, units[u]["resume"]["optimizer"]["never_stepped_groups"]) for u in order]
    groups_ns = sorted({k for _, d in ns for k in d})
    if groups_ns:
        ident = []
        for k in groups_ns:
            vals = [(units[u]["best"]["vs_init"].get(k, {}).get("n_identical"), units[u]["best"]["vs_init"].get(k, {}).get("n_tensors")) for u in order]
            ident.append(f"{k}: identical to the reconstruction in {sum(1 for a, b in vals if a is not None and a == b)}/{len(vals)} units")
        A("\nNever-stepped trainable tensors (no AdamW state: no gradient path in `forward_logits`) — " + "; ".join(ident) + ".\n")
    return L


def markdown(T: dict, CI: list, AT: dict, INV: dict, IC: list, GC: list, src: Path, env: dict, validated: bool, PA: dict | None = None, J: list | None = None) -> str:
    L = []; A = L.append
    A(f"# V81_05 — input / gradient checks and checkpoint-level inference (V8.1 fork 3, {time.strftime('%Y-%m-%d', time.gmtime())})\n")
    A("Facts only (FULL_PLAN_CN §04 D05 checkpoint level, D06; §03 §3–5; §07 §3).  Every number below is read from `results/{checkpoint_inference,input_checks,gradient_checks}.json`, "
      f"which `scripts/v81/v81_fork3_post.py` writes from the GPU summaries under `{src}`.  No V8 artefact was modified; no GenAI-Rank / GenAI-Bench / TIFA160 label or prediction was read; "
      "the only labels used are the FIT-AUDIT and CAL_SELECT development labels (package roles allowed for labelled diagnostics).  Seed policy: per seed, then the metric mean (no prediction ensemble).\n")
    A("## 0. Environment, standard-batch timing, inference budget\n")
    if J:
        A("GPU jobs: " + "; ".join(f"{j['job']} `{j.get('name')}` on {j.get('host')} ({j.get('gpu')}), {j.get('start')} → {j.get('end')}, exit {j.get('exit')}, {j['hours']:.2f} GPU-h" if j.get("hours") is not None else f"{j['job']} (log not found)" for j in J)
          + f" — total {sum(j['hours'] or 0 for j in J):.2f} GPU-h of the 6 GPU-h cap.\n")
    if T:
        fm, vo, qf = T["full_model"], T["vit_only"], T["qformer_head_only"]; est = T["estimates_s"]
        A(f"GPU `{T['env']['gpu']}` on {T['env']['host']}, torch {T['env']['torch']}, CUDA {T['env']['cuda']}, numpy {T['env']['numpy']}, python {T['env']['python']}; "
          f"job start {T['env']['utc_start']}.  Standard load: batch {T['batch']}, full-frame resize {T['resolution']} (BLIP eval processor), FIT-AUDIT prompts "
          f"(tokens mean {T['prompt_tokens']['mean']:.1f}, max {T['prompt_tokens']['max']}), {T['n_images_loaded']} images loaded; warm-up 3 batches, then {fm['n_timed_batches']} timed batches (CUDA-synchronised).\n")
        A(table(["path", "s / batch (mean ± sd)", "ms / image", "images / s", "peak GPU mem (GB)"],
                [[fm["label"], f"{fm['s_per_batch_mean']:.3f} ± {fm['s_per_batch_std']:.3f}", f"{fm['s_per_image'] * 1000:.1f}", f"{fm['images_per_s']:.1f}", f"{fm['peak_mem_gb']:.1f}"],
                 [vo["label"], f"{vo['s_per_batch_mean']:.3f} ± {vo['s_per_batch_std']:.3f}", f"{vo['s_per_image'] * 1000:.1f}", f"{vo['images_per_s']:.1f}", f"{vo['peak_mem_gb']:.1f}"],
                 [qf["label"], f"{qf['s_per_batch_mean']:.3f} ± {qf['s_per_batch_std']:.3f}", f"{qf['s_per_image'] * 1000:.1f}", f"{qf['images_per_s']:.1f}", f"{qf['peak_mem_gb']:.1f}"]]))
        A(f"\nCPU decode + preprocess, single process: {T['cpu_decode_preprocess_s_per_image_single_process'] * 1000:.1f} ms / image.  Derived full-model budget on this GPU: FIT-AUDIT {est['fit_audit']:.0f} s, "
          f"CAL_SELECT {est['cal_select']:.0f} s, a 15-candidate pool {est['candidate_pool_15']:.2f} s, a 7-candidate pool {est['candidate_pool_7']:.2f} s; one checkpoint on both roles "
          f"{est['one_checkpoint_both_roles_full']:.0f} s through the full model vs {est['one_checkpoint_both_roles_cached']:.0f} s through the cached-ViT path used below.\n")
    else:
        A("Timing stage not run (`timing/timing.json` absent).\n")
    A("## 1. Checkpoint inventory (what exists per unit)\n")
    A("Only `best.pt` (selected epoch) and the final `resume.pt` (epoch 4, with optimiser state) exist per unit; no per-epoch weights (§03 §3: other epochs are `not_saved`).  "
      "`best == resume` means the selected epoch is the last one, in which case the two files hold the same trainable weights (max |Δparam| shown).\n")
    inv_rows = []
    for name in sorted(INV, key=lambda n: (ARMS.index(n.split("-")[2]) if n.split("-")[2] in ARMS else 99, n)):
        v = INV[name]; bf = v["resume"].get("best_field") or [None, None]
        dlb, dlr = (v["best"].get("cal_select_vs_log") or {}), (v["resume"].get("cal_select_vs_log") or {})
        inv_rows.append([name, v["best"]["epoch"], v["resume"]["epoch"], v["log_selected_epoch"], f"e{bf[1]} / {f(bf[0], 4)}", "yes" if v["best_equals_resume_epoch"] else "no",
                         g(v["best_resume_max_abs_param_diff"], 3) if v["best_equals_resume_epoch"] else "—",
                         f"{dlb.get('mean_mse_prompt_macro', float('nan')):+.5f} / {dlr.get('mean_mse_prompt_macro', float('nan')):+.5f}", v["best"]["sha256"][:12], v["resume"]["sha256"][:12]])
    A(table(["unit", "best epoch", "resume epoch", "log selected", "resume.pt best field (epoch / MSE)", "best == resume", "max abs Δparam", "CAL_SELECT MSE re-inferred − log (best / resume)", "best sha256", "resume sha256"], inv_rows))
    dl_all = [abs(r[k]) for r in CI for k in r if k.startswith("vs_log_d_mean_mse") and r[k] is not None]
    dl_other = {k: max(abs(r[k]) for r in CI if r.get(k) is not None) for k in ("vs_log_d_rps_micro", "vs_log_d_ce_micro", "vs_log_d_within_prompt_pairwise_acc_macro") if any(r.get(k) is not None for r in CI)}
    if dl_all:
        A(f"\nRe-inferred CAL_SELECT metrics vs the training-time log of the same epoch over the {len(dl_all)} checkpoints: max abs Δ mean-rating MSE {max(dl_all):.5f}, "
          + ", ".join(f"{k.replace('vs_log_d_', '')} {v:.5f}" for k, v in dl_other.items()) + ".\n")
    A("\n## 2. Checkpoint-level inference (eval mode, FIT-AUDIT and CAL_SELECT)\n")
    A("Rows in the package schema per (unit, checkpoint, role) under `outputs_v81/fork3/infer/rows/` (paths + sha256 in `checkpoint_inference.csv`"
      + (", every file re-validated with the package reference validator in labelled mode" if validated else "") + ").  RECONSTRUCTED_INIT = V8's `build(seed)` with no training "
      "(BLIP-2 COCO weights, seeded head, FIT log-histogram bias; shared by the eight arms at a seed; not claimed bit-identical to the formal start).  "
      "The forward path caches the frozen EVA ViT-g output once per image; its agreement with the frozen CAL_SELECT export of each unit's `best.pt` is the `vs export` column (max |Δμ| over the 1,653 images).\n")
    ex = [r for r in CI if r["role"] == "CAL_SELECT" and r["kind"] == "best" and r.get("vs_export_max_abs_dmu") is not None]
    if ex:
        top = max(ex, key=lambda r: r["vs_export_max_abs_dmu"])
        A(f"Agreement with the frozen exports (best.pt, CAL_SELECT, {len(ex)} units, each exported on its training GPU; re-inference on {env.get('gpu')}): max abs Δμ {top['vs_export_max_abs_dmu']:.4f} "
          f"({top['tag'].split('__')[0]}, exported on {top.get('vs_export_gpu')}); per-unit mean abs Δμ {min(r['vs_export_mean_abs_dmu'] for r in ex):.5f}–{max(r['vs_export_mean_abs_dmu'] for r in ex):.5f}.\n")
    for role in ("FIT_AUDIT", "CAL_SELECT"):
        A(f"### 2.{1 if role == 'FIT_AUDIT' else 2} {role} — seed means (per-seed values in `checkpoint_inference.csv`)\n")
        rows = []
        for method in ["ALL_ARMS"] + ARMS:
            for kind in ("init", "best", "resume"):
                a = AT.get(f"{role}|{method}|{kind}")
                if not a:
                    continue
                rows.append([method if kind != "init" else "RECONSTRUCTED_INIT (all arms)", kind, "/".join("—" if e is None else str(e) for e in a["epochs"]), f(a["mse_prompt_macro"]["mean"], 4), f(a["pairwise_acc_macro"]["mean"]),
                             f(a["spearman_pooled"]["mean"]), f(a["spearman_prompt_macro"]["mean"]), f(a["rps_micro"]["mean"], 4), f(a["ce_micro"]["mean"], 4), f(a["within_prompt_mu_std"]["mean"]), f(a["mu_std"]["mean"]),
                             g(a["vs_export_max_abs_dmu"]["mean"], 2) if kind == "best" and role == "CAL_SELECT" else "—"])
        A(table(["method", "ckpt", "epochs (s0/s1/s2)", "mean-rating MSE (prompt macro)", "pairwise acc", "Spearman pooled", "Spearman prompt macro", "RPS", "CE", "within-prompt μ sd", "μ sd", "vs export max abs Δμ (seed mean of unit max)"], rows))
        A("")
    A("## 3. D06 input interventions on FIT-AUDIT (best.pt of the 24 units; RECONSTRUCTED_INIT per seed)\n")
    A("Conditions (same eval path, same preprocessing and tokenisation): **normal** pairs; **permuted** = every image moved to a different prompt by one fixed seeded derangement "
      "(`interventions/permutation.json`); **rotation** = the within-prompt candidate change derived from the normal predictions (each row vs the next lexical candidate of its own prompt; "
      "Δμ compared with the human-mean difference of the same two images; pairs with equal human means excluded); **grey** = one all-grey 364×364 image (RGB 128) paired with each of the 128 prompts.  "
      "Permuted predictions are never scored against the original image's human rating.  ρ(text) = Spearman of the permuted prediction with the normal prediction of the same row "
      "(same prompt, different image); ρ(image) = with the normal prediction of the image that was moved in (same image, different prompt).\n")
    def ic_rows(filter_fn):
        out = []
        for r in sorted(IC, key=lambda r: ((ARMS.index(r["method"]) if r["method"] in ARMS else -1), r["seed"])):
            if not filter_fn(r):
                continue
            out.append([r["method"], r["seed"], f(r["normal_spearman_pooled"]), f(r["normal_pairwise_acc_macro"]), f(r["normal_within_prompt_mu_std"]), f(r["normal_between_prompt_mu_std"]),
                        f(r["perm_abs_dmu_mean"]), f(r["perm_abs_dmu_p90"]), f(r["perm_frac_abs_dmu_lt_0.05"], 2), f(r["perm_q_l1_mean"]), f(r["perm_rho_text_following"], 2), f(r["perm_rho_image_following"], 2),
                        f(r["rot_spearman_dmu_vs_dhuman"], 2), f(r["rot_sign_agreement"], 2), f(r["grey_mu_mean"], 2), f(r["grey_dmu_vs_prompt_mean"], 2), f(r["grey_frac_ge_prompt_max"], 2)])
        return out
    cols = ["method", "seed", "normal ρ pooled", "normal pairwise", "within-prompt μ sd", "between-prompt μ sd", "perm mean abs Δμ", "perm p90 abs Δμ", "perm frac abs Δμ < .05", "perm q L1", "ρ(text)", "ρ(image)",
            "rot ρ(Δμ, Δhuman)", "rot sign agr.", "grey μ", "grey μ − prompt mean", "grey ≥ prompt max"]
    A("### 3.1 Per unit\n"); A(table(cols, ic_rows(lambda r: True)))
    byarm = defaultdict(list)
    for r in IC:
        byarm[r["method"]].append(r)
    keys = ["normal_spearman_pooled", "normal_pairwise_acc_macro", "normal_within_prompt_mu_std", "normal_between_prompt_mu_std", "perm_abs_dmu_mean", "perm_abs_dmu_p90", "perm_frac_abs_dmu_lt_0.05", "perm_q_l1_mean",
            "perm_rho_text_following", "perm_rho_image_following", "rot_spearman_dmu_vs_dhuman", "rot_sign_agreement", "grey_mu_mean", "grey_dmu_vs_prompt_mean", "grey_frac_ge_prompt_max"]
    A("\n### 3.2 Seed means per arm\n")
    rows = []
    for method in ["RECONSTRUCTED_INIT"] + ARMS:
        rs = byarm.get(method)
        if not rs:
            continue
        m = {k: mean_or_none([r[k] for r in rs]) for k in keys}
        rows.append([method, len(rs)] + [f(m[k], 2 if "rho" in k or "frac" in k or "sign" in k or "grey" in k else 3) for k in keys])
    A(table(["method", "n seeds"] + cols[2:], rows))
    A("\n### 3.3 Gradient response (eval-mode autograd, first 64 FIT-AUDIT rows; ‖∂μ/∂pixels‖₂ through the frozen ViT, ‖∂μ/∂(ViT output)‖₂ at the Q-former input; normal vs permuted pairing)\n")
    rows = []
    for r in sorted(IC, key=lambda r: ((ARMS.index(r["method"]) if r["method"] in ARMS else -1), r["seed"])):
        rows.append([r["method"], r["seed"], g(r["grad_px_normal_mean"]), g(r["grad_px_normal_median"]), g(r["grad_px_permuted_mean"]), g(r["grad_vit_normal_mean"]), g(r["grad_vit_normal_median"]),
                     g(r["grad_vit_permuted_mean"]), f(r["grad_vit_cls_share_normal"], 2), f(r["grad_vit_max_patch_share_normal"], 2), r["grad_n_images"]])
    A(table(["method", "seed", "px grad mean (normal)", "px grad median", "px grad mean (permuted)", "ViT-out grad mean (normal)", "ViT-out grad median", "ViT-out grad mean (permuted)", "CLS-token share", "max-patch share", "n"], rows))
    A("\n## 4. Gradient checks on the frozen epoch-0 batches (U-RANK / U-LIST, U-CS as reference)\n")
    A("First 20 batches of `v8_core.batch_manifest(FIT, seed, epoch 0)` (hash checked against each unit's `log.json`), eval mode (dropout off), autocast fp16 forward as in training, "
      "fp32 losses from the unchanged `v8_core.batch_loss`, unscaled autograd (no GradScaler, no optimiser step).  Point term = the normalised mean-rating MSE (for U-RANK / U-LIST) or the CS loss; "
      "ranking term = the λ-weighted RankNet / ListNet part (total − point).  Norms are L2 over the group's parameters; cos = cosine between the point and ranking gradients; "
      "‖g‖ and cos are means over the 20 batches; `rank/point` = the median over batches of the per-batch norm ratio.  μ saturation = fraction of μ within 0.05 of 1 or 5.\n")
    rows = []
    for r in sorted(GC, key=lambda r: (["U-RANK", "U-LIST", "U-CS"].index(r["arm"]), r["seed"], r["checkpoint"])):
        rows.append([r["arm"], r["seed"], r["checkpoint"] + ("" if r["epoch"] is None else f" (e{r['epoch']})"), g(r["head_point_norm"]), g(r["head_rank_norm"]), f(r["head_cos_point_rank"], 2), f(r["head_rank_over_point_median"], 2),
                     g(r["qformer_point_norm"]), g(r["qformer_rank_norm"]), f(r["qformer_cos_point_rank"], 2), f(r["qformer_rank_over_point_median"], 2), g(r["ln_vision_point_norm"]), g(r["query_tokens_point_norm"]),
                     f(r["point_loss"], 4), f(r["ranking_loss"], 4), f(r["ranking_pairs"], 1), f(r["ranking_subgroups"], 1), f(r["mu_saturated_frac"], 3), f(r["mu_within_batch_std"], 3), r["nonfinite_batches"], "yes" if r["plan_matches_unit_log"] else "NO"])
    A(table(["arm", "seed", "ckpt", "head ‖g_point‖", "head ‖g_rank‖", "head cos", "head rank/point (median)", "Q-former ‖g_point‖", "Q-former ‖g_rank‖", "Q-former cos", "Q-former rank/point (median)", "ln_vision ‖g_point‖",
             "query ‖g_point‖", "point loss", "ranking loss", "pairs / batch", "subgroups / batch", "μ saturated", "μ sd in batch", "non-finite batches", "plan hash ok"], rows))
    if PA:
        L.extend(param_section(PA))
    A("\n## 6. Not done / limits\n")
    lim = ["No per-epoch checkpoint exists for any unit (only `best.pt` and the final `resume.pt`), so the checkpoint-level selection analysis of D05 is limited to these two states per unit; "
           "no training was re-run to recover the trajectory (`not_saved`).",
           ("RECONSTRUCTED_INIT is a reconstruction from the same recipe and seed, labelled as such; §5 reports where it can be checked against the checkpoints (frozen and never-stepped tensors); "
            "the trained tensors' starting values cannot be checked." if PA else
            "RECONSTRUCTED_INIT is a reconstruction from the same recipe and seed, labelled as such; bitwise identity with the formal units' start is not claimed."),
           "The gradient checks run in eval mode without the GradScaler; the training step used train mode (Q-former dropout) and fp16 loss scaling.  They describe gradient magnitude and direction, "
           + ("not the realised update (§5 gives the realised parameter movement)." if PA else "not the realised update."),
           "The within-prompt \"rotation\" condition is derived from the normal predictions (the same prompt with another of its own candidates), not a separate forward pass.",
           "FIT-AUDIT is a training-set audit sample (every image was seen in every epoch); its metrics describe fit, not generalisation.",
           "No interpretation is written here; the phase report (V81_08) reads these facts together with D00–D05 and D07."]
    A("".join(f"- {x}\n" for x in lim))
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--src", default=str(SRC_DEFAULT)); ap.add_argument("--results", default=str(RESULTS_DEFAULT)); ap.add_argument("--md", default=str(MD_DEFAULT))
    ap.add_argument("--no-validate", action="store_true"); ap.add_argument("--jobs", default="1042190,1042191", help="GPU job ids whose log headers give host / GPU / wall time")
    a = ap.parse_args(argv); src = Path(a.src); res = Path(a.results); res.mkdir(parents=True, exist_ok=True)
    core = None if a.no_validate else load_core(); validate = None if core is None else core.validate_rows
    T = json.loads((src / "timing" / "timing.json").read_text()) if (src / "timing" / "timing.json").exists() else None
    SI = json.loads((src / "infer" / "summary.json").read_text()); SX = json.loads((src / "interventions" / "summary.json").read_text()); SG = json.loads((src / "gradcheck" / "summary.json").read_text())
    env = SI.get("env", {}); CI, INV = checkpoint_inference(SI, validate); AT = arm_tables(CI); IC = input_checks(SX); GC = gradient_checks(SG)
    done = {s: (src / s / "done.json").exists() for s in ("infer", "interventions", "gradcheck")}
    J = job_info([x for x in a.jobs.split(",") if x]); out_ci = {"source": str(src), "env": env, "jobs": J, "fit_audit": SI.get("fit_audit"), "stages_done": done, "timing": T, "inventory": INV, "rows": CI, "arm_tables": AT, "rows_validated_with_package_core": validate is not None}
    write_json = lambda p, o: p.write_text(json.dumps(o, indent=1, allow_nan=False))
    write_json(res / "checkpoint_inference.json", out_ci); write_csv(res / "checkpoint_inference.csv", CI, list(dict.fromkeys(k for r in CI for k in r)))
    write_json(res / "input_checks.json", {"source": str(src), "env": SX.get("env"), "fit_audit": SX.get("fit_audit"), "permutation_seed": SX.get("permutation_seed"), "grey_rgb": SX.get("grey_rgb"),
                                           "grad_rows": SX.get("grad_rows"), "rows": IC}); write_csv(res / "input_checks.csv", IC, list(IC[0].keys()))
    write_json(res / "gradient_checks.json", {"source": str(src), "env": SG.get("env"), "mode": SG.get("mode"), "n_batches": SG.get("n_batches"), "rows": GC}); write_csv(res / "gradient_checks.csv", GC, list(GC[0].keys()))
    pa_path = res / "param_audit.json"; PA = json.loads(pa_path.read_text()) if pa_path.exists() else None
    md = markdown(T, CI, AT, INV, IC, GC, src, env, validate is not None, PA, J); Path(a.md).parent.mkdir(parents=True, exist_ok=True); Path(a.md).write_text(md)
    print(f"[fork3 post] {len(CI)} inference rows, {len(IC)} intervention entries, {len(GC)} gradient-check entries; stages done {done}; -> {res}, {a.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
