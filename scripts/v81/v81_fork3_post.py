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
def checkpoint_inference(S: dict, validate) -> tuple[list, dict]:
    rows = []; inv = {}
    for tag, e in S["entries"].items():
        for role in ("FIT_AUDIT", "CAL_SELECT"):
            r = e["roles"][role]; m = r["metrics"]; vs = r.get("vs_frozen_export") or {}
            rows.append({"tag": tag, "method": e["method"], "seed": e["seed"], "checkpoint_id": e["checkpoint_id"], "kind": e["kind"], "epoch": e.get("epoch"), "role": role, "n": r["n"],
                         "mse_prompt_macro": m["mean_mse_prompt_macro"], "rps_micro": m["rps_micro"], "ce_micro": m["ce_micro"], "pairwise_acc_macro": m["within_prompt_pairwise_acc_macro"],
                         "spearman_pooled": m["spearman_pooled"], "spearman_prompt_macro": m["spearman_prompt_macro"], "mu_mean": m["mu_mean"], "mu_std": m["mu_std"],
                         "within_prompt_mu_std": m["within_prompt_mu_std_mean"], "within_prompt_human_std": m["within_prompt_human_std_mean"], "between_prompt_mu_std": m["between_prompt_mu_std"],
                         "vs_export_max_abs_dmu": vs.get("max_abs_dmu"), "vs_export_mean_abs_dmu": vs.get("mean_abs_dmu"), "vs_export_applies": vs.get("applies"),
                         "best_epoch": e.get("best_epoch"), "resume_epoch": e.get("resume_epoch"), "best_equals_resume_epoch": e.get("best_equals_resume_epoch"),
                         "best_resume_max_abs_param_diff": e.get("best_resume_max_abs_param_diff"), "ckpt_sha256": e.get("sha256"), "ckpt_bytes": e.get("bytes"),
                         "rows_file": r["rows"], "rows_sha256": r["rows_sha256"]})
            if validate is not None:
                recs = [json.loads(l) for l in open(r["rows"])]; validate(recs, labelled=True)
                if hashlib.sha256(Path(r["rows"]).read_bytes()).hexdigest() != r["rows_sha256"]:
                    raise SystemExit(f"rows sha256 mismatch for {tag} {role}")
        if e["kind"] in ("best", "resume"):
            inv.setdefault(e["unit"].split("/")[-1], {}).update({e["kind"]: {"epoch": e["epoch"], "sha256": e["sha256"], "bytes": e["bytes"], "step": e.get("step"), "best_field": e.get("ckpt_best_field")},
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
def markdown(T: dict, CI: list, AT: dict, INV: dict, IC: list, GC: list, src: Path, env: dict, validated: bool) -> str:
    L = []; A = L.append
    A(f"# V81_05 — input / gradient checks and checkpoint-level inference (V8.1 fork 3, {time.strftime('%Y-%m-%d', time.gmtime())})\n")
    A("Facts only (FULL_PLAN_CN §04 D05 checkpoint level, D06; §03 §3–5; §07 §3).  Every number below is read from `results/{checkpoint_inference,input_checks,gradient_checks}.json`, "
      f"which `scripts/v81/v81_fork3_post.py` writes from the GPU summaries under `{src}`.  No V8 artefact was modified; no GenAI-Rank / GenAI-Bench / TIFA160 label or prediction was read; "
      "the only labels used are the FIT-AUDIT and CAL_SELECT development labels (package roles allowed for labelled diagnostics).  Seed policy: per seed, then the metric mean (no prediction ensemble).\n")
    A("## 0. Environment, standard-batch timing, inference budget\n")
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
        v = INV[name]; inv_rows.append([name, v["best"]["epoch"], v["resume"]["epoch"], v["log_selected_epoch"], "yes" if v["best_equals_resume_epoch"] else "no",
                                        g(v["best_resume_max_abs_param_diff"], 3) if v["best_equals_resume_epoch"] else "—", v["best"]["sha256"][:12], v["resume"]["sha256"][:12]])
    A(table(["unit", "best epoch", "resume epoch", "log selected", "best == resume", "max abs Δparam", "best sha256", "resume sha256"], inv_rows))
    A("\n## 2. Checkpoint-level inference (eval mode, FIT-AUDIT and CAL_SELECT)\n")
    A("Rows in the package schema per (unit, checkpoint, role) under `outputs_v81/fork3/infer/rows/` (paths + sha256 in `checkpoint_inference.csv`"
      + (", every file re-validated with the package reference validator in labelled mode" if validated else "") + ").  RECONSTRUCTED_INIT = V8's `build(seed)` with no training "
      "(BLIP-2 COCO weights, seeded head, FIT log-histogram bias; shared by the eight arms at a seed; not claimed bit-identical to the formal start).  "
      "The forward path caches the frozen EVA ViT-g output once per image; its agreement with the frozen CAL_SELECT export of each unit's `best.pt` is the `vs export` column (max |Δμ| over the 1,653 images).\n")
    for role in ("FIT_AUDIT", "CAL_SELECT"):
        A(f"### 2.{1 if role == 'FIT_AUDIT' else 2} {role} — seed means (per-seed values in `checkpoint_inference.csv`)\n")
        rows = []
        for method in ["ALL_ARMS"] + ARMS:
            for kind in ("init", "best", "resume"):
                a = AT.get(f"{role}|{method}|{kind}")
                if not a:
                    continue
                rows.append([method if kind != "init" else "RECONSTRUCTED_INIT (all arms)", kind, "/".join(str(e) for e in a["epochs"]), f(a["mse_prompt_macro"]["mean"], 4), f(a["pairwise_acc_macro"]["mean"]),
                             f(a["spearman_pooled"]["mean"]), f(a["spearman_prompt_macro"]["mean"]), f(a["rps_micro"]["mean"], 4), f(a["ce_micro"]["mean"], 4), f(a["within_prompt_mu_std"]["mean"]), f(a["mu_std"]["mean"]),
                             g(a["vs_export_max_abs_dmu"]["mean"], 2) if kind == "best" and role == "CAL_SELECT" else "—"])
        A(table(["method", "ckpt", "epochs (s0/s1/s2)", "mean-rating MSE (prompt macro)", "pairwise acc", "Spearman pooled", "Spearman prompt macro", "RPS", "CE", "within-prompt μ sd", "μ sd", "vs export max|Δμ|"], rows))
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
    cols = ["method", "seed", "normal ρ pooled", "normal pairwise", "within-prompt μ sd", "between-prompt μ sd", "perm mean|Δμ|", "perm p90|Δμ|", "perm frac|Δμ|<.05", "perm q L1", "ρ(text)", "ρ(image)",
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
      "`rank/point` = norm ratio.  μ saturation = fraction of μ within 0.05 of 1 or 5.\n")
    rows = []
    for r in sorted(GC, key=lambda r: (["U-RANK", "U-LIST", "U-CS"].index(r["arm"]), r["seed"], r["checkpoint"])):
        rows.append([r["arm"], r["seed"], r["checkpoint"] + ("" if r["epoch"] is None else f" (e{r['epoch']})"), g(r["head_point_norm"]), g(r["head_rank_norm"]), f(r["head_cos_point_rank"], 2), g(r["head_rank_over_point"], 2),
                     g(r["qformer_point_norm"]), g(r["qformer_rank_norm"]), f(r["qformer_cos_point_rank"], 2), g(r["qformer_rank_over_point"], 2), g(r["ln_vision_point_norm"]), g(r["query_tokens_point_norm"]),
                     f(r["point_loss"], 4), f(r["ranking_loss"], 4), f(r["ranking_pairs"], 1), f(r["ranking_subgroups"], 1), f(r["mu_saturated_frac"], 3), f(r["mu_within_batch_std"], 3), r["nonfinite_batches"], "yes" if r["plan_matches_unit_log"] else "NO"])
    A(table(["arm", "seed", "ckpt", "head ‖g_point‖", "head ‖g_rank‖", "head cos", "head rank/point", "Q-former ‖g_point‖", "Q-former ‖g_rank‖", "Q-former cos", "Q-former rank/point", "ln_vision ‖g_point‖",
             "query ‖g_point‖", "point loss", "ranking loss", "pairs / batch", "subgroups / batch", "μ saturated", "μ sd in batch", "non-finite batches", "plan hash ok"], rows))
    A("\n## 5. Not done / limits\n")
    A("- No per-epoch checkpoint exists for any unit (only `best.pt` and the final `resume.pt`), so the checkpoint-level selection analysis of D05 is limited to these two states per unit; "
      "no training was re-run to recover the trajectory (`not_saved`).\n- RECONSTRUCTED_INIT is a reconstruction from the same recipe and seed, labelled as such; bitwise identity with the formal units' start is not claimed.\n"
      "- The gradient checks run in eval mode without the GradScaler; the training step used train mode (Q-former dropout) and fp16 loss scaling.  The checks describe gradient magnitude and direction, not the realised update.\n"
      "- The within-prompt \"rotation\" condition is derived from the normal predictions (the same prompt with another of its own candidates), not a separate forward pass.\n"
      "- No interpretation is written here; the phase report (V81_08) reads these facts together with D00–D05 and D07.\n")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--src", default=str(SRC_DEFAULT)); ap.add_argument("--results", default=str(RESULTS_DEFAULT)); ap.add_argument("--md", default=str(MD_DEFAULT))
    ap.add_argument("--no-validate", action="store_true"); a = ap.parse_args(argv); src = Path(a.src); res = Path(a.results); res.mkdir(parents=True, exist_ok=True)
    core = None if a.no_validate else load_core(); validate = None if core is None else core.validate_rows
    T = json.loads((src / "timing" / "timing.json").read_text()) if (src / "timing" / "timing.json").exists() else None
    SI = json.loads((src / "infer" / "summary.json").read_text()); SX = json.loads((src / "interventions" / "summary.json").read_text()); SG = json.loads((src / "gradcheck" / "summary.json").read_text())
    env = SI.get("env", {}); CI, INV = checkpoint_inference(SI, validate); AT = arm_tables(CI); IC = input_checks(SX); GC = gradient_checks(SG)
    done = {s: (src / s / "done.json").exists() for s in ("infer", "interventions", "gradcheck")}
    out_ci = {"source": str(src), "env": env, "fit_audit": SI.get("fit_audit"), "stages_done": done, "timing": T, "inventory": INV, "rows": CI, "arm_tables": AT, "rows_validated_with_package_core": validate is not None}
    write_json = lambda p, o: p.write_text(json.dumps(o, indent=1, allow_nan=False))
    write_json(res / "checkpoint_inference.json", out_ci); write_csv(res / "checkpoint_inference.csv", CI, list(CI[0].keys()))
    write_json(res / "input_checks.json", {"source": str(src), "env": SX.get("env"), "fit_audit": SX.get("fit_audit"), "permutation_seed": SX.get("permutation_seed"), "grey_rgb": SX.get("grey_rgb"),
                                           "grad_rows": SX.get("grad_rows"), "rows": IC}); write_csv(res / "input_checks.csv", IC, list(IC[0].keys()))
    write_json(res / "gradient_checks.json", {"source": str(src), "env": SG.get("env"), "mode": SG.get("mode"), "n_batches": SG.get("n_batches"), "rows": GC}); write_csv(res / "gradient_checks.csv", GC, list(GC[0].keys()))
    md = markdown(T, CI, AT, INV, IC, GC, src, env, validate is not None); Path(a.md).parent.mkdir(parents=True, exist_ok=True); Path(a.md).write_text(md)
    print(f"[fork3 post] {len(CI)} inference rows, {len(IC)} intervention entries, {len(GC)} gradient-check entries; stages done {done}; -> {res}, {a.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
