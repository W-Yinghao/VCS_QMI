#!/usr/bin/env python
"""V8.1 C1 — writes reports/V8_1_DIAGNOSTICS/V81_C1_REQUEST.yaml (package template templates/CONDITIONAL_REQUEST.yaml, filled) and
V81_C1_PROTOCOL_FROZEN_20261015.yaml from the committed result files and the smoke job; every number is read from those files.  Refuses to
overwrite an existing frozen protocol.  Both outputs must pass yaml.safe_load.

    python -I scripts/v81/c1_write_protocol.py --smoke-log /home/infres/yinwang/CS_QMI/slurm_logs/v81_c1_smoke_<job>.out
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
D = REPO / "reports" / "V8_1_DIAGNOSTICS"; R = D / "results"
ARMS, SEEDS = ["U-CS", "U-RPS", "U-MSE"], [0, 1, 2]
OWNER_LINE = "「这俩都可以提交」"
OWNER_CONTEXT = "owner reply 2026-10-15 ~20:55 UTC to 'V8.1 C1: 9 units, ~6-8 GPU-h; code, smoke and measured budget first' (main reports/OWNER_DECISIONS_20261015.md, commit df4ced4)"


def jl(p):
    return json.loads(Path(p).read_text())


def smoke_steps(log: Path) -> dict:
    txt = log.read_text(errors="replace"); head = txt.splitlines()[0]
    steps = {m.group(1): {"rc": int(m.group(2)), "seconds": int(m.group(3))} for m in re.finditer(r"^=== (\w+) rc=(\d+) (\d+) s", txt, re.M)}
    ex = re.search(r"^exit=(\d+) end=(\S+)", txt, re.M)
    return {"job": re.search(r"_(\d+)\.out$", log.name).group(1), "host": re.search(r"host=(\S+)", head).group(1), "gpu": re.search(r"gpu=([^,]+)", head).group(1),
            "start": re.search(r"start=(\S+)", head).group(1), "end": ex.group(2) if ex else None, "exit": int(ex.group(1)) if ex else None, "steps": steps}


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke-log", required=True)
    ap.add_argument("--audit-log", default=None, help="log of the separate audit check when the smoke's own audit step did not run to completion")
    a = ap.parse_args(argv)
    proto = D / "V81_C1_PROTOCOL_FROZEN_20261015.yaml"
    if proto.exists():
        raise SystemExit(f"{proto} exists — a frozen protocol is never overwritten (append an execution note instead)")
    S = jl(R / "c1_selftest.json"); INV = jl(R / "c1_param_inventory.json"); LC = jl(R / "label_ceiling.json"); D5 = jl(R / "d05_log_level.json")["summary"]
    CI = jl(R / "checkpoint_inference.json"); SM = smoke_steps(Path(a.smoke_log)); commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    tv8, tc1 = S["timing_same_gpu"]["V8"], S["timing_same_gpu"]["C1"]; spe = S["steps_per_epoch"]
    v8logs = {(arm, s): jl(Path("/projects/EEG-foundation-model/yinghao/outputs_v8/units") / f"V8-FIT-{arm}-s{s}" / "log.json") for arm in ARMS for s in SEEDS}
    lr = {arm: sorted({v8logs[(arm, s)]["lr"] for s in SEEDS}) for arm in ARMS}
    if any(len(v) != 1 for v in lr.values()):
        raise SystemExit(f"parent lr not unique per arm: {lr}")
    # budget on the smoke GPU: 5 epochs x steps x measured C1 s/step + 5 CAL evals (V8 logs of the same GPU type when available, else the mean) + export + audit
    gpu = S["gpu"]; same = [e["eval_s"] for L in v8logs.values() if L["gpu"] == gpu for e in L["epochs_log"]]
    alll = [e["eval_s"] for L in v8logs.values() for e in L["epochs_log"]]; eval_s = sum(same) / len(same) if same else sum(alll) / len(alll)
    n_roles = 1653 + 1630 + 3257; export_s = SM["steps"]["export"]["seconds"] * n_roles / 1653; audit_s = SM["steps"]["audit"]["seconds"]; audit_src = "smoke " + SM["job"]
    if a.audit_log:
        from datetime import datetime
        al = Path(a.audit_log).read_text(); st = re.search(r"start=(\S+)", al).group(1); en = re.search(r"exit=0 end=(\S+)", al).group(1)
        audit_s = int((datetime.strptime(en, "%Y-%m-%dT%H:%M:%SZ") - datetime.strptime(st, "%Y-%m-%dT%H:%M:%SZ")).total_seconds()); audit_src = "audit check " + re.search(r"_(\d+)\.out$", a.audit_log).group(1)
    train_s = 5 * spe * tc1["s_per_step_mean"]; unit_h = (train_s + 5 * eval_s + export_s + audit_s + 3 * 120) / 3600
    v8_train_s = [sum(e["train_s"] for e in L["epochs_log"]) for L in v8logs.values() if L["gpu"] == gpu]
    budget = {"measured_gpu": gpu, "smoke_job": SM["job"], "selftest_job": SM["job"], "steps_per_epoch": spe, "c1_s_per_step": round(tc1["s_per_step_mean"], 4), "v8_s_per_step_same_gpu": round(tv8["s_per_step_mean"], 4),
              "c1_over_v8_step_time": round(tc1["s_per_step_mean"] / tv8["s_per_step_mean"], 3), "c1_peak_mem_gb": round(tc1["peak_mem_gb"], 1), "v8_peak_mem_gb": round(tv8["peak_mem_gb"], 1),
              "cal_eval_s_per_epoch": round(eval_s, 1), "cal_eval_source": "parent V8 units (U-CS / U-RPS / U-MSE) on the same GPU type" if same else "parent V8 units (U-CS on H100, U-RPS / U-MSE on L40S): none ran on this GPU type, so their mean (conservative)",
              "export_s_3_roles": round(export_s), "export_source": f"smoke export of CAL_SELECT ({SM['steps']['export']['seconds']} s, 1,653 images) scaled to 6,540 images",
              "audit_s": audit_s, "audit_source": audit_src, "model_builds_s": 360, "unit_hours_on_measured_gpu": round(unit_h, 2), "nine_units_gpu_hours": round(9 * unit_h, 1),
              "v8_unit_train_hours_same_gpu": [round(x / 3600, 2) for x in v8_train_s] or None,
              "note": "other GPU types scale with their step time; the L40S V8 units trained ~2x slower per step than the RTX PRO 6000 / H100 ones (V8 formal logs)"}
    ev = {"label_ceiling": {role: {"ceiling_gaussian_all_pairs": round(LC["roles"][role]["A_gaussian"]["ceiling_pair_accuracy"], 3),
                                   "ceiling_gaussian_both_3vote": round(LC["roles"][role]["A_gaussian_both3"]["ceiling_pair_accuracy"], 3),
                                   "estimator_bias_on_synthetic_3vote": round(LC["roles"][role]["A_validation_on_synthetic_3vote"]["bias_A_minus_oracle"], 3) if "A_validation_on_synthetic_3vote" in LC["roles"][role] else None,
                                   "model_pair_accuracy_all_pairs": {arm: round(LC["roles"][role]["model_pair_accuracy_arm_mean"][arm]["pair_accuracy_all"], 3) for arm in ARMS}}
                            for role in ("DEV", "CAL_SELECT", "CAL_PROB")},
          "cal_select_curves_seed_mean": {arm: {"pair_acc_by_epoch": [round(x, 4) for x in D5[f"FIT/{arm}"]["mean_pairacc_curve"]], "mse_by_epoch": [round(x, 4) for x in D5[f"FIT/{arm}"]["mean_mse_curve"]],
                                                "train_loss_by_epoch": [round(x, 4) for x in D5[f"FIT/{arm}"]["mean_train_loss_curve"]], "selected_epochs": D5[f"FIT/{arm}"]["selected_epochs"]} for arm in ARMS},
          "fit_audit_vs_cal_select_best_to_resume": {arm: {role: {k: [round(CI["arm_tables"][f"{role}|{arm}|best"][k]["mean"], 4), round(CI["arm_tables"][f"{role}|{arm}|resume"][k]["mean"], 4)]
                                                                  for k in ("mse_prompt_macro", "pairwise_acc_macro")} for role in ("FIT_AUDIT", "CAL_SELECT")} for arm in ARMS}}
    peak_drop = {arm: round(max(D5[f"FIT/{arm}"]["mean_pairacc_curve"]) - D5[f"FIT/{arm}"]["mean_pairacc_curve"][-1], 4) for arm in ARMS}
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    request = {
        "version": "V8.1", "approved": True, "candidate": "C1 partial Q-former (FULL_PLAN_CN.md 06 C1)",
        "owner_decision_reference": {"verbatim": OWNER_LINE, "context": OWNER_CONTEXT, "commit": "df4ced4"},
        "supporting_diagnostics": [
            "V81_02_DECISION_AGREEMENT.md / V81_03_SCALE_VS_SELECTION.md: the six scale arms make near-identical within-prompt decisions; the held-out within-prompt pair accuracy is ~0.70 for every arm",
            {"results/label_ceiling.json (development diagnostic, not pre-registered)": ev["label_ceiling"]},
            {"V81_05 / results/checkpoint_inference.json: FIT-AUDIT keeps improving from best to the final epoch while CAL_SELECT does not ([best, resume] seed means)": ev["fit_audit_vs_cal_select_best_to_resume"]},
            {"results/d05_log_level.json: CAL_SELECT curves and training loss by epoch (seed means)": ev["cal_select_curves_seed_mean"]},
            "V81_05 §5 (results/param_audit.json): the full-adaptation units moved 0.4-1.1 % (relative L2) from initialisation; AdamW applied 11,890-11,894 of 11,895 steps",
            "V81_05 §3: every trained arm responds to the image (cross-prompt permutation mean abs delta-mu 1.03-1.78 vs 0.14-0.26 at RECONSTRUCTED_INIT)"],
        "contradicting_evidence": [
            {"CAL_SELECT pair accuracy falls only slightly after its peak (seed-mean peak minus last epoch)": peak_drop},
            "C1 changes capacity (fewer trainable parameters), not only the adaptation scope; a difference is a fixed-recipe adaptation-scope effect, not an optimum over adaptation schemes",
            "DEV was already read by the V8.1 diagnostics D02-D04 (development exposure); the C1 DEV readout is a development comparison, not an independent test",
            "No per-epoch V8 checkpoint exists; the FIT-AUDIT best-vs-resume contrast is the only checkpoint-level trajectory evidence"],
        "one_factor_changed": "trainable set: last 2 Q-former BertLayer blocks (resolved from the real encoder ModuleList) + ln_vision + query tokens + head; every other Q-former parameter frozen and its modules in eval mode during training",
        "baseline_units_reused": [f"V8-FIT-{arm}-s{s}" for arm in ARMS for s in SEEDS],
        "new_units": [f"V81-C1-{arm}-s{s}" for arm in ARMS for s in SEEDS],
        "measured_gpu_type": gpu, "measured_unit_hours": budget["unit_hours_on_measured_gpu"], "total_training_and_export_hours": budget["nine_units_gpu_hours"], "budget_detail": budget,
        "selection_data": "CAL_SELECT", "new_test_label_access": False, "old_results_overwritten": False,
        "stop_rule": "if the pre-registered primary rule does not return 'improves', V8 is closed out: no distillation, residual, backbone or tuning rescue of this branch, no further C variant without a new owner decision",
        "written_utc": now, "code_commit": commit}
    trainable = INV["n_trainable_c1"]
    if "n_effective_trainable_c1" not in INV:
        raise SystemExit("c1_param_inventory.json lacks the selftest's dead-tensor fields (run the smoke selftest with --inventory first)")
    protocol = {
        "version": "V8.1-C1", "status": "FROZEN before any C1 formal unit was submitted", "frozen_utc": now, "code_commit": commit,
        "owner_decision": {"verbatim": OWNER_LINE, "context": OWNER_CONTEXT, "commit": "df4ced4"},
        "package": "v8_1/package_v81_20261015 FULL_PLAN_CN.md 06 C1; configs/CONDITIONAL_UNITS.csv (V81-C1-* x 9); request reports/V8_1_DIAGNOSTICS/V81_C1_REQUEST.yaml",
        "design": {
            "one_factor_changed": request["one_factor_changed"],
            "c1_blocks": {"encoder_layers": INV["n_encoder_layers"], "indices": INV["c1_block_indices"], "name_prefixes": INV["c1_block_name_prefixes"],
                          "have_cross_attention": INV["c1_blocks_have_cross_attention"]},
            "trainable_parameters": {"c1": trainable, "v8": INV["n_trainable_v8"], "by_group": {g: v for g, v in INV["by_group"].items()},
                                     "c1_effective_on_loss_path": INV["n_effective_trainable_c1"],
                                     "c1_dead_on_loss_path": INV["dead_on_loss_path_c1"], "v8_dead_on_loss_path": INV["dead_on_loss_path_v8"],
                                     "dead_check": INV["dead_check"] + "; the pooled readout uses only the 32 query outputs, so the last block's text-path FFN has no path to the loss (same in V8)"},
            "modes_during_training": "model.eval(); the two C1 blocks, ln_vision and the head .train(); verified on the real stack (results/c1_selftest.json modes_during_training)",
            "unchanged_from_v8": ["FGA-BLIP2 COCO construction (V7 build_model), frozen EVA ViT-g in eval mode, 364 full-frame input, raw prompt <= 128 tokens",
                                  "v8_train.build(seed, FIT histogram): seeded head weight, FIT log-histogram bias",
                                  "FIT role (26,177 images), v8_core.batch_manifest per (seed, epoch) — plan hashes logged and comparable with the V8 logs",
                                  "AdamW (0.9, 0.999), weight decay 0.05 on matrices (V8 rule applied to the C1 trainable set), 100-step warm-up from 1e-8 + cosine to 0 over 11,895 steps",
                                  "AMP fp16 forward + GradScaler, fp32 losses from v8_core (unchanged)", "5 epochs; CAL_SELECT prompt-macro mean-rating MSE selection, ties -> earlier epoch",
                                  "best.pt / resume.pt as V8; export = v8_train.cmd_export (best.pt, eval mode)"],
            "arms_and_lr": {arm: lr[arm][0] for arm in ARMS}, "lr_source": "each arm's formal V8 unit log.json (the frozen lr rule's choice); identical across seeds",
            "seeds": SEEDS, "units": request["new_units"], "paired_baselines": request["baseline_units_reused"],
            "code": "scripts/v81/c1_train.py = V8 cmd_train + the 5 substitutions V8_TO_C1 (tests/v81/test_c1.py asserts source equality); smoke " + SM["job"],
            "extra_artifact": "trainable tensors after every epoch (<unit>/epochs/epoch{e}.pt); never used for selection or for the primary readout"},
        "compute": {"partitions": "RTX6000PRO,H100,L40S", "exclude": "node51,node60", "nice": 300, "jobs": "3 grouped jobs, one per arm (seeds 0, 1, 2 in sequence), slurm/v81/c1_group.sbatch",
                    "outputs": "/projects/EEG-foundation-model/yinghao/outputs_v81/c1/units/V81-C1-<arm>-s<seed> (never outputs_v8)", "budget": budget},
        "labels": {"read": ["FIT (training)", "CAL_SELECT (epoch selection)", "DEV (primary readout)", "CAL_PROB (exported, descriptive)", "FIT-AUDIT (audit, descriptive)"],
                   "never_read": ["V8-05 GenAI-Bench confirmation labels", "GenAI-Rank", "TIFA160", "EvalMuse test"], "exports": ["CAL_SELECT", "CAL_PROB", "DEV"]},
        "primary_readout": {
            "data": "DEV (3,257 images, 307 prompts), predictions from each unit's best.pt export (images_DEV.jsonl, raw mu)",
            "contrast": "C1 unit minus the original full-adaptation V8-FIT unit of the same arm and seed; seed mean of the three per-seed differences (no prediction ensemble)",
            "metrics": {"a_pair_accuracy": "pooled within-prompt pair accuracy over all pairs whose exact human means differ (vote sums compared exactly), model ties 0.5",
                        "b_selected_human_mean": "per prompt the image with the highest raw mu (ties -> lowest candidate index; candidates in lexical image-id order, as V81_02), its human mean, averaged over prompts"},
            "uncertainty": "prompt_group cluster bootstrap (V7 near-duplicate component), 2,000 replicates, seed 2718, one resample per replicate shared by all arms / seeds / variants; percentile 95 % interval of the seed-mean difference",
            "decision": "improves = interval lower bound > 0 for at least 2 of the 3 arms on (a), or for at least 2 of the 3 arms on (b); otherwise 'does not improve'",
            "code": "scripts/v81/c1_analyze.py (self-check with V8 on both sides: every difference exactly 0, job 1042289)"},
        "secondary_readouts": ["DEV pooled Spearman difference (same bootstrap)", "prompt-macro mean-rating MSE and its decomposition (prompt bias^2 + within-prompt centred error)",
                               "RPS / CE", "CAL_SELECT curves per epoch (logs)", "FIT-AUDIT best vs resume (C1 audit vs V81_05 D05 rows)", "cost: s/step, epoch times, peak memory, GPU type"],
        "stop_rule": request["stop_rule"],
        "exposure": "DEV was read by V8.1 D02-D04; the C1 DEV readout is a development comparison.  The V8-05 confirmation prompts are not used to choose or evaluate C1 and do not become a C1 confirmation set.",
        "execution_notes": []}
    for name, obj in (("V81_C1_REQUEST.yaml", request), (proto.name, protocol)):
        txt = yaml.safe_dump(obj, sort_keys=False, allow_unicode=True, width=160); (D / name).write_text(txt); yaml.safe_load((D / name).read_text())
    print(json.dumps(budget, indent=1)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
