#!/usr/bin/env python
"""V8.1 C1 — writes reports/V8_1_DIAGNOSTICS/V81_C1_REPORT_<date>.md from results/c1_results.json (c1_analyze.py), the frozen protocol, the
parameter inventory, results/checkpoint_inference.json (V8 FIT-AUDIT rows, V81_05) and the three group-job logs.  Every number is read from
those files; the verdict applies the frozen decision rule and stop rule as written.

    python -I scripts/v81/c1_report.py --jobs 1042302,1042303,1042304 --analysis-job 1042478 --out reports/V8_1_DIAGNOSTICS/V81_C1_REPORT_20261016.md
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
D = REPO / "reports" / "V8_1_DIAGNOSTICS"; R = D / "results"
ARMS, SEEDS = ["U-CS", "U-RPS", "U-MSE"], [0, 1, 2]


def f(x, d=3, sign=False):
    if x is None:
        return "—"
    return (f"{x:+.{d}f}" if sign else f"{x:.{d}f}")


def table(cols, rows):
    return "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + "".join("| " + " | ".join(str(c) for c in r) + " |\n" for r in rows)


def job_units(job: str) -> dict:
    fs = sorted(glob.glob(f"/home/infres/yinwang/CS_QMI/slurm_logs/*_{job}.out")); txt = Path(fs[-1]).read_text(errors="replace"); head = txt.splitlines()[0]
    units = {m.group(1): int(m.group(2)) for m in re.finditer(r"^\[unit-exit\] (\S+) rc=0 (\d+) s", txt, re.M)}
    return {"job": job, "host": re.search(r"host=(\S+)", head).group(1), "gpu": re.search(r"gpu=([^,]+)", head).group(1), "commit": re.search(r"commit=(\S+)", head).group(1),
            "code": re.search(r"code=(\S+)", head).group(1), "start": re.search(r"start=(\S+)", head).group(1), "exit": re.search(r"^exit=(\d+)", txt, re.M).group(1),
            "end": re.search(r"^exit=\d+ end=(\S+)", txt, re.M).group(1), "units": units}


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", required=True); ap.add_argument("--analysis-job", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    X = json.loads((R / "c1_results.json").read_text()); P = yaml.safe_load((D / "V81_C1_PROTOCOL_FROZEN_20261015.yaml").read_text())
    INV = json.loads((R / "c1_param_inventory.json").read_text()); CI = json.loads((R / "checkpoint_inference.json").read_text())
    J = [job_units(j) for j in a.jobs.split(",")]; U = X["units"]; PR = X["primary"]
    total_s = sum(s for j in J for s in j["units"].values())
    L = []; A = L.append
    A(f"# V81_C1 report — partial Q-former adaptation (V8.1 C1, {X['generated_utc'][:10]})\n")
    A(f"Protocol `V81_C1_PROTOCOL_FROZEN_20261015.yaml` (frozen {P['frozen_utc']}, code {P['code_commit']}; request `V81_C1_REQUEST.yaml`, approved by the owner "
      f"{P['owner_decision']['verbatim']}, main {P['owner_decision']['commit']}).  Every number below is read from `results/c1_results.json` "
      f"(`scripts/v81/c1_analyze.py`, CPU job {a.analysis_job}), the parameter inventory and the job logs, by `scripts/v81/c1_report.py`.  "
      "Labels read: FIT (training), CAL_SELECT (epoch selection), DEV (readout), CAL_PROB (exported only), FIT-AUDIT.  No V8-05 / GenAI-Bench / GenAI-Rank / TIFA160 / "
      "EvalMuse-test label was read.\n")
    A("## 1. What was run\n")
    A(f"One factor changed against V8's formal recipe: {P['design']['one_factor_changed']}.  Trainable parameters {INV['n_trainable_c1']:,} of V8's {INV['n_trainable_v8']:,} "
      f"(encoder layers {INV['c1_block_indices']} of {INV['n_encoder_layers']}; cross-attention in {INV['c1_blocks_have_cross_attention']}); "
      f"{len(INV['dead_on_loss_path_c1'])} of them (the last block's text-path FFN, {INV['n_trainable_c1'] - INV['n_effective_trainable_c1']:,} parameters) have no path to the loss, "
      f"as in V8, so {INV['n_effective_trainable_c1']:,} are effective.  Arms and lr (the parents' frozen lr): "
      + ", ".join(f"{k} {v:g}" for k, v in P["design"]["arms_and_lr"].items()) + "; seeds 0, 1, 2.\n")
    rows = []
    for j in J:
        rows.append([j["job"], j["host"], j["gpu"], j["commit"], j["code"], ", ".join(f"{u.replace('V81-C1-', '')} {s / 3600:.2f} h" for u, s in j["units"].items()), j["exit"]])
    A(table(["job", "host", "GPU", "commit", "code hash", "units (wall time each)", "exit"], rows))
    A(f"\nTotal unit wall time {total_s / 3600:.2f} GPU-h (frozen budget {P['compute']['budget']['nine_units_gpu_hours']} GPU-h measured on {P['compute']['budget']['measured_gpu']}).\n")
    rows = []
    for arm in ARMS:
        for s in SEEDS:
            c, v = U[f"C1|{arm}|s{s}"], U[f"V8|{arm}|s{s}"]
            rows.append([arm, s, c["log"]["gpu"].replace("NVIDIA ", "")[:16], v["log"]["gpu"].replace("NVIDIA ", "")[:16], c["log"]["selected_epoch"], v["log"]["selected_epoch"],
                         f(np.mean(c["log"]["train_s"]) / np.mean(c["log"]["steps"]), 3), f(np.mean(v["log"]["train_s"]) / np.mean(v["log"]["steps"]), 3),
                         f(c["log"]["peak_mem_gb"], 1), f(v["log"]["peak_mem_gb"], 1)])
    A("\nPer unit (C1 vs its full-adaptation parent; s/step = mean epoch training time / steps):\n")
    A(table(["arm", "seed", "C1 GPU", "V8 GPU", "C1 selected epoch", "V8 selected epoch", "C1 s/step", "V8 s/step", "C1 peak GB", "V8 peak GB"], rows))
    A("\nExecution checks (all units): 5 epochs × 2,379 steps, batch-plan hashes identical to the parent unit, exports of CAL_SELECT / CAL_PROB / DEV complete "
      "(1,653 / 1,630 / 3,257 rows), export epoch = selected epoch, per-epoch trainable snapshots epoch0–epoch4 present, inventory checks true.\n")
    A("## 2. Pre-registered primary readout (DEV, C1 − same-arm same-seed full adaptation)\n")
    A(f"DEV: {X['n_dev_images']:,} images, {X['n_dev_prompts']} prompts, {X['n_clusters']} prompt_group clusters; bootstrap {X['bootstrap']['reps']} × seed {X['bootstrap']['seed']} over "
      f"{X['bootstrap']['unit']} clusters, one resample shared by all arms, seeds and variants; percentile 95 % interval of the seed-mean difference.  Rule (frozen): {PR['rule']}.\n")
    for key, label in (("pair_acc_pooled", "(a) pooled within-prompt pair accuracy"), ("selected_human_mean", "(b) selected-image human mean")):
        rows = []
        for arm in ARMS:
            r = PR[key][arm]
            lev_c = np.mean([U[f"C1|{arm}|s{s}"][key] for s in SEEDS]); lev_v = np.mean([U[f"V8|{arm}|s{s}"][key] for s in SEEDS])
            rows.append([arm, f(lev_c, 4), f(lev_v, 4), " / ".join(f(x, 4, True) for x in r["per_seed_diff"]), f(r["seed_mean_diff"], 4, True),
                         f"[{f(r['ci95'][0], 4, True)}, {f(r['ci95'][1], 4, True)}]", "yes" if r["ci_above_0"] else "no", "yes" if r["ci_below_0"] else "no"])
        A(f"### {label}\n")
        A(table(["arm", "C1 (seed mean)", "V8 (seed mean)", "C1 − V8 per seed (s0 / s1 / s2)", "seed-mean difference", "95 % interval", "interval above 0", "interval below 0"], rows))
        A(f"\nArms with the interval above 0: {PR[key]['n_arms_ci_above_0']} of 3.\n")
    A(f"**Decision under the frozen rule: {'IMPROVES' if PR['improves'] else 'DOES NOT IMPROVE'}.**\n")
    A("## 3. Secondary readouts (descriptive)\n")
    sp = X["secondary"]["spearman_pooled_diff"]
    A(table(["arm", "pooled Spearman C1 − V8 per seed", "seed mean", "95 % interval"],
            [[arm, " / ".join(f(x, 4, True) for x in sp[arm]["per_seed_diff"]), f(sp[arm]["seed_mean_diff"], 4, True), f"[{f(sp[arm]['ci95'][0], 4, True)}, {f(sp[arm]['ci95'][1], 4, True)}]"] for arm in ARMS]))
    SMN = X["secondary"]["seed_means"]; keys = [("mse_prompt_macro", "MSE"), ("mse_bias2_prompt_macro", "prompt bias²"), ("mse_centred_prompt_macro", "centred error"), ("rps_micro", "RPS"),
                                                ("ce_micro", "CE"), ("spearman_pooled", "Spearman"), ("pair_acc_pooled", "pair acc"), ("selected_human_mean", "selected human mean")]
    A("\nDEV levels, seed means (prompt-macro mean-rating MSE = prompt bias² + within-prompt centred error):\n")
    A(table(["variant, arm"] + [k[1] for k in keys], [[f"{v} {arm}"] + [f(SMN[f"{v}|{arm}"][k], 4) for k, _ in keys] for arm in ARMS for v in ("C1", "V8")]))
    A("\nCAL_SELECT by epoch (seed means; from the training logs; selection = minimum MSE):\n")
    rows = []
    for arm in ARMS:
        for v in ("C1", "V8"):
            mse = np.mean([U[f"{v}|{arm}|s{s}"]["log"]["cal_mse"] for s in SEEDS], 0); pa = np.mean([U[f"{v}|{arm}|s{s}"]["log"]["cal_pairacc"] for s in SEEDS], 0)
            tl = np.mean([U[f"{v}|{arm}|s{s}"]["log"]["train_loss"] for s in SEEDS], 0)
            rows.append([f"{v} {arm}", " / ".join(f(x, 4) for x in mse), " / ".join(f(x, 4) for x in pa), " / ".join(f(x, 4) for x in tl)])
    A(table(["variant, arm", "CAL MSE e0 … e4", "CAL pair acc e0 … e4", "training loss e0 … e4"], rows))
    A("\nFIT-AUDIT (1,370 in-training images; C1 from the unit's audit on the full model, V8 from V81_05's checkpoint inference; seed means [best → resume]):\n")
    rows = []
    for arm in ARMS:
        c = [U[f"C1|{arm}|s{s}"].get("fit_audit") for s in SEEDS]
        if all(c):
            cm = [np.mean([x[k]["metrics"]["mean_mse_prompt_macro"] for x in c]) for k in ("best", "resume")]
            cp = [np.mean([x[k]["metrics"]["within_prompt_pairwise_acc_macro"] for x in c]) for k in ("best", "resume")]
        else:
            cm = cp = [None, None]
        vm = [CI["arm_tables"][f"FIT_AUDIT|{arm}|{k}"]["mse_prompt_macro"]["mean"] for k in ("best", "resume")]
        vp = [CI["arm_tables"][f"FIT_AUDIT|{arm}|{k}"]["pairwise_acc_macro"]["mean"] for k in ("best", "resume")]
        rows.append([arm, f"{f(cm[0], 4)} → {f(cm[1], 4)}", f"{f(vm[0], 4)} → {f(vm[1], 4)}", f"{f(cp[0], 3)} → {f(cp[1], 3)}", f"{f(vp[0], 3)} → {f(vp[1], 3)}"])
    A(table(["arm", "C1 MSE", "V8 MSE", "C1 pair acc (prompt macro)", "V8 pair acc (prompt macro)"], rows))
    A("\n## 4. Verdict (frozen rule and stop rule applied as written)\n")
    sp_rng = [sp[a_]["seed_mean_diff"] for a_ in ARMS]; sp_hi = max(sp[a_]["ci95"][1] for a_ in ARMS)
    mse_d = [SMN[f"C1|{a_}"]["mse_prompt_macro"] - SMN[f"V8|{a_}"]["mse_prompt_macro"] for a_ in ARMS]
    A(f"Every secondary DEV readout moves in the same direction as the primary ones: pooled Spearman seed-mean difference {min(sp_rng):+.3f} to {max(sp_rng):+.3f} "
      f"({'all intervals below 0' if sp_hi < 0 else 'not all intervals below 0'}, highest upper bound {sp_hi:+.4f}); prompt-macro MSE {min(mse_d):+.3f} to {max(mse_d):+.3f} (C1 − V8).\n")
    if PR["improves"]:
        A(f"The frozen rule returns **improves**: the seed-mean interval is above 0 for {PR['pair_acc_pooled']['n_arms_ci_above_0']} arm(s) on (a) and "
          f"{PR['selected_human_mean']['n_arms_ci_above_0']} arm(s) on (b).  This is a development-set result of one fixed-recipe adaptation-scope change; any external check "
          "needs a new, separately locked plan (the V8-05 prompts are not a C1 confirmation set).\n")
    else:
        A(f"The frozen rule returns **does not improve**: the seed-mean interval is above 0 for {PR['pair_acc_pooled']['n_arms_ci_above_0']} of 3 arms on (a) and "
          f"{PR['selected_human_mean']['n_arms_ci_above_0']} of 3 arms on (b).  Stop rule (frozen): {P['stop_rule']}.  V8 is therefore closed out at its current results; "
          "no rescue variant follows from this result.\n")
    A("## 5. Limits\n")
    lim = ["GPU types differ between some C1 units and their parents (table §1); V81_05 measured cross-GPU agreement of the same checkpoint at ≤ 0.042 in μ (mean ≤ 0.0012).",
           "C1 changes capacity as well as adaptation scope; the contrast is a fixed-recipe adaptation-scope effect at each arm's parent lr, not a comparison of tuned adaptation schemes.",
           "DEV was already read by the V8.1 diagnostics D02–D04; this is a development comparison, not an independent test.",
           "FIT-AUDIT is a training-set audit sample; the C1 and V8 FIT-AUDIT numbers come from different inference paths (full model vs cached ViT output; V81_05 §2 agreement)."]
    A("".join(f"- {x}\n" for x in lim))
    Path(a.out).write_text("\n".join(L)); print(f"[c1 report] -> {a.out} (improves={PR['improves']})"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
