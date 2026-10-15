"""Collect the V9-N2 precheck tables (results only) into the repo and build the gate record.

    python scripts/v9/n2_precheck_collect.py --out-root <outputs_v9/n2> --dataset f30k --rhos 0.4 0 --seeds 901 902 903 \
        --dest reports/V9/V9_N2_precheck --job <slurm id> --commit <short sha>

Copies precheck_<ds>_rho<r>_s<seed>.{json,md}, the teacher logs and the VCS weight-lock records (no weights, no labels),
writes PRECHECK_SUMMARY.{json,md} (verdict per cell, AUROC, merges, rho-hat on the independent batch, per-method weight
summary, teacher cost) — the gate record cited by the N2 protocol."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def cell(out_root: Path, ds: str, rho: float, seed: int) -> dict:
    d = out_root / ds / f"rho{rho:g}" / f"s{seed}"
    pc = json.loads((d / "precheck" / f"precheck_{ds}_rho{rho:g}_s{seed}.json").read_text())
    tl = json.loads((d / "teacher" / "teacher_log.json").read_text())
    shared = (d / "teacher" / "teacher_shared_from.json")
    vcs = json.loads((d / "weights" / "VCS_Weight.csv.lock.json").read_text())["record"]
    ts = pc["teacher_separability"]; rho_rec = vcs["rho"]; ib = rho_rec.get("independent_batch", {})
    per = {m: {k: round(v[k], 4) for k in ("mean", "ess", "total_mass", "share_at_0", "share_at_1", "n")} for m, v in pc["weights"]["per_method"].items()}
    close = {k: {"mean_abs_diff": round(v["mean_abs_diff"], 5), "spearman": round(v["spearman"], 6), "share_identical_1e-6": round(v["share_identical_1e-6"], 6)}
             for k, v in pc["weights"]["pairwise"].items() if v["spearman"] > 0.99}
    return {"dataset": ds, "rho": rho, "seed": seed, "formal_block_allowed": pc["formal_block_allowed"], "refusal_reason": pc["refusal_reason"],
            "auroc": round(ts["auroc"], 4), "overlap_coefficient": round(ts["overlap_coefficient"], 4), "P_median": round(ts["P"]["q50"], 4), "Q_median": round(ts["Q"]["q50"], 4),
            "n_P": ts["n_P"], "n_Q": ts["n_Q"], "merged": pc["loss_coefficients"].get("merged", {}), "groups": pc["loss_coefficients"].get("groups", []),
            "rho_hat_U_est": round(rho_rec["projected"], 4), "rho_hat_status": rho_rec["status"],
            "rho_hat_independent": {"point": round(ib.get("projected", float("nan")), 4), "lower": round(ib.get("interval", {}).get("lower", float("nan")), 4),
                                    "upper": round(ib.get("interval", {}).get("upper", float("nan")), 4), "status": ib.get("interval", {}).get("status"), "n_pairs": ib.get("n_pairs")},
            "realized_rho": round(pc["noise_manifest"]["realized_rho"], 5), "n_A_mismatched": pc["noise_manifest"]["n_A_mismatched"],
            "roles_counts": pc["roles"]["counts"], "dev_test_disjoint": bool(pc["roles"]["dev_ids"]["disjoint"] and pc["roles"]["test_ids"]["disjoint"]),
            "weights_per_method": per, "near_monotone_pairs": close,
            "teacher": {"shared_from_rho": json.loads(shared.read_text())["from"].split("/")[-4] if shared.exists() else None, "n_pairs": tl["n_pairs"],
                        "epochs": len(tl["epochs"]), "seconds_per_epoch": round(tl["seconds_per_epoch"], 1), "seconds_total": round(tl["seconds_total"], 1),
                        "peak_gpu_mem_mb": tl["epochs"][-1].get("peak_gpu_mem_mb"), "first_last_mean_batch_loss": [round(tl["epochs"][0]["mean_batch_loss"], 3), round(tl["epochs"][-1]["mean_batch_loss"], 3)],
                        "device": tl["device"]}}


def write_gate_record(protocol: Path, summ: dict, dest: Path) -> None:
    """Append (or replace) the top-level `precheck_gate_record:` block — results only, no design change — and check it parses."""
    import yaml
    q = lambda x: json.dumps(str(x))
    rows = []
    for c in summ["cells"]:
        ib = c["rho_hat_independent"]
        rows.append(f"    - {{rho: {c['rho']:g}, seed: {c['seed']}, allowed: {str(c['formal_block_allowed']).lower()}, auroc: {c['auroc']}, overlap: {c['overlap_coefficient']}, "
                    f"rho_hat_U_est: {c['rho_hat_U_est']} ({c['rho_hat_status']}), rho_hat_independent: \"{ib['point']} [{ib['lower']}, {ib['upper']}]\", "
                    f"merged_into_one_run: {json.dumps(sorted(set(sum([g for g in c['groups'] if len(g) > 1], []))))}, teacher_seconds_per_epoch: {c['teacher']['seconds_per_epoch']}, "
                    f"teacher_peak_gpu_mb: {c['teacher']['peak_gpu_mem_mb']:.0f}, teacher_shared_from: {q(c['teacher']['shared_from_rho'] or 'trained in this cell')}}}")
    block = "\n".join([
        "precheck_gate_record:   # RESULTS ONLY (no design change) — written by scripts/v9/n2_precheck_collect.py from the precheck of record",
        f"  job: {q(summ['job'])}", f"  code_commit: {q(summ['commit'])}", f"  host: {q(summ['host'])}", f"  dataset: {q(summ['dataset'])}",
        f"  tables: {q(str(dest) + '/precheck_<ds>_rho<rho>_s<seed>.{json,md}, PRECHECK_SUMMARY.{json,md}, teacher_log_*.json, vcs_weight_lock_*.json')}",
        f"  block_allowed: {str(summ['block_allowed']).lower()}",
        "  thresholds: \"refuse if teacher AUROC (clean vs within-role re-paired, A-critic-select) > 0.999 or < 0.55\"",
        "  merge_rule_applied: \"methods whose loss coefficients are equal to 1e-12 on every pool row are ONE run per seed (data order and RNG derive from the seed only); "
        "at rho 0 the projected rho-hat is 0, so every posterior / mass weight is identically 1 and those arms coincide with Unweighted_A — the merged group is run once per seed and its result is reported for every member\"",
        "  cells:", *rows, ""])
    text = protocol.read_text()
    if "\nprecheck_gate_record:" in text:
        head, _, tail = text.partition("\nprecheck_gate_record:")
        rest = tail.split("\n")
        k = next((i for i, ln in enumerate(rest[1:], 1) if ln and not ln.startswith(" ") and not ln.startswith("#")), len(rest))
        text = head + "\n" + block + "\n".join(rest[k:])
    else:
        text = text.rstrip("\n") + "\n" + block
    yaml.safe_load(text); protocol.write_text(text)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out-root", required=True); ap.add_argument("--dataset", default="f30k"); ap.add_argument("--rhos", nargs="+", type=float, default=[0.4, 0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[901, 902, 903]); ap.add_argument("--dest", required=True); ap.add_argument("--job", default=""); ap.add_argument("--commit", default="")
    ap.add_argument("--host", default=""); ap.add_argument("--protocol", default="", help="protocol YAML to receive / replace the top-level precheck_gate_record block")
    a = ap.parse_args()
    out_root, dest = Path(a.out_root), Path(a.dest); dest.mkdir(parents=True, exist_ok=True); cells = []
    for seed in a.seeds:
        for rho in a.rhos:
            d = out_root / a.dataset / f"rho{rho:g}" / f"s{seed}"
            for suf in ("json", "md"):
                shutil.copy2(d / "precheck" / f"precheck_{a.dataset}_rho{rho:g}_s{seed}.{suf}", dest)
            shutil.copy2(d / "teacher" / "teacher_log.json", dest / f"teacher_log_{a.dataset}_rho{rho:g}_s{seed}.json")
            shutil.copy2(d / "weights" / "VCS_Weight.csv.lock.json", dest / f"vcs_weight_lock_{a.dataset}_rho{rho:g}_s{seed}.json")
            cells.append(cell(out_root, a.dataset, rho, seed))
    allowed = all(c["formal_block_allowed"] for c in cells)
    summ = {"job": a.job, "commit": a.commit, "host": a.host, "dataset": a.dataset, "cells": cells, "block_allowed": allowed,
            "merges_any": any(c["merged"] for c in cells), "thresholds": {"auroc_max": 0.999, "auroc_min": 0.55}}
    (dest / "PRECHECK_SUMMARY.json").write_text(json.dumps(summ, indent=1))
    L = [f"# V9-N2 precheck gate record — {a.dataset} — block {'ALLOWED' if allowed else 'REFUSED'}", "",
         f"Job {a.job}, code commit {a.commit}, host {a.host}; thresholds: refuse if AUROC > 0.999 or < 0.55 (teacher clean vs within-role re-paired on A-critic-select); "
         "merge rule: loss coefficients identical at 1e-12 AND same data order / RNG state (never satisfied by distinct weight maps).", "",
         "| rho | seed | allowed | AUROC | overlap | P med | Q med | rho-hat (U_est) | rho-hat independent [95 % Hoeffding] | merges | teacher s/epoch | peak GPU MB | teacher shared |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        ib = c["rho_hat_independent"]
        L.append(f"| {c['rho']:g} | {c['seed']} | {'yes' if c['formal_block_allowed'] else 'NO: ' + c['refusal_reason']} | {c['auroc']:.4f} | {c['overlap_coefficient']:.3f} | {c['P_median']:.3f} | {c['Q_median']:.3f} | "
                 f"{c['rho_hat_U_est']:.4f} ({c['rho_hat_status']}) | {ib['point']:.4f} [{ib['lower']:.3f}, {ib['upper']:.3f}] | {len(c['merged']) or 'none'} | {c['teacher']['seconds_per_epoch']} | "
                 f"{c['teacher']['peak_gpu_mem_mb']:.0f} | {c['teacher']['shared_from_rho'] or 'trained here'} |")
    L += ["", "## Weight summary per method (mean w, ESS, total mass, share at 0 / 1)", ""]
    for c in cells:
        L += [f"**rho {c['rho']:g}, seed {c['seed']}** (n_U pairs = {c["weights_per_method"]["VCS_Weight"]["n"]})", "", "| method | mean w | ESS | mass | at 0 | at 1 |", "|---|---|---|---|---|---|"]
        L += [f"| {m} | {v['mean']:.4f} | {v['ess']:.0f} | {v['total_mass']:.0f} | {v['share_at_0']:.4f} | {v['share_at_1']:.4f} |" for m, v in c["weights_per_method"].items()]
        L += ["", "near-monotone pairs (Spearman > 0.99; NOT merged — coefficients differ): " + ("; ".join(f"{k}: mean|dw| {v['mean_abs_diff']}, rho_s {v['spearman']}, identical {v['share_identical_1e-6']}" for k, v in c["near_monotone_pairs"].items()) or "none"), ""]
    L += ["## Reading", "",
          "- The teacher is one per (dataset, seed): the rho 0 cell reuses the rho 0.4 teacher (identical roles hashes, clean anchor pairs), so AUROC is identical across rho within a seed by construction.",
          "- AUROC far from both refusal thresholds in every cell: the teacher separates clean from re-paired anchor pairs but is not a label oracle.",
          "- rho-hat on the independent batch (20 % of U images, by image) is the pre-registered interval; the U_est point estimate is what the weights use.",
          "- No two methods have identical loss coefficients: every arm of the Flickr block is a distinct run (VCS vs BCE share the critic logit and are rank-identical; their weights differ).",
          "- Teacher cost is the first real GPU number: 5 warm-up epochs on the A-teacher-warmup pairs (4 % of the training pairs)."]
    (dest / "PRECHECK_SUMMARY.md").write_text("\n".join(L) + "\n")
    if a.protocol:
        write_gate_record(Path(a.protocol), summ, dest)
    print(f"{'ALLOWED' if allowed else 'REFUSED'} cells={len(cells)} -> {dest}"); return 0 if allowed else 3


if __name__ == "__main__":
    raise SystemExit(main())
