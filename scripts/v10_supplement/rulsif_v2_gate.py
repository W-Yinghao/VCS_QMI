"""Formal FIT-only functional gate for RuLSIF_exact_v2, with v1's values on the same gate (CPU Slurm).

Pre-registered in v10/supplement/RULSIF_V2_PREREGISTRATION.json (thresholds and levels fixed
before any v2 number). Seed 1017 (the CAL seed): level A evaluates exactly the CAL fits, level B
repeats the identical fitting procedure on 4 core folds. Reads FIT only; never CAL/DEV/TEST/EPFL.

If the labelled post-gate amendment record exists, v2 is evaluated on the whole frozen grid,
eligibility follows the amendment's required levels (the existing per-level verdicts, unchanged),
and every number is compared with the original gate run (--compare-report / --compare-rows); the
family passes only if exactly the declared eligible candidates pass and no verdict differs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from v10.supplement.contract import (RULSIF_V2_AMENDMENT, RULSIF_V2_RECORD, candidates,  # noqa: E402
                                     digest, read, rulsif_v2_amendment, rulsif_v2_grid, sha,
                                     source_inventory)

SEED = 1017
FOLDS = 4


def fingerprint(model):
    w = model.coefficients.detach().double().cpu()
    return {"bandwidth_left": getattr(model, "bandwidth_left", model.bandwidth),
            "bandwidth_right": getattr(model, "bandwidth_right", model.bandwidth),
            "coefficients": len(w), "coefficient_sum": float(w.sum()),
            "coefficient_l2": float(w.norm()), "constant": float(w[0]) if len(w) == len(model.centres_left) + 1 else None,
            "centres_sha256": hashlib.sha256(model.centres_left.detach().cpu().numpy().tobytes()
                                             + model.centres_right.detach().cpu().numpy().tobytes()).hexdigest()}


ROW_FIELDS = ("F_true", "F_imposed", "truth_dependence_term", "truth_quadratic_term")


def reproduction(report_path, rows_path, families, new_rows, family="RuLSIF_exact_v2"):
    """Compare a re-run with the original gate report and rows: values, summaries and verdicts."""
    original = read(report_path)
    old_rows = {}
    with Path(rows_path).open() as stream:
        for line in stream:
            r = json.loads(line)
            if r["family"] == family:
                old_rows[(r["order"], r["level"], r["core"], r["perturbation"])] = r
    if set(old_rows) != set(new_rows):
        raise RuntimeError("Re-run rows do not cover the same samples as the original gate")
    largest, identical = 0.0, 0
    for key, new in new_rows.items():
        old = old_rows[key]
        values = [(old[f], new[f]) for f in ROW_FIELDS] + list(zip(old["F_local"], new["F_local"]))
        largest = max([largest] + [abs(a - b) for a, b in values])
        identical += all(a == b for a, b in values)
    summary_fields, verdicts = [], []
    old_entries = {e["candidate"]["order"]: e for e in original["families"][family]["candidates"]}
    for entry in families[family]["candidates"]:
        old = old_entries[entry["candidate"]["order"]]
        if old["candidate"] != entry["candidate"]:
            raise RuntimeError("Re-run evaluated a different candidate definition")
        for level in ("deployed_position_heldout", "crossfit4_core_heldout"):
            summary_fields += [f"{entry['candidate']['order']}:{level}:{k}" for k in entry[level]["summary"]
                               if entry[level]["summary"][k] != old[level]["summary"].get(k)]
            if entry[level]["verdict"] != old[level]["verdict"]:
                verdicts.append({"order": entry["candidate"]["order"], "level": level,
                                 "original": old[level]["verdict"], "rerun": entry[level]["verdict"]})
    return {"original_report": str(report_path), "original_report_sha256": sha(report_path),
            "original_job": original["job"], "original_rows": str(rows_path), "original_rows_sha256": sha(rows_path),
            "rows_compared": len(new_rows), "rows_identical": identical, "max_abs_difference": largest,
            "summary_fields_differing": summary_fields, "verdict_differences": verdicts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="report JSON")
    parser.add_argument("--rows", type=Path, required=True, help="per-sample rows JSONL (energies only)")
    parser.add_argument("--families", nargs="+", default=["RuLSIF_exact", "RuLSIF_exact_v2"])
    parser.add_argument("--compare-report", type=Path, help="original gate report (required under the amendment)")
    parser.add_argument("--compare-rows", type=Path, help="original gate rows (required under the amendment)")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("The functional gate runs under Slurm")
    started = time.time()
    record = read(RULSIF_V2_RECORD)
    thresholds = record["functional_gate"]["thresholds"]
    preregistered = record["functional_gate"]["required_levels"]
    if not set(preregistered) <= {"deployed_position_heldout", "crossfit4_core_heldout"} or not preregistered:
        raise RuntimeError("Unknown or empty required gate levels")
    amendment = rulsif_v2_amendment() if "RuLSIF_exact_v2" in args.families else None
    if amendment is not None:
        rule = amendment["amended_rule"]
        cited = amendment["amends"]["original_gate_reports"]["RuLSIF_exact_v2"]
        if (args.compare_report is None or args.compare_rows is None
                or sha(args.compare_report) != cited["sha256"] or sha(args.compare_rows) != cited["rows_sha256"]):
            raise RuntimeError("The amended gate must be compared with the cited original gate run")

    def required_for(family):
        return rule["required_levels"] if amendment is not None and family == "RuLSIF_exact_v2" else preregistered
    inventory = source_inventory()
    import torch
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "2")))
    from v10 import adapter as A
    from v10.supplement import rulsif_gate as G
    from v10.supplement import rulsif_v2 as V2
    from v10.supplement.registration import RuLSIFRigidEnergy
    from v10.supplement.training import KernelTrainer
    guard = A.DataGuard()
    fit = A.fit_cores(guard, 1.0, SEED)
    if len(fit) != 32:
        raise RuntimeError("FIT32 required")
    inputs = G.fit_inputs(guard, fit, "cpu")
    folds = G.crossfit_folds(fit, FOLDS)
    args.rows.parent.mkdir(parents=True, exist_ok=True)
    rows_tmp = args.rows.with_name("." + args.rows.name + ".tmp")
    families = {}
    new_rows = {}
    with rows_tmp.open("w") as rows_out:
        def emit(family, order, level, r):
            rows_out.write(json.dumps({"family": family, "order": order, "level": level, **r}) + "\n")
            if family == "RuLSIF_exact_v2":
                new_rows[(order, level, r["core"], r["perturbation"])] = r

        def block(family, candidate, level, model, energy, cores, extra=None):
            rows = G.evaluate(model, energy, inputs, guard, cores, shg_input=candidate.get("shg_input", "intensity"))
            for r in rows:
                emit(family, candidate["order"], level, r)
            summary = G.summarize(rows)
            return {"summary": summary, "verdict": G.verdict(summary, thresholds), **(extra or {})}

        for family in args.families:
            results = []
            required = required_for(family)
            grid = rulsif_v2_grid() if family == "RuLSIF_exact_v2" else candidates(family)
            for candidate in grid:
                entry = {"candidate": candidate}
                if family == "RuLSIF_exact":
                    trainer = KernelTrainer(candidate, SEED, guard, "cpu")
                    trainer.step()
                    deployed = trainer.ratio
                    settings = dict(constant=False, support="square_v1", bandwidth_rule="joint_median",
                                    centres_per_core=candidate["centres"] // 32,
                                    positions_per_core=candidate["fit_positions_per_core"],
                                    bandwidth_multiplier=candidate["bandwidth_multiplier"],
                                    ridge=candidate["ridge"], alpha=candidate["alpha"])
                    # The shared fitting code reproduces v1's deployed fit (same stream and draw order).
                    twin, scenes, _ = V2.fit_kernel_model(guard, fit, stream_tag=f"V10|RuLSIF|FIT|{SEED}",
                                                          device="cpu", purpose="v1 twin check", **settings)
                    twin.fit(scenes)
                    if not (torch.equal(twin.centres_left, deployed.centres_left)
                            and torch.equal(twin.centres_right, deployed.centres_right)
                            and twin.bandwidth == deployed.bandwidth
                            and torch.equal(twin.coefficients, deployed.coefficients)):
                        raise RuntimeError("Shared fitting code does not reproduce v1's deployed fit")
                    entry["shared_code_reproduces_deployed_fit"] = True
                    energy = RuLSIFRigidEnergy

                    def crossfit(train, j):
                        model, scenes, _ = V2.fit_kernel_model(
                            guard, train, stream_tag=f"V10|RuLSIF|FIT|{SEED}|crossfit{j}", device="cpu",
                            purpose="V10 RuLSIF functional gate cross-fit (v1 rule)", **settings)
                        model.fit(scenes)
                        return model
                else:
                    trainer = V2.KernelTrainerV2(candidate, SEED, guard, "cpu")
                    trainer.step()
                    deployed = trainer.ratio
                    energy = V2.RuLSIFV2RigidEnergy

                    def crossfit(train, j):
                        t = V2.KernelTrainerV2(candidate, SEED, guard, "cpu", cores=train,
                                               tag=V2.stream_tag(SEED) + f"|crossfit{j}")
                        t.step()
                        return t.ratio
                entry["deployed_fit"] = fingerprint(deployed)
                entry["deployed_position_heldout"] = block(family, candidate, "deployed_position_heldout",
                                                           deployed, energy, fit)
                fold_rows, fold_meta = [], []
                for j, held in enumerate(folds):
                    model = crossfit([c for c in fit if c not in held], j)
                    fold_rows.extend(G.evaluate(model, energy, inputs, guard, held,
                                                shg_input=candidate.get("shg_input", "intensity")))
                    fold_meta.append({"fold": j, "held_out": held, **fingerprint(model)})
                for r in fold_rows:
                    emit(family, candidate["order"], "crossfit4_core_heldout", r)
                summary = G.summarize(fold_rows)
                entry["crossfit4_core_heldout"] = {"summary": summary, "verdict": G.verdict(summary, thresholds),
                                                   "folds": fold_meta}
                entry["status"] = "PASS" if all(entry[level]["verdict"]["status"] == "PASS" for level in required) else "FAIL"
                results.append(entry)
                print("GATE", family, candidate["order"], entry["status"],
                      "A meanJ %.4f" % entry["deployed_position_heldout"]["summary"]["mean_F_true"],
                      "B meanJ %.4f" % summary["mean_F_true"], "%.0fs" % (time.time() - started), flush=True)
            passing = [e["candidate"]["order"] for e in results if e["status"] == "PASS"]
            if amendment is not None and family == "RuLSIF_exact_v2":
                # Exactly the declared eligible candidates must pass; the rest are excluded, not failed.
                status = "PASS" if passing and passing == rule["eligible_orders"] else "FAIL"
            else:
                status = "PASS" if len(passing) == len(results) else "FAIL"
            families[family] = {"status": status, "candidates": results, "required_levels": required,
                                "eligible_orders": passing,
                                "excluded_orders": [e["candidate"]["order"] for e in results if e["status"] != "PASS"]}
    os.replace(rows_tmp, args.rows)
    replay = None
    if amendment is not None:
        replay = reproduction(args.compare_report, args.compare_rows, families, new_rows)
        if replay["verdict_differences"]:
            families["RuLSIF_exact_v2"]["status"] = "FAIL"
    roles = {r["role"] for r in guard.reads}
    if roles != {"FIT"}:
        raise RuntimeError("Gate touched a non-FIT role: " + repr(roles))
    commit = subprocess.run(["git", "-C", os.environ.get("V10R2_REPO", str(REPO)), "rev-parse",
                             os.environ.get("V10R2_COMMIT", "HEAD")], capture_output=True, text=True).stdout.strip()
    report = {"schema": "v10.rulsif_exact_v2.fit_gate.v1", "stage": "FIT_FUNCTIONAL_GATE",
              "family": "RuLSIF_exact_v2", "status": families.get("RuLSIF_exact_v2", {}).get("status", "NOT_RUN"),
              "job": os.environ["SLURM_JOB_ID"], "utc": time.strftime("%FT%TZ", time.gmtime()),
              "source_commit": commit, "code_sha256": digest(inventory), "source_files": inventory,
              "preregistration_sha256": sha(RULSIF_V2_RECORD), "candidate_grid_sha256": record.get("candidate_grid_sha256"),
              "thresholds": thresholds,
              "rule_mode": "amended" if amendment is not None else "preregistered",
              "required_levels": required_for("RuLSIF_exact_v2"),
              "amendment_sha256": sha(RULSIF_V2_AMENDMENT) if amendment is not None else None,
              "pass_rule": (amendment["amended_rule"]["candidate_rule"] if amendment is not None
                            else record["functional_gate"]["pass_rule"]),
              "reproduction": replay, "seed": SEED,
              "folds": folds, "roles_read": sorted(roles), "held_test_read": False, "cal_read": False,
              "dev_read": False, "fit_reads": sum(r.get("kind") != "rule" for r in guard.reads),
              "rows": str(args.rows), "rows_sha256": sha(args.rows), "seconds": time.time() - started,
              "families": families}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name("." + args.out.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
    os.replace(temporary, args.out)
    print("FIT_GATE", report["status"], {f: v["status"] for f, v in families.items()}, flush=True)
    if "RuLSIF_exact_v2" in families and report["status"] != "PASS":
        sys.exit(3)


if __name__ == "__main__":
    main()
