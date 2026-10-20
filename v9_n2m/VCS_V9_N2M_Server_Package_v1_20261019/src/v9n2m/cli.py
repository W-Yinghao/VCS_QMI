"""Torch-free command discovery; scientific imports occur only in their commands."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .common import PACKAGE_ROOT, assert_same_file, read_json, sha256_file, utc_now, write_json


def _site_arguments(parser):
    parser.add_argument("--base", required=True, help="Original immutable handoff/v9_full_20261017 directory")
    parser.add_argument("--out", required=True, help="Separate new-round output directory")


def _input_arguments(parser):
    parser.add_argument("--old-output", help="Relocated outputs_v9/n2/f30k root; source digests remain fixed")
    parser.add_argument("--data", help="Relocated f30k_precomp train+DEV directory; preserve size+mtime")
    parser.add_argument("--vocab", help="Relocated frozen vocabulary JSON")


def parser():
    ap = argparse.ArgumentParser(description="V9 N2M: Constant-U and Shuffle-VCS full-training handoff (6–9 new runs).")
    ap.add_argument("--version", action="version", version="VCS_V9_N2M_v1_20261019 / 1.0.0")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("verify-package", help="Check packaged file hashes; no Torch, data or network")
    p = sub.add_parser("inspect", help="Rebuild old N2F 33-unit census from artifacts")
    _site_arguments(p); _input_arguments(p)
    p.add_argument("--metadata-only", action="store_true", help="Read report snapshot without checkpoints/data; cannot release training")
    p = sub.add_parser("bootstrap", help="Copy verified old runtime and patch telemetry in the new output only")
    _site_arguments(p)
    p = sub.add_parser("verify-torch", help="Actual tiny SGR equivalence/resume/coefficients check; CUDA needed on server")
    _site_arguments(p); p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p = sub.add_parser("prepare", help="Freeze controls and one auto/replay9/reuse6 execution plan")
    _site_arguments(p); _input_arguments(p)
    p.add_argument("--mode", choices=("auto", "replay9", "reuse6"), default="auto")
    p.add_argument("--verification", help="SERVER_VERIFICATION.json from the actual allocated GPU")
    p.add_argument("--historical-environments", help="Optional JSON with archived original-job environment evidence; never a new probe relabelled as history")
    p = sub.add_parser("run", help="Train/resume one frozen seed group or unit; requires allocated CUDA")
    p.add_argument("--experiment", required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--seed", type=int, choices=(901,902,903))
    group.add_argument("--unit", help="For example s901__ConstantU; frozen predecessor order enforced")
    p = sub.add_parser("score", help="Offline EVAL_ONLY probability and mass diagnostics for one seed")
    p.add_argument("--experiment", required=True); p.add_argument("--seed", type=int, choices=(901,902,903), required=True)
    p.add_argument("--labels", required=True, help="Separate pairs.eval_only.json, read only by this scorer")
    p.add_argument("--out", required=True)
    p = sub.add_parser("export-fixed-losses", help="GPU inference: original VCS checkpoint, 8 fixed observed batches, no labels")
    p.add_argument("--experiment", required=True); p.add_argument("--seed", type=int, choices=(901,902,903), required=True)
    p.add_argument("--out-csv", required=True)
    p = sub.add_parser("score-fixed-losses", help="EVAL_ONLY reweighting of one shared frozen-student loss export")
    p.add_argument("--experiment", required=True); p.add_argument("--seed", type=int, choices=(901,902,903), required=True)
    p.add_argument("--labels", required=True); p.add_argument("--loss-csv", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("summarize", help="Recompute full endpoints, paired contrasts, failures and collapse from artifacts")
    p.add_argument("--experiment", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("accounting", help="Read-only sacct collection, deduplicated by allocation")
    p.add_argument("--experiment", required=True); p.add_argument("--out", required=True)
    p.add_argument("--sacct-text", help="Optional already collected parsable sacct file")
    return ap


def _verify_package():
    manifest = read_json(PACKAGE_ROOT / "PACKAGE_MANIFEST.json")
    for rel, digest in manifest["files_sha256"].items():
        candidate = (PACKAGE_ROOT / rel).resolve()
        if PACKAGE_ROOT.resolve() not in candidate.parents:
            raise ValueError("Package manifest contains an escaping path")
        assert_same_file(candidate, digest)
    return {"status": "PASS", "files_checked": len(manifest["files_sha256"]),
            "protocol_id": manifest["protocol_id"], "note": "File identity check; does not imply GPU training has run."}


def _verify_torch(args):
    from .runtime import bootstrap_runtime, collect_environment, configure_precision, package_code_identity
    from .verify_torch import verify_runtime
    out = Path(args.out).resolve()
    info = bootstrap_runtime(args.base, out)
    raw_path = out / f"TORCH_VERIFICATION_{args.device}.json"
    if raw_path.exists() or (out / "experiment.json").exists():
        raise ValueError("Verification output already exists or experiment is frozen; preserve it and use a fresh output root for a different environment")
    try:
        configure_precision()
    except ModuleNotFoundError:
        pass  # The verifier writes an honest NOT_RUN receipt when Torch is absent.
    result = verify_runtime(out / "runtime", Path(args.base).resolve() / "source", raw_path, device=args.device)
    if result.get("status") == "PASS":
        configure_precision()
        environment = collect_environment()
    else:
        environment = result.get("environment")
    envelope = {"schema_version": 1, "created_utc": utc_now(), "status": result.get("status", "FAIL"),
                "device": args.device, "environment": environment,
                "verification_report": str(raw_path), "verification_report_sha256": sha256_file(raw_path),
                "runtime_receipt_sha256": sha256_file(out / "runtime/RUNTIME.json"),
                "effective_runtime_digest": info["effective_digest"],
                "package_code_digest": package_code_identity()["digest"]}
    write_json(out / ("SERVER_VERIFICATION.json" if args.device == "cuda" else "CPU_VERIFICATION.json"), envelope)
    return envelope


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "verify-package":
        result = _verify_package()
    elif args.command == "inspect":
        from .identity import closeout_census, resolve_inputs, verify_source, verify_training_inputs
        out = Path(args.out)
        source = verify_source(args.base, include_cache=not args.metadata_only)
        census = closeout_census(args.base, check_binary=not args.metadata_only, out_path=out / "N2F_CENSUS.json")
        inputs = None
        if not args.metadata_only:
            paths = resolve_inputs(args.base, args.old_output, args.data, args.vocab)
            inputs = verify_training_inputs(args.base, paths)
        write_json(out / "INSPECTION.json", {"source": source, "census": census, "inputs": inputs})
        result = {"status": "READY" if census["ready_for_new_training"] else "NOT_READY", "metadata_complete": census["metadata_complete"],
                  "binary_verified_complete": census["complete"], "expected": 33, "source_files_verified": source["files_checked"],
                  "report": str(out / "INSPECTION.json")}
    elif args.command == "bootstrap":
        from .runtime import bootstrap_runtime
        info = bootstrap_runtime(args.base, args.out)
        result = {"status": "PASS", "runtime": info["runtime"], "changed_files": info["changed_source_files"], "effective_digest": info["effective_digest"]}
    elif args.command == "verify-torch":
        result = _verify_torch(args)
    elif args.command == "prepare":
        from .prepare import prepare_experiment
        exp = prepare_experiment(args.base, args.out, mode=args.mode, old_output=args.old_output, data=args.data, vocab=args.vocab,
                                 verification_path=args.verification, historical_environments=args.historical_environments)
        result = {"status": "FROZEN", "experiment": str(Path(args.out).resolve() / "experiment.json"),
                  "new_full_runs": exp["new_full_runs"], "reference_actions": {s: r["VCS_action"] for s,r in exp["reference_decisions"].items()}}
    elif args.command == "run":
        from .runner import run_seed, run_unit
        result = run_seed(args.experiment, args.seed) if args.seed is not None else run_unit(args.experiment, args.unit)
    elif args.command in ("score", "score-fixed-losses"):
        from .offline import score_weights, score_fixed_losses
        from .prepare import verify_experiment
        exp = verify_experiment(args.experiment)
        u = next(u for u in exp["units"] if u["seed"] == args.seed and u["arm"] == "VCS")
        source = Path(exp["inputs"]["old_output"]) / "rho0.4" / f"s{args.seed}" / "weights/VCS_Weight.csv"
        # Bind evaluator labels to the original freeze only in this explicitly offline command.
        from .identity import pinned_freeze
        frozen = pinned_freeze(exp["base_handoff"])
        label_key = str(Path(frozen["old_output"]) / "rho0.4" / f"s{args.seed}" / "eval_only/pairs.eval_only.json")
        assert_same_file(args.labels, frozen["input_sha256"][label_key])
        if args.command == "score":
            report = score_weights(u["observed_path"], args.labels, source, str(source)+".lock.json", u["controls_dir"], args.out,
                                   noise_manifest_path=u["noise_manifest_path"])
            result = {"status": "PASS", "seed": args.seed, "diagnostic_only": True, "output": args.out, "rho": report["rho"]}
        else:
            historical_result_path = Path(u["historical_reference_dir"]) / "full_result.json"
            assert_same_file(historical_result_path, exp["reference_decisions"][str(args.seed)]["reference_result_sha256"])
            historical_result = read_json(historical_result_path)
            report = score_fixed_losses(u["observed_path"], args.labels, u["controls_dir"], args.loss_csv, args.out,
                                         expected_experiment_sha256=sha256_file(args.experiment),
                                         expected_checkpoint_sha256=historical_result["artifact_sha256"]["best.pt"])
            result = {"status": "PASS", "seed": args.seed, "diagnostic_only": True, "output": args.out,
                      "batch_arm_rows": len(report["batches"]), "identity": report.get("identity"), "warnings": report.get("warnings", [])}
    elif args.command == "export-fixed-losses":
        from .diagnostics import export_fixed_losses
        report = export_fixed_losses(args.experiment, args.seed, args.out_csv)
        result = {"status": "PASS", "seed": args.seed, "output": args.out_csv, "checkpoint_sha256": report["checkpoint_sha256"]}
    elif args.command == "summarize":
        from .aggregate import summarize_experiment
        report = summarize_experiment(args.experiment, args.out)
        result = {"status": "SUMMARIZED", "output": args.out, "counts": report.get("counts"), "note": "Read RESULTS_CN.md and summary.json; incomplete units are not imputed."}
    elif args.command == "accounting":
        from .accounting import collect_accounting
        report = collect_accounting(args.experiment, args.out, args.sacct_text)
        result = {"status": "ACCOUNTING_COMPLETE" if report["complete_terminal_accounting"] else "ACCOUNTING_PARTIAL",
                  "output": args.out, "gpu_allocation_hours_known": report["gpu_allocation_hours_known"], "missing_jobs": report["missing_jobs"]}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    status = result.get("status")
    if status in ("INTERRUPTED", "COMPLETE_STOP_REQUESTED"):
        return 75
    if status in ("NOT_READY", "NOT_RUN", "FAIL"):
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
