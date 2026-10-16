"""Read-only inventory for a V10 server handoff. Never imports host training code."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


HISTORICAL_HOST = {
    "commit": "27117c9ae78d64d39a6548c40b00ff58f70cc4d4",
    "paths": ["src/v9/reg/g1.py", "src/v9/reg/r6_reg_pilot.py"],
    "status": "reported_in_recovered_design_not_refetched_this_turn",
}


def git_read(repo: Path, *args: str) -> str:
    env = dict(os.environ)
    env["GIT_OPTIONAL_LOCKS"] = "0"
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT,
        env=env, timeout=15,
    ).strip()


def inspect_source(path: Path) -> dict:
    if not path.is_file():
        return {"exists": False}
    raw = path.read_bytes()
    out = {"exists": True, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    try:
        tree = ast.parse(raw.decode("utf-8"))
        out["top_level_symbols"] = [
            {"name": node.name, "kind": type(node).__name__}
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
    except (ValueError, SyntaxError, UnicodeDecodeError) as exc:
        out["parse_error"] = str(exc)
    return out


def inspect_manifest(path: Path, data_root: Path) -> dict:
    required = {"dataset", "record_id", "group_id", "fixed_path", "moving_path",
                "role", "exposure", "source_version", "inclusion", "exclusion_reason"}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        rows = list(reader)
    errors, warnings, roles, ids, sources, confirmation_groups = [], [], {}, set(), {}, set()
    if not required.issubset(columns):
        errors.append({"missing_columns": sorted(required - columns)})
    if not rows:
        errors.append({"empty_manifest": True})
    counts = {}
    for line, row in enumerate(rows, start=2):
        if not required.issubset(row):
            continue
        key = (row["dataset"], row["record_id"])
        if key in ids:
            errors.append({"line": line, "duplicate_record": list(key)})
        ids.add(key)
        if row["inclusion"] not in {"include", "exclude"}:
            errors.append({"line": line, "invalid_inclusion": row["inclusion"]})
            continue
        if row["inclusion"] == "exclude":
            if not row["exclusion_reason"].strip():
                errors.append({"line": line, "missing_exclusion_reason": True})
            continue
        if any(not row[x].strip() for x in required - {"exclusion_reason"}):
            errors.append({"line": line, "blank_required_value": True})
        group = (row["dataset"], row["group_id"])
        role = row["role"]
        if role not in {"FIT", "CAL", "DEV", "CONFIRM"}:
            errors.append({"line": line, "invalid_role": role})
        roles.setdefault(group, set()).add(role)
        counts[(row["dataset"], role)] = counts.get((row["dataset"], role), 0) + 1
        if row["exposure"] not in {"used_for_design", "never_read", "unknown"}:
            errors.append({"line": line, "invalid_exposure": row["exposure"]})
        if role == "CONFIRM":
            confirmation_groups.add(group)
            if row["exposure"] != "never_read":
                errors.append({"line": line, "confirmation_exposure_unverified": True})
        if row["exposure"] == "unknown":
            warnings.append({"line": line, "exposure_unknown": True})
        for side in ["fixed_path", "moving_path"]:
            candidate = Path(row[side]).expanduser()
            if not candidate.is_absolute():
                candidate = data_root / candidate
            candidate = candidate.resolve()
            if not candidate.is_file():
                errors.append({"line": line, "missing_image": str(candidate)})
            sources.setdefault(str(candidate), set()).add(role)
    for group, role_set in roles.items():
        if len(role_set) > 1:
            errors.append({"group": list(group), "cross_role_group": sorted(role_set)})
    for source, role_set in sources.items():
        if len(role_set) > 1:
            errors.append({"source": source, "cross_role_image_path": sorted(role_set)})
    return {
        "path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "included_records_by_dataset_role": [
            {"dataset": d, "role": r, "count": n} for (d, r), n in sorted(counts.items())
        ],
        "independent_group_count_declared": len(roles),
        "confirmation_group_count_declared": len(confirmation_groups),
        "errors": errors, "warnings": warnings,
        "limitations": ["group IDs and never_read declarations require source evidence",
                        "does not detect visual near-duplicates or undocumented patient relationships",
                        "does not validate licenses, image decoding, registration masks or alignment"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()
    result = {"stage": "T00", "historical_host_reference": HISTORICAL_HOST,
              "server_repo": str(repo), "python": sys.version.split()[0],
              "remote_jobs_started": False, "source_review_complete": False}
    errors = []
    try:
        result["current_commit"] = git_read(repo, "rev-parse", "HEAD")
        result["current_branch"] = git_read(repo, "branch", "--show-current")
        result["dirty_paths"] = git_read(repo, "status", "--porcelain", "--untracked-files=no").splitlines()
    except (OSError, subprocess.SubprocessError) as exc:
        errors.append({"git_inventory_error": str(exc)})
    result["historical_paths_at_current_checkout"] = {
        rel: inspect_source(repo / rel) for rel in HISTORICAL_HOST["paths"]
    }
    result["host_path_mapping_required"] = any(
        not value["exists"] for value in result["historical_paths_at_current_checkout"].values()
    )
    if args.manifest:
        try:
            result["data_inventory"] = inspect_manifest(args.manifest, args.data_root)
            errors.extend(result["data_inventory"]["errors"])
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            errors.append({"manifest_error": str(exc)})
    else:
        result["data_inventory"] = {"status": "NOT_PROVIDED"}
    result["errors"] = errors
    result["status"] = "ERROR" if errors else "INVENTORY_WRITTEN_REQUIRES_HOST_REVIEW"
    result["does_not_certify"] = ["V9 completion", "GPU timings", "host adapter fidelity",
                                 "scientific outcomes", "independent confirmation"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output), "errors": len(errors)}))
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
