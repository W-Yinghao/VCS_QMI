"""Verify the actual N2F execution snapshot and reconstruct its closeout census."""
from __future__ import annotations

import math
from pathlib import Path

from .common import PACKAGE_ROOT, assert_same_file, read_json, sha256_file, utc_now, write_json


def pinned_freeze(base):
    base = Path(base).resolve()
    pinned = PACKAGE_ROOT / "provenance/N2F_FROZEN.json"
    assert_same_file(base / "FROZEN.json", sha256_file(pinned))
    assert_same_file(base / "GROUPS.json", sha256_file(PACKAGE_ROOT / "provenance/N2F_GROUPS.json"))
    return read_json(pinned)


def verify_source(base, include_cache=False):
    base = Path(base).resolve()
    frozen = pinned_freeze(base)
    checked = {}
    for rel, expected in frozen["code_sha256"].items():
        if not include_cache and rel.startswith("source/token_cache/"):
            continue
        checked[rel] = assert_same_file(base / rel, expected)
    return {"status": "PASS", "files_checked": len(checked), "sha256": checked,
            "cache_checked": include_cache, "base_freeze_sha256": sha256_file(base / "FROZEN.json")}


def expected_steps(n, batch=128):
    # iterate_batches in the inherited host drops only singletons.
    q, r = divmod(n, batch)
    return q + int(r >= 2)


def validate_complete_unit(path, *, expected_identity=None, check_binary=True, result_override=None):
    p = Path(path)
    result = read_json(p / "full_result.json") if result_override is None else result_override
    identity = result["identity"]
    if read_json(p / "identity.json") != identity:
        raise ValueError(f"Result/identity mismatch: {p}")
    for k, v in (expected_identity or {}).items():
        if identity.get(k) != v:
            raise ValueError(f"Unexpected {k} in {p}: {identity.get(k)!r} != {v!r}")
    required = {"best.pt", "last.pt", "train_log.json", "run_manifest.json", "identity.json"}
    digests = result.get("artifact_sha256", {})
    if not required.issubset(digests):
        raise ValueError(f"Missing artifact identities: {p}")
    checked, omitted = {}, []
    for rel, expected in digests.items():
        candidate = p / rel
        if candidate.resolve().parent != p.resolve():
            raise ValueError(f"Artifact is outside unit directory: {rel}")
        if not check_binary and rel.endswith(".pt"):
            omitted.append(rel)
            continue
        checked[rel] = assert_same_file(candidate, expected)
    log = read_json(p / "train_log.json")
    epochs = log["epochs"]
    if [r["epoch"] for r in epochs] != list(range(40)):
        raise ValueError(f"Not exactly 40 complete epochs: {p}")
    n_pool = int(result["n_pool"])
    nsteps = expected_steps(n_pool)
    if any(r.get("steps_in_epoch") != nsteps for r in epochs):
        raise ValueError(f"Incomplete epoch or changed batch behavior: {p}")
    vals = [float(r["dev"]["rsum"]) for r in epochs]
    if not all(math.isfinite(x) for x in vals):
        raise ValueError(f"Nonfinite DEV: {p}")
    best_epoch = max(range(40), key=lambda i: vals[i])
    best = result["done"]["best"]
    if best["epoch"] != best_epoch or best["rsum"] != vals[best_epoch]:
        raise ValueError(f"Best/earliest-tie selection inconsistent: {p}")
    if result.get("full_epochs") != 40 or result["done"].get("epochs_run") != 40:
        raise ValueError(f"Completion marker is inconsistent: {p}")
    if result.get("test_read") is not False:
        raise ValueError(f"Unexpected TEST role: {p}")
    manifest = read_json(p / "run_manifest.json")
    if result["done"]["manifest_sha256"] != sha256_file(p / "run_manifest.json"):
        raise ValueError(f"Done marker refers to a different run manifest: {p}")
    if manifest.get("seed") != identity["seed"] or manifest.get("n_pool") != n_pool:
        raise ValueError(f"Training manifest identity mismatch: {p}")
    if (p / "done.json").exists() and read_json(p / "done.json") != result["done"]:
        raise ValueError(f"Result contains a different done receipt: {p}")
    return {"status": "COMPLETE" if not omitted else "METADATA_COMPLETE_CHECKPOINTS_NOT_VERIFIED",
            "best_epoch": best_epoch, "best_dev_rsum": vals[best_epoch],
            "updates": nsteps * 40, "verified_artifacts": checked,
            "unverified_artifacts": omitted, "hardware": result.get("hardware", {}),
            "result_sha256": sha256_file(p / "full_result.json") if result_override is None else None}


def closeout_census(base, *, check_binary=True, out_path=None):
    base = Path(base).resolve()
    frozen = pinned_freeze(base)
    rows = []
    for group in read_json(base / "GROUPS.json"):
        for item in group["units"]:
            p = base / "results" / group["id"] / f"{item['method']}_{item['scale']}"
            row = {"group": group["id"], "seed": group["seed"], **item, "path": str(p)}
            try:
                row.update(validate_complete_unit(p, expected_identity={
                    "group": group["id"], "seed": group["seed"], "method": item["method"],
                    "scale": item["scale"], "epochs": 40,
                    "protocol_sha256": frozen["code_sha256"]["PROTOCOL_CN.md"],
                    "learning_rate_multiplier": 1.0, "design_rho": 0.4,
                    "freeze_sha256": sha256_file(base / "FROZEN.json")}, check_binary=check_binary))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                row.update(status="INCOMPLETE_OR_INVALID", reason=str(exc))
            rows.append(row)
    if len(rows) != 33 or len({(r['seed'], r['method'], r['scale']) for r in rows}) != 33:
        raise ValueError("The inherited N2F census must have 33 distinct units")
    complete = sum(r["status"] == "COMPLETE" for r in rows)
    metadata_complete = sum(r["status"].startswith(("COMPLETE", "METADATA_COMPLETE")) for r in rows)
    report = {"schema_version": 1, "created_utc": utc_now(), "base": str(base),
              "base_freeze_sha256": sha256_file(base / "FROZEN.json"), "expected": 33,
              "complete": complete, "metadata_complete": metadata_complete,
              "checkpoints_checked": check_binary, "ready_for_new_training": complete == 33,
              "units": rows, "note": "Rebuilt from actual result/log/artifact identities, not FULL_UNITS.csv status text."}
    if out_path:
        write_json(out_path, report)
    return report


def validate_vcs_reference(path, base, seed):
    """Verify treatment and recipe, in addition to the generic artifact census."""
    from .runner import check_config
    p = Path(path)
    frozen = pinned_freeze(base)
    result = read_json(p / "full_result.json")
    manifest = read_json(p / "run_manifest.json")
    check_config(manifest["cfg"])
    original_csv = Path(frozen["old_output"]) / "rho0.4" / f"s{seed}" / "weights/VCS_Weight.csv"
    expected = {"seed": seed, "n_pool": 145000, "n_A": 14500, "n_U": 130500, "lr_mult": 1.0,
                "weight_lock_sha256": frozen["input_sha256"][str(original_csv)]}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Historical VCS manifest does not match frozen {key}: {p}")
    for key, value in {"method": "VCS_Weight", "scale": "sum", "seed": seed,
                       "protocol_sha256": frozen["code_sha256"]["PROTOCOL_CN.md"],
                       "freeze_sha256": sha256_file(Path(base) / "FROZEN.json"),
                       "learning_rate_multiplier": 1.0, "design_rho": 0.4}.items():
        if result["identity"].get(key) != value:
            raise ValueError(f"Historical VCS identity does not match {key}: {p}")
    return {"status": "PASS", "weight_sha256": expected["weight_lock_sha256"],
            "run_manifest_sha256": sha256_file(p / "run_manifest.json")}


def resolve_inputs(base, old_output=None, data=None, vocab=None):
    frozen = pinned_freeze(base)
    return {"old_output": str(Path(old_output or frozen["old_output"]).resolve()),
            "data": str(Path(data or frozen["data"]).resolve()),
            "vocab": str(Path(vocab or frozen["vocab"]).resolve()),
            "cache": str(Path(base).resolve() / "source/token_cache")}


def verify_training_inputs(base, inputs, *, seeds=(901, 902, 903)):
    """Never open eval_only labels. Exact small-file hashes; inherited size/mtime for features."""
    frozen = pinned_freeze(base)
    checked = {}
    original = Path(frozen["old_output"])
    actual = Path(inputs["old_output"])
    for seed in seeds:
        for name in ("pairs.observed_pairs.json", "pairs.noise_manifest.json",
                     "weights/VCS_Weight.csv", "weights/VCS_Weight.csv.lock.json"):
            suffix = Path("rho0.4") / f"s{seed}" / name
            expected = frozen["input_sha256"][str(original / suffix)]
            p = actual / suffix
            checked[str(p)] = assert_same_file(p, expected)
    checked[inputs["vocab"]] = assert_same_file(inputs["vocab"], frozen["input_sha256"][frozen["vocab"]])
    cache = Path(inputs["cache"])
    if not cache.is_dir():
        raise FileNotFoundError(f"Frozen token cache is unavailable: {cache}")
    existing = {p.name for p in cache.iterdir() if p.is_file()}
    if existing != set(frozen["cache_names"]):
        raise ValueError(f"Token cache does not exactly match frozen names: {existing}")
    for name in frozen["cache_names"]:
        checked[str(cache / name)] = assert_same_file(cache / name, frozen["code_sha256"]["source/token_cache/" + name])
    if Path(inputs["data"]).name != "f30k_precomp":
        raise ValueError("Relocated data directory must retain name f30k_precomp for the frozen token cache key")
    identities = {}
    for old_path, expected in frozen["data_files"].items():
        p = Path(inputs["data"]) / Path(old_path).name
        st = p.stat()
        found = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
        if any(found[k] != expected[k] for k in found):
            raise ValueError(f"Data size/mtime differ from inherited freeze: {p}")
        identities[str(p)] = found
    return {"status": "PASS", "sha256": checked, "data_size_mtime": identities,
            "data_identity_note": "Inherited N2F identity uses size+mtime for train/dev files; no full feature-content hash was supplied.",
            "eval_only_labels_read": False, "TEST_read": False}
