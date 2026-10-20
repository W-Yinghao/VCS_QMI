"""Frozen Constant-U and image-block Shuffle-VCS controls, without Torch.

Only observed identities and an already locked production weight vector enter
these functions.  Mechanism labels are a separate offline-scoring input and are
rejected here.  The output directory is published only after every artifact has
been written and checked; an existing experiment is never overwritten.
"""
from __future__ import annotations

import csv
import math
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from v9n2m.common import canonical_sha256, read_json, sha256_file, sort_id, utc_now, write_csv, write_json

CONTROL_VERSION = "v9n2m-controls-v1"
OBSERVED_FIELDS = frozenset(("pair_id", "caption_id", "observed_image_id", "role"))
ANCHOR_ROLES = frozenset(("teacher_warmup", "critic_fit", "critic_select", "reference_moments"))
EVALUATOR_FIELDS = frozenset(("original_image_id", "mismatched", "clean", "is_clean", "label", "labels", "eval_only"))
ARMS = ("VCS", "ConstantU", "ShuffleVCS")


def _training_input_path(path):
    path = Path(path).expanduser().resolve()
    if any("eval_only" in part.lower() or "eval-only" in part.lower() for part in path.parts):
        raise ValueError(f"Evaluator-only path cannot enter control generation: {path}")
    return path


def _identifier(value, field, row_number):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"Row {row_number}: {field} must be a string or integer ID")
    value = str(value)
    if not value or value != value.strip() or any(c in value for c in "\r\n\x00"):
        raise ValueError(f"Row {row_number}: malformed {field}")
    return value


def load_observed(path):
    """Return validated observed rows in original order, with IDs as strings.

    The complete schema is deliberate: unexpected fields cannot silently carry
    evaluator labels or caption/image content into a new processing path.
    """
    path = _training_input_path(path)
    rows = read_json(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Observed input must be a nonempty JSON list of observed-pair rows")
    pair_ids, caption_ids, image_roles = set(), set(), {}
    result = []
    for k, row in enumerate(rows, 1):
        if not isinstance(row, dict) or set(row) != OBSERVED_FIELDS:
            fields = set(row) if isinstance(row, dict) else set()
            raise ValueError(f"Observed row {k} has unexpected/missing fields: "
                             f"extra={sorted(fields - OBSERVED_FIELDS)}, missing={sorted(OBSERVED_FIELDS - fields)}; "
                             "evaluator labels are forbidden")
        item = {name: _identifier(row[name], name, k) for name in OBSERVED_FIELDS - {"role"}}
        if not isinstance(row["role"], str) or row["role"] not in ANCHOR_ROLES | {"U"}:
            raise ValueError(f"Observed row {k} has unsupported role {row['role']!r}")
        item["role"] = row["role"]
        for field, seen in (("pair_id", pair_ids), ("caption_id", caption_ids)):
            if item[field] in seen:
                raise ValueError(f"Duplicate {field} in observed rows: {item[field]}")
            seen.add(item[field])
        image_id = item["observed_image_id"]
        if image_id in image_roles and image_roles[image_id] != item["role"]:
            raise ValueError(f"Observed image {image_id} crosses roles; frozen A/U roles must remain disjoint")
        image_roles[image_id] = item["role"]
        result.append(item)
    return result


def load_weights(path, expected_ids=None, lock_path=None):
    """Return a strict ``{pair_id: float64-compatible weight}`` mapping.

    ``lock_path`` enables content-hash and row-count checks.  Original score,
    rho_hat, fit_hash and reference_hash columns are allowed and not used to
    choose controls.  Duplicate IDs, missing/extra rows and label fields fail.
    """
    path = _training_input_path(path)
    lock = None
    if lock_path is not None:
        lock_path = _training_input_path(lock_path)
        lock = read_json(lock_path)
        if not isinstance(lock, dict) or not isinstance(lock.get("sha256"), str):
            raise ValueError("Weight lock must contain its source CSV sha256")
        if sha256_file(path) != lock["sha256"]:
            raise ValueError(f"Weight CSV SHA256 does not match lock: {path}")
        if isinstance(lock.get("n"), bool) or not isinstance(lock.get("n"), int) or lock["n"] <= 0:
            raise ValueError("Weight lock must contain a positive integer row count n")
    result = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not {"pair_id", "w"}.issubset(fields):
            raise ValueError("Weight CSV must have unique columns including pair_id,w")
        leaked = EVALUATOR_FIELDS.intersection(name.lower() for name in fields)
        if leaked:
            raise ValueError(f"Weight CSV carries forbidden evaluator columns: {sorted(leaked)}")
        for k, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Weight CSV row {k} has a different column count from its header")
            pair_id = _identifier(row["pair_id"], "pair_id", k)
            if pair_id in result:
                raise ValueError(f"Duplicate pair_id in weight CSV: {pair_id}")
            try:
                weight = float(row["w"])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid weight for {pair_id}") from exc
            if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
                raise ValueError(f"Weight for {pair_id} must be finite in [0,1]")
            result[pair_id] = weight
    if not result:
        raise ValueError("Weight CSV must not be empty")
    if lock is not None and len(result) != lock["n"]:
        raise ValueError("Weight CSV row count does not match lock n")
    if expected_ids is not None:
        expected_ids = [_identifier(value, "expected pair_id", k) for k, value in enumerate(expected_ids, 1)]
        expected = set(expected_ids)
        if len(expected) != len(expected_ids):
            raise ValueError("Expected pair IDs contain duplicates")
        actual = set(result)
        if actual != expected:
            raise ValueError(f"Weight ID coverage mismatch: missing={len(expected - actual)}, "
                             f"extra={len(actual - expected)}; "
                             f"examples missing={sorted(expected - actual)[:5]}, extra={sorted(actual - expected)[:5]}")
    return result


def _stats(weights, n_a):
    weights = np.asarray(weights, dtype=np.float64)
    mass = float(np.sum(weights, dtype=np.float64))
    square_mass = float(np.sum(weights * weights, dtype=np.float64))
    return {"n_U": int(len(weights)), "total_mass_U": mass,
            "mean_w_U": float(np.mean(weights, dtype=np.float64)),
            "ess_U": mass * mass / square_mass if square_mass > 0 else 0.0,
            "mean_v_full_pool": (int(n_a) + mass) / (int(n_a) + len(weights)),
            "min_w_U": float(np.min(weights)), "max_w_U": float(np.max(weights)),
            "anchor_weight": 1.0, "n_A": int(n_a)}


def _roundoff_equal(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def _validate_invariants(original, constant, shuffled, n_a):
    original, constant, shuffled = [np.asarray(x, dtype=np.float64) for x in (original, constant, shuffled)]
    stats = {arm: _stats(w, n_a) for arm, w in zip(ARMS, (original, constant, shuffled))}
    mu = float(np.mean(original, dtype=np.float64))
    if not np.all(constant == mu):
        raise ValueError("Constant-U weights are not exactly the float64 production mean")
    if not np.array_equal(np.sort(original), np.sort(shuffled)):
        raise ValueError("Shuffle-VCS does not preserve the exact float64 weight multiset")
    for arm in ARMS[1:]:
        for field in ("total_mass_U", "mean_v_full_pool"):
            if not _roundoff_equal(stats[arm][field], stats["VCS"][field]):
                raise ValueError(f"{arm} failed mass/normalizer invariant: {field}")
    if not _roundoff_equal(stats["ShuffleVCS"]["ess_U"], stats["VCS"]["ess_U"]):
        raise ValueError("Shuffle-VCS failed ESS invariant")
    # Zero total mass has ESS 0/0 and is conventionally reported as zero, not n.
    expected_constant_ess = float(len(constant)) if mu > 0 else 0.0
    if not _roundoff_equal(stats["ConstantU"]["ess_U"], expected_constant_ess):
        raise ValueError("Constant-U has unexpected ESS")
    return stats


def _block_matrix(u_rows, weight_map, block_size):
    groups = defaultdict(list)
    for row in u_rows:
        groups[row["observed_image_id"]].append(row)
    wrong = {image_id: len(rows) for image_id, rows in groups.items() if len(rows) != block_size}
    if wrong:
        examples = [(k, wrong[k]) for k in sorted(wrong, key=sort_id)[:8]]
        raise ValueError(f"Observed-image U blocks must each have {block_size} rows; "
                         f"{len(wrong)} blocks differ, examples={examples}. "
                         "No silent pair-shuffle fallback is permitted.")
    image_ids = sorted(groups, key=sort_id)
    ordered_rows = [sorted(groups[image_id], key=lambda row: sort_id(row["caption_id"])) for image_id in image_ids]
    matrix = np.asarray([[weight_map[row["pair_id"]] for row in rows] for rows in ordered_rows], dtype=np.float64)
    return image_ids, ordered_rows, matrix


def _pearson(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    aa, bb = a - a.mean(), b - b.mean()
    norm = float(np.sqrt(np.dot(aa, aa) * np.dot(bb, bb)))
    return float(np.dot(aa, bb) / norm) if norm > 0 else None


def _unchanged_sources(sources):
    for item in sources.values():
        if sha256_file(item["path"]) != item["sha256"]:
            raise ValueError(f"Source input changed during control generation: {item['path']}")


def _check_existing(out, request, expected_arrays, pair_ids, permutation, n_a):
    manifest = read_json(out / "manifest.json")
    if manifest.get("request_sha256") != canonical_sha256(request) or manifest.get("request") != request:
        raise ValueError("Existing controls have a different source/configuration identity; use a new output directory")
    original, constant, shuffled = [], [], []
    for arm, destination in zip(ARMS, (original, constant, shuffled)):
        entry = manifest.get("arms", {}).get(arm, {})
        csv_path, lock_path = out / f"{arm}.csv", out / f"{arm}.csv.lock.json"
        if entry.get("csv") != str(csv_path) or entry.get("lock") != str(lock_path):
            raise ValueError("Existing control manifest has an unexpected artifact path")
        if sha256_file(csv_path) != entry.get("sha256") or sha256_file(lock_path) != entry.get("lock_sha256"):
            raise ValueError(f"Existing {arm} artifact differs from its manifest")
        weights = load_weights(csv_path, pair_ids, lock_path)
        destination.extend(weights[pair_id] for pair_id in pair_ids)
        if not np.array_equal(np.asarray(destination, dtype=np.float64), expected_arrays[arm]):
            raise ValueError(f"Existing {arm} weights differ from the frozen control definition")
    stats = _validate_invariants(original, constant, shuffled, n_a)
    expected_meta = {"seed": request["seed"], "n_pool": request["expected_n_pool"],
                     "n_U": request["expected_n_u"], "n_A": n_a,
                     "mean_v_full_pool": stats["VCS"]["mean_v_full_pool"],
                     "mu_float64": stats["VCS"]["mean_w_U"], "sources": request["sources"],
                     "label_fields_read": False, "random_permutation_selected_using_labels": False}
    for field, expected in expected_meta.items():
        if manifest.get(field) != expected:
            raise ValueError(f"Existing controls manifest has inconsistent {field}")
    for arm in ARMS:
        for field, expected in stats[arm].items():
            if manifest["arms"][arm].get(field) != expected:
                raise ValueError(f"Existing {arm} manifest has inconsistent {field}")
    permutation_path = out / "permutation.npy"
    saved = np.load(permutation_path, allow_pickle=False)
    if saved.dtype != np.dtype("int64") or not np.array_equal(saved, permutation):
        raise ValueError("Saved permutation differs from the prescribed one-shot PCG64 permutation")
    if sha256_file(permutation_path) != manifest.get("permutation", {}).get("sha256"):
        raise ValueError("Saved permutation SHA256 differs from the manifest")
    permutation_meta = {"file": str(permutation_path), "seed_sequence": request["seed_sequence"],
                        "block_size": request["block_size"], "n_blocks": len(permutation),
                        "fixed_points_allowed": True,
                        "n_fixed_points": int(np.sum(permutation == np.arange(len(permutation)))),
                        "fixed_point_fraction": float(np.mean(permutation == np.arange(len(permutation))))}
    for field, expected in permutation_meta.items():
        if manifest["permutation"].get(field) != expected:
            raise ValueError(f"Existing permutation manifest has inconsistent {field}")
    return manifest


def generate_controls(observed_path, vcs_csv, vcs_lock, seed, out_dir, *,
                      expected_n_pool=145000, expected_n_u=130500, block_size=5):
    """Prepare all three arms, returning the immutable artifact manifest.

    Production defaults enforce 145000 training pairs, 130500 U pairs and five
    rows per observed-image block.  Smaller counts are permitted solely for
    explicit synthetic implementation checks; the experiment freeze enforces
    the production protocol.  Existing outputs are verified and returned
    unchanged; changed inputs or tampered outputs fail rather than overwrite.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    for name, value in (("expected_n_pool", expected_n_pool), ("expected_n_u", expected_n_u), ("block_size", block_size)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    expected_n_pool, expected_n_u, block_size = int(expected_n_pool), int(expected_n_u), int(block_size)
    if expected_n_u >= expected_n_pool:
        raise ValueError("The full pool must include trusted A rows as well as U rows")
    observed_path, vcs_csv, vcs_lock = map(_training_input_path, (observed_path, vcs_csv, vcs_lock))
    sources = {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in
               (("observed", observed_path), ("vcs_csv", vcs_csv), ("vcs_lock", vcs_lock))}
    out = Path(out_dir).expanduser().absolute()
    if out.is_symlink():
        raise ValueError("Controls output must not be a symlink")
    out = out.resolve()
    for source in (observed_path, vcs_csv, vcs_lock):
        if out == source or out in source.parents:
            raise ValueError("Control output directory cannot contain or replace a source input")
    rows = load_observed(observed_path)
    u_rows = [row for row in rows if row["role"] == "U"]
    if len(rows) != expected_n_pool or len(u_rows) != expected_n_u:
        raise ValueError(f"Observed counts disagree with declared protocol: "
                         f"pool={len(rows)} (expected {expected_n_pool}), U={len(u_rows)} (expected {expected_n_u})")
    pair_ids = [row["pair_id"] for row in u_rows]
    weights = load_weights(vcs_csv, pair_ids, vcs_lock)
    parent_lock = read_json(vcs_lock)
    record = parent_lock.get("record")
    if not isinstance(record, dict) or record.get("method") != "VCS_Weight":
        raise ValueError("Source lock must identify the original production VCS_Weight vector")
    if record.get("deployable") is False or record.get("rho_strategy", "estimated") != "estimated":
        raise ValueError("Controls require deployable estimated-rho production VCS weights, not an oracle diagnostic")
    image_ids, ordered_rows, matrix = _block_matrix(u_rows, weights, block_size)
    permutation = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 20261019, 9101]))).permutation(len(image_ids)).astype(np.int64)
    shuffled_map = {row["pair_id"]: float(matrix[int(permutation[k]), j])
                    for k, block in enumerate(ordered_rows) for j, row in enumerate(block)}
    original = np.asarray([weights[pair_id] for pair_id in pair_ids], dtype=np.float64)
    mu = float(np.mean(original, dtype=np.float64))
    arrays = {"VCS": original, "ConstantU": np.full(len(original), mu, dtype=np.float64),
              "ShuffleVCS": np.asarray([shuffled_map[pair_id] for pair_id in pair_ids], dtype=np.float64)}
    n_a = len(rows) - len(u_rows)
    stats = _validate_invariants(arrays["VCS"], arrays["ConstantU"], arrays["ShuffleVCS"], n_a)
    _unchanged_sources(sources)
    if parent_lock["sha256"] != sources["vcs_csv"]["sha256"]:
        raise ValueError("Production lock identity changed while reading source weights")
    request = {"schema_version": 1, "control_definition_version": CONTROL_VERSION,
               "generator_source_sha256": sha256_file(__file__), "seed": seed,
               "expected_n_pool": expected_n_pool, "expected_n_u": expected_n_u, "block_size": block_size,
               "sources": sources, "seed_sequence": [seed, 20261019, 9101]}
    if (out / "manifest.json").is_file():
        return _check_existing(out, request, arrays, pair_ids, permutation, n_a)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("Controls output is nonempty without a completed manifest; use a fresh output directory")
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=out.name + ".building.", dir=out.parent))
    try:
        np.save(stage / "permutation.npy", permutation, allow_pickle=False)
        permutation_record = {"file": str(out / "permutation.npy"), "sha256": sha256_file(stage / "permutation.npy"),
                              "rng": "numpy.random.Generator(PCG64(SeedSequence([seed,20261019,9101])))",
                              "seed_sequence": [seed, 20261019, 9101], "numpy_version": np.__version__,
                              "definition": "target block k receives W[permutation[k], :]; one draw, frozen for all epochs",
                              "block_size": block_size, "n_blocks": len(image_ids), "fixed_points_allowed": True,
                              "image_id_order": "integer IDs first by numeric value (string tie break), then other IDs lexicographically",
                              "within_block_order": "caption_id, using the same total ordering as image IDs",
                              "image_ids_sha256": canonical_sha256(image_ids),
                              "ordered_pair_ids_sha256": canonical_sha256([[r["pair_id"] for r in block] for block in ordered_rows]),
                              "n_fixed_points": int(np.sum(permutation == np.arange(len(permutation)))),
                              "fixed_point_fraction": float(np.mean(permutation == np.arange(len(permutation)))),
                              "weight_pearson_r": _pearson(arrays["VCS"], arrays["ShuffleVCS"]),
                              "dependence_scope": "preserves observed-image weight blocks, not every caption-origin dependency"}
        arms = {}
        for arm in ARMS:
            csv_path, lock_path = stage / f"{arm}.csv", stage / f"{arm}.csv.lock.json"
            if arm == "VCS":
                shutil.copyfile(vcs_csv, csv_path)
            else:
                write_csv(csv_path, ({"pair_id": pair_id, "w": format(float(w), ".17g")}
                                     for pair_id, w in zip(pair_ids, arrays[arm])), ["pair_id", "w"])
            digest = sha256_file(csv_path)
            arm_record = {"method": "VCS_Weight" if arm == "VCS" else arm, "arm": arm,
                          "control_definition_version": CONTROL_VERSION, "seed": seed,
                          "production_vcs_csv_sha256": sources["vcs_csv"]["sha256"],
                          "production_vcs_lock_sha256": sources["vcs_lock"]["sha256"],
                          "observed_sha256": sources["observed"]["sha256"], **stats[arm],
                          "total_mass": stats[arm]["total_mass_U"], "mean_w": stats[arm]["mean_w_U"],
                          "ess": stats[arm]["ess_U"], "n_U": len(u_rows), "label_fields_read": False,
                          "anchor_rule": "A weight is exactly 1 in the trainer; CSV contains only U rows"}
            if arm == "VCS":
                arm_record["production_record"] = record
                arm_record["source_csv_copied_byte_for_byte"] = True
            elif arm == "ConstantU":
                arm_record["mu_float64"] = mu
                arm_record["ess_note"] = "n_U for positive mass; zero for all-zero weights (undefined ESS reported as 0)"
            else:
                arm_record["permutation_sha256"] = permutation_record["sha256"]
            write_json(lock_path, {"file": str(out / csv_path.name), "sha256": digest, "n": len(u_rows), "record": arm_record})
            reread = load_weights(csv_path, pair_ids, lock_path)
            if not np.array_equal(arrays[arm], np.asarray([reread[p] for p in pair_ids], dtype=np.float64)):
                raise ValueError(f"{arm} CSV serialization did not preserve float64 values")
            arms[arm] = {"csv": str(out / csv_path.name), "lock": str(out / lock_path.name),
                         "sha256": digest, "lock_sha256": sha256_file(lock_path), **stats[arm]}
        manifest = {"schema_version": 1, "control_definition_version": CONTROL_VERSION,
                    "created_utc": utc_now(), "request": request, "request_sha256": canonical_sha256(request),
                    "seed": seed, "n_pool": len(rows), "n_A": n_a, "n_U": len(u_rows),
                    "mean_v_full_pool": stats["VCS"]["mean_v_full_pool"], "mu_float64": mu,
                    "sources": sources, "arms": arms, "permutation": permutation_record,
                    "invariants": {"id_coverage_exact": True, "constant_float64_mean_exact": True,
                                   "constant_mass_preserved_within_float64_roundoff": True,
                                   "shuffle_float64_multiset_exact": True,
                                   "shuffle_mass_ess_preserved_within_float64_roundoff": True,
                                   "roundoff_relative_tolerance": 1e-12, "roundoff_absolute_tolerance": 1e-12,
                                   "not_a_performance_gate": True},
                    "label_fields_read": False, "random_permutation_selected_using_labels": False}
        write_json(stage / "manifest.json", manifest)
        _unchanged_sources(sources)
        # Publish only a complete directory.  A concurrent publisher is never
        # replaced, and a failed build leaves the source and prior outputs intact.
        if out.exists():
            if not out.is_dir() or any(out.iterdir()):
                raise ValueError("Controls output changed while preparing; refusing to overwrite it")
            out.rmdir()
        os.rename(stage, out)
        return _check_existing(out, request, arrays, pair_ids, permutation, n_a)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
