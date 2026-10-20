"""Evaluator-only, content-free N2M probability and retained-mass diagnostics.

This module never creates training weights. Mechanism labels describe the injected
mapping, not the semantic compatibility of a caption with the observed image.
"""
from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path

import numpy as np

from .common import ARMS, read_json, sha256_file, utc_now, write_csv, write_json

EPS = 1e-12
BIN_EDGES = np.linspace(0.0, 1.0, 11)


def _vector(x, name):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must be a nonempty finite vector")
    return x


def probability_metrics(clean, weights):
    """Clean=1; grouped ties for ROC AUC and noninterpolated average precision."""
    y, w = _vector(clean, "clean"), _vector(weights, "weights")
    if y.shape != w.shape or np.any((y != 0) & (y != 1)) or np.any((w < 0) | (w > 1)):
        raise ValueError("labels must be binary and weights in [0,1], with matching shapes")
    n, positives = len(y), int(y.sum())
    negatives = n - positives
    auc, ap, auc_reason, ap_reason = None, None, None, None
    # Mann-Whitney credit: all positive-negative ties count one half.
    order = np.argsort(w, kind="stable")
    sy, sw = y[order], w[order]
    ends = np.r_[np.flatnonzero(sw[1:] != sw[:-1]) + 1, n]
    previous, lower_negatives, credit = 0, 0, 0.0
    groups = []
    for end in ends:
        count = int(end - previous)
        pos = int(sy[previous:end].sum())
        neg = count - pos
        credit += pos * (lower_negatives + 0.5 * neg)
        lower_negatives += neg
        groups.append((count, pos))
        previous = int(end)
    if positives and negatives:
        auc = float(credit / (positives * negatives))
    else:
        auc_reason = "AUROC undefined: mechanism labels contain one class"
    if positives:
        cumulative_count, cumulative_positive, area = 0, 0, 0.0
        for count, pos in reversed(groups):
            cumulative_count += count
            cumulative_positive += pos
            area += (pos / positives) * (cumulative_positive / cumulative_count)
        ap = float(area)
    else:
        ap_reason = "Average precision undefined: no clean positive examples"
    clipped = np.clip(w, EPS, 1.0 - EPS)
    bins = np.minimum((w * 10).astype(int), 9)
    calibration, ece = [], 0.0
    for k in range(10):
        mask = bins == k
        count = int(mask.sum())
        pw = float(w[mask].mean()) if count else None
        observed = float(y[mask].mean()) if count else None
        gap = abs(pw - observed) if count else None
        if count:
            ece += count / n * gap
        calibration.append({"bin": k, "left": float(BIN_EDGES[k]), "right": float(BIN_EDGES[k + 1]),
                            "right_inclusive": k == 9, "n": count, "mean_weight": pw,
                            "clean_fraction": observed, "absolute_gap": gap})
    mass = float(w.sum(dtype=np.float64))
    squares = float(np.dot(w, w))
    cm = float(w[y == 1].sum(dtype=np.float64))
    nm = float(w[y == 0].sum(dtype=np.float64))
    return {
        "n": n, "n_clean": positives, "n_mismatched": negatives,
        "positive_label": "clean = not mismatched", "realized_rho": negatives / n,
        "mean_weight": mass / n, "total_mass": mass,
        "ess": mass * mass / squares if squares else 0.0,
        "brier": float(np.mean((w - y) ** 2)),
        "logloss": float(-np.mean(y * np.log(clipped) + (1.0 - y) * np.log1p(-clipped))),
        "logloss_clip_epsilon": EPS, "logloss_n_clipped": int(np.count_nonzero(clipped != w)),
        "auroc": auc, "auroc_unavailable_reason": auc_reason,
        "auprc_average_precision": ap, "auprc_unavailable_reason": ap_reason,
        "auprc_definition": "noninterpolated average precision; equal scores grouped at one threshold",
        "clean_retained_mass": cm, "mismatched_retained_mass": nm,
        "clean_mean_retention": cm / positives if positives else None,
        "mismatched_mean_retention": nm / negatives if negatives else None,
        "residual_noise_mass_fraction": nm / mass if mass else None,
        "retained_clean_mass_fraction": cm / mass if mass else None,
        "calibration_ece_10_equal_width": float(ece), "calibration": calibration,
        "inference_unit": "descriptive U pairs; no pair-IID confidence interval",
    }


def _observed_and_labels(observed_path, labels_path, noise_manifest_path=None):
    # Importing the training-safe observed loader does not import this evaluator
    # into any training path. It also checks that the observed file is label-free.
    from .controls import load_observed
    rows = load_observed(observed_path)
    labels = read_json(labels_path)
    if not isinstance(labels, dict):
        raise ValueError("EVAL_ONLY labels must be a dictionary keyed by pair_id")
    ids = [str(r["pair_id"]) for r in rows]
    if set(labels) != set(ids):
        raise ValueError("Evaluator labels must cover exactly the observed pool's pair IDs")
    for row in rows:
        lab = labels[str(row["pair_id"])]
        if not isinstance(lab, dict) or type(lab.get("mismatched")) is not bool or "original_image_id" not in lab:
            raise ValueError("Each label needs original_image_id and a JSON boolean mismatched")
        actual = str(row["observed_image_id"]) != str(lab["original_image_id"])
        if lab["mismatched"] != actual:
            raise ValueError(f"Mechanism label disagrees with observed mapping: {row['pair_id']}")
        if row["role"] != "U" and actual:
            raise ValueError("An anchor is mismatched; wrong label/observed mapping")
    urows = [r for r in rows if r["role"] == "U"]
    if not urows:
        raise ValueError("Observed table has no U pairs")
    clean = np.array([not labels[str(r["pair_id"])] ["mismatched"] for r in urows], dtype=np.float64)
    manifest = None
    if noise_manifest_path is not None:
        manifest = read_json(noise_manifest_path)
        for key, path in (("observed_sha256", observed_path), ("eval_only_sha256", labels_path)):
            if key in manifest and manifest[key] != sha256_file(path):
                raise ValueError(f"Noise manifest {key} mismatch")
        mapping = hashlib.sha256("\n".join(f"{r['pair_id']}\t{r['observed_image_id']}" for r in rows).encode()).hexdigest()
        if "mapping_sha256" in manifest and manifest["mapping_sha256"] != mapping:
            raise ValueError("Noise manifest observed mapping hash mismatch")
        n_mis = int((1 - clean).sum())
        expected = {"n_pairs": len(rows), "n_U_pairs": len(urows), "n_mismatched": n_mis,
                    "n_A_mismatched": 0}
        for key, value in expected.items():
            if key in manifest and manifest[key] != value:
                raise ValueError(f"Noise manifest {key} differs from evaluator reconstruction")
        if "realized_rho" in manifest and not math.isclose(float(manifest["realized_rho"]), n_mis / len(urows), rel_tol=0, abs_tol=1e-12):
            raise ValueError("Noise manifest realized_rho mismatch")
    return rows, urows, labels, clean, manifest


def _weight_table(path, lock_path, expected_ids):
    # Local strict reader keeps the scorer independent of the generator's return
    # representation and retains original score strings for reconstruction.
    from .controls import load_weights
    load_weights(path, expected_ids=expected_ids, lock_path=lock_path)
    lock = read_json(lock_path)
    if lock.get("sha256") != sha256_file(path):
        raise ValueError(f"Weight SHA256 does not match lock: {path}")
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"pair_id", "w"}.issubset(reader.fieldnames or []):
            raise ValueError("Weight CSV requires pair_id,w")
        entries = {}
        for row in reader:
            pid = str(row["pair_id"])
            if pid in entries:
                raise ValueError(f"Duplicate weight pair ID: {pid}")
            entries[pid] = row
    if set(entries) != set(expected_ids) or len(entries) != lock.get("n"):
        raise ValueError("Weight lock/CSV/U pair-ID coverage mismatch")
    w = _vector([float(entries[pid]["w"]) for pid in expected_ids], "weights")
    if np.any((w < 0) | (w > 1)):
        raise ValueError("Weights outside [0,1]")
    return w, lock, entries


def _posterior_from_t(t, rho):
    t = np.asarray(t, dtype=np.float64)
    if not 0 <= rho <= 1 or np.any(abs(t) > 1):
        raise ValueError("Invalid critic values or rho")
    # The frozen production function explicitly returns ones when rho <= 0.
    if rho == 0:
        return np.ones_like(t)
    num = (1.0 - rho) * (1.0 + t)
    den = num + rho * (1.0 - t)
    if np.any(den <= 0):
        raise ValueError("Posterior is undefined at a zero-mixture-support point")
    return num / den


def score_weights(observed_path, labels_path, vcs_csv, vcs_lock, controls_dir, out_dir, *, noise_manifest_path=None):
    """Score already-frozen controls. Label reads happen only in this function."""
    out, control_root = Path(out_dir), Path(controls_dir)
    control_manifest_path = control_root / "manifest.json"
    if not control_manifest_path.exists():
        raise ValueError("Controls must be generated and frozen before evaluator labels are read")
    cm = read_json(control_manifest_path)
    sources = cm.get("sources", {})
    for name, path in (("observed", observed_path), ("vcs_csv", vcs_csv), ("vcs_lock", vcs_lock)):
        if sources.get(name, {}).get("sha256") != sha256_file(path):
            raise ValueError(f"Frozen controls source identity differs: {name}")
    # Verify every frozen control's bytes and lock before opening evaluator labels.
    from .controls import load_observed
    observed = load_observed(observed_path)
    ids = [str(r["pair_id"]) for r in observed if r["role"] == "U"]
    source_w, lock, entries = _weight_table(vcs_csv, vcs_lock, ids)
    frozen_arms = {}
    for arm in ARMS:
        item = cm.get("arms", {}).get(arm, {})
        csv_path = Path(item.get("csv", control_root / f"{arm}.csv"))
        lock_path = Path(item.get("lock", str(csv_path) + ".lock.json"))
        if item.get("sha256") != sha256_file(csv_path) or item.get("lock_sha256") != sha256_file(lock_path):
            raise ValueError(f"Frozen control CSV/lock identity differs: {arm}")
        frozen_arms[arm] = (csv_path, lock_path, *_weight_table(csv_path, lock_path, ids))
    rows, urows, labels, clean, noise = _observed_and_labels(observed_path, labels_path, noise_manifest_path)
    arms = {}
    hashes = {}
    for arm in ARMS:
        csv_path, lock_path, w, arm_lock, _ = frozen_arms[arm]
        if arm == "VCS" and not np.array_equal(w, source_w):
            raise ValueError("Control VCS differs from the locked production weight vector")
        metrics = probability_metrics(clean, w)
        n_a = len(rows) - len(urows)
        metrics.update({"n_A": n_a, "n_pool": len(rows),
                        "mean_v_full_pool": (n_a + metrics["total_mass"]) / len(rows),
                        "anchor_fraction_of_retained_full_mass": n_a / (n_a + metrics["total_mass"])})
        arms[arm] = metrics
        hashes[arm] = {"csv_sha256": sha256_file(csv_path), "lock_sha256": sha256_file(lock_path)}
    record = lock.get("record", {})
    rho_record = record.get("rho", {})
    if not isinstance(rho_record, dict):
        rho_record = {}
    rho_used = record.get("rho_used", rho_record.get("projected"))
    critic = record.get("critic", {})
    oracle = {"available": False, "unavailable_reason": None, "diagnostic_only": True,
              "deployable": False, "same_critic_no_refit": True}
    consistency = {"available": False, "unavailable_reason": None}
    scores_present = all(entries[pid].get("score") not in (None, "") for pid in ids)
    if not scores_present:
        oracle["unavailable_reason"] = consistency["unavailable_reason"] = "Locked source CSV lacks teacher score for at least one U pair"
    elif not all(key in critic for key in ("a", "b")):
        oracle["unavailable_reason"] = consistency["unavailable_reason"] = "Source lock lacks scalar critic a,b; no critic fitted by scorer"
    else:
        score = _vector([float(entries[pid]["score"]) for pid in ids], "teacher scores")
        a, b = float(critic["a"]), float(critic["b"])
        if not np.isfinite(a) or not np.isfinite(b) or a < 0:
            raise ValueError("Scalar critic must have finite a>=0 and b")
        t = np.tanh(a * score + b)
        try:
            ow = _posterior_from_t(t, float(1.0 - clean.mean()))
            oracle.update({"available": True, "rho_used": float(1.0 - clean.mean()),
                           "metrics": probability_metrics(clean, ow),
                           "T_reconstruction": "tanh(a * locked_CSV_score + b)",
                           "n_T_saturated": int(np.count_nonzero(abs(t) == 1)),
                           "difference_from_production": {"mean_absolute_weight": float(np.mean(abs(ow - source_w))),
                                                           "brier": float(np.mean((ow - clean) ** 2) - arms["VCS"]["brier"])}})
        except ValueError as exc:
            oracle["unavailable_reason"] = str(exc)
        if rho_used is None or not math.isfinite(float(rho_used)):
            consistency["unavailable_reason"] = "Source lock lacks a finite rho_used/projected"
        else:
            rw = _posterior_from_t(t, float(rho_used))
            # Original writer rounds both scores and weights to 10 decimals.
            # |dw/ds| = 2*a*w*(1-w) <= a/2, plus weight rounding and fp64 roundoff.
            tol = 5e-11 * (1 + abs(a) / 2) + 128 * np.finfo(np.float64).eps
            errors = abs(rw - source_w)
            consistency = {"available": True, "rho_used_from": "exact JSON lock record, not rounded CSV rho_hat",
                           "rho_used": float(rho_used), "absolute_tolerance": float(tol),
                           "tolerance_basis": "5e-11 weight rounding + (a/2)*5e-11 score rounding + 128 fp64 eps",
                           "max_absolute_error": float(errors.max()), "n_outside_tolerance": int(np.count_nonzero(errors > tol)),
                           "consistent": bool(np.all(errors <= tol)),
                           "rounded_score_and_weight_not_original_full_precision": True}
            # Inversion is poorly conditioned near posterior boundaries. Compare
            # the score-derived T with the interval induced by CSV rounding;
            # never use a single universal T tolerance or clipped inverse.
            r = float(rho_used)
            if 0 < r < 1:
                def inverse(w):
                    return (w - (1-r)) / ((1-r) - (1-2*r)*w)
                lower_w = np.clip(source_w - 5e-11, 0, 1)
                upper_w = np.clip(source_w + 5e-11, 0, 1)
                lower_t, upper_t = inverse(lower_w), inverse(upper_w)
                score_t_bound = abs(a) * 5e-11 + 128 * np.finfo(np.float64).eps
                inside = (t >= lower_t - score_t_bound) & (t <= upper_t + score_t_bound)
                consistency["inverse_T_interval_check"] = {"available": True, "n_outside_rounding_interval": int((~inside).sum()),
                                                           "all_inside": bool(inside.all()), "score_T_tolerance": float(score_t_bound)}
            else:
                consistency["inverse_T_interval_check"] = {"available": False, "unavailable_reason": "rho is at a boundary; posterior does not identify T"}
    report = {
        "schema_version": 1, "analysis": "V9 N2M EVAL_ONLY mechanism diagnostics", "created_utc": utc_now(),
        "training_label_access": False, "evaluator_label_access": True, "test_read": False,
        "labels_are": "Injected mapping mechanism only; not semantic truth labels",
        "source": {"observed_sha256": sha256_file(observed_path), "labels_sha256": sha256_file(labels_path),
                   "vcs_csv_sha256": sha256_file(vcs_csv), "vcs_lock_sha256": sha256_file(vcs_lock),
                   "controls_manifest_sha256": sha256_file(control_manifest_path),
                   "noise_manifest_sha256": sha256_file(noise_manifest_path) if noise_manifest_path else None,
                   "arm_hashes": hashes},
        "rho": {"raw": rho_record.get("raw"), "projected": rho_record.get("projected"), "used": rho_used,
                "realized": float(1-clean.mean()), "design": noise.get("design_rho") if noise else None,
                "moments": {k: rho_record.get(k) for k in ("a", "b", "c", "denominator", "status", "n_P", "n_Q", "n_U")},
                "original_record": rho_record,
                "interval_interpretation": "Diagnostic only: P/Q share reference images, original Q cluster IDs and role-prefix checks do not establish independence, and U donor links can cross the original estimation/held-out split; no confirmed coverage claim."},
        "arms": arms, "production_reconstruction": consistency, "same_T_oracle_rho": oracle,
        "inference": {"unit": "seed-level experiment; pair-level values are descriptive",
                      "confidence_intervals": None, "p_values": None,
                      "no_metric_threshold_blocks_training": True,
                      "calibration_bins_frozen": "10 equal-width bins on [0,1]; left-inclusive and last bin includes 1"},
        "warnings": ["Teacher scores and production weights in the source CSV were rounded to 10 decimals; reconstructed critic outputs and oracle-rho weights are diagnostics.",
                     "Labels are read after the controls have already been frozen; the scorer never writes a training control or selects a shuffle.",
                     "Fixed receiving-image block shuffling preserves five-row blocks but not every dependency involving original caption images."],
    }
    if consistency.get("available") and not consistency.get("consistent"):
        report["warnings"].append("Production score/critic/rho reconstruction is inconsistent beyond known rounding; inspect identity and numerical provenance before interpreting this diagnostic.")
    write_json(out / "offline_metrics.json", report)
    table = []
    calibration = []
    for arm, metrics in arms.items():
        table.append({"arm": arm, **{k: metrics[k] for k in ("n", "n_clean", "n_mismatched", "mean_weight", "ess", "brier", "logloss", "auroc", "auprc_average_precision", "clean_mean_retention", "mismatched_mean_retention", "residual_noise_mass_fraction", "mean_v_full_pool")}})
        calibration.extend({"arm": arm, **row} for row in metrics["calibration"])
    if oracle["available"]:
        metrics = oracle["metrics"]
        table.append({"arm": "VCS_same_T_oracle_rho_EVAL_ONLY", **{k: metrics.get(k) for k in table[0] if k != "arm"}})
        calibration.extend({"arm": "VCS_same_T_oracle_rho_EVAL_ONLY", **row} for row in metrics["calibration"])
    write_csv(out / "offline_metrics.csv", table)
    write_csv(out / "calibration_10bins.csv", calibration)
    text = ["# V9 N2M 离线机制诊断", "", "标签含义：clean = 未被注入机制实际错配；不等同于语义兼容性真值。以下是开发性、描述性结果，没有 pair-IID 区间或性能门控。", "",
            f"U 共 {len(urows)} 对，实际错配率 {report['rho']['realized']:.9f}；估计 rho 的原始值/投影值为 {report['rho']['raw']} / {report['rho']['projected']}。", "",
            "| arm | mean(w) | ESS | Brier | logloss | AUROC (clean=1) | AP | 残余噪声质量占比 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in table:
        fmt = lambda v: "NA" if v is None else f"{v:.6f}"
        text.append("| " + " | ".join([row["arm"]] + [fmt(row[k]) for k in ("mean_weight", "ess", "brier", "logloss", "auroc", "auprc_average_precision", "residual_noise_mass_fraction")]) + " |")
    text += ["", "所有评分从已有锁定 CSV 读取，原始生产权重未重拟合。AUROC 给并列正负对半分；AP 按整组相同分数合并，恒定分数的 AP 等于 clean 比例。", "", "rho 原区间仅作无保证诊断：P/Q 共用参考图像、Q 的原聚类标识和角色前缀检查未建立独立性、U 的 donor 关联可能跨原划分。", "", "同一 T 的 oracle-rho 只改变诊断后验中的混合率，不能用于新控制生成或调参。CSV 中 score/w 的 10 位小数精度限制、重建误差及所有输入哈希见 offline_metrics.json。", ""]
    out.mkdir(parents=True, exist_ok=True)
    (out / "OFFLINE_READOUT_CN.md").write_text("\n".join(text), encoding="utf-8")
    return report


def score_fixed_losses(observed_path, labels_path, controls_dir, loss_csv, out_dir, *,
                       expected_experiment_sha256=None, expected_checkpoint_sha256=None):
    """Reweight one shared loss export; this is not a full-training result.

    CSV requires batch_id,pair_id,ell and companion <CSV>.meta.json. Root's
    exporter additionally supplies loss_csv_sha256 and per-batch pool_indices.
    Older external exports can be described with explicit unverified metadata;
    a declared identity mismatch always fails before evaluator labels are read.
    """
    from .controls import load_observed
    manifest_path = Path(controls_dir) / "manifest.json"
    manifest = read_json(manifest_path)
    observed_digest = sha256_file(observed_path)
    if manifest.get("sources", {}).get("observed", {}).get("sha256") != observed_digest:
        raise ValueError("Fixed-loss observed map differs from frozen controls")
    rows = load_observed(observed_path)
    ids = [str(r["pair_id"]) for r in rows if r["role"] == "U"]
    row_by_id = {str(r["pair_id"]): r for r in rows}
    vectors, normalizers = {}, {}
    for arm in ARMS:
        item = manifest["arms"][arm]
        if item.get("sha256") != sha256_file(item["csv"]) or item.get("lock_sha256") != sha256_file(item["lock"]):
            raise ValueError(f"Fixed-loss frozen control identity differs: {arm}")
        w, _, _ = _weight_table(item["csv"], item["lock"], ids)
        lookup = dict(zip(ids, w))
        vectors[arm] = {pid: 1.0 if row["role"] != "U" else float(lookup[pid]) for pid, row in row_by_id.items()}
        normalizers[arm] = math.fsum(vectors[arm].values()) / len(rows)
    meta_path = Path(str(loss_csv) + ".meta.json")
    meta = read_json(meta_path)
    for key in ("checkpoint_sha256", "observed_sha256", "batch_definition", "ell_definition", "same_negative_pool"):
        if key not in meta:
            raise ValueError(f"Fixed-loss metadata missing {key}")
    if meta["observed_sha256"] != observed_digest or meta["same_negative_pool"] is not True:
        raise ValueError("Fixed-loss metadata must bind the observed map and shared negative pool")
    digest = str(meta["checkpoint_sha256"])
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
        raise ValueError("Fixed-loss checkpoint identity must be a SHA256")
    for key in ("test_read", "evaluator_labels_read", "weights_applied_during_export"):
        if key in meta and meta[key] is not False:
            raise ValueError(f"Fixed-loss exporter metadata must declare {key}=false")
    if "seed" in meta and (type(meta["seed"]) is not int or meta["seed"] != manifest.get("seed")):
        raise ValueError("Fixed-loss export seed differs from frozen controls")
    if expected_experiment_sha256 is not None and meta.get("experiment_sha256") != expected_experiment_sha256:
        raise ValueError("Fixed-loss export experiment SHA256 differs from requested experiment")
    if expected_checkpoint_sha256 is not None and meta.get("checkpoint_sha256") != expected_checkpoint_sha256:
        raise ValueError("Fixed-loss checkpoint SHA256 differs from requested reference checkpoint")
    csv_digest = sha256_file(loss_csv)
    csv_identity_verified = "loss_csv_sha256" in meta
    if csv_identity_verified and meta["loss_csv_sha256"] != csv_digest:
        raise ValueError("Fixed-loss CSV differs from metadata loss_csv_sha256")
    warnings = []
    if not csv_identity_verified:
        warnings.append("Metadata has no loss_csv_sha256: these numbers describe the current CSV only; its link to the claimed export is unverified. Do not call the scientific export complete.")
    batches, seen = {}, set()
    with Path(loss_csv).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not {"batch_id", "pair_id", "ell"}.issubset(fields):
            raise ValueError("Fixed-loss CSV requires unique columns batch_id,pair_id,ell")
        for rec in reader:
            if None in rec or any(v is None for v in rec.values()):
                raise ValueError("Fixed-loss CSV row has malformed column count")
            bid, pid, ell = str(rec["batch_id"]), str(rec["pair_id"]), float(rec["ell"])
            if not bid or pid not in row_by_id or (bid, pid) in seen or not math.isfinite(ell) or ell < 0:
                raise ValueError("Invalid, duplicate, or unknown fixed-loss observation")
            seen.add((bid, pid))
            batches.setdefault(bid, []).append((pid, ell))
    if not batches:
        raise ValueError("No fixed-loss batches")
    batch_identity_verified = False
    if "batches" in meta:
        declared = meta["batches"]
        if not isinstance(declared, list) or any(not isinstance(b, dict) for b in declared):
            raise ValueError("Fixed-loss batches metadata must be a list of objects")
        declared_ids = [str(b.get("batch_id")) for b in declared]
        if len(set(declared_ids)) != len(declared_ids) or set(declared_ids) != set(batches):
            raise ValueError("Fixed-loss batch ID coverage differs from export metadata")
        all_indices_present = True
        for b in declared:
            bid = str(b["batch_id"])
            if type(b.get("n")) is not int or b["n"] != len(batches[bid]):
                raise ValueError("Fixed-loss batch row count differs from metadata")
            if "pool_indices" not in b:
                all_indices_present = False
                continue
            indices = b["pool_indices"]
            if (not isinstance(indices, list) or len(indices) != b["n"]
                    or any(type(i) is not int or i < 0 or i >= len(rows) for i in indices)
                    or len(set(indices)) != len(indices)):
                raise ValueError("Fixed-loss batch pool_indices are invalid")
            expected_ids = [str(rows[i]["pair_id"]) for i in indices]
            if expected_ids != [pid for pid, ell in batches[bid]]:
                raise ValueError("Fixed-loss pair order differs from declared observed pool_indices")
        batch_identity_verified = all_indices_present
    if not batch_identity_verified:
        warnings.append("Exact per-batch pool-index/order metadata is unavailable; claimed batches are not independently bound to ordered observed rows.")
    if expected_experiment_sha256 is None:
        warnings.append("No expected experiment SHA256 was supplied to this API; the metadata's experiment identity, if present, is reported without an external comparison.")
    if expected_checkpoint_sha256 is None:
        warnings.append("No expected reference checkpoint SHA256 was supplied to this API; the named checkpoint identity is a recorded export claim.")
    # Every control/CSV/metadata identity check precedes this evaluator-only read.
    _, _, labels, _, _ = _observed_and_labels(observed_path, labels_path)
    vectors["CleanOracle_EVAL_ONLY"] = {pid: float(not labels[pid]["mismatched"]) for pid in row_by_id}
    normalizers["CleanOracle_EVAL_ONLY"] = math.fsum(vectors["CleanOracle_EVAL_ONLY"].values()) / len(rows)
    table = []
    for bid, pairs in batches.items():
        for arm, weights in vectors.items():
            numerator = math.fsum(weights[pid] * ell for pid, ell in pairs)
            den = normalizers[arm]
            table.append({"batch_id": bid, "arm": arm, "n_pairs": len(pairs), "mean_v_full_pool": den,
                          "sum_objective": numerator / den if den else None,
                          "normalized_mean_objective": numerator / (len(pairs) * den) if den else None})
    # Catch concurrent export modifications instead of attaching the new hash to
    # numbers read from an earlier version of the file.
    if sha256_file(loss_csv) != csv_digest:
        raise ValueError("Fixed-loss CSV changed during scoring")
    report = {"schema_version": 1, "diagnostic_only": True, "test_read": False,
              "labels_used_only_by_clean_oracle": True, "loss_csv_sha256": csv_digest,
              "metadata_sha256": sha256_file(meta_path), "controls_manifest_sha256": sha256_file(manifest_path),
              "source_metadata": meta, "batches": table,
              "identity": {"csv_identity_verified": csv_identity_verified,
                           "batch_identity_verified": batch_identity_verified,
                           "observed_control_identity_verified": True,
                           "experiment_identity_verified": expected_experiment_sha256 is not None,
                           "checkpoint_identity_compared_with_expected": expected_checkpoint_sha256 is not None,
                           "checkpoint_model_execution_independently_verified_by_scorer": False},
              "scientific_training_complete": False,
              "warnings": warnings + ["The scorer reads exported losses without loading the model. Hash binding does not independently prove model execution; no full-training completion or retrained-endpoint claim is made."],
              "interpretation": "Same student, per-anchor losses, batches and negative candidates as declared by the exporter. Reweighting risk diagnostic does not identify the independently trained endpoint or a clean-negative-pool risk."}
    write_json(Path(out_dir) / "fixed_loss_metrics.json", report)
    write_csv(Path(out_dir) / "fixed_loss_metrics.csv", table)
    return report
