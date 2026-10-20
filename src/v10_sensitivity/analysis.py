"""Pre-registered descriptive read-outs (core is the unit; perturbations are not independent)."""
from __future__ import annotations

import math
import statistics

import numpy as np

from .common import require

STRATA = ((0.0, 10.0), (10.0, 20.0), (20.0, 30.0))
BOOTSTRAP = {"resamples": 10000, "seed": 20261019, "level": 0.95}
REPLAY_KEYS = ("err_full", "theta_hat", "t_hat", "F_hat", "F_true")


def metrics(rows):
    """Identical to the frozen records.metrics (fractions over registrations; statistics.median)."""
    errors = [row["err_full"] for row in rows]
    require(errors and all(math.isfinite(e) and e >= 0 for e in errors), "Incomplete or invalid registration errors")
    return {"n": len(errors), "success42": sum(e < 42 for e in errors) / len(errors),
            "success9": sum(e < 9 for e in errors) / len(errors), "median_err": statistics.median(errors)}


def stratum_of(theta):
    """|theta*| in [0,10), [10,20), [20,30] (the last stratum includes its upper bound)."""
    value = abs(float(theta))
    require(value <= 30.0, f"|theta*| {value} outside the common range")
    if value < 10.0:
        return "0-10"
    if value < 20.0:
        return "10-20"
    return "20-30"


def rotation_strata(rows):
    out = {}
    for low, high in STRATA:
        name = f"{low:g}-{high:g}"
        selected = [r for r in rows if stratum_of(r["theta"]) == name]
        out[name] = metrics(selected) if selected else {"n": 0}
    filled = [v["success42"] for v in out.values() if v["n"]]
    out["spread_success42"] = (max(filled) - min(filled)) if filled else None
    return out


def per_core(rows, threshold):
    cores = {}
    for row in rows:
        cores.setdefault(row["core"], []).append(row["err_full"] < threshold)
    require(all(len(v) == 10 for v in cores.values()), "Every core must carry its 10 perturbations")
    return {core: sum(v) / len(v) for core, v in sorted(cores.items())}


def paired_core_differences(new_rows, old_rows, threshold):
    """Per-core success difference (new minus old) on identical (core, perturbation) keys."""
    new_keys = {(r["core"], r["perturbation"]) for r in new_rows}
    old_keys = {(r["core"], r["perturbation"]) for r in old_rows}
    require(new_keys == old_keys and len(new_keys) == len(new_rows) == len(old_rows),
            "Paired read-outs must cover identical core/perturbation keys")
    new, old = per_core(new_rows, threshold), per_core(old_rows, threshold)
    return {core: new[core] - old[core] for core in new}


def bootstrap_mean(values, resamples=BOOTSTRAP["resamples"], seed=BOOTSTRAP["seed"], level=BOOTSTRAP["level"]):
    """Mean and percentile CI over the units (cores); deterministic seed."""
    data = np.asarray(list(values), dtype=np.float64)
    require(data.size > 0 and np.isfinite(data).all(), "Bootstrap needs finite values")
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, data.size, size=(resamples, data.size))].mean(1)
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return {"mean": float(data.mean()), "ci_low": float(low), "ci_high": float(high),
            "units": int(data.size), **{k: BOOTSTRAP[k] for k in ("resamples", "seed", "level")}}


def averaged_core_differences(per_seed):
    """Average the per-core differences over seeds (same cores in every seed)."""
    cores = sorted(per_seed[0])
    require(all(sorted(d) == cores for d in per_seed), "Seeds cover different cores")
    return {core: sum(d[core] for d in per_seed) / len(per_seed) for core in cores}


def failure_split(rows, threshold=42.0):
    failures = [r for r in rows if r["err_full"] >= threshold]
    energy = sum(r["F_hat"] >= r["F_true"] for r in failures)
    return {"failures": len(failures), "energy_optimum_wrong": energy, "search_missed_higher_F": len(failures) - energy}


def _flat(value):
    return [float(x) for x in value] if isinstance(value, (list, tuple)) else [float(value)]


def replay_compare(new_rows, frozen_rows, keys=REPLAY_KEYS):
    """Exact-equality count and max |difference| per field between a replay and frozen rows."""
    frozen = {(r["core"], r["perturbation"]): r for r in frozen_rows}
    new = {(r["core"], r["perturbation"]): r for r in new_rows}
    require(set(new) == set(frozen) and len(new) == len(new_rows) and len(frozen) == len(frozen_rows),
            "Replay must cover exactly the frozen core/perturbation keys")
    identical_rows, fields = 0, {}
    for key in sorted(new):
        same = True
        for name in keys:
            a, b = _flat(new[key][name]), _flat(frozen[key][name])
            difference = max(abs(x - y) for x, y in zip(a, b, strict=True))
            fields[name] = max(fields.get(name, 0.0), difference)
            same &= a == b
        identical_rows += same
    return {"rows": len(new), "identical_rows": identical_rows, "max_abs_difference": fields,
            "metrics_new": metrics(list(new.values())), "metrics_frozen": metrics(list(frozen.values())),
            "status": "IDENTICAL" if identical_rows == len(new) else "MISMATCH_DISCLOSED"}
