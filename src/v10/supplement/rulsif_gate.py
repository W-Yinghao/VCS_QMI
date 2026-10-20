"""FIT-only functional gate for the kernel-ratio registration energies (RuLSIF v1 and v2).

Reads FIT images only, every read through the V10 DataGuard with role FIT. A FIT sample is a
(core, perturbation index i) pair, built with the registration's own rules:
  positions    Q.RigidEnergy draw with seed 'R6-REG-P2|pos|<core>|<i>' (the CAL/DEV rule);
  images       the host registration inputs: BF luminance reference and the moving SHG intensity
               produced by the host moving-image pipeline at the zero perturbation (aligned);
  perturbation the frozen R6-REG-P2 rule Q.perturbations_p2(core)[i] (logged as a FIT rule read).
F_true is the energy at the true alignment (identity pose), F_imposed the energy with the moving
image misaligned by the imposed perturbation (theta*, t*), and F_local the energies at six fixed
near-truth offsets (diagnostic only). J(T = 0) = 0, so F_true > 0 means the fitted T beats T = 0.
"""
from __future__ import annotations

import math
import statistics

import numpy as np
import torch

from v10 import adapter as A

LOCAL_POSES = ((6.0, 0.0, 0.0), (-6.0, 0.0, 0.0), (0.0, 8.0, 0.0), (0.0, -8.0, 0.0),
               (0.0, 0.0, 8.0), (0.0, 0.0, -8.0))


def crossfit_folds(cores, folds):
    """Deterministic core-level folds: fold j holds sorted(cores)[j::folds]."""
    ordered = sorted(cores)
    if folds < 2 or len(ordered) % folds:
        raise ValueError("Core folds must partition the FIT cores evenly")
    return [ordered[j::folds] for j in range(folds)]


def fit_inputs(guard, cores, device):
    identity = {core: [[0.0, 0.0, 0.0]] for core in cores}
    return A.GuardedInputs(guard, {"FIT"}, "V10 RuLSIF functional gate (FIT only)", identity, device)


def truth_terms(energy):
    """Diagnostic split of F_true = J(g - 1) at the identity pose (exact V moments):
    dependence = E_P g - E_Q g = E_P T - E_Q T and quadratic = -(E_P T^2 + E_Q T^2) / 2."""
    ratio = energy.ratio
    values = A.Q.sample_pts(torch, torch.nn.functional, energy.mov, energy.patch_positions[None])[0, :, 0]
    values = values.reshape(len(energy.x), 81)
    if hasattr(ratio, "right_features"):
        right = ratio.right_features(values)
    else:
        from .rulsif import gaussian_features
        right = gaussian_features(values, ratio.centres_right, ratio.bandwidth)
    w = ratio.coefficients
    gp = (energy.left * right) @ w
    gq = (energy.left.mean(0) * right.mean(0)) @ w
    gq2 = ((right @ energy.q_quadratic) * right).sum(-1).mean()
    tp2 = (gp - 1).square().mean()
    tq2 = gq2 - 2 * gq + 1
    return float(gp.mean() - gq), float(-0.5 * (tp2 + tq2))


def evaluate(model, energy_class, inputs, guard, cores, shg_input="intensity"):
    """Gate rows for every (core, i), i = 0..9, of the given FIT cores.

    shg_input selects the host moving path that matches the model's FIT input: 'intensity'
    (robust-normalised SHG, index 0) or 'log1p' (robust-normalised log1p SHG, index 1).
    """
    A.freeze_for_test(model)
    device = model.centres_left.device
    index = {"intensity": 0, "log1p": 1}[shg_input]
    rows = []
    for core in cores:
        reference = inputs.reference(core)[0]
        moving = inputs.moving(core, 0)[index]
        guard.log_rule(core, {"FIT"}, "V10 RuLSIF functional gate: frozen R6-REG-P2 imposed perturbation rule")
        for i, (theta, ty, tx) in enumerate(A.Q.perturbations_p2(core)):
            rng = np.random.default_rng(A.Q.seed_of(f"R6-REG-P2|pos|{core}|{i}"))
            energy = energy_class(reference, moving, rng, ratio=model)
            poses = ((0.0, 0.0, 0.0), (theta, ty, tx)) + LOCAL_POSES
            angles = torch.tensor([math.radians(p[0]) for p in poses], device=device, dtype=torch.float32)
            shifts = torch.tensor([[p[1], p[2]] for p in poses], device=device, dtype=torch.float32)
            with torch.no_grad():
                values = [float(v) for v in energy.batch(angles, shifts)]
                dependence, quadratic = truth_terms(energy)
            if not all(math.isfinite(v) for v in values + [dependence, quadratic]):
                raise FloatingPointError("Non-finite gate energy")
            rows.append({"core": core, "perturbation": i, "imposed": [theta, ty, tx],
                         "F_true": values[0], "F_imposed": values[1], "F_local": values[2:],
                         "truth_dependence_term": dependence, "truth_quadratic_term": quadratic})
    return rows


def summarize(rows):
    if not rows:
        raise ValueError("Empty gate block")
    true = [r["F_true"] for r in rows]
    margin = [r["F_true"] - r["F_imposed"] for r in rows]
    local = [r["F_true"] > max(r["F_local"]) for r in rows]
    per_core = {}
    for r in rows:
        per_core.setdefault(r["core"], []).append(r)
    return {"n": len(rows), "cores": len(per_core),
            "mean_F_true": statistics.fmean(true), "median_F_true": statistics.median(true),
            "min_F_true": min(true), "max_F_true": max(true),
            "frac_F_true_positive": sum(t > 0 for t in true) / len(true),
            "frac_true_beats_imposed": sum(m > 0 for m in margin) / len(margin),
            "median_true_minus_imposed": statistics.median(margin),
            "frac_true_beats_local6": sum(local) / len(local),
            "mean_truth_dependence_term": statistics.fmean(r["truth_dependence_term"] for r in rows),
            "mean_truth_quadratic_term": statistics.fmean(r["truth_quadratic_term"] for r in rows),
            "frac_truth_dependence_positive": sum(r["truth_dependence_term"] > 0 for r in rows) / len(rows),
            "cores_majority_beats_imposed": sum(
                sum(x["F_true"] > x["F_imposed"] for x in v) * 2 > len(v) for v in per_core.values()),
            "per_core_mean_F_true": {c: statistics.fmean(x["F_true"] for x in v) for c, v in sorted(per_core.items())}}


def verdict(summary, thresholds):
    """Pre-registered pass rule (thresholds from the pre-registration record)."""
    conditions = {
        "mean_F_true_positive": summary["mean_F_true"] > thresholds["mean_F_true_gt"],
        "F_true_positive_fraction": summary["frac_F_true_positive"] >= thresholds["F_true_positive_fraction_min"],
        "true_beats_imposed_fraction": summary["frac_true_beats_imposed"] >= thresholds["true_beats_imposed_fraction_min"],
    }
    return {"status": "PASS" if all(conditions.values()) else "FAIL", "conditions": conditions}
