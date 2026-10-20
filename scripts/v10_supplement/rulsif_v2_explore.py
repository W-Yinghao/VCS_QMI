"""FIT-only design evidence for RuLSIF_exact_v2 (CPU Slurm). Never reads CAL, DEV, TEST or EPFL.

4-fold core-level cross-fit on FIT32 with seed-1117 streams (distinct from the gate/CAL seed
1017). For each structural variant and (bandwidth multiplier, ridge): fit on 24 FIT cores with
the shared fitting code, evaluate the functional-gate statistics on the 8 held-out FIT cores
(registration position rule, aligned host moving image, frozen imposed perturbations).
Variants sharing a support share positions (paired comparison of the constant basis).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

J, PM = "joint_median", "per_modality_median"
VARIANTS = {
    # Round 1 (job 1048595/1048598 at 4c495a7): v1's joint width; the constant under the ridge.
    "v1_rule": {"constant": False, "support": "square_v1", "centres_per_core": 4, "bandwidth_rule": J, "penalize_constant": None},
    "a_only": {"constant": True, "support": "square_v1", "centres_per_core": 4, "bandwidth_rule": J, "penalize_constant": True},
    "b_only": {"constant": False, "support": "disk_r129", "centres_per_core": 4, "bandwidth_rule": J, "penalize_constant": None},
    "ab_c4": {"constant": True, "support": "disk_r129", "centres_per_core": 4, "bandwidth_rule": J, "penalize_constant": True},
    "ab_c8": {"constant": True, "support": "disk_r129", "centres_per_core": 8, "bandwidth_rule": J, "penalize_constant": True},
    "ab_c16": {"constant": True, "support": "disk_r129", "centres_per_core": 16, "bandwidth_rule": J, "penalize_constant": True},
    # Round 2: per-modality widths (c) and the unpenalised constant.
    "c_only": {"constant": False, "support": "disk_r129", "centres_per_core": 4, "bandwidth_rule": PM, "penalize_constant": None},
    "abc_c4_pen": {"constant": True, "support": "disk_r129", "centres_per_core": 4, "bandwidth_rule": PM, "penalize_constant": True},
    "abc_c4": {"constant": True, "support": "disk_r129", "centres_per_core": 4, "bandwidth_rule": PM, "penalize_constant": False},
    "ac_square_c4": {"constant": True, "support": "square_v1", "centres_per_core": 4, "bandwidth_rule": PM, "penalize_constant": False},
    "abc_c8": {"constant": True, "support": "disk_r129", "centres_per_core": 8, "bandwidth_rule": PM, "penalize_constant": False},
    "abc_c16": {"constant": True, "support": "disk_r129", "centres_per_core": 16, "bandwidth_rule": PM, "penalize_constant": False},
    # Round 3: SHG on the host log1p path (d), with (a)+(b)+(c).
    "abcd_c4": {"constant": True, "support": "disk_r129", "centres_per_core": 4, "bandwidth_rule": PM, "penalize_constant": False, "shg_input": "log1p"},
    "abcd_c8": {"constant": True, "support": "disk_r129", "centres_per_core": 8, "bandwidth_rule": PM, "penalize_constant": False, "shg_input": "log1p"},
    "abcd_c16": {"constant": True, "support": "disk_r129", "centres_per_core": 16, "bandwidth_rule": PM, "penalize_constant": False, "shg_input": "log1p"},
    "abd_joint_c8": {"constant": True, "support": "disk_r129", "centres_per_core": 8, "bandwidth_rule": J, "penalize_constant": False, "shg_input": "log1p"},
    "bcd_noconst_c8": {"constant": False, "support": "disk_r129", "centres_per_core": 8, "bandwidth_rule": PM, "penalize_constant": None, "shg_input": "log1p"},
    # Round 5: per-image empirical-CDF (copula) transform of both modalities (e), with (a)+(b)+(c).
    "abce_c8": {"constant": True, "support": "disk_r129", "centres_per_core": 8, "bandwidth_rule": PM, "penalize_constant": False, "input_transform": "copula"},
    "abce_c16": {"constant": True, "support": "disk_r129", "centres_per_core": 16, "bandwidth_rule": PM, "penalize_constant": False, "input_transform": "copula"},
    "abe_joint_c8": {"constant": True, "support": "disk_r129", "centres_per_core": 8, "bandwidth_rule": J, "penalize_constant": False, "input_transform": "copula"},
    # Round 6: more centres with (a)+(b)+(c)+(e).
    "abce_c32": {"constant": True, "support": "disk_r129", "centres_per_core": 32, "bandwidth_rule": PM, "penalize_constant": False, "input_transform": "copula"},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1117)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--mults", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--ridges", nargs="+", type=float, default=[1e-4, 1e-3, 1e-2])
    parser.add_argument("--positions", type=int, default=2048)
    parser.add_argument("--folds", type=int, default=4)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("FIT design evidence runs under Slurm")
    if args.seed == 1017:
        raise ValueError("The tuning/gate seed 1017 is reserved for the formal gate and CAL")
    import torch
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "2")))
    from v10 import adapter as A
    from v10.supplement import rulsif_gate as G
    from v10.supplement import rulsif_v2 as V2
    from v10.supplement.registration import RuLSIFRigidEnergy
    guard = A.DataGuard()
    fit = A.fit_cores(guard, 1.0, args.seed)
    if len(fit) != 32:
        raise RuntimeError("FIT32 required")
    cache = {}

    def loader(core, key):
        if (core, key) not in cache:
            cache[(core, key)] = A.P.load_img(guard.r6_root, core, ("bf", key))
        return cache[(core, key)]

    inputs = G.fit_inputs(guard, fit, "cpu")
    folds = G.crossfit_folds(fit, args.folds)
    started = time.time()
    configs = []
    for name in args.variants:
        variant = VARIANTS[name]
        energy_class = (RuLSIFRigidEnergy if not variant["constant"] and variant["bandwidth_rule"] == J
                        and variant.get("input_transform", "none") == "none" else V2.RuLSIFV2RigidEnergy)
        blocks = {}
        for j, held in enumerate(folds):
            train = [c for c in fit if c not in held]
            tag = f"V10|RuLSIF_exact_v2|DESIGN|{args.seed}|{variant['support']}|fold{j}"
            scenes, cx, cy, medians, meta = V2.fit_scenes(
                guard, train, support=variant["support"], centres_per_core=variant["centres_per_core"],
                positions_per_core=args.positions, stream_tag=tag, device="cpu",
                purpose="V10 RuLSIF_exact_v2 FIT-only design evidence", loader=loader,
                shg_input=variant.get("shg_input", "intensity"),
                input_transform=variant.get("input_transform", "none"))
            for mult in args.mults:
                for ridge in args.ridges:
                    model = V2.kernel_model(cx, cy, medians, constant=variant["constant"],
                                            bandwidth_rule=variant["bandwidth_rule"], bandwidth_multiplier=mult,
                                            ridge=ridge, alpha=0.5,
                                            penalize_constant=bool(variant["penalize_constant"]),
                                            input_transform=variant.get("input_transform", "none"))
                    info = model.fit(scenes, q_mode="v")
                    with torch.no_grad():
                        in_sample = statistics.fmean(float(model.common_j(x, y)) for x, y in scenes)
                    rows = G.evaluate(model, energy_class, inputs, guard, held,
                                      shg_input=variant.get("shg_input", "intensity"))
                    block = blocks.setdefault((mult, ridge), {"rows": [], "folds": []})
                    block["rows"].extend(rows)
                    block["folds"].append({"fold": j, "held_out": held, "medians": medians,
                                           "bandwidth_left": getattr(model, "bandwidth_left", model.bandwidth),
                                           "bandwidth_right": getattr(model, "bandwidth_right", model.bandwidth),
                                           "J_fit_in_sample": in_sample, "J_fit_moments": info.get("J_fit_moments"),
                                           "mixture_mean_of_g": info.get("mixture_mean_of_g"),
                                           "negative_coefficients": info["negative_coefficients"],
                                           "constant_coefficient": info.get("constant_coefficient"),
                                           "coefficient_abs_max": float(model.coefficients.abs().max()),
                                           "linear_solve_residual": info["linear_solve_residual"]})
                    print(name, j, mult, ridge, "J_fit %.4f" % in_sample,
                          "heldout mean F_true %.4f" % statistics.fmean(r["F_true"] for r in rows),
                          "%.0fs" % (time.time() - started), flush=True)
        for (mult, ridge), block in blocks.items():
            summary = G.summarize(block["rows"])
            configs.append({"variant": name, "shg_input": "intensity", "input_transform": "none", **variant,
                            "bandwidth_multiplier": mult, "ridge": ridge,
                            "positions_per_core": args.positions, "crossfit": summary,
                            "folds": block["folds"]})
    reads = {r["role"] for r in guard.reads}
    if reads != {"FIT"}:
        raise RuntimeError("Design evidence touched a non-FIT role: " + repr(reads))
    result = {"schema": "v10.rulsif_exact_v2.design_evidence.v1", "status": "COMPLETE",
              "job": os.environ["SLURM_JOB_ID"], "utc": time.strftime("%FT%TZ", time.gmtime()),
              "seed_streams": args.seed, "folds": folds, "roles_read": sorted(reads),
              "fit_reads": sum(r.get("kind") != "rule" for r in guard.reads),
              "rule_reads": sum(r.get("kind") == "rule" for r in guard.reads),
              "held_test_read": False, "cal_read": False, "dev_read": False,
              "seconds": time.time() - started, "torch_threads": torch.get_num_threads(),
              "local_poses": [list(p) for p in G.LOCAL_POSES], "configs": configs}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name("." + args.out.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    os.replace(temporary, args.out)
    print("DESIGN_EVIDENCE_COMPLETE", args.out, flush=True)


if __name__ == "__main__":
    main()
