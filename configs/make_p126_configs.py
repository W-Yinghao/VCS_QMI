"""P126 — v6 V6-CURVE (VCS_Server_Tasks_v6_CN.md §7.1): four seed-0 800-epoch units of the fixed anchored-quadratic scorer
    f(s) = a[(s - kappa) + lambda s(1 - s)],  T = tanh f,  (a, kappa) = (2, 0.5)  [b = -1],  lambda in {-0.25, +0.25},
each = the A-P3 config of that dataset (fixed f = 2s - 1, all-view tokens, full negative gradient) with ONLY
    model.critic.affine_mode   fixed -> fixed_curved
    model.critic.curvature_lambda (absent) -> lambda
    run.stage                  -> P126_v6_curve
The lambda = 0 baselines are the existing A-P3 runs: P107_AP3_views4_800ep_seed0 (CIFAR-10) and P107_AP3_c100_views4_800ep_seed0 (CIFAR-100).
kappa is the AFFINE anchor; the actual zero s0 of f (lambda = +0.25: 0.4384, lambda = -0.25: 0.5616) is recorded in each run manifest.

    python configs/make_p126_configs.py [--write]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "configs"))
from make_p104_configs import flat  # noqa: E402

STAGE = "P126_v6_curve"
PARENTS = {"c10": ("cifar10_hpY_AP3_views4_800ep_seed0.yaml", "P107_AP3_views4_800ep_seed0"),
           "c100": ("cifar100_hpY_AP3_c100_views4_800ep_seed0.yaml", "P107_AP3_c100_views4_800ep_seed0")}
LAMBDAS = (-0.25, 0.25)


def tag_of(lam: float) -> str:
    return ("m" if lam < 0 else "p") + f"{abs(lam):g}".replace(".", "")


def apply(c: dict, lam: float) -> dict:
    cr, pr = c["model"]["critic"], c["pairing"]
    assert cr["affine_mode"] == "fixed" and float(cr["cosine_scale_init"]) == 2.0 and float(cr["cosine_bias_init"]) == -1.0
    assert pr["pair_scope"] == "all_view_tokens" and pr["negative_detach"] is False and c["objective"]["loss"] == "negative_J"
    cr["affine_mode"] = "fixed_curved"
    cr["curvature_lambda"] = float(lam)
    c["run"]["stage"] = STAGE
    return c


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    shas, lines = {}, []
    for ds, (parent, baseline) in PARENTS.items():
        base = yaml.safe_load(open(ROOT / "configs" / parent))
        for lam in LAMBDAS:
            c = apply(json.loads(json.dumps(base)), lam)
            pre = "cifar100" if ds == "c100" else "cifar10"
            name = f"{pre}_hpCV_AP3_{ds}_lam{tag_of(lam)}_views4_800ep_seed0.yaml"
            run_id = f"P126_CV_AP3_{ds}_lam{tag_of(lam)}_views4_800ep_seed0"
            txt = yaml.safe_dump(c, sort_keys=False); sha = hashlib.sha256(txt.encode()).hexdigest()
            fb, fc = flat(base), flat(c)
            diff = {k: [fb.get(k, "<absent>"), fc.get(k, "<absent>")] for k in sorted(set(fb) | set(fc)) if fb.get(k, "<absent>") != fc.get(k, "<absent>")}
            shas[name] = {"sha256": sha, "run_id": run_id, "parent": parent, "lambda_zero_baseline_run": baseline, "a": 2.0, "kappa_anchor": 0.5,
                          "lambda": lam, "diff_vs_parent": diff}
            print(f"{run_id:44s} {sha[:16]}  " + "; ".join(f"{k}: {v[0]} -> {v[1]}" for k, v in diff.items()))
            lines.append(f"sbatch --parsable --partition=RTX6000PRO,H100 --job-name=p126_{ds}_lam{tag_of(lam)} "
                         f"--export=ALL,CFG=configs/{name},RUN_ID={run_id},FINAL_CKPT=epoch_800.pt slurm/run_unit.sbatch")
            if a.write:
                (ROOT / "configs" / name).write_text(txt)
    if a.write:
        (ROOT / "configs" / "P126_SHA256.json").write_text(json.dumps({"stage": STAGE, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                                       "configs": shas}, indent=1))
        (ROOT / "slurm" / "p126_lines.txt").write_text(
            "# P126 (v6 V6-CURVE) launch lines — DRAFT, NOT SUBMITTED.  Start condition: stage-B review + separate batch approval (v6 §7.1).\n"
            "# Normal QOS only (never runfill); node60 allowed (no --exclude).  lambda = 0 baselines = existing A-P3 runs (P107 seed 0, both datasets).\n"
            + "\n".join(lines) + "\n")
        print("wrote configs, configs/P126_SHA256.json, slurm/p126_lines.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
