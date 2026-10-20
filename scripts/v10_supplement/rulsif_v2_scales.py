"""FIT-only kernel-scale diagnostic for RuLSIF_exact_v2 design (CPU Slurm; FIT reads only).

Per modality: median pairwise distance of 9 x 9 patches drawn on each support, the share of
the joint squared distance contributed by each side, and the kernel-value spread on each side
under v1's joint-median bandwidth rule. Seed-1117 streams; no CAL, DEV, TEST or EPFL image.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Runs under Slurm")
    import torch
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "2")))
    from v10 import adapter as A
    from v10.supplement import rulsif_v2 as V2
    from v10.supplement.rulsif import gaussian_features
    guard = A.DataGuard()
    fit = A.fit_cores(guard, 1.0, 1117)
    out = {}
    for support in ("square_v1", "disk_r129"):
        scenes, cx, cy, median, meta = V2.fit_scenes(
            guard, fit, support=support, centres_per_core=4, positions_per_core=2048,
            stream_tag=f"V10|RuLSIF_exact_v2|SCALES|1117|{support}", device="cpu",
            purpose="V10 RuLSIF_exact_v2 FIT-only kernel scale diagnostic")
        dx, dy = torch.pdist(cx.double()), torch.pdist(cy.double())
        joint = torch.pdist(torch.cat((cx, cy), -1).double())
        x = torch.cat([s[0] for s in scenes]); y = torch.cat([s[1] for s in scenes])
        row = {"median_joint": median, "median_bf": float(dx.median()), "median_shg": float(dy.median()),
               "mean_share_of_joint_sq_distance_shg": float((dy.square() / joint.square().clamp_min(1e-12)).mean()),
               "patch_value_quantiles_bf": [float(v) for v in torch.quantile(x.flatten()[::97].double(), torch.tensor([.01, .25, .5, .75, .99], dtype=torch.float64))],
               "patch_value_quantiles_shg": [float(v) for v in torch.quantile(y.flatten()[::97].double(), torch.tensor([.01, .25, .5, .75, .99], dtype=torch.float64))]}
        for mult in (0.5, 1.0, 2.0):
            h = mult * median
            kx = gaussian_features(x[::16], cx, h); ky = gaussian_features(y[::16], cy, h)
            row[f"x{mult}"] = {
                "bf_kernel_quantiles": [float(v) for v in torch.quantile(kx.flatten().double(), torch.tensor([.05, .5, .95], dtype=torch.float64))],
                "shg_kernel_quantiles": [float(v) for v in torch.quantile(ky.flatten().double(), torch.tensor([.05, .5, .95], dtype=torch.float64))],
                "product_max_per_point_median": float((kx * ky).max(1).values.median()),
                "product_sum_per_point_median": float((kx * ky).sum(1).median())}
        out[support] = row
        print(support, json.dumps(row), flush=True)
    if {r["role"] for r in guard.reads} != {"FIT"}:
        raise RuntimeError("Non-FIT read")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"job": os.environ["SLURM_JOB_ID"], "utc": time.strftime("%FT%TZ", time.gmtime()),
                                    "roles_read": ["FIT"], "supports": out}, indent=1) + "\n")


if __name__ == "__main__":
    main()
