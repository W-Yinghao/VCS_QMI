"""P115 — v5 NEXT-A-AUG (VCS_Server_Next_Round_v5_CN.md §3): split the strong augmentation into its crop and colour-jitter parts for P107 A-P3
and tuned SimCLR (P41 recipe), seed 0, 800 epochs — 4 new units; the standard and both-strong cells reuse existing runs:

  cell                crop scale min   jitter B/C/S/H        A-P3 run                               SimCLR run
  standard            0.20             0.4/0.4/0.4/0.1       P107_AP3_views4_800ep_seed0 (exists)   P41_simclr_views4_800ep_seed0 (exists)
  crop-only strong    0.08             0.4/0.4/0.4/0.1       P115_AP3_croponly_seed0 (new)          P115_simclr_croponly_seed0 (new)
  jitter-only strong  0.20             0.8/0.8/0.8/0.2       P115_AP3_jitteronly_seed0 (new)        P115_simclr_jitteronly_seed0 (new)
  both strong         0.08             0.8/0.8/0.8/0.2       P107_AP3_augstrong_views4_800ep_seed0  P89_simclr_views4_800ep_augstrong_seed0

Each new YAML = the method's standard seed-0 config (A-P3: cifar10_hpY_AP3_views4_800ep_seed0.yaml; SimCLR: cifar10_hpN_simclr_views4_800ep_seed0.yaml)
with ONLY views.random_resized_crop.scale[0] or views.color_jitter.{brightness,contrast,saturation,hue} set to the strong values (the values of the
existing both-strong configs), plus run.stage and checkpoint epochs 20 / 100 / 400 / 800 added to the existing list.  Flip, grayscale, jitter p unchanged.

    python configs/make_p115_configs.py [--write]
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

STAGE = "P115_v5_aug_factors"
METHODS = {"AP3": ("cifar10_hpY_AP3_views4_800ep_seed0.yaml", "cifar10_hpY_AP3_augstrong_views4_800ep_seed0.yaml"),
           "simclr": ("cifar10_hpN_simclr_views4_800ep_seed0.yaml", "cifar10_hpS_simclr_views4_800ep_augstrong_seed0.yaml")}


def load(n: str) -> dict:
    return yaml.safe_load(open(ROOT / "configs" / n))


def apply(c: dict, strong_ref: dict, factor: str) -> dict:
    v, sv = c["views"], strong_ref["views"]
    assert v["random_resized_crop"]["scale"] == [0.2, 1.0] and v["color_jitter"]["brightness"] == 0.4
    if factor == "croponly":
        v["random_resized_crop"]["scale"] = list(sv["random_resized_crop"]["scale"])
    elif factor == "jitteronly":
        for k in ("brightness", "contrast", "saturation", "hue"):
            v["color_jitter"][k] = sv["color_jitter"][k]
    else:
        raise ValueError(factor)
    c["run"]["stage"] = STAGE
    c["logging"]["checkpoint_epochs"] = sorted(set(c["logging"]["checkpoint_epochs"]) | {20, 100, 400, 800})
    return c


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    shas, lines = {}, []
    for m, (std_name, strong_name) in METHODS.items():
        std, strong = load(std_name), load(strong_name)
        fs, fst = flat(std), flat(strong)
        both = {k for k in set(fs) | set(fst) if fs.get(k) != fst.get(k)} - {"run.stage"}
        assert both == {"views.random_resized_crop.scale", "views.color_jitter.brightness", "views.color_jitter.contrast",
                        "views.color_jitter.saturation", "views.color_jitter.hue"}, both  # the strong block differs only in crop + jitter
        for factor in ("croponly", "jitteronly"):
            c = apply(json.loads(json.dumps(std)), strong, factor)
            name = f"cifar10_hpAF_{m}_{factor}_views4_800ep_seed0.yaml"; run_id = f"P115_{m}_{factor}_views4_800ep_seed0"
            txt = yaml.safe_dump(c, sort_keys=False); sha = hashlib.sha256(txt.encode()).hexdigest()
            fc = flat(c)
            diff = {k: [fs.get(k, "<absent>"), fc.get(k, "<absent>")] for k in sorted(set(fs) | set(fc)) if fs.get(k, "<absent>") != fc.get(k, "<absent>")}
            shas[name] = {"sha256": sha, "run_id": run_id, "parent": std_name, "strong_reference": strong_name, "factor": factor, "diff_vs_parent": diff}
            print(f"{run_id:40s} {sha[:16]}  " + "; ".join(f"{k}: {v[0]} -> {v[1]}" for k, v in diff.items()))
            lines.append(f"sbatch --parsable --partition=RTX6000PRO,H100 --job-name=p115_{m.lower()}_{factor} "
                         f"--export=ALL,CFG=configs/{name},RUN_ID={run_id},FINAL_CKPT=epoch_800.pt slurm/run_unit.sbatch")
            if a.write:
                (ROOT / "configs" / name).write_text(txt)
    if a.write:
        (ROOT / "configs" / "P115_SHA256.json").write_text(json.dumps({"stage": STAGE, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "configs": shas}, indent=1))
        (ROOT / "slurm" / "p115_lines.txt").write_text(
            "# P115 (v5 NEXT-A-AUG) launch lines — DRAFT, NOT SUBMITTED. Conditional on P107 layer 2 + P112 follow-up (v5 §3). Normal QOS; node60 allowed.\n"
            + "\n".join(lines) + "\n")
        print("wrote configs, configs/P115_SHA256.json, slurm/p115_lines.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
