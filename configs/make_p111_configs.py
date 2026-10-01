"""P111 — G2 confirmation layer 2 (package v3 §10.2 / v4 A0, A4; owner 2026-10-02 "再提交十个任务"): ten full 800-epoch units.

  fresh seeds 3, 4   : G2 (P104 G2 recipe, seed s) and tuned SimCLR (P41 recipe, seed s) — paired new seeds
  strong augmentation: G2 seeds 0, 1, 2 with the P89 strong block (crop scale min 0.08; colour jitter 0.8 / 0.8 / 0.8 / 0.2), vs P89 SimCLR strong
  CIFAR-100          : G2 seeds 0, 1, 2 on the P91 CIFAR-100 8x recipe (dataset / manifest from cifar100_hpS_vcs_a5_views4_800ep_seed{s}.yaml)

G2's critic settings come from `make_p104_configs.apply(c, "G2")` (fixed a, b = 2, −1), so nothing about the scorer is re-typed here.  Seed-s bases for
s = 3, 4 are the seed-0 recipe files with run.seed = s (seed files 0–2 differ only in run.seed — verified by diff).

    python configs/make_p111_configs.py [--write]
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
from make_p104_configs import apply, flat  # noqa: E402

STAGE = "P111_G2_confirm2"
STRONG = {"crop_scale_min": 0.08, "cj": {"brightness": 0.8, "contrast": 0.8, "saturation": 0.8, "hue": 0.2}}


def load(name: str) -> dict:
    return yaml.safe_load(open(ROOT / "configs" / name))


def strong(c: dict, ref: dict) -> dict:
    """Copy the whole views block of the P89 strong-augmentation VCS config (only crop scale and colour jitter differ from the recipe)."""
    keep_count = c["views"]["count"]
    c["views"] = json.loads(json.dumps(ref["views"])); assert c["views"]["count"] == keep_count
    return c


def cells() -> list[tuple[str, dict, str]]:
    out = []
    for s in (3, 4):
        b = load("cifar10_hpK_a5_views4_800ep_vcs_seed0.yaml"); b["run"]["seed"] = s
        out.append((f"P111_G2_views4_800ep_seed{s}", apply(b, "G2"), "cifar10_hpK_a5_views4_800ep_vcs_seed0.yaml (seed→%d)" % s))
        b = load("cifar10_hpN_simclr_views4_800ep_seed0.yaml"); b["run"]["seed"] = s
        out.append((f"P111_simclr_views4_800ep_seed{s}", b, "cifar10_hpN_simclr_views4_800ep_seed0.yaml (seed→%d)" % s))
    for s in (0, 1, 2):
        ref = load("cifar10_hpO_a5_views4_800ep_augstrong_vcs_seed0.yaml" if s == 0 else f"cifar10_hpS_a5_views4_800ep_augstrong_vcs_seed{s}.yaml")
        b = load(f"cifar10_hpK_a5_views4_800ep_vcs_seed{s}.yaml")
        out.append((f"P111_G2_augstrong_views4_800ep_seed{s}", strong(apply(b, "G2"), ref), f"cifar10_hpK_a5_views4_800ep_vcs_seed{s}.yaml + P89 strong views"))
    for s in (0, 1, 2):
        b = load(f"cifar100_hpS_vcs_a5_views4_800ep_seed{s}.yaml")
        out.append((f"P111_G2_c100_views4_800ep_seed{s}", apply(b, "G2"), f"cifar100_hpS_vcs_a5_views4_800ep_seed{s}.yaml"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    shas, lines = {}, []
    for run_id, c, base in cells():
        c["run"]["stage"] = STAGE
        ce = c["logging"]["checkpoint_epochs"]
        if 20 not in ce and run_id.startswith("P111_G2"):
            c["logging"]["checkpoint_epochs"] = sorted(ce + [20])  # as P104 G2
        name = "cifar10_hpZ_" + run_id[len("P111_"):] + ".yaml" if "c100" not in run_id else "cifar100_hpZ_" + run_id[len("P111_"):] + ".yaml"
        txt = yaml.safe_dump(c, sort_keys=False); sha = hashlib.sha256(txt.encode()).hexdigest()
        shas[name] = {"sha256": sha, "run_id": run_id, "base": base}
        print(f"{run_id:42s} {sha[:16]}  seed={c['run']['seed']} data={c['data']['name']} crop={c['views'].get('crop_scale', c['views'].get('scale'))}")
        lines.append(f"sbatch --parsable --partition=RTX6000PRO,H100 --exclude=node60 --job-name={run_id.lower()[:28]} "
                     f"--export=ALL,CFG=configs/{name},RUN_ID={run_id},FINAL_CKPT=epoch_800.pt slurm/run_unit.sbatch")
        if a.write:
            (ROOT / "configs" / name).write_text(txt)
    if a.write:
        (ROOT / "configs" / "P111_SHA256.json").write_text(json.dumps({"stage": STAGE, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "configs": shas}, indent=1))
        (ROOT / "slurm" / "p111_lines.txt").write_text("# P111 G2 confirmation layer 2 — normal QOS (never runfill)\n" + "\n".join(lines) + "\n")
        print("wrote configs, configs/P111_SHA256.json, slurm/p111_lines.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
