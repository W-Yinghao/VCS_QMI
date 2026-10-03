"""P114 — v5 NEXT-A-JS-AP3 (owner 2026-10-03: execute VCS_Server_Next_Round_v5_CN.md §2): the matched-JS control of P107 A-P3, seeds 0, 1, 2.

Each YAML is `configs/cifar10_hpY_AP3_views4_800ep_seed{s}.yaml` (A-P3: fixed scorer f = 2s - 1, all-view-token pairing, chunk 256,
negative_detach false) with ONLY these fields changed:
    objective.loss            negative_J -> js_matched_logistic   (L = mean_P softplus(-2f) + mean_Q softplus(2f), same logits / tokens / masks)
    objective.js_fixed_scorer (absent)   -> true                  (explicit marker, as P107 A-L1)
    objective.js_all_view_tokens (absent) -> true                 (second explicit marker: JS on the all-view path, P114 only)
    run.stage                 -> P114_v5_js_ap3
Encoder / projector init, augmentation, data order and every RNG role are inherited unchanged (run.seed is A-P3's).

    python configs/make_p114_configs.py [--write]
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

STAGE = "P114_v5_js_ap3"
PARENT = "cifar10_hpY_AP3_views4_800ep_seed{s}.yaml"


def apply(c: dict) -> dict:
    assert c["objective"]["loss"] == "negative_J" and c["pairing"]["pair_scope"] == "all_view_tokens" and c["pairing"]["negative_detach"] is False
    assert c["model"]["critic"]["affine_mode"] == "fixed" and float(c["model"]["critic"]["cosine_scale_init"]) == 2.0
    c["objective"]["loss"] = "js_matched_logistic"
    c["objective"]["js_fixed_scorer"] = True
    c["objective"]["js_all_view_tokens"] = True
    c["run"]["stage"] = STAGE
    return c


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); ap.add_argument("--seeds", default="0,1,2"); a = ap.parse_args()
    shas, lines = {}, []
    for s in [int(x) for x in a.seeds.split(",")]:
        parent = PARENT.format(s=s)
        base = yaml.safe_load(open(ROOT / "configs" / parent))
        c = apply(json.loads(json.dumps(base)))
        name = f"cifar10_hpJS_AP3_views4_800ep_seed{s}.yaml"; run_id = f"P114_JSAP3_views4_800ep_seed{s}"
        txt = yaml.safe_dump(c, sort_keys=False); sha = hashlib.sha256(txt.encode()).hexdigest()
        fb, fc = flat(base), flat(c)
        diff = {k: [fb.get(k, "<absent>"), fc.get(k, "<absent>")] for k in sorted(set(fb) | set(fc)) if fb.get(k, "<absent>") != fc.get(k, "<absent>")}
        shas[name] = {"sha256": sha, "run_id": run_id, "parent": parent, "seed": s, "diff_vs_parent": diff}
        print(f"{run_id:32s} {sha[:16]}  " + "; ".join(f"{k}: {v[0]} -> {v[1]}" for k, v in diff.items()))
        lines.append(f"sbatch --parsable --partition=RTX6000PRO,H100 --job-name=p114_jsap3_s{s} "
                     f"--export=ALL,CFG=configs/{name},RUN_ID={run_id},FINAL_CKPT=epoch_800.pt slurm/run_unit.sbatch")
        if a.write:
            (ROOT / "configs" / name).write_text(txt)
    if a.write:
        (ROOT / "configs" / "P114_SHA256.json").write_text(json.dumps({"stage": STAGE, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "configs": shas}, indent=1))
        (ROOT / "slurm" / "p114_lines.txt").write_text(
            "# P114 (v5 NEXT-A-JS-AP3) launch lines — DRAFT, NOT SUBMITTED. Main session freezes the prereg first. Normal QOS (never runfill); node60 allowed.\n"
            + "\n".join(lines) + "\n")
        print("wrote configs, configs/P114_SHA256.json, slurm/p114_lines.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
