"""P104 — package v3 first batch (owner 2026-09-30: "你来解压并看看这个文件夹中的计划，并执行"): eight full 800-epoch seed-0 cells mapped
from the logical manifest `configs/experiments_v3.yaml` of VCS_SSL_Performance_Research_and_Server_v3.zip onto the current schema.

Every YAML is the frozen 8x recipe `cifar10_hpK_a5_views4_800ep_vcs_seed{s}.yaml` (cosine critic a0 = 5, K = 8, negative right detach,
4 views, B 256, 800 epochs, AdamW 1e-3, critic wd 0) with run.stage = P104_v3_first_batch, epoch 20 added to logging.checkpoint_epochs
(package §9 diagnostics at 0/20/100/400/800), and only the listed fields changed:

  G1  critic.affine_mode fixed, a = 1, b = 0                     (negative right detach)
  G2  critic.affine_mode fixed, a = 2, b = -1                    (negative right detach)
  G3  critic.affine_mode fixed, a = 1, b = 0                     pairing.negative_detach false (full routing)
  G4  learned a0 = 5, b0 = 0                                     pairing.negative_detach false (full routing)
  U1  learned a0 = 5, b0 = 0                                     pairing.pair_scope all_view_tokens (chunk 256), detach
  U2  critic.affine_mode fixed, a = 1, b = 0                     pairing.pair_scope all_view_tokens (chunk 256), detach
  N1  learned a0 = 5, b0 = 0, observation_noise_tau 0.3, noise_repeats 4, noise_eval_repeats 16      (detach)
  N2  objective.loss js_matched_logistic (matched-JS control), tau 0.3, noise_repeats 1, noise_eval_repeats 16 (detach)

Learned cells carry critic.cosine_bias_init: 0.0 explicitly (= the recipe's b0; numerically identical init) so that every P104 run logs
the P104 cost counters.  --write writes the YAMLs, configs/P104_SHA256.json and slurm/p104_units.txt ("run_id config final_ckpt").
    python configs/make_p104_configs.py [--write] [--seeds 0]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = "cifar10_hpK_a5_views4_800ep_vcs_seed{s}.yaml"
STAGE = "P104_v3_first_batch"
ORDER = ["G1", "G3", "U1", "N1", "N2", "G2", "G4", "U2"]  # package v3 §4: G1, G3, U1, N1, N2 first


def apply(c: dict, uid: str) -> dict:
    crit, pair, obj = c["model"]["critic"], c["pairing"], c["objective"]
    def fixed(a: float, b: float) -> None:
        crit["affine_mode"] = "fixed"; crit["cosine_scale_init"] = float(a); crit["cosine_bias_init"] = float(b)
    def learned() -> None:
        crit["cosine_bias_init"] = 0.0
        assert float(crit["cosine_scale_init"]) == 5.0
    if uid == "G1":
        fixed(1.0, 0.0)
    elif uid == "G2":
        fixed(2.0, -1.0)
    elif uid == "G3":
        fixed(1.0, 0.0); pair["negative_detach"] = False
    elif uid == "G4":
        learned(); pair["negative_detach"] = False
    elif uid == "U1":
        learned(); pair["pair_scope"] = "all_view_tokens"; pair["all_view_chunk"] = 256
    elif uid == "U2":
        fixed(1.0, 0.0); pair["pair_scope"] = "all_view_tokens"; pair["all_view_chunk"] = 256
    elif uid == "N1":
        learned(); crit["observation_noise_tau"] = 0.3; crit["noise_repeats"] = 4; crit["noise_eval_repeats"] = 16
    elif uid == "N2":
        learned(); obj["loss"] = "js_matched_logistic"; crit["observation_noise_tau"] = 0.3; crit["noise_repeats"] = 1; crit["noise_eval_repeats"] = 16
    else:
        raise ValueError(uid)
    return c


def flat(d: dict, pre: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{pre}.{k}" if pre else k
        if isinstance(v, dict):
            out.update(flat(v, key))
        else:
            out[key] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); ap.add_argument("--seeds", default="0")
    a = ap.parse_args()
    rows, shas = [], {}
    for s in [int(x) for x in a.seeds.split(",")]:
        base = yaml.safe_load(open(ROOT / "configs" / BASE.format(s=s)))
        assert base["train"]["epochs"] == 800 and base["views"]["count"] == 4 and base["pairing"]["negative_detach"] is True
        for uid in ORDER:
            c = apply(json.loads(json.dumps(base)), uid)
            c["run"]["stage"] = STAGE
            ce = c["logging"]["checkpoint_epochs"]
            if 20 not in ce:
                c["logging"]["checkpoint_epochs"] = sorted(ce + [20])
            name = f"cifar10_hpX_{uid}_views4_800ep_seed{s}.yaml"
            txt = yaml.safe_dump(c, sort_keys=False)
            sha = hashlib.sha256(txt.encode()).hexdigest()
            fb, fc = flat(base), flat(c)
            diff = {k: (fb.get(k, "<absent>"), fc.get(k, "<absent>")) for k in sorted(set(fb) | set(fc)) if fb.get(k, "<absent>") != fc.get(k, "<absent>")}
            run_id = f"P104_{uid}_views4_800ep_seed{s}"
            rows.append((run_id, f"configs/{name}", "epoch_800.pt"))
            shas[name] = {"sha256": sha, "logical_variant": uid, "seed": s, "base": BASE.format(s=s), "diff_vs_base": {k: list(v) for k, v in diff.items()}}
            print(f"{run_id:34s} {sha[:16]}  " + "; ".join(f"{k}: {v[0]} -> {v[1]}" for k, v in diff.items()))
            if a.write:
                (ROOT / "configs" / name).write_text(txt)
    if a.write:
        (ROOT / "configs" / "P104_SHA256.json").write_text(json.dumps({"stage": STAGE, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "configs": shas}, indent=1))
        (ROOT / "slurm" / "p104_units.txt").write_text("".join(f"{r} {c} {f}\n" for r, c, f in rows))
        print("wrote configs, configs/P104_SHA256.json, slurm/p104_units.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
