"""P95 — package-v1 estimator improvements as online critic variants inside full CIFAR-10 SSL at 8x (owner 2026-09-29:
"我建议都放进ssl里试试，这种probe还是不能代替全量实验。就在cifar10尝试").

Every YAML is the frozen 8x recipe `cifar10_hpK_a5_views4_800ep_vcs_seed{s}.yaml` (cosine critic a0 = 5, K = 8, negative detach, 4 views,
B 256, 800 epochs, AdamW 1e-3) with run.stage = P95_v1_in_ssl and only the listed critic / objective fields changed.

    python configs/make_p95_configs.py [--write]                                        # stage A: 8 cells, seed 0
    python configs/make_p95_configs.py --stageB residual=0.25 dictionary noise=0.1 refresh=200 js [--write]   # seeds 1-2 of the selected cells

Stage-A grid: residual λ ∈ {0.25, 0.5}; dictionary (one cell); observation noise τ ∈ {0.1, 0.3}; critic refresh R ∈ {50, 200} epochs;
matched balanced-logistic (JS) control.  --write writes the YAMLs, merges configs/P95_SHA256.json per stage and writes
slurm/p95_units_stageA.txt / slurm/p95_units_stageB.txt ("run_id config final_ckpt" per line).
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
CFG = ROOT / "configs"
SLURM = ROOT / "slurm"
SHA_JSON = CFG / "P95_SHA256.json"
BASE = "cifar10_hpK_a5_views4_800ep_vcs_seed{s}.yaml"
STAGE = "P95_v1_in_ssl"
GRID = {"residual": [0.25, 0.5], "dictionary": [None], "noise": [0.1, 0.3], "refresh": [50, 200], "js": [None]}


def fmt(x) -> str:
    return "" if x is None else f"{x:g}"


def apply(c: dict, family: str, val) -> tuple[dict, str]:
    crit = c["model"]["critic"]
    if family == "residual":
        crit["input"] = "residual_cosine_mlp"; crit["residual_lambda"] = float(val)
        return {"model.critic.input": "residual_cosine_mlp", "model.critic.residual_lambda": float(val)}, f"residual_lam{fmt(val)}"
    if family == "dictionary":
        crit["input"] = "dictionary_simplex"
        return {"model.critic.input": "dictionary_simplex"}, "dictionary"
    if family == "noise":
        crit["observation_noise_tau"] = float(val)
        return {"model.critic.observation_noise_tau": float(val)}, f"noise_tau{fmt(val)}"
    if family == "refresh":
        crit["refresh_every_epochs"] = int(val); crit["refresh_batches"] = 16
        return {"model.critic.refresh_every_epochs": int(val), "model.critic.refresh_batches": 16}, f"refresh_R{int(val)}"
    if family == "js":
        c["objective"]["loss"] = "js_matched_logistic"
        return {"objective.loss": "js_matched_logistic"}, "js_matched"
    raise ValueError(family)


def make(family: str, val, seed: int) -> dict:
    base = BASE.format(s=seed)
    c = yaml.safe_load((CFG / base).read_text(encoding="utf-8"))
    assert c["model"]["critic"]["input"] == "cosine" and c["pairing"]["negative_detach"] and c["views"]["count"] == 4 and c["train"]["epochs"] == 800
    c["run"]["stage"] = STAGE
    ch, tag = apply(c, family, val)
    return {"stage": STAGE, "family": family, "value": val, "seed": seed, "base_config": base, "config": f"configs/cifar10_hpV_{tag}_views4_800ep_seed{seed}.yaml",
            "run_id": f"P95_{tag}_views4_800ep_seed{seed}", "final_ckpt": "epoch_800.pt", "changes": ch, "_cfg": c}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stageB", nargs="*", default=None, help="family=value (value omitted for dictionary / js), one per selected family")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    if a.stageB is None:
        units = [make(f, v, 0) for f, vals in GRID.items() for v in vals]; which = "A"
    else:
        units = []
        for item in a.stageB:
            f, _, v = item.partition("=")
            if f not in GRID:
                ap.error(f"unknown family {f}")
            val = None if not v else (int(v) if f == "refresh" else float(v))
            if val not in GRID[f]:
                ap.error(f"{f}={v} is not a stage-A grid value {GRID[f]}")
            units += [make(f, val, s) for s in (1, 2)]
        which = "B"
    for u in units:
        print(f"{u['run_id']:<46} {u['config']:<64} {u['final_ckpt']}  <- {u['base_config']}  {u['changes']}")
    print(f"{len(units)} units (stage {which})")
    if not a.write:
        print("dry run (pass --write)"); return 0
    reg = json.loads(SHA_JSON.read_text()) if SHA_JSON.is_file() else {"stages": {}}
    for u in units:
        p = CFG / Path(u["config"]).name
        p.write_text(yaml.safe_dump(u["_cfg"], sort_keys=False, allow_unicode=True), encoding="utf-8")
        u["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    key = f"{STAGE}_stage{which}"
    reg["stages"][key] = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                          "units": [{k: v for k, v in u.items() if k != "_cfg"} for u in units],
                          "files": {Path(u["config"]).name: u["sha256"] for u in units}}
    SHA_JSON.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    uf = SLURM / f"p95_units_stage{which}.txt"
    uf.write_text("".join(f"{u['run_id']} {u['config']} {u['final_ckpt']}\n" for u in units), encoding="utf-8")
    print(f"wrote {uf} and {SHA_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
