"""Derive the frozen 200-epoch confirmation configs from the frozen pilot configs (explicit, auditable overrides only).

Stage P5_confirm200 (spec §8/§19 and configs/next_stage_plan.yaml): same recipe, restart from the same initial weights per seed,
200 epochs, warm-up 10 epochs, seeds {0,1,2}, checkpoints at 20/50/100/150/200, kNN monitor at 0/10/20/50/100/150/200.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent
OVERRIDES = {
    ("run", "stage"): "P5_confirm200",
    ("train", "epochs"): 200,
    ("train", "warmup_epochs"): 10,
    ("logging", "checkpoint_epochs"): [20, 50, 100, 150, 200],
    ("evaluation", "knn_epochs"): [0, 10, 20, 50, 100, 150, 200],
    ("evaluation", "linear_epochs_of_pretrain"): [200],
}
out = {}
for name in ("vcs", "simclr", "vicreg"):
    base = yaml.safe_load((ROOT / "configs" / f"cifar10_pilot_{name}.yaml").read_text())
    for seed in (0, 1, 2):
        c = json.loads(json.dumps(base))
        for path, val in OVERRIDES.items():
            d = c
            for k in path[:-1]:
                d = d[k]
            d[path[-1]] = val
        c["run"]["seed"] = seed
        p = ROOT / "configs" / f"cifar10_confirm200_{name}_seed{seed}.yaml"
        p.write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True), encoding="utf-8")
        out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
(ROOT / "configs" / "CONFIRM200_SHA256.json").write_text(json.dumps({"derived_from": {f"cifar10_pilot_{n}.yaml": hashlib.sha256((ROOT / "configs" / f"cifar10_pilot_{n}.yaml").read_bytes()).hexdigest() for n in ("vcs", "simclr", "vicreg")}, "overrides": {".".join(k): v for k, v in OVERRIDES.items()}, "seeds": [0, 1, 2], "files": out}, indent=2))
print(json.dumps(out, indent=2))
