"""B-group (shared recipe) LR / schedule single-factor configs for VCS, derived from the 200-epoch VCS seed-0 confirmation config.

Stage P14_vcs_hparamB_lr (owner question 2026-09-25: is slow convergence due to a too-small LR?).  Seed 0, 200 epochs, K=1, one field each.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "configs" / "cifar10_confirm200_vcs_seed0.yaml"
UNITS = [
    ("lr3e-3", ("optimizer", "lr"), 3e-3),
    ("lr1e-2", ("optimizer", "lr"), 1e-2),
    ("lr3e-4", ("optimizer", "lr"), 3e-4),
    ("floor0.1", ("schedule", "min_lr_ratio"), 0.1),
    ("const", ("schedule", "min_lr_ratio"), 1.0),   # cosine with floor 1.0 == constant LR after warm-up
]
out, units = {}, []
for label, path, val in UNITS:
    c = yaml.safe_load(BASE.read_text())
    c["run"]["stage"] = "P14_vcs_hparamB_lr"
    d = c
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = val
    assert c["run"]["seed"] == 0 and c["train"]["epochs"] == 200 and c["pairing"]["k"] == 1
    p = ROOT / "configs" / f"cifar10_hpB_{label}_vcs_seed0.yaml"
    p.write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True), encoding="utf-8")
    out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    units.append({"label": label, "field": ".".join(path), "value": val, "config": f"configs/{p.name}", "run_id": f"P14_vcs_{label}_seed0"})
(ROOT / "configs" / "HPARAM_B_LR_SHA256.json").write_text(json.dumps({"derived_from": {BASE.name: hashlib.sha256(BASE.read_bytes()).hexdigest()},
                                                                      "baseline_run": "P5_vcs_seed0", "units": units, "files": out}, indent=2))
(ROOT / "slurm" / "hparamB_lr_units.txt").write_text("".join(f"{u['run_id']} {u['config']}\n" for u in units))
print("\n".join(f"{u['run_id']}: {u['field']}={u['value']}" for u in units))
