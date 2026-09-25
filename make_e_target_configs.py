"""Wave E: target-branch / predictor / projector-depth variants (plan-compatible SSL wiring), derived from the best 200-epoch config
(K=8 + critic lr x10). Stage P22_vcs_target_branch. Seed 0, 200 epochs, one change each (ema_pred = two, disclosed)."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "configs" / "cifar10_hpB_k8_clr10_vcs_seed0.yaml"

def setp(c, path, val):
    d = c
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = val

UNITS = [
    ("ema0.99", [(("train", "target_branch"), "ema_0.99")]),
    ("ema0.996", [(("train", "target_branch"), "ema_0.996")]),
    ("stopgrad", [(("train", "target_branch"), "stopgrad")]),
    ("sg_pred", [(("train", "target_branch"), "stopgrad"), (("model", "projector", "predictor"), True)]),
    ("ema0.99_pred", [(("train", "target_branch"), "ema_0.99"), (("model", "projector", "predictor"), True)]),
    ("proj_depth3", [(("model", "projector", "depth"), 3)]),
]
out, units = {}, []
for label, changes in UNITS:
    c = yaml.safe_load(BASE.read_text())
    c["run"]["stage"] = "P22_vcs_target_branch"
    for path, val in changes:
        setp(c, path, val)
    p = ROOT / "configs" / f"cifar10_hpE_{label}_vcs_seed0.yaml"
    p.write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True), encoding="utf-8")
    out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    units.append({"label": label, "changes": [(".".join(pth), v) for pth, v in changes], "config": f"configs/{p.name}", "run_id": f"P22_vcs_{label}_seed0"})
(ROOT / "configs" / "HPARAM_E_TARGET_SHA256.json").write_text(json.dumps({"derived_from": {BASE.name: hashlib.sha256(BASE.read_bytes()).hexdigest()},
    "baseline_run": "P16_vcs_k8_clr10_seed0", "units": units, "files": out}, indent=2))
(ROOT / "slurm" / "hparamE_target_units.txt").write_text("".join(f"{u['run_id']} {u['config']} epoch_200.pt\n" for u in units))
print("\n".join(f"{u['run_id']}: {u['changes']}" for u in units))
