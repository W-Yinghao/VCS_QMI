"""Derive the frozen A-group (VCS-only) hyper-parameter configs from the 200-epoch VCS seed-0 confirmation config.

Stage P12_vcs_hparamA (owner request 2026-09-24): single seed 0, 200 epochs, warm-up 10; each config changes ONE field relative to
P5_vcs_seed0 (K=1, critic [512,512], gain 0.1, critic lr x1, critic wd 0, projector 512/128, l2 critic input).
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "configs" / "cifar10_confirm200_vcs_seed0.yaml"
# (label, path, value) — submission priority order
UNITS = [
    ("cw2048", ("model", "critic", "hidden_dims"), [2048, 2048]),
    ("cw256", ("model", "critic", "hidden_dims"), [256, 256]),
    ("po512", ("model", "projector", "output_dim"), 512),
    ("po256", ("model", "projector", "output_dim"), 256),
    ("clr10", ("optimizer", "critic_lr_multiplier"), 10.0),
    ("clr0.1", ("optimizer", "critic_lr_multiplier"), 0.1),
    ("k255", ("pairing", "k"), 255),
    ("cd3", ("model", "critic", "hidden_dims"), [512, 512, 512]),
    ("cd1", ("model", "critic", "hidden_dims"), [512]),
    ("cw1024", ("model", "critic", "hidden_dims"), [1024, 1024]),
    ("cw128", ("model", "critic", "hidden_dims"), [128, 128]),
    ("ph2048", ("model", "projector", "hidden_dim"), 2048),
    ("ph1024", ("model", "projector", "hidden_dim"), 1024),
    ("clr3", ("optimizer", "critic_lr_multiplier"), 3.0),
    ("clr0.3", ("optimizer", "critic_lr_multiplier"), 0.3),
    ("po64", ("model", "projector", "output_dim"), 64),
    ("cwd1e-4", ("optimizer", "critic_weight_decay"), 1e-4),
    ("cwd1e-3", ("optimizer", "critic_weight_decay"), 1e-3),
    ("gain1.0", ("model", "critic", "last_layer_xavier_gain"), 1.0),
    ("gain0.01", ("model", "critic", "last_layer_xavier_gain"), 0.01),
    ("rawp", ("model", "normalization", "vcs_and_simclr"), "none"),
]
out, units = {}, []
for label, path, val in UNITS:
    c = yaml.safe_load(BASE.read_text())
    c["run"]["stage"] = "P12_vcs_hparamA"
    d = c
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = val
    assert c["run"]["seed"] == 0 and c["train"]["epochs"] == 200 and c["run"]["method"] == "vcs_qmi"
    p = ROOT / "configs" / f"cifar10_hpA_{label}_vcs_seed0.yaml"
    p.write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True), encoding="utf-8")
    out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    units.append({"label": label, "field": ".".join(path), "value": val, "config": f"configs/{p.name}", "run_id": f"P12_vcs_{label}_seed0"})
(ROOT / "configs" / "HPARAM_A_SHA256.json").write_text(json.dumps({"derived_from": {BASE.name: hashlib.sha256(BASE.read_bytes()).hexdigest()},
                                                                   "baseline_run": "P5_vcs_seed0", "units": units, "files": out}, indent=2))
(ROOT / "slurm" / "hparamA_units.txt").write_text("".join(f"{u['run_id']} {u['config']}\n" for u in units))
print(len(units), "configs;", "\n".join(f"{u['run_id']}: {u['field']}={u['value']}" for u in units))
