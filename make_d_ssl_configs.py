"""SSL-wiring variants permitted by the collaborator's plan (owner 2026-09-25: only the plan's content is fixed), derived from the current
best 200-epoch VCS config (K=8 + critic lr x10).  Stage P20_vcs_ssl_wiring.  Seed 0, 200 epochs, one change each.
"""
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
    ("crit_on_h", [(("model", "critic", "feature_source"), "h_l2")]),           # critic reads L2-normalized encoder output (512-d); projector unused
    ("neg_detach", [(("pairing", "negative_detach"), True)]),                    # shifted partner detached in negative pairs
    ("crit_steps5", [(("train", "mode"), "joint_critic_steps_5")]),             # 4 critic-only steps per batch on the best base
]
out, units = {}, []
for label, changes in UNITS:
    c = yaml.safe_load(BASE.read_text())
    c["run"]["stage"] = "P20_vcs_ssl_wiring"
    for path, val in changes:
        setp(c, path, val)
    assert c["pairing"]["k"] == 8 and c["optimizer"]["critic_lr_multiplier"] == 10.0 and c["train"]["epochs"] == 200 and c["run"]["seed"] == 0
    p = ROOT / "configs" / f"cifar10_hpD_{label}_vcs_seed0.yaml"
    p.write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True), encoding="utf-8")
    out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    units.append({"label": label, "changes": [(".".join(pth), v) for pth, v in changes], "config": f"configs/{p.name}", "run_id": f"P20_vcs_{label}_seed0"})
(ROOT / "configs" / "HPARAM_D_SSL_SHA256.json").write_text(json.dumps({"derived_from": {BASE.name: hashlib.sha256(BASE.read_bytes()).hexdigest()},
    "baseline_run": "P16_vcs_k8_clr10_seed0", "units": units, "files": out}, indent=2))
(ROOT / "slurm" / "hparamD_ssl_units.txt").write_text("".join(f"{u['run_id']} {u['config']} epoch_200.pt\n" for u in units))
print("\n".join(f"{u['run_id']}: {u['changes']}" for u in units))
