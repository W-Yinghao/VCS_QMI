"""Critic-form / pairing / update-schedule named variants for VCS (owner: 把 critic 变体那批提交上去), derived from the K=8 seed-0 config.

Stage P18_vcs_critic_variants: seed 0, 200 epochs, K=8, critic lr x1; each unit changes one thing relative to P10_vcs_k8_seed0.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "configs" / "cifar10_k8_vcs_seed0.yaml"

def setp(c, path, val):
    d = c
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = val

UNITS = [
    ("crit_interact", [(("model", "critic", "input"), "concat_interact")]),
    ("crit_bilinear", [(("model", "critic", "input"), "bilinear_concat")]),
    ("crit_cosine", [(("model", "critic", "input"), "cosine")]),
    ("pair_sym", [(("pairing", "sampler"), "random_nonzero_cyclic_shift_symmetric")]),
    ("crit_steps2", [(("train", "mode"), "joint_critic_steps_2")]),
    ("crit_steps5", [(("train", "mode"), "joint_critic_steps_5")]),
]
out, units = {}, []
for label, changes in UNITS:
    c = yaml.safe_load(BASE.read_text())
    c["run"]["stage"] = "P18_vcs_critic_variants"
    for path, val in changes:
        setp(c, path, val)
    assert c["run"]["seed"] == 0 and c["pairing"]["k"] == 8 and c["train"]["epochs"] == 200
    p = ROOT / "configs" / f"cifar10_hpC_{label}_vcs_seed0.yaml"
    p.write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True), encoding="utf-8")
    out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    units.append({"label": label, "changes": [(".".join(pth), v) for pth, v in changes], "config": f"configs/{p.name}", "run_id": f"P18_vcs_{label}_seed0"})
(ROOT / "configs" / "HPARAM_C_CRITIC_SHA256.json").write_text(json.dumps({"derived_from": {BASE.name: hashlib.sha256(BASE.read_bytes()).hexdigest()},
    "baseline_run": "P10_vcs_k8_seed0", "units": units, "files": out}, indent=2))
(ROOT / "slurm" / "hparamC_critic_units.txt").write_text("".join(f"{u['run_id']} {u['config']} epoch_200.pt\n" for u in units))
print("\n".join(f"{u['run_id']}: {u['changes']}" for u in units))
