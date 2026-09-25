"""Wave H: new candidates on the neg-detach cosine base (stage P28_vcs_new_candidates) + seed fills for earlier cosine bases (stage P29_vcs_seedfill)."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent
def setp(c, path, val):
    d = c
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = val

BASE = ROOT / "configs" / "cifar10_hpF_cos_negdetach_vcs_seed0.yaml"
NEW = [
    ("mono_spline", [(("model", "critic", "input"), "mono_spline"), (("model", "critic", "hidden_dims"), [8, 8])]),
    ("diag_metric", [(("model", "critic", "input"), "diag_metric")]),
    ("views4", [(("views", "count"), 4)]),
]
SEEDS = [  # (label, source config, seed)
    ("cosK8", ROOT / "configs" / "cifar10_hpC_crit_cosine_vcs_seed0.yaml", 1),
    ("cosK8", ROOT / "configs" / "cifar10_hpC_crit_cosine_vcs_seed0.yaml", 2),
    ("cosK255", ROOT / "configs" / "cifar10_hpF_cos_k255_vcs_seed0.yaml", 1),
    ("cosK255", ROOT / "configs" / "cifar10_hpF_cos_k255_vcs_seed0.yaml", 2),
]
out, units_new, units_seed = {}, [], []
for label, changes in NEW:
    c = yaml.safe_load(BASE.read_text()); c["run"]["stage"] = "P28_vcs_new_candidates"
    for path, val in changes:
        setp(c, path, val)
    p = ROOT / "configs" / f"cifar10_hpH_{label}_vcs_seed0.yaml"
    p.write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True), encoding="utf-8"); out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    units_new.append({"label": label, "changes": [(".".join(pth), v) for pth, v in changes], "config": f"configs/{p.name}", "run_id": f"P28_vcs_{label}_seed0"})
for label, src, seed in SEEDS:
    c = yaml.safe_load(src.read_text()); c["run"]["stage"] = "P29_vcs_seedfill"; c["run"]["seed"] = seed
    p = ROOT / "configs" / f"cifar10_hpH_{label}_vcs_seed{seed}.yaml"
    p.write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True), encoding="utf-8"); out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    units_seed.append({"label": label, "seed": seed, "source": src.name, "config": f"configs/{p.name}", "run_id": f"P29_vcs_{label}_seed{seed}"})
(ROOT / "configs" / "HPARAM_H_SHA256.json").write_text(json.dumps({"new_candidates_base": BASE.name, "new": units_new, "seedfill": units_seed, "files": out}, indent=2))
(ROOT / "slurm" / "hparamH_new_units.txt").write_text("".join(f"{u['run_id']} {u['config']} epoch_200.pt\n" for u in units_new))
(ROOT / "slurm" / "hparamH_seed_units.txt").write_text("".join(f"{u['run_id']} {u['config']} epoch_200.pt\n" for u in units_seed))
print("\n".join(f"{u['run_id']}: {u['changes']}" for u in units_new)); print("\n".join(f"{u['run_id']} <- {u['source']}" for u in units_seed))
